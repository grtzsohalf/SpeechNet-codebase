import os
import copy
import random

import torch
import torchaudio
from librosa.util import find_files
from torch.utils.data.dataset import Dataset


class NoisyCleanDataset(Dataset):
    def __init__(self, roots, sample_rate, max_time, target_level=-25, noise_proportion=0, io_normalization=False,
                 noise_type='gaussian', target_type='clean', snrs=[3], min_time=0, eps=1e-8, **kwargs):
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.min_time = min_time
        self.target_level = target_level
        self.io_normalization = io_normalization
        self.eps = eps

        self.filepths = []
        for root in roots:
            self.filepths += find_files(root)
        assert len(self.filepths) > 0, 'No audio file detected'

        self.noise_proportion = noise_proportion
        self.snrs = snrs
        if noise_type == 'gaussian':
            self.noise_sampler = torch.distributions.Normal(0, 1)
        else:
            self.noise_wavpths = find_files(noise_type)

        self.target_type = target_type
        if os.path.isdir(target_type):
            self.tar_filepths = find_files(target_type)
            assert len(self.tar_filepths) > 0
            self.regex_searcher = re.compile('fileid_\d+')

    def get_subset(self, *args, **kwargs):
        def sample_files(files, ratio=0.2, select_sampled=True, seed=0):
            random.seed(seed)
            files = copy.deepcopy(files)
            random.shuffle(files)
            sampled_num = round(ratio * len(files))
            return files[:sampled_num] if select_sampled else files[sampled_num:]
        subset = copy.deepcopy(self)
        subset.filepths = sample_files(subset.filepths, *args, **kwargs)
        if hasattr(subset, 'noise_wavpths'):
            subset.noise_wavpths = sample_files(subset.noise_wavpths, *args, **kwargs)
        return subset

    @classmethod
    def normalize_wav_decibel(cls, audio, target_level):
        '''Normalize the signal to the target level'''
        rms = audio.pow(2).mean().pow(0.5)
        scalar = (10 ** (target_level / 20)) / (rms + 1e-10)
        audio = audio * scalar
        return audio

    @classmethod
    def load_data(cls, wav_path, sample_rate=16000, max_time=40000, target_level=-25, min_time=0, **kwargs):
        wav, sr = torchaudio.load(wav_path)
        assert sr == sample_rate, f'Sample rate mismatch: real {sr}, config {sample_rate}'
        wav = wav.view(-1)
        maxpoints = int(sr / 1000) * max_time
        minpoints = int(sr / 1000) * min_time
        if len(wav) < minpoints:
            times = minpoints // len(wav) + 1
            wav = wav.unsqueeze(0).expand(times, -1).reshape(-1)
        if len(wav) > maxpoints:
            start = random.randint(0, len(wav) - maxpoints)
            wav = wav[start:start + maxpoints]
        wav = cls.normalize_wav_decibel(wav, target_level)
        return wav

    @classmethod
    def add_noise(cls, speech, noise, snrs, eps=1e-10):
        # speech, noise: (batch_size, seqlen)
        if speech.size(-1) >= noise.size(-1):
            times = speech.size(-1) // noise.size(-1)
            remainder = speech.size(-1) % noise.size(-1)
            noise_expanded = noise.unsqueeze(-2).expand(-1, times, -1).reshape(speech.size(0), -1)
            noise = torch.cat([noise_expanded, noise[:, :remainder]], dim=-1)
        else:
            start = random.randint(0, noise.size(-1) - speech.size(-1))
            noise = noise[:, start:start + speech.size(-1)]
        assert noise.size(-1) == speech.size(-1)

        snr = float(snrs[random.randint(0, len(snrs) - 1)])
        snr_exp = 10.0 ** (snr / 10.0)
        speech_power = speech.pow(2).sum(dim=-1, keepdim=True)
        noise_power = noise.pow(2).sum(dim=-1, keepdim=True)
        scalar = (speech_power / (snr_exp * noise_power + eps)).pow(0.5)
        scaled_noise = scalar * noise
        noisy = speech + scaled_noise
        assert torch.isnan(noisy).sum() == 0 and torch.isinf(noisy).sum() == 0 
        return noisy, scaled_noise

    def __getitem__(self, idx):
        load_config = [self.sample_rate, self.max_time, self.target_level, self.min_time]
        src_pth = self.filepths[idx]
        wav = NoisyCleanDataset.load_data(src_pth, *load_config)

        # build input
        dice = random.random()
        if dice < self.noise_proportion:
            if hasattr(self, 'noise_sampler'):
                noise = self.noise_sampler.sample(wav.shape)
            elif hasattr(self, 'noise_wavpths'):
                noise_idx = random.randint(0, len(self.noise_wavpths) - 1)
                noise, noise_sr = torchaudio.load(self.noise_wavpths[noise_idx])
                if noise_sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(noise_sr, self.sample_rate)
                    noise = resampler(noise)
                    noise_sr = self.sample_rate
                noise = noise.squeeze(0)
            noisy, scaled_noise = NoisyCleanDataset.add_noise(wav.unsqueeze(0), noise.unsqueeze(0), self.snrs, self.eps)
            noisy, scaled_noise = noisy.squeeze(0), scaled_noise.squeeze(0)
            wav_inp = noisy
        else:
            wav_inp = wav

        # build target
        if self.target_type == 'clean':
            wav_tar = wav
        elif self.target_type == 'noise':
            assert 'scaled_noise' in locals()
            wav_tar = scaled_noise
        else:
            result = self.regex_searcher.search(src_pth)
            assert result is not None
            fileid = result.group()
            tar_candidates = [pth for pth in self.tar_filepths if fileid in pth]
            tar_searcher = re.compile(fileid + '\D')
            tar_pths = [pth for pth in tar_candidates if tar_searcher.search(pth) is not None]
            assert len(tar_pths) == 1, f'{tar_pths}'
            tar_pth = tar_pths[0]
            wav_tar = NoisyCleanDataset.load_data(tar_pth, *load_config)

        if self.io_normalization:
            wav_inp = NoisyCleanDataset.normalize_wav_decibel(wav_inp, self.target_level)
            wav_tar = NoisyCleanDataset.normalize_wav_decibel(wav_tar, self.target_level)

        wavs = torch.stack([wav_inp, wav_tar], dim=-1)
        return wavs, src_pth

    def __len__(self):
        return len(self.filepths)
