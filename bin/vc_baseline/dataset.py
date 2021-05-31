import os
import copy
import random
import glob
import re
from collections import defaultdict

import torch
import torchaudio
from librosa.util import find_files
from torch.utils.data.dataset import Dataset

class VcDataset(Dataset):
    def __init__(self, split, test_speakers, data_path, speaker_info_path, test_proportion):
        if split == 'train':
            self.mode = 'train'
        else:
            self.mode = 'val'
        self.root = os.path.join(data_path, split)
        self.speaker_info_path = speaker_info_path

        article2filenames, spk2filenames, filename2spk = VcDataset.read_filenames(self.root)
        self.train_path_list = []
        self.example_path_list = []
        self.spk_1_list = []
        self.spk_2_list = []
        article_ids = article2filenames.keys()
        for article in article_ids:
            path_list = article2filenames[article]
            for i in range(len(path_list)):
                for j in range(len(path_list)):
                    if i != j:
                        self.train_path_list.append([path_list[i], path_list[j]])
                        self.example_path_list.append([spk2filenames[filename2spk[path_list[i]]][0],
                                                       spk2filenames[filename2spk[path_list[j]]][0]])
                        self.spk_1_list.append(filename2spk[path_list[i]])
                        self.spk_2_list.append(filename2spk[path_list[j]])
    @classmethod
    def read_filenames(cls, root_dir):
        article2filenames = defaultdict(lambda : [])
        spk2filenames = defaultdict(lambda : [])
        filename2spk = {}
        for path in sorted(glob.glob(os.path.join(root_dir, '*/*/*.wav'))):
            spk_id = path.strip().split('/')[-3]
            utt_id = path.strip().split('/')[-1]
            article2filenames[utt_id].append(path)
            spk2filenames[spk_id].append(path)
            filename2spk[path] = spk_id
        return article2filenames, spk2filenames, filename2spk

    @classmethod
    def load_data(cls, wav_path, sample_rate=16000, max_time=128000, target_level=-25, min_time=0, **kwargs):
        wav, sr = torchaudio.load(wav_path)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            wav = resampler(wav)
            sr = sample_rate
        # assert sr == sample_rate, f'Sample rate mismatch: real {sr}, config {sample_rate}'
        wav = wav.view(-1)
        length = len(wav)
        maxpoints = int(sr / 1000) * max_time
        minpoints = int(sr / 1000) * min_time
        if length < minpoints:
            times = minpoints // length + 1
            wav = wav.unsqueeze(0).expand(times, -1).reshape(-1)
        if length > maxpoints:
            start = random.randint(0, length - maxpoints)
            wav = wav[start:start + maxpoints]
        wav = cls.normalize_wav_decibel(wav, target_level)
        return wav, length

    @classmethod
    def normalize_wav_decibel(cls, audio, target_level):
        '''Normalize the signal to the target level'''
        rms = audio.pow(2).mean().pow(0.5)
        scalar = (10 ** (target_level / 20)) / (rms + 1e-10)
        audio = audio * scalar
        return audio
    
    def __getitem__(self, idx):
        src_pth, tgt_pth = self.train_path_list[idx]
        spk_1 = self.spk_1_list[idx]
        spk_2 = self.spk_2_list[idx]
        src_wav, src_length = VcDataset.load_data(src_pth)
        tgt_wav, tgt_length = VcDataset.load_data(tgt_pth)
        if self.mode == 'val':
            example_src_pth, example_tgt_pth = self.example_path_list[idx]
            example_src_wav, example_src_length = VcDataset.load_data(example_src_pth)
            example_tgt_wav, example_tgt_length = VcDataset.load_data(example_tgt_pth)
        # wavs = torch.stack([wav], dim=-1)
        if self.mode == 'val':
            return src_wav, torch.tensor([src_length]), tgt_wav, torch.tensor(tgt_length), \
                    example_src_wav, torch.tensor([example_src_length]), example_tgt_wav, torch.tensor(example_tgt_length), spk_1, spk_2
        else:
            return src_wav, torch.tensor([src_length]), tgt_wav, torch.tensor(tgt_length), None, None, None, None, spk_1, spk_2

    def __len__(self):
        return len(self.train_path_list)
