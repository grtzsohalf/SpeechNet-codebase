import os
import glob
import math
import copy
import time
import random
from functools import partial
from collections import defaultdict
from joblib import Parallel, delayed

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from asteroid.data import LibriMix
from asteroid.losses import PITLossWrapper

from src.audio import *
from bin.ss.preprocessor import *
from bin.ss.dataset import *
from bin.ss.model import *
from bin.ss.objective import *
from bin.ss.evaluation import *
from bin.ss.utils import *

MAX_TIMESTAMPS = 50
LOG_WAV_NUM = 5


def logging(logger, step, tag, data, mode='scalar', preprocessor=None):
    if type(data) is torch.Tensor:
        data = data.detach().cpu()
    tag = f'ss/{tag}'

    if mode == 'scalar':
        # data is a int or float
        logger.add_scalar(tag, data, global_step=step)
    elif mode == 'audio':
        # data: (seqlen, )
        assert preprocessor is not None
        data = data / data.abs().max().item()
        # log wavform
        logger.add_audio(f'{tag}.wav', data.reshape(-1, 1), global_step=step, sample_rate=preprocessor._sample_rate)
        # log log-linear-scale spectrogram
        feat_config = OnlinePreprocessor.get_feat_config(feat_type='linear', log=True)
        linear = preprocessor(data.reshape(1, 1, -1), [feat_config])[0]
        figure = plot_spectrogram(linear)
        logger.add_figure(f'{tag}.png', figure, global_step=step)
    else:
        raise NotImplementedError


def collate_fn(wavs, input_transform):

    wavs = [torch.cat([m.reshape(1,-1), s]).transpose(-1,-2) for m, s in wavs]
    # wavs: [(seqlen, n_src+1), ...]

    lengths = torch.LongTensor([len(s) for s in wavs])
    padded_wavs = pad_sequence(wavs, batch_first=True).transpose(-1, -2).contiguous()
    # lengths: (batch_size, )
    # padded_wavs: (batch_size, n_src+1, seqlen)
    # mixture: padded_wavs[:, 0]
    # sources: padded_wavs[:, 1:]
    return padded_wavs, lengths


class SolverSS():
    ''' Handler for complete training and evaluation progress of downstream models '''
    def __init__(self, gpu_rank, world_size, rank, config, paras, mode, log, eps=1e-6):
        self.device = gpu_rank
        self.world_size = world_size
        self.rank = rank

        self.mode = mode
        self.args = paras
        self.config = config
        self.datarc = config['data']['ss_corpus']
        self.modelrc = config['model']['ss']

        self.sr = self.datarc['sample_rate']
        self.n_src = self.datarc['n_src']

        # Preprocessors
        # used for extract input feature for all tasks
        self.input_extracter, self.input_featdim, sr = create_transform(config['data']['audio'])
        assert self.sr == sr
        # used for extract noisy phase and target clean linear
        self.preprocessor = OnlinePreprocessor(
            sample_rate=self.sr,
            **self.datarc['preprocessor'],
        ).to(self.device)

        # Datasets
        self.trainset = LibriMix(**self.datarc['trainset'])
        self.devset = LibriMix(**self.datarc['devset'])
        self.testset = LibriMix(**self.datarc['testset'])

        # Reset dataloaders for train/dev/test
        self.set_trainloader()
        self.set_devloader()
        self.set_testloader()

        self.criterion = PITLossWrapper(eval(f'{self.modelrc["objective"]["name"]}')(**self.modelrc["objective"]), pit_from='pw_pt')
        self.metrics = [eval(f'{m}_eval') for m in self.modelrc['metrics']]
        self.ascending = torch.arange(MAX_TIMESTAMPS * self.sr).to(device=self.device)
        self.eps = eps

        self.logger = partial(logging, logger=log, preprocessor=copy.deepcopy(self.preprocessor).cpu())


    def transform_data(self, padded_wavs, lengths, audio_transform):
        # lengths: (batch_size, )
        # padded_wavs: (batch_size, n_src+1, seqlen)
        # mixture: padded_wavs[:, 0]
        # sources: padded_wavs[:, 1:]
        padded_wavs, lengths = padded_wavs.to(self.device), lengths.to(self.device)
        with torch.no_grad():
            # input_features = [audio_transform(wav[0, :lengths[i]].view(1, -1))\
                              # for i, wav in enumerate(padded_wavs)]
            # stft_lengths = torch.LongTensor([len(feature) for feature in input_features]).to(self.device)
            input_features, stft_lengths = audio_transform((padded_wavs[:, 0], lengths))
            padded_features = pad_sequence(input_features, batch_first=True)
            # padded_features: (batch_size, seqlen, featdim)
        return padded_wavs, lengths, padded_features, stft_lengths


    def set_trainloader(self):
        self.trainloader = DataLoader(
            self.trainset,
            num_workers=self.args.njobs,
            collate_fn=partial(collate_fn, input_transform=self.input_extracter[:-1]),
            shuffle=True if self.rank is None else False,
            sampler=None if self.rank is None else DistributedSampler(self.trainset, num_replicas=self.world_size, rank=self.rank),
            pin_memory=self.args.gpu,
            **self.datarc['trainloader'],
        )


    def set_devloader(self):
        self.devloader = DataLoader(
            self.devset,
            num_workers=self.args.njobs,
            collate_fn=partial(collate_fn, input_transform=self.input_extracter[:-1]),
            shuffle=True,
            pin_memory=self.args.gpu,
            **self.datarc['evalloader'],
        )


    def set_testloader(self):
        self.testloader = DataLoader(
            self.testset,
            num_workers=self.args.njobs,
            collate_fn=partial(collate_fn, input_transform=self.input_extracter[:-1]),
            pin_memory=self.args.gpu,
            **self.datarc['evalloader'],
        )


    def get_model(self):
        input_dim = self.config['model']['audio_decoder']['out_dim']
        target_dim = self.preprocessor(feat_list=[OnlinePreprocessor.get_feat_config('linear')])[0].size(-1)
        model_mel2lin = eval(f'{self.modelrc["model"]["mel2lin"]}')(input_dim, target_dim, **self.modelrc["model"])

        enc_dim = self.config['model']['audio_encoder']['dim']
        model_separator = Separator(enc_dim * 2, enc_dim * 2,
                                    eval(f'{self.modelrc["model"]["separator"]}'), self.n_src, **self.modelrc["model"])
        return model_mel2lin, model_separator


    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = self.ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1)).long()
        return length_masks


    def forward(self, ss_data, audio_encoder, audio_decoder, audio_transform, mel2lin, separator, **kwargs):
        padded_wavs, lengths, padded_features, stft_lengths = self.transform_data(*ss_data, audio_transform)

        phase_inp, *linear_tar = self.preprocessor(padded_wavs)
        if self.datarc['io_sample_pipeline']:
            stft_lengths = (lengths - self.preprocessor._win_args['win_length']) // self.preprocessor._win_args['hop_length']
            padded_features = mel_inp[:, :max(stft_lengths), :]
        else:
            padded_features, stft_lengths = padded_features.to(self.device), stft_lengths.to(self.device)
        # padded_wavs: [B, n+1, T]
        # lengths: [B,]
        # padded_features: [B, T, D]
        # stft_lengths: [B,]

        enc_len, input_x_content, enc_mask_content, input_x_speaker = audio_encoder(padded_features, stft_lengths)
        # enc_len: [B,]
        # inpu_x_content: [B, T, D]
        # enc_mask_content: [B, 1, D]??
        # inpu_x_speaker: [B, D]

        input_x = torch.cat([input_x_content, input_x_speaker.unsqueeze(1).expand_as(input_x_content)], -1)
        input_xs = [x for x, _ in separator(input_x)]
        input_xs = torch.cat(input_xs, 0)
        input_xs_content = input_xs[:, :, :separator.dim//2]
        input_xs_speaker = input_xs[:, :, separator.dim//2:].mean(1)
        # inpu_x_content: [Bn, T, D]
        # inpu_x_speaker: [Bn, D]
        enc_mask_content = enc_mask_content.expand(-1, self.n_src, -1).reshape(input_xs.shape[0], 1, -1).contiguous()
        # enc_mask_content: [Bn, 1, D]??
        enc_len = enc_len.repeat(self.n_src, 1).transpose(0, 1).reshape(-1).contiguous()
        # enc_len: [Bn,]

        stft_lengths, dec_x, stft_length_masks = audio_decoder(enc_len,
                                                               input_xs_content, enc_mask_content,
                                                               input_xs_speaker)

        stft_lengths = stft_lengths.reshape(-1, self.n_src)[:, 0]
        batch_size = len(stft_lengths)
        dec_x = dec_x.reshape(-1, self.n_src, *dec_x.shape[1:])
        stft_length_masks = stft_length_masks.reshape(-1, self.n_src, *stft_length_masks.shape[1:])
        # shape: [B, n_src, T]
        predicted, model_results = mel2lin(features=dec_x)
        # shape: [B, n_src, T, D]

        phase_inp = phase_inp[:, :max(stft_lengths), :]
        # shape: [B, T, D]
        linear_tar = [tar[:, :max(stft_lengths), :] for tar in linear_tar]
        linear_tar = torch.stack(linear_tar, 1)
        # shape: [B, n_src, T, D]

        vs = locals()
        return {key:item for key, item in vs.items() if key in [
            'predicted', 'linear_tar', 'phase_inp', 'padded_wavs',
            'lengths', 'stft_length_masks',
        ]}


    def compute_loss(self, predicted, linear_tar, stft_length_masks, step, split='train', **kwargs):
        # loss = self.criterion(linear_tar, predicted, stft_length_masks=stft_length_masks)
        if self.modelrc["objective"]["name"] == 'SISDR':
            predicted.masked_fill_(~stft_length_masks.unsqueeze(-1), -float('inf'))
            linear_tar.masked_fill_(~stft_length_masks.unsqueeze(-1), -float('inf'))
        elif self.modelrc["objective"]["name"] == 'L1':
            predicted.masked_fill_(~stft_length_masks.unsqueeze(-1), 0)
            linear_tar.masked_fill_(~stft_length_masks.unsqueeze(-1), 0)
        loss = self.criterion(predicted, linear_tar)
        if self.rank == 0 and split=='train' and step % self.modelrc['log_step'] == 0:
            self.logger(step=step, tag='train_loss', data=loss.item(), mode='scalar')
        return loss


    def init_evaluation(self, split='test'):
        random.seed(self.datarc['seed'])
        setattr(self, f'{split}_buffer', defaultdict(list))


    def compute_metrics(self, predicted, linear_tar, phase_inp, stft_length_masks,
                        padded_wavs, lengths, split='test', **kwargs):
        assert hasattr(self, f'{split}_buffer')
        buffer = getattr(self, f'{split}_buffer')

        wav_inp = padded_wavs[:, 0, :]    
        wav1_tar = padded_wavs[:, 1, :]
        wav2_tar = padded_wavs[:, 2, :]

        # compute loss
        loss, reordered = self.criterion(linear_tar, predicted, return_est=True, stft_length_masks=stft_length_masks)
        predicted1 = reordered[:, 0]
        predicted2 = reordered[:, 1]

        # reconstruct waveform
        wav1_predicted = self.preprocessor.istft(predicted1.exp(), phase_inp)
        wav1_predicted = torch.cat([wav1_predicted, wav1_predicted.new_zeros(wav1_predicted.size(0), max(lengths) - wav1_predicted.size(1))], dim=1)
        wav1_predicted = masked_normalize_decibel(wav1_predicted, wav1_tar, self._get_length_masks(lengths))
        wav2_predicted = self.preprocessor.istft(predicted2.exp(), phase_inp)
        wav2_predicted = torch.cat([wav2_predicted, wav2_predicted.new_zeros(wav2_predicted.size(0), max(lengths) - wav2_predicted.size(1))], dim=1)
        wav2_predicted = masked_normalize_decibel(wav2_predicted, wav2_tar, self._get_length_masks(lengths))

        # split batch into list of utterances and duplicate N_METRICS times
        batch_size = len(wav1_predicted)
        wav1_predicted_list = wav1_predicted.detach().cpu().chunk(batch_size) * len(self.metrics)
        wav2_predicted_list = wav2_predicted.detach().cpu().chunk(batch_size) * len(self.metrics)
        wav1_tar_list = wav1_tar.detach().cpu().chunk(batch_size) * len(self.metrics)
        wav2_tar_list = wav2_tar.detach().cpu().chunk(batch_size) * len(self.metrics)
        lengths_list = lengths.detach().cpu().tolist() * len(self.metrics)

        # prepare metric function for each utterance in the duplicated list
        ones = torch.ones(batch_size).long().unsqueeze(0).expand(len(self.metrics), -1)
        metric_ids = ones * torch.arange(len(self.metrics)).unsqueeze(-1)
        metric_fns = [self.metrics[idx.item()] for idx in metric_ids.reshape(-1)]
        
        def calculate_metric(length, predicted, target, metric_fn):
            return metric_fn(predicted.squeeze()[:length], target.squeeze()[:length])
        scores1 = Parallel(n_jobs=self.args.njobs)(delayed(calculate_metric)(l, p, t, f)
                            for l, p, t, f in zip(lengths_list, wav1_predicted_list, wav1_tar_list, metric_fns))
        scores2 = Parallel(n_jobs=self.args.njobs)(delayed(calculate_metric)(l, p, t, f)
                            for l, p, t, f in zip(lengths_list, wav2_predicted_list, wav2_tar_list, metric_fns))

        scores1 = torch.FloatTensor(scores1).view(len(self.metrics), batch_size).mean(dim=1)
        scores2 = torch.FloatTensor(scores2).view(len(self.metrics), batch_size).mean(dim=1)
        buffer['scores'].append(scores1)
        buffer['scores'].append(scores2)
        buffer['loss'].append(loss.item())
        if len(buffer['predicted1.wav']) < LOG_WAV_NUM:
            buffer['predicted1.wav'].append(wav1_predicted[0].detach().cpu())
            buffer['predicted2.wav'].append(wav2_predicted[0].detach().cpu())
            buffer['mixture.wav'].append(wav_inp[0].detach().cpu())
            buffer['s1.wav'].append(wav1_tar[0].detach().cpu())
            buffer['s2.wav'].append(wav2_tar[0].detach().cpu())


    def log_evaluation(self, step, split='test'):
        if self.rank != 0:
            return

        assert hasattr(self, f'{split}_buffer')
        buffer = getattr(self, f'{split}_buffer')

        for key, item in buffer.items():
            assert type(item) is list
            if 'scores' in key:
                scores = torch.stack(item, dim=0).mean(dim=0).unbind(0)
                for score, name in zip(scores, self.modelrc['metrics']):
                    self.logger(step=step, tag=f'{split}_{name}', data=score.item(), mode='scalar')
            elif 'loss' in key:
                loss = torch.FloatTensor(item).mean()
                self.logger(step=step, tag=f'{split}_loss', data=loss, mode='scalar')
            elif 'wav' in key:
                for i, wav in enumerate(item):
                    self.logger(step=step, tag=f'{i}.{split}_{key}', data=wav, mode='audio')
        delattr(self, f'{split}_buffer')
