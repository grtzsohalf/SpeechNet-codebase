import os
import glob
import math
import copy
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

from src.audio import *
from bin.ssw.preprocessor import *
from bin.ssw.dataset import *
from bin.ssw.model import *
from bin.ssw.objective import *
from bin.ssw.evaluation import *
from bin.ssw.utils import *

MAX_TIMESTAMPS = 50
LOG_WAV_NUM = 5


def logging(logger, step, tag, data, mode='scalar', preprocessor=None):
    if type(data) is torch.Tensor:
        data = data.detach().cpu()
    tag = f'ssw/{tag}'

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


def collate_fn(feats, input_transform, sr):
    # mix: [(seqlen, channel), ...]
    mix = torch.stack([feat[0] for feat in feats])
    sources = torch.stack([feat[1] for feat in feats])
    ids = torch.stack([feat[2] for feat in feats])

    # sources: (batch_size * n_src, seqlen, channel)
    batch_size, n_src, channel, seqlen = sources.size()
    sources = sources.view(batch_size * n_src, seqlen, channel)

    mix_lengths = torch.LongTensor([len(s) for s in mix])

    srcs_lengths = torch.LongTensor([len(s) for s in sources])
    # padded_mix: (batch_size, channel, seqlen)
    padded_mix = pad_sequence(mix, batch_first=True).transpose(-1, -2).contiguous()
    # mix_feats: (batch_size, seqlen, channel)
    mix_feats = torch.stack([input_transform(m.view(1, -1)) for m in mix])

    stft_mix_lengths = torch.LongTensor([len(feature) for feature in mix_feats])
    # padded_srcs: (batch_size * n_src, channel, seqlen)
    padded_srcs = pad_sequence(sources, batch_first=True).transpose(-1, -2).contiguous()
    # srcs_feats: (batch_size * n_src, seqlen, channel)
    srcs_feats = [input_transform(s.view(1, -1)) for s in sources]
    stft_srcs_lengths = torch.LongTensor([len(feature) for feature in srcs_feats])

    # padded_mix_feats: (batch_size, channel, max_seqlen)
    padded_mix_feats = pad_sequence(mix_feats, batch_first=True).transpose(-1, -2).contiguous()
    # padded_srcs_feats: (batch_size * n_src, channel, max_seqlen)
    padded_srcs_feats = pad_sequence(srcs_feats, batch_first=True).transpose(-1, -2).contiguous()
    return padded_mix, padded_srcs, mix_lengths, srcs_lengths, padded_mix_feats, padded_srcs_feats, stft_mix_lengths, stft_srcs_lengths


class SolverSSW():
    ''' Handler for complete training and evaluation progress of downstream models '''
    def __init__(self, gpu_rank, world_size, rank, config, paras, mode, log, eps=1e-6):
        self.device = gpu_rank
        self.world_size = world_size
        self.rank = rank

        self.mode = mode
        self.args = paras
        self.config = config
        self.datarc = config['data']['ssw_corpus']
        self.modelrc = config['model']['ssw']

        self.sr = self.datarc['sample_rate']

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
        self.trainset = WaveSplitWhamDataset(
            self.datarc['train_file_path'], task=self.datarc['task'], sample_rate=self.sr, segment=self.datarc['segment'], nondefault_nsrc=self.datarc['nondefault_nsrc']
        )
        self.devset = WaveSplitWhamDataset(
            self.datarc['dev_file_path'], task=self.datarc['task'], sample_rate=self.sr, segment=self.datarc['segment'],nondefault_nsrc=self.datarc['nondefault_nsrc']
        )
       
        if self.datarc.get('dev_size') is not None:
            random.seed(self.datarc['seed'])
            self.devset.filepths = random.sample(self.devset.filepths, self.datarc['dev_size'])
        # self.testset = NoisyCleanDataset(sample_rate=self.sr, **self.datarc['testset'])

        # Reset dataloaders for train/dev/test
        self.set_trainloader()
        self.set_devloader()
        # self.set_testloader()

        self.criterion = eval(f'{self.modelrc["objective"]["name"]}')(**self.modelrc["objective"])
        self.metrics = [eval(f'{m}_eval') for m in self.modelrc['metrics']]
        self.ascending = torch.arange(MAX_TIMESTAMPS * self.sr).to(device=self.device)
        self.eps = eps

        self.logger = partial(logging, logger=log, preprocessor=copy.deepcopy(self.preprocessor).cpu())


    def set_trainloader(self):
        self.trainloader = DataLoader(
            self.trainset,
            num_workers=self.args.njobs,
            collate_fn=partial(collate_fn, input_transform=self.input_extracter[:-1], sr=self.sr),
            shuffle=True if self.rank is None else False,
            sampler=None if self.rank is None else DistributedSampler(self.trainset, num_replicas=self.world_size, rank=self.rank),
            **self.datarc['trainloader'],
        )


    def set_devloader(self):
        self.devloader = DataLoader(
            self.devset,
            num_workers=self.args.njobs,
            collate_fn=partial(collate_fn, input_transform=self.input_extracter[:-1], sr=self.sr),
            shuffle=True if self.rank is None else False,
            sampler=None if self.rank is None else DistributedSampler(self.devset, num_replicas=self.world_size, rank=self.rank),
            **self.datarc['evalloader'],
        )


    def set_testloader(self):
        self.testloader = DataLoader(
            self.testset,
            num_workers=self.args.njobs,collate_fn=partial(collate_fn, input_transform=self.input_extracter[:-1], sr=self.sr),
            **self.datarc['evalloader'],
        )


    def get_model(self):
        input_dim = self.config['model']['audio_encoder']['dim']
        target_dim = self.preprocessor(feat_list=[OnlinePreprocessor.get_feat_config('linear')])[0].size(-2)
        print('target dim: ', target_dim)
        return eval(f'{self.modelrc["model"]["name"]}')(2, target_dim, 240)
    
    def get_linear_model(self):
        input_dim = self.config['model']['audio_decoder']['out_dim']
        target_dim = self.preprocessor(feat_list=[OnlinePreprocessor.get_feat_config('linear')])[0].size(-2)
        return eval(f'{self.modelrc["model"]["name_linear"]}')(input_dim, target_dim, **self.modelrc["model"])
        
    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = self.ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1)).long()
        return length_masks


    def forward(self, ss_data, audio_encoder, audio_decoder, model, mel_to_linear, **kwargs):
        padded_mix, padded_srcs, mix_lengths, srcs_lengths, padded_mix_feats, padded_srcs_feats, stft_mix_lengths, stft_srcs_lengths = ss_data
        padded_mix, padded_srcs, padded_srcs_feats, stft_mix_lengths, stft_srcs_lengths, srcs_lengths = padded_mix.to(self.device), padded_srcs.to(self.device), padded_srcs_feats.to(self.device), stft_mix_lengths.to(self.device), stft_srcs_lengths.to(self.device), srcs_lengths.to(self.device)
        
        print(padded_mix.shape)
        print(padded_mix_feats.shape)
        print(stft_mix_lengths)
        import sys
        sys.exit()

        phase_mix, linear_mix = self.preprocessor(torch.cat((padded_mix, padded_mix), 1))
        phase_mix = phase_mix[:, :, :max(stft_mix_lengths)]
        linear_mix = linear_mix[:, :, :max(stft_mix_lengths)]
        
        phase_srcs, linear_srcs = self.preprocessor(torch.cat((padded_srcs, padded_srcs), 1))
        
        phase_srcs = phase_srcs[:, :, :max(stft_srcs_lengths)]
        linear_srcs = linear_srcs[:, :, :max(stft_srcs_lengths)]
        
        # embs: (batch, n_src, n_emb, n_frames)
        embs = model(linear_mix)
        stft_srcs_lengths, dec_x, stft_length_masks = audio_decoder(*audio_encoder((embs * padded_srcs_feats).transpose(-1, -2), stft_srcs_lengths))
        predicted, model_results = mel_to_linear(features=dec_x)
        
        vs = locals()
        return {key:item for key, item in vs.items() if key in [
            'predicted', 'padded_mix', 'padded_srcs', 'phase_mix', 'linear_srcs', 'stft_mix_lengths', 'stft_srcs_lengths', 'srcs_lengths'
        ]}


    def compute_loss(self, predicted, padded_mix, phase_mix, linear_srcs, stft_mix_lengths, stft_srcs_lengths, srcs_lengths, step, split='train', **kwargs):
        first, seqlen, channel = predicted.size()
        batch_size = padded_mix.size()[0]
        
        predicted = predicted.view(batch_size, -1, seqlen, channel)
        linear_srcs = linear_srcs[:, :channel, :max(stft_srcs_lengths)].reshape(batch_size, -1, seqlen, channel)
        loss, objective_results = self.criterion(predicted, linear_srcs)
        if self.rank == 0 and split=='train' and step % self.modelrc['log_step'] == 0:
            self.logger(step=step, tag='train_loss', data=loss.item(), mode='scalar')
        return loss


    def init_evaluation(self, split='test'):
        random.seed(self.datarc['seed'])
        setattr(self, f'{split}_buffer', defaultdict(list))


    def compute_metrics(self, predicted, padded_mix, padded_srcs, phase_mix, linear_srcs, stft_mix_lengths, stft_srcs_lengths, srcs_lengths, split='test', **kwargs):
        buffer = defaultdict(list)
        first, seqlen, channel = predicted.size()
        batch_size = padded_mix.size()[0]
        phase_mix = phase_mix[:, :channel, :max(stft_srcs_lengths)].transpose(-1, -2)
        phase_mix = torch.cat((phase_mix, phase_mix), dim=0)
        linear_srcs = linear_srcs[:, :channel, :max(stft_srcs_lengths)]           
        # prepare metric function for each utterance in the duplicated list
        ones = torch.ones(batch_size).long().unsqueeze(0).expand(len(self.metrics), -1)
        metric_ids = ones * torch.arange(len(self.metrics)).unsqueeze(-1)
        metric_fns = [self.metrics[idx.item()] for idx in metric_ids.reshape(-1)]

        # compute loss
        # predicted = predicted.view(batch_size, -1, seqlen, channel)
        loss, objective_results = self.criterion(predicted.view(batch_size, -1, seqlen, channel), linear_srcs.reshape(batch_size, -1, seqlen, channel))
        # reconstruct waveform
        predicted = predicted[:, :max(stft_srcs_lengths), :]
        wav_predicted = self.preprocessor.istft(predicted[objective_results].exp(), phase_mix)
        wav_predicted = torch.cat([wav_predicted, wav_predicted.new_zeros(wav_predicted.size(0), max(srcs_lengths) - wav_predicted.size(1))], dim=1)
        padded_srcs = padded_srcs.squeeze(1)
        wav_predicted = masked_normalize_decibel(wav_predicted, padded_srcs.squeeze(1), self._get_length_masks(srcs_lengths))
        
        # split batch into list of utterances and duplicate N_METRICS times
        wav_predicted_list = wav_predicted.detach().cpu().chunk(batch_size) * len(self.metrics)
        srcs_list = padded_srcs.detach().cpu().chunk(batch_size) * len(self.metrics)
        srcs_lengths_list = srcs_lengths.detach().cpu().tolist() * len(self.metrics)

        def calculate_metric(length, predicted, target, metric_fn):
            return metric_fn(predicted[0, :length], target[0, :length])
        scores = Parallel(n_jobs=self.args.njobs)(delayed(calculate_metric)(l, p, t, f) for l, p, t, f in zip(srcs_lengths_list, wav_predicted_list, srcs_list, metric_fns))
        scores = torch.FloatTensor(scores).view(len(self.metrics), batch_size).mean(dim=1)
        
        buffer['scores'] = scores
        buffer['loss'] = loss.item()
        
        if len(buffer['predicted.wav']) < LOG_WAV_NUM:
            buffer['predicted.wav'] = wav_predicted[0].detach().cpu()
            buffer['s1.wav'] = srcs_list[0][0].detach().cpu()
            buffer['s2.wav'] = srcs_list[0][1].detach().cpu()
        return buffer


    def log_evaluation(self, buffer, step, split='test'):
        if self.rank != 0:
            return
        print('logging evaluation')
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
