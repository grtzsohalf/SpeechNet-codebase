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
from bin.se.preprocessor import *
from bin.se.dataset import *
from bin.se.model import *
from bin.se.objective import *
from bin.se.evaluation import *
from bin.se.utils import *

from bin.tts.solver import log_mel_to_wav
from src.transformer.nets_utils import make_non_pad_mask

MAX_TIMESTAMPS = 50
LOG_WAV_NUM = 5


def logging(logger, step, tag, data, mode='scalar', preprocessor=None):
    if type(data) is torch.Tensor:
        data = data.detach().cpu()
    tag = f'se/{tag}'

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


def collate_fn(data):
    # wavs: [(seqlen, channel), ...]
    wavs = [d[0] for d in data]
    paths = [d[1] for d in data]
    lengths = torch.LongTensor([len(s) for s in wavs])
    padded_wavs = pad_sequence(wavs, batch_first=True).transpose(-1, -2).contiguous()
    # padded_wavs: (batch_size, channel, seqlen)
    # padded_features: (batch_size, seqlen, featdim)
    return padded_wavs, lengths, paths


class SolverSE():
    ''' Handler for complete training and evaluation progress of downstream models '''
    def __init__(self, gpu_rank, world_size, rank, config, paras, mode, log, eps=1e-6):
        self.device = gpu_rank
        self.world_size = world_size
        self.rank = rank

        self.mode = mode
        self.args = paras
        self.config = config
        self.datarc = config['data']['se_corpus']
        self.modelrc = config['model']['se']

        # audio 
        self.sr = self.datarc['sample_rate']
        self.n_fft = self.datarc['n_fft']
        self.hop_length = self.datarc['hop_length']
        self.win_length = self.datarc['win_length']

        # Preprocessors
        # used for extract input feature for all tasks
        self.input_extracter, self.input_featdim, sr = create_transform(config['data']['audio'])
        self.input_extracter = self.input_extracter[:-1].to(device=self.device)
        assert self.sr == sr

        # used for extract noisy phase and target clean linear
        self.preprocessor = OnlinePreprocessor(
            sample_rate=self.sr,
            **self.datarc['preprocessor'],
        ).to(self.device)

        # Datasets
        all_trainset = NoisyCleanDataset(sample_rate=self.sr, **self.datarc['trainset'])
        self.trainset = all_trainset.get_subset(ratio=self.datarc['dev_ratio'], select_sampled=False, seed=self.datarc['seed'])
        self.devset = all_trainset.get_subset(ratio=self.datarc['dev_ratio'], select_sampled=True, seed=self.datarc['seed'])
        self.testset = NoisyCleanDataset(sample_rate=self.sr, **self.datarc['testset'])

        # Reset dataloaders for train/dev/test
        self.set_trainloader()
        self.set_devloader()
        self.set_testloader()

        self.criterion = eval(f'{self.modelrc["objective"]["name"]}')(**self.modelrc["objective"])
        self.mse = nn.MSELoss()
        self.metrics = [eval(f'{m}_eval') for m in self.modelrc['metrics']]
        self.ascending = torch.arange(MAX_TIMESTAMPS * self.sr).to(device=self.device)
        self.eps = eps

        self.logger = partial(logging, logger=log, preprocessor=copy.deepcopy(self.preprocessor).cpu())
        self.best_metrics = torch.ones(len(self.modelrc['metrics'])) * -100


    def set_trainloader(self):
        self.tr_sampler = DistributedSampler(self.trainset, num_replicas=self.world_size, rank=self.rank)
        self.trainloader = DataLoader(
            self.trainset,
            num_workers=self.args.njobs,
            collate_fn=collate_fn,
            shuffle=True if self.rank is None else False,
            sampler=None if self.rank is None else self.tr_sampler,
            **self.datarc['trainloader'],
        )


    def set_devloader(self):
        self.devloader = DataLoader(
            self.devset,
            num_workers=self.args.njobs,
            collate_fn=collate_fn,
            **self.datarc['evalloader'],
        )


    def set_testloader(self):
        self.testloader = DataLoader(
            self.testset,
            num_workers=self.args.njobs,
            collate_fn=collate_fn,
            **self.datarc['evalloader'],
        )


    def get_model(self):
        input_dim = self.config['model']['audio_decoder']['out_dim']
        target_dim = self.preprocessor(feat_list=[OnlinePreprocessor.get_feat_config('linear')])[0].size(-1)
        return eval(f'{self.modelrc["model"]["name"]}')(input_dim, target_dim, **self.modelrc["model"])


    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = self.ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1)).long()
        return length_masks


    #def forward(self, se_data, audio_encoder, audio_decoder, model, **kwargs):
    def forward(self, se_data, audio_encoder, audio_decoder, prosody_predictor, model, **kwargs):
        padded_wavs, lengths, paths = se_data
        padded_wavs, lengths = padded_wavs.to(self.device), lengths.to(self.device)

        spk = []
        for f in paths:
            spk.append(f.split('/')[-3])

        def extracter_wrapper(wavs, lens, extracter):
            bsz, chn, seqlen = wavs.shape
            wavs = wavs.view(-1, seqlen)
            lens = lens.unsqueeze(-1).expand(-1, 2).reshape(-1)
            feats, stft_lens = extracter((wavs, lens))
            feats = feats.view(bsz, chn, feats.size(1), -1)
            stft_lens = stft_lens.view(bsz, chn)
            return feats, stft_lens
            # feats: (batch_size, channel, seqlen, featdim)
            # stft_lens: (batch_size, channel)

        padded_features, stft_lengths = extracter_wrapper(padded_wavs, lengths, self.input_extracter)
        padded_features = padded_features[:, 0]
        stft_lengths = stft_lengths[:, 0]

        phase_inp, linear_tar, mel_inp = self.preprocessor(padded_wavs)
        if self.datarc['io_sample_pipeline']:
            stft_lengths = (lengths - self.preprocessor._win_args['win_length']) // self.preprocessor._win_args['hop_length']
            padded_features = mel_inp[:, :max(stft_lengths), :]
        else:
            padded_features, stft_lengths = padded_features.to(self.device), stft_lengths.to(self.device)

        encode_len, encode_feature_content, encode_mask_content, \
            encode_feature_speaker, encode_mask_speaker, pooled_encode_feature_speaker = \
            audio_encoder(padded_features, stft_lengths)
        _, predicted_prosody, _ = prosody_predictor(encode_feature_content + pooled_encode_feature_speaker.unsqueeze(1), encode_len)
        '''
        predicted_prosody = None
        '''
        stft_lengths, dec_x, stft_length_masks = audio_decoder(encode_len, encode_feature_content, encode_mask_content,
                encode_feature_speaker, encode_mask_speaker)

        predicted, model_results = model(features=dec_x)
        '''
        predicted = dec_x
        '''

        phase_inp = phase_inp[:, :max(stft_lengths), :]
        linear_tar = linear_tar[:, :max(stft_lengths), :]
        #padded_features = padded_features[:, :max(stft_lengths), :]

        spk_emb = pooled_encode_feature_speaker.cpu().detach().numpy().tolist()

        vs = locals()
        return {key:item for key, item in vs.items() if key in [
            'predicted', 'linear_tar', 'phase_inp', 'padded_wavs',
            'lengths', 'stft_length_masks', 'predicted_prosody', 'encode_feature_speaker', 'spk_emb', 'spk'
        ]}


    def compute_loss(self, predicted, linear_tar, stft_length_masks, predicted_prosody, encode_feature_speaker, step, split='train', **kwargs):
        loss, objective_results = self.criterion(predicted, linear_tar, stft_length_masks)

        prosody_loss = self.mse(predicted_prosody.masked_select(stft_length_masks.unsqueeze(-1)), 
                encode_feature_speaker[:, :predicted_prosody.shape[1]].masked_select(stft_length_masks[:, :predicted_prosody.shape[1]].unsqueeze(-1))) 
        '''
        prosody_loss = None
        '''
        if self.rank == 0 and split=='train' and step % self.modelrc['log_step'] == 0:
            self.logger(step=step, tag='train_loss', data=loss.item(), mode='scalar')
            self.logger(step=step, tag='prosody_loss', data=prosody_loss.item(), mode='scalar')
        return loss + prosody_loss


    def init_evaluation(self, split='test'):
        random.seed(self.datarc['seed'])
        setattr(self, f'{split}_buffer', defaultdict(list))


    def compute_metrics(self, predicted, linear_tar, phase_inp, stft_length_masks,
                        padded_wavs, lengths, split='test', **kwargs):
        assert hasattr(self, f'{split}_buffer')
        buffer = getattr(self, f'{split}_buffer')

        wav_inp = padded_wavs[:, 0, :]    
        wav_tar = padded_wavs[:, 1, :]

        # compute loss
        loss, objective_results = self.criterion(predicted, linear_tar, stft_length_masks)
        '''
        loss, objective_results = self.criterion(predicted, padded_features[:, :, :predicted.shape[2]], stft_length_masks)
        '''

        # reconstruct waveform
        wav_predicted = self.preprocessor.istft(predicted.exp(), phase_inp)
        wav_predicted = torch.cat([wav_predicted, wav_predicted.new_zeros(wav_predicted.size(0), max(lengths) - wav_predicted.size(1))], dim=1)
        wav_predicted = masked_normalize_decibel(wav_predicted, wav_tar, self._get_length_masks(lengths))
        '''
        wav_predicted_list = []
        for mel_predicted, mask in zip(predicted, stft_length_masks):
            wav_predicted_list.append(torch.tensor(log_mel_to_wav(mel_predicted.masked_select(mask.unsqueeze(-1)).view(-1, 80).cpu().detach().numpy(), 
                self.sr, self.n_fft, self.hop_length, self.win_length)))
        '''

        # split batch into list of utterances and duplicate N_METRICS times
        batch_size = len(wav_predicted)
        wav_predicted_list = wav_predicted.detach().cpu().chunk(batch_size) * len(self.metrics)
        '''
        wav_predicted_list = wav_predicted_list * len(self.metrics)
        '''
        wav_tar_list = wav_tar.detach().cpu().chunk(batch_size) * len(self.metrics)
        lengths_list = lengths.detach().cpu().tolist() * len(self.metrics)

        # prepare metric function for each utterance in the duplicated list
        ones = torch.ones(batch_size).long().unsqueeze(0).expand(len(self.metrics), -1)
        metric_ids = ones * torch.arange(len(self.metrics)).unsqueeze(-1)
        metric_fns = [self.metrics[idx.item()] for idx in metric_ids.reshape(-1)]
        
        def calculate_metric(length, predicted, target, metric_fn):
            return metric_fn(predicted.squeeze()[:length], target.squeeze()[:length])
        scores = Parallel(n_jobs=self.args.njobs)(delayed(calculate_metric)(l, p, t, f)
                            for l, p, t, f in zip(lengths_list, wav_predicted_list, wav_tar_list, metric_fns))

        scores = torch.FloatTensor(scores).view(len(self.metrics), batch_size).mean(dim=1)
        buffer['scores'].append(scores)
        buffer['loss'].append(loss.item())
        if len(buffer['predicted.wav']) < LOG_WAV_NUM:
            buffer['predicted.wav'].append(wav_predicted[0].detach().cpu())
            buffer['noisy.wav'].append(wav_inp[0].detach().cpu())
            buffer['clean.wav'].append(wav_tar[0].detach().cpu())


    def log_evaluation(self, step, split='test', save_fn=None):
        if self.rank != 0:
            return

        assert hasattr(self, f'{split}_buffer')
        buffer = getattr(self, f'{split}_buffer')

        for key, item in buffer.items():
            assert type(item) is list

            if 'scores' in key:
                scores = torch.stack(item, dim=0).mean(dim=0).unbind(0)
                for idx, (score, name) in enumerate(zip(scores, self.modelrc['metrics'])):
                    self.logger(step=step, tag=f'{split}_{name}', data=score.item(), mode='scalar')
                    print(f'[SE] - {split}_{name} = {score.item()}')

                    if score > self.best_metrics[idx]:
                        self.best_metrics[idx] = score
                        save_fn(f'se_best_{name}.pth', {f'se_best_{name}': score.item()}, show_msg=True)

            elif 'loss' in key:
                loss = torch.FloatTensor(item).mean()
                self.logger(step=step, tag=f'{split}_loss', data=loss, mode='scalar')
                print(f'[SE] - {split}_loss = {loss}')

            elif 'wav' in key:
                for i, wav in enumerate(item):
                    self.logger(step=step, tag=f'{i}.{split}_{key}', data=wav, mode='audio')

        delattr(self, f'{split}_buffer')
