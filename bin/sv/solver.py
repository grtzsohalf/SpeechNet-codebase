import os
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tensorboardX import SummaryWriter

from bin.sv.model import *
from bin.sv.utils_eer import *
from src.audio import *
from bin.sv.dataset import *


def collate_test_fn(data_sample, input_transform):
    # wavs: [(wavs[0:utterance_length],lengths[0:utterance_length]), ...]
    wavs1 = []
    wavs2 = []
    ylabels = []
    lengths1 = []
    lengths2 = []

    for samples in data_sample:
        wavs1.append(samples[0])
        wavs2.append(samples[1])
        ylabels.append(samples[2])
        lengths1.append(samples[3])
        lengths2.append(samples[4])

    all_wav = []
    all_wav.extend(wavs1)
    all_wav.extend(wavs2)

    all_length = []
    all_length.extend(lengths1)
    all_length.extend(lengths2)

    length_tensor = torch.stack(all_length)
    back_wavs = pad_sequence(all_wav, batch_first=True)
    back_length = length_tensor

    return back_wavs, back_length, ylabels


def collate_fn(data_sample, input_transform):
    # wavs: [(wavs[0:utterance_length],lengths[0:utterance_length]), ...]
    wavs = []
    lengths = []
    indexes = []

    for samples in data_sample:
        wavs.extend(samples[0])
        lengths.extend(samples[1])

    length_tensor = torch.stack(lengths)
    back_wavs = pad_sequence(wavs, batch_first=True)
    back_length = length_tensor

    return back_wavs, back_length


class SolverSV():
    '''Solver for training'''

    def __init__(self, gpu_rank, world_size, rank, config, paras, mode, log, eps=1e-6):
        self.device = gpu_rank
        self.world_size = world_size
        self.rank = rank

        self.mode = mode
        self.args = paras
        self.config = config

        self.datarc = config['data']['sv_corpus']
        self.post_process = config['model']['sv']

        self.sr = self.datarc['sample_rate']
        self.utter_num = self.datarc['utterance_sample']
        self.maxtimestep = self.datarc['max_timestep']

        self.log_step = self.post_process['log_step']

        # use to extract same feature on all task
        self.input_extracter, self.input_featdim, self.sample_rate = create_transform(
            config['data']['audio'])

        with open(self.datarc["valid_spkr_path"], "r") as f:
            spkr_ids = [line.rstrip() for line in f]

        # Create dataset
        self.TrainSet = Voxceleb_Dataset(file_path=self.datarc['train_file_path'],
                                         max_timestep=self.datarc['max_timestep'],
                                         dev_spkr_ids=spkr_ids,
                                         utterance_sample=self.datarc['utterance_sample'])

        self.DevSet = Voxceleb_Dataset(file_path=self.datarc['dev_file_path'],
                                       is_dev=True,
                                       meta_data=self.datarc['dev_meta_data'],
                                       dev_spkr_ids=spkr_ids)

        self.EvalSet = Voxceleb_Dataset(file_path=self.datarc['test_file_path'],
                                        is_eval=True,
                                        meta_data=self.datarc['test_meta_data'])

        # Reset dataloaders for train/dev/test
        self.set_train_dataloader()
        self.set_dev_dataloader()
        self.set_test_dataloader()

        self.post_module = eval(f'{self.post_process["model"]["name"]}')(
            self.config['model']['audio_encoder']['dim'])
        self.score_fn = nn.CosineSimilarity(dim=2)
        self.criterion = eval(f'{self.post_process["objective"]["name"]}')()
        self.metric = eval(f'{self.post_process["metric"]["name"]}')
        self.eps = eps
        self.recorder = log
    
    def transform_data(self,wavs, lengths, audio_transform):
        wave_lengths = [torch.tensor(lengths[i][0]).long() for i in range(wavs.shape[0])]  

        with torch.no_grad():
            feats, feat_lengths = audio_transform((wavs.to(self.device), torch.stack(wave_lengths).to(self.device)))
            
        # with torch.no_grad():
        #     for i in range(len(lengths)):
        #         feat=audio_transform(wavs[i][:lengths[i]].view(1,-1).to(self.device))
        #         feat_length = torch.tensor(feat.shape[0]).long()
        #         feats.append(feat)
        #         feat_lengths.append(feat_length)
        # import ipdb; ipdb.set_trace()
        # print(feats)

        return wavs.to(self.device), lengths.to(self.device), feats, feat_lengths

        

    def forward(self,sv_data, audio_encoder, audio_transform, **kwargs):

        padded_wavs, lengths = sv_data

        padded_wavs, lengths, padded_features, stft_lengths = self.transform_data(padded_wavs, lengths, audio_transform)

        enc_len, input_x_content, enc_mask_content, input_x_speaker = audio_encoder(padded_features, stft_lengths)

        agg_vec = input_x_speaker

        vs = locals()

        return {key:item for key, item in vs.items() if key in ['agg_vec','input_x_speaker','padded_wavs','lengths']}
    
    def compute_loss(self, agg_vec, step, split="train", **kwargs):
        agg_vec = agg_vec.reshape(-1, self.utter_num, agg_vec.shape[-1])
        loss=self.criterion(agg_vec)    

        return loss
    
    def compute_metrics(self, input_x_speaker, ylabel, split="test", **kwargs):
        wav1 = []
        wav2 = []
        for i in range(len(ylabel)):
            wav1.append(input_x_speaker[i].unsqueeze(0))
            wav2.append(input_x_speaker[len(ylabel)+i].unsqueeze(0))
        wav1 = torch.stack(wav1)
        wav1 = wav1 / torch.norm(wav1, dim=-1).unsqueeze(-1)
        wav2 = torch.stack(wav2)
        wav2 = wav2 / torch.norm(wav2, dim=-1).unsqueeze(-1)
        ylabel = torch.stack(ylabel).cpu().detach().long().tolist()
        scores = self.score_fn(wav1,wav2).squeeze().cpu().detach().tolist()
        return scores, ylabel
    
    def calculate_metric(self, scores,labels):
        return self.metric(np.array(labels),np.array(scores))
        


    def set_train_dataloader(self):
        self.tr_sampler = DistributedSampler(self.TrainSet, num_replicas=self.world_size, rank=self.rank)
        self.train_dataloader = DataLoader(self.TrainSet,
                                           num_workers=self.args.njobs,
                                           collate_fn=partial(
                                               collate_fn, input_transform=self.input_extracter),
                                           shuffle=True if self.rank is None else False,
                                           sampler=None if self.rank is None else self.tr_sampler,
                                           **self.datarc['train_dataloader'])

    def set_dev_dataloader(self):
        self.dev_dataloader = DataLoader(self.DevSet,
                                         num_workers=self.args.njobs,
                                         collate_fn=partial(
                                             collate_test_fn, input_transform=self.input_extracter[:-1]),
                                        sampler=None if self.rank is None else DistributedSampler(
                                               self.DevSet, num_replicas=self.world_size, rank=self.rank),
                                         **self.datarc['dev_dataloader'],)

    def set_test_dataloader(self):
        self.test_dataloader = DataLoader(self.EvalSet,
                                          num_workers=self.args.njobs,
                                          collate_fn=partial(
                                              collate_test_fn, input_transform=self.input_extracter[:-1]),
                                          **self.datarc['eval_dataloader'],)
    
    def logging(self, string, score, step):
        self.recorder.add_scalars(string, {string:score}, step)
