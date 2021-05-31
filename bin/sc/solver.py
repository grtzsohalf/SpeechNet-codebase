import os
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tensorboardX import SummaryWriter

from bin.sc.model import *
from src.audio import *
from bin.sc.dataset import *


def collate_fn(samples):
    wavs, lengths, labels = [], [], []

    for wav,length,label in samples:
        wavs.append(wav)
        lengths.append(length)
        labels.append(label)
    
    back_wavs = pad_sequence(wavs, batch_first=True)
    length_tensor = torch.stack(lengths)
    labels_tensor = torch.stack(labels)

    return back_wavs, length_tensor, labels_tensor


class SolverSC():
    '''Solver for training'''

    def __init__(self, gpu_rank, world_size, rank, config, paras, mode, log, eps=1e-6):
        self.device = gpu_rank
        self.world_size = world_size
        self.rank = rank

        self.mode = mode
        self.args = paras
        self.config = config

        self.datarc = config['data']['sc_corpus']
        self.post_process = config['model']['sc']

        self.sr = self.datarc['sample_rate']
        self.log_step = self.post_process['log_step']

        # use to extract same feature on all task
        self.input_extracter, self.input_featdim, self.sample_rate = create_transform(
            config['data']['audio'])
        self.input_extracter = self.input_extracter[:-1]

        # Create dataset
        self.TrainSet = SpeakerClassifiDataset('train', file_path=self.datarc['file_path'],meta_data=self.datarc['meta_data'], max_timestep=self.datarc['max_timestep'])

        self.DevSet = SpeakerClassifiDataset('dev', file_path=self.datarc['file_path'],meta_data=self.datarc['meta_data'], max_timestep=self.datarc['max_eval_timestep'])

        self.EvalSet = SpeakerClassifiDataset('test', file_path=self.datarc['file_path'],meta_data=self.datarc['meta_data'],max_timestep=self.datarc['max_eval_timestep'])

        # Reset dataloaders for train/dev/test
        self.set_train_dataloader()
        self.set_dev_dataloader()
        self.set_test_dataloader()

        self.post_module = eval(f'{self.post_process["model"]["name"]}')(
            self.config['model']['audio_encoder']['dim'], self.datarc['speaker_num'])
        
        self.objective = nn.CrossEntropyLoss()
        self.eps = eps
        self.recorder = log
        self.best_acc = {'accuracy': 0.05}

    
    def transform_data(self, wavs, lengths, audio_transform):
        wave_lengths = [torch.tensor(lengths[i][0]).long() for i in range(wavs.shape[0])]  
        
        with torch.no_grad():
            feats, feat_lengths = audio_transform((wavs.to(self.device), torch.stack(wave_lengths).to(self.device)))
        
        return wavs.to(self.device), lengths.to(self.device), feats, feat_lengths

        

    def forward(self,sc_data, audio_encoder, audio_transform, **kwargs):
        padded_wavs, lengths, labels = sc_data
        padded_wavs, lengths, padded_features, stft_lengths = self.transform_data(padded_wavs, lengths, audio_transform)
         
        enc_len, input_x_content, enc_mask_content, \
            input_x_speaker, enc_mask_speaker, pooled_input_x_speaker = audio_encoder(padded_features, stft_lengths)
        agg_vec = pooled_input_x_speaker
        predicted=self.post_module(agg_vec)

        spk_emb = pooled_input_x_speaker.cpu().detach().numpy().tolist()
        spk = labels.cpu().detach().numpy().tolist()

        vs = locals()

        return {key:item for key, item in vs.items() if key in ['agg_vec','pooled_input_x_speaker','padded_wavs','lengths', 'labels', 'predicted', 'spk_emb', 'spk']}
    
    def compute_loss(self, predicted, labels, step, split="train", **kwargs):
        
        loss = self.objective(predicted, labels.squeeze(1).cuda(self.device))    
        return loss
    
    def compute_metrics(self, predicted, labels, split="dev", **kwargs):
        predicted_classid = predicted.max(dim=-1).indices
        return (predicted_classid == labels.cuda(self.device).squeeze(1)).view(-1).cpu().float().tolist()
    
    def calculate_metric(self, true_list):
        return np.sum(np.array(true_list)) / len(np.array(true_list))
        


    def set_train_dataloader(self):
        self.tr_sampler = DistributedSampler(self.TrainSet, num_replicas=self.world_size, rank=self.rank)
        self.train_dataloader = DataLoader(self.TrainSet,
                                           num_workers=self.args.njobs,
                                           collate_fn=partial(
                                               collate_fn),
                                           shuffle=True if self.rank is None else False,
                                           sampler=None if self.rank is None else self.tr_sampler,
                                           **self.datarc['train_dataloader'])

    def set_dev_dataloader(self):
        self.dev_dataloader = DataLoader(self.DevSet,
                                         num_workers=self.args.njobs,
                                         collate_fn=partial(
                                             collate_fn),
                                        sampler=None if self.rank is None else DistributedSampler(
                                               self.DevSet, num_replicas=self.world_size, rank=self.rank),
                                         **self.datarc['dev_dataloader'],)

    def set_test_dataloader(self):
        self.test_dataloader = DataLoader(self.EvalSet,
                                          num_workers=self.args.njobs,
                                          collate_fn=partial(
                                              collate_fn),
                                          **self.datarc['eval_dataloader'],)
    
    def logging(self, string, score, step):
        self.recorder.add_scalar(string, score, step)
