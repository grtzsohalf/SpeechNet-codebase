import os
import glob
import math
import copy
import random
from functools import partial
from collections import defaultdict
from joblib import Parallel, delayed
import librosa

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from src.module import VGGExtractor, CNNExtractor, CNNUpsampler, LinearUpsampler, PseudoUpsampler, RNNLayer

from bin.tts.dataset import *
from src.audio import *
from bin.tts.preprocessor import *
from bin.tts.loss import FastSpeech2Loss
from bin.tts.utils import get_mask_from_lengths
from bin.tts.model import SpeakerIntegrator, VarianceAdaptor

from src.transformer.nets_utils import make_non_pad_mask

import wandb


import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

MAX_TIMESTAMPS = 50
LOG_WAV_NUM = 5


def log_mel_to_wav(log_mel, sr, n_fft, hop_length, win_length):
    mel = np.exp(log_mel) - 1e-6
    mel = mel.T # 80(num_mel) is the first dim

    S = librosa.feature.inverse.mel_to_stft(
        mel, power=1, sr=sr, n_fft=n_fft, fmin=0)

    wav = librosa.core.griffinlim(
        S, n_iter=32, hop_length=hop_length, win_length=win_length)   
    
    return wav

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

class SolverTTS():
    ''' Handler for complete training and evaluation progress of downstream models '''
    def __init__(self, gpu_rank, world_size, rank, config, paras, mode, log, eps=1e-6):
        self.device = gpu_rank
        self.world_size = world_size
        self.rank = rank

        self.mode = mode
        self.args = paras
        self.config = config

        #!!
        self.datarc = config['data']['tts_corpus']
        self.modelrc = config['model']

        # audio 
        self.sr = self.datarc['sampling_rate']
        self.n_fft = self.datarc['n_fft']
        self.hop_length = self.datarc['hop_length']
        self.win_length = self.datarc['win_length']

        self.log_step = config['model']['tts']['log_step']
        # Preprocessors
        # used for extract input feature for all tasks
        #!!
        self.input_extracter, self.input_featdim, sr = create_transform(config['data']['audio'])
        # used for extract noisy phase and target clean linear
        """
        self.preprocessor = OnlinePreprocessor(
            sample_rate=self.sr,
            **self.datarc['preprocessor'],
        ).to(self.device)
        """
        #! Datasets
        self.trainset = LibriTTSDataset(self.datarc, "train.txt", "train.txt")
        self.devset   = LibriTTSDataset(self.datarc, "val.txt", "train.txt")
        self.testset  = LibriTTSDataset(self.datarc, "val.txt", "train.txt")
        # Reset dataloaders for train/dev/test
        self.set_trainloader()
        self.set_devloader()
        self.set_testloader()
        # === comment first === #
        # ask
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()
        #self.criterion = eval(f'{self.modelrc["objective"]["name"]}')(**self.modelrc["objective"])
        #self.metrics = [eval(f'{m}_eval') for m in self.modelrc['metrics']]
        self.ascending = torch.arange(MAX_TIMESTAMPS * self.sr).to(device=self.device)
        self.eps = eps
        self.recorder = log

        self.best_loss = 40666888.0

    def set_trainloader(self):
        self.tr_sampler = DistributedSampler(self.trainset, num_replicas=self.world_size, rank=self.rank)
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=self.datarc['batch_size'],
            num_workers=self.args.njobs,
            collate_fn=self.trainset.collate_fn,
            shuffle=True if self.rank is None else False,
            sampler=None if self.rank is None else self.tr_sampler,
            drop_last=True
        )


    def set_devloader(self):
        self.devloader = DataLoader(
            self.devset,
            batch_size=self.datarc['batch_size']//4,
            num_workers=self.args.njobs,
            collate_fn=self.devset.collate_fn,
            shuffle=True if self.rank is None else False,
            sampler=None if self.rank is None else DistributedSampler(self.devset, num_replicas=self.world_size, rank=self.rank),
            drop_last=False
        )


    def set_testloader(self):
        self.testloader = DataLoader(
            self.testset,
            num_workers=self.args.njobs,
            collate_fn=self.testset.collate_fn
        )


    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = self.ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1)).long()
        return length_masks
    
    def get_model(self):
        spk_table, inv_spk_table = self.trainset.get_spk_table()
        speaker_integrator = SpeakerIntegrator(self.datarc, spk_table)
        variance_adaptor = VarianceAdaptor(self.modelrc['variance_adaptor'], self.datarc, self.device)

        prenet = self.config['model']['audio_encoder']['prenet']
        dim = self.config['model'] ['text_encoder']['att_d_feat']
        if prenet=='vgg':
            extractor = VGGExtractor(dim, hide_dim=dim)
        elif prenet=='cnn':
            extractor = CNNExtractor(dim, out_dim=dim)
        else:
            extractor = PseudoDownsampler(dim, dim)
        
        return speaker_integrator, variance_adaptor, extractor


    def forward(self, mode, tts_data, audio_transform, audio_encoder, text_encoder, audio_decoder, prosody_predictor,
                speaker_integrator, variance_adaptor, extractor, **kwargs):
        device = self.device
        """ For debugging, print parameters of text encoder and audio decoder
        p_text = sum(p.numel() for p in text_encoder.parameters())
        p_audio =sum(p.numel() for p in audio_decoder.parameters())
        print(p_text, p_audio)

        for name, param in text_encoder.named_parameters():
            if param.requires_grad:
                print (name, param.data.shape)
        print("-----------------------------")
        for name, param in audio_decoder.named_parameters():
            if param.requires_grad:
                print (name, param.data.shape)        
        assert 1==2
        """
        

        # tts_data : list : 16
        # dict_keys(['id', 'text', 'mel_target', 'D', 'log_D', 'f0', 'energy', 'src_len', 'mel_len', 'spk_ids']) 
        spk_ids     = torch.tensor(tts_data["spk_ids"]).to(torch.int64).to(device)
        text        = torch.from_numpy(tts_data["text"]).long().to(device)
        mel_target  = torch.from_numpy(tts_data["mel_target"]).float().to(device)
        D           = torch.from_numpy(tts_data["D"]).long().to(device)
        log_D       = torch.from_numpy(tts_data["log_D"]).float().to(device)
        f0          = torch.from_numpy(tts_data["f0"]).float().to(device)
        energy      = torch.from_numpy(tts_data["energy"]).float().to(device)
        src_len     = torch.from_numpy(tts_data["src_len"]).long().to(device)
        mel_len     = torch.from_numpy(tts_data["mel_len"]).long().to(device)
        max_src_len = np.max(tts_data["src_len"]).astype(np.int32)
        max_mel_len = np.max(tts_data["mel_len"]).astype(np.int32)

        # Split into data and examples
        if mode != 'train':
            batch_size = len(spk_ids)

            example_spk_ids = spk_ids[batch_size:]
            example_text = text[batch_size:]
            example_mel_target = mel_target[batch_size:]
            example_D = D[batch_size:]
            example_log_D = log_D[batch_size:]
            example_f0 = f0[batch_size:]
            example_energy = energy[batch_size:]
            example_src_len = src_len[batch_size:]
            example_mel_len = mel_len[batch_size:]

            spk_ids = spk_ids[:batch_size]
            text = text[:batch_size]
            mel_target = mel_target[:batch_size]
            D = D[:batch_size]
            log_D = log_D[:batch_size]
            f0 = f0[:batch_size]
            energy = energy[:batch_size]
            src_len = src_len[:batch_size]
            mel_len = mel_len[:batch_size]
        
        # ===   text encoder   === #
        src_mask = get_mask_from_lengths(self.device, src_len, max_src_len)
        mel_mask = get_mask_from_lengths(self.device, mel_len, max_mel_len)
        encoder_output = text_encoder(text, src_mask) 
        
        # === Variance and Speaker Modules === #
        table_speaker_embedding = speaker_integrator.get_speaker_embedding(spk_ids)
        # encoder_output = speaker_integrator(speaker_embedding, encoder_output)

        if mode == 'train':
            postprocessed_mel, _ = audio_transform[1:-1]((mel_target.permute(0, 2, 1), mel_len))
        else:
            postprocessed_mel, _ = audio_transform[1:-1]((example_mel_target.permute(0, 2, 1), mel_len))
        _, _, _, speaker_embedding, speaker_mask, pooled_speaker_embedding = audio_encoder(postprocessed_mel, mel_len)

        spk_emb = pooled_speaker_embedding.cpu().detach().numpy().tolist()
        spk = tts_data['spk_ids']

        
        '''
        # Downsample
        mel_len = mel_len//audio_encoder.sample_rate
        max_mel_len = max_mel_len//audio_encoder.sample_rate
        downsampled_mel_mask = get_mask_from_lengths(self.device, mel_len, max_mel_len)
        D = torch.clamp(D//audio_encoder.sample_rate, min=1)
        log_D = torch.log(torch.exp(log_D-1)/audio_encoder.sample_rate) + 1
        '''

        if mode == 'train':
            d_target = D
            p_target = f0
            e_target = energy
        else:
            d_target = example_D
            p_target = example_f0
            e_target = example_energy
        
        #variance_adaptor_output, d_prediction, p_prediction, e_prediction, _, _ = variance_adaptor(
        #    encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        variance_adaptor_output, d_prediction,  _, _ = variance_adaptor(
                table_speaker_embedding, encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)

        #variance_adaptor_output = speaker_integrator(speaker_embedding, variance_adaptor_output)
        
        
        # === audio decoder === #
        # mel_len : torch.Size()
        # variance_adaptor_output : torch.Size([2, 872, 256])  [batch_size, seq_len, h_dim]
        # mel_mask : torch.Size([2, 872]) [batch_size, seq_len]
        # speaker_embedding : torch.Size([2, 256])


        # dec_len, mel_out, dec_mask = audio_decoder(mel_len , None, ~mel_mask.unsqueeze(1), speaker_embedding, upsampled_x=variance_adaptor_output)
        variance_adaptor_output, mel_len = extractor(variance_adaptor_output, mel_len)
        downsampled_mel_mask = get_mask_from_lengths(self.device, mel_len, max_mel_len//audio_encoder.sample_rate)

        _, predicted_prosody, _ = prosody_predictor(variance_adaptor_output + table_speaker_embedding.unsqueeze(1), mel_len)
        '''
        predicted_prosody = speaker_embedding
        '''

        dec_len, mel_out, dec_mask = audio_decoder(mel_len , variance_adaptor_output, ~downsampled_mel_mask.unsqueeze(1), 
                                                   predicted_prosody, speaker_mask, upsampled_x=None)
        
        mel_target = mel_target[:, :mel_out.shape[1], :]
        mel_mask = mel_mask[:, :mel_out.shape[1]]
        
        vs = locals()
        return {key:item for key, item in vs.items() if key in [
            'mel_out', 'mel_target',
            'd_prediction', 'log_D',
            'src_mask', 'mel_mask',
            'speaker_embedding',
            'pooled_speaker_embedding',
            'table_speaker_embedding',
            'predicted_prosody',
            'spk_emb',
            'spk'
        ]}
        

    def compute_loss(self, mel_out , mel_target, 
                     d_prediction, log_D, 
                     src_mask , mel_mask,
                     speaker_embedding,
                     pooled_speaker_embedding,
                     table_speaker_embedding,
                     predicted_prosody, **kwargs):
        log_D.requires_grad = False
        #f0.requires_grad = False
        #energy.requires_grad = False
        mel_target.requires_grad = False
        
        src_mask = ~src_mask
        mel_mask = ~mel_mask

        d_prediction = d_prediction.masked_select(src_mask)
        log_D        = log_D.masked_select(src_mask)
        
        #p_prediction = p_prediction.masked_select(mel_mask)
        #f0           = f0.masked_select(mel_mask)
        
        #e_prediction = e_prediction.masked_select(mel_mask)
        #energy       = energy.masked_select(mel_mask)
        mel_out = mel_out.masked_select(mel_mask.unsqueeze(-1))
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

        speaker_loss = self.mse_loss(pooled_speaker_embedding, table_speaker_embedding)
        mel_loss = self.mse_loss(mel_out, mel_target)
        d_loss = self.mae_loss(d_prediction, log_D)
        #p_loss = self.mae_loss(p_prediction, f0)
        #e_loss = self.mae_loss(e_prediction, energy)        
        prosody_loss = self.mse_loss(predicted_prosody.masked_select(mel_mask.unsqueeze(-1)), 
                speaker_embedding[:, :mel_mask.shape[1], :].masked_select(mel_mask.unsqueeze(-1)))
        '''
        prosody_loss = None
        '''
        
        total_loss = mel_loss + d_loss + speaker_loss + prosody_loss
        
        return total_loss, mel_loss, d_loss, speaker_loss, prosody_loss
    
    # for dev set and test set but not using test set now 
    def compute_metrics(self, mel_out , mel_target, 
                        d_prediction, log_D, 
                        src_mask , mel_mask, 
                        speaker_embedding,
                        pooled_speaker_embedding,
                        table_speaker_embedding,
                        predicted_prosody,
                        split="dev", synthesis=False, **kwargs):
        
        total_loss, mel_loss, d_loss, speaker_loss, prosody_loss = self.compute_loss(
            mel_out , mel_target, 
            d_prediction, log_D, 
            src_mask , mel_mask,
            speaker_embedding,
            pooled_speaker_embedding,
            table_speaker_embedding,
            predicted_prosody)
        if synthesis:
            idx = random.randint(0, len(mel_out)-1)
            _mel_mask = ~mel_mask[idx]
            _mel_mask = _mel_mask.unsqueeze(-1).repeat(1, 80) #num_mel
            log_mel_model = torch.reshape(mel_out[idx].masked_select(_mel_mask), (-1, 80)).cpu().detach().numpy()
            log_mel_target = torch.reshape(mel_target[idx].masked_select(_mel_mask), (-1, 80)).cpu().detach().numpy()
            #print(log_mel_model.shape, log_mel_target.shape)
            #print(self.sr, self.n_fft, self.hop_length, self.win_length)
            wav_model = log_mel_to_wav(log_mel_model, self.sr, self.n_fft, self.hop_length, self.win_length)
            wav_target = log_mel_to_wav(log_mel_target, self.sr, self.n_fft, self.hop_length, self.win_length)
        else:
            log_mel_model = None
            log_mel_target = None
            wav_model = None
            wav_target = None
        return total_loss, mel_loss, d_loss, speaker_loss, prosody_loss, log_mel_model, log_mel_target, wav_model, wav_target
        
    def logging(self, tag, data, step, mode):
        if mode == 'scalar':
            self.recorder.add_scalar(tag, data, step)
            #wandb.log({tag:data}, step=step)
        
        if mode == 'image':
            self.recorder.add_image(tag, plot_spectrogram_to_numpy(data.T), step, dataformats='HWC')

        elif mode == 'wav':
            data = data.reshape(1, -1)
            self.recorder.add_audio(f'{tag}.wav', data, global_step=step, sample_rate=self.sr)
