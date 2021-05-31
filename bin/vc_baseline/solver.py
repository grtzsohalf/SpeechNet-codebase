import os
import glob
import math
import copy
import random
from functools import partial
from collections import defaultdict
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

import soundfile as sf
import matplotlib.pyplot as plt

from src.audio import *
from bin.vc_baseline.dataset import *
from bin.vc_baseline.model import *
from bin.vc_baseline.preprocessor import *
from bin.se.utils import *

from bin.tts.solver import log_mel_to_wav

MAX_TIMESTAMPS = 50
LOG_WAV_NUM = 10

def logging(logger, step, tag, data, mode='scalar', preprocessor=None, sample_dir=None):
    if type(data) is torch.Tensor:
        data = data.detach().cpu()
    tag = f'vc_baseline/{tag}'

    if mode == 'scalar':
        # data is a int or float
        logger.add_scalar(tag, data, global_step=step)
    elif mode == 'audio':
        # data: (seqlen, )
        assert preprocessor is not None
        data = data / data.abs().max().item()
        '''
        data = data / np.amax(np.absolute(data))
        '''
        # log wavform            
        logger.add_audio(f'{tag}.wav', data.reshape(-1, 1), global_step=step, sample_rate=preprocessor._sample_rate)
        # log log-linear-scale spectrogram
        feat_config = OnlinePreprocessor.get_feat_config(feat_type='linear', log=True)
        linear = preprocessor(data.reshape(1, 1, -1), [feat_config])[0]
        if sample_dir is not None:
            os.makedirs(sample_dir, exist_ok=True)
            os.makedirs(os.path.join(sample_dir, 'wavs'), exist_ok=True)
            os.makedirs(os.path.join(sample_dir, 'imgs'), exist_ok=True)
            suffix = tag.split('/')[-1]
            sf.write(os.path.join(sample_dir, 'wavs', f'{step}-{suffix}'), data.reshape(-1, 1), 16000)
            figure = plot_spectrogram(linear, path=os.path.join(sample_dir, 'imgs', f'{step}-{suffix}.png'))
        else:
            figure = plot_spectrogram(linear)
        logger.add_figure(f'{tag}.png', figure, global_step=step)
    else:
        raise NotImplementedError

def collate_fn(inputs, input_transform):
    # inputs: [(wav, length), ...]
    wavs, src_lengths, tgt_lengths, spk_1_list, spk_2_list = [], [], [], [], []
    for src_wav, src_length, tgt_wav, tgt_length, example_src_wav, example_src_length, example_tgt_wav, example_tgt_length, spk_1, spk_2 in inputs:
        spk_1_list.append(spk_1)
        spk_2_list.append(spk_2)
        wavs.append(src_wav)
        wavs.append(tgt_wav)
        src_lengths.append(src_length)
        tgt_lengths.append(tgt_length)
    for src_wav, src_length, tgt_wav, tgt_length, example_src_wav, example_src_length, example_tgt_wav, example_tgt_length, spk_1, spk_2 in inputs:
        if example_src_wav is None:
            break
        wavs.append(example_src_wav)
        wavs.append(example_tgt_wav)
        src_lengths.append(example_src_length)
        tgt_lengths.append(example_tgt_length)
    wavs_tensor = pad_sequence(wavs, batch_first=True)
    src_wavs_tensor = wavs_tensor[range(0, len(wavs_tensor), 2)]
    tgt_wavs_tensor = wavs_tensor[range(1, len(wavs_tensor), 2)]
    src_lengths_tensor = torch.tensor(src_lengths)
    tgt_lengths_tensor = torch.tensor(tgt_lengths)
    
    return src_wavs_tensor, src_lengths_tensor, tgt_wavs_tensor, tgt_lengths_tensor, spk_1_list, spk_2_list

class SolverVCB():
    def __init__(self, gpu_rank, world_size, rank, config, paras, mode, log, eps=1e-6):
        self.device = gpu_rank
        self.world_size = world_size
        self.rank = rank

        self.mode = mode
        self.args = paras
        self.config = config
        self.datarc = config['data']['vcb_corpus']
        self.modelrc = config['model']['vcb']
        self.log_step = self.modelrc['log_step']
        self.milestone = self.modelrc['milestone']

        # audio 
        self.sr = self.datarc['sample_rate']
        self.n_fft = self.datarc['n_fft']
        self.hop_length = self.datarc['hop_length']
        self.win_length = self.datarc['win_length']

        # used for extract input feature for all tasks
        self.input_extracter, self.input_featdim, sr = create_transform(config['data']['audio'])
        assert self.sr == sr

        self.preprocessor = OnlinePreprocessor(
            sample_rate=self.sr,
            **self.datarc['preprocessor'],
        ).to(self.device)

        self.trainset = VcDataset(split='train', data_path=self.datarc['data_path'], speaker_info_path=self.datarc['speaker_info_path'],test_speakers=self.datarc['test_speakers'],test_proportion=self.datarc['test_proportion'])
        self.devset = VcDataset(split='dev', data_path=self.datarc['data_path'], speaker_info_path=self.datarc['speaker_info_path'],test_speakers=self.datarc['test_speakers'],test_proportion=self.datarc['test_proportion'])

        self.set_trainloader()
        self.set_devloader()

        self.eps = eps
        self.recorder = log
        #self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.L1Loss()
        self.ascending = torch.arange(MAX_TIMESTAMPS * self.sr).to(device=self.device)

        self.logger = partial(logging, logger=log, preprocessor=copy.deepcopy(self.preprocessor).cpu())

        # Logger settings
        self.sample_dir = paras.sampledir
        self.best_losses = {'reconstruction': 40666888.0, 'conversion': 40666888.0, 'prosody': 40666888.0}


    def set_trainloader(self):
        self.tr_sampler = DistributedSampler(self.trainset, num_replicas=self.world_size, rank=self.rank)
        self.trainloader = DataLoader(
            self.trainset,
            num_workers=self.args.njobs,
            collate_fn=partial(collate_fn, input_transform=self.input_extracter[:-1]),
            shuffle=True if self.rank is None else False,
            sampler=None if self.rank is None else self.tr_sampler,
            **self.datarc['train_loader']
        )

    def set_devloader(self):
        self.devloader = DataLoader(
            self.devset,
            num_workers=self.args.njobs,
            collate_fn=partial(collate_fn, input_transform=self.input_extracter[:-1]),
            shuffle=True if self.rank is None else False,
            sampler=None if self.rank is None else DistributedSampler(self.devset, num_replicas=self.world_size, rank=self.rank),
            **self.datarc['dev_loader']
        )

    def init_evaluation(self, split='test'):
        random.seed(self.datarc['seed'])
        setattr(self, f'{split}_buffer', defaultdict(list))

    def get_model(self):
        input_dim = self.config['model']['audio_decoder']['out_dim']
        target_dim = self.preprocessor(feat_list=[OnlinePreprocessor.get_feat_config('linear')])[0].size(-1)
        mel2lin = eval(f'{self.modelrc["model"]}')(input_dim, target_dim)
        return mel2lin

    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = self.ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1)).long()
        return length_masks

    def transform_data(self, wavs, lengths, audio_transform):   
        wave_lengths = [torch.tensor(lengths[i]).long() for i in range(wavs.shape[0])]  
        with torch.no_grad():
            feats, feat_lengths = audio_transform((wavs.to(self.device), torch.stack(wave_lengths).to(self.device)))
        
        return wavs.to(self.device), lengths.to(self.device), feats, feat_lengths

    def forward(self, mode, vcb_data, audio_encoder, audio_decoder, prosody_predictor, audio_transform, step=None, model=None, **kwargs):
        #torch.autograd.set_detect_anomaly(True)
        src_padded_wavs, src_lengths, tgt_padded_wavs, tgt_lengths, spk_1, spk_2 = vcb_data
        src_padded_wavs, src_lengths = src_padded_wavs.to(self.device), src_lengths.to(self.device)
        tgt_padded_wavs, tgt_lengths = tgt_padded_wavs.to(self.device), tgt_lengths.to(self.device)

        # for spk examples
        if mode != 'train':
            batch_size = len(src_padded_wavs) // 2

            example_src_padded_wavs = src_padded_wavs[batch_size:]
            example_src_lengths = src_lengths[batch_size:]
            example_tgt_padded_wavs = tgt_padded_wavs[batch_size:]
            example_tgt_lengths = tgt_lengths[batch_size:]

            src_padded_wavs = src_padded_wavs[:batch_size]
            src_lengths = src_lengths[:batch_size]
            tgt_padded_wavs = tgt_padded_wavs[:batch_size]
            tgt_lengths = tgt_lengths[:batch_size]

        padded_wavs = torch.cat((src_padded_wavs, tgt_padded_wavs), 0)
        lengths = torch.cat((src_lengths, tgt_lengths), 0)
        padded_wavs, lengths, padded_features, stft_lengths = self.transform_data(padded_wavs, lengths, audio_transform)

        batch_size = padded_wavs.size(0) // 2
        src_padded_wavs, tgt_padded_wavs = padded_wavs[:batch_size], padded_wavs[batch_size:]
        src_lengths, tgt_lengths = lengths[:batch_size], lengths[batch_size:]
        src_padded_features, tgt_padded_features = padded_features[:batch_size], padded_features[batch_size:]
        src_stft_lengths, tgt_stft_lengths = stft_lengths[:batch_size], stft_lengths[batch_size:]

        # for spk examples
        if mode != 'train':
            example_padded_wavs = torch.cat((example_src_padded_wavs, example_tgt_padded_wavs), 0)
            example_lengths = torch.cat((example_src_lengths, example_tgt_lengths), 0)
            example_padded_wavs, example_lengths, example_padded_features, example_stft_lengths = \
                    self.transform_data(example_padded_wavs, example_lengths, audio_transform)

            example_src_padded_wavs, example_tgt_padded_wavs = example_padded_wavs[:batch_size], example_padded_wavs[batch_size:]
            example_src_lengths, example_tgt_lengths = example_lengths[:batch_size], example_lengths[batch_size:]
            example_src_padded_features, example_tgt_padded_features = example_padded_features[:batch_size], example_padded_features[batch_size:]
            example_src_stft_lengths, example_tgt_stft_lengths = example_stft_lengths[:batch_size], example_stft_lengths[batch_size:]

        # src_padded_wavs, src_lengths, src_padded_features, src_stft_lengths = self.transform_data(src_padded_wavs, src_lengths, audio_transform)

        # tgt_padded_wavs, tgt_lengths, tgt_padded_features, tgt_stft_lengths = self.transform_data(tgt_padded_wavs, tgt_lengths, audio_transform)

        src_phase, src_linear = self.preprocessor(src_padded_wavs.unsqueeze(1))
        tgt_phase, tgt_linear = self.preprocessor(tgt_padded_wavs.unsqueeze(1))
        
        max_stft_lengths = torch.tensor([ max(i, j) for i, j in zip(src_stft_lengths, tgt_stft_lengths)]).to(self.device)

        src_enc_len, src_content, src_enc_mask_content, \
            src_speaker, src_enc_mask_speaker, pooled_src_speaker = audio_encoder(src_padded_features, max_stft_lengths)

        tgt_enc_len, tgt_content, tgt_enc_mask_content, \
            tgt_speaker, tgt_enc_mask_speaker, pooled_tgt_speaker = audio_encoder(tgt_padded_features, max_stft_lengths)

        spk_emb = pooled_src_speaker.cpu().detach().numpy().tolist()
        spk = spk_1

        if mode != 'train':
            example_src_enc_len, example_src_content, example_src_enc_mask_content, \
                    example_src_speaker, example_src_enc_mask_speaker, example_pooled_src_speaker = \
                    audio_encoder(example_src_padded_features, max_stft_lengths)

            example_tgt_enc_len, example_tgt_content, example_tgt_enc_mask_content, \
                    example_tgt_speaker, example_tgt_enc_mask_speaker, example_pooled_tgt_speaker = \
                    audio_encoder(example_tgt_padded_features, max_stft_lengths)
        
        use_convert = True
        '''
        if step <= self.milestone: 
            use_convert = random.random() <= (step) / (self.milestone)
        '''
        src_recon_stft_lengths, src_recon, src_recon_stft_length_masks = audio_decoder(src_enc_len, src_content, src_enc_mask_content, 
                                                                           src_speaker, src_enc_mask_speaker)
        tgt_recon_stft_lengths, tgt_recon, tgt_recon_stft_length_masks = audio_decoder(tgt_enc_len, tgt_content, tgt_enc_mask_content, 
                                                                           tgt_speaker, tgt_enc_mask_speaker)
        src_recon, _ = model(features=src_recon)
        tgt_recon, _ = model(features=tgt_recon)

        # prosody prediction
        if mode == 'train':
            _, tgt_convert_predicted_prosody, _ = prosody_predictor(src_content + pooled_tgt_speaker.unsqueeze(1), tgt_enc_len)
            _, src_convert_predicted_prosody, _ = prosody_predictor(tgt_content + pooled_src_speaker.unsqueeze(1), src_enc_len)
        else:
            _, tgt_convert_predicted_prosody, _ = prosody_predictor(src_content + example_pooled_tgt_speaker.unsqueeze(1), tgt_enc_len)
            _, src_convert_predicted_prosody, _ = prosody_predictor(tgt_content + example_pooled_src_speaker.unsqueeze(1), src_enc_len)
        '''
        if mode == 'train':
            tgt_convert_predicted_prosody = tgt_speaker
            src_convert_predicted_prosody = src_speaker
        else:
            tgt_convert_predicted_prosody = example_tgt_speaker
            src_convert_predicted_prosody = example_src_speaker
        '''

        src_convert, tgt_convert = None, None
        # if use_convert:
        tgt_stft_lengths, tgt_convert, tgt_stft_length_masks = audio_decoder(tgt_enc_len, src_content, tgt_enc_mask_content, 
                                                                           tgt_convert_predicted_prosody, tgt_enc_mask_speaker)
        src_stft_lengths, src_convert, src_stft_length_masks = audio_decoder(src_enc_len, tgt_content, src_enc_mask_content, 
                                                                           src_convert_predicted_prosody, src_enc_mask_speaker)
        src_convert, _ = model(features=src_convert)
        tgt_convert, _ = model(features=tgt_convert) 

        
        src_linear = src_linear[:, :src_stft_length_masks.size(1), :]
        tgt_linear = tgt_linear[:, :tgt_stft_length_masks.size(1), :]

        src_phase = src_phase[:, :max(src_stft_lengths), :]
        tgt_phase = tgt_phase[:, :max(tgt_stft_lengths), :]

        vs = locals()

        return { key: item for key, item in vs.items() if key in [
            'use_convert', 'src_recon', 'tgt_recon', 'src_convert', 'tgt_convert', 'src_linear', 'tgt_linear','src_phase', 'tgt_phase', 'src_padded_wavs', 'tgt_padded_wavs', 'src_padded_features', 'tgt_padded_features', 'src_lengths', 'tgt_lengths', 'src_stft_length_masks', 'tgt_stft_length_masks', 'src_convert_predicted_prosody', 'tgt_convert_predicted_prosody', 'src_speaker', 'tgt_speaker', 'spk_emb', 'spk'
        ]}

    def compute_metrics(self, use_convert, src_recon, tgt_recon, src_convert, tgt_convert, src_linear, tgt_linear, src_phase, tgt_phase, src_padded_wavs, tgt_padded_wavs, src_padded_features, tgt_padded_features, src_lengths, tgt_lengths, src_stft_length_masks, tgt_stft_length_masks, split='test', **kwargs):
        if hasattr(self, f'{split}_buffer') == False:
            setattr(self, f'{split}_buffer', defaultdict(list))
        buffer = getattr(self, f'{split}_buffer')
        if len(buffer['source.wav']) >= LOG_WAV_NUM:
            return
        # conversion waveform
        maximum = max(max(src_lengths), max(tgt_lengths))
        max_lengths = torch.tensor([maximum for i in range(src_padded_wavs.size(0))]).to(self.device)

        src_recon = self.preprocessor.istft(src_recon[:, :src_phase.shape[1], :].exp(), src_phase)
        src_recon = torch.cat([src_recon, src_recon.new_zeros(src_recon.size(0), maximum - src_recon.size(1))], dim = 1)

        tgt_recon = self.preprocessor.istft(tgt_recon[:, :tgt_phase.shape[1], :].exp(), tgt_phase)
        tgt_recon = torch.cat([tgt_recon, tgt_recon.new_zeros(tgt_recon.size(0), maximum - tgt_recon.size(1))], dim = 1)
        '''
        src_recon_predicted_list = []
        for mel_predicted, mask in zip(src_recon, src_stft_length_masks):
            src_recon_predicted_list.append(log_mel_to_wav(mel_predicted.masked_select(mask.unsqueeze(-1)).view(-1, 80).cpu().detach().numpy(), 
                self.sr, self.n_fft, self.hop_length, self.win_length))
            break

        tgt_recon_predicted_list = []
        for mel_predicted, mask in zip(tgt_recon, tgt_stft_length_masks):
            tgt_recon_predicted_list.append(log_mel_to_wav(mel_predicted.masked_select(mask.unsqueeze(-1)).view(-1, 80).cpu().detach().numpy(), 
                self.sr, self.n_fft, self.hop_length, self.win_length))
            break
        '''

        buffer['src_recon.wav'].append(src_recon[0].detach().cpu())
        buffer['tgt_recon.wav'].append(tgt_recon[0].detach().cpu())
        buffer['source.wav'].append(src_padded_wavs[0].detach().cpu())
        buffer['target.wav'].append(tgt_padded_wavs[0].detach().cpu())
        '''
        buffer['src_recon.wav'].append(torch.tensor(src_recon_predicted_list[0]))
        buffer['tgt_recon.wav'].append(torch.tensor(tgt_recon_predicted_list[0]))
        buffer['source.wav'].append(src_padded_wavs[0].detach().cpu())
        buffer['target.wav'].append(tgt_padded_wavs[0].detach().cpu())
        '''
        
        src_to_tgt = self.preprocessor.istft(tgt_convert[:, :tgt_phase.shape[1], :].exp(), tgt_phase)
        src_to_tgt = torch.cat([src_to_tgt, src_to_tgt.new_zeros(src_to_tgt.size(0), maximum - src_to_tgt.size(1))], dim=1)
        # src_to_tgt = masked_normalize_decibel(src_to_tgt, tgt_linear, self._get_length_masks(max_lengths))

        tgt_to_src = self.preprocessor.istft(src_convert[:, :src_phase.shape[1], :].exp(), src_phase)
        tgt_to_src = torch.cat([tgt_to_src, tgt_to_src.new_zeros(tgt_to_src.size(0), maximum - tgt_to_src.size(1))], dim=1)
        # tgt_to_src = masked_normalize_decibel(tgt_to_src, src_linear, self._get_length_masks(max_lengths))
        '''
        src_to_tgt_predicted_list = []
        for mel_predicted, mask in zip(tgt_convert, tgt_stft_length_masks):
            src_to_tgt_predicted_list.append(log_mel_to_wav(mel_predicted.masked_select(mask.unsqueeze(-1)).view(-1, 80).cpu().detach().numpy(), 
                self.sr, self.n_fft, self.hop_length, self.win_length))
            break

        tgt_to_src_predicted_list = []
        for mel_predicted, mask in zip(src_convert, src_stft_length_masks):
            tgt_to_src_predicted_list.append(log_mel_to_wav(mel_predicted.masked_select(mask.unsqueeze(-1)).view(-1, 80).cpu().detach().numpy(), 
                self.sr, self.n_fft, self.hop_length, self.win_length))
            break
        '''

        buffer['src_to_tgt.wav'].append(src_to_tgt[0].detach().cpu())
        buffer['tgt_to_src.wav'].append(tgt_to_src[0].detach().cpu())
        '''
        buffer['src_to_tgt.wav'].append(torch.tensor(src_to_tgt_predicted_list[0]))
        buffer['tgt_to_src.wav'].append(torch.tensor(tgt_to_src_predicted_list[0]))
        '''

    def compute_loss(self, use_convert, src_recon, tgt_recon, src_convert, tgt_convert, src_padded_features, tgt_padded_features, src_linear, tgt_linear, src_stft_length_masks, tgt_stft_length_masks, src_convert_predicted_prosody, tgt_convert_predicted_prosody, src_speaker, tgt_speaker, 
            step=0, split='train', **kwargs):

        src_recon.masked_fill_(~src_stft_length_masks.unsqueeze(-1), 0)
        src_linear.masked_fill_(~src_stft_length_masks.unsqueeze(-1), 0)
        
        tgt_recon.masked_fill_(~tgt_stft_length_masks.unsqueeze(-1), 0)
        tgt_linear.masked_fill_(~tgt_stft_length_masks.unsqueeze(-1), 0)
        
        src_len = min(src_recon.size(1), src_linear.size(1))
        tgt_len = min(tgt_recon.size(1), tgt_linear.size(1))

        loss_recon = self.criterion(src_recon[:, :src_len, :], src_linear[:, :src_len, :]) + self.criterion(tgt_recon[:, :tgt_len, :], tgt_linear[:, :tgt_len, :])
        '''
        loss_recon = self.criterion(src_recon[:, :src_len, :].masked_select(src_stft_length_masks.unsqueeze(-1).bool()), 
                src_padded_features[:, :src_len, :src_recon.shape[2]].masked_select(src_stft_length_masks.unsqueeze(-1).bool())) + \
                self.criterion(tgt_recon[:, :tgt_len, :].masked_select(tgt_stft_length_masks.unsqueeze(-1).bool()), 
                        tgt_padded_features[:, :tgt_len, :tgt_recon.shape[2]].masked_select(tgt_stft_length_masks.unsqueeze(-1).bool()))
        '''

        # if use_convert:
        src_convert.masked_fill_(~src_stft_length_masks.unsqueeze(-1), 0)
        tgt_convert.masked_fill_(~tgt_stft_length_masks.unsqueeze(-1), 0)
        
        src_len = min(src_convert.size(1), src_linear.size(1))
        tgt_len = min(tgt_convert.size(1), tgt_linear.size(1))
        loss_convert = self.criterion(src_convert[:, :src_len, :], src_linear[:, :src_len, :]) + self.criterion(tgt_convert[:, :tgt_len, :], tgt_linear[:, :tgt_len, :])
        '''
        loss_convert = self.criterion(src_convert[:, :src_len, :].masked_select(src_stft_length_masks.unsqueeze(-1).bool()), 
                src_padded_features[:, :src_len, :src_convert.shape[2]].masked_select(src_stft_length_masks.unsqueeze(-1).bool())) + \
                self.criterion(tgt_convert[:, :tgt_len, :].masked_select(tgt_stft_length_masks.unsqueeze(-1).bool()), 
                        tgt_padded_features[:, :tgt_len, :tgt_convert.shape[2]].masked_select(tgt_stft_length_masks.unsqueeze(-1).bool()))
        '''

        loss_prosody = self.criterion(src_convert_predicted_prosody[:, :src_len, :].masked_select(src_stft_length_masks.unsqueeze(-1).bool()), 
                src_speaker[:, :src_len, :].masked_select(src_stft_length_masks.unsqueeze(-1).bool())) + \
                        self.criterion(tgt_convert_predicted_prosody[:, :tgt_len, :].masked_select(tgt_stft_length_masks.unsqueeze(-1).bool()), 
                tgt_speaker[:, :tgt_len, :].masked_select(tgt_stft_length_masks.unsqueeze(-1).bool()))
        '''
        loss_prosody = None
        '''
            
        return loss_recon, loss_convert, loss_prosody
    
    def log(self, tag, data, step, mode='scalar', preprocessor=None):
        self.recorder.add_scalar(tag, data, global_step=step)

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
                    #print(key, i, wav)
                    self.logger(step=step, tag=f'{i}.{split}_{key}', data=wav, mode='audio', sample_dir=self.sample_dir)
        delattr(self, f'{split}_buffer')
