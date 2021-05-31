import math
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.util import init_weights, init_gate
from src.module import ScaleDotAttention, LocationAwareAttention
from src.text_encoder import TextEncoder
from src.text_decoder import TextDecoder
from src.audio_encoder import AudioEncoder
from src.audio_decoder import AudioDecoder
from src.prosody_predictor import ProsodyPredictor
from src.audio import create_transform

from src.transformer.mask import target_mask
from src.transformer.nets_utils import make_non_pad_mask

import torch.cuda.nvtx as nvtx


class Model(nn.Module):
    ''' General model, including AudioEncoder/TextDecoder(s) and more in the future'''

    def __init__(self, paras, config, mode, asr_task, se_task, tts_task,
                 sc_task, vcb_task, input_size, vocab_size, init_adadelta, log_step, ctc_weight, lsm_weight, 
                 audio_encoder, attention, audio_decoder, text_encoder, text_decoder, prosody_predictor, emb_drop=0.0, 
                 emb_tying=None, emb_dim=None, module=None, dim=None, n_layers=None, dropout=None,
                 solver_asr=None, solver_se=None, solver_tts=None,solver_sc=None, solver_vcb=None, process_group=None, **kwargs):
        super(Model, self).__init__()

        self.log_step = log_step
        # Preprocessing modules
        audio_config = config['data']['audio']
        self.audio_transform, self.feat_dim, self.sample_rate = create_transform(audio_config.copy())

        self.asr_task = asr_task
        self.se_task = se_task
        self.tts_task = tts_task
        self.sc_task = sc_task
        self.vcb_task = vcb_task

        # log_sigmas for multi-task loss weights
        self.log_sigma_asr = nn.Parameter(torch.zeros(1))
        self.log_sigma_se = nn.Parameter(torch.zeros(1))
        self.log_sigma_sc = nn.Parameter(torch.zeros(1))
        self.log_sigma_tts = nn.Parameter(torch.zeros(1))
        self.log_sigma_vcb = nn.Parameter(torch.zeros(1))

        # audio encoder
        self.audio_encoder = AudioEncoder(process_group, input_size, **audio_encoder)
        assert hasattr(self.audio_encoder, 'out_dim')

        # audio decoder
        self.audio_decoder = AudioDecoder(process_group, self.audio_encoder.out_dim, **audio_decoder)
        assert hasattr(self.audio_decoder, 'out_dim')

        # text decoder
        self.dec_dim = text_decoder['dim']
        query_dim = self.dec_dim*text_decoder['layer']
        if text_decoder['module'] in ['LSTM', 'GRU']:
            self.text_decoder = TextDecoder(
                self.audio_encoder.out_dim+self.dec_dim, vocab_size, **text_decoder)
            self.attention = Attention(
                self.audio_encoder.out_dim, query_dim, **attention)
        elif text_decoder['module'] == 'Transformer':
            self.text_decoder = TextDecoder(
                self.dec_dim, vocab_size, **text_decoder)
            self.attention = None
            
        # text encoder
        self.text_encoder = TextEncoder(**text_encoder)

        # prosody predictor
        self.prosody_predictor = ProsodyPredictor(**prosody_predictor)
        #self.prosody_predictor = None
        

        ### ASR
        self.solver_asr = solver_asr
        if asr_task:
            self.asr = nn.ModuleDict()
            self.asr.solver = solver_asr
            # Setup
            assert 0 <= ctc_weight <= 1
            self.vocab_size = vocab_size
            self.lsm_weight = lsm_weight
            self.ctc_weight = ctc_weight
            self.enable_ctc = ctc_weight > 0
            self.enable_att = ctc_weight != 1

            # Modules
            if self.enable_ctc:
                self.asr.ctc_layer = nn.Linear(self.audio_encoder.out_dim, vocab_size)
            else:
                self.asr.ctc_layer = None

            if self.enable_att:
                self.asr.pre_embed = nn.Embedding(vocab_size, self.dec_dim)
                self.asr.embed_drop = nn.Dropout(emb_drop)
            else:
                self.asr.pre_embed = None
                self.asr.embed_drop = None

            '''
            if not paras.single_task:
                self.asr.output = nn.Sequential(
                    nn.Linear(self.audio_decoder.out_dim, audio_config['feat_dim']),
                    nn.ReLU(),
                    nn.Linear(audio_config['feat_dim'], audio_config['feat_dim'])
                )
            else:
                self.asr.output = None
            '''
            self.asr.output = nn.Sequential(
                nn.Linear(self.audio_decoder.out_dim, audio_config['feat_dim']),
                nn.ReLU(),
                nn.Linear(audio_config['feat_dim'], audio_config['feat_dim'])
            )

        if se_task:
            self.se = nn.ModuleDict()
            self.se.solver = solver_se
            self.se.model = solver_se.get_model()
        
        if sc_task:
            self.sc = nn.ModuleDict()
            self.sc.solver = solver_sc
            self.sc.transform = solver_sc.input_extracter
            self.sc.model = solver_sc.post_module
            self.sc.objective =solver_sc.objective

        ## TTS
        if tts_task:
            self.tts = nn.ModuleDict()
            self.tts.solver = solver_tts
            self.tts.speaker_integrator, self.tts.variance_adaptor, self.tts.extractor = solver_tts.get_model()

        ## VC Baseline
        if vcb_task:
            self.vcb = nn.ModuleDict()
            self.vcb.solver = solver_vcb
            self.vcb.model = solver_vcb.get_model()

        # Init
        if init_adadelta:
            self.apply(init_weights)
            for l in range(self.text_decoder.layer):
                bias = getattr(self.text_decoder.layers, 'bias_ih_l{}'.format(l))
                bias = init_gate(bias)

    def set_state(self, prev_state, prev_attn):
        ''' Setting up all memory states for beam decoding'''
        self.text_decoder.set_state(prev_state)
        if self.text_decoder.module in ['LSTM', 'GRU']:
            self.attention.set_mem(prev_attn)

    def create_msg(self):
        # Messages for user
        msg = []
        if self.asr_task:
            msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(
                self.audio_encoder.sample_rate))
            if self.audio_encoder.vgg:
                msg.append(
                    '           | VGG Extractor w/ time downsampling rate = 4 in encoder enabled.')
            if self.audio_encoder.cnn:
                msg.append(
                    '           | CNN Extractor w/ time downsampling rate = 4 in encoder enabled.')
            if self.enable_ctc:
                msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(
                    self.ctc_weight))
            if self.enable_att:
                msg.append('           | attention-based decoder enabled ( lambda = {}).'.format(
                    1-self.ctc_weight))
        return msg

    def forward(self, asr_inputs, se_data, tts_data, sc_data, vcb_data, step, cur_lr, progress_step, mode):
        '''
        ASR Arguments:
            audio_feature - [BxTxD] Acoustic feature with shape 
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps 
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxL]   Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return text_decoder state [BxLxD] for other purpose
            txt_len       - [B]     Length of txt in a batch

        LM Arguments:
            x
            lens
            hidden
        '''
        # Compute loss and others

        loss = 0.

        # torch.autograd.set_detect_anomaly(True)

        if mode == 'train':
            # Multi-task loss weights: https://arxiv.org/pdf/1705.07115.pdf
            squared_sigma_asr = torch.exp(self.log_sigma_asr)**2
            squared_sigma_se = torch.exp(self.log_sigma_se)**2
            squared_sigma_sc = torch.exp(self.log_sigma_sc)**2
            squared_sigma_tts = torch.exp(self.log_sigma_tts)**2
            squared_sigma_vcb = torch.exp(self.log_sigma_vcb)**2

        asr_outputs = None
        if self.asr_task and asr_inputs is not None:
            '''
            if not self.training:
                audio_transform = self.audio_transform[:-1]
            else:
                audio_transform = self.audio_transform
            '''
            audio_transform = self.audio_transform[:-1]
            asr_outputs, lm_outputs = self.solver_asr.forward(asr_inputs, self.sample_rate, 
                                                              audio_transform, self.audio_encoder, self.audio_decoder, self.text_decoder, 
                                                              self.prosody_predictor, self.asr.output, self.attention, self.vocab_size, 
                                                              self.lsm_weight, self.ctc_weight, self.enable_ctc, self.enable_att, 
                                                              self.asr.ctc_layer, self.asr.pre_embed, self.asr.embed_drop, if_reconstruction=True)
            if mode == 'train':
                asr_exec_inputs = {'dec_state': asr_outputs['dec_state'],
                                   'att_output': asr_outputs['att_output'],
                                   'txt': asr_outputs['txt'],
                                   'txt_len': asr_outputs['txt_len'],
                                   'ctc_output': asr_outputs['ctc_output'],
                                   'encode_len': asr_outputs['encode_len'],
                                   'asr_feat': asr_outputs['asr_feat'],
                                   'asr_feat_len': asr_outputs['asr_feat_len'],
                                   'reconstructed': asr_outputs['reconstructed'],
                                   'encode_feature_speaker': asr_outputs['encode_feature_speaker'],
                                   'predicted_prosody': asr_outputs['predicted_prosody']}
                # asr_exec_outputs: att_loss, ctc_loss, emb_loss, att_output, ctc_output
                asr_loss, asr_exec_outputs = self.solver_asr._exec_asr(
                    asr_exec_inputs, self.ctc_weight)
                # total_loss, att_loss, ctc_loss, emb_loss, att_output, ctc_output = \
                # self.solver_asr._exec_asr(self, dec_state, att_output, txt, txt_len, total_loss, ctc_output, encode_len)
                scaled_asr_loss = asr_loss / squared_sigma_asr[0] + self.log_sigma_asr[0]
                loss = loss + scaled_asr_loss

                # Logger
                # self.solver_asr.step += 1
                if step % self.log_step == 0:
                    self.solver_asr._write_asr_log(
                        scaled_asr_loss, asr_loss, cur_lr, asr_exec_outputs, asr_outputs['txt'], step, progress_step)

        se_outputs = None
        if self.se_task and se_data:
            se_outputs = self.se.solver.forward(se_data, self.audio_encoder, self.audio_decoder, self.prosody_predictor, **self.se)

            if mode == 'train':
                se_loss = self.se.solver.compute_loss(**se_outputs, step=step, split='train')
                scaled_se_loss= se_loss * 0.5 / squared_sigma_se[0] + self.log_sigma_se[0]
                loss = loss + scaled_se_loss
        
        sc_outputs = None
        if self.sc_task and sc_data:
            sc_outputs = self.sc.solver.forward(sc_data, self.audio_encoder, self.sc.transform,)

            if mode == 'train':
                sc_loss = self.sc.solver.compute_loss(**sc_outputs, step=step)

                if step % int(self.sc.solver.log_step) == 0:
                    if self.sc.solver.rank == 0:
                        self.sc.solver.logging("sc_loss", sc_loss.item(), step)
                        self.sc.solver.logging("lr", cur_lr, step)
                        print(f'[SC] loss ={sc_loss.item()}')
                        print(f'[LR] learning rate = {cur_lr}' )
                scaled_sc_loss = sc_loss / squared_sigma_sc[0] + self.log_sigma_sc[0]
                loss = loss + scaled_sc_loss

        tts_outputs = None
        if self.tts_task and tts_data:
            tts_outputs = self.tts.solver.forward(mode, tts_data, self.audio_transform, self.audio_encoder, self.text_encoder, self.audio_decoder, 
                                                  self.prosody_predictor, **self.tts)

            if mode == 'train':
                tts_total_loss, tts_mel_loss, tts_d_loss, tts_speaker_loss, tts_prosody_loss = \
                    self.tts.solver.compute_loss(**tts_outputs) 

                scaled_tts_total_loss = tts_total_loss * 0.5 / squared_sigma_tts[0] + self.log_sigma_tts[0]
                loss = loss + scaled_tts_total_loss
                
                if step % int(self.tts.solver.log_step) == 0 :
                    if self.tts.solver.rank == 0:
                        self.tts.solver.logging("tts_total_loss", tts_total_loss.item(), step, mode='scalar')
                        self.tts.solver.logging("tts_mel_loss", tts_mel_loss.item(), step, mode='scalar')
                        self.tts.solver.logging("tts_duration_loss", tts_d_loss.item(), step, mode='scalar')
                        self.tts.solver.logging("tts_speaker_loss", tts_speaker_loss.item(), step, mode='scalar')
                        self.tts.solver.logging("tts_prosody_loss", tts_prosody_loss.item(), step, mode='scalar')
                        #self.tts.solver.logging("tts_pitch_loss", tts_p_loss.item(), step, mode='scalar')
                        #self.tts.solver.logging("tts_energy_loss", tts_e_loss.item(), step, mode='scalar')

        vcb_outputs = None
        if self.vcb_task and vcb_data:
            vcb_outputs = self.vcb.solver.forward(mode, vcb_data, self.audio_encoder, self.audio_decoder, self.prosody_predictor, 
                    self.audio_transform[:-1], step, **self.vcb)

            if mode == 'train':
                vcb_loss_recon, vcb_loss_convert, vcb_loss_prosody = self.vcb.solver.compute_loss(**vcb_outputs, step=step)
                scaled_vcb_total_loss = vcb_loss_recon * 0.5 / squared_sigma_vcb[0] + self.log_sigma_vcb[0]
                if vcb_outputs['use_convert'] == True:
                    scaled_vcb_total_loss = scaled_vcb_total_loss + vcb_loss_prosody * 0.5 / squared_sigma_vcb[0] + self.log_sigma_vcb[0]
                    scaled_vcb_total_loss = scaled_vcb_total_loss + vcb_loss_convert * 0.5 / squared_sigma_vcb[0] + self.log_sigma_vcb[0]
                loss = loss + scaled_vcb_total_loss

                if step % int(self.vcb.solver.log_step) == 0:
                    if self.vcb.solver.rank == 0:
                        self.vcb.solver.log("vcb_loss_recon", vcb_loss_recon.item(), step)
                        self.vcb.solver.log("vcb_loss_convert", vcb_loss_convert.item(), step)
                        self.vcb.solver.log("vcb_loss_prosody", vcb_loss_prosody.item(), step)

        return asr_outputs, se_outputs, sc_outputs, tts_outputs, vcb_outputs, loss



class Attention(nn.Module):
    ''' Attention mechanism
        please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
        Input : TextDecoder state                      with shape [batch size, text_decoder hidden dimension]
                Compressed feature from AudioEncoder   with shape [batch size, T, audio_encoder feature dimension]
        Output: Attention score                        with shape [batch size, num head, T (attention score of each time step)]
                Context vector                         with shape [batch size, audio_encoder feature dimension]
                (i.e. weighted (by attention score) sum of all timesteps T's feature) '''

    def __init__(self, v_dim, q_dim, mode, dim, num_head, temperature, v_proj,
                 loc_kernel_size, loc_kernel_num):
        super(Attention, self).__init__()

        # Setup
        self.v_dim = v_dim
        self.dim = dim
        self.mode = mode.lower()
        self.num_head = num_head

        # Linear proj. before attention
        self.proj_q = nn.Linear(q_dim, dim*num_head)
        self.proj_k = nn.Linear(v_dim, dim*num_head)
        self.v_proj = v_proj
        if v_proj:
            self.proj_v = nn.Linear(v_dim, v_dim*num_head)

        # Attention
        if self.mode == 'dot':
            self.att_layer = ScaleDotAttention(temperature, self.num_head)
        elif self.mode == 'loc':
            self.att_layer = LocationAwareAttention(
                loc_kernel_size, loc_kernel_num, dim, num_head, temperature)
        else:
            raise NotImplementedError

        # Layer for merging MHA
        if self.num_head > 1:
            self.merge_head = nn.Linear(v_dim*num_head, v_dim)

        # Stored feature
        self.key = None
        self.value = None
        self.mask = None

    def reset_mem(self):
        self.key = None
        self.value = None
        self.mask = None
        self.att_layer.reset_mem()

    def set_mem(self, prev_attn):
        self.att_layer.set_mem(prev_attn)

    def forward(self, dec_state, enc_feat, enc_len):

        # Preprecessing
        bs, ts, _ = enc_feat.shape
        query = torch.tanh(self.proj_q(dec_state))
        query = query.view(bs, self.num_head, self.dim).view(
            bs*self.num_head, self.dim)  # BNxD

        if self.key is None:
            # Maskout attention score for padded states
            self.att_layer.compute_mask(enc_feat, enc_len.to(enc_feat.device))

            # Store enc state to lower computational cost
            self.key = torch.tanh(self.proj_k(enc_feat))
            self.value = torch.tanh(self.proj_v(
                enc_feat)) if self.v_proj else enc_feat  # BxTxN

            if self.num_head > 1:
                self.key = self.key.view(bs, ts, self.num_head, self.dim).permute(
                    0, 2, 1, 3)  # BxNxTxD
                self.key = self.key.contiguous().view(bs*self.num_head, ts, self.dim)  # BNxTxD
                if self.v_proj:
                    self.value = self.value.view(
                        bs, ts, self.num_head, self.v_dim).permute(0, 2, 1, 3)  # BxNxTxD
                    self.value = self.value.contiguous().view(
                        bs*self.num_head, ts, self.v_dim)  # BNxTxD
                else:
                    self.value = self.value.repeat(self.num_head, 1, 1)

        # Calculate attention
        context, attn = self.att_layer(query, self.key, self.value)
        if self.num_head > 1:
            context = context.view(
                bs, self.num_head*self.v_dim)    # BNxD  -> BxND
            context = self.merge_head(context)  # BxD

        return attn, context

