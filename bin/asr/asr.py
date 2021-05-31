import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.util import init_weights, init_gate
from src.module import ScaleDotAttention, LocationAwareAttention
from src.text_decoder import TextDecoder
from src.audio_encoder import AudioEncoder

from src.transformer.mask import target_mask
from src.transformer.nets_utils import make_non_pad_mask


class ASR(nn.Module):
    ''' ASR model, including AudioEncoder/TextDecoder(s)'''

    def __init__(self, input_size, vocab_size, init_adadelta, ctc_weight, lsm_weight, 
                 audio_encoder, attention, text_decoder, emb_drop=0.0):
        super(ASR, self).__init__()

        # Setup
        assert 0 <= ctc_weight <= 1
        self.vocab_size = vocab_size
        self.lsm_weight = lsm_weight
        self.ctc_weight = ctc_weight
        self.enable_ctc = ctc_weight > 0
        self.enable_att = ctc_weight != 1
        self.lm = None

        # Modules
        self.audio_encoder = AudioEncoder(input_size, **audio_encoder)
        if self.enable_ctc:
            self.ctc_layer = nn.Linear(self.audio_encoder.out_dim, vocab_size)
        if self.enable_att:
            self.dec_dim = text_decoder['dim']
            self.pre_embed = nn.Embedding(vocab_size, self.dec_dim)
            self.embed_drop = nn.Dropout(emb_drop)

            query_dim = self.dec_dim*text_decoder['layer']

            if text_decoder['module'] in ['LSTM', 'GRU']:
                self.text_decoder = TextDecoder(
                    self.audio_encoder.out_dim+self.dec_dim, vocab_size, **text_decoder)
                self.attention = Attention(
                    self.audio_encoder.out_dim, query_dim, **attention)

            elif text_decoder['module'] == 'Transformer':
                self.text_decoder = TextDecoder(
                    self.dec_dim, vocab_size, **text_decoder)

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

    def forward(self, audio_feature, feature_len, decode_step, tf_rate=0.0, teacher=None,
                emb_decoder=None, get_dec_state=False, txt_len=None):
        '''
        Arguments
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
        '''
        # Init
        bs = audio_feature.shape[0]
        ctc_output, att_output, att_seq = None, None, None
        dec_state = [] if get_dec_state else None

        # Encode
        encode_len, encode_feature_content, encode_mask_content, \
            encode_feature_speaker, encode_mask_speaker = self.audio_encoder(audio_feature, feature_len)

        # CTC based decoding
        if self.enable_ctc:
            ctc_output = F.log_softmax(self.ctc_layer(encode_feature_content), dim=-1)

        # Attention based decoding
        if self.enable_att:
            # Init (init char = <SOS>, reset all rnn state and cell)
            self.text_decoder.init_state(bs)
            if self.text_decoder.module in ['LSTM', 'GRU']:
                self.attention.reset_mem()
            last_char = self.pre_embed(torch.zeros(
                (bs), dtype=torch.long, device=encode_feature_content.device))
            att_seq, output_seq = [], []

            # Preprocess data for teacher forcing
            if teacher is not None:
                teacher = self.embed_drop(self.pre_embed(teacher))

            # Decode
            if self.text_decoder.module == 'Transformer' and teacher is not None:
                # Pad sos
                txt_input = torch.cat((torch.zeros(last_char.shape, device=teacher.device).unsqueeze(1), teacher[:, :-1]), 1)
                txt_mask = make_non_pad_mask((txt_len).tolist()).to(teacher.device)
                txt_mask = target_mask(txt_mask, 0) 
                dec_state = self.text_decoder(encode_feature_content, encode_mask_content, txt_input, txt_mask)
                for t in range(decode_step):
                    cur_char = self.text_decoder.char_trans(self.text_decoder.final_dropout(dec_state[:, t]))
                    output_seq.append(cur_char)
                att_output = torch.stack(output_seq, dim=1)  # BxTxV
            else:
                if self.text_decoder.module == 'Transformer':
                    decoded = torch.zeros((last_char.shape[0], decode_step+1, last_char.shape[1]), 
                                          device=encode_feature_content.device)
                for t in range(decode_step):
                    if self.text_decoder.module in ['LSTM', 'GRU']:
                        # Attend (inputs current state of first layer, encoded features)
                        attn, context = self.attention(
                            self.text_decoder.get_query(), encode_feature_content, encode_len)
                        # Decode (inputs context + embedded last character)
                        text_decoder_input = torch.cat([last_char, context], dim=-1)
                        cur_char, d_state = self.text_decoder(text_decoder_input)
                    elif self.text_decoder.module == 'Transformer':
                        txt_input = decoded[:, :t+1]
                        txt_mask = make_non_pad_mask([t+1]*decoded.shape[0]).to(txt_input.device)
                        txt_mask = target_mask(txt_mask, 0) 
                        d_state = self.text_decoder(encode_feature_content, encode_mask_content, txt_input, txt_mask)[:, -1]
                        cur_char = self.text_decoder.char_trans(self.text_decoder.final_dropout(d_state))

                    # Prepare output as input of next step
                    if (teacher is not None):
                        # Training stage
                        if (tf_rate == 1) or (torch.rand(1).item() <= tf_rate):
                            # teacher forcing
                            last_char = teacher[:, t, :]
                        else:
                            # self-sampling (replace by argmax may be another choice)
                            with torch.no_grad():
                                if (emb_decoder is not None) and emb_decoder.apply_fuse:
                                    _, cur_prob = emb_decoder(
                                        d_state, cur_char, return_loss=False)
                                else:
                                    cur_prob = cur_char.softmax(dim=-1)
                                sampled_char = Categorical(cur_prob).sample()
                            last_char = self.embed_drop(
                                self.pre_embed(sampled_char))
                    else:
                        # Inference stage
                        if (emb_decoder is not None) and emb_decoder.apply_fuse:
                            _, cur_char = emb_decoder(
                                d_state, cur_char, return_loss=False)
                        # argmax for inference
                        last_char = self.pre_embed(torch.argmax(cur_char, dim=-1))
                        if self.text_decoder.module == 'Transformer':
                            decoded[:, t+1] = last_char


                    # save output of each step
                    output_seq.append(cur_char)
                    if self.text_decoder.module in ['LSTM', 'GRU']:
                        att_seq.append(attn)
                    if get_dec_state:
                        dec_state.append(d_state)

                att_output = torch.stack(output_seq, dim=1)  # BxTxV
                if self.text_decoder.module in ['LSTM', 'GRU']:
                    att_seq = torch.stack(att_seq, dim=2)       # BxNxDtxT
                else:
                    att_seq = None
                if get_dec_state:
                    dec_state = torch.stack(dec_state, dim=1)

        return ctc_output, encode_len, att_output, att_seq, dec_state


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

