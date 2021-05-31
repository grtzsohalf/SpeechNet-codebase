import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from bin.tts.text.symbols import symbols
from bin.tts.utils import get_sinusoid_encoding_table
from src.transformer.attention import MultiHeadedAttention


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn



class MultiHeadAttention_tts(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn



class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(d_in, d_hid, 
                             kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2)
        # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in,
                             kernel_size=kernel_size[1], padding=(kernel_size[1]-1)//2)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, att_n_head, att_d_feat, fft_filter ,fft_kernel, dropout=0.1):
        super(FFTBlock, self).__init__()
        # we use MultiHeadedAttention from this project, the dimension of key and value
        # are calculated based on n_head and feature dim. Howerver, for PositionwiseFeedForward
        # , we used the one from the original project.
        self.slf_attn = MultiHeadAttention_tts(att_n_head, att_d_feat, att_d_feat//att_n_head, att_d_feat//att_n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(att_d_feat, fft_filter, fft_kernel, dropout=dropout)

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        return enc_output
    
        #return enc_output, enc_slf_attn

class TextEncoder(nn.Module):
    ''' Encoder '''
    """
    def __init__(self,
                 n_src_vocab=len(symbols)+1,               
                 len_max_seq=hp.max_seq_len,                --->max_seq_len
                 d_word_vec=hp.encoder_hidden,              --->word_dim
                 n_layers=hp.encoder_layer,                 --->n_layers
                 n_head=hp.encoder_head,                    --->att_n_head
                 d_k=hp.encoder_hidden // hp.encoder_head,  --->att_k
                 d_v=hp.encoder_hidden // hp.encoder_head,  --->att_v
                 d_model=hp.encoder_hidden,                 --->att_d_feat
                 d_inner=hp.fft_conv1d_filter_size,         --->fft_filter
                 dropout=hp.encoder_dropout):
    """
    def __init__(self, max_seq_len, word_dim, n_layer,
                 att_n_head, att_d_feat, 
                 fft_filter, fft_kernel, dropout, n_src_vocab=len(symbols)+1):
        super(TextEncoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.word_dim = word_dim

        PAD = 0
        n_position = max_seq_len + 1
        self.src_word_emb = nn.Embedding(n_src_vocab, word_dim, padding_idx=PAD)
        self.position_enc = nn.Parameter(get_sinusoid_encoding_table(n_position, word_dim).unsqueeze(0), requires_grad=False)
        self.layer_stack = nn.ModuleList([FFTBlock(att_n_head, att_d_feat, fft_filter, fft_kernel, dropout=dropout) for _ in range(n_layer)])

    def forward(self, src_seq, mask, return_attns=False):

        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        # slf_attn_mask = ~slf_attn_mask
        # print(slf_attn_mask.long())
        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(src_seq.shape[1], self.word_dim)[:src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, mask=mask, slf_attn_mask=slf_attn_mask)
        return enc_output