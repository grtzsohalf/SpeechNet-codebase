import torch
import torch.nn as nn

from src.transformer.attention import MultiHeadedAttention
from src.transformer.decoder_layer import DecoderLayer
from src.transformer.embedding import PositionalEncoding
from src.transformer.layer_norm import LayerNorm
from src.transformer.positionwise_feed_forward import PositionwiseFeedForward
from src.transformer.repeat import repeat


class TextDecoder(nn.Module):
    ''' Decoder (a.k.a. Speller in LAS) '''
    # ToDo:　More elegant way to implement decoder

    def __init__(self, input_dim, vocab_size, module, dim, layer, dropout, 
                 head=None, linear_unit=None, normalized_before=None, concat_after=None):
        super(TextDecoder, self).__init__()
        self.in_dim = input_dim
        self.vocab_size = vocab_size
        self.module = module
        self.dim = dim
        self.layer = layer
        self.dropout = dropout

        # Transformer specific
        self.head = head
        self.linear_unit = linear_unit
        self.normalized_before = normalized_before
        self.concat_after = concat_after

        # Init
        assert module in ['Transformer', 'LSTM', 'GRU'], NotImplementedError
        self.hidden_state = None
        self.enable_cell = module == 'LSTM'

        # Modules
        if module in ['LSTM', 'GRU']:
            self.layers = getattr(nn, module)(
                input_dim, dim, num_layers=layer, dropout=dropout, batch_first=True)
        elif module =='Transformer':
            self.layers = repeat(
                layer,
                lambda lnum: DecoderLayer(
                    dim,
                    MultiHeadedAttention(head, dim, dropout),
                    MultiHeadedAttention(head, dim, dropout),
                    PositionwiseFeedForward(dim, linear_unit, dropout),
                    dropout,
                    normalized_before,
                    concat_after,
                ),
            )
        self.char_trans = nn.Linear(dim, vocab_size)
        self.final_dropout = nn.Dropout(dropout)

    def init_state(self, bs):
        ''' Set all hidden states to zeros '''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (torch.zeros((self.layer, bs, self.dim), device=device),
                                 torch.zeros((self.layer, bs, self.dim), device=device))
        else:
            self.hidden_state = torch.zeros(
                (self.layer, bs, self.dim), device=device)
        return self.get_state()

    def set_state(self, hidden_state):
        ''' Set all hidden states/cells, for decoding purpose'''
        device = next(self.parameters()).device
        if self.enable_cell:
            self.hidden_state = (hidden_state[0].to(
                device), hidden_state[1].to(device))
        else:
            self.hidden_state = hidden_state.to(device)

    def get_state(self):
        ''' Return all hidden states/cells, for decoding purpose'''
        if self.enable_cell:
            return (self.hidden_state[0].cpu(), self.hidden_state[1].cpu())
        else:
            return self.hidden_state.cpu()

    def get_query(self):
        ''' Return state of all layers as query for attention '''
        if self.enable_cell:
            return self.hidden_state[0].transpose(0, 1).reshape(-1, self.dim*self.layer)
        else:
            return self.hidden_state.transpose(0, 1).reshape(-1, self.dim*self.layer)

    def forward(self, x, x_mask=None, txt=None, txt_mask=None):
        ''' Decode and transform into vocab '''
        if self.module in ['LSTM', 'GRU']:
            if not self.training:
                self.layers.flatten_parameters()
            x, self.hidden_state = self.layers(x.unsqueeze(1), self.hidden_state)
            x = x.squeeze(1)
            char = self.char_trans(self.final_dropout(x))
            return char, x
        elif self.module == 'Transformer':
            _, _, decoded_txt, _ = self.layers(x, x_mask, txt, txt_mask)
            return decoded_txt
