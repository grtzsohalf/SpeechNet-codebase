import torch
import torch.nn as nn

from src.conformer.convolution import ConvolutionModule
from src.conformer.encoder_layer import EncoderLayer

from src.module import VGGExtractor, CNNExtractor, CNNUpsampler, LinearUpsampler, PseudoUpsampler, RNNLayer

from src.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from src.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
    RelPositionalEncoding,
)
from src.transformer.layer_norm import LayerNorm
from src.transformer.positionwise_feed_forward import PositionwiseFeedForward
from src.transformer.repeat import repeat
from src.transformer.nets_utils import make_non_pad_mask
from bin.sv.model import SAP

import torch.cuda.nvtx as nvtx


def get_activation(act):
    """Return activation function."""
    from src.conformer.swish import Swish

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
        "gelu": torch.nn.GELU,
    }

    return activation_funcs[act]()


class ProsodyPredictor(nn.Module):

    def __init__(self, dim, dropout, 
                 layer=None,
                 head=None, linear_unit=None, normalized_before=None, concat_after=None,
                 macaron_style=False,
                 pos_enc_layer_type="abs_pos",
                 selfattention_layer_type="selfattn",
                 use_cnn_module=False,
                 cnn_activation_type="swish",
                 cnn_module_kernel=31
                 ):
        super(ProsodyPredictor, self).__init__()

        # Hyper-parameters checking
        self.layer = layer

        # Construct model

        self.dim = dim
        self.dropout = dropout

        # Transformer specific
        self.head = head
        self.linear_unit = linear_unit
        self.normalized_before = normalized_before
        self.concat_after = concat_after

        # Recurrent or self-attention encoder
        # get positional embedding class
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        else:
            pos_enc_class = None

        # get positional embedding module
        self.pos_embedding = pos_enc_class(dim, dropout) if pos_enc_class else None

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (head, dim, dropout)
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (head, dim, dropout)
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        # get activation class
        cnn_activation = get_activation(cnn_activation_type)

        # Content layers in Encoder
        self.predictor = repeat(
            layer,
            lambda lnum: EncoderLayer(
                dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                PositionwiseFeedForward(dim, linear_unit, dropout),
                PositionwiseFeedForward(dim, linear_unit, dropout) if macaron_style else None,
                ConvolutionModule(None, dim, cnn_module_kernel, cnn_activation) if use_cnn_module else None,
                dropout,
                normalized_before,
                concat_after,
            ),
        )
        self.upsampler = CNNUpsampler(dim, dim)

        # Build model

    def forward(self, input_x, enc_len):
        input_x, enc_len = self.upsampler(input_x, enc_len)

        nvtx.range_push('Make non pad mask')
        enc_mask = make_non_pad_mask(enc_len.tolist(), torch.zeros((input_x.shape[0], input_x.shape[1]))).to(input_x.device).unsqueeze(-2)
        nvtx.range_pop()

        if self.pos_embedding:
            nvtx.range_push('Pos embedding')
            input_x = self.pos_embedding(input_x)
            nvtx.range_pop()

        nvtx.range_push('Encoder forward')
        input_x, enc_mask = self.predictor(input_x, enc_mask)
        if isinstance(input_x, tuple):
            input_x = input_x[0]
        nvtx.range_pop()
                
        return enc_len, input_x, enc_mask
