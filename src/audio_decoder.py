import torch
import torch.nn as nn

from src.module import VGGExtractor, CNNExtractor, CNNUpsampler, LinearUpsampler, PseudoUpsampler, RNNLayer

from src.conformer.convolution import ConvolutionModule
from src.conformer.encoder_layer import EncoderLayer

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


class AudioDecoder(nn.Module):
    def __init__(self, process_group, input_size, dim, dropout, out_dim=80, upsampler='cnn',
                 layer_share=None, layer_content=None, layer_speaker=None,
                 head=None, linear_unit=None, normalized_before=None, concat_after=None,
                 macaron_style=False,
                 pos_enc_layer_type="abs_pos",
                 selfattention_layer_type="selfattn",
                 use_cnn_module=False,
                 cnn_activation_type="swish",
                 cnn_module_kernel=31,
                 **kwargs):
        super(AudioDecoder, self).__init__()

        self.input_size = input_size
        self.out_dim = out_dim

       # Transformer specific
        self.layer_share = layer_share
        self.layer_content = layer_content
        self.layer_speaker = layer_speaker
        self.head = head
        self.linear_unit = linear_unit
        self.normalized_before = normalized_before
        self.concat_after = concat_after

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
        self.pos_embedding_before_upsample = pos_enc_class(dim, dropout) if pos_enc_class else None
        self.pos_embedding_after_upsample = pos_enc_class(dim, dropout) if pos_enc_class else None

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
        if layer_content > 0:
            self.content = repeat(
                layer_content,
                lambda lnum: EncoderLayer(
                    dim,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(dim, linear_unit, dropout),
                    PositionwiseFeedForward(dim, linear_unit, dropout) if macaron_style else None,
                    ConvolutionModule(process_group, dim, cnn_module_kernel, cnn_activation) if use_cnn_module else None,
                    dropout,
                    normalized_before,
                    concat_after,
                ),
            )

        # upsample before joining content and speaker
        if upsampler == 'linear':
            self.upsampler = LinearUpsampler(dim, dim)
        elif upsampler == 'cnn':
            self.upsampler = CNNUpsampler(dim, dim)
        else:
            # only linear transform, no upsample
            self.upsampler = PseudoUpsampler(dim, dim)

        # Shared layers in Encoder
        self.share = repeat(
            layer_share,
            lambda lnum: EncoderLayer(
                dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                PositionwiseFeedForward(dim, linear_unit, dropout),
                PositionwiseFeedForward(dim, linear_unit, dropout) if macaron_style else None,
                ConvolutionModule(process_group, dim, cnn_module_kernel, cnn_activation) if use_cnn_module else None,
                dropout,
                normalized_before,
                concat_after,
            ),
        )

        # project to specified mel-bin number
        self.output = nn.Linear(dim, self.out_dim)

    def forward(self, enc_len, input_x_content, enc_mask_content, 
                input_x_speaker, enc_mask_speaker, upsampled_x=None):
        """
        Args:
            enc_len: LongTensor in (batch_size, )
            input_x_content: FloatTensor in (batch_size, seq_len, feat_dim)
            enc_mask_content: BoolTensor in (batch_size, 1, seq_len)
            input_x_speaker: FloatTensor in (batch_size, seq_len, feat_dim)
            enc_mask_speaker: BoolTensor in (batch_size, 1, seq_len)
            pooled_input_x_speaker: FloatTensor in (batch_size, )

        Return:
            dec_len: LongTensor in (batch_size, )
            dec_x: FloatTensor in (batch_size, seq_len, feat_dim)
            dec_mask: FloatTensor in (batch_size, seq_len)

        """

        if upsampled_x is None:
            input_x_content = input_x_content * enc_mask_content.transpose(-1, -2)
            if hasattr(self, 'content'):
                if self.pos_embedding_before_upsample:
                    input_x_content = self.pos_embedding_before_upsample(input_x_content)
                input_x_content, enc_mask_content = self.content(input_x_content, enc_mask_content)
                if isinstance(input_x_content, tuple):
                    input_x_content = input_x_content[0]
            upsampled_x, dec_len = self.upsampler(input_x_content, enc_len)
        else:
            dec_len = enc_len

        join_x = upsampled_x #+ pooled_input_x_speaker.unsqueeze(1)
        if input_x_speaker is not None:
            join_x = join_x + input_x_speaker[:, :join_x.shape[1]]
        dec_mask = torch.lt(torch.arange(join_x.shape[1]).unsqueeze(0).to(dec_len.device), dec_len.unsqueeze(-1)).unsqueeze(1)

        if self.pos_embedding_after_upsample:
            join_x = self.pos_embedding_after_upsample(join_x)
        dec_x, dec_mask = self.share(join_x, dec_mask)
        if isinstance(dec_x, tuple):
            dec_x = dec_x[0]

        dec_mask = dec_mask.squeeze(1)

        # transform to bins of mel
        dec_x = self.output(dec_x)

        #assert dec_len.max().item() == dec_x.size(1) == dec_mask.size(1)
        assert dec_x.size(1) == dec_mask.size(1)
        return dec_len, dec_x, dec_mask
