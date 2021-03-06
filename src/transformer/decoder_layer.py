#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""

import torch
from torch import nn

from src.transformer.layer_norm import LayerNorm

import torch.cuda.nvtx as nvtx


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward: feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, memory, memory_mask, tgt, tgt_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor):
                decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, 1, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)

        """
        residual = tgt
        if self.normalize_before:
            nvtx.range_push('Norm 1')
            tgt = self.norm1(tgt)
            nvtx.range_pop()

        nvtx.range_push('Cache')
        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]
        nvtx.range_pop()

        nvtx.range_push('Self attn and concat')
        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        nvtx.range_pop()
        if not self.normalize_before:
            nvtx.range_push('Norm 1')
            x = self.norm1(x)
            nvtx.range_pop()

        residual = x
        if self.normalize_before:
            nvtx.range_push('Norm 2')
            x = self.norm2(x)
            nvtx.range_pop()
        nvtx.range_push('Self attn and concat')
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        nvtx.range_pop()
        if not self.normalize_before:
            nvtx.range_push('Norm 2')
            x = self.norm2(x)
            nvtx.range_pop()

        residual = x
        if self.normalize_before:
            nvtx.range_push('Norm 3')
            x = self.norm3(x)
            nvtx.range_pop()
        nvtx.range_push('Feedforward')
        x = residual + self.dropout(self.feed_forward(x))
        nvtx.range_pop()
        if not self.normalize_before:
            nvtx.range_push('Norm 3')
            x = self.norm3(x)
            nvtx.range_pop()

        nvtx.range_push('Cache concat')
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        nvtx.range_pop()

        return memory, memory_mask, x, tgt_mask
