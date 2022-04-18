#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Hang Le (hangtp.le@gmail.com)

"""Dual-decoder layer definition."""
import logging
import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class TripleDecoderLayer(nn.Module):
    """Single decoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, size, size_asr, size_conv,
                 self_attn, src_attn, feed_forward, 
                 self_attn_asr, src_attn_asr, feed_forward_asr,
                 self_attn_conv, src_attn_conv, feed_forward_conv,
                 cross_self_attn_asr2st, cross_self_attn_st2asr, 
                 cross_self_attn_asr2conv, cross_self_attn_conv2asr, 
                 cross_self_attn_st2conv, cross_self_attn_conv2st, 
                 cross_src_attn_asr2st, cross_src_attn_st2asr, 
                 cross_src_attn_asr2conv, cross_src_attn_conv2asr, 
                 cross_src_attn_st2conv, cross_src_attn_conv2st, 
                 dropout_rate, 
                 normalize_before=True, 
                 concat_after=False, 
                 cross_operator=None,
                 cross_weight_learnable=False, 
                 cross_weight=0.0,
                 cross_st2asr=True, cross_asr2st=True,
                 cross_conv2asr=True, cross_asr2conv=True,
                 cross_st2conv=True, cross_conv2st=True,
                 adapters=None,
                 ):
        """Construct an DecoderLayer object."""
        super(TripleDecoderLayer, self).__init__()
        self.size = size
        self.size_asr = size_asr
        self.size_conv = size_conv

        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.self_attn_asr = self_attn_asr
        self.src_attn_asr = src_attn_asr
        self.feed_forward_asr = feed_forward_asr
        
        self.self_attn_conv = self_attn_conv
        self.src_attn_conv = src_attn_conv
        self.feed_forward_conv = feed_forward_conv

        self.cross_self_attn_asr2st = cross_self_attn_asr2st
        self.cross_self_attn_st2asr = cross_self_attn_st2asr
        self.cross_self_attn_asr2conv = cross_self_attn_asr2conv
        self.cross_self_attn_conv2asr = cross_self_attn_conv2asr
        self.cross_self_attn_conv2st = cross_self_attn_conv2st
        self.cross_self_attn_st2conv = cross_self_attn_st2conv

        self.cross_src_attn_asr2st = cross_src_attn_asr2st
        self.cross_src_attn_st2asr = cross_src_attn_st2asr
        self.cross_src_attn_asr2conv = cross_src_attn_asr2conv
        self.cross_src_attn_conv2asr = cross_src_attn_conv2asr
        self.cross_src_attn_conv2st = cross_src_attn_conv2st
        self.cross_src_attn_st2conv = cross_src_attn_st2conv
      
        self.cross_st2asr = cross_st2asr
        self.cross_asr2st = cross_asr2st
        self.cross_conv2asr = cross_conv2asr
        self.cross_asr2conv = cross_asr2conv
        self.cross_st2conv = cross_st2conv
        self.cross_conv2st = cross_conv2st

        self.cross_operator = cross_operator
        if cross_operator == "concat":
            raise NotImplementedError
#            if cross_self_attn_asr2st is not None: 
#                self.cross_concat_linear1_asr2st = nn.Linear(size + size, size)
#            if cross_self_attn_st2asr is not None: 
#                self.cross_concat_linear1_st2asr = nn.Linear(size + size, size)
#            if cross_self_attn_conv2st is not None: 
#                self.cross_concat_linear1_conv2st = nn.Linear(size + size, size)
#            if cross_self_attn_st2conv is not None: 
#                self.cross_concat_linear1_st2conv = nn.Linear(size + size, size)
#            if cross_self_attn_asr2conv is not None: 
#                self.cross_concat_linear1_asr2conv = nn.Linear(size + size, size)
#            if cross_self_attn_conv2asr is not None: 
#                self.cross_concat_linear1_conv2asr = nn.Linear(size + size, size)
#            if cross_src_attn_asr2st is not None: 
#                self.cross_concat_linear2_asr2st = nn.Linear(size + size, size)
#            if cross_src_attn_st2asr is not None: 
#                self.cross_concat_linear2_st2asr = nn.Linear(size + size, size)
#            if cross_src_attn_conv2st is not None: 
#                self.cross_concat_linear2_conv2st = nn.Linear(size + size, size)
#            if cross_src_attn_st2conv is not None: 
#                self.cross_concat_linear2_st2conv = nn.Linear(size + size, size)
#            if cross_src_attn_asr2conv is not None: 
#                self.cross_concat_linear2_asr2conv = nn.Linear(size + size, size)
#            if cross_src_attn_conv2asr is not None: 
#                self.cross_concat_linear2_conv2asr = nn.Linear(size + size, size)
        elif cross_operator == "sum":
            if cross_weight_learnable:
                assert float(cross_weight) > 0.0
                if self.cross_asr2st:
                    self.cross_weight_asr2st = torch.nn.Parameter(torch.tensor(cross_weight))
                if self.cross_st2asr:
                    self.cross_weight_st2asr = torch.nn.Parameter(torch.tensor(cross_weight))
                if self.cross_conv2st:
                    self.cross_weight_conv2st = torch.nn.Parameter(torch.tensor(cross_weight))
                if self.cross_st2conv:
                    self.cross_weight_st2conv = torch.nn.Parameter(torch.tensor(cross_weight))
                if self.cross_asr2conv:
                    self.cross_weight_asr2conv = torch.nn.Parameter(torch.tensor(cross_weight))
                if self.cross_conv2asr:
                    self.cross_weight_conv2asr = torch.nn.Parameter(torch.tensor(cross_weight))
            else:
                if self.cross_asr2st:
                    self.cross_weight_asr2st = cross_weight
                if self.cross_st2asr:
                    self.cross_weight_st2asr = cross_weight
                if self.cross_conv2st:
                    self.cross_weight_conv2st = cross_weight
                if self.cross_st2conv:
                    self.cross_weight_st2conv = cross_weight
                if self.cross_asr2conv:
                    self.cross_weight_asr2conv = cross_weight
                if self.cross_conv2asr:
                    self.cross_weight_conv2asr = cross_weight

        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

        self.norm1_asr = LayerNorm(size_asr)
        self.norm2_asr = LayerNorm(size_asr)
        self.norm3_asr = LayerNorm(size_asr)
        self.dropout_asr = nn.Dropout(dropout_rate)

        self.norm1_conv = LayerNorm(size_conv)
        self.norm2_conv = LayerNorm(size_conv)
        self.norm3_conv = LayerNorm(size_conv)
        self.dropout_conv = nn.Dropout(dropout_rate)

        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            raise NotImplementedError
#            self.concat_linear1 = nn.Linear(size + size, size)
#            self.concat_linear2 = nn.Linear(size + size, size)
#            self.concat_linear1_asr = nn.Linear(size + size, size)
#            self.concat_linear2_asr = nn.Linear(size + size, size)
#
        self.adapters = adapters

    def forward(self, tgt, tgt_mask, tgt_asr, tgt_mask_asr, tgt_conv, tgt_mask_conv,
                memory, memory_mask, 
                cross_mask_asr2st, cross_mask_st2asr, 
                cross_mask_conv2st, cross_mask_st2conv, 
                cross_mask_asr2conv, cross_mask_conv2asr, 
                cross_self=False, cross_src=False,
                cross_self_from="before-self", cross_src_from="before-src",
                lang_id=None, 
                cache=None, cache_asr=None, cache_conv=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, 1, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)
            cross (torch.Tensor): decoded previous target from another decoder (batch, max_time_out, size)
        """
        residual = tgt
        residual_asr = tgt_asr
        residual_conv = tgt_conv

        if self.normalize_before:
            tgt = self.norm1(tgt)
            tgt_asr = self.norm1_asr(tgt_asr)
            tgt_conv = self.norm1_conv(tgt_conv)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (tgt.shape[0], tgt.shape[1] - 1, self.size), \
                f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if cache_asr is None:
            tgt_q_asr = tgt_asr
            tgt_q_mask_asr = tgt_mask_asr
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache_asr.shape == (tgt_asr.shape[0], tgt_asr.shape[1] - 1, self.size), \
                f"{cache_asr.shape} == {(tgt_asr.shape[0], tgt_asr.shape[1] - 1, self.size)}"
            tgt_q_asr = tgt_asr[:, -1:, :]
            residual_asr = residual_asr[:, -1:, :]
            tgt_q_mask_asr = None
            if tgt_mask_asr is not None:
                tgt_q_mask_asr = tgt_mask_asr[:, -1:, :]

        if cache_conv is None:
            tgt_q_conv = tgt_conv
            tgt_q_mask_conv = tgt_mask_conv
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache_conv.shape == (tgt_conv.shape[0], tgt_conv.shape[1] - 1, self.size), \
                f"{cache_conv.shape} == {(tgt_conv.shape[0], tgt_conv.shape[1] - 1, self.size)}"
            tgt_q_conv = tgt_conv[:, -1:, :]
            residual_conv = residual_conv[:, -1:, :]
            tgt_q_mask_conv = None
            if tgt_mask_conv is not None:
                tgt_q_mask_conv = tgt_mask_conv[:, -1:, :]

        # Self-attention
        if self.concat_after:
            raise NotImplementedError
#            tgt_concat = torch.cat((tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1)
#            x = self.concat_linear1(tgt_concat)
#            tgt_concat_asr = torch.cat((tgt_q_asr, self.self_attn_asr(tgt_q_asr, tgt_asr, tgt_asr, tgt_q_mask_asr)), dim=-1)
#            x_asr = self.concat_linear1_asr(tgt_concat_asr)
        else:
            x = self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
            x_asr = self.dropout_asr(self.self_attn_asr(tgt_q_asr, tgt_asr, tgt_asr, tgt_q_mask_asr))
            x_conv = self.dropout_conv(self.self_attn_conv(tgt_q_conv, tgt_conv, tgt_conv, tgt_q_mask_conv))
        
        # Cross-self attention
        if cross_self and cross_self_from == "before-self":
            if self.cross_asr2st:
                z = self.dropout(self.cross_self_attn_asr2st(tgt_q, tgt_asr, tgt_asr, cross_mask_asr2st))
                if self.cross_operator == 'sum':
                    x = x + self.cross_weight_asr2st * z
                else:
                    raise NotImplementedError
            if self.cross_st2asr:
                z_asr = self.dropout_asr(self.cross_self_attn_st2asr(tgt_q_asr, tgt, tgt, cross_mask_st2asr))
                if self.cross_operator == 'sum':
                    x_asr = x_asr + self.cross_weight_st2asr * z_asr
                else:
                    raise NotImplementedError
            if self.cross_conv2st:
                z = self.dropout(self.cross_self_attn_conv2st(tgt_q, tgt_conv, tgt_conv, cross_mask_conv2st))
                if self.cross_operator == 'sum':
                    x = x + self.cross_weight_conv2st * z
                else:
                    raise NotImplementedError
            if self.cross_st2conv:
                z_conv = self.dropout_conv(self.cross_self_attn_st2conv(tgt_q_conv, tgt, tgt, cross_mask_st2conv))
                if self.cross_operator == 'sum':
                    x_conv = x_conv + self.cross_weight_st2conv * z_conv
                else:
                    raise NotImplementedError
            if self.cross_asr2conv:
                z_conv = self.dropout_conv(self.cross_self_attn_asr2conv(tgt_q_conv, tgt_asr, tgt_asr, cross_mask_asr2conv))
                if self.cross_operator == 'sum':
                    x_conv = x_conv + self.cross_weight_asr2conv * z_conv
                else:
                    raise NotImplementedError
            if self.cross_conv2asr:
                z_asr = self.dropout_asr(self.cross_self_attn_conv2asr(tgt_q_asr, tgt_conv, tgt_conv, cross_mask_conv2asr))
                if self.cross_operator == 'sum':
                    x_asr = x_asr + self.cross_weight_conv2asr * z_asr
                else:
                    raise NotImplementedError

        x = x + residual
        x_asr = x_asr + residual_asr
        x_conv = x_conv + residual_conv

        if not self.normalize_before:
            x = self.norm1(x)
            x_asr = self.norm1_asr(x_asr)
            x_conv = self.norm1_conv(x_conv)

        # Source attention
        residual = x
        residual_asr = x_asr
        residual_conv = x_conv
        if self.normalize_before:
            x = self.norm2(x)
            x_asr = self.norm2_asr(x_asr)
            x_conv = self.norm2_conv(x_conv)
        y = x
        y_asr = x_asr
        y_conv = x_conv

        if self.concat_after:
            raise NotImplementedError
#            x_concat = torch.cat((x, self.src_attn(x, memory, memory, memory_mask)), dim=-1)
#            x = self.concat_linear2(x_concat)
#            x_concat_asr = torch.cat((x_asr, self.src_attn_asr(x_asr, memory, memory, memory_mask)), dim=-1)
#            x_asr = self.concat_linear2_asr(x_concat_asr)
        else:
            x = self.dropout(self.src_attn(x, memory, memory, memory_mask))
            x_asr = self.dropout_asr(self.src_attn_asr(x_asr, memory, memory, memory_mask))
            x_conv = self.dropout_conv(self.src_attn_conv(x_conv, memory, memory, memory_mask))
        
        # Cross-source attention
        if cross_src and cross_src_from == "before-src":
            if self.cross_asr2st:
                z = self.dropout(self.cross_src_attn_asr2st(y, y_asr, y_asr, cross_mask_asr2st))
                if self.cross_operator == 'sum':
                    x = x + self.cross_weight_asr2st * z
                else:
                    raise NotImplementedError
            if self.cross_st2asr:
                z_asr = self.dropout_asr(self.cross_src_attn_st2asr(y_asr, y, y, cross_mask_st2asr))
                if self.cross_operator == 'sum':
                    x_asr = x_asr + self.cross_weight_st2asr * z_asr
                else:
                    raise NotImplementedError
            if self.cross_conv2st:
                z = self.dropout(self.cross_src_attn_conv2st(y, y_conv, y_conv, cross_mask_conv2st))
                if self.cross_operator == 'sum':
                    x = x + self.cross_weight_conv2st * z
                else:
                    raise NotImplementedError
            if self.cross_st2conv:
                z_conv = self.dropout_conv(self.cross_src_attn_st2conv(y_conv, y, y, cross_mask_st2conv))
                if self.cross_operator == 'sum':
                    x_conv = x_conv + self.cross_weight_st2conv * z_conv
                else:
                    raise NotImplementedError
            if self.cross_asr2conv:
                z_conv = self.dropout_conv(self.cross_src_attn_asr2conv(y_conv, y_asr, y_asr, cross_mask_asr2conv))
                if self.cross_operator == 'sum':
                    x_conv = x_conv + self.cross_weight_asr2conv * z_conv
                else:
                    raise NotImplementedError
            if self.cross_conv2asr:
                z_asr = self.dropout_asr(self.cross_src_attn_conv2asr(y_asr, y_conv, y_conv, cross_mask_conv2asr))
                if self.cross_operator == 'sum':
                    x_asr = x_asr + self.cross_weight_conv2asr * z_asr
                else:
                    raise NotImplementedError
        
        x = x + residual
        x_asr = x_asr + residual_asr
        x_conv = x_conv + residual_conv
        
        if not self.normalize_before:
            x = self.norm2(x)
            x_asr = self.norm2_asr(x)
            x_conv = self.norm2_conv(x)
        
        # Feed forward
        residual = x
        residual_asr = x_asr
        residual_conv = x_conv

        if self.normalize_before:
            x = self.norm3(x)
            x_asr = self.norm3_asr(x_asr)
            x_conv = self.norm3_conv(x_conv)
        x = residual + self.dropout(self.feed_forward(x))
        x_asr = residual_asr + self.dropout_asr(self.feed_forward_asr(x_asr))
        x_conv = residual_conv + self.dropout_conv(self.feed_forward_conv(x_conv))
        if not self.normalize_before:
            x = self.norm3(x)
            x_asr = self.norm3_asr(x_asr)
            x_conv = self.norm3_conv(x_conv)
        
        # Adapters
        if lang_id is not None and self.adapters is not None:
            x = self.adapters[lang_id](x, x)[0]

        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        if cache_asr is not None:
            x_asr = torch.cat([cache_asr, x_asr], dim=1)
        if cache_conv is not None:
            x_conv = torch.cat([cache_conv, x_conv], dim=1)

        return x, tgt_mask, x_asr, tgt_mask_asr, x_conv, tgt_mask_conv, \
                memory, memory_mask, \
                cross_mask_asr2st, cross_mask_st2asr, \
                cross_mask_conv2st, cross_mask_st2conv, \
                cross_mask_asr2conv, cross_mask_conv2asr, \
                cross_self, cross_src, cross_self_from, cross_src_from
