#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Hang Le (hangtp.le@gmail.com)

"""Dual-decoder definition."""

import logging
import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer_triple import TripleDecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import ScorerInterface
from espnet.nets.pytorch_backend.transformer.adapter import Adapter


class TripleDecoder(ScorerInterface, torch.nn.Module):
    """Transfomer decoder module.

    :param int odim: output dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate for attention
    :param str or torch.nn.Module input_layer: input layer type
    :param bool use_output_layer: whether to use output layer
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, 
                 odim_tgt,
                 odim_src,
                 odim_wrsrc,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 self_attention_dropout_rate=0.0,
                 src_attention_dropout_rate=0.0,
                 input_layer="embed",
                 use_output_layer=True,
                 pos_enc_class=PositionalEncoding,
                 normalize_before=True,
                 concat_after=False,
                 cross_operator=None,
                 cross_weight_learnable=False,
                 cross_weight=0.0,
                 cross_self=False,
                 cross_src=False,
                 cross_st2asr=True,
                 cross_asr2st=True,
                 cross_st2conv=True,
                 cross_conv2st=True,
                 cross_conv2asr=True,
                 cross_asr2conv=True,
                 adapter_names=None,
                 reduction_factor=8,
                 ):
        """Construct an Decoder object."""
        torch.nn.Module.__init__(self)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim_tgt, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            self.embed_asr = torch.nn.Sequential(
                torch.nn.Embedding(odim_src, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            self.embed_conv = torch.nn.Sequential(
                torch.nn.Embedding(odim_wrsrc, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim_tgt, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            self.embed_asr = torch.nn.Sequential(
                torch.nn.Linear(odim_src, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            self.embed_conv = torch.nn.Sequential(
                torch.nn.Linear(odim_wrsrc, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            self.embed_asr = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
            self.embed_conv = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before

        self.adapter_names = adapter_names
        self.triple_decoders = repeat(
            num_blocks,
            lambda: TripleDecoderLayer(
                attention_dim, attention_dim, attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate), 
                MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate), 
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate), 
                MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate), 
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate), 
                MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate), 
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                cross_self_attn_asr2st=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate) if (cross_self and cross_asr2st) else None, 
                cross_self_attn_st2asr=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate) if (cross_self and cross_st2asr) else None, 
                cross_self_attn_conv2st=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate) if (cross_self and cross_conv2st) else None, 
                cross_self_attn_st2conv=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate) if (cross_self and cross_st2conv) else None, 
                cross_self_attn_asr2conv=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate) if (cross_self and cross_asr2conv) else None, 
                cross_self_attn_conv2asr=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate) if (cross_self and cross_conv2asr) else None, 
                cross_src_attn_asr2st=MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate) if (cross_src and cross_asr2st) else None, 
                cross_src_attn_st2asr=MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate) if (cross_src and cross_st2asr) else None, 
                cross_src_attn_conv2st=MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate) if (cross_src and cross_conv2st) else None, 
                cross_src_attn_st2conv=MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate) if (cross_src and cross_st2conv) else None, 
                cross_src_attn_asr2conv=MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate) if (cross_src and cross_asr2conv) else None, 
                cross_src_attn_conv2asr=MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate) if (cross_src and cross_conv2asr) else None, 
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after,
                cross_operator=cross_operator,
                cross_weight_learnable=cross_weight_learnable,
                cross_weight=cross_weight,
                cross_st2asr=cross_st2asr,
                cross_conv2st=cross_conv2st,
                cross_st2conv=cross_st2conv,
                cross_asr2conv=cross_asr2conv,
                cross_conv2asr=cross_conv2asr,
                cross_asr2st=cross_asr2st,
                adapters=nn.ModuleDict({k: Adapter(attention_dim, attention_dim//reduction_factor) 
                        for k in adapter_names}) if adapter_names else None,
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
            self.after_norm_asr = LayerNorm(attention_dim)
            self.after_norm_conv = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, odim_tgt)
            self.output_layer_asr = torch.nn.Linear(attention_dim, odim_src)
            self.output_layer_conv = torch.nn.Linear(attention_dim, odim_wrsrc)
        else:
            self.output_layer = None
            self.output_layer_asr = None
            self.output_layer_conv = None

    def forward(self, tgt, tgt_mask, tgt_asr, tgt_mask_asr, tgt_conv, tgt_mask_conv, 
                memory, memory_mask, 
                cross_mask_asr2st, cross_mask_st2asr, 
                cross_mask_conv2st, cross_mask_st2conv, 
                cross_mask_asr2conv, cross_mask_conv2asr, 
                cross_self=False, cross_src=False,
                cross_self_from="before-self", cross_src_from="before-src"):
        """Forward decoder.

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out) if input_layer == "embed"
                                 input tensor (batch, maxlen_out, #mels) in the other cases
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask,  (batch, maxlen_in)
                                         dtype=torch.uint8 in PyTorch 1.2-
                                         dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :return x: decoded token score before softmax (batch, maxlen_out, token) if use_output_layer is True,
                   final block outputs (batch, maxlen_out, attention_dim) in the other cases
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        x_asr = self.embed_asr(tgt_asr)
        x_conv = self.embed_conv(tgt_conv)
        if self.adapter_names:
            lang_id = str(tgt[:, 0:1][0].item())
        else:
            lang_id = None
        x, tgt_mask, x_asr, tgt_mask_asr, x_conv, tgt_mask_conv, memory, memory_mask, _, _, _, _, _, _, _, _, _, _ = self.triple_decoders(x, tgt_mask, x_asr, tgt_mask_asr, x_conv, tgt_mask_conv, 
                                                                                        memory, memory_mask, 
                                                                                        cross_mask_asr2st, cross_mask_st2asr, 
                                                                                        cross_mask_conv2st, cross_mask_st2conv, 
                                                                                        cross_mask_asr2conv, cross_mask_conv2asr, 
                                                                                        cross_self, cross_src, cross_self_from, cross_src_from,
                                                                                        lang_id)
        if self.normalize_before:
            x = self.after_norm(x)
            x_asr = self.after_norm_asr(x_asr)
            x_conv = self.after_norm_conv(x_conv)
        if self.output_layer is not None:
            x = self.output_layer(x)
            x_asr = self.output_layer_asr(x_asr)
            x_conv = self.output_layer_conv(x_conv)
        return x, tgt_mask, x_asr, tgt_mask_asr, x_conv, tgt_mask_conv

    def forward_one_step(self, tgt, tgt_mask, 
                        tgt_asr, tgt_mask_asr, 
                        tgt_conv, tgt_mask_conv, 
                        memory, 
                        cross_mask_asr2st=None, cross_mask_st2asr=None, 
                        cross_mask_conv2st=None, cross_mask_st2conv=None, 
                        cross_mask_asr2conv=None, cross_mask_conv2asr=None, 
                        cross_self=False, cross_src=False,
                        cross_self_from="before-self", cross_src_from="before-src",
                        cache=None, cache_asr=None, cache_conv=None):
        """Forward one step.

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param List[torch.Tensor] cache: cached output list of (batch, max_time_out-1, size)
        :return y, cache: NN output value and cache per `self.decoders`.
            `y.shape` is (batch, maxlen_out, token)
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        x = self.embed(tgt)
        x_asr = self.embed_asr(tgt_asr)
        x_conv = self.embed_conv(tgt_conv)
        if cache is None:
            cache = self.init_state()
        if cache_asr is None:
            cache_asr = self.init_state()
        if cache_conv is None:
            cache_conv = self.init_state()
        new_cache = []
        new_cache_asr = []
        new_cache_conv = []
        for c, c_asr, c_conv, triple_decoder in zip(cache, cache_asr, cache_conv, self.triple_decoders):
            x, tgt_mask, x_asr, tgt_mask_asr, x_conv, tgt_mask_conv, memory, _, _, _, _, _, _, _, _, _, _, _ = triple_decoder(x, tgt_mask, x_asr, tgt_mask_asr, x_conv, tgt_mask_conv, 
                                                                                    memory, None, 
                                                                                    cross_mask_asr2st, cross_mask_st2asr, 
                                                                                    cross_mask_conv2st, cross_mask_st2conv, 
                                                                                    cross_mask_asr2conv, cross_mask_conv2asr, 
                                                                                    cross_self, cross_src, 
                                                                                    cross_self_from, cross_src_from,
                                                                                    cache=c, cache_asr=c_asr, cache_conv=c_conv)
            new_cache.append(x)
            new_cache_asr.append(x_asr)
            new_cache_conv.append(x_conv)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
            y_asr = self.after_norm_asr(x_asr[:, -1])
            y_conv = self.after_norm_conv(x_conv[:, -1])
        else:
            y = x[:, -1]
            y_asr = x_asr[:, -1]
            y_conv = x_conv[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
            y_asr = torch.log_softmax(self.output_layer_asr(y_asr), dim=-1)
            y_conv = torch.log_softmax(self.output_layer_conv(y_conv), dim=-1)

        return y, new_cache, y_asr, new_cache_asr, y_conv, new_cache_conv

    # beam search API (see ScorerInterface)
    def init_state(self, x=None):
        """Get an initial state for decoding."""
        return [None for i in range(len(self.triple_decoders))]
