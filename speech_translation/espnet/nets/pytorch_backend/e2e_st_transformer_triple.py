# Copyright 2019 Hang Le (hangtp.le@gmail.com)

"""Transformer speech translation model (pytorch)."""

import sys

from argparse import Namespace
from distutils.util import strtobool

import logging
import math
import six
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_st import TripleDecoderReporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_triple import TripleDecoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask, create_cross_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.st_interface import STInterface
from espnet.nets.e2e_asr_common import end_detect

from nltk.translate.bleu_score import corpus_bleu


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
    

def build_embedding(dictionary, embed_dim, padding_idx=0):
    num_embeddings = max(list(dictionary.values())) + 1
    emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
    return emb


def end_sentence(sen):
    ended_sen = []
    for word in sen:
        if word == '<eos>':
            break
        ended_sen.append(word)
    return ended_sen


class E2ETripleDecoder(STInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')

        # Adapters
        group.add_argument('--adapter-reduction-factor', default=8, type=int,
                           help='Reduction factor in bottle neck of adapter modules')
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim_tgt, odim_src, odim_wrsrc, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.char_list_tgt = args.char_list_tgt

        # special tokens and model dimensions
        self.pad = 0
        self.sos_tgt = odim_tgt - 1
        self.eos_tgt = odim_tgt - 1
        self.odim_tgt = odim_tgt
        self.sos_src = odim_src - 1
        self.eos_src = odim_src - 1
        self.odim_src = odim_src
        self.sos_wrsrc = odim_wrsrc - 1
        self.eos_wrsrc = odim_wrsrc - 1
        self.odim_wrsrc = odim_wrsrc
        self.idim = idim
        self.adim = args.adim
        self.ignore_id = ignore_id

        # submodule for ASR and conversion task
        self.mtlalpha = args.mtlalpha
        self.asr_weight = getattr(args, "asr_weight", 0.0)
        self.conv_weight = getattr(args, "conv_weight", 0.0)
        self.do_asr = self.asr_weight > 0 and args.mtlalpha < 1
        self.do_conv = self.conv_weight > 0

        # cross-attention parameters
        self.cross_weight = getattr(args, "cross_weight", 0.0)
        self.cross_self = getattr(args, "cross_self", False)
        self.cross_src = getattr(args, "cross_src", False)
        self.cross_operator = getattr(args, "cross_operator", None)
        self.cross_st2asr = getattr(args, "cross_st2asr", False)
        self.cross_conv2asr = getattr(args, "cross_conv2asr", False)
        self.cross_st2conv = getattr(args, "cross_st2conv", False)
        self.cross_asr2conv = getattr(args, "cross_asr2conv", False)
        self.cross_conv2st = getattr(args, "cross_conv2st", False)
        self.cross_asr2st = getattr(args, "cross_asr2st", False)
        self.num_decoders = getattr(args, "num_decoders", 1)
        self.wait_k_asr = getattr(args, "wait_k_asr", 0)
        self.wait_k_conv = getattr(args, "wait_k_conv", 0)
        self.cross_src_from = getattr(args, "cross_src_from", "embedding")
        self.cross_self_from = getattr(args, "cross_self_from", "embedding")
        self.cross_weight_learnable = getattr(args, "cross_weight_learnable", False)

        # one-to-many ST experiments
        self.use_joint_dict = getattr(args, "use_joint_dict", True)
        self.one_to_many = getattr(args, "one_to_many", False)
        self.use_lid = getattr(args, "use_lid", False)
        if self.use_joint_dict:
            self.langs_dict = getattr(args, "langs_dict_tgt", None)
        self.lang_tok = getattr(args, "lang_tok", None)

        self.subsample = get_subsample(args, mode='st', arch='transformer')
        self.reporter = TripleDecoderReporter()
        self.normalize_before = getattr(args, "normalize_before", True)

        # Check parameters
        if self.one_to_many:
            # assert self.use_lid
            self.use_lid = True
        if self.cross_operator == 'sum' and self.cross_weight <= 0:
            assert (not self.cross_st2asr) and (not self.cross_conv2asr) \
                    and (not self.cross_st2conv) and (not self.cross_asr2conv) \
                    and (not self.cross_asr2st) and (not self.cross_conv2st)
        if self.cross_st2asr or self.cross_asr2st \
                or self.cross_conv2asr or self.cross_asr2conv:
            assert self.do_asr
            assert self.cross_self or self.cross_src
        if self.cross_st2conv or self.cross_conv2st \
                or self.cross_conv2asr or self.cross_asr2conv:
            assert self.do_conv
            assert self.cross_self or self.cross_src
        assert bool(self.cross_operator) == ((self.do_asr and (self.cross_st2asr or self.cross_asr2st or self.cross_conv2asr or self.cross_asr2conv)) \
               or (self.do_conv and (self.cross_st2conv or self.cross_conv2st or self.cross_conv2asr or self.cross_asr2conv)))
        if self.cross_src_from != "embedding" or self.cross_self_from != "embedding":
            assert self.normalize_before
        assert self.wait_k_conv >= 0
        assert self.wait_k_asr >= 0

        logging.info("*** Parallel attention parameters ***")
        if self.cross_st2asr:
            logging.info("| Cross ST to ASR")
        if self.cross_asr2st:
            logging.info("| Cross ASR to ST")
        if self.cross_conv2asr:
            logging.info("| Cross Conversion to ASR")
        if self.cross_asr2conv:
            logging.info("| Cross ASR to Conversion")
        if self.cross_st2conv:
            logging.info("| Cross ST to Conversion")
        if self.cross_conv2st:
            logging.info("| Cross Conversion to ST")
        if self.cross_self:
            logging.info("| Cross at Self")
        if self.cross_src:
            logging.info("| Cross at Source")
        if self.cross_st2asr or self.cross_asr2st \
                or self.cross_conv2asr or self.cross_asr2conv \
                or self.cross_st2conv or self.cross_conv2st:
            logging.info(f'| Cross operator: {self.cross_operator}')
            logging.info(f'| Cross sum weight: {self.cross_weight}')
            if self.cross_src:
                logging.info(f'| Cross source from: {self.cross_src_from}')
            if self.cross_self:
                logging.info(f'| Cross self from: {self.cross_self_from}')
        logging.info(f'| wait_k_conv = {self.wait_k_conv}')
        logging.info(f'| wait_k_asr = {self.wait_k_asr}')
        
        if (self.cross_src_from != "embedding" and self.cross_src) and (not self.normalize_before):
            logging.warning(f'WARNING: Resort to using self.cross_src_from == embedding for cross at source attention.')
        if (self.cross_self_from != "embedding" and self.cross_self) and (not self.normalize_before):
            logging.warning(f'WARNING: Resort to using self.cross_self_from == embedding for cross at self attention.')
        
        # Adapters
        self.use_adapters = getattr(args, "use_adapters", False)
        adapter_reduction_factor = getattr(args, "adapter_reduction_factor", 8)
        adapter_names = getattr(args, "adapters", None)
        # convert target language tokens to ids
        if adapter_names:
            adapter_names = [str(args.char_list_tgt.index(f'<2{l}>')) for l in adapter_names]
        logging.info(f'| adapters = {adapter_names}')

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            adapter_names=adapter_names,
            reduction_factor=adapter_reduction_factor,
        )

        self.triple_decoder = TripleDecoder(
                odim_tgt,
                odim_src,
                odim_wrsrc,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                normalize_before=self.normalize_before,
                cross_operator=self.cross_operator,
                cross_weight_learnable=self.cross_weight_learnable,
                cross_weight=self.cross_weight,
                cross_self=self.cross_self,
                cross_src=self.cross_src,
                cross_st2asr=self.cross_st2asr,
                cross_asr2st=self.cross_asr2st,
                cross_st2conv=self.cross_st2conv,
                cross_conv2st=self.cross_conv2st,
                cross_conv2asr=self.cross_conv2asr,
                cross_asr2conv=self.cross_asr2conv,
                use_output_layer=True if self.use_joint_dict else False,
                adapter_names=adapter_names,
                reduction_factor=adapter_reduction_factor,
        )

        if not self.use_joint_dict:
            self.output_layer = torch.nn.Linear(args.adim, odim_tgt)
            if self.do_asr:
                self.output_layer_asr = torch.nn.Linear(args.adim, odim_src)
            if self.do_conv:
                self.output_layer_conv = torch.nn.Linear(args.adim, odim_wrsrc)
       
        # submodule for MT task
        self.mt_weight = getattr(args, "mt_weight", 0.0)
        if self.mt_weight > 0:
            self.encoder_mt = Encoder(
                idim=odim_tgt,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                input_layer='embed',
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                padding_idx=0
            )
        self.reset_parameters(args)  # place after the submodule initialization
        if args.mtlalpha > 0.0:
            self.ctc = CTC(odim_tgt, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        else:
            self.ctc = None

        if self.asr_weight > 0 and (args.report_cer or args.report_wer):
            from espnet.nets.e2e_asr_common import ErrorCalculator
            self.error_calculator = ErrorCalculator(args.char_list,
                                                    args.sym_space, args.sym_blank,
                                                    args.report_cer, args.report_wer)
        else:
            self.error_calculator = None
        self.rnnlm = None

        self.criterion_st = LabelSmoothingLoss(self.odim_tgt, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        self.criterion_asr = LabelSmoothingLoss(self.odim_src, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        self.criterion_conv = LabelSmoothingLoss(self.odim_wrsrc, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)

        # Language embedding layer
        if self.lang_tok == "encoder-pre-sum":
            self.language_embeddings = build_embedding(self.langs_dict, self.idim, padding_idx=self.pad)
            logging.info(f'language_embeddings: {self.language_embeddings}')

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)
        if self.mt_weight > 0:
            torch.nn.init.normal_(self.encoder_mt.embed[0].weight, mean=0, std=args.adim ** -0.5)
            torch.nn.init.constant_(self.encoder_mt.embed[0].weight[self.pad], 0)

    def forward(self, xs_pad, ilens, ys_pad, ys_pad_src, ys_pad_wrsrc):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_pad_src: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 0. Extract target language ID
        # src_lang_ids = None
        tgt_lang_ids, tgt_lang_ids_src, enc_lang_ids = None, None, None
        if self.use_lid:
            tgt_lang_ids = ys_pad[:, 0:1]
            enc_lang_ids = str(tgt_lang_ids[0].data.cpu().numpy()[0])
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
            
            if self.do_asr:
                tgt_lang_ids_src = ys_pad_src[:, 0:1]
                ys_pad_src = ys_pad_src[:, 1:]  # remove target language ID in the beggining

            if self.do_conv:
                tgt_lang_ids_wrsrc = ys_pad_wrsrc[:, 0:1]
                ys_pad_wrsrc = ys_pad_wrsrc[:, 1:]  # remove target language ID in the beggining

        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel # bs x max_ilens x idim

        if self.lang_tok == "encoder-pre-sum":
            lang_embed = self.language_embeddings(tgt_lang_ids) # bs x 1 x idim
            xs_pad = xs_pad + lang_embed

        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2) # bs x 1 x max_ilens

        hs_pad, hs_mask = self.encoder(xs_pad, src_mask, enc_lang_ids)

        self.hs_pad = hs_pad

        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos_tgt, self.eos_tgt, self.ignore_id) # bs x max_lens

        if self.do_asr:
            ys_in_pad_src, ys_out_pad_src = add_sos_eos(ys_pad_src, self.sos_src, self.eos_src, self.ignore_id) # bs x max_lens_src

        if self.do_conv:
            ys_in_pad_wrsrc, ys_out_pad_wrsrc = add_sos_eos(ys_pad_wrsrc, self.sos_wrsrc, self.eos_wrsrc, self.ignore_id) # bs x max_lens_wrsrc

        if self.lang_tok == "decoder-pre":
            ys_in_pad = torch.cat([tgt_lang_ids, ys_in_pad[:, 1:]], dim=1)
            if self.do_asr:
                ys_in_pad_src = torch.cat([tgt_lang_ids_src, ys_in_pad_src[:, 1:]], dim=1)
            if self.do_conv:
                ys_in_pad_wrsrc = torch.cat([tgt_lang_ids_wrsrc, ys_in_pad_wrsrc[:, 1:]], dim=1)

        ys_mask = target_mask(ys_in_pad, self.ignore_id) # bs x max_lens x max_lens

        if self.do_asr:
            ys_mask_src = target_mask(ys_in_pad_src, self.ignore_id) # bs x max_lens x max_lens_src

        if self.do_conv:
            ys_mask_wrsrc = target_mask(ys_in_pad_wrsrc, self.ignore_id) # bs x max_lens x max_lens_wrsrc

        if self.wait_k_conv > 0:
            pass
        else:
            self.wait_k_conv = 0

        if self.wait_k_asr > 0:
            pass
        else:
            self.wait_k_asr = 0

        cross_mask_asr2st = create_cross_mask(ys_in_pad, ys_in_pad_src, self.ignore_id, wait_k_cross=self.wait_k_asr+self.wait_k_conv)
        cross_mask_st2asr = create_cross_mask(ys_in_pad_src, ys_in_pad, self.ignore_id, wait_k_cross=-self.wait_k_asr-self.wait_k_conv)
        cross_mask_conv2st = create_cross_mask(ys_in_pad, ys_in_pad_wrsrc, self.ignore_id, wait_k_cross=self.wait_k_conv)
        cross_mask_st2conv = create_cross_mask(ys_in_pad_wrsrc, ys_in_pad, self.ignore_id, wait_k_cross=-self.wait_k_conv)
        cross_mask_asr2conv = create_cross_mask(ys_in_pad_wrsrc, ys_in_pad_src, self.ignore_id, wait_k_cross=self.wait_k_asr)
        cross_mask_conv2asr = create_cross_mask(ys_in_pad_src, ys_in_pad_wrsrc, self.ignore_id, wait_k_cross=-self.wait_k_asr)

        pred_pad, pred_mask, pred_pad_asr, pred_mask_asr, pred_pad_conv, pred_mask_conv = self.triple_decoder(ys_in_pad, ys_mask, ys_in_pad_src, ys_mask_src, ys_in_pad_wrsrc, ys_mask_wrsrc,
                                                                                hs_pad, hs_mask, 
                                                                                cross_mask_asr2st, cross_mask_st2asr,
                                                                                cross_mask_conv2st, cross_mask_st2conv,
                                                                                cross_mask_asr2conv, cross_mask_conv2asr,
                                                                                cross_self=self.cross_self, cross_src=self.cross_src,
                                                                                cross_self_from=self.cross_self_from,
                                                                                cross_src_from=self.cross_src_from)
        if not self.use_joint_dict:
            pred_pad = self.output_layer(pred_pad)
            pred_pad_asr = self.output_layer_asr(pred_pad_asr)
            pred_pad_conv = self.output_layer_conv(pred_pad_conv)

        self.pred_pad = pred_pad
        self.pred_pad_asr = pred_pad_asr
        self.pred_pad_conv = pred_pad_conv
        pred_pad_mt = None

        # 3. compute attention loss
        loss_asr, loss_mt, loss_conv = 0.0, 0.0, 0.0
        loss_att = self.criterion_st(pred_pad, ys_out_pad)

        # compute loss
        loss_asr = self.criterion_asr(pred_pad_asr, ys_out_pad_src)
        loss_conv = self.criterion_conv(pred_pad_conv, ys_out_pad_wrsrc)
        # Multi-task w/ MT
        if self.mt_weight > 0:
            # forward MT encoder
            ilens_mt = torch.sum(ys_pad_src != self.ignore_id, dim=1).cpu().numpy()
            # NOTE: ys_pad_src is padded with -1
            ys_src = [y[y != self.ignore_id] for y in ys_pad_src]  # parse padded ys_src
            ys_zero_pad_src = pad_list(ys_src, self.pad)  # re-pad with zero
            ys_zero_pad_src = ys_zero_pad_src[:, :max(ilens_mt)]  # for data parallel
            src_mask_mt = (~make_pad_mask(ilens_mt.tolist())).to(ys_zero_pad_src.device).unsqueeze(-2)

            hs_pad_mt, hs_mask_mt = self.encoder_mt(ys_zero_pad_src, src_mask_mt)
            # forward MT decoder
            pred_pad_mt, _ = self.decoder(ys_in_pad, ys_mask, hs_pad_mt, hs_mask_mt)
            # compute loss
            loss_mt = self.criterion_st(pred_pad_mt, ys_out_pad)

        self.acc = th_accuracy(pred_pad.view(-1, self.odim_tgt), ys_out_pad,
                               ignore_label=self.ignore_id)
        if pred_pad_asr is not None:
            self.acc_asr = th_accuracy(pred_pad_asr.view(-1, self.odim_src), ys_out_pad_src,
                                       ignore_label=self.ignore_id)
        else:
            self.acc_asr = 0.0
        if pred_pad_conv is not None:
            self.acc_conv = th_accuracy(pred_pad_conv.view(-1, self.odim_wrsrc), ys_out_pad_wrsrc,
                                       ignore_label=self.ignore_id)
        else:
            self.acc_conv = 0.0
        if pred_pad_mt is not None:
            self.acc_mt = th_accuracy(pred_pad_mt.view(-1, self.odim_tgt), ys_out_pad,
                                      ignore_label=self.ignore_id)
        else:
            self.acc_mt = 0.0

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0 or self.asr_weight == 0:
            loss_ctc = 0.0
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad_src)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad_src.cpu(), is_ctc=True)

        # 5. compute cer/wer
        cer, wer = None, None  # TODO(hirofumi0810): fix later
        # if self.training or (self.asr_weight == 0 or self.mtlalpha == 1 or not (self.report_cer or self.report_wer)):
        #     cer, wer = None, None
        # else:
        #     ys_hat = pred_pad.argmax(dim=-1)
        #     cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copyied from e2e_asr

        y_hyp = pred_pad.argmax(2)
        seq_hyp = [[self.char_list_tgt[int(idx)] for idx in sen if int(idx) != -1] for sen in y_hyp]
        seq_ref = [[self.char_list_tgt[int(idx)] for idx in sen if int(idx) != -1] for sen in ys_out_pad]
        seq_hyp = [end_sentence(sen) for sen in seq_hyp]
        seq_ref = [[end_sentence(sen)] for sen in seq_ref]

        for sen_hyp, sen_ref in zip(seq_hyp, seq_ref):
            sen_hyp = " ".join(sen_hyp)
            sen_ref = " ".join(sen_ref[0])
            logging.info(f"HYP: {sen_hyp}")
            logging.info(f"REF: {sen_ref}")

        bleu = corpus_bleu(seq_ref, seq_hyp) * 100

        alpha = self.mtlalpha
        self.loss = (1 - self.asr_weight - self.mt_weight - self.conv_weight) * loss_att \
                + self.asr_weight * (alpha * loss_ctc + (1 - alpha) * loss_asr) \
                + self.conv_weight * loss_conv \
                + self.mt_weight * loss_mt
        loss_asr_data = float(alpha * loss_ctc + (1 - alpha) * loss_asr)
        loss_conv_data = float(loss_conv)
        loss_mt_data = None if self.mt_weight == 0 else float(loss_mt)
        loss_st_data = float(loss_att)

        loss_data = float(self.loss)

        cw_asr2st_l8 = float(self.triple_decoder.triple_decoders[7].cross_weight_asr2st) if hasattr(self.triple_decoder.triple_decoders[7], 'cross_weight_asr2st') else -1
        cw_st2asr_l8 = float(self.triple_decoder.triple_decoders[7].cross_weight_st2asr) if hasattr(self.triple_decoder.triple_decoders[7], 'cross_weight_st2asr') else -1
        cw_conv2st_l8 = float(self.triple_decoder.triple_decoders[7].cross_weight_conv2st) if hasattr(self.triple_decoder.triple_decoders[7], 'cross_weight_conv2st') else -1
        cw_st2conv_l8 = float(self.triple_decoder.triple_decoders[7].cross_weight_st2conv) if hasattr(self.triple_decoder.triple_decoders[7], 'cross_weight_st2conv') else -1
        cw_asr2conv_l8 = float(self.triple_decoder.triple_decoders[7].cross_weight_asr2conv) if hasattr(self.triple_decoder.triple_decoders[7], 'cross_weight_asr2conv') else -1
        cw_conv2asr_l8 = float(self.triple_decoder.triple_decoders[7].cross_weight_conv2asr) if hasattr(self.triple_decoder.triple_decoders[7], 'cross_weight_conv2asr') else -1

        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_asr_data, loss_conv_data, loss_mt_data, loss_st_data,
                                 self.acc_asr, self.acc_conv, self.acc_mt, self.acc,
                                 cer_ctc, cer, wer, bleu, loss_data,
                                 cw_asr2st_l8, cw_st2asr_l8,
                                 cw_conv2st_l8, cw_st2conv_l8,
                                 cw_asr2conv_l8, cw_conv2asr_l8)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder)

    def encode(self, x):
        """Encode source acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def recognize_convert_and_translate_sum(self, x, trans_args, 
                                    char_list_tgt=None,
                                    char_list_src=None,
                                    char_list_wrsrc=None,
                                    rnnlm=None,
                                    use_jit=False, 
                                    decode_asr_weight=1.0, 
                                    decode_conv_weight=1.0, 
                                    score_is_prob=False, 
                                    ratio_diverse_st=0.0,
                                    ratio_diverse_asr=0.0,
                                    ratio_diverse_conv=0.0,
                                    use_rev_triu_width=0,
                                    use_diag=False,
                                    debug=False):
        """Recognize and translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """

        assert self.do_asr and self.do_conv, "Recognize, convert and translate are performed simultaneously."
        assert 0 <= ratio_diverse_asr < 1
        assert 0 <= ratio_diverse_conv < 1
        assert 0 <= ratio_diverse_st < 1

        if self.use_lid and self.lang_tok == 'decoder-pre':
            tgt_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[-1])
            y = char_list_tgt.index(tgt_lang_id)
            logging.info(f'tgt_lang_id: {tgt_lang_id} - y: {y}')

            src_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[0])
            y_asr = char_list_src.index(src_lang_id)
            logging.info(f'src_lang_id: {src_lang_id} - y_asr: {y_asr}')

            wrsrc_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[0])
            y_conv = char_list_wrsrc.index(wrsrc_lang_id)
            logging.info(f'wrsrc_lang_id: {wrsrc_lang_id} - y_conv: {y_conv}')
        else:
            y = self.sos_tgt
            y_asr = self.sos_src
            y_conv = self.sos_wrsrc
        
        logging.info(f'<sos> index: {str(y)}; <sos> mark: {char_list_tgt[y]}')
        logging.info(f'<sos> index asr: {str(y_asr)}; <sos> mark asr: {char_list_src[y_asr]}')
        logging.info(f'<sos> index conv: {str(y_conv)}; <sos> mark conv: {char_list_wrsrc[y_conv]}')

        enc_output = self.encode(x).unsqueeze(0)
        h = enc_output.squeeze(0)
        logging.info('input lengths: ' + str(h.size(0)))

        # search params
        beam = trans_args.beam_size
        penalty = trans_args.penalty
        cr = math.ceil((1 - ratio_diverse_asr) * beam)
        cc = math.ceil((1 - ratio_diverse_conv) * beam)
        ct = math.ceil((1 - ratio_diverse_st) * beam)

        vy = h.new_zeros(1).long()

        if trans_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            maxlen = max(1, int(trans_args.maxlenratio * h.size(0)))
        if trans_args.maxlenratio_asr == 0:
            maxlen_asr = h.shape[0]
        else:
            maxlen_asr = max(1, int(trans_args.maxlenratio_asr * h.size(0)))
        if trans_args.maxlenratio_conv == 0:
            maxlen_conv = h.shape[0]
        else:
            maxlen_conv = max(1, int(trans_args.maxlenratio_conv * h.size(0)))
        minlen = int(trans_args.minlenratio * h.size(0))
        minlen_asr = int(trans_args.minlenratio_asr * h.size(0))
        minlen_conv = int(trans_args.minlenratio_conv * h.size(0))
        logging.info(f'max output length: {str(maxlen)}; min output length: {str(minlen)}')
        logging.info(f'max output length asr: {str(maxlen_asr)}; min output length asr: {str(minlen_asr)}')
        logging.info(f'max output length conv: {str(maxlen_conv)}; min output length conv: {str(minlen_conv)}')

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            logging.info('initializing hypothesis...')
            hyp = {'score': 0.0, 'yseq': [y], 'yseq_asr': [y_asr], 'yseq_conv': [y_conv]}

        hyps = [hyp]
        ended_hyps = []

        traced_decoder = None
        for i in six.moves.range(max(maxlen, maxlen_asr, maxlen_conv)):
            logging.info('position ' + str(i))

            hyps_best_kept = []

            for idx, hyp in enumerate(hyps):
                ys_mask = subsequent_mask(len(hyp['yseq'])).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)

                ys_mask_asr = subsequent_mask(len(hyp['yseq_asr'])).unsqueeze(0)
                ys_asr = torch.tensor(hyp['yseq_asr']).unsqueeze(0)

                ys_mask_conv = subsequent_mask(len(hyp['yseq_conv'])).unsqueeze(0)
                ys_conv = torch.tensor(hyp['yseq_conv']).unsqueeze(0)

                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    if hyp['yseq'][-1] != self.eos_tgt or hyp['yseq_asr'][-1] != self.eos_src or hyp['yseq_conv'][-1] != self.eos_wrsrc or i < 2:
                        cross_mask_asr2st = create_cross_mask(ys, ys_asr, self.ignore_id, wait_k_cross=self.wait_k_asr+self.wait_k_conv)
                        cross_mask_st2asr = create_cross_mask(ys_asr, ys, self.ignore_id, wait_k_cross=-self.wait_k_asr-self.wait_k_conv)
                        cross_mask_conv2st = create_cross_mask(ys, ys_conv, self.ignore_id, wait_k_cross=self.wait_k_conv)
                        cross_mask_st2conv = create_cross_mask(ys_conv, ys, self.ignore_id, wait_k_cross=-self.wait_k_conv)
                        cross_mask_asr2conv = create_cross_mask(ys_conv, ys_asr, self.ignore_id, wait_k_cross=self.wait_k_asr)
                        cross_mask_conv2asr = create_cross_mask(ys_asr, ys_conv, self.ignore_id, wait_k_cross=-self.wait_k_asr)
                        local_att_scores, _, local_att_scores_asr, _, local_att_scores_conv, _ = self.triple_decoder.forward_one_step(ys, ys_mask, ys_asr, ys_mask_asr, ys_conv, ys_mask_conv, enc_output,
                                                                                                          cross_mask_asr2st=cross_mask_asr2st, cross_mask_st2asr=cross_mask_st2asr,
                                                                                                          cross_mask_conv2st=cross_mask_conv2st, cross_mask_st2conv=cross_mask_st2conv,
                                                                                                          cross_mask_asr2conv=cross_mask_asr2conv, cross_mask_conv2asr=cross_mask_conv2asr,
                                                                                                          cross_self=self.cross_self, cross_src=self.cross_src,
                                                                                                          cross_self_from=self.cross_self_from,
                                                                                                          cross_src_from=self.cross_src_from)
                        # If using 2 separate dictionaries
                        if not self.use_joint_dict:
                            local_att_scores_conv = self.output_layer_conv(local_att_scores_conv)
                            local_att_scores_asr = self.output_layer_asr(local_att_scores_asr)
                            local_att_scores = self.output_layer(local_att_scores)

                    if (hyp['yseq'][-1] == self.eos_tgt and i > 0) or i < self.wait_k_asr + self.wait_k_conv:
                        local_att_scores = None
                    if (hyp['yseq_asr'][-1] == self.eos_src and i > 0):
                        local_att_scores_asr = None
                    if (hyp['yseq_conv'][-1] == self.eos_wrsrc and i > 0) or i < self.wait_k_asr:
                        local_att_scores_conv = None

                if local_att_scores is not None and local_att_scores_asr is not None and local_att_scores_conv is not None:
                    local_att_scores_asr = decode_asr_weight * local_att_scores_asr
                    local_att_scores_conv = decode_conv_weight * local_att_scores_conv
                    xk, ixk = local_att_scores.topk(beam)
                    yk, iyk = local_att_scores_asr.topk(beam)
                    zk, izk = local_att_scores_conv.topk(beam)
                    S = torch.mm(torch.t(torch.ones_like(xk)), xk) + torch.mm(torch.t(yk), torch.ones_like(yk))
                    S = zk.reshape(-1, 1, 1) + S
                    s2v = torch.LongTensor([[i, j, k] for i in ixk.squeeze(0) for j in iyk.squeeze(0) for k in izk.squeeze(0)]) # (k^3) x 3

                    # Get best tokens
                    s2v = s2v.reshape(beam, beam, beam, 3)
                    Sc = S[:cc, :cr, :ct]
                    if use_diag:
                        mask = np.add.outer(np.arange(cr), -np.arange(ct)[::-1])
                        mask = np.add.outer(-np.arange(cc)[::-1], mask) > 0
                        mask = torch.from_numpy(mask)
                        Sc = Sc.masked_fill(mask, float('-inf'))
                    if use_rev_triu_width > 0:
                        raise NotImplementedError
                    local_best_scores, id3k = Sc.flatten().topk(beam)
                    I = s2v[:cc, :cr, :ct]
                    I = I.reshape(-1, 3)
                    I = I[id3k]
                    local_best_ids_st = I[:,0]
                    local_best_ids_asr = I[:,1]
                    local_best_ids_conv = I[:,2]

                elif local_att_scores is not None and local_att_scores_asr is not None:
                    local_att_scores_asr = decode_asr_weight * local_att_scores_asr
                    xk, ixk = local_att_scores.topk(beam)
                    yk, iyk = local_att_scores_asr.topk(beam)
                    S = (torch.mm(torch.t(xk), torch.ones_like(xk))
                                    + torch.mm(torch.t(torch.ones_like(yk)), yk))
                    s2v = torch.LongTensor([[i, j] for i in ixk.squeeze(0) for j in iyk.squeeze(0)]) # (k^2) x 2

                    # Get best tokens
                    s2v = s2v.reshape(beam, beam, 2)
                    Sc = S[:cr, :ct]
                    if use_diag:
                        mask = np.add.outer(np.arange(cr), -np.arange(ct)[::-1]) > 0
                        mask = torch.from_numpy(mask)
                        Sc = Sc.masked_fill(mask, float('-inf'))
                    if use_rev_triu_width > 0:
                        mask = np.abs(np.add.outer(np.arange(cr), -np.arange(ct))) >= use_rev_triu_width
                        mask = torch.from_numpy(mask)
                        Sc = Sc.masked_fill(mask, float('-inf'))
                    local_best_scores, id2k = Sc.flatten().topk(beam)
                    I = s2v[:cr, :ct]
                    I = I.reshape(-1, 2)
                    I = I[id2k]
                    local_best_ids_st = I[:,0]
                    local_best_ids_asr = I[:,1]
                elif local_att_scores is not None and local_att_scores_conv is not None:
                    local_att_scores_conv = decode_conv_weight * local_att_scores_conv
                    xk, ixk = local_att_scores.topk(beam)
                    yk, iyk = local_att_scores_conv.topk(beam)
                    S = (torch.mm(torch.t(xk), torch.ones_like(xk))
                                    + torch.mm(torch.t(torch.ones_like(yk)), yk))
                    s2v = torch.LongTensor([[i, j] for i in ixk.squeeze(0) for j in iyk.squeeze(0)]) # (k^2) x 2

                    # Get best tokens
                    s2v = s2v.reshape(beam, beam, 2)
                    Sc = S[:cc, :ct]
                    if use_diag:
                        mask = np.add.outer(np.arange(cc), -np.arange(ct)[::-1]) > 0
                        mask = torch.from_numpy(mask)
                        Sc = Sc.masked_fill(mask, float('-inf'))
                    if use_rev_triu_width > 0:
                        mask = np.abs(np.add.outer(np.arange(cc), -np.arange(ct))) >= use_rev_triu_width
                        mask = torch.from_numpy(mask)
                        Sc = Sc.masked_fill(mask, float('-inf'))
                    local_best_scores, id2k = Sc.flatten().topk(beam)
                    I = s2v[:cc, :ct]
                    I = I.reshape(-1, 2)
                    I = I[id2k]
                    local_best_ids_st = I[:,0]
                    local_best_ids_conv = I[:,1]
                elif local_att_scores_conv is not None and local_att_scores_asr is not None:
                    local_att_scores_asr = decode_asr_weight * local_att_scores_asr
                    local_att_scores_conv = decode_conv_weight * local_att_scores_conv
                    xk, ixk = local_att_scores_conv.topk(beam)
                    yk, iyk = local_att_scores_asr.topk(beam)
                    S = (torch.mm(torch.t(xk), torch.ones_like(xk))
                                    + torch.mm(torch.t(torch.ones_like(yk)), yk))
                    s2v = torch.LongTensor([[i, j] for i in ixk.squeeze(0) for j in iyk.squeeze(0)]) # (k^2) x 2

                    # Get best tokens
                    s2v = s2v.reshape(beam, beam, 2)
                    Sc = S[:cr, :cc]
                    if use_diag:
                        mask = np.add.outer(np.arange(cr), -np.arange(cc)[::-1]) > 0
                        mask = torch.from_numpy(mask)
                        Sc = Sc.masked_fill(mask, float('-inf'))
                    if use_rev_triu_width > 0:
                        mask = np.abs(np.add.outer(np.arange(cr), -np.arange(cc))) >= use_rev_triu_width
                        mask = torch.from_numpy(mask)
                        Sc = Sc.masked_fill(mask, float('-inf'))
                    local_best_scores, id2k = Sc.flatten().topk(beam)
                    I = s2v[:cr, :cc]
                    I = I.reshape(-1, 2)
                    I = I[id2k]
                    local_best_ids_conv = I[:,0]
                    local_best_ids_asr = I[:,1]

                elif local_att_scores is not None:
                    local_best_scores, local_best_ids_st = torch.topk(local_att_scores, beam, dim=1)
                    local_best_scores = local_best_scores.squeeze(0)
                    local_best_ids_st = local_best_ids_st.squeeze(0)
                elif local_att_scores_asr is not None:
                    local_best_scores, local_best_ids_asr = torch.topk(local_att_scores_asr, beam, dim=1)
                    local_best_ids_asr = local_best_ids_asr.squeeze(0)
                    local_best_scores = local_best_scores.squeeze(0) 
                elif local_att_scores_conv is not None:
                    local_best_scores, local_best_ids_conv = torch.topk(local_att_scores_conv, beam, dim=1)
                    local_best_ids_conv = local_best_ids_conv.squeeze(0)
                    local_best_scores = local_best_scores.squeeze(0) 
                else:
                    raise NotImplementedError

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']

                    new_hyp['yseq_asr'] = [0] * (1 + len(hyp['yseq_asr']))
                    new_hyp['yseq_asr'][:len(hyp['yseq_asr'])] = hyp['yseq_asr']

                    new_hyp['yseq_conv'] = [0] * (1 + len(hyp['yseq_conv']))
                    new_hyp['yseq_conv'][:len(hyp['yseq_conv'])] = hyp['yseq_conv']

                    if local_att_scores is not None:
                        new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids_st[j])
                    else:
                        if i >= self.wait_k_asr + self.wait_k_conv:
                            new_hyp['yseq'][len(hyp['yseq'])] = self.eos_tgt
                        else:
                            new_hyp['yseq'] = hyp['yseq'] # v3
                    
                    if local_att_scores_asr is not None:
                        new_hyp['yseq_asr'][len(hyp['yseq_asr'])] = int(local_best_ids_asr[j])
                    else:
                        new_hyp['yseq_asr'] = hyp['yseq_asr'] # v3

                    if local_att_scores_conv is not None:
                        new_hyp['yseq_conv'][len(hyp['yseq_conv'])] = int(local_best_ids_conv[j])
                    else:
                        if i >= self.wait_k_asr:
                            new_hyp['yseq_conv'][len(hyp['yseq_conv'])] = self.eos_wrsrc
                        else:
                            new_hyp['yseq_conv'] = hyp['yseq_conv'] # v3

                    hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # if ratio_diverse_st==0 and ratio_diverse_asr==0:
            #     hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]
            # else:
            #     hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)
            #     hyps_best_kept_tmp = []
            #     last_tokens_st = dict.fromkeys(set([h['yseq'][-1] for h in hyps_best_kept]), 0)
            #     last_tokens_asr = dict.fromkeys(set([h['yseq_asr'][-1] for h in hyps_best_kept]), 0)
            #     count = 0
            #     for ii, hyps in enumerate(hyps_best_kept):
            #         if last_tokens_asr[hyps['yseq_asr'][-1]] < cr and last_tokens_st[hyps['yseq'][-1]] < ct:
            #             hyps_best_kept_tmp.append(hyps)
            #             last_tokens_st[hyps['yseq'][-1]] += 1
            #             last_tokens_asr[hyps['yseq_asr'][-1]] += 1
            #             count += 1
            #             if count == beam:
            #                 break
            #     hyps_best_kept = hyps_best_kept_tmp

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    if hyp['yseq'][-1] != self.eos_tgt:
                        hyp['yseq'].append(self.eos_tgt)
            if i == maxlen_asr - 1:
                logging.info('adding <eos> in the last postion in the loop for asr')
                for hyp in hyps:
                    if hyp['yseq_asr'][-1] != self.eos_src:
                        hyp['yseq_asr'].append(self.eos_src)
            if i == maxlen_conv - 1:
                logging.info('adding <eos> in the last postion in the loop for conversion')
                for hyp in hyps:
                    if hyp['yseq_conv'][-1] != self.eos_wrsrc:
                        hyp['yseq_conv'].append(self.eos_wrsrc)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []

            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos_tgt and hyp['yseq_asr'][-1] == self.eos_src and hyp['yseq_conv'][-1] == self.eos_wrsrc:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen and len(hyp['yseq_asr']) > minlen_asr and len(hyp['yseq_conv']) > minlen_conv:
                        hyp['score'] += (i + 1) * penalty
 
                        # if rnnlm:  # Word LM needs to add final <eos> score
                        #     hyp['score'] += trans_args.lm_weight * rnnlm.final(
                        #         hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection          
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.info('remained hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list_tgt is not None and char_list_src is not None and char_list_wrsrc is not None:
                for hyp in hyps:
                    logging.info('hypo: ' + ''.join([char_list_tgt[int(x)] for x in hyp['yseq']]))
                    logging.info('hypo asr: ' + ''.join([char_list_src[int(x)] for x in hyp['yseq_asr']]))
                    logging.info('hypo conv: ' + ''.join([char_list_wrsrc[int(x)] for x in hyp['yseq_conv']]))
                logging.info('best hypo: ' + ''.join([char_list_tgt[int(x)] for x in hyps[0]['yseq']]))
                logging.info('best hypo asr: ' + ''.join([char_list_src[int(x)] for x in hyps[0]['yseq_asr']]))
                logging.info('best hypo conv: ' + ''.join([char_list_wrsrc[int(x)] for x in hyps[0]['yseq_conv']]))
            logging.info('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), trans_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            trans_args.minlenratio_asr = max(0.0, trans_args.minlenratio_asr - 0.1)
            trans_args.minlenratio_conv = max(0.0, trans_args.minlenratio_conv - 0.1)
            return self.recognize_convert_and_translate_sum(x, trans_args, char_list_tgt, char_list_src, char_list_wrsrc, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability st: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        logging.info('normalized log probability asr: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq_asr'])))
        logging.info('normalized log probability conv: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq_conv'])))

        return nbest_hyps

    def recognize_convert_and_translate_half_joint(self, x, trans_args, 
                                        char_list_tgt=None,
                                        char_list_src=None,
                                        char_list_wrsrc=None,
                                        rnnlm=None,
                                        use_jit=False):
        raise NotImplementedError

    def recognize_convert_and_translate_st(self, x, trans_args, 
                                    char_list_tgt=None,
                                    char_list_src=None,
                                    char_list_wrsrc=None,
                                    rnnlm=None,
                                    use_jit=False, 
                                    decode_asr_weight=1.0, 
                                    decode_conv_weight=1.0, 
                                    score_is_prob=False, 
                                    ratio_diverse_st=0.0,
                                    ratio_diverse_asr=0.0,
                                    ratio_diverse_conv=0.0,
                                    use_rev_triu_width=0,
                                    use_diag=False,
                                    debug=False):
        """Recognize and translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """

        assert self.do_asr and self.do_conv, "Recognize, convert and translate are performed simultaneously."
        assert 0 <= ratio_diverse_asr < 1
        assert 0 <= ratio_diverse_conv < 1
        assert 0 <= ratio_diverse_st < 1

        if self.use_lid and self.lang_tok == 'decoder-pre':
            tgt_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[-1])
            y = char_list_tgt.index(tgt_lang_id)
            logging.info(f'tgt_lang_id: {tgt_lang_id} - y: {y}')

            src_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[0])
            y_asr = char_list_src.index(src_lang_id)
            logging.info(f'src_lang_id: {src_lang_id} - y_asr: {y_asr}')

            wrsrc_lang_id = '<2{}>'.format(trans_args.config.split('.')[-2].split('-')[0])
            y_conv = char_list_wrsrc.index(wrsrc_lang_id)
            logging.info(f'wrsrc_lang_id: {wrsrc_lang_id} - y_conv: {y_conv}')
        else:
            y = self.sos_tgt
            y_asr = self.sos_src
            y_conv = self.sos_wrsrc
        
        logging.info(f'<sos> index: {str(y)}; <sos> mark: {char_list_tgt[y]}')
        logging.info(f'<sos> index asr: {str(y_asr)}; <sos> mark asr: {char_list_src[y_asr]}')
        logging.info(f'<sos> index conv: {str(y_conv)}; <sos> mark conv: {char_list_wrsrc[y_conv]}')

        enc_output = self.encode(x).unsqueeze(0)
        h = enc_output.squeeze(0)
        logging.info('input lengths: ' + str(h.size(0)))

        # search params
        beam = trans_args.beam_size
        penalty = trans_args.penalty
        cr = math.ceil((1 - ratio_diverse_asr) * beam)
        cc = math.ceil((1 - ratio_diverse_conv) * beam)
        ct = math.ceil((1 - ratio_diverse_st) * beam)

        vy = h.new_zeros(1).long()

        if trans_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            maxlen = max(1, int(trans_args.maxlenratio * h.size(0)))
        if trans_args.maxlenratio_asr == 0:
            maxlen_asr = h.shape[0]
        else:
            maxlen_asr = max(1, int(trans_args.maxlenratio_asr * h.size(0)))
        if trans_args.maxlenratio_conv == 0:
            maxlen_conv = h.shape[0]
        else:
            maxlen_conv = max(1, int(trans_args.maxlenratio_conv * h.size(0)))
        minlen = int(trans_args.minlenratio * h.size(0))
        minlen_asr = int(trans_args.minlenratio_asr * h.size(0))
        minlen_conv = int(trans_args.minlenratio_conv * h.size(0))
        logging.info(f'max output length: {str(maxlen)}; min output length: {str(minlen)}')
        logging.info(f'max output length asr: {str(maxlen_asr)}; min output length asr: {str(minlen_asr)}')
        logging.info(f'max output length conv: {str(maxlen_conv)}; min output length conv: {str(minlen_conv)}')

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            logging.info('initializing hypothesis...')
            hyp = {'score': 0.0, 'yseq': [y], 'yseq_asr': [y_asr], 'yseq_conv': [y_conv]}

        hyps = [hyp]
        ended_hyps = []

        traced_decoder = None
        for i in six.moves.range(max(maxlen, maxlen_asr, maxlen_conv)):
            logging.info('position ' + str(i))

            hyps_best_kept = []

            for idx, hyp in enumerate(hyps):
                ys_mask = subsequent_mask(len(hyp['yseq'])).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)

                ys_mask_asr = subsequent_mask(len(hyp['yseq_asr'])).unsqueeze(0)
                ys_asr = torch.tensor(hyp['yseq_asr']).unsqueeze(0)

                ys_mask_conv = subsequent_mask(len(hyp['yseq_conv'])).unsqueeze(0)
                ys_conv = torch.tensor(hyp['yseq_conv']).unsqueeze(0)

                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    if hyp['yseq'][-1] != self.eos_tgt or hyp['yseq_asr'][-1] != self.eos_src or hyp['yseq_conv'][-1] != self.eos_wrsrc or i < 2:
                        cross_mask_asr2st = create_cross_mask(ys, ys_asr, self.ignore_id, wait_k_cross=self.wait_k_asr+self.wait_k_conv)
                        cross_mask_st2asr = create_cross_mask(ys_asr, ys, self.ignore_id, wait_k_cross=-self.wait_k_asr-self.wait_k_conv)
                        cross_mask_conv2st = create_cross_mask(ys, ys_conv, self.ignore_id, wait_k_cross=self.wait_k_conv)
                        cross_mask_st2conv = create_cross_mask(ys_conv, ys, self.ignore_id, wait_k_cross=-self.wait_k_conv)
                        cross_mask_asr2conv = create_cross_mask(ys_conv, ys_asr, self.ignore_id, wait_k_cross=self.wait_k_asr)
                        cross_mask_conv2asr = create_cross_mask(ys_asr, ys_conv, self.ignore_id, wait_k_cross=-self.wait_k_asr)
                        local_att_scores, _, local_att_scores_asr, _, local_att_scores_conv, _ = self.triple_decoder.forward_one_step(ys, ys_mask, ys_asr, ys_mask_asr, ys_conv, ys_mask_conv, enc_output,
                                                                                                          cross_mask_asr2st=cross_mask_asr2st, cross_mask_st2asr=cross_mask_st2asr,
                                                                                                          cross_mask_conv2st=cross_mask_conv2st, cross_mask_st2conv=cross_mask_st2conv,
                                                                                                          cross_mask_asr2conv=cross_mask_asr2conv, cross_mask_conv2asr=cross_mask_conv2asr,
                                                                                                          cross_self=self.cross_self, cross_src=self.cross_src,
                                                                                                          cross_self_from=self.cross_self_from,
                                                                                                          cross_src_from=self.cross_src_from)
                        # If using 2 separate dictionaries
                        if not self.use_joint_dict:
                            local_att_scores_conv = self.output_layer_conv(local_att_scores_conv)
                            local_att_scores_asr = self.output_layer_asr(local_att_scores_asr)
                            local_att_scores = self.output_layer(local_att_scores)

                    if (hyp['yseq'][-1] == self.eos_tgt and i > 0) or i < self.wait_k_asr + self.wait_k_conv:
                        local_att_scores = None
                    if (hyp['yseq_asr'][-1] == self.eos_src and i > 0):
                        local_att_scores_asr = None
                    if (hyp['yseq_conv'][-1] == self.eos_wrsrc and i > 0) or i < self.wait_k_asr:
                        local_att_scores_conv = None

                if local_att_scores is not None and local_att_scores_asr is not None and local_att_scores_conv is not None:
                    local_best_scores, local_best_ids_st = torch.topk(local_att_scores, beam, dim=1)
                    _, local_best_ids_asr = torch.topk(local_att_scores_asr, 1, dim=1)
                    _, local_best_ids_conv = torch.topk(local_att_scores_conv, 1, dim=1)
                    local_best_scores = local_best_scores.squeeze(0)
                    local_best_ids_st = local_best_ids_st.squeeze(0)
                    local_best_ids_asr = local_best_ids_asr.squeeze(0)
                    local_best_ids_asr = torch.cat([local_best_ids_asr] * beam)
                    local_best_ids_conv = local_best_ids_conv.squeeze(0)
                    local_best_ids_conv = torch.cat([local_best_ids_conv] * beam)
                elif local_att_scores is not None and local_att_scores_asr is not None:
                    local_best_scores, local_best_ids_st = torch.topk(local_att_scores, beam, dim=1)
                    _, local_best_ids_asr = torch.topk(local_att_scores_asr, 1, dim=1)
                    local_best_scores = local_best_scores.squeeze(0)
                    local_best_ids_st = local_best_ids_st.squeeze(0)
                    local_best_ids_asr = local_best_ids_asr.squeeze(0)
                    local_best_ids_asr = torch.cat([local_best_ids_asr] * beam)
                elif local_att_scores is not None and local_att_scores_conv is not None:
                    local_best_scores, local_best_ids_st = torch.topk(local_att_scores, beam, dim=1)
                    _, local_best_ids_conv = torch.topk(local_att_scores_conv, 1, dim=1)
                    local_best_scores = local_best_scores.squeeze(0)
                    local_best_ids_st = local_best_ids_st.squeeze(0)
                    local_best_ids_conv = local_best_ids_conv.squeeze(0)
                    local_best_ids_conv = torch.cat([local_best_ids_conv] * beam)
                elif local_att_scores_conv is not None and local_att_scores_asr is not None:
                    local_best_scores, local_best_ids_conv = torch.topk(local_att_scores_conv, 1, dim=1)
                    _, local_best_ids_asr = torch.topk(local_att_scores_asr, 1, dim=1)
                    local_best_scores = local_best_scores.squeeze(0)
                    local_best_scores = torch.cat([local_best_scores] * beam)
                    local_best_ids_conv = local_best_ids_conv.squeeze(0)
                    local_best_ids_conv = torch.cat([local_best_ids_conv] * beam)
                    local_best_ids_asr = local_best_ids_asr.squeeze(0)
                    local_best_ids_asr = torch.cat([local_best_ids_asr] * beam)
                elif local_att_scores is not None:
                    local_best_scores, local_best_ids_st = torch.topk(local_att_scores, beam, dim=1)
                    local_best_scores = local_best_scores.squeeze(0)
                    local_best_ids_st = local_best_ids_st.squeeze(0)
                elif local_att_scores_asr is not None:
                    local_best_scores, local_best_ids_asr = torch.topk(local_att_scores_asr, 1, dim=1)
                    local_best_scores = local_best_scores.squeeze(0) 
                    local_best_scores = torch.cat([local_best_scores] * beam)
                    local_best_ids_asr = local_best_ids_asr.squeeze(0)
                    local_best_ids_asr = torch.cat([local_best_ids_asr] * beam)
                elif local_att_scores_conv is not None:
                    local_best_scores, local_best_ids_conv = torch.topk(local_att_scores_conv, 1, dim=1)
                    local_best_scores = local_best_scores.squeeze(0) 
                    local_best_scores = torch.cat([local_best_scores] * beam)
                    local_best_ids_conv = local_best_ids_conv.squeeze(0)
                    local_best_ids_conv = torch.cat([local_best_ids_conv] * beam)
                else:
                    raise NotImplementedError

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']

                    new_hyp['yseq_asr'] = [0] * (1 + len(hyp['yseq_asr']))
                    new_hyp['yseq_asr'][:len(hyp['yseq_asr'])] = hyp['yseq_asr']

                    new_hyp['yseq_conv'] = [0] * (1 + len(hyp['yseq_conv']))
                    new_hyp['yseq_conv'][:len(hyp['yseq_conv'])] = hyp['yseq_conv']

                    if local_att_scores is not None:
                        new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids_st[j])
                    else:
                        if i >= self.wait_k_asr + self.wait_k_conv:
                            new_hyp['yseq'][len(hyp['yseq'])] = self.eos_tgt
                        else:
                            new_hyp['yseq'] = hyp['yseq'] # v3
                    
                    if local_att_scores_asr is not None:
                        new_hyp['yseq_asr'][len(hyp['yseq_asr'])] = int(local_best_ids_asr[j])
                    else:
                        new_hyp['yseq_asr'] = hyp['yseq_asr'] # v3

                    if local_att_scores_conv is not None:
                        new_hyp['yseq_conv'][len(hyp['yseq_conv'])] = int(local_best_ids_conv[j])
                    else:
                        if i >= self.wait_k_asr:
                            new_hyp['yseq_conv'][len(hyp['yseq_conv'])] = self.eos_wrsrc
                        else:
                            new_hyp['yseq_conv'] = hyp['yseq_conv'] # v3

                    hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # if ratio_diverse_st==0 and ratio_diverse_asr==0:
            #     hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]
            # else:
            #     hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)
            #     hyps_best_kept_tmp = []
            #     last_tokens_st = dict.fromkeys(set([h['yseq'][-1] for h in hyps_best_kept]), 0)
            #     last_tokens_asr = dict.fromkeys(set([h['yseq_asr'][-1] for h in hyps_best_kept]), 0)
            #     count = 0
            #     for ii, hyps in enumerate(hyps_best_kept):
            #         if last_tokens_asr[hyps['yseq_asr'][-1]] < cr and last_tokens_st[hyps['yseq'][-1]] < ct:
            #             hyps_best_kept_tmp.append(hyps)
            #             last_tokens_st[hyps['yseq'][-1]] += 1
            #             last_tokens_asr[hyps['yseq_asr'][-1]] += 1
            #             count += 1
            #             if count == beam:
            #                 break
            #     hyps_best_kept = hyps_best_kept_tmp

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    if hyp['yseq'][-1] != self.eos_tgt:
                        hyp['yseq'].append(self.eos_tgt)
            if i == maxlen_asr - 1:
                logging.info('adding <eos> in the last postion in the loop for asr')
                for hyp in hyps:
                    if hyp['yseq_asr'][-1] != self.eos_src:
                        hyp['yseq_asr'].append(self.eos_src)
            if i == maxlen_conv - 1:
                logging.info('adding <eos> in the last postion in the loop for conversion')
                for hyp in hyps:
                    if hyp['yseq_conv'][-1] != self.eos_wrsrc:
                        hyp['yseq_conv'].append(self.eos_wrsrc)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []

            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos_tgt and hyp['yseq_asr'][-1] == self.eos_src and hyp['yseq_conv'][-1] == self.eos_wrsrc:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen and len(hyp['yseq_asr']) > minlen_asr and len(hyp['yseq_conv']) > minlen_conv:
                        hyp['score'] += (i + 1) * penalty
 
                        # if rnnlm:  # Word LM needs to add final <eos> score
                        #     hyp['score'] += trans_args.lm_weight * rnnlm.final(
                        #         hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection          
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.info('remained hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list_tgt is not None and char_list_src is not None and char_list_wrsrc is not None:
                for hyp in hyps:
                    logging.info('hypo: ' + ''.join([char_list_tgt[int(x)] for x in hyp['yseq']]))
                    logging.info('hypo asr: ' + ''.join([char_list_src[int(x)] for x in hyp['yseq_asr']]))
                    logging.info('hypo conv: ' + ''.join([char_list_wrsrc[int(x)] for x in hyp['yseq_conv']]))
                logging.info('best hypo: ' + ''.join([char_list_tgt[int(x)] for x in hyps[0]['yseq']]))
                logging.info('best hypo asr: ' + ''.join([char_list_src[int(x)] for x in hyps[0]['yseq_asr']]))
                logging.info('best hypo conv: ' + ''.join([char_list_wrsrc[int(x)] for x in hyps[0]['yseq_conv']]))
            logging.info('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), trans_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            trans_args.minlenratio_asr = max(0.0, trans_args.minlenratio_asr - 0.1)
            trans_args.minlenratio_conv = max(0.0, trans_args.minlenratio_conv - 0.1)
            return self.recognize_convert_and_translate_sum(x, trans_args, char_list_tgt, char_list_src, char_list_wrsrc, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability st: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        logging.info('normalized log probability asr: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq_asr'])))
        logging.info('normalized log probability conv: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq_conv'])))

        return nbest_hyps



    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src, ys_pad_wrsrc):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src, ys_pad_wrsrc)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention) and m.attn is not None:  # skip MHA for submodules
                ret[name] = m.attn.cpu().numpy()
        return ret


