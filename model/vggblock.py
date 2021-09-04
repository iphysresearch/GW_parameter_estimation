# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Ref: 
# https://github.com/pytorch/fairseq/blob/14c5bd027f04aae9dbb32f1bd7b34591b61af97f/fairseq/modules/vggblock.py#L38
#
# Modified by He Wang. 2021/09/04

from __future__ import absolute_import, division, print_function, unicode_literals

from collections.abc import Iterable
from itertools import repeat

import torch
import torch.nn as nn


def _pair(v, causal):
    if causal:
        if isinstance(v, Iterable):
            assert len(v) == 2, "len(v) != 2"
            return (1, v[1])
        return (1, v)
    else:
        if isinstance(v, Iterable):
            assert len(v) == 2, "len(v) != 2"
            return v
        return tuple(repeat(v, 2))


def infer_conv_output_dim(conv_op, input_dim, sample_inchannel):
    sample_seq_len = 200
    sample_bsz = 10
    x = torch.randn(sample_bsz, sample_inchannel, sample_seq_len, input_dim)
    # N x C x H x W
    # N: sample_bsz, C: sample_inchannel, H: sample_seq_len, W: input_dim
    x = conv_op(x)
    # N x C x H x W
    x = x.transpose(1, 2)
    # N x H x C x W
    bsz, seq = x.size()[:2]
    per_channel_dim = x.size()[3]
    # bsz: N, seq: H, CxW the rest
    return x.contiguous().view(bsz, seq, -1).size(-1), per_channel_dim


class VGGBlock(torch.nn.Module):
    """
    VGG motibated cnn module https://arxiv.org/pdf/1409.1556.pdf
    Args:
        in_channels: (int) number of input channels (typically 1)
        out_channels: (int) number of output channels
        conv_kernel_size: convolution channels
        pooling_kernel_size: the size of the pooling window to take a max over
        num_conv_layers: (int) number of convolution layers
        input_dim: (int) input dimension
        conv_stride: the stride of the convolving kernel.
            Can be a single number or a tuple (sH, sW)  Default: 1
        conv_padding: implicit paddings on both sides of the input for conv.
            Can be a single number or a tuple (padH, padW). Default: None
        pool_padding: implicit paddings on both sides of the input for pool.
            Can be a single number or a tuple (padH, padW). Default: None
        layer_norm: (bool) if layer norm is going to be applied. Default: False
        causal: (bool) if in causal.
            'kernel/padding/stride_size' would be (1, ...) for 1D series.
            Set 'padding' = (0, 'dilation'*('kernel_size'-1)).
            'conv_padding' and 'conv_padding' are not working. Default: False
    Shape:
        Input: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
        Output: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size,
        pooling_kernel_size,
        num_conv_layers,
        input_dim,
        conv_stride=1,
        conv_padding=None,
        pool_padding=None,
        conv_dilation=1,
        pool_dilation=1,
        layer_norm=False,
        causal=False
    ):
        assert (
            input_dim is not None
        ), "Need input_dim for LayerNorm and infer_conv_output_dim"
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = _pair(conv_kernel_size, causal)
        self.pooling_kernel_size = _pair(pooling_kernel_size, causal)
        self.num_conv_layers = num_conv_layers
        self.conv_padding = (0, conv_dilation * (self.conv_kernel_size[-1] - 1)) if causal else (
            tuple(e // 2 for e in self.conv_kernel_size)
            if conv_padding is None
            else _pair(conv_padding, causal)
        )
        self.pool_padding = (0, pool_dilation * (self.pooling_kernel_size[-1] - 1)) if causal else (
            tuple(e // 2 for e in self.pooling_kernel_size)
            if pool_padding is None
            else _pair(pool_padding, causal)
        )
        self.conv_stride = _pair(conv_stride, causal)

        self.layers = nn.ModuleList()
        for layer in range(num_conv_layers):
            conv_op = nn.Conv2d(
                in_channels if layer == 0 else out_channels,
                out_channels,
                self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.conv_padding,
                dilation=conv_dilation,
            )
            self.layers.append(conv_op)
            if layer_norm:
                conv_output_dim, per_channel_dim = infer_conv_output_dim(
                    conv_op, input_dim, in_channels if layer == 0 else out_channels
                )
                self.layers.append(nn.LayerNorm(per_channel_dim))
                input_dim = per_channel_dim
            self.layers.append(nn.ReLU())

        if self.pooling_kernel_size is not None:
            pool_op = nn.MaxPool2d(
                kernel_size=self.pooling_kernel_size,
                padding=self.pool_padding,
                ceil_mode=True,
                dilation=pool_dilation)
            self.layers.append(pool_op)
            self.total_output_dim, self.output_dim = infer_conv_output_dim(
                pool_op, input_dim, out_channels
            )

    def forward(self, x):
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x)
        return x
