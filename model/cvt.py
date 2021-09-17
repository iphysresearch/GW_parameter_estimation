import torch
from torch import nn
from einops import repeat
from itertools import repeat as repeat_
from einops.layers.torch import Rearrange
from cvt_module import ConvAttention, PreNorm, FeedForward
import numpy as np
from collections.abc import Iterable


def _pair(v):
    if isinstance(v, Iterable):
        assert len(v) == 2, "len(v) != 2"
        return v
    return tuple(repeat_(v, 2))


def infer_output_dim(conv_op, input_shape):
    sample_bsz = 1
    x = torch.randn(sample_bsz, *input_shape)
    # N x C x H x W
    # N: sample_bsz, C: sample_inchannel, H: sample_seq_len, W: input_dim
    x = conv_op(x)
    return x.shape[1:]
    # N x C x H x W


class Transformer(nn.Module):
    def __init__(self, dim, img_size_h, img_size_w, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvAttention(dim, img_size_h, img_size_w, heads=heads, dim_head=dim_head, dropout=dropout,
                        last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CvT(nn.Module):
    def __init__(self, image_size_h, image_size_w, in_channels, num_classes, dim=64,
                 kernels=[7, 3, 3], strides=[4, 2, 2],
                 heads=[1, 3, 6], depth=[1, 2, 10], pool='cls', dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim

        # #### Stage 1 #######
        conv1 = nn.Conv2d(in_channels, dim, _pair(kernels[0]), _pair(strides[0]), 2)
        _, h, w = infer_output_dim(conv1, [in_channels, image_size_h, image_size_w])
        self.stage1_conv_embed = nn.Sequential(
            conv1,
            Rearrange('b c h w -> b (h w) c', h=h, w=w),
            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            Transformer(dim=dim, img_size_h=h, img_size_w=w, depth=depth[0], heads=heads[0], dim_head=self.dim,
                        mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h=h, w=w),
        )

        # #### Stage 2 #######
        in_channels = dim
        scale = heads[1]//heads[0]
        dim = scale*dim
        conv2 = nn.Conv2d(in_channels, dim, _pair(kernels[1]), _pair(strides[1]), 1)
        _, h, w = infer_output_dim(conv2, [in_channels, h, w])
        self.stage2_conv_embed = nn.Sequential(
            conv2,
            Rearrange('b c h w -> b (h w) c', h=h, w=w),
            nn.LayerNorm(dim)
        )
        self.stage2_transformer = nn.Sequential(
            Transformer(dim=dim, img_size_h=h, img_size_w=w, depth=depth[1], heads=heads[1], dim_head=self.dim,
                        mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h=h, w=w)
        )

        # #### Stage 3 #######
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim
        conv3 = nn.Conv2d(in_channels, dim, _pair(kernels[2]), _pair(strides[2]), 1)
        _, h, w = infer_output_dim(conv3, [in_channels, h, w])
        self.stage3_conv_embed = nn.Sequential(
            conv3,
            Rearrange('b c h w -> b (h w) c', h=h, w=w),
            nn.LayerNorm(dim)
        )
        self.stage3_transformer = nn.Sequential(
            Transformer(dim=dim, img_size_h=h, img_size_w=w, depth=depth[2], heads=heads[2], dim_head=self.dim,
                        mlp_dim=dim * scale_dim, dropout=dropout, last_stage=True),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        xs = self.stage1_conv_embed(img)
        xs = self.stage1_transformer(xs)

        xs = self.stage2_conv_embed(xs)
        xs = self.stage2_transformer(xs)

        xs = self.stage3_conv_embed(xs)
        b, n, _ = xs.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        xs = torch.cat((cls_tokens, xs), dim=1)
        xs = self.stage3_transformer(xs)
        return xs
        # xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]

        # xs = self.mlp_head(xs)
        # return xs


if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    model = CvT(224, 3, 1000)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum(np.prod(p.size()) for p in parameters) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model(img)

    print("Shape of out :", out.shape)  # [B, num_classes]
