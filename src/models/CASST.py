"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from einops import rearrange
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, out_channel, band=1):
        super().__init__()

        self.conv1 = nn.Conv2d(band, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(128, out_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, num=1):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)

        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.adapt_bn = nn.BatchNorm2d(num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CTMF(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 num_branches=2
                 ):
        super(CTMF, self).__init__()
        self.norm = nn.ModuleList([norm_layer(dim) for i in range(num_branches * 2)])
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.mlp1= Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.attn = Attention(dim, num_heads=num_heads)
        self.attn1 = Attention(dim, num_heads=num_heads)

    def forward(self, x, y):
        x_ = self.drop_path(self.attn(self.norm[1](torch.cat([y[:, 0:1], x[:, 1:]], dim=1))))
        y_ = self.drop_path(self.attn(self.norm[1](torch.cat([x[:, 0:1], y[:, 1:]], dim=1))))

        x = x + torch.cat([(x[:, 0:1] + y_[:, 0:1]), x_[:, 1:]], dim=1)
        y = y + torch.cat([(y[:, 0:1] + x_[:, 0:1]), y_[:, 1:]], dim=1)

        x = x + self.drop_path(self.mlp(self.norm[2](x)))
        y = y + self.drop_path(self.mlp(self.norm[2](y)))

        # x = x + self.drop_path(self.attn(self.norm[0](x)))
        # y = y + self.drop_path(self.attn1(self.norm[1](y)))
        # x = x + self.drop_path(self.mlp(self.norm[2](x)))
        # y = y + self.drop_path(self.mlp1(self.norm[3](y)))

        return x, y


class CASST(nn.Module):
    def __init__(self, params):
        super(CASST, self).__init__()
        num_classes =  params['data'].get('num_classes', 16)
        input_channel= params['data'].get('spectral_size', 30)
        embed_dim=512
        blocks=2
        num_heads=8
        drop_ratio=0.4
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        self.cnn = CNN(out_channel=embed_dim)

        # hsi branch
        self.spa_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spa_pos_embed = nn.Parameter(torch.zeros(1, 122, embed_dim))

        # lidar branch
        self.spe_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spe_pos_embed = nn.Parameter(torch.zeros(1, input_channel+1, embed_dim))

        self.num_blocks = blocks
        self.CTMF = nn.Sequential(*[
            CTMF(dim=embed_dim, num_heads=num_heads)
            for j in range(self.num_blocks)
        ])
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.spe_norm = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.cls_head = nn.Linear(embed_dim * 2, num_classes)
        self.pre_logits = nn.Identity()

        torch.nn.init.xavier_uniform_(self.cls_head.weight)
        torch.nn.init.normal_(self.cls_head.bias, std=1e-6)
        nn.init.normal_(self.spa_pos_embed, std=.02)
        nn.init.normal_(self.spa_cls_token, std=.02)
        nn.init.normal_(self.spe_pos_embed, std=.02)
        nn.init.normal_(self.spe_cls_token, std=.02)

    def forward_spa(self, x):
        x = self.conv_h(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_token = self.spa_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.spa_pos_embed)

        return x

    def forward_spe(self, y):
        cnn_output = []
        for index in range(y.shape[1]):
            each_band_input = torch.unsqueeze(y[:, index, :, :], dim=1)
            # print("each_band_input=", each_band_input.shape)
            each_band_output = torch.unsqueeze(self.cnn(each_band_input, 1), dim=2)
            # print("each_band_output=", each_band_output.shape)
            if index == 0:
                cnn_output = each_band_output
            else:
                cnn_output = torch.cat([cnn_output, each_band_output], dim=2)
        # print('cnn_output=', cnn_output.shape)
        y = cnn_output.permute(0, 2, 1)
        # y = self.spe_norm(y)

        cls_token = self.spe_cls_token.expand(y.shape[0], -1, -1)
        y = torch.cat((cls_token, y), dim=1)
        y = self.pos_drop(y + self.spe_pos_embed)

        return y

    def forward(self, x):
        x_spa = self.forward_spa(x)
        x_spe = self.forward_spe(x)

        for i in range(self.num_blocks):
            x_spa, x_spe = self.CTMF[i](x_spa, x_spe)

        spa_cls = self.norm1(self.pre_logits(x_spa[:, 0]))
        spe_cls = self.norm2(self.pre_logits(x_spe[:, 0]))

        cls = torch.cat((spa_cls, spe_cls), dim=1)
        out = self.cls_head(cls)

        return out, 0, 0


if __name__ == '__main__':
    data = torch.randn([1, 30, 11, 11])
    model = CASST(embed_dim=512, blocks=2, num_heads=8, num_classes=16, input_channel=30)
    out_data = model(data)
    print(out_data.shape)
