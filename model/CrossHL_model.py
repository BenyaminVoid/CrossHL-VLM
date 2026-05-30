import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear


class HetConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=None,
        p=64,
        g=64,
    ):
        super().__init__()
        self.groupwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=g,
            padding=kernel_size // 3,
            stride=stride,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=p,
            stride=stride,
        )

    def forward(self, x):
        return self.groupwise_conv(x) + self.pointwise_conv(x)


class CrossHL_attention(nn.Module):
    def __init__(
        self,
        dim,
        patches,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.dim = dim
        self.Wq = nn.Linear(patches, dim * num_heads, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(patches + 1, dim, bias=qkv_bias)
        self.linear_projection = nn.Linear(dim * num_heads, dim)
        self.linear_projection_drop = nn.Dropout(proj_drop)

    def forward(self, x, x2):
        batch_size, num_tokens, channels = x.shape
        query = self.Wq(x2).reshape(
            batch_size,
            self.num_heads,
            self.num_heads,
            self.dim // self.num_heads,
        )
        key = self.Wk(x).reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.dim // self.num_heads,
        ).permute(0, 2, 1, 3)
        value = self.Wv(x.transpose(1, 2)).reshape(
            batch_size,
            channels,
            self.num_heads,
            self.dim // self.num_heads,
        ).permute(0, 2, 3, 1)

        attention = torch.einsum("bhid,bhjd->bhij", key, query) * self.scale
        attention = attention.softmax(dim=-1)

        x = torch.einsum("bhij,bhjd->bhid", attention, value)
        x = x.reshape(batch_size, num_tokens, -1)
        x = self.linear_projection(x)
        x = self.linear_projection_drop(x)
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fclayer1 = Linear(dim, mlp_dim)
        self.fclayer2 = Linear(mlp_dim, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fclayer1.weight)
        nn.init.xavier_uniform_(self.fclayer2.weight)
        nn.init.normal_(self.fclayer1.bias, std=1e-6)
        nn.init.normal_(self.fclayer2.bias, std=1e-6)

    def forward(self, x):
        x = self.fclayer1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fclayer2(x)
        x = self.dropout(x)
        return x


class SingleEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, patchsize):
        super().__init__()
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = MultiLayerPerceptron(dim, mlp_dim)
        self.cross_hl_attention = CrossHL_attention(dim=dim, patches=patchsize**2)

    def forward(self, x1, x2):
        residual = x1
        x = self.attention_norm(x1)
        x = self.cross_hl_attention(x, x2)
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x


class Encoder(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=512, depth=2, patchsize=11):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(depth):
            layer = SingleEncoderBlock(dim, num_heads, mlp_dim, patchsize)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x, x2):
        for layer_block in self.layer:
            x = layer_block(x, x2)
        encoded = self.encoder_norm(x)
        return encoded[:, 0]


class CrossHL_Transformer(nn.Module):
    """Cross-HL backbone with an optional CLIP-space semantic projection head.

    The classification path is unchanged: fused CLS features still go to the
    original linear classifier. The VLM head is only used when return_embed=True
    and therefore acts as a training regularizer, not a classifier.
    """

    def __init__(
        self,
        FM,
        NC,
        NCLidar,
        Classes,
        patchsize,
        vlm_dim=512,
        num_heads=8,
        mlp_dim=512,
        depth=2,
    ):
        super().__init__()
        self.patchsize = patchsize
        self.NCLidar = NCLidar
        self.feature_dim = FM * 4
        self.vlm_dim = vlm_dim

        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.hetconv_layer = nn.Sequential(
            HetConv(
                8 * (NC - 8),
                self.feature_dim,
                p=1,
                g=(self.feature_dim) // 4
                if (8 * (NC - 8)) % FM == 0
                else (self.feature_dim) // 8,
            ),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(),
        )

        self.ca = Encoder(
            self.feature_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            depth=depth,
            patchsize=patchsize,
        )
        self.fclayer = nn.Linear(self.feature_dim, Classes)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, (patchsize**2) + 1, self.feature_dim)
        )
        self.dropout = nn.Dropout(0.1)
        self.clsTok = nn.Parameter(torch.zeros(1, 1, self.feature_dim))

        torch.nn.init.xavier_uniform_(self.fclayer.weight)
        torch.nn.init.normal_(self.fclayer.bias, std=1e-6)
        self.vlm_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.vlm_dim),
            nn.LayerNorm(self.vlm_dim),
        )

    def encode_features(self, x1, x2):
        x1 = x1.reshape(x1.shape[0], -1, self.patchsize, self.patchsize)
        x2 = x2.reshape(x1.shape[0], -1, self.patchsize * self.patchsize)
        x1 = x1.unsqueeze(1)

        if x2.shape[1] > 0:
            x2 = (
                F.adaptive_avg_pool1d(x2.flatten(2).transpose(1, 2), 1)
                .transpose(1, 2)
                .reshape(x1.shape[0], -1, self.patchsize * self.patchsize)
            )

        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0], -1, self.patchsize, self.patchsize)
        x1 = self.hetconv_layer(x1)
        x1 = x1.flatten(2).transpose(-1, -2)

        cls_tokens = self.clsTok.expand(x1.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x1), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return self.ca(x, x2)

    def semantic_embedding(self, features):
        return F.normalize(self.vlm_proj(features), dim=-1)

    def freeze_early_layers(self):
        for param in self.conv5.parameters():
            param.requires_grad = False
        for param in self.hetconv_layer.parameters():
            param.requires_grad = False

    def forward(self, x1, x2, return_embed=False, return_features=False):
        features = self.encode_features(x1, x2)
        logits = self.fclayer(features)

        outputs = [logits]
        if return_embed:
            outputs.append(self.semantic_embedding(features))
        if return_features:
            outputs.append(features)

        if len(outputs) == 1:
            return logits
        return tuple(outputs)
