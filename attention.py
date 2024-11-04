from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation_fn: Callable[[Tensor], Tensor] = F.gelu,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation_fn = activation_fn
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SinCosPosEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.dim = dim

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(dim: int, pos: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(dim // 2).float()
        omega /= dim / 2.0
        omega = 1.0 / 10000 ** omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        return torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)

    def forward(self, h: int, w: int) -> torch.Tensor:
        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, h, w])
        emb_h = self._get_1d_sincos_pos_embed_from_grid(self.dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(self.dim // 2, grid[1])
        pos_embed = torch.concatenate([emb_h, emb_w], dim=1)  # (H*W, D)
        return pos_embed


class GroupedQuerySelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_query_heads: int,
        num_key_value_heads: int,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert (
            num_query_heads % num_key_value_heads == 0
        ), "num_query_heads must be divisible by num_key_value_heads"
        assert dim % num_query_heads == 0, "dim must be divisible by num_query_heads"

        self.num_groups = num_query_heads // num_key_value_heads
        self.num_key_value_heads = num_key_value_heads
        self.scale = num_query_heads ** 0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, (dim // self.num_groups) * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q(x) / self.scale
        k, v = self.kv(x).chunk(2, dim=-1)

        q = rearrange(
            q, "b n (h g c) -> b g h n c", h=self.num_key_value_heads, g=self.num_groups
        )
        k = rearrange(k, "b s (h c) -> b h s c", h=self.num_key_value_heads)
        v = rearrange(v, "b s (h c) -> b h s c", h=self.num_key_value_heads)

        similarity = einsum(q, k, "b g h n c, b h s c -> b g h n s")
        attn = similarity.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum(attn, v, "b g h n s, b h s c -> b g h n c")
        x = rearrange(x, "b g h n c -> b n (h g c)")
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_query_heads: int,
        num_key_value_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        activation_fn: Callable[[Tensor], Tensor] = F.gelu,
    ):
        super().__init__()
        self.attn = GroupedQuerySelfAttention(
            dim=dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.attn_norm = nn.RMSNorm(dim)

        self.mlp = MLP(
            input_dim=dim,
            hidden_dim=dim * mlp_ratio,
            output_dim=dim,
            activation_fn=activation_fn,
            dropout=mlp_dropout,
        )
        self.mlp_norm = nn.RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x