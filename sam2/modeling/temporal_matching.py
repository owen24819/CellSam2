# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Temporal matching head and components for cell tracking across frames."""

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from hydra.utils import instantiate

from sam2.modeling.sam2_utils import MLP


def _bin_init_exp(cutoff: float, n_bins: int) -> torch.Tensor:
    """Exponentially spaced bin edges from 0 to cutoff (for distance bucketing)."""
    t = torch.linspace(0, 1, n_bins + 1, dtype=torch.float32)
    bins = cutoff * (t ** 2)
    return bins


class FourierPositionEncoding(torch.nn.Module):
    """Maps 2-D normalised cell centroids to a ``hidden_dim`` embedding using
    multi-scale Fourier (sinusoidal) features.

    For each coordinate axis (cx, cy) and each of the ``num_freqs`` frequency
    bands (geometric progression 1, 2, 4, …, 2^(F-1)):

        angle  = 2π · coord · freq
        feature = [sin(angle), cos(angle)]

    This yields a 4·F-dimensional descriptor per cell (2 axes × 2 functions ×
    F bands) which is then projected linearly to ``hidden_dim``.
    """

    def __init__(self, hidden_dim: int = 256, num_freqs: int = 16):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)
        self.proj = torch.nn.Linear(4 * num_freqs, hidden_dim)
        torch.nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            centroids: [N, 2]  normalised (cx, cy) in [0, 1]

        Returns:
            pe: [N, hidden_dim]
        """
        angles = centroids * (2.0 * math.pi)
        angles = angles.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        enc = torch.cat([angles.sin(), angles.cos()], dim=-1)
        enc = enc.flatten(1)
        return self.proj(enc)


class RelativeSpatialBias(torch.nn.Module):
    """Relative positional bias for attention: add to attention scores from pairwise centroid distances."""

    def __init__(
        self,
        mode: str = "distance",
        sigma: float = 0.2,
        n_bins: int = 16,
        cutoff: float = 1.5,
    ):
        super().__init__()
        self.mode = mode
        self.sigma = sigma
        self.n_bins = n_bins
        self.cutoff = cutoff
        if mode == "distance":
            self.scale = torch.nn.Parameter(torch.tensor(0.1))
        elif mode == "bins":
            bins = _bin_init_exp(cutoff, n_bins)
            self.register_buffer("bins", bins)
            self.bias_table = torch.nn.Parameter(torch.zeros(n_bins))
        elif mode != "none":
            raise ValueError(f"RelativeSpatialBias mode must be 'distance', 'bins', or 'none', got {mode!r}")

    def forward(
        self,
        query_centroids: torch.Tensor,
        key_centroids: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros(
                query_centroids.shape[0],
                key_centroids.shape[0],
                device=query_centroids.device,
                dtype=query_centroids.dtype,
            )
        dist = torch.cdist(query_centroids.float(), key_centroids.float(), p=2)
        if self.mode == "distance":
            bias = self.scale * torch.exp(-(dist ** 2) / (2 * self.sigma ** 2))
            return bias.to(query_centroids.dtype)
        idx = torch.bucketize(dist.contiguous().view(-1), self.bins.to(dist.device))
        idx = idx.clamp(max=self.n_bins - 1)
        bias = self.bias_table[idx].view(dist.shape).to(query_centroids.dtype)
        return bias


class SACABlock(torch.nn.Module):
    """One interleaved Self-Attention → Cross-Attention → FFN block (standard transformer)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_ratio: int = 4):
        super().__init__()
        self.sa_norm = torch.nn.LayerNorm(hidden_dim)
        self.sa = torch.nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.ca_norm = torch.nn.LayerNorm(hidden_dim)
        self.ca = torch.nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.ffn_norm = torch.nn.LayerNorm(hidden_dim)
        self.ffn = MLP(
            hidden_dim,
            hidden_dim * ffn_ratio,
            hidden_dim,
            num_layers=2,
            activation=torch.nn.GELU,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_mask_sa: Optional[torch.Tensor] = None,
        attn_mask_ca: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_n = self.sa_norm(q)
        out, _ = self.sa(q_n, q_n, q_n, attn_mask=attn_mask_sa)
        q = q + out
        q_n = self.ca_norm(q)
        out, _ = self.ca(q_n, k, k, attn_mask=attn_mask_ca)
        q = q + out
        q_n = self.ffn_norm(q)
        q = q + self.ffn(q_n)
        return q


class KeyConstellationEncoder(torch.nn.Module):
    """Pre-norm self-attention + FFN stack on key tokens to build a spatial constellation map."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int = 1,
        ffn_ratio: int = 4,
    ):
        super().__init__()
        self.sa_layers = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.sa_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        self.ffn_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        self.ffns = torch.nn.ModuleList([
            MLP(hidden_dim, hidden_dim * ffn_ratio, hidden_dim, num_layers=2, activation=torch.nn.GELU)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        key_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key_tokens.shape[0] == 0:
            return key_tokens
        x = key_tokens.unsqueeze(0)
        for sa, sa_norm, ffn_norm, ffn in zip(
            self.sa_layers, self.sa_norms, self.ffn_norms, self.ffns
        ):
            x_n = sa_norm(x)
            out, _ = sa(x_n, x_n, x_n, attn_mask=attn_mask)
            x = x + out
            x_n = ffn_norm(x)
            x = x + ffn(x_n)
        return x.squeeze(0)


class TemporalMatchingHead(torch.nn.Module):
    """SuperGlue-inspired cell matcher for CellSAM2."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        use_aux_loss: bool,
        pos_enc: Any,
        pos_bias: Any,
        key_encoder: Any,
        blocks: Any,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.obj_ptr_norm = torch.nn.LayerNorm(hidden_dim)
        self.roi_feat_norm = torch.nn.LayerNorm(hidden_dim)
        self.token_proj = MLP(
            hidden_dim * 2, hidden_dim, hidden_dim, 2,
            activation=torch.nn.GELU,
        )

        self.pos_enc = instantiate(pos_enc, hidden_dim=hidden_dim, _convert_="all")
        self.key_encoder = instantiate(
            key_encoder, hidden_dim=hidden_dim, num_heads=num_heads, _convert_="all"
        )

        get = blocks.get if hasattr(blocks, "get") else lambda k, d=None: getattr(blocks, k, d)
        num_blocks = get("num_blocks", 1)
        block_cfg = get("block")
        self.blocks = torch.nn.ModuleList([
            instantiate(block_cfg, hidden_dim=hidden_dim, num_heads=num_heads, _convert_="all")
            for _ in range(num_blocks)
        ])
        self.use_aux_loss = use_aux_loss

        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        self.null_key = torch.nn.Parameter(torch.empty(1, hidden_dim))
        torch.nn.init.trunc_normal_(self.null_key, std=0.02)

        self.pos_bias = instantiate(pos_bias, _convert_="all")

    def get_centroids_from_mask_logits(self, mask_logits: torch.Tensor) -> torch.Tensor:
        device = mask_logits.device
        _, _, H_mask, W_mask = mask_logits.shape
        mask_prob = mask_logits.float().sigmoid()
        mask_bin = (mask_prob.squeeze(1) > 0.5)
        N = mask_bin.shape[0]
        H_norm = max(H_mask - 1, 1)
        W_norm = max(W_mask - 1, 1)
        centroids_list = []
        for i in range(N):
            ys, xs = torch.where(mask_bin[i])
            if len(xs) == 0:
                centroids_list.append(torch.zeros(2, device=device, dtype=torch.float32))
                continue
            cx = xs.float().mean()
            cy = ys.float().mean()
            cy_int = int(round(cy.item()))
            cx_int = int(round(cx.item()))
            if not mask_bin[i, cy_int, cx_int]:
                mid = len(xs) // 2
                cx = xs[mid].float()
                cy = ys[mid].float()
            centroids_list.append(torch.stack([cx, cy]))
        centroids_px = torch.stack(centroids_list, dim=0)
        return torch.stack(
            [centroids_px[:, 0] / W_norm, centroids_px[:, 1] / H_norm], dim=1
        )

    def build_matching_tokens(
        self,
        obj_ptr: torch.Tensor,
        pix_feat: torch.Tensor,
        mask_logits: torch.Tensor,
        centroids: Optional[torch.Tensor] = None,
    ):
        _, C, H, W = pix_feat.shape
        device = pix_feat.device

        mask_prob = F.interpolate(
            mask_logits.float().sigmoid(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        mask_sum = mask_prob.sum(dim=(2, 3)).clamp(min=1e-6)
        roi_feat = (pix_feat * mask_prob).sum(dim=(2, 3)) / mask_sum

        obj_ptr_detach = obj_ptr.detach()
        roi_feat_detach = roi_feat.detach()

        if centroids is None:
            centroids = self.get_centroids_from_mask_logits(mask_logits)
        else:
            centroids = centroids.to(device=device)

        token = self.token_proj(
            torch.cat(
                [self.obj_ptr_norm(obj_ptr_detach), self.roi_feat_norm(roi_feat_detach)],
                dim=1,
            )
        )
        return token, centroids

    def _encode_keys(
        self,
        key_tokens: torch.Tensor,
        key_centroids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask_kk = None
        if key_centroids is not None and key_centroids.shape[0] == key_tokens.shape[0]:
            attn_mask_kk = self.pos_bias(key_centroids, key_centroids)
        return self.key_encoder(key_tokens, attn_mask=attn_mask_kk)

    def _match_logits_from_q(
        self,
        q: torch.Tensor,
        key_tokens: torch.Tensor,
        N_k: int,
        query_centroids: torch.Tensor,
        key_centroids: torch.Tensor,
    ) -> torch.Tensor:
        N_q = q.shape[0]
        keys_with_null = (
            torch.cat([key_tokens, self.null_key.expand(1, -1)], dim=0)
            if N_k > 0
            else self.null_key
        )
        H, D = self.num_heads, self.head_dim
        Q = self.q_proj(q).view(N_q, H, D)
        K = self.k_proj(keys_with_null).view(N_k + 1, H, D)
        attn = torch.einsum("qhd,khd->qkh", Q, K) / (D ** 0.5)
        return attn.mean(dim=-1)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_tokens: torch.Tensor,
        query_centroids: torch.Tensor,
        key_centroids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        N_q = query_tokens.shape[0]
        N_k = key_tokens.shape[0]

        if N_k == 0:
            match_logits = torch.zeros(
                N_q, 1,
                device=query_tokens.device,
                dtype=query_tokens.dtype,
            )
            return match_logits, None

        query_tokens = query_tokens + self.pos_enc(query_centroids)
        key_tokens = key_tokens + self.pos_enc(key_centroids)

        key_tokens = self._encode_keys(key_tokens, key_centroids)

        keys_with_null = torch.cat(
            [key_tokens, self.null_key.expand(1, -1)], dim=0
        )

        bias_qq = self.pos_bias(query_centroids, query_centroids)
        q = query_tokens.unsqueeze(0)
        match_logits_aux = [] if self.use_aux_loss else None

        k = keys_with_null.unsqueeze(0)
        bias_qk = self.pos_bias(query_centroids, key_centroids)
        null_bias = torch.zeros(
            N_q, 1,
            device=query_centroids.device,
            dtype=bias_qk.dtype,
        )
        bias_qk = torch.cat([bias_qk, null_bias], dim=1)
        for i, block in enumerate(self.blocks):
            q = block(q, k, attn_mask_sa=bias_qq, attn_mask_ca=bias_qk)
            if self.use_aux_loss and i < len(self.blocks) - 1:
                match_logits_aux.append(
                    self._match_logits_from_q(
                        q.squeeze(0), key_tokens, N_k,
                        query_centroids, key_centroids,
                    )
                )
        q = q.squeeze(0)
        match_logits = self._match_logits_from_q(
            q, key_tokens, N_k, query_centroids, key_centroids,
        )

        return match_logits, match_logits_aux
