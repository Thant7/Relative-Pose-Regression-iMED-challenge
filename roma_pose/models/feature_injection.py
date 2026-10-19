from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class RoMaCorrespondenceFeatureInjection(nn.Module):
    """
    Inject RoMa correspondence information into Reloc3r encoder FEATURES.

    Reloc inputs:
        feat1, feat2 : [B, N, C]       float
        pos1, pos2   : [B, N, 2]       int64, (y, x) patch coordinates

    RoMa inputs:
        warp12       : [B, Hr, Wr, 2]  normalized target coords in image 2
        warp21       : [B, Hr, Wr, 2]  normalized target coords in image 1
        cert12       : [B, 1, Hr, Wr]
        cert21       : [B, 1, Hr, Wr]

    Output:
        feat1_new    : [B, N, C]
        feat2_new    : [B, N, C]

    pos1 / pos2 are NOT modified.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        patch_size: int = 16,
        geom_hidden: int = 128,
        feat_hidden: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.geom_mlp = nn.Sequential(
            nn.Linear(6, geom_hidden),
            nn.GELU(),
            nn.Linear(geom_hidden, feature_dim),
        )
        self.feat_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feat_hidden),
            nn.GELU(),
            nn.Linear(feat_hidden, feature_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _certainty_to_bchw(cert: torch.Tensor) -> torch.Tensor:
        if cert.dim() == 3:
            cert = cert.unsqueeze(1)
        elif cert.dim() == 4 and cert.shape[-1] == 1:
            cert = cert.permute(0, 3, 1, 2)
        if cert.dim() != 4 or cert.shape[1] != 1:
            raise ValueError(f"Expected certainty [B,1,H,W], got {cert.shape}")
        return cert

    @staticmethod
    def _tokens_to_map(feat: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        B, N, C = feat.shape
        if N != patch_h * patch_w:
            raise ValueError(f"N={N}, but patch grid is {patch_h}x{patch_w}={patch_h * patch_w}")
        return feat.transpose(1, 2).contiguous().view(B, C, patch_h, patch_w)

    @staticmethod
    def _map_to_tokens(feat_map: torch.Tensor) -> torch.Tensor:
        return feat_map.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _normalize_patch_positions(pos: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        pos = pos.float()
        y = pos[..., 0]
        x = pos[..., 1]
        x_norm = 2.0 * (x + 0.5) / patch_w - 1.0
        y_norm = 2.0 * (y + 0.5) / patch_h - 1.0
        return torch.stack([x_norm, y_norm], dim=-1)

    def _one_direction(
        self,
        feat_src: torch.Tensor,
        feat_tgt: torch.Tensor,
        pos_src: torch.Tensor,
        warp_src_to_tgt: torch.Tensor,
        certainty: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ):
        B, N, C = feat_src.shape
        warp = warp_src_to_tgt.permute(0, 3, 1, 2)
        warp = F.interpolate(warp, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
        warp_grid = warp.permute(0, 2, 3, 1).contiguous()

        certainty = self._certainty_to_bchw(certainty)
        certainty = F.interpolate(certainty, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
        certainty_tokens = certainty.flatten(2).transpose(1, 2).contiguous().clamp(0.0, 1.0)

        tgt_map = self._tokens_to_map(feat_tgt, patch_h, patch_w)
        matched_tgt_map = F.grid_sample(
            tgt_map,
            warp_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        matched_tgt = self._map_to_tokens(matched_tgt_map)

        src_xy = self._normalize_patch_positions(pos_src, patch_h, patch_w)
        tgt_xy = warp_grid.view(B, N, 2)
        delta_xy = tgt_xy - src_xy
        geom = torch.cat([src_xy, tgt_xy, delta_xy], dim=-1)
        geom_hint = self.geom_mlp(geom)

        feat_pair = torch.cat([feat_src, matched_tgt], dim=-1)
        feat_hint = self.feat_mlp(feat_pair)

        valid = (
            (tgt_xy[..., 0] >= -1.0)
            & (tgt_xy[..., 0] <= 1.0)
            & (tgt_xy[..., 1] >= -1.0)
            & (tgt_xy[..., 1] <= 1.0)
        ).unsqueeze(-1)
        gate = certainty_tokens * valid.float()
        correction = gate * (self.alpha * geom_hint + self.beta * feat_hint)
        feat_new = feat_src + correction

        aux = {
            "matched_tgt": matched_tgt,
            "geom": geom,
            "geom_hint": geom_hint,
            "feat_hint": feat_hint,
            "certainty": certainty_tokens,
            "valid": valid,
            "correction": correction,
        }
        return feat_new, aux

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
        warp12: torch.Tensor,
        cert12: torch.Tensor,
        warp21: torch.Tensor,
        cert21: torch.Tensor,
        img_shape,
    ):
        H = int(img_shape[0])
        W = int(img_shape[1])
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        B, N, C = feat1.shape
        if C != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {C}")
        if N != patch_h * patch_w:
            raise ValueError(f"N={N}, patch grid={patch_h}x{patch_w}")

        feat1_new, aux12 = self._one_direction(feat1, feat2, pos1, warp12, cert12, patch_h, patch_w)
        feat2_new, aux21 = self._one_direction(feat2, feat1, pos2, warp21, cert21, patch_h, patch_w)
        return feat1_new, feat2_new, {"12": aux12, "21": aux21}
