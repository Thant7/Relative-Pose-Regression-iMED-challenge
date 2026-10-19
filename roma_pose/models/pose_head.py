from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.modules import reset_module_parameters

class WrappedPoseHead(nn.Module):
    """
    Standard Reloc3r Pose Head wrapper without late fusion concat.
    """
    def __init__(self, base_head: nn.Module) -> None:
        super().__init__()
        self.patch_size = base_head.patch_size
        self.proj = copy.deepcopy(base_head.proj)
        self.res_conv = copy.deepcopy(base_head.res_conv)
        self.avgpool = copy.deepcopy(base_head.avgpool)
        self.decoder_dim = base_head.proj.in_features
        self.feature_dim = base_head.fc_t.in_features
        self.rot_representation = base_head.rot_representation
        
        self.more_mlps = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(self.feature_dim, 3)
        self.fc_rot = nn.Linear(self.feature_dim, 9 if self.rot_representation == "9D" else 6)
        
        reset_module_parameters(self.more_mlps)
        reset_module_parameters(self.fc_t)
        reset_module_parameters(self.fc_rot)

    def forward(self, decout: list[torch.Tensor], img_shape: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = img_shape
        H, W = int(H), int(W)
        
        with torch.cuda.amp.autocast(enabled=False):
            tokens = decout[-1].float()
            B, S, D = tokens.shape
            
            feat = self.proj(tokens)
            reloc_spatial = feat.transpose(-1, -2).contiguous().view(B, self.feature_dim, H // self.patch_size, W // self.patch_size)
            
            for block in self.res_conv:
                reloc_spatial = block(reloc_spatial)

            pooled = self.avgpool(reloc_spatial).view(B, -1)
            head_feat = self.more_mlps(pooled)
            out_t = self.fc_t(head_feat)
            out_r = self.fc_rot(head_feat)
            pose = self.convert_pose_to_4x4(B, out_r, out_t, tokens.device)
            
        return {
            "pose": pose,
            "out_t": out_t,
            "out_r": out_r,
            "head_feat": head_feat,
        }

    def convert_pose_to_4x4(self, B: int, out_r: torch.Tensor, out_t: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.rot_representation == "9D":
            out_r = self.svd_orthogonalize(out_r)
        else:
            out_r = self.rotation_6d_to_matrix(out_r)
        pose = torch.zeros((B, 4, 4), device=device, dtype=out_t.dtype)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.0
        return pose

    def svd_orthogonalize(self, m: torch.Tensor) -> torch.Tensor:
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, _, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        return torch.matmul(torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2), u.transpose(-2, -1))

    def rotation_6d_to_matrix(self, d6: torch.Tensor) -> torch.Tensor:
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
