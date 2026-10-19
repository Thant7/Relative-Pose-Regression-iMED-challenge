from __future__ import annotations

import math
import torch
import torch.nn.functional as F

def matrix_to_pose_components(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return T[:, :3, :3], T[:, :3, 3]



def move_view(view: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in view.items()}



def se3_inverse_torch(T: torch.Tensor) -> torch.Tensor:
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    out = torch.eye(4, device=T.device, dtype=T.dtype).repeat(T.shape[0], 1, 1)
    Rt = R.transpose(1, 2)
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -(Rt @ t[..., None]).squeeze(-1)
    return out



def challenge_normalize(raw_first_challenge: torch.Tensor, raw_current_challenge: torch.Tensor) -> torch.Tensor:
    return se3_inverse_torch(raw_first_challenge) @ raw_current_challenge



def rotation_geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rel = R_pred.transpose(1, 2) @ R_gt
    tr = torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1)
    cos = torch.clamp((tr - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos).mean()



def pose_loss(pred: torch.Tensor, target: torch.Tensor, lambda_r: float, lambda_t: float) -> tuple[torch.Tensor, dict[str, float]]:
    R_pred, t_pred = matrix_to_pose_components(pred)
    R_gt, t_gt = matrix_to_pose_components(target)
    rot = rotation_geodesic_loss(R_pred, R_gt)
    trans = F.smooth_l1_loss(t_pred, t_gt)
    loss = lambda_r * rot + lambda_t * trans
    return loss, {
        "loss": float(loss.detach().cpu()),
        "rot_rad": float(rot.detach().cpu()),
        "rot_deg": float((rot * 180.0 / math.pi).detach().cpu()),
        "trans_smooth_l1": float(trans.detach().cpu()),
    }
