from __future__ import annotations

import torch
import torch.nn as nn
from ..runtime import setup_external_paths
setup_external_paths()
try:
    from romav2 import RoMaV2
except ModuleNotFoundError:
    RoMaV2 = None
from reloc3r.reloc3r_relpose import Reloc3rRelpose
from .feature_injection import RoMaCorrespondenceFeatureInjection
from .pose_head import WrappedPoseHead
from ..utils.modules import set_requires_grad

class LockedDualReloc3rRoMaV2Fusion(nn.Module):
    def __init__(
        self,
        pretrained: str = "siyan824/reloc3r-512",
        image_size: int = 512,
        roma_setting: str = "fast",
        use_roma_model: bool = True,
        conf_threshold: float = 0.5,
        freeze_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.roma_setting = roma_setting
        self.use_roma_model = use_roma_model
        
        if pretrained in {"", "local", "none"}:
            self.reloc = Reloc3rRelpose(img_size=image_size)
        else:
            self.reloc = Reloc3rRelpose.from_pretrained(pretrained)
            
        self.feature_injector = RoMaCorrespondenceFeatureInjection(
            feature_dim=self.reloc.enc_embed_dim,
            patch_size=self.reloc.pose_head.patch_size,
        )
        self.wrapped_pose_head = WrappedPoseHead(self.reloc.pose_head)
        
        if use_roma_model and RoMaV2 is None:
            raise ModuleNotFoundError(
                "RoMaV2 is required only for online matching. Install it, or pass "
                "--roma-cache-root to train from precomputed RoMa tensors."
            )
        self.roma = RoMaV2(RoMaV2.Cfg(compile=False)).eval() if use_roma_model else nn.Module()
        if use_roma_model:
            self.roma.apply_setting(roma_setting)

        set_requires_grad(self.reloc, False)
        set_requires_grad(self.roma, False)
        set_requires_grad(self.reloc.decoder_embed, not freeze_decoder)
        set_requires_grad(self.reloc.dec_blocks, not freeze_decoder)
        set_requires_grad(self.reloc.dec_norm, not freeze_decoder)
        set_requires_grad(self.feature_injector, True)
        set_requires_grad(self.wrapped_pose_head, True)

    def roma_flow_input(self, img1: torch.Tensor, img2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            preds = self.roma.match(img1, img2)
            warp_ab = preds["warp_AB"].detach()
            certainty = preds["overlap_AB"].detach()
            if certainty.dim() == 3:
                certainty = certainty.unsqueeze(1)
            elif certainty.dim() == 4 and certainty.shape[-1] == 1:
                certainty = certainty.permute(0, 3, 1, 2)
        return warp_ab, certainty

    def forward(
        self,
        view1: dict[str, torch.Tensor],
        view2: dict[str, torch.Tensor],
        roma_img1: torch.Tensor,
        roma_img2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.use_roma_model:
            raise RuntimeError("Use forward_precomputed() with cached RoMa tensors.")
            
        W12, C12 = self.roma_flow_input(roma_img1, roma_img2)
        W21, C21 = self.roma_flow_input(roma_img2, roma_img1)
        return self.forward_decoded(view1, view2, W12, C12, W21, C21)

    def forward_precomputed(
        self,
        view1: dict[str, torch.Tensor],
        view2: dict[str, torch.Tensor],
        W12: torch.Tensor,
        C12: torch.Tensor,
        W21: torch.Tensor,
        C21: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.forward_decoded(view1, view2, W12, C12, W21, C21)

    def forward_decoded(
        self,
        view1: dict[str, torch.Tensor],
        view2: dict[str, torch.Tensor],
        W12: torch.Tensor,
        C12: torch.Tensor,
        W21: torch.Tensor,
        C21: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.reloc._encoder(view1, view2)
            
        feat1_updated, feat2_updated, inject_aux = self.feature_injector(
            feat1.detach(),
            feat2.detach(),
            pos1,
            pos2,
            W12,
            C12,
            W21,
            C21,
            shape1[0],
        )
        dec1, dec2 = self.reloc._decoder(feat1_updated, pos1, feat2_updated, pos2)
        
        out12 = self.wrapped_pose_head(dec1, shape1[0])
        out21 = self.wrapped_pose_head(dec2, shape2[0])
        
        return {
            "pose": out12["pose"],
            "out_t": out12["out_t"],
            "out_r": out12["out_r"],
            "P12": out12["pose"],
            "P21": out21["pose"],
            "roma_warp12": W12,
            "roma_warp21": W21,
            "roma_certainty12": C12,
            "roma_certainty21": C21,
            "inject_aux": inject_aux,
            "dec1_last": dec1[-1],
            "dec2_last": dec2[-1],
        }
