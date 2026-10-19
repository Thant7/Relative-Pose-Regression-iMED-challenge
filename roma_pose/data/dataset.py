from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torchvision.io import read_image
from ..runtime import setup_external_paths
setup_external_paths()
from reloc3r.utils.image import load_images

def load_roma_image_locked(path: Path, image_size: int) -> torch.Tensor:
    img = read_image(os.fspath(path))
    if img.shape[0] > 3:
        img = img[:3]
    return img.float() / 255.0



def roma_cache_path(cache_root: Path, split: str, roma_setting: str, sequence: str, frame_idx: int) -> Path:
    return cache_root / split / f"roma_{roma_setting}" / sequence / f"frame_{int(frame_idx):06d}.pt"



def cached_flow_input_to_warp_cert(flow_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert cached [dx_norm, dy_norm, certainty] to RoMa-style warp/certainty.

    Existing RoMa cache files store:
        flow_input: [3,H,W]
        dx_norm = (x_target_norm - x_source_norm) / 2
        dy_norm = (y_target_norm - y_source_norm) / 2
        certainty = overlap/certainty

    PositionalInjection expects:
        warp: [H,W,2] with normalized target coordinates [x_target_norm,y_target_norm]
        cert: [1,H,W]
    """
    if flow_input.dim() != 3 or flow_input.shape[0] != 3:
        raise ValueError(f"expected cached flow_input [3,H,W], got {tuple(flow_input.shape)}")
    _, h, w = flow_input.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, dtype=flow_input.dtype),
        torch.linspace(-1.0, 1.0, w, dtype=flow_input.dtype),
        indexing="ij",
    )
    warp = torch.stack(
        [
            x + 2.0 * flow_input[0],
            y + 2.0 * flow_input[1],
        ],
        dim=-1,
    ).clamp(-1.0, 1.0)
    cert = flow_input[2:3].clamp(0.0, 1.0)
    return warp, cert



class LockedFirstCurrentRoMaDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Any], image_size: int, roma_cache_root: Path | None = None, split: str = "train", roma_setting: str = "fast") -> None:
        self.samples = samples
        self.image_size = image_size
        self.roma_cache_root = roma_cache_root
        self.split = split
        self.roma_setting = roma_setting

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        paths = [os.fspath(s.e1_l_0), os.fspath(s.e2_l_0), os.fspath(s.e1_l_t), os.fspath(s.e2_l_t)]
        imgs = load_images(paths, size=self.image_size, verbose=False)
        item = {
            "sequence": s.sequence,
            "frame0": s.frame0,
            "frame_t": s.frame_t,
            "first_view1": imgs[0],
            "first_view2": imgs[1],
            "curr_view1": imgs[2],
            "curr_view2": imgs[3],
            "target": torch.from_numpy(s.target),
        }
        if self.roma_cache_root is None:
            item.update(
                {
                    "roma_first1": load_roma_image_locked(s.e1_l_0, self.image_size),
                    "roma_first2": load_roma_image_locked(s.e2_l_0, self.image_size),
                    "roma_curr1": load_roma_image_locked(s.e1_l_t, self.image_size),
                    "roma_curr2": load_roma_image_locked(s.e2_l_t, self.image_size),
                }
            )
        else:
            first_cache = torch.load(roma_cache_path(self.roma_cache_root, self.split, self.roma_setting, s.sequence, s.frame0), map_location="cpu", weights_only=True)
            curr_cache = torch.load(roma_cache_path(self.roma_cache_root, self.split, self.roma_setting, s.sequence, s.frame_t), map_location="cpu", weights_only=True)
            if "warp_ab" in first_cache:
                first_warp12 = first_cache["warp_ab"].float()
                first_cert12 = first_cache["certainty"].float()
                first_warp21 = first_cache["warp_ba"].float()
                first_cert21 = first_cache["certainty_ba"].float()
                curr_warp12 = curr_cache["warp_ab"].float()
                curr_cert12 = curr_cache["certainty"].float()
                curr_warp21 = curr_cache["warp_ba"].float()
                curr_cert21 = curr_cache["certainty_ba"].float()
            elif "flow_input12" in first_cache:
                first_warp12, first_cert12 = cached_flow_input_to_warp_cert(first_cache["flow_input12"].float())
                first_warp21, first_cert21 = cached_flow_input_to_warp_cert(first_cache["flow_input21"].float())
                curr_warp12, curr_cert12 = cached_flow_input_to_warp_cert(curr_cache["flow_input12"].float())
                curr_warp21, curr_cert21 = cached_flow_input_to_warp_cert(curr_cache["flow_input21"].float())
            else:
                raise KeyError(
                    f"Unsupported RoMa cache keys in {roma_cache_path(self.roma_cache_root, self.split, self.roma_setting, s.sequence, s.frame0)}: "
                    f"{sorted(first_cache.keys())}"
                )
            item.update(
                {
                    "roma_first_warp12": first_warp12,
                    "roma_first_cert12": first_cert12,
                    "roma_first_warp21": first_warp21,
                    "roma_first_cert21": first_cert21,
                    
                    "roma_curr_warp12": curr_warp12,
                    "roma_curr_cert12": curr_cert12,
                    "roma_curr_warp21": curr_warp21,
                    "roma_curr_cert21": curr_cert21,
                }
            )
        return item



class MixedSplitRoMaDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[LockedFirstCurrentRoMaDataset]) -> None:
        self.datasets = datasets
        self.offsets: list[int] = []
        total = 0
        for ds in datasets:
            self.offsets.append(total)
            total += len(ds)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self.total
        for ds_idx in range(len(self.datasets) - 1, -1, -1):
            start = self.offsets[ds_idx]
            if idx >= start:
                return self.datasets[ds_idx][idx - start]
        raise IndexError(idx)



def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    def stack_view(key: str) -> dict[str, torch.Tensor]:
        return {
            "img": torch.cat([it[key]["img"] for it in items], dim=0),
            "true_shape": torch.as_tensor(np.concatenate([it[key]["true_shape"] for it in items], axis=0)),
        }

    batch = {
        "sequence": [it["sequence"] for it in items],
        "frame0": torch.tensor([it["frame0"] for it in items], dtype=torch.long),
        "frame_t": torch.tensor([it["frame_t"] for it in items], dtype=torch.long),
        "first_view1": stack_view("first_view1"),
        "first_view2": stack_view("first_view2"),
        "curr_view1": stack_view("curr_view1"),
        "curr_view2": stack_view("curr_view2"),
        "target": torch.stack([it["target"] for it in items], dim=0).float(),
    }
    if "roma_first_warp12" in items[0]:
        batch.update(
            {
                "roma_first_warp12": torch.stack([it["roma_first_warp12"] for it in items], dim=0),
                "roma_first_cert12": torch.stack([it["roma_first_cert12"] for it in items], dim=0),
                "roma_first_warp21": torch.stack([it["roma_first_warp21"] for it in items], dim=0),
                "roma_first_cert21": torch.stack([it["roma_first_cert21"] for it in items], dim=0),
                
                "roma_curr_warp12": torch.stack([it["roma_curr_warp12"] for it in items], dim=0),
                "roma_curr_cert12": torch.stack([it["roma_curr_cert12"] for it in items], dim=0),
                "roma_curr_warp21": torch.stack([it["roma_curr_warp21"] for it in items], dim=0),
                "roma_curr_cert21": torch.stack([it["roma_curr_cert21"] for it in items], dim=0),
            }
        )
    else:
        batch.update(
            {
                "roma_first1": torch.stack([it["roma_first1"] for it in items], dim=0),
                "roma_first2": torch.stack([it["roma_first2"] for it in items], dim=0),
                "roma_curr1": torch.stack([it["roma_curr1"] for it in items], dim=0),
                "roma_curr2": torch.stack([it["roma_curr2"] for it in items], dim=0),
            }
        )
    return batch
