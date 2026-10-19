"""Predict one batch without backward, optimization, or writing checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from roma_pose.runtime import PROJECT_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--roma-cache-root", type=Path, default=PROJECT_ROOT / "outputs" / "roma_cache")
    p.add_argument("--split", default="train")
    p.add_argument("--sequences", nargs="*")
    p.add_argument("--max-frames-per-sequence", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--roma-setting", default="fast")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pretrained", default="siyan824/reloc3r-512")
    p.add_argument("--resume-checkpoint", type=Path)
    args = p.parse_args()

    from roma_pose.data.samples import build_samples
    from roma_pose.engine import make_loader, forward_normalized, initialize_from_checkpoint
    from roma_pose.models.reloc_roma import LockedDualReloc3rRoMaV2Fusion

    args.batch_size = 1
    samples = build_samples(args.data_root, args.split, args.sequences, args.max_frames_per_sequence)
    if not samples:
        p.error("No matching image/pose samples found; check --data-root and --split.")
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    sample = next((s for s in samples if s.frame_t != s.frame0), samples[0])
    batch = next(iter(make_loader([sample], args, shuffle=False)))
    model = LockedDualReloc3rRoMaV2Fusion(
        args.pretrained, image_size=args.image_size,
        roma_setting=args.roma_setting, use_roma_model=False,
    ).to(device).eval()
    if args.resume_checkpoint:
        initialize_from_checkpoint(model, args.resume_checkpoint, device)
    with torch.no_grad():
        result = forward_normalized(model, batch, device, amp=False)
    print(json.dumps({
        "mode": "forward_only",
        "device": str(device),
        "sequence": batch["sequence"],
        "frame0": batch["frame0"].tolist(),
        "frame_t": batch["frame_t"].tolist(),
        "prediction_shape": list(result["pred"].shape),
        "prediction": result["pred"].cpu().tolist(),
        "warp_shape": list(result["curr"]["roma_warp12"].shape),
        "certainty_shape": list(result["curr"]["roma_certainty12"].shape),
        "note": "Without --resume-checkpoint the wrapped head's regression layers are random.",
    }, indent=2))


if __name__ == "__main__":
    main()
