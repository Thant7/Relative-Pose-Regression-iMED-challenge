from __future__ import annotations
import argparse
from pathlib import Path
from .runtime import PROJECT_ROOT as ROOT

def main() -> None:
    p = argparse.ArgumentParser(description="Locked dual-direction Reloc3r + frozen RoMaV2 correspondence fusion training.")
    p.add_argument("--online-roma", action="store_true", help="Run online RoMaV2 instead of cached correspondences.")
    p.add_argument("--train", action="store_true", help="Run real train/test monitoring. Without this, run one-batch sanity only.")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--split", default="train")
    p.add_argument("--extra-train-splits", nargs="*", default=[], help="Additional splits to append to training only, e.g. --extra-train-splits test.")
    p.add_argument("--test-split", default="test")
    p.add_argument("--sequences", nargs="*")
    p.add_argument("--test-sequences", nargs="*")
    p.add_argument("--max-frames-per-sequence", type=int)
    p.add_argument("--test-max-frames-per-sequence", type=int)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/reloc3r_romav2_fusion_inject"))
    p.add_argument("--resume-checkpoint", type=Path, default=None, help="Load compatible same-architecture state, or compatible encoder/decoder tensors from another architecture.")
    p.add_argument("--roma-cache-root", type=Path, default=ROOT / "outputs" / "roma_cache", help="Directory containing precomputed RoMa tensors (default: outputs/roma_cache under the project root).")
    p.add_argument("--pretrained", default="siyan824/reloc3r-512")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--decoder-lr", type=float, default=5e-6)
    p.add_argument("--freeze-decoder", action="store_true", help="Freeze Reloc3r decoder; train only feature injector and wrapped pose head.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps-per-epoch", type=int)
    p.add_argument("--eval-every", type=int, default=5, help="Run full test/ATE evaluation every N epochs.")
    p.add_argument("--lambda-r", type=float, default=1.0)
    p.add_argument("--lambda-t", type=float, default=1.0)
    p.add_argument("--conf-threshold", type=float, default=0.5, help="Reserved for compatibility; feature injection uses continuous RoMa certainty as a gate.")
    p.add_argument("--roma-setting", default="fast", choices=["turbo", "fast", "base", "precise", "mega1500", "scannet1500", "wxbs", "satast"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--identity-tolerance", type=float, default=3e-2)
    args = p.parse_args()
    from .engine import train, sanity
    if args.online_roma:
        args.roma_cache_root = None
    if args.train:
        train(args)
    else:
        if args.max_frames_per_sequence is None:
            args.max_frames_per_sequence = 3
        if args.batch_size != 1:
            args.batch_size = 1
        sanity(args)


if __name__ == "__main__":
    main()
