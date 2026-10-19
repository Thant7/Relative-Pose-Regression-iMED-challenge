from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .models.reloc_roma import LockedDualReloc3rRoMaV2Fusion
from .data.dataset import LockedFirstCurrentRoMaDataset, MixedSplitRoMaDataset, collate_batch
from .data.samples import build_samples, matrix_to_pose_row
from .utils.geometry import challenge_normalize, move_view, pose_loss, se3_inverse_torch
from .utils.modules import set_seed, count_params, grad_norm
from .evaluation.pose_metrics import evaluate_sequence
from .runtime import PROJECT_ROOT as ROOT
torch.set_float32_matmul_precision("highest")

RELOC3R_STREAM_MAPPING = {
    "original_code": "reloc3r_relpose.forward() computes pose1 from dec1 and pose2 from dec2; inference_relpose() returns pose2['pose'] as pose2to1.",
    "dec1": "original Reloc3r pose1 stream",
    "dec2": "original Reloc3r pose2 stream, documented in original code as native pose2to1",
    "locked_architecture_use": "Reloc3r encoder features are frozen; RoMaV2 correspondences sample matched target features and inject a learned residual into source encoder tokens before the shared Reloc3r decoder.",
    "challenge_output": "one final fused raw challenge-direction pose C_t; normalized as P_t = inverse(C_0) @ C_t",
}

def initialize_from_checkpoint(model: LockedDualReloc3rRoMaV2Fusion, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    own = model.state_dict()
    same_architecture = any(k.startswith("feature_injector.") for k in state) and any(k.startswith("wrapped_pose_head.") for k in state)

    allowed_prefixes = (
        "reloc.patch_embed.",
        "reloc.enc_blocks.",
        "reloc.enc_norm.",
        "reloc.decoder_embed.",
        "reloc.dec_blocks.",
        "reloc.dec_norm.",
    )
    compatible = {
        k: v
        for k, v in state.items()
        if (
            (same_architecture or k.startswith(allowed_prefixes))
            and k in own
            and own[k].shape == v.shape
        )
    }
    missing_compatible = model.load_state_dict(compatible, strict=False)

    init_report: dict[str, Any] = {
        "checkpoint": os.fspath(checkpoint),
        "init_mode": "full_same_architecture_resume" if same_architecture else "partial_reloc_encoder_decoder_load",
        "compatible_tensors_loaded": len(compatible),
        "missing_keys_count": len(missing_compatible.missing_keys),
        "source_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "next_epoch": int(ckpt.get("epoch", -1)) + 1 if same_architecture and isinstance(ckpt, dict) and "epoch" in ckpt else 0,
        "source_metrics": ckpt.get("metrics") if isinstance(ckpt, dict) else None,
    }
    return init_report



def load_optimizer_from_checkpoint(optimizer: torch.optim.Optimizer, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    if not isinstance(ckpt, dict) or "optimizer" not in ckpt:
        return {"optimizer_loaded": False, "reason": "checkpoint has no optimizer state"}
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        return {"optimizer_loaded": True}
    except Exception as exc:
        return {"optimizer_loaded": False, "reason": repr(exc)}



def forward_normalized(model: LockedDualReloc3rRoMaV2Fusion, batch: dict[str, Any], device: torch.device, amp: bool) -> dict[str, torch.Tensor]:
    first_v1 = move_view(batch["first_view1"], device)
    first_v2 = move_view(batch["first_view2"], device)
    curr_v1 = move_view(batch["curr_view1"], device)
    curr_v2 = move_view(batch["curr_view2"], device)
    
    with torch.cuda.amp.autocast(enabled=amp):
        if "roma_first_warp12" in batch:
            first = model.forward_precomputed(
                first_v1, first_v2,
                batch["roma_first_warp12"].to(device, non_blocking=True),
                batch["roma_first_cert12"].to(device, non_blocking=True),
                batch["roma_first_warp21"].to(device, non_blocking=True),
                batch["roma_first_cert21"].to(device, non_blocking=True),
            )
            curr = model.forward_precomputed(
                curr_v1, curr_v2,
                batch["roma_curr_warp12"].to(device, non_blocking=True),
                batch["roma_curr_cert12"].to(device, non_blocking=True),
                batch["roma_curr_warp21"].to(device, non_blocking=True),
                batch["roma_curr_cert21"].to(device, non_blocking=True),
            )
        else:
            roma_first1 = batch["roma_first1"].to(device, non_blocking=True)
            roma_first2 = batch["roma_first2"].to(device, non_blocking=True)
            roma_curr1 = batch["roma_curr1"].to(device, non_blocking=True)
            roma_curr2 = batch["roma_curr2"].to(device, non_blocking=True)
            first = model(first_v1, first_v2, roma_first1, roma_first2)
            curr = model(curr_v1, curr_v2, roma_curr1, roma_curr2)
            
    pred = challenge_normalize(first["pose"].float(), curr["pose"].float())
    return {"pred": pred, "first": first, "curr": curr}



def total_pose_loss(out: dict[str, torch.Tensor], target: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    total, stats = pose_loss(out["pred"], target, args.lambda_r, args.lambda_t)
    stats = dict(stats)
    stats["pose_loss"] = stats["loss"]
    return total, stats



def make_optimizer(model: LockedDualReloc3rRoMaV2Fusion, lr: float, weight_decay: float, decoder_lr: float | None = None) -> torch.optim.Optimizer:
    decoder_lr = float(lr if decoder_lr is None else decoder_lr)
    groups = [
        ("reloc_decoder", list(model.reloc.decoder_embed.parameters()) + list(model.reloc.dec_blocks.parameters()) + list(model.reloc.dec_norm.parameters()), decoder_lr),
        ("feature_injector", list(model.feature_injector.parameters()), lr),
        ("wrapped_pose_head", list(model.wrapped_pose_head.parameters()), lr),
    ]
    seen: set[int] = set()
    opt_groups = []
    for name, params, group_lr in groups:
        params = [p for p in params if p.requires_grad]
        if not params:
            continue
        for p in params:
            seen.add(id(p))
        opt_groups.append({"name": name, "params": params, "lr": float(group_lr)})
    return torch.optim.AdamW(opt_groups, weight_decay=weight_decay)



def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "metadata": metadata,
            "metrics": metrics,
        },
        path,
    )



def make_loader(samples: list[Any], args: argparse.Namespace, shuffle: bool, split: str | None = None) -> DataLoader:
    dataset = LockedFirstCurrentRoMaDataset(
        samples,
        args.image_size,
        roma_cache_root=args.roma_cache_root,
        split=args.split if split is None else split,
        roma_setting=args.roma_setting,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )



def make_mixed_train_loader(split_to_samples: list[tuple[str, list[Any]]], args: argparse.Namespace) -> DataLoader:
    datasets = [
        LockedFirstCurrentRoMaDataset(
            samples,
            args.image_size,
            roma_cache_root=args.roma_cache_root,
            split=split,
            roma_setting=args.roma_setting,
        )
        for split, samples in split_to_samples
    ]
    dataset = MixedSplitRoMaDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )



def train_one_epoch(model: LockedDualReloc3rRoMaV2Fusion, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    max_steps = min(len(loader), args.max_steps_per_epoch) if args.max_steps_per_epoch else len(loader)
    progress = tqdm(enumerate(loader), total=max_steps, desc=f"train epoch {args.current_epoch}", dynamic_ncols=True, leave=False)
    for step, batch in progress:
        out = forward_normalized(model, batch, device, bool(args.amp))
        target = batch["target"].to(device, non_blocking=True)
        loss, stats = total_pose_loss(out, target, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        bs = int(target.shape[0])
        count += bs
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + float(v) * bs
        progress.set_postfix(
            loss=totals.get("loss", 0.0) / max(count, 1),
            rot_deg=totals.get("rot_deg", 0.0) / max(count, 1),
            t=totals.get("trans_smooth_l1", 0.0) / max(count, 1),
        )
        if args.max_steps_per_epoch and step + 1 >= args.max_steps_per_epoch:
            break
    return {f"train_{k}": v / max(count, 1) for k, v in totals.items()} | {"train_num_samples": float(count)}



@torch.no_grad()
def evaluate_samples(model: LockedDualReloc3rRoMaV2Fusion, samples: list[Any], device: torch.device, args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    if not samples:
        return {
            f"{prefix}_loss": float("nan"),
            f"{prefix}_rot_deg": float("nan"),
            f"{prefix}_trans_smooth_l1": float("nan"),
            f"{prefix}_mean_ate": float("nan"),
            f"{prefix}_median_per_sequence_ate": float("nan"),
            f"{prefix}_registered_pct": float("nan"),
            f"{prefix}_num_sequences": 0,
            f"{prefix}_num_samples": 0,
            f"{prefix}_per_sequence": [],
        }
    eval_split = args.test_split if prefix == "test" else args.split
    loader = make_loader(samples, args, shuffle=False, split=eval_split)
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    pred_by_seq: dict[str, list[Any]] = {}
    gt_by_seq: dict[str, list[Any]] = {}
    progress = tqdm(loader, total=len(loader), desc=f"{prefix} eval", dynamic_ncols=True, leave=False)
    for batch in progress:
        out = forward_normalized(model, batch, device, bool(args.amp))
        target = batch["target"].to(device, non_blocking=True)
        loss, stats = total_pose_loss(out, target, args)
        bs = int(target.shape[0])
        count += bs
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + float(v) * bs
        progress.set_postfix(loss=totals.get("loss", 0.0) / max(count, 1))

        pred_np = out["pred"].detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        frame0 = batch["frame0"].cpu().numpy().tolist()
        frame_t = batch["frame_t"].cpu().numpy().tolist()
        for i, seq_name in enumerate(batch["sequence"]):
            force_identity = int(frame_t[i]) == int(frame0[i])
            pred_by_seq.setdefault(seq_name, []).append(matrix_to_pose_row(int(frame_t[i]), pred_np[i], force_identity=force_identity))
            gt_by_seq.setdefault(seq_name, []).append(matrix_to_pose_row(int(frame_t[i]), target_np[i], force_identity=force_identity))

    per_seq: list[dict[str, Any]] = []
    for seq_name in sorted(pred_by_seq):
        pred_rows = sorted(pred_by_seq[seq_name], key=lambda r: r.frame_idx)
        gt_rows = sorted(gt_by_seq[seq_name], key=lambda r: r.frame_idx)
        metrics = evaluate_sequence(gt_rows, pred_rows)
        per_seq.append({"sequence": seq_name, **metrics})
    valid = [r for r in per_seq if int(r.get("num_matched_poses", 0)) > 0 and np.isfinite(float(r["mean_ate"]))]
    return {
        f"{prefix}_loss": totals.get("loss", float("nan")) / max(count, 1),
        f"{prefix}_rot_deg": totals.get("rot_deg", float("nan")) / max(count, 1),
        f"{prefix}_trans_smooth_l1": totals.get("trans_smooth_l1", float("nan")) / max(count, 1),
        f"{prefix}_mean_ate": float(np.mean([float(r["mean_ate"]) for r in valid])) if valid else float("nan"),
        f"{prefix}_median_per_sequence_ate": float(np.median([float(r["mean_ate"]) for r in valid])) if valid else float("nan"),
        f"{prefix}_registered_pct": float(np.mean([float(r["registered_pct"]) for r in per_seq])) if per_seq else float("nan"),
        f"{prefix}_num_sequences": len(per_seq),
        f"{prefix}_num_samples": count,
        f"{prefix}_per_sequence": per_seq,
    }



def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_samples = build_samples(args.data_root, args.split, args.sequences, args.max_frames_per_sequence)
    extra_train: list[tuple[str, list[Any]]] = []
    for extra_split in args.extra_train_splits:
        extra_samples = build_samples(args.data_root, extra_split, None, args.max_frames_per_sequence)
        extra_train.append((extra_split, extra_samples))
    test_samples = [] if args.eval_every == 0 else build_samples(args.data_root, args.test_split, args.test_sequences, args.test_max_frames_per_sequence)
    train_loader = make_mixed_train_loader([(args.split, train_samples), *extra_train], args)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    
    # 💡 fb_alpha 인자 제거 (버그 해결)
    model = LockedDualReloc3rRoMaV2Fusion(
        args.pretrained,
        image_size=args.image_size,
        roma_setting=args.roma_setting,
        use_roma_model=args.roma_cache_root is None,
        conf_threshold=args.conf_threshold,
        freeze_decoder=args.freeze_decoder,
    ).to(device)
    
    init_report = initialize_from_checkpoint(model, args.resume_checkpoint, device) if args.resume_checkpoint else {
        "checkpoint": None,
        "init_mode": "clean_public_pretrained_only",
        "reloc_initialization": f"Reloc3rRelpose.from_pretrained({args.pretrained!r}); encoder/decoder reused",
        "roma_initialization": f"RoMaV2 public weights, setting={args.roma_setting!r}" if args.roma_cache_root is None else f"disabled during training; using cache at {args.roma_cache_root}",
        "fusion_pose_head_initialization": "proj/res_conv copied from pretrained Reloc3r pose_head; more_mlps/fc_t/fc_rot randomly initialized",
        "feature_injector_initialization": "zero residual start: alpha=0, beta=0; geom_mlp/feat_mlp random",
        "next_epoch": 0,
    }
    optimizer = make_optimizer(model, args.lr, args.weight_decay, args.decoder_lr)
    if args.resume_checkpoint and init_report.get("init_mode") == "full_same_architecture_resume" and not args.freeze_decoder:
        init_report["optimizer"] = load_optimizer_from_checkpoint(optimizer, args.resume_checkpoint, device)
    start_epoch = int(init_report.get("next_epoch", 0))

    metadata = {
        "model": "locked_dual_reloc3r_romav2_correspondence_feature_injection",
        "input_sizes": {"reloc3r": "400x512", "roma_setting": args.roma_setting},
        "roma_cache_root": os.fspath(args.roma_cache_root) if args.roma_cache_root else None,
        "train_split": args.split,
        "extra_train_splits": args.extra_train_splits,
        "num_train_samples": len(train_samples) + sum(len(samples) for _, samples in extra_train),
        "num_test_samples": len(test_samples),
        "eval_every": args.eval_every,
        "initialization": init_report,
        "frozen": ["Reloc3r encoder", "RoMaV2"] + (["Reloc3r decoder"] if args.freeze_decoder else []),
        "trainable": ([] if args.freeze_decoder else ["Reloc3r decoder"]) + ["RoMaCorrespondenceFeatureInjection", "WrappedPoseHead"],
        "challenge_normalization": "P_t = inverse(C_0) @ C_t",
        "fusion": "RoMa correspondence feature injection: frozen encoder feat1/feat2 [B,800,1024] are enriched using geometry MLP and matched target features sampled by RoMa warp before Reloc3r decoder; pos1/pos2 are unchanged.",
    }
    (args.output_dir / "config.json").write_text(json.dumps({**vars(args), **metadata}, indent=2, default=os.fspath), encoding="utf-8")
    log_path = args.output_dir / "log.jsonl"
    best_test = float("inf")
    best_epoch = -1
    rows: list[dict[str, Any]] = []

    for epoch in range(start_epoch, start_epoch + args.epochs):
        start = time.perf_counter()
        args.current_epoch = epoch
        train_stats = train_one_epoch(model, train_loader, optimizer, device, args)
        should_eval = (
            args.eval_every > 0
            and (
                epoch == 0
                or (epoch + 1) % args.eval_every == 0
                or epoch == args.epochs - 1
            )
        )
        test_stats = evaluate_samples(model, test_samples, device, args, "test") if should_eval else {
            "test_loss": float("nan"),
            "test_rot_deg": float("nan"),
            "test_trans_smooth_l1": float("nan"),
            "test_mean_ate": float("nan"),
            "test_median_per_sequence_ate": float("nan"),
            "test_registered_pct": float("nan"),
            "test_per_sequence": [],
        }
        row = {
            "epoch": epoch,
            **train_stats,
            "evaluated_test": bool(should_eval),
            "test_loss": test_stats["test_loss"],
            "test_rot_deg": test_stats["test_rot_deg"],
            "test_trans_smooth_l1": test_stats["test_trans_smooth_l1"],
            "test_mean_ate": test_stats["test_mean_ate"],
            "test_median_per_sequence_ate": test_stats["test_median_per_sequence_ate"],
            "test_registered_pct": test_stats["test_registered_pct"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "runtime_s": time.perf_counter() - start,
        }
        rows.append(row)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        save_checkpoint(args.output_dir / "checkpoint-last.pth", model, optimizer, epoch, args, metadata, row)
        if np.isfinite(row["test_mean_ate"]) and row["test_mean_ate"] < best_test:
            best_test = float(row["test_mean_ate"])
            best_epoch = epoch
            save_checkpoint(args.output_dir / "checkpoint-best-test.pth", model, optimizer, epoch, args, metadata, row)
            (args.output_dir / "test_per_sequence_best.json").write_text(json.dumps(test_stats["test_per_sequence"], indent=2), encoding="utf-8")

    summary = {
        "best_test_epoch": best_epoch,
        "best_test_mean_ate": best_test if np.isfinite(best_test) else float("nan"),
        "start_epoch": start_epoch,
        "final_epoch": start_epoch + args.epochs - 1,
        "num_train_samples": len(train_samples) + sum(len(samples) for _, samples in extra_train),
        "extra_train_splits": args.extra_train_splits,
        "extra_train_counts": {split: len(samples) for split, samples in extra_train},
        "num_test_samples": len(test_samples),
        "eval_every": args.eval_every,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"training_summary": summary}, indent=2), flush=True)



def sanity(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    samples = build_samples(args.data_root, args.split, args.sequences, args.max_frames_per_sequence)
    # Comparing the reference frame with itself can cancel the head gradient.
    samples = [sample for sample in samples if sample.frame_t != sample.frame0]
    if not samples:
        raise ValueError("Sanity gradient checks require a non-reference frame; use at least two frames per sequence.")
    loader = DataLoader(
        LockedFirstCurrentRoMaDataset(samples, args.image_size, roma_cache_root=args.roma_cache_root, split=args.split, roma_setting=args.roma_setting),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = LockedDualReloc3rRoMaV2Fusion(
        args.pretrained,
        image_size=args.image_size,
        roma_setting=args.roma_setting,
        use_roma_model=args.roma_cache_root is None,
        conf_threshold=args.conf_threshold,
        freeze_decoder=args.freeze_decoder,
    ).to(device)
    
    init_report = initialize_from_checkpoint(model, args.resume_checkpoint, device) if args.resume_checkpoint else {
        "checkpoint": None,
        "init_mode": "clean_public_pretrained_only",
        "reloc_initialization": f"Reloc3rRelpose.from_pretrained({args.pretrained!r}); encoder/decoder reused",
        "roma_initialization": f"RoMaV2 public weights, setting={args.roma_setting!r}" if args.roma_cache_root is None else f"disabled during sanity; using cache at {args.roma_cache_root}",
        "fusion_pose_head_initialization": "proj/res_conv copied from pretrained Reloc3r pose_head; more_mlps/fc_t/fc_rot randomly initialized",
        "feature_injector_initialization": "zero residual start: alpha=0, beta=0; geom_mlp/feat_mlp random",
    }
    opt = make_optimizer(model, args.lr, args.weight_decay, args.decoder_lr)

    batch = next(iter(loader))
    out = forward_normalized(model, batch, device, bool(args.amp))
    target = batch["target"].to(device, non_blocking=True)
    loss, loss_stats = total_pose_loss(out, target, args)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    modules = {
        "Reloc3r encoder": nn.ModuleList([model.reloc.patch_embed, model.reloc.enc_blocks, model.reloc.enc_norm]),
        "Reloc3r decoder": nn.ModuleList([model.reloc.decoder_embed, model.reloc.dec_blocks, model.reloc.dec_norm]),
        "Shared FeatureInjector": model.feature_injector,
        "Shared WrappedPoseHead": model.wrapped_pose_head,
        "RoMaV2 full": model.roma,
    }
    param_table = {
        name: {"total": count_params(module)[0], "trainable": count_params(module)[1], "grad_norm": grad_norm(module)}
        for name, module in modules.items()
    }
    
    if "roma_curr1" in batch:
        roma_input_shape1 = list(batch["roma_curr1"].shape)
        roma_input_shape2 = list(batch["roma_curr2"].shape)
        roma_input_source = "online_original_images"
    else:
        roma_input_shape1 = list(batch["roma_curr_warp12"].shape)
        roma_input_shape2 = list(batch["roma_curr_warp21"].shape)
        roma_input_source = "precomputed_flow_input"

    shape_table = {
        "I1": roma_input_shape1,
        "I2": roma_input_shape2,
        "roma_input_source": roma_input_source,
        "dec1_last": list(out["curr"]["dec1_last"].shape),
        "dec2_last": list(out["curr"]["dec2_last"].shape),
        "roma_warp12": list(out["curr"]["roma_warp12"].shape),
        "roma_warp21": list(out["curr"]["roma_warp21"].shape),
        "roma_certainty12": list(out["curr"]["roma_certainty12"].shape),
        "roma_certainty21": list(out["curr"]["roma_certainty21"].shape),
        "out_t": list(out["curr"]["out_t"].shape),
        "out_r": list(out["curr"]["out_r"].shape),
        "P12": list(out["curr"]["P12"].shape),
        "P21": list(out["curr"]["P21"].shape),
        "pose": list(out["curr"]["pose"].shape),
        "pred": list(out["pred"].shape),
    }
    ident = challenge_normalize(out["first"]["pose"], out["first"]["pose"])
    optimizer_groups = [
        {"name": g["name"], "lr": g["lr"], "num_params": int(sum(p.numel() for p in g["params"]))}
        for g in opt.param_groups
    ]
    sanity_report = {
        "stream_mapping": RELOC3R_STREAM_MAPPING,
        "initialization": init_report,
        "shapes": shape_table,
        "param_table": param_table,
        "loss": loss_stats,
        "checks": {
            "reloc_encoder_trainable_params": count_params(modules["Reloc3r encoder"])[1],
            "reloc_decoder_trainable_params": count_params(modules["Reloc3r decoder"])[1],
            "roma_trainable_params": count_params(model.roma)[1],
            "feature_injector_grad_norm": grad_norm(model.feature_injector),
            "feature_injector_alpha": float(model.feature_injector.alpha.detach().cpu()),
            "feature_injector_beta": float(model.feature_injector.beta.detach().cpu()),
            "feature_injector_grad_note": "alpha/beta should receive gradients; MLP weights receive gradients after alpha/beta move away from zero",
            "wrapped_pose_head_grad_nonzero": grad_norm(model.wrapped_pose_head) > 0, 
            "dual_branch_output": True,
            "challenge_pose_uses_P12": True,
            "same_feature_injector_object_12_21": id(model.feature_injector) == id(model.feature_injector),
            "same_wrapped_pose_head_object_12_21": id(model.wrapped_pose_head) == id(model.wrapped_pose_head),
            "identity_t_max_abs": float(ident[:, :3, 3].abs().max().detach().cpu()),
            "optimizer_groups": optimizer_groups,
        },
    }
    print(json.dumps({"locked_dual_fusion_sanity": sanity_report}, indent=2), flush=True)

    expected_shapes = {
        "dec1_last": [args.batch_size, 800, 768],
        "dec2_last": [args.batch_size, 800, 768],
    }
    for key, expected in expected_shapes.items():
        if shape_table[key] != expected:
            raise AssertionError(f"expected {key} {expected}, got {shape_table[key]}")
    if shape_table["out_t"] != [args.batch_size, 3]:
        raise AssertionError("invalid translation shape")
    if shape_table["out_r"] != [args.batch_size, 9]:
        raise AssertionError("invalid raw rotation shape")
    if shape_table["pose"] != [args.batch_size, 4, 4]:
        raise AssertionError("invalid pose shape")
    if sanity_report["checks"]["reloc_encoder_trainable_params"] != 0:
        raise AssertionError("Reloc3r encoder is not frozen")
    if not args.freeze_decoder and sanity_report["checks"]["reloc_decoder_trainable_params"] == 0:
        raise AssertionError("Reloc3r decoder is not trainable")
    if args.freeze_decoder and sanity_report["checks"]["reloc_decoder_trainable_params"] != 0:
        raise AssertionError("Reloc3r decoder is not frozen")
    if sanity_report["checks"]["roma_trainable_params"] != 0:
        raise AssertionError("RoMaV2 is not frozen")
    if param_table["Reloc3r encoder"]["grad_norm"] != 0 or param_table["RoMaV2 full"]["grad_norm"] != 0:
        raise AssertionError("Frozen modules received gradients")
    if not sanity_report["checks"]["wrapped_pose_head_grad_nonzero"]:
        raise AssertionError("Shared WrappedPoseHead did not receive gradients")
    if sanity_report["checks"]["identity_t_max_abs"] > args.identity_tolerance:
        raise AssertionError("frame-0 normalization identity check failed")
