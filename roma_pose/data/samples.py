from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..evaluation.io_pose import PoseRow, read_pose_txt

@dataclass(frozen=True)
class Sample:
    sequence: str
    frame0: int
    frame_t: int
    e1_l_0: Path
    e2_l_0: Path
    e1_l_t: Path
    e2_l_t: Path
    target: np.ndarray



def pose_row_to_matrix(row: PoseRow) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat(row.q_xyzw).as_matrix().astype(np.float32)
    T[:3, 3] = row.t.astype(np.float32)
    return T



def matrix_to_pose_row(frame_idx: int, T: np.ndarray, force_identity: bool = False) -> PoseRow:
    if force_identity:
        return PoseRow(
            frame_idx=frame_idx,
            t=np.zeros(3, dtype=np.float64),
            q_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
    return PoseRow(
        frame_idx=frame_idx,
        t=T[:3, 3].astype(np.float64),
        q_xyzw=Rotation.from_matrix(T[:3, :3]).as_quat().astype(np.float64),
    )



def image_maps(seq_dir: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    e1 = {int(p.stem.split("_")[1]): p for p in sorted((seq_dir / "endoscope1" / "L").glob("frame_*.png"))}
    e2 = {int(p.stem.split("_")[1]): p for p in sorted((seq_dir / "endoscope2" / "L").glob("frame_*.png"))}
    return e1, e2



def build_samples(data_root: Path, split: str, sequences: list[str] | None, max_frames_per_sequence: int | None) -> list[Sample]:
    split_dir = data_root / split
    wanted = set(sequences) if sequences else None
    samples: list[Sample] = []
    for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if wanted is not None and seq_dir.name not in wanted:
            continue
        pose_path = seq_dir / "pose.txt"
        if not pose_path.exists():
            continue
        rows = read_pose_txt(pose_path)
        e1, e2 = image_maps(seq_dir)
        common = [r for r in rows if r.frame_idx in e1 and r.frame_idx in e2]
        if max_frames_per_sequence is not None:
            common = common[:max_frames_per_sequence]
        if not common:
            continue
        first = common[0]
        for row in common:
            samples.append(
                Sample(
                    sequence=seq_dir.name,
                    frame0=first.frame_idx,
                    frame_t=row.frame_idx,
                    e1_l_0=e1[first.frame_idx],
                    e2_l_0=e2[first.frame_idx],
                    e1_l_t=e1[row.frame_idx],
                    e2_l_t=e2[row.frame_idx],
                    target=pose_row_to_matrix(row),
                )
            )
    return samples
