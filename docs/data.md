# Images, ground truth and correspondence cache

Expected image layout:

```text
data/{train,test}/{sequence}/
  endoscope1/L/frame_000002.png
  endoscope2/L/frame_000002.png
  pose.txt
```

Pose rows are `frame_idx tx ty tz qx qy qz qw`. The loader intersects image IDs with pose IDs. The first available matched frame is the sample's reference frame; ground truth is used as supplied and is expected to follow the challenge's first-frame convention. Dataset units are preserved.

Cache lookup:

```text
{roma_cache_root}/{split}/roma_{setting}/{sequence}/frame_{frame_idx:06d}.pt
```

The existing workspace cache is `D:/MICCAI_PE/outputs/roma_cache` with setting `fast`. At preparation time it contained 6,095 train and 1,895 test tensors. A loaded example used a 200 x 256 correspondence grid. These counts describe local files, not files included in Git.

Two cache representations are supported:

1. `warp_ab`, `certainty`, `warp_ba`, `certainty_ba`: normalized target-coordinate maps and confidence maps.
2. `flow_input12`, `flow_input21`: each `[3,H,W]`, storing normalized half-displacement in the first two channels and certainty in the third.

For representation 2, the original code reconstructs a normalized source grid with `linspace(-1,1)` and computes `target = source + 2 * displacement`. Coordinates are clamped to `[-1,1]` and certainty to `[0,1]`. The resulting maps are later sampled with `align_corners=False`. These inherited endpoint and clamping choices are preserved; their geometric consequences should be evaluated separately rather than silently changed during organization.

`W12[b,y,x]` is the matched target coordinate `(x_target,y_target)` in image 2 for a source location in image 1. It is not an intensity and is not itself displacement. `W21` reverses source and target. `C12` and `C21` are their confidence maps, not camera calibration matrices.

No images, pose files, pretrained weights or `.pt` caches are distributed by this repository. Training requires the corresponding files and permission to use them.
