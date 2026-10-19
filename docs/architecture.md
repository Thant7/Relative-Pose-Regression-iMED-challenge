# Architecture and exact conventions

## Feature injection

For each direction, the frozen image encoder supplies source and target features `[B,N,C]`. With the default landscape input, image size is 400 x 512, patches are 16 x 16, the token grid is 25 x 32, and encoder feature width is 1024. The decoder width is 768.

The warp and certainty maps are bilinearly resized to the patch grid. `grid_sample(..., align_corners=False, padding_mode="zeros")` samples the target feature map. Coordinates are `(x,y)`; positional indices supplied by Reloc3r are `(y,x)`.

The six geometry inputs per token are source coordinates, matched target coordinates, and their difference. A `6 -> 128 -> C` GELU MLP produces a geometry hint. A separate `LayerNorm(2C) -> Linear(2C,512) -> GELU -> Linear(512,C)` network combines the source and sampled target features.

The update is:

```text
gate       = clamp(certainty, 0, 1) * target_coordinate_in_bounds
correction = gate * (alpha * geometry_hint + beta * feature_hint)
new_source = source + correction
```

`alpha` and `beta` start at zero. Initial encoder features are therefore unchanged; the gates can receive gradients on the first backward pass, while the two hint MLPs start receiving nonzero gradients after those gates move away from zero. This does not mean the whole pretrained model is initially preserved: the wrapped head's final MLPs and regressors are random.

The same injector operates in both directions. Each direction samples the other image's **original** encoder features. Patch positional indices are unchanged. The code's `conf_threshold` argument is retained for compatibility but is not used as a hard threshold; confidence gates updates continuously.

## Pose head and directions

The shared head uses the last decoder tokens, a copied pretrained projection and residual convolutions, spatial average pooling, then two new feature-width Linear/ReLU blocks. Separate linear layers produce translation and rotation. The copied spatial layers and the new MLP/regression layers are trainable.

Nine rotation values are projected onto a proper rotation matrix by the inherited SVD routine; the optional 6D path orthonormalizes two vectors. Translation fills the last column of a 4 x 4 matrix; the bottom row is `[0,0,0,1]`.

The code applies the head to `dec1` and `dec2`, returning `P12` and `P21`. Its selected `pose` is **P12**. The original Reloc3r inference path describes its `dec2` output as native pose2to1; those native conventions must not be inferred just from these wrapper names. This release preserves the experiment's P12 selection without adding an inversion.

For each training sample, the model predicts the first stereo pair and current stereo pair, and normalizes:

```text
prediction_t = inverse(first["pose"]) @ current["pose"]
```

The loss supervises that normalized prediction with rotation geodesic loss plus translation Smooth L1 loss. The inherited objective does not independently supervise `P21` and has no explicit bidirectional consistency term.

## Optimization and evaluation

AdamW uses separate groups for the Reloc3r decoder (default learning rate 5e-6), feature injector (2e-5) and wrapped head (2e-5), weight decay 0.05, and gradient clipping at 1.0. `--freeze-decoder` removes frozen decoder parameters from optimization.

The engine preserves the local evaluator, including Sim(3) alignment and relative-pose error helpers. See `evaluation/pose_metrics.py` for the full aggregation and invalid-pose handling. Test split evaluation is monitoring in this research workflow; it must not be described as an untouched final test when used to select epochs.

Checkpoint loading preserves parameter key names. A same-architecture checkpoint loads compatible model tensors and optimizer state where available; a different architecture is filtered to compatible encoder/decoder tensors. This is not a guarantee of bitwise deterministic continuation: RNG and data-loader states are not restored.
