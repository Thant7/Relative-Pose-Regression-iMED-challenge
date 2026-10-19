# Research provenance

Prepared from the local workspace on 2026-10-19. This release covers the feature-injection v1 script, not every RAFT, late-fusion, weighted-pooling or positional-injection experiment in the workspace.

| Release module | Original source |
|---|---|
| models, dataset, engine, CLI | `train_reloc3r_romav2_feature_inject_v1.py` |
| sample enumeration, SE(3) utilities, pose loss | selected definitions from `train_reloc3r_raft_flow_head.py` |
| pose metrics, alignment, RPE and pose I/O | `imedpe/src/imcpe/{pose_metrics,alignment,rpe,io_pose}.py` |

The organization preserves model parameter names, feature-update equations, pose-head initialization, selected P12 branch, losses and training behavior. External source paths are centralized in `runtime.py`. The release adds a forward-only entry point and an explicit `--online-roma` option; cache remains the default. Historical scripts are not imported by the release package.

Historical configurations, including `outputs/reloc3r_romav2_feature_inject_v1_train_only_e10/config.json`, used `outputs/roma_cache`. Some historical metadata still calls the model `rope_positional_injection`; the current implementation injects into encoder features and does not modify positional indices. The archived experiment arguments are transcribed in `configs/train_only_e10.args`; this file is an argument reference, not a config parsed automatically by the CLI.

The release fixes one sanity-check issue: it excludes reference-equals-current samples when checking for nonzero pose-head gradients. Comparing a frame with itself can cancel those gradients through first-frame normalization and fail the old assertion despite correct autograd. The training sampler and model equations are unchanged.

The project author reports that this method won the iMED Challenge 2026 with **2.193 mm ATE RMSE**. This historical result is recorded from the author's statement; an official leaderboard link, exact submitted checkpoint and full evaluation protocol have not been attached to this release. The code organization and local checks do not independently reproduce that score.

No new benchmark evaluation or completed full training run was performed during this refactor. Refactor checks and their results are recorded in `VALIDATION.md`. Existing trained weights are external.
