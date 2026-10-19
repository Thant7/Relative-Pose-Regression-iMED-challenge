# Validation record

Checked on 2026-10-19 in the existing Windows `miccai_pe` environment, using the local Reloc3r checkout and cached correspondences. No full training run was performed.

| Check | Result |
|---|---|
| Unit tests | 4 passed: identity coordinate sampling, confidence/in-bounds gating, zero-start residual gradients, SE(3) normalization |
| Original-definition comparison | All 37 extracted definitions matched original ASTs before the documented sanity-sampler correction; the remaining 36 are unchanged after it |
| CLI help | Passed without loading external models |
| Engine imports | Passed with `RELOC3R_ROOT` set to the local checkout |
| CUDA forward-only | Passed on frames 2 and 5 of `session_001_chickenbreast_10deg_circular`; normalized pose `[1,4,4]`, warp `[1,200,256,2]`, certainty `[1,1,200,256]` |
| CUDA forward/backward sanity | Passed after excluding reference-equals-current samples; encoder gradient zero, pose-head gradients nonzero, injector gradient norm about 0.001417 |

The original first-sample sanity check failed its nonzero-head-gradient assertion when reference and current were the same frame. The organized version selects non-reference samples for that diagnostic; training sample enumeration is unchanged.

Expected runtime messages include the inherited deprecated autocast API warning and the local checkout's use of a PyTorch RoPE fallback instead of a compiled CUDA extension. Both checks completed with these messages. No pretrained experiment checkpoint was supplied: the newly initialized regression head's predictions establish execution, not accuracy.

Source hashes are recorded in `source-sha256.json`, and selected environment versions in `environment-local.txt`. A fresh machine still needs external model sources, weights, data and caches; online RoMaV2 and a fresh external-source installation were not tested.
