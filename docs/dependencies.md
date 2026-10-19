# External sources and environment

This release does not vendor Reloc3r, CroCo or RoMaV2. It also does not grant rights to their code, weights or datasets. Their upstream terms continue to apply; a project license has not been selected on the user's behalf.

The original local Reloc3r checkout records remote `https://github.com/ffrivera0/reloc3r.git` and revision `761fac648e9c21fd7dcda01ab2ccd4fc20058102`. To obtain that source:

```powershell
git clone --recursive https://github.com/ffrivera0/reloc3r.git third_party/reloc3r
git -C third_party/reloc3r checkout 761fac648e9c21fd7dcda01ab2ccd4fc20058102
git -C third_party/reloc3r submodule update --init --recursive
```

Follow that checkout's own installation instructions for its environment and any required CUDA/CroCo positional-embedding extensions. The package dependencies in `pyproject.toml` are the direct Python requirements for this release, not a tested cross-platform lockfile for every external backend.

External lookup is controlled by `RELOC3R_ROOT` (default `third_party/reloc3r`) and `ROMAV2_ROOT` (default `third_party/RoMaV2`). Cached runs need Reloc3r but do not need RoMaV2. Online runs require an importable `romav2.RoMaV2` with the API used in `models/reloc_roma.py`; no RoMaV2 revision is claimed because its source is absent in the local workspace.

`environment-local.txt` records selected package versions in the existing `miccai_pe` environment at preparation time. It is provenance, not a portable installation command.
