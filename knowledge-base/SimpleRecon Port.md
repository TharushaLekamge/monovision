---
tags: [simplerecon, port, results]
date: 2026-10-21
---
# SimpleRecon Port (inference)

Goal: run SimpleRecon in the same env as the GAN (`monovision`, Python 3.11, current torch) instead of its pinned Python 3.9.7 / torch 1.10 / CUDA 11.3 env, which cannot target this machine's RTX 5060 (sm_120). Scope: inference + fusion on `main`; **training is ported on branch `simplerecon-training`** (see below). Location: patch series, so `third_party/simplerecon` stays at the pinned commit (see [[Third-Party and Licensing]]).

## Use
```bash
bash scripts/01_setup_simplerecon.sh --weights    # copies submodule to build/simplerecon, applies patches/simplerecon/*.patch, fetches hero_model.ckpt
pip install -r requirements-recon.txt             # into the monovision env
cd build/simplerecon && python test.py --name HERO_MODEL --output_base_path OUT \
  --config_file configs/models/hero_model.yaml --load_weights_from_checkpoint weights/hero_model.ckpt \
  --data_config configs/data/vdr_dense.yaml --batch_size 1 --fast_cost_volume --run_fusion --depth_fuser open3d --fuse_color
```
`build/` is gitignored; `build/simplerecon` is disposable and rebuilt on every run, while the checkpoint lives in `build/weights/` (SHA-256 checked on download, trust-on-first-use) and is symlinked in, so re-running never deletes it. `SIMPLERECON_BUILD_DIR` overrides the output dir. Edit `dataset_path` in the data config to point at your scans.

## What had to change (4 patches)
| Patch | Problem on new versions | Change |
|---|---|---|
| 0001 | `torch.jit.script` cannot compile `kornia.filters` calls (`pyrdown`, `NormalGenerator`) | run as plain Python; same maths |
| 0002 | torch ≥ 2.6 `weights_only=True` rejects the pickled `options.Options`; lightning ≥ 2 needs a version key the stripped checkpoints lack | allowlist `options.Options`; override `DepthModel.load_from_checkpoint` to tag the version (the files hold only weights + hparams, nothing to migrate); full training checkpoints already have the key and are passed through untouched, the key check uses `mmap` so weights aren't read twice |
| 0003 | `moviepy.editor`, `write_videofile(verbose=)` and `plt.cm.get_cmap` removed | use `import moviepy`, drop `verbose`, `plt.get_cmap` |
| 0004 | `marching_cubes(single_mesh=True)` exists only in the authors' scikit-image fork (no wheel for this Python) | fall back to stock `marching_cubes` with a warning |

## Findings worth remembering
- **open3d 0.20.0 is broken for this**: `ScalableTSDFVolume` returned empty or tiny meshes even for a flat wall (reproduced with a 20-line script, with numpy 1 and 2). 0.18.0 and 0.19.0 work. Pinned `open3d==0.19.0`. The first run with 0.20 *looked* successful: depth metrics fine, `.ply` written, but only 1.5k and 0.2k vertices. Always check mesh size.
- Installing `torchvision` upgraded torch in the env from 2.11.0+cu128 to **2.14.1+cu130**. GPU and all 26 tests still pass.
- **`git apply` silently skips files when run from a subdirectory of a repo** (our gitignored `build/`), yet exits 0, so the setup script printed "applied" with nothing patched. The script now uses `patch -p1 -d`. Always diff the result after changing patch tooling.
- moviepy 2 also dropped `write_videofile(verbose=)`; my first version of patch 0003 missed it (found in review; `save_viz_video_frames` was never exercised by `test.py`).
- Google Drive via `gdown` failed for the sample scans; `curl -L "https://drive.usercontent.google.com/download?id=<id>&export=download&confirm=t"` worked.

## Results (Niantic's two sample VDR scans, `vdr_dense`, hero_model, batch 1, RTX 5060)
| Fuser | house | living_room | wall time |
|---|---|---|---|
| `open3d` (0.19, colour) | 113,784 verts | 128,875 verts | 4 min 53 s |
| `ours` (stock marching cubes) | 178,209 verts | 175,920 verts | 7 min 8 s |

Depth vs the scans' ARKit LiDAR depth (VDR has no ground-truth mesh; identical across runs): abs_diff 0.0778, abs_rel 0.0454, rmse 0.116, a25 98.8 %, a1 98.8 %. Model time about 90 s for the two scans.

## Not verified
- **No numerical comparison with the original pinned env.** It cannot run here. Evidence of correctness is the depth metrics against ground truth and plausible meshes, not bit-parity. A sample-scan reference number from the authors would settle it.
- `--depth_fuser ours` meshes may differ from the paper's because of the missing `single_mesh` option.
- Not run: ScanNet, ios_logger / `arkit_dataset.py` input (the path this project will use), `pc_fusion.py`, live meshing, `--fast_cost_volume` off, batch > 1, peak VRAM.
- `torch.jit.script` deprecation warnings remain from other modules.

## Training port (branch `simplerecon-training`, worktree `../monovision-train`)
Added patch `0005-Port-training-to-pytorch-lightning-2.patch` (only in the worktree's `patches/simplerecon/`, not yet on `main`). The worktree was cut from the initial commit, so the patch tooling (`patches/`, `scripts/01_setup_simplerecon.sh`, `requirements-recon.txt`) was copied in and `third_party/simplerecon` symlinks to the main checkout. Nothing is committed on either side. SimpleRecon's code is never tracked here, only patches.

| Change in `train.py` | Why |
|---|---|
| `gpus=` → `accelerator`/`devices` | removed in lightning 2 |
| `precision` 16/32 → `"16-mixed"`/`"32-true"` | string form required |
| `DDPPlugin` → `DDPStrategy`, only when `--gpus > 1` | plugin removed; single GPU uses `"auto"` |
| `resume_from_checkpoint=` → `fit(ckpt_path=)` | moved in lightning 2 |
| `depth_model.py`: log `min(4, batch)` images | original loop did `range(4)`, crashing for batch < 4 (needed on 8 GB) |

Tests (VDR sample scans with hand-made train/val tuple files, hero_model config, RTX 5060, fp16):
- From scratch, batch 1, 20 steps, about 2.4 it/s: runs; val loss 1.68 → 1.21 (meaningless, tiny data).
- `--resume last.ckpt`: restores and continues to step 30.
- `--load_weights_from_checkpoint` (released hero_model): val loss 0.29 → 0.21.
- Batch 2: runs. A saved checkpoint reloads through `DepthModel.load_from_checkpoint` with identical weights.
- Checkpoints are about 600 MB each (weights + optimiser); mind the disk.

Not verified: multi-GPU DDP (one GPU here), real ScanNet training, `--precision 32`, matching the paper's training results, the `unet_encoder` variant, long-run stability. The paper-scale recipe (batch 16, 2×A100) will not fit on 8 GB.

## Review follow-ups (2026-10-21, python-pr-review)
All eight findings addressed: moviepy `verbose` fixed; setup script keeps weights and verifies their SHA-256; lighter checkpoint loader; trailing whitespace removed from 0005 (patches regenerated, `git diff --check` clean); `requirements-recon.txt` pins `torchvision==0.29.1` and `pytorch-lightning>=2,<2.7`; the worktree's absolute `third_party/` symlink is now in the worktree's `info/exclude` so it can't be committed (replace it with a real submodule checkout once `main` commits the submodule).

Tests added in `tests/` (`conftest.py`, `test_simplerecon_port.py`, 5 tests, ~12 s): every patch applies via the script; video writing; `to_mesh` fallback warns and returns a mesh; stripped checkpoint loads (needs `--weights`, else skipped); open3d flat-wall fusion is non-empty (guards the 0.20.0 bug). The first three and the loader test fail on unpatched upstream code (verified). Not unit-tested: the Trainer argument mapping in `train.py` (inline in `main`; covered only by the manual smoke runs). A fine-tune smoke run after the loader change: val loss 0.157 after 8 steps.
