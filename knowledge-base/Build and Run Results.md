---
tags: [results, run]
date: 2026-10-21
---
# Build and Run Results (2026-10-21)

First attempt to build and run the project on this machine. Only the GAN stage exists in code, so that is all that could run. See [[Status]].

## Machine
Fedora, Linux 7.0.12. NVIDIA RTX 5060 Laptop GPU (sm_120), **8 GB VRAM** (about 1 GB used by the desktop, so about 7 GB usable). Driver 595.80.

## Build
- The `monovision` conda env **already existed**: `~/.conda/envs/monovision`, Python 3.11.17, torch 2.11.0+cu128, numpy 2.4.6, tensorflow-cpu 2.21.0, tf-slim, pytest, and `monovision` 0.1.0 as an editable install.
- The shell default `python` is the system Python 3.14.6 with no torch. **Use the env**: `conda activate monovision` or `~/.conda/envs/monovision/bin/python`.
- `python scripts/check_env.py`: all required items OK, including the GPU kernel test (torch 2.11+cu128 supports sm_120).
- No compile step. The package is pure Python.

## Tests (`bash scripts/run_tests.sh`)
| Suite | Result |
|---|---|
| `tests/test_port.py` | 17 passed, 3 s |
| `tests/test_parity_tf.py` (vs original TF graph) | 9 passed, 52 s (5 `layer.apply` deprecation warnings from TF) |

This also extends the verified range: the tests now pass on Python 3.11 / torch 2.11 / TF 2.21, not only the 3.13 / 2.14 combination in [[AU-GAN Port]]. They ran on CPU.

## Run: GAN smoke training and inference
There is no dataset on disk (`datasets/` absent). I generated 16 synthetic image pairs per split (random blobs; A darkened ×0.2 as fake night), kept outside the repo, so this checks that the code runs, **not** that it learns.

| Run | Result |
|---|---|
| pt-AUGAN (`pt_augan_bdd100k.yaml`), 256×256, **batch 1**, CUDA | trains; checkpoints (`last.pt`, `weights_epoch00.pt`) and sample images written; about 0.4 s/iter |
| AU-GAN original (`augan_bdd100k.yaml`), 128×128, batch 2, CUDA | trains, 2 epochs, losses finite |
| pt-AUGAN, 256×256, batch 2 or 4 | **CUDA OOM** |
| AU-GAN original, 256×256, batch 1 | **CUDA OOM** (needs more than about 6.3 GB; still OOM with `expandable_segments:True`) |
| `monovision.enhance.infer` with the pt-AUGAN checkpoint on 16 images | 16 outputs, same filenames and native 320×320 size |

Takeaways:
- The configs' `batch_size: 8` at 256×256 will **not fit** on this GPU. The paper used a V100 (16–32 GB). Plan for batch 1–2 on pt-AUGAN, or a smaller `fine_size`, or a bigger GPU. The original AU-GAN architecture (deconv, no demod) uses noticeably more memory than pt-AUGAN here.
- The training loop, image pool, checkpointing, LR schedule hook and inference path all execute end to end on GPU.
- Not tested: real data, convergence, `--init_from` conversion of a TF checkpoint, or `--continue_train`.

## Not run
- **SimpleRecon** (superseded: it now runs, see [[SimpleRecon Port]]): submodule present (`third_party/simplerecon`, includes `weights/` dir and `simplerecon_env.yml`), but its Python 3.9 / torch 1.10 / CUDA 11.3 env was not built. CUDA 11.3 has no sm_120 support, so it **likely cannot run on this GPU** as pinned. Needs a newer torch/open3d or a different machine. See [[Open Questions]].
- **`monovision run`, capture, data, eval**: no code yet (the `monovision/{capture,data,recon,eval}` packages contain only `__init__.py`). `pyproject.toml` defines no console script.
- **Data**: BDD100K and ScanNet not downloaded.

## Follow-ups
Added to the TODO list on [[Home]].
