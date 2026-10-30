---
tags: [setup]
updated: 2026-10-21
---
# Environment

**Update 2026-10-21:** SimpleRecon inference now runs in the `monovision` env too, so a second env is optional ([[SimpleRecon Port]]). Originally two environments, because SimpleRecon pins old versions (Python 3.9.7, PyTorch 1.10.0, CUDA 11.3, open3d 0.14.1).

| Env | Used for |
|---|---|
| `monovision` | GAN training/inference, tests. Python ≥ 3.9, current PyTorch |
| `simplerecon` | SimpleRecon depth + TSDF fusion |

Files: `environment.yml`, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`.
Check readiness: `python scripts/check_env.py [--recon]`. Full guide: `docs/SETUP.md`.

Tested: `monovision` code on Linux, Python 3.13, torch 2.14, CPU.
Untested: Windows (use WSL2), GPUs, Python 3.9 / torch 1.10, the `simplerecon` env steps.
GPU needed for training; the paper used one V100 at about 10 h/epoch.
Host (this machine): Fedora, Linux 7.0.

## Observed on this machine (2026-10-21)
`~/.conda/envs/monovision`: Python 3.11.17, torch 2.14.1+cu130 (was 2.11.0+cu128 until installing torchvision upgraded it), TF 2.21 CPU, open3d 0.19.0. GPU: RTX 5060 Laptop, 8 GB. System `python` is 3.14 without torch. Details: [[Build and Run Results]].
