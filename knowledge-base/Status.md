---
tags: [status]
updated: 2026-10-21
---
# Status

| Phase | Component | State |
|---|---|---|
| 1 | Repo, environment, submodules | scaffolded (`scripts/00_setup.sh`, `docs/SETUP.md`, `scripts/check_env.py`) |
| 2 | Data: BDD100K split, ScanNet, ios_logger, synthetic night | BDD100K **10k** split done by brightness heuristic ([[Decisions and Assumptions]]); rest not started |
| 3 | SimpleRecon baselines + mesh evaluation | SimpleRecon **inference ported and running; training ported on branch `simplerecon-training`** ([[SimpleRecon Port]]); baselines and mesh evaluation not started |
| 4 | AU-GAN PyTorch port + pt-AUGAN | **done and verified against the original TF graph; not yet trained** |
| 5 | Pipeline glue (`monovision run`) | not started |
| 6 | Evaluation (FID, P/R/F, significance) | not started |

## What is verified
- Port matches the TF graph: outputs, losses, gradients (float64), legacy bicubic. Details in [[AU-GAN Port]].
- 17 TF-free tests plus the parity suite pass (CPU, Python 3.13, torch 2.14).

## What is not
- Training on real data or to convergence. A short GPU smoke run on synthetic images works ([[Build and Run Results]]).
- The pretrained BDD100K checkpoint (deliberately not downloaded).
- Python 3.9 / torch 1.10, Windows.
- Port parity with the original SimpleRecon env (cannot run here).

## Git
Branch `main`, one commit. Staged: `.gitmodules`, the two submodules. Modified: `.gitignore`, `README.md`. Untracked: all project code and docs.

Next steps live in the TODO list on [[Home]].
