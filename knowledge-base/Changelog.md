---
tags: [log]
---
# Changelog

Newest first. One line per meaningful change; link the relevant note.

## 2026-11-05
- Added `monovision/eval` (FID, mesh P/R/F, paired significance tests, CLI) with tests. First FID: 10k model 288, 100K model 188, untranslated night 114. See [[Evaluation]].
- Built the BDD100K 100K night/day split from real `timeofday` labels and trained pt-AUGAN for 12k iterations. See [[Decisions and Assumptions]].

## 2026-10-21
- Addressed python-pr-review: fixed moviepy call, hardened setup script (weights kept, checksum, `patch` instead of `git apply`), lighter loader, pins, 5 new tests. Patches regenerated (0005 on the training branch). See [[SimpleRecon Port]].
- Ported SimpleRecon training to lightning 2 on worktree branch `simplerecon-training` (patch 0005); resume, fine-tune and batch 1–2 smoke runs pass. See [[SimpleRecon Port]].
- Ported SimpleRecon inference to Python 3.11 / torch 2.14 (4 patches, `scripts/01_setup_simplerecon.sh`, `requirements-recon.txt`). Found open3d 0.20.0 TSDF bug. See [[SimpleRecon Port]].
- First build/test/run: 26 tests pass, GPU smoke training and inference work, 8 GB VRAM limits batch size. SimpleRecon not run. See [[Build and Run Results]].
- Initialised the knowledge base from the repo state: README, `docs/`, `THIRD_PARTY.md`, git status. See [[Home]].

## 2026-10-10
- Repo scaffolded; submodules AU-GAN and SimpleRecon added at upstream HEAD ([[Third-Party and Licensing]]).
- AU-GAN PyTorch port finished and verified against the original TF graph ([[AU-GAN Port]]).
