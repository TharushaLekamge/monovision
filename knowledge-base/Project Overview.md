---
tags: [project]
updated: 2026-10-21
---
# Project Overview

**MonoVision** is a reproduction repo for *Monocular 3D Reconstruction in Poorly Visible Environments* (Ilesinghe, Lekamge, Samarutilake, IJRC 3(1), 2024, pp. 27–34).

Idea: monocular 3D reconstruction degrades in low light. Translate night frames to day with a GAN first, then reconstruct.

Pipeline summary (detail in [[Pipeline]]):
`ios_logger` capture → frames → night→day GAN (AU-GAN / pt-AUGAN) → SimpleRecon (depth + TSDF fusion) → mesh.

The paper's own GAN variant is **pt-AUGAN**: AU-GAN with upsample+conv instead of transposed conv, and weight demodulation. See [[Decisions and Assumptions]].

Repo state: git repo with a single "Initial commit" on `main`. Most work is still uncommitted. See [[Status]].

Layout: see `README.md` in the repo root. Key paths:
- `monovision/enhance/`: the AU-GAN port ([[AU-GAN Port]])
- `monovision/{capture,data,recon,eval}/`: placeholders for later phases
- `configs/gan/`: `augan_bdd100k.yaml`, `pt_augan_bdd100k.yaml`
- `data_splits/scannet16.txt`: the paper's 16 ScanNet test scans
- `third_party/`: submodules ([[Third-Party and Licensing]])
- `docs/SETUP.md`, `docs/PORTING_NOTES.md`
