---
tags: [pipeline]
updated: 2026-10-21
---
# Pipeline

1. **Capture**: `ios_logger` (iOS app, built separately with Xcode) records images + poses. Code target: `monovision/capture` (not started).
2. **Enhance**: night→day translation, `monovision.enhance.infer`. Filenames are preserved so camera poses stay aligned.
3. **Reconstruct**: SimpleRecon predicts depth and fuses with TSDF into a mesh. Runs in its own env ([[Environment]]).
4. **Evaluate**: FID for the GAN; precision / recall / F-score on meshes; significance tests (`monovision/eval`, not started).

The two envs exchange image folders on disk, so there is no cross-env import.

Data conventions: ForkGAN/AU-GAN layout, `A` = night, `B` = day: `datasets/bdd100k/{trainA,trainB,testA,testB}/*.jpg`.

Test set: 16 ScanNet scans in `data_splits/scannet16.txt` (`_00` suffix is an assumption).

Commands: see the Quickstart in the root `README.md`.
