---
tags: [evaluation]
updated: 2026-11-05
---
# Evaluation

Code: `monovision/eval/`. CLI: `python -m monovision.eval {fid,mesh,compare}`. Tests: `tests/test_eval.py` (synthetic data only).

| Piece | What | Caveat |
|---|---|---|
| `fid.py` | Frechet distance between two image folders | Features come from torchvision Inception-v3 (ImageNet weights, downloaded on first use), not the TF-Inception of reference FID. Numbers compare inside this project only. |
| `mesh.py` | acc, comp, precision, recall, F-score at 5 cm from points sampled on two meshes | No masking of gt regions the camera never saw (SimpleRecon's eval does this), so completeness is pessimistic. Not checked against SimpleRecon's numbers. |
| `stats.py` | paired t-test, Wilcoxon, bootstrap 95% CI of the mean difference | Inputs must be per-scan values in the same order. |
| `compare` CLI | paired test on one column of two CSVs that have an `id` column | |

## First FID numbers (2026-11-05)
500 BDD100K test night images translated to day, against 500 real test day images, 256×256 model, 500 samples per side (FID is biased high at this sample size).

| Images | FID vs real day |
|---|---|
| untranslated night (reference) | 113.6 |
| `pt_augan_bdd100k_10k_20ep` (248 night images, 4,960 iters) | 288.5 |
| `pt_augan_bdd100k_100k` (28k night images, 12,000 iters) | 188.0 |

- The 100K model is clearly closer to real day than the 10k model, matching what the samples look like.
- Both are **farther** from real day than the untouched night images are. So by this metric neither model is a good translator yet; the artifacts cost more than the brightening gains. Not investigated whether the generated-image resize (native size via `keep_size`) adds to this.
- One seed, one 500-image sample, no confidence interval.

## Why is generated FID worse than untranslated night? (2026-11-05)
Controls at 256×256, 500 vs 500 images, same Inception features:

| Comparison | FID |
|---|---|
| real day vs real day (different 500 images): noise floor | 45–46 |
| real day after a 256 round trip vs real day: cost of the resolution change | 48.5 (about +2) |
| untranslated night vs real day | 110–114 |
| 10k model output vs real day | 286 |
| 100K model output vs real day | 183 |

- It is **not** the metric setup: the noise floor is far below the gaps, and downscaling to 256 and back barely moves FID. The generated images really are farther from real day than raw night is, in Inception feature space. Visible GAN artifacts (smeared texture, blown-out sky) are the likely cause; this was not tested directly.
- FID across the 100K run's per-epoch checkpoints (2,000 iterations per epoch, 500 images, batch 1): 330, 280, 224, 279, 226, **183**. Falling overall but noisy (the epoch 3 rise shows GAN instability), and the lowest value is at the end of the learning-rate decay. Longer training looks worthwhile, but nothing here shows it reaches the 110 of raw night.
- Single seed, 500 images per side; differences of about 50 between neighbouring checkpoints are within the noise seen here.

## 128×128 vs 256×256 (2026-11-05)
`pt_augan_bdd100k_100k_128`: batch 4 (batch 8 OOMs), 14,400 iterations (about 57,600 images, 108 min); the 256 run was 12,000 images in 155 min. FID at 128, 500 test night images vs 500 real day images resized to 128:

| | FID |
|---|---|
| real day vs real day (floor) | 48.6 |
| untranslated night | 119.5 |
| 128 model per epoch (2,400 iters each) | 290, 250, 274, 175, 201, **174** |
| 256 model final (not strictly comparable) | 183 |

- FID is about the same for both (174 vs 183, inside the checkpoint-to-checkpoint noise of about 50), and 128 is also noisy epoch to epoch.
- Visually 128 is worse on the four held-out samples: blurrier, and it invents structure (extra yellow lane lines, blobs) so the scene geometry is less faithful. That matters for reconstruction, so the longer run uses 256.
- Cross-resolution FID compares features of upsampled 128 images with 256 ones; treat the 174 vs 183 as indicative only.
