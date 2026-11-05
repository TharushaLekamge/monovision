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
