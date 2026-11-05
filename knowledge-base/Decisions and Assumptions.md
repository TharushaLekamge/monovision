---
tags: [decisions]
updated: 2026-10-21
---
# Decisions and Assumptions

| Item | Choice | Basis |
|---|---|---|
| Port via submodule, no AU-GAN code copied | yes | AU-GAN has no licence |
| Two conda envs | yes | SimpleRecon pins old versions |
| pt-AUGAN upsample | `--upsample upconv`: bilinear ×2 + 3×3 conv replacing every transposed conv (decoders and confidence head); `--up_mode nearest` available | **assumption**: paper names the change but not placement or mode |
| pt-AUGAN demodulation | `--demod true`: per-filter weight normalisation (no style vector in AU-GAN) | **assumption** about intent |
| Aspect ratio | flag; paper uses 256×256 (`--aspect 1`) | upstream default is 256×512 |
| Test set suffix | `_00` appended to the 16 scan IDs | **assumption** |
| Noise at inference | off | deliberate difference; no effect on night→day output |

Models trained with either pt-AUGAN switch cannot load the original TF checkpoint; `convert_tf.py` supports the original architecture only.

## BDD100K 10k split (2026-10-30)
- Only `raw_data/bdd100k_images_10k.zip` is available (gitignored, `raw_data/`); the 100K images are not downloaded. The zip has no labels, so there is no `timeofday` attribute.
- Assumption: night = mean luma < 50, day = mean luma >= 80 on a 160×90 thumbnail; 50–80 (dawn/dusk, ambiguous) is dropped. Luma is bimodal on the train set, so these cut points sit in the valley. A dark daytime tunnel could land in A; not checked by eye.
- BDD100K `val` becomes `test`; the unlabelled `test` folder is unused.
- Result is small for A (248 train / 29 test night), so GAN training on it is a smoke/dev run, not a paper-scale run.
- Output goes to `datasets/bdd100k` (gitignored); the per-image decision is in `data_splits/bdd100k_10k_split.csv` (committed).
- With the full dataset: `python -m monovision.data.bdd100k --zip raw_data/bdd100k_images_100k.zip --labels raw_data/bdd100k_labels.zip --out datasets/bdd100k_100k --manifest data_splits/bdd100k_100k_split.csv` uses the real `timeofday` (labels zip = one JSON per image). Result: train 28,028 night / 36,800 day, test 3,929 / 5,258; dawn/dusk and untagged skipped. The 100K manifest (2.4 MB) is gitignored.
