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
