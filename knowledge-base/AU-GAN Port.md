---
tags: [gan, verification]
updated: 2026-10-21
---
# AU-GAN Port

PyTorch port of [AU-GAN](https://github.com/jgkwak95/AU-GAN) (TensorFlow 1.14) in `monovision/enhance/`. Full write-up: `docs/PORTING_NOTES.md`.

## Verification
`tests/test_parity_tf.py` builds the original TF graph from the submodule under TF2 v1-compat, fills it with random weights, converts the checkpoint, and compares:
- generator outputs and all losses (2e-4 / 1e-4)
- odd input sizes (TF asymmetric SAME padding)
- gradients in float64 (1e-9); float32 was too noisy and could hide real bugs
- legacy TF1 bicubic resize

TF and PyTorch suites run as separate pytest processes (`scripts/run_tests.sh`) because one process crashed natively; cause not investigated.

## TF behaviours reproduced
Asymmetric SAME padding, transpose-conv cropping, no conv biases, hand-written instance norm, legacy bicubic discriminator pyramid, upstream cross-scale wiring, image-pool quirk, weight layout `permute(3,2,0,1)`.

## Deliberate differences
- Reconstruction-branch noise only in train mode
- Boolean flags parse correctly (upstream `type=bool` bug)
- Entry point is `python -m monovision.enhance.train`
- Aspect ratio is a flag (upstream 256×512, paper 256×256)

## Not verified
Training to convergence, the pretrained checkpoint, torch 1.10 / Python 3.9.
