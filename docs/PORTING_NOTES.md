# AU-GAN → PyTorch: porting notes

Source: https://github.com/jgkwak95/AU-GAN (TensorFlow 1.14). Files ported: `models.py`, `ops.py`, `loss_utils.py`, `AUGAN.py`, `utils.py` (data loading and image pool). Port: `monovision/enhance/`.

## How it was verified

`tests/test_parity_tf.py` builds the **original** TF graph from `third_party/AU-GAN` under TF2's v1-compat mode. `tests/tf_reference.py` applies import-only patches; the maths is unchanged. The test fills every variable with random values, saves a TF checkpoint, converts it with `convert_tf.py`, and compares the two implementations. No pretrained weights are needed.

| Test | What is compared | Tolerance | Result |
|---|---|---|---|
| `test_generator_and_losses_match` | G_A2B outputs (translation, reconstruction, r4/t4 features, confidence map), G_B2A output, total G loss, perceptual/adversarial/cycle parts, both D losses (float32, 32×64) | 2e-4 abs on outputs, 1e-4 on losses | pass |
| `test_generator_odd_sizes` | generator at 36×52, 30×50, 33×47 (TF asymmetric SAME padding and transpose-conv cropping) | 2e-4 | pass |
| `test_gradients_match` | dL/dW for 7 weights across G_A2B, G_B2A and both discriminators, in **float64** | 1e-9 (relative to the largest gradient) | pass |
| `test_legacy_bicubic` | TF1 `resize_bicubic` (legacy mode) at 2× and non-integer factors | 1e-5 | pass |
| `test_bicubic_2x_is_subsampling` | TF 2× bicubic equals `x[::2, ::2]` in both the forward and backward pass | exact | pass |

Why gradients are checked in float64: in float32, the first and last generator layers differed from a float64 reference by about 1e-3 (TF) and 3e-4 (PyTorch). That is round-off; TF's oneDNN CPU kernels sum in a different order. It was large enough to hide or imitate a real bug. In float64 the two implementations agree to 1e-9.

TF-free tests (`tests/test_port.py`, 17 tests) cover:
- output shapes and the pt-AUGAN switches;
- demodulation producing unit-norm filters;
- train-only noise;
- gradients reaching every generator parameter;
- config and boolean parsing;
- a 3-iteration train → checkpoint → folder-inference smoke run for both the original and pt-AUGAN variants.

Run both suites with `scripts/run_tests.sh`. They run as two separate pytest processes, because running TF and PyTorch in one process crashed natively in the test environment; the cause hasn't been investigated.

**Not verified:**
- Training to convergence; no GPU was available.
- The pretrained BDD100K checkpoint itself; deliberately not downloaded.
- Running on torch 1.10 / Python 3.9, the SimpleRecon environment. Tests ran on Python 3.13 / torch 2.14 (CPU); the code avoids newer APIs.

## TF behaviours reproduced explicitly

- **'SAME' padding** is asymmetric when the total is odd. For example, a stride-2 3×3 conv on an even input pads (0, 1), not (1, 1). Implemented in `layers.tf_same_pad_amounts`.
- **conv2d_transpose 'SAME'** gives out = in × s and is not `ConvTranspose2d(padding=1, output_padding=1)`. The port runs an unpadded transpose conv, then crops at the matching forward-pad offset.
- **"Dilated" residual blocks** use rate 1 upstream, so they're plain 3×3 convs.
- **No conv biases** anywhere (slim `biases_initializer=None`).
- **Instance norm**: biased variance, eps 1e-5. It's hand-written because `nn.InstanceNorm2d` raises on 1×1 maps in train mode, whereas TF returns the offset.
- **Discriminator pyramid** uses TF1 legacy bicubic: a = −0.75, no half-pixel centres, 1024-entry weight table. At 2× this equals strided subsampling.
- **Discriminator cross-scale wiring** is reproduced as upstream: at each level, branch 0 reads branch 1 through a stride-1 conv, and branch 1 reads branch 0 through two stride-2 convs.
- **Image pool** quirk reproduced: it stores whole (fake_A, fake_B) batch pairs and swaps A and B from independent random slots.
- **Weight layout**: TF conv `[kh,kw,in,out]` and TF transpose `[kh,kw,out,in]` both map with `permute(3,2,0,1)`; no kernel flip.

## Deliberate differences from upstream

- **Reconstruction-branch noise** (N(0, 0.02)) is applied only in `train()` mode. Upstream also applies it at test time. It doesn't affect the night→day output.
- **Boolean flags** parse correctly. Upstream's argparse `type=bool` makes `--use_uncertainty False` evaluate to True.
- **Command name**: the upstream README says `main_uncer.py`, but only `main.py` exists. Here it's `python -m monovision.enhance.train`.
- **Aspect ratio** is a flag. Upstream trains at 256×512 (`--aspect 2`); the MonoVision paper reports 256×256 (`--aspect 1`).

## pt-AUGAN (MonoVision paper §III-B)

The paper names two changes; their exact placement is not stated, so both are switches:
- `--upsample upconv`: ×2 upsample (bilinear by default; `--up_mode nearest` available) followed by a 3×3 conv, in place of every transposed conv (both decoders and the confidence head). **[assumption: placement and interpolation mode]**
- `--demod true`: StyleGAN2-style weight demodulation on every generator conv. AU-GAN has no style vector, so this reduces to per-filter weight normalisation. **[assumption: that this is what was intended]**

Either switch changes the weights, so a model trained this way can't load the original TF checkpoint. `convert_tf.py` supports the original architecture only.
