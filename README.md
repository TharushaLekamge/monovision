# MonoVision

Reproduction repo for *Monocular 3D Reconstruction in Poorly Visible Environments* (Ilesinghe, Lekamge, Samarutilake, IJRC 3(1), 2024, pp. 27–34).

Pipeline: `ios_logger` capture → frames → **night→day GAN** (AU-GAN / pt-AUGAN) → **SimpleRecon** (depth + TSDF fusion) → mesh.

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | Repo, environment, submodules | scaffolded (`scripts/00_setup.sh`, `docs/SETUP.md`, `scripts/check_env.py`) |
| 2 | Data: BDD100K split, ScanNet, ios_logger, synthetic night | BDD100K 10k split done (`monovision.data.bdd100k`); rest not started |
| 3 | SimpleRecon baselines + mesh evaluation | not started |
| 4 | **AU-GAN PyTorch port + pt-AUGAN** | **done, verified against the original TF graph** (see `docs/PORTING_NOTES.md`); not yet trained |
| 5 | Pipeline glue (`monovision run`) | not started (`monovision/recon`) |
| 6 | Evaluation (FID, P/R/F, significance) | code done, tested on synthetic data (`monovision/eval`); no real mesh results yet |

## Setup

Full instructions, including Windows/WSL2, GPU drivers and the separate SimpleRecon env: **[docs/SETUP.md](docs/SETUP.md)**.

Check whether a machine is already set up:

```bash
python scripts/check_env.py            # GAN env
python scripts/check_env.py --recon    # SimpleRecon env
```

Short version (Linux or WSL2, conda installed):

```bash
bash scripts/00_setup.sh                  # git init + AU-GAN / SimpleRecon submodules
conda env create -f environment.yml && conda activate monovision
pip install torch                         # or the CUDA command from pytorch.org's selector
pip install -e . -r requirements-dev.txt  # TF only needed for parity tests / checkpoint conversion
bash scripts/run_tests.sh
```

## Quickstart (GAN)

```bash
# data layout (ForkGAN/AU-GAN convention; A = night, B = day)
#   datasets/bdd100k/{trainA,trainB,testA,testB}/*.jpg

# train: original AU-GAN or the paper's pt-AUGAN variant
python -m monovision.enhance.train --config configs/gan/augan_bdd100k.yaml
python -m monovision.enhance.train --config configs/gan/pt_augan_bdd100k.yaml

# translate a folder of frames (filenames kept, so camera poses stay aligned)
python -m monovision.enhance.infer --weights checkpoints/pt_augan_bdd100k/last.pt \
    --input <scan>/images --output <scan_day>/images

# optional: convert an original AU-GAN TF checkpoint (needs only TF 2.x, not 1.14)
python -m monovision.enhance.convert_tf --tf_ckpt <dir with cyclegan.model-*> --out augan_torch.pt
python -m monovision.enhance.train --config configs/gan/augan_bdd100k.yaml --init_from augan_torch.pt
```

`--init_from` loads converted weights into the original architecture only (`upsample: deconv`, `demod: false`).

## Layout

```
monovision/enhance/   AU-GAN port: layers, models, losses, data, train, infer, convert_tf
monovision/data/       BDD100K night/day split builder
monovision/eval/       FID, mesh precision/recall/F-score, paired significance tests
monovision/{capture,recon}/   later phases (placeholders)
configs/gan/          augan_bdd100k.yaml (upstream defaults), pt_augan_bdd100k.yaml (paper)
data_splits/          scannet16.txt — the paper's 16 test scans (_00 suffix assumed)
tests/                test_port.py (no TF), test_parity_tf.py + tf_reference.py (vs original TF)
third_party/          AU-GAN, SimpleRecon (git submodules; not vendored)
docs/SETUP.md         environment setup (both envs), troubleshooting
docs/PORTING_NOTES.md what was verified, TF quirks reproduced, deliberate differences
scripts/check_env.py  checks whether an env is ready
```

## Licensing

There's no LICENSE file yet; choose one before publishing. Constraints:
- SimpleRecon is under a non-commercial Niantic licence.
- AU-GAN ships without a licence, so its code is used here via a submodule and not copied.
- BDD100K and ScanNet have their own terms of use.

Keep this repo academic / non-commercial.

## Citation

```
I. T. A. Ilesinghe, N. L. N. T. Lekamge, G. D. N. N. Samarutilake, "Monocular 3D Reconstruction
in Poorly Visible Environments," International Journal of Research in Computing, vol. 3, no. 1,
pp. 27-34, 2024.
```
(Author order as printed on the PDF; the journal's article page lists the authors in reverse order.)
