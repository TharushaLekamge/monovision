# Environment setup

MonoVision uses **two environments**, because SimpleRecon pins old versions (Python 3.9.7, PyTorch 1.10.0, CUDA 11.3, open3d 0.14.1):

| Env | Used for | Python / PyTorch |
|---|---|---|
| `monovision` | GAN training and inference (`monovision.enhance`), tests | Python ≥ 3.9, current PyTorch |
| `simplerecon` | 3D reconstruction (SimpleRecon depth + TSDF fusion) | SimpleRecon's own pinned env |

The pipeline translates frames in `monovision`, then reconstructs in `simplerecon`. They exchange image folders on disk.

**What has been tested:**
- The `monovision` code and tests on Linux, Python 3.13, PyTorch 2.14, CPU only.
- Not yet tested: Windows, GPUs, Python 3.9 / PyTorch 1.10, and the steps in the `simplerecon` section below. Those steps follow SimpleRecon's README.

---

## 0. Already set up? Check first

From the repo root, inside the environment you want to check:

```bash
python scripts/check_env.py            # monovision env
python scripts/check_env.py --recon    # simplerecon env (also checks its packages and weights)
```

Every `[FAIL]` line names the step below that fixes it. If everything passes, skip to [§6 Verify](#6-verify).

## 1. Prerequisites

- **OS:** Linux is recommended. SimpleRecon's authors tested on Debian 10, and its env pins `clang`/`llvm-openmp`.
  - **Windows:** use **WSL2 with Ubuntu** (`wsl --install -d Ubuntu` in an admin PowerShell, then reboot). Run everything below inside WSL.
  - The `monovision` env alone is pure PyTorch and may work on native Windows, but that's untested.
- **NVIDIA driver** (GPU only): `nvidia-smi` should list your GPU. On WSL2, install the driver on Windows, not inside Linux.
  - A GPU is needed for training. The paper used one V100, at about 10 hours per epoch.
  - CPU is fine for the tests.
- **git:** `git --version`.
- **Conda:** [Miniforge](https://github.com/conda-forge/miniforge) is recommended, though any conda or mamba works. Check with `conda --version`.

Linux / WSL example:

```bash
sudo apt update && sudo apt install -y git
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b && ~/miniforge3/bin/conda init && exec $SHELL
```

## 2. Get the code and upstream submodules

```bash
cd ~ && unzip /path/to/monovision-repo.zip -d monovision && cd monovision
# On WSL, your Windows folder is under /mnt/c (e.g. /mnt/c/monovision/repo). Working inside the
# Linux filesystem (~/monovision) is much faster for datasets and training.
bash scripts/00_setup.sh          # git init + AU-GAN and SimpleRecon as submodules; prints their commits
```

Record the two printed commit hashes in `THIRD_PARTY.md`.

## 3. `monovision` env (GAN)

```bash
conda env create -f environment.yml
conda activate monovision
```

Install PyTorch. Use the selector at <https://pytorch.org/get-started/locally/> (Pip, your OS, your CUDA version) and run the command it gives. Typical cases:

```bash
pip install torch                 # Linux: the default PyPI wheel includes CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only (smaller)
```

On native Windows, the default PyPI wheel has historically been CPU-only, so take the CUDA command from the selector.

Then install the repo:

```bash
pip install -e .                       # the monovision package
pip install -r requirements-dev.txt    # optional: pytest + TensorFlow/tf-slim for parity tests
```

TensorFlow is only needed for `tests/test_parity_tf.py` and for converting original AU-GAN TF checkpoints (`convert_tf.py`). Any TF 2.x works; TF 1.14 is **not** needed.

## 4. Data folders (for training)

```
datasets/bdd100k/{trainA,trainB,testA,testB}/*.jpg     # A = night, B = day
```

Data preparation scripts are in a later phase (`monovision/data`, not written yet).

## 5. `simplerecon` env (reconstruction)

```bash
conda env create -f third_party/simplerecon/simplerecon_env.yml   # creates env "simplerecon"
conda activate simplerecon
pip install -e . --no-deps            # makes `monovision` importable here; deps already present
```

Download the SimpleRecon weights. Follow the link in `third_party/simplerecon/README.md` ("Models" section) and put `hero_model.ckpt` at `third_party/simplerecon/weights/hero_model.ckpt`.

**GPU note:** PyTorch 1.10 with CUDA 11.3 predates recent GPU generations. If `check_env.py --recon` warns that your GPU architecture isn't covered by the torch build, SimpleRecon may not run as pinned. Check SimpleRecon's GitHub issues before changing versions; newer versions haven't been tested here.

If `conda env create` fails, SimpleRecon's README notes that a May 2023 update fixed pins for `llvm-openmp`, `clang` and `protobuf`. Make sure the submodule is on a recent commit.

## 6. Verify

```bash
conda activate monovision
python scripts/check_env.py
bash scripts/run_tests.sh             # 17 PyTorch tests, then 9 TF parity tests if TF is installed

conda activate simplerecon
python scripts/check_env.py --recon
```

Expected: `All required items present.` and all tests passing. A tiny end-to-end GAN smoke run, which also exercises training and inference, is part of `tests/test_port.py`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `bash: scripts/...sh: /bin/bash^M: bad interpreter` | Windows line endings. The repo's `.gitattributes` keeps `*.sh` as LF; for an existing checkout run `sed -i 's/\r$//' scripts/*.sh` |
| Native crash when running all tests in one `pytest` call | TF and PyTorch in one process crashed in our environment; `scripts/run_tests.sh` runs them separately |
| `torch.cuda.is_available()` is False | Driver missing (`nvidia-smi`), or a CPU-only torch wheel is installed. Reinstall torch with the selector's CUDA command |
| `third_party/... missing` | Run `bash scripts/00_setup.sh`, or `git submodule update --init` in an existing clone |
| `monovision package not importable` | Run `pip install -e .` from the repo root in the active env |
