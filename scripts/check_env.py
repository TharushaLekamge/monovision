#!/usr/bin/env python
"""Check whether this machine/environment is ready for MonoVision. Prints a checklist.

  python scripts/check_env.py            # GAN env (monovision)
  python scripts/check_env.py --recon    # also check the SimpleRecon env + weights

Exit code 0 if every required item passes, 1 otherwise. Optional items never fail the run.
"""
import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OK, FAIL, WARN = "[ ok ]", "[FAIL]", "[warn]"
failures = []


def report(status, msg, fix=None):
    print(f"{status} {msg}")
    if fix and status != OK:
        print(f"       -> {fix}")
    if status == FAIL:
        failures.append(msg)


def version_of(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def check_python():
    v = sys.version_info
    ok = v >= (3, 9)
    report(OK if ok else FAIL, f"Python {platform.python_version()} ({sys.executable})",
           "Python >= 3.9 required; see docs/SETUP.md step 2")


def check_core():
    for mod, pip_name in [("numpy", "numpy"), ("PIL", "pillow"), ("yaml", "pyyaml")]:
        v = version_of(mod)
        report(OK if v else FAIL, f"{pip_name} {v or 'missing'}", f"pip install {pip_name}")
    v = version_of("monovision")
    report(OK if v else FAIL, f"monovision package {v or 'not importable'}", "pip install -e .  (from the repo root)")


def check_torch(require_gpu):
    v = version_of("torch")
    if not v:
        report(FAIL, "torch missing", "install from https://pytorch.org/get-started/locally/ (see docs/SETUP.md step 3)")
        return
    import torch

    report(OK, f"torch {v} (built for CUDA {torch.version.cuda or 'none / CPU-only'})")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            name = torch.cuda.get_device_name(i)
            arch = f"sm_{cap[0]}{cap[1]}"
            built = torch.cuda.get_arch_list()
            supported = arch in built or any(a.startswith("compute_") and int(a.split("_")[1]) <= cap[0] * 10 + cap[1]
                                             for a in built)
            report(OK if supported else WARN, f"GPU {i}: {name} ({arch}); this torch build targets {', '.join(built)}",
                   "this torch build may not support your GPU; install a newer CUDA build of torch")
            try:
                (torch.ones(2, device=f"cuda:{i}") * 2).sum().item()
                report(OK, f"GPU {i}: test kernel ran")
            except Exception as e:  # noqa: BLE001
                report(FAIL, f"GPU {i}: test kernel failed: {e}", "driver/torch CUDA mismatch; see docs/SETUP.md")
    else:
        status = FAIL if require_gpu else WARN
        report(status, "no CUDA GPU visible to torch (CPU only: fine for tests, too slow for training)",
               "check `nvidia-smi`, then install a CUDA build of torch (docs/SETUP.md step 3)")


def check_tools():
    report(OK if shutil.which("git") else FAIL, f"git {'found' if shutil.which('git') else 'missing'}",
           "install git (https://git-scm.com)")
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            report(OK, f"nvidia-smi: {out.splitlines()[0] if out else 'no GPUs listed'}")
        except Exception as e:  # noqa: BLE001
            report(WARN, f"nvidia-smi failed: {e}")
    else:
        report(WARN, "nvidia-smi not found (no NVIDIA driver, or not on PATH)",
               "install the NVIDIA driver for GPU training/reconstruction")


def check_submodules():
    for name, marker in [("AU-GAN", "models.py"), ("simplerecon", "test.py")]:
        p = os.path.join(ROOT, "third_party", name, marker)
        report(OK if os.path.isfile(p) else FAIL, f"third_party/{name} {'present' if os.path.isfile(p) else 'missing'}",
               "run scripts/00_setup.sh (or `git submodule update --init`)")


def installed(mod):
    """Presence check without importing (importing TensorFlow is slow and noisy)."""
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def check_optional_tf():
    tf_v = "installed" if installed("tensorflow") else None
    slim_v = "installed" if installed("tf_slim") else None
    status = OK if tf_v and slim_v else WARN
    report(status, f"optional: tensorflow {tf_v or 'missing'}, tf-slim {slim_v or 'missing'} "
                   "(only for TF parity tests / converting TF checkpoints)",
           "pip install -r requirements-dev.txt")
    v = version_of("pytest")
    report(OK if v else WARN, f"optional: pytest {v or 'missing'}", "pip install pytest")


def check_recon():
    for mod in ["pytorch_lightning", "kornia", "open3d", "trimesh", "timm", "transforms3d", "skimage"]:
        v = version_of(mod)
        report(OK if v else FAIL, f"simplerecon dep {mod} {v or 'missing'}",
               "activate the `simplerecon` env (docs/SETUP.md step 5)")
    w = os.path.join(ROOT, "third_party", "simplerecon", "weights", "hero_model.ckpt")
    report(OK if os.path.isfile(w) else FAIL, f"SimpleRecon weights {'found' if os.path.isfile(w) else 'missing'}: {w}",
           "download hero_model.ckpt per the SimpleRecon README into third_party/simplerecon/weights/")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recon", action="store_true", help="also check SimpleRecon env and weights")
    p.add_argument("--require_gpu", action="store_true", help="treat missing CUDA GPU as a failure")
    a = p.parse_args()
    print(f"MonoVision environment check  ({platform.system()} {platform.release()}, repo: {ROOT})\n")
    check_python()
    check_core()
    check_torch(a.require_gpu or a.recon)
    check_tools()
    check_submodules()
    check_optional_tf()
    if a.recon:
        check_recon()
    print()
    if failures:
        print(f"{len(failures)} required item(s) missing. See docs/SETUP.md.")
        sys.exit(1)
    print("All required items present.")


if __name__ == "__main__":
    main()
