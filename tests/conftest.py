"""Shared fixtures. The SimpleRecon build is created once per session in a temp dir."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "build" / "weights" / "hero_model.ckpt"


@pytest.fixture(scope="session")
def simplerecon_build(tmp_path_factory):
    """Runs scripts/01_setup_simplerecon.sh into a temp dir; returns (build_dir, script_stdout)."""
    if not (REPO / "third_party" / "simplerecon" / "test.py").exists():
        pytest.skip("SimpleRecon submodule not checked out (scripts/00_setup.sh)")
    build_dir = tmp_path_factory.mktemp("simplerecon") / "simplerecon"
    result = subprocess.run(
        ["bash", str(REPO / "scripts" / "01_setup_simplerecon.sh")],
        env={**os.environ, "SIMPLERECON_BUILD_DIR": str(build_dir)},
        capture_output=True, text=True, check=True,
    )
    return build_dir, result.stdout


@pytest.fixture
def run_in_simplerecon(simplerecon_build):
    """Runs a python snippet with the patched SimpleRecon as the working directory."""
    build_dir, _ = simplerecon_build

    def run(code):
        return subprocess.run(
            [sys.executable, "-W", "ignore", "-c", code],
            cwd=build_dir, env={**os.environ, "PYTHONPATH": str(build_dir)},
            capture_output=True, text=True,
        )
    return run
