"""Regression tests for patches/simplerecon (SimpleRecon on current torch / lightning).

They exercise our patches, not the libraries: each test failed before the patch that fixes it.
SimpleRecon runs in a subprocess because its top-level packages (utils, tools, ...) would
otherwise leak into this process.
"""

import numpy as np
import pytest

from conftest import CHECKPOINT, REPO


def test_setup_script_with_pinned_submodule_applies_every_patch(simplerecon_build):
    _, stdout = simplerecon_build
    patches = sorted(p.name for p in (REPO / "patches" / "simplerecon").glob("*.patch"))
    applied = [line.split()[1] for line in stdout.splitlines() if line.startswith("applied ")]
    assert applied == patches


def test_save_viz_video_frames_with_rgb_frames_writes_video(run_in_simplerecon, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    out = tmp_path / "viz.mp4"
    result = run_in_simplerecon(
        "import numpy as np\n"
        "from utils.visualization_utils import save_viz_video_frames\n"
        f"frames = [np.full((64, 64, 3), 40 * i, np.uint8) for i in range(4)]\n"
        f"save_viz_video_frames(frames, r'{out}', fps=4)\n"
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert out.stat().st_size > 0


def test_to_mesh_with_single_mesh_requested_falls_back_and_warns(run_in_simplerecon):
    result = run_in_simplerecon(
        "import warnings, torch\n"
        "from tools.tsdf import TSDF\n"
        "bounds = {f'{a}{m}': v for a in 'xyz' for m, v in (('min', -1.0), ('max', 1.0))}\n"
        "tsdf = TSDF.from_bounds(bounds, voxel_size=0.1)\n"
        "n = tsdf.tsdf_values.shape\n"
        "ax = torch.linspace(-1, 1, n[0]).half()\n"
        "x, y, z = torch.meshgrid(ax, ax, ax, indexing='ij')\n"
        "tsdf.tsdf_values = (torch.sqrt(x**2 + y**2 + z**2) - 0.5).clamp(-1, 1)\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    mesh = tsdf.to_mesh(export_single_mesh=True)\n"
        "assert len(mesh.vertices) > 0\n"
        "assert any('single_mesh' in str(w.message) for w in caught)\n"
    )
    assert result.returncode == 0, result.stderr[-500:]


def test_load_from_checkpoint_with_stripped_released_checkpoint_restores_weights(run_in_simplerecon):
    if not CHECKPOINT.exists():
        pytest.skip("run scripts/01_setup_simplerecon.sh --weights")
    result = run_in_simplerecon(
        f"import torch, options\n"
        f"from experiment_modules.depth_model import DepthModel\n"
        f"path = r'{CHECKPOINT}'\n"
        f"model = DepthModel.load_from_checkpoint(path, args=None)\n"
        f"saved = torch.load(path, map_location='cpu')['state_dict']\n"
        f"assert set(saved) == set(model.state_dict())\n"
        f"key = next(iter(saved))\n"
        f"assert torch.equal(saved[key], model.state_dict()[key])\n"
    )
    assert result.returncode == 0, result.stderr[-500:]


def test_scalable_tsdf_volume_with_flat_wall_returns_nonempty_mesh():
    """open3d 0.20.0 returned an empty mesh here; requirements-recon.txt pins 0.19.0."""
    o3d = pytest.importorskip("open3d")
    integration = o3d.pipelines.integration
    height, width = 96, 128
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 100, 100, width / 2, height / 2)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.full((height, width, 3), 128, np.uint8)),
        o3d.geometry.Image(np.full((height, width), 2.013, np.float32)),
        depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False,
    )
    volume = integration.ScalableTSDFVolume(
        voxel_length=0.04, sdf_trunc=0.12, color_type=integration.TSDFVolumeColorType.RGB8)
    volume.integrate(rgbd, intrinsic, np.eye(4))
    assert len(volume.extract_triangle_mesh().vertices) > 0
