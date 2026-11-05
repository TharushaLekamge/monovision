import numpy as np
import pytest

from monovision.eval import fid, mesh, stats


def test_frechet_zero_for_same_gaussian():
    f = np.random.default_rng(0).normal(size=(500, 8))
    assert fid.fid_from_features(f, f) == pytest.approx(0, abs=1e-6)


def test_frechet_mean_shift():
    f = np.random.default_rng(0).normal(size=(2000, 4))
    assert fid.fid_from_features(f, f + 2.0) == pytest.approx(16.0, rel=1e-3)  # 4 dims * 2^2


def test_point_metrics_identical_and_shifted():
    pts = np.random.default_rng(0).uniform(size=(1000, 3))
    m = mesh.point_metrics(pts, pts)
    assert m["fscore"] == 1.0 and m["acc"] == 0 and m["comp"] == 0
    far = mesh.point_metrics(pts + 10, pts)
    assert far["prec"] == 0 and far["recall"] == 0 and far["fscore"] == 0


def test_point_metrics_partial_recall():
    gt = np.stack([np.linspace(0, 1, 1000), np.zeros(1000), np.zeros(1000)], 1)
    pred = gt[:500]  # perfect precision, half the surface
    m = mesh.point_metrics(pred, gt, thresh=0.001)
    assert m["prec"] == 1.0 and 0.49 < m["recall"] < 0.52


def test_mesh_metrics_from_file(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    p = tmp_path / "box.ply"
    trimesh.creation.box((1, 1, 1)).export(p)
    assert mesh.mesh_metrics(p, p, n=2000)["fscore"] == 1.0


def test_paired_test():
    rng = np.random.default_rng(0)
    b = rng.uniform(0.3, 0.6, 30)
    r = stats.paired_test(b + 0.1 + rng.normal(0, 0.01, 30), b)
    assert r["mean_diff"] == pytest.approx(0.1, abs=0.01)
    assert r["p_ttest"] < 1e-6 and r["p_wilcoxon"] < 1e-3 and r["ci95"][0] > 0
    assert stats.paired_test(b, b)["p_wilcoxon"] == 1.0
    with pytest.raises(ValueError):
        stats.paired_test([1, 2], [1])
