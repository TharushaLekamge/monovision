"""Mesh accuracy / completeness / precision / recall / F-score from sampled surface points.

acc  = mean distance pred -> gt      comp   = mean distance gt -> pred
prec = share of pred points within `thresh` of gt     recall = share of gt points within `thresh` of pred
Distances in metres, default threshold 5 cm (the ScanNet convention). Not implemented: masking out
gt regions the camera never saw (SimpleRecon's evaluation does this), so completeness is pessimistic.
"""
import numpy as np
from scipy.spatial import cKDTree


def sample_mesh(path, n=200000, seed=0):
    import trimesh

    m = trimesh.load(path, force="mesh")
    if len(m.faces) == 0:
        return np.asarray(m.vertices)
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(pts)


def point_metrics(pred, gt, thresh=0.05):
    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    prec, rec = float((d_pg < thresh).mean()), float((d_gp < thresh).mean())
    return {"acc": float(d_pg.mean()), "comp": float(d_gp.mean()), "prec": prec, "recall": rec,
            "fscore": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def mesh_metrics(pred_path, gt_path, thresh=0.05, n=200000, seed=0):
    return point_metrics(sample_mesh(pred_path, n, seed), sample_mesh(gt_path, n, seed), thresh)
