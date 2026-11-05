"""Frechet Inception Distance between two image folders.

Features are the 2048-d pooled output of torchvision's ImageNet Inception-v3, not the TF-Inception
used by the reference FID implementations, so values are comparable inside this project only,
not with numbers from papers. Use the same sample count and resize on both sides.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import linalg

IMG_EXT = {".jpg", ".jpeg", ".png"}


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        off = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + off).dot(sigma2 + off))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid_from_features(f1, f2):
    f1, f2 = np.asarray(f1, np.float64), np.asarray(f2, np.float64)
    return frechet_distance(f1.mean(0), np.cov(f1, rowvar=False), f2.mean(0), np.cov(f2, rowvar=False))


def list_images(folder, limit=None):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)
    return files[:limit] if limit else files


def inception_features(folder, limit=None, batch_size=50, device=None):
    import torch
    import torchvision

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = torchvision.models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True, transform_input=False)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    files = list_images(folder, limit)
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    feats = []
    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            x = np.stack([np.asarray(Image.open(f).convert("RGB").resize((299, 299), Image.BICUBIC)) for f in files[i:i + batch_size]])
            x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255.0
            feats.append(net((x - mean) / std).cpu().numpy())
    return np.concatenate(feats)


def fid(folder_a, folder_b, limit=None, **kw):
    return fid_from_features(inception_features(folder_a, limit, **kw), inception_features(folder_b, limit, **kw))
