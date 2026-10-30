"""Unpaired A/B image data, matching upstream utils.load_train_data / load_test_data.

Layout (ForkGAN/AU-GAN convention):  <root>/{trainA,trainB,testA,testB}/*.jpg
A = adverse domain (night), B = normal domain (day).

Upstream resizes to (load_size, load_size*aspect), random-crops (fine_size, fine_size*aspect),
random h-flip (same flip for A and B), scales to [-1, 1]. Upstream aspect is 2 (256x512);
the MonoVision paper reports 256x256 training, i.e. aspect=1.
Resize uses PIL bilinear, which is what scipy.misc.imresize used.
"""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*.*")) if p.lower().endswith(IMG_EXT))


def load_rgb(path, h, w):
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 127.5 - 1.0


def to_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class UnpairedDataset(Dataset):
    """Each epoch reshuffles A and B independently and pairs them by index (as upstream)."""

    def __init__(self, root, split="train", load_size=286, fine_size=256, aspect=2, train=True, seed=None):
        self.A = list_images(os.path.join(root, split + "A"))
        self.B = list_images(os.path.join(root, split + "B"))
        if not self.A or not self.B:
            raise FileNotFoundError(f"No images in {root}/{split}A or {split}B")
        self.load_size, self.fine_size, self.aspect, self.train = load_size, fine_size, aspect, train
        self.rng = random.Random(seed)
        self.reshuffle()

    def reshuffle(self):
        self.rng.shuffle(self.A)
        self.rng.shuffle(self.B)

    def __len__(self):
        return min(len(self.A), len(self.B))

    def __getitem__(self, i):
        fs, ls, a = self.fine_size, self.load_size, self.aspect
        if self.train:
            A = load_rgb(self.A[i], ls, ls * a)
            B = load_rgb(self.B[i], ls, ls * a)
            h1 = int(np.ceil(np.random.uniform(1e-2, ls - fs)))
            w1 = int(np.ceil(np.random.uniform(1e-2, (ls - fs) * a)))
            A, B = A[h1:h1 + fs, w1:w1 + fs * a], B[h1:h1 + fs, w1:w1 + fs * a]
            if np.random.random() > 0.5:
                A, B = A[:, ::-1], B[:, ::-1]
        else:
            A = load_rgb(self.A[i], fs, fs * a)
            B = load_rgb(self.B[i], fs, fs * a)
        return to_tensor(A), to_tensor(B)


class ImagePool:
    """Port of upstream utils.ImagePool, including its quirk: it stores and swaps whole
    (fake_A batch, fake_B batch) pairs, and replaces A and B from independent random slots."""

    def __init__(self, maxsize=50):
        self.maxsize, self.images = maxsize, []

    def __call__(self, fake_A, fake_B):
        if self.maxsize <= 0:
            return fake_A, fake_B
        pair = [fake_A.detach().clone(), fake_B.detach().clone()]
        if len(self.images) < self.maxsize:
            self.images.append(pair)
            return pair[0], pair[1]
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = self.images[idx][0].clone()
            self.images[idx][0] = pair[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = self.images[idx][1].clone()
            self.images[idx][1] = pair[1]
            return tmp1, tmp2
        return pair[0], pair[1]
