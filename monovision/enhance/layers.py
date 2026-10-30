"""Building blocks that reproduce TensorFlow-1.x semantics exactly in PyTorch.

Why not plain nn.Conv2d(padding=k//2)?  TF 'SAME' padding is asymmetric when the total
padding is odd (extra pixel goes bottom/right). For AU-GAN's stride-2 3x3 convs on even
inputs TF pads (top=0, bottom=1), whereas PyTorch padding=1 pads (1, 1); the outputs differ.
Likewise TF's 'SAME' conv2d_transpose is not PyTorch's ConvTranspose2d(padding=1,
output_padding=1). Both are verified numerically against TF in tests/test_parity_tf.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def tf_same_pad_amounts(size, k, s, d=1):
    """(before, after) padding TF uses for 'SAME' along one axis."""
    out = math.ceil(size / s)
    eff_k = (k - 1) * d + 1
    total = max((out - 1) * s + eff_k - size, 0)
    return total // 2, total - total // 2


class Conv2dTF(nn.Module):
    """Bias-free conv with TF padding ('SAME' or 'VALID').

    demod=True applies StyleGAN2-style weight demodulation (each output filter rescaled to unit
    L2 norm). This is one of the pt-AUGAN changes described in the MonoVision paper; the paper
    does not say exactly where it was applied, so it is a switch on every generator conv.
    Without a style vector, demodulation reduces to per-filter weight normalisation.
    """

    def __init__(self, cin, cout, k, s=1, padding="SAME", demod=False, eps=1e-8):
        super().__init__()
        self.k, self.s, self.padding, self.demod, self.eps = k, s, padding.upper(), demod, eps
        self.weight = nn.Parameter(torch.empty(cout, cin, k, k))
        nn.init.trunc_normal_(self.weight, std=0.02, a=-0.04, b=0.04)  # TF truncated_normal(0.02)

    def effective_weight(self):
        w = self.weight
        if self.demod:
            w = w * torch.rsqrt(w.pow(2).sum(dim=[1, 2, 3], keepdim=True) + self.eps)
        return w

    def forward(self, x):
        if self.padding == "SAME":
            ph = tf_same_pad_amounts(x.shape[-2], self.k, self.s)
            pw = tf_same_pad_amounts(x.shape[-1], self.k, self.s)
            x = F.pad(x, (pw[0], pw[1], ph[0], ph[1]))
        return F.conv2d(x, self.effective_weight(), stride=self.s)


class ConvTranspose2dTF(nn.Module):
    """Bias-free transposed conv matching TF/slim conv2d_transpose(padding='SAME').

    TF's SAME transpose conv yields out = in * s and is the adjoint of a SAME conv whose
    padding is (before, after) = tf_same_pad_amounts(out, k, s). Equivalent: a PyTorch
    transpose conv with no padding, then crop [before : before + out].
    """

    def __init__(self, cin, cout, k=3, s=2):
        super().__init__()
        self.k, self.s = k, s
        self.weight = nn.Parameter(torch.empty(cin, cout, k, k))
        nn.init.trunc_normal_(self.weight, std=0.02, a=-0.04, b=0.04)

    def forward(self, x):
        h, w = x.shape[-2] * self.s, x.shape[-1] * self.s
        y = F.conv_transpose2d(x, self.weight, stride=self.s)
        th = tf_same_pad_amounts(h, self.k, self.s)[0]
        tw = tf_same_pad_amounts(w, self.k, self.s)[0]
        return y[..., th:th + h, tw:tw + w]


class UpsampleConv(nn.Module):
    """x2 upsample followed by a stride-1 conv (pt-AUGAN replacement for transposed conv).

    The paper doesn't state the interpolation mode; 'bilinear' is the default [assumption],
    'nearest' is selectable.
    """

    def __init__(self, cin, cout, k=3, mode="bilinear", demod=False):
        super().__init__()
        self.mode = mode
        self.conv = Conv2dTF(cin, cout, k, 1, "SAME", demod=demod)

    def forward(self, x):
        kw = {"align_corners": False} if self.mode in ("bilinear", "bicubic") else {}
        return self.conv(F.interpolate(x, scale_factor=2, mode=self.mode, **kw))


class InstanceNormTF(nn.Module):
    """Matches AU-GAN's instance_norm: biased variance, eps 1e-5, scale ~ N(1, 0.02), offset 0.

    Implemented by hand instead of nn.InstanceNorm2d because PyTorch raises on 1x1 feature
    maps in train mode, while TF simply returns the offset there (deep discriminator layers
    reach 1x1 on small crops).
    """

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(c).normal_(1.0, 0.02))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def reflect_pad3(x):
    return F.pad(x, (3, 3, 3, 3), mode="reflect")


def lrelu(x):
    return torch.maximum(x, 0.2 * x)


def tf_legacy_resize_bicubic(x, out_h, out_w):
    """TF1 tf.image.resize_bicubic(align_corners=False, half_pixel_centers=False).

    Uses the cubic kernel with a=-0.75 (TF legacy), source coord = dst * (in / out), edge-clamped.
    For an exact 2x downscale this reduces to x[..., ::2, ::2].
    """

    def weights_idx(in_size, out_size, device):
        scale = in_size / out_size
        src = torch.arange(out_size, device=device, dtype=torch.float64) * scale
        i0 = torch.floor(src)
        # TF evaluates the kernel from a 1024-entry lookup table: t is rounded to 1/1024.
        t = (torch.round((src - i0) * 1024) / 1024).unsqueeze(1)  # (out, 1)
        offs = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device, dtype=torch.float64)
        dist = (t - offs).abs()  # (out, 4)
        a = -0.75  # TF legacy (half_pixel_centers=False) uses -0.75; -0.5 only with half-pixel centers
        w = torch.where(
            dist <= 1,
            ((a + 2) * dist - (a + 3)) * dist * dist + 1,
            torch.where(dist < 2, ((a * dist - 5 * a) * dist + 8 * a) * dist - 4 * a, torch.zeros_like(dist)),
        )
        idx = (i0.unsqueeze(1) + offs).clamp(0, in_size - 1).long()
        return w.to(torch.float32), idx

    wh, ih = weights_idx(x.shape[-2], out_h, x.device)
    ww, iw = weights_idx(x.shape[-1], out_w, x.device)
    # rows
    xr = x[..., ih, :]  # (..., out_h, 4, W)
    xr = (xr * wh.to(x.dtype)[..., None]).sum(-2)
    xc = xr[..., iw]  # (..., out_h, out_w, 4)
    return (xc * ww.to(x.dtype)).sum(-1)
