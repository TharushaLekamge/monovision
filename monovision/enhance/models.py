"""PyTorch port of AU-GAN (Kwak et al., 2021) — generator and multi-scale discriminator.

Ported line-by-line from third_party/AU-GAN/models.py and ops.py. Module names mirror the TF
variable scopes (g_e1_c, g_r1_c1, ...) so TF checkpoints convert mechanically
(see convert_tf.py).

Faithfulness notes (all verified against TF in tests/test_parity_tf.py):
  * The "dilated" residual blocks are called with rate=1 upstream, i.e. plain 3x3 convs.
  * All convs are bias-free (slim biases_initializer=None).
  * Discriminator input pyramid uses TF1 legacy bicubic resize.

Deliberate difference: upstream adds N(0, 0.02) noise in the reconstruction branch at train
AND test time. Here it is applied only in train() mode (noise_std can be set to 0). The
reconstruction branch is not used by the night->day output, so inference is unaffected.

pt-AUGAN switches (paper §III-B, placement is our assumption):
  * upsample='upconv'  -> UpsampleConv instead of transposed conv in every decoder
  * demod=True         -> weight-demodulated generator convs
Either switch makes the model incompatible with the original TF checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    Conv2dTF,
    ConvTranspose2dTF,
    InstanceNormTF,
    UpsampleConv,
    lrelu,
    reflect_pad3,
    tf_legacy_resize_bicubic,
)


class ResBlock(nn.Module):
    """residule_block_dilated (rate 1) and residual_block (T-Net) share this structure."""

    def __init__(self, dim, demod=False):
        super().__init__()
        self.c1 = Conv2dTF(dim, dim, 3, 1, demod=demod)
        self.n1 = InstanceNormTF(dim)
        self.c2 = Conv2dTF(dim, dim, 3, 1, demod=demod)
        self.n2 = InstanceNormTF(dim)

    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return y + x


class Decoder(nn.Module):
    """4 res blocks -> 2x (up, IN, relu) -> reflect pad -> 7x7 conv -> tanh."""

    def __init__(self, ngf, out_c, upsample="deconv", up_mode="bilinear", demod=False):
        super().__init__()
        dim = ngf * 4
        self.res = nn.ModuleList([ResBlock(dim, demod) for _ in range(4)])
        self.d1 = _up(dim, ngf * 2, upsample, up_mode, demod)
        self.n_d1 = InstanceNormTF(ngf * 2)
        self.d2 = _up(ngf * 2, ngf, upsample, up_mode, demod)
        self.n_d2 = InstanceNormTF(ngf)
        self.pred = Conv2dTF(ngf, out_c, 7, 1, "VALID", demod=demod)

    def forward(self, x, noise_fn=None):
        for i, r in enumerate(self.res):
            x = r(x)
            if i == 0 and noise_fn is not None:
                x = noise_fn(x)
        d1 = F.relu(self.n_d1(self.d1(x)))
        d2 = F.relu(self.n_d2(self.d2(d1)))
        return torch.tanh(self.pred(reflect_pad3(d2))), d1


def _up(cin, cout, upsample, up_mode, demod):
    if upsample == "deconv":
        if demod:
            raise ValueError("demod is not supported on transposed convs; use upsample='upconv'")
        return ConvTranspose2dTF(cin, cout, 3, 2)
    if upsample == "upconv":
        return UpsampleConv(cin, cout, 3, mode=up_mode, demod=demod)
    raise ValueError(upsample)


class Generator(nn.Module):
    """generator_resnet. transfer=True builds the A2B (adverse->normal) generator with the
    T-Net feature-transfer blocks and the confidence head; transfer=False builds B2A."""

    def __init__(self, ngf=64, in_c=3, out_c=3, transfer=True, upsample="deconv", up_mode="bilinear",
                 demod=False, noise_std=0.02):
        super().__init__()
        self.transfer, self.noise_std = transfer, noise_std
        self.e1 = Conv2dTF(in_c, ngf, 7, 1, "VALID", demod=demod)
        self.n_e1 = InstanceNormTF(ngf)
        self.e2 = Conv2dTF(ngf, ngf * 2, 3, 2, demod=demod)
        self.n_e2 = InstanceNormTF(ngf * 2)
        self.e3 = Conv2dTF(ngf * 2, ngf * 4, 3, 2, demod=demod)
        self.n_e3 = InstanceNormTF(ngf * 4)
        self.enc_res = nn.ModuleList([ResBlock(ngf * 4, demod) for _ in range(4)])  # g_r1..g_r4
        if transfer:
            self.tnet = nn.ModuleList([ResBlock(ngf * 4, demod) for _ in range(4)])  # g_t1..g_t4
        self.dec = Decoder(ngf, out_c, upsample, up_mode, demod)  # g_r6..g_r9, g_d1, g_d2, g_pred_c
        self.dec_rec = Decoder(ngf, out_c, upsample, up_mode, demod)  # *_rec
        if transfer:
            self.d_conf = _up(ngf * 2, ngf, upsample, up_mode, demod)  # g_d_dc_conf
            self.n_conf = InstanceNormTF(ngf)
            self.pred_conf = Conv2dTF(ngf, 1, 7, 1, "VALID", demod=demod)

    def _noise(self, x):
        if self.training and self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x

    def forward(self, x):
        """Returns (pred, pred_rec, r4, t4, conf) like the TF function. t4/conf are None for B2A."""
        c = F.relu(self.n_e1(self.e1(reflect_pad3(x))))
        c = F.relu(self.n_e2(self.e2(c)))
        c = F.relu(self.n_e3(self.e3(c)))
        r4 = c
        for r in self.enc_res:
            r4 = r(r4)
        t4 = None
        feature = r4
        if self.transfer:
            t4 = r4
            for t in self.tnet:
                t4 = t(t4)
            feature = t4
        pred, d1 = self.dec(feature)
        pred_rec, _ = self.dec_rec(self._noise(r4), noise_fn=self._noise)
        conf = None
        if self.transfer:
            dc = F.relu(self.n_conf(self.d_conf(d1)))
            conf = F.softplus(self.pred_conf(reflect_pad3(dc)))
        return pred, pred_rec, r4, t4, conf

    @torch.no_grad()
    def translate(self, x):
        """Inference helper: only the translation output."""
        return self.forward(x)[0]


class Discriminator(nn.Module):
    """AU-GAN multi-scale discriminator (ops.dis_down / final_conv), n_scale image pyramid.

    Faithful to the upstream wiring, including its unusual cross-scale routing: at each level,
    branch i < n_scale-1 takes the previous branch i+1 through a stride-1 conv, and the last
    branch takes the previous branch 0 through n_scale stride-2 convs.
    """

    def __init__(self, ndf=32, in_c=3, n_scale=2, n_levels=4):
        super().__init__()
        self.n_scale = n_scale
        self.levels = nn.ModuleList()
        chans = [in_c] * n_scale
        for lvl in range(n_levels):
            ch = ndf * (2 ** lvl)
            branches = nn.ModuleList()
            new_chans = []
            for i in range(n_scale):
                if i == n_scale - 1:
                    steps = nn.ModuleList()
                    cin = chans[0]
                    for _ in range(n_scale):
                        steps.append(nn.ModuleList([Conv2dTF(cin, ch, 4, 2), InstanceNormTF(ch)]))
                        cin = ch
                    branches.append(steps)
                else:
                    branches.append(nn.ModuleList([nn.ModuleList([Conv2dTF(chans[i + 1], ch, 4, 1), InstanceNormTF(ch)])]))
                new_chans.append(ch)
            chans = new_chans
            self.levels.append(branches)
        self.final = nn.ModuleList([Conv2dTF(chans[i], 1, 4, 1) for i in range(n_scale)])

    def forward(self, x):
        h, w = x.shape[-2:]
        images = [x] + [tf_legacy_resize_bicubic(x, h // (2 ** i), w // (2 ** i)) for i in range(1, self.n_scale)]
        for branches in self.levels:
            backpack = images[0]
            new = list(images)
            for i, steps in enumerate(branches):
                y = backpack if i == self.n_scale - 1 else images[i + 1]
                for conv, norm in steps:
                    y = lrelu(norm(conv(y)))
                new[i] = y
                # upstream mutates the list in place: branch i reads images[i+1] before it is
                # overwritten, and the last branch reads the saved backpack, so a copy is equivalent.
            images = new
        return [f(im) for f, im in zip(self.final, images)]


class AUGAN(nn.Module):
    """Container with the same parts as the TF AUGAN class: G_A2B, G_B2A, n_d D's per domain."""

    def __init__(self, ngf=64, ndf=64, n_d=2, n_scale=2, upsample="deconv", up_mode="bilinear", demod=False,
                 noise_std=0.02):
        super().__init__()
        self.G_A2B = Generator(ngf, transfer=True, upsample=upsample, up_mode=up_mode, demod=demod, noise_std=noise_std)
        self.G_B2A = Generator(ngf, transfer=False, upsample=upsample, up_mode=up_mode, demod=demod, noise_std=noise_std)
        df = ndf // n_d  # upstream: df_dim = ndf // n_d
        self.D_A = nn.ModuleList([Discriminator(df, n_scale=n_scale) for _ in range(n_d)])
        self.D_B = nn.ModuleList([Discriminator(df, n_scale=n_scale) for _ in range(n_d)])
