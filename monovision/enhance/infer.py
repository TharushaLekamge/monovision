"""Night -> day translation of a folder of frames (pipeline step before SimpleRecon).

  python -m monovision.enhance.infer --weights checkpoints/bdd_exp/weights_epoch19.pt \
      --input scans/<scan>/images --output scans_day/<scan>/images

* Keeps filenames, so camera poses/intrinsics stay aligned with frames.
* --size HxW resizes before translation (e.g. 256x512 to match training, as upstream test did).
  Default: keep native resolution (the generator is fully convolutional) [assumption: the paper
  doesn't say what resolution frames were translated at]. Input is reflect-padded to a
  multiple of 4 and cropped back, so output size == input size.
* --keep_size resizes the result back to the original frame size when --size is used.
* Architecture flags (upsample/demod/ngf) are read from the training checkpoint's args when
  the weights file is a full training checkpoint; pass them explicitly for bare state_dicts.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import list_images
from .models import Generator
from .train import str2bool


def load_generator(weights, device, ngf=64, upsample="deconv", up_mode="bilinear", demod=False):
    ck = torch.load(weights, map_location=device)
    state = ck
    if isinstance(ck, dict) and "model" in ck:
        a = ck.get("args", {})
        ngf, upsample = a.get("ngf", ngf), a.get("upsample", upsample)
        up_mode, demod = a.get("up_mode", up_mode), a.get("demod", demod)
        state = ck["model"]
    g = Generator(ngf, transfer=True, upsample=upsample, up_mode=up_mode, demod=demod)
    prefix = "G_A2B."
    g.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    return g.to(device).eval()


@torch.no_grad()
def translate_image(g, img, device, size=None, keep_size=True):
    """img: PIL RGB. Returns PIL RGB."""
    w0, h0 = img.size
    if size:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].to(device)
    h, w = x.shape[-2:]
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    y = g.translate(x)[..., :h, :w]
    out = Image.fromarray(((y[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8))
    if size and keep_size:
        out = out.resize((w0, h0), Image.BILINEAR)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--size", default=None, help="HxW, e.g. 256x512")
    p.add_argument("--keep_size", type=str2bool, default=True)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--upsample", choices=["deconv", "upconv"], default="deconv")
    p.add_argument("--up_mode", choices=["bilinear", "nearest"], default="bilinear")
    p.add_argument("--demod", type=str2bool, default=False)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    dev = torch.device(a.device)
    g = load_generator(a.weights, dev, a.ngf, a.upsample, a.up_mode, a.demod)
    os.makedirs(a.output, exist_ok=True)
    files = list_images(a.input)
    for i, f in enumerate(files):
        out = translate_image(g, Image.open(f).convert("RGB"), dev, size, a.keep_size)
        out.save(os.path.join(a.output, os.path.basename(f)), quality=95)
        if i % 50 == 0:
            print(f"[{i + 1}/{len(files)}] {f}", flush=True)


if __name__ == "__main__":
    main()
