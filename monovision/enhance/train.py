"""Train AU-GAN / pt-AUGAN in PyTorch. Mirrors AUGAN.train() upstream.

  python -m monovision.enhance.train --config configs/gan/augan_bdd100k.yaml
  python -m monovision.enhance.train --dataset_dir datasets/bdd100k --experiment_name bdd_exp

Per iteration (as upstream): one Adam step on both generators, push fakes through the image
pool, then one Adam step per discriminator pair i in range(n_d). LR is constant for
`epoch_step` epochs, then decays linearly to 0 at `epoch`.

Fixed vs upstream: boolean flags are parsed properly (upstream argparse type=bool makes
`--use_uncertainty False` evaluate True).
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImagePool, UnpairedDataset
from .losses import discriminator_loss, generator_forward
from .models import AUGAN


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"boolean expected, got {v!r}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML file; its keys override the defaults below")
    p.add_argument("--dataset_dir", default="datasets/bdd100k")
    p.add_argument("--experiment_name", default="bdd_exp")
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--epoch", type=int, default=20)
    p.add_argument("--epoch_step", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--train_size", type=int, default=int(1e8))
    p.add_argument("--load_size", type=int, default=286)
    p.add_argument("--fine_size", type=int, default=256)
    p.add_argument("--aspect", type=int, default=2, help="width/height ratio; upstream 2 (256x512), paper 1")
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_d", type=int, default=2)
    p.add_argument("--n_scale", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--L1_lambda", type=float, default=10.0)
    p.add_argument("--conf_lambda", type=float, default=1.0)
    p.add_argument("--use_lsgan", type=str2bool, default=True)
    p.add_argument("--use_uncertainty", type=str2bool, default=True)
    p.add_argument("--max_size", type=int, default=50, help="image pool size")
    p.add_argument("--upsample", choices=["deconv", "upconv"], default="deconv", help="pt-AUGAN: upconv")
    p.add_argument("--up_mode", choices=["bilinear", "nearest"], default="bilinear")
    p.add_argument("--demod", type=str2bool, default=False, help="pt-AUGAN: weight demodulation")
    p.add_argument("--save_freq", type=int, default=1000)
    p.add_argument("--print_freq", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--continue_train", type=str2bool, default=False)
    p.add_argument("--init_from", default=None, help="state_dict .pt to start from (e.g. converted TF weights)")
    p.add_argument("--max_iters", type=int, default=0, help="stop after N iterations (0 = no limit; for smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def parse_args(argv=None):
    p = build_parser()
    args, _ = p.parse_known_args(argv)
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        unknown = set(cfg) - set(vars(args))
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        p.set_defaults(**cfg)
    return p.parse_args(argv)


def model_from_args(args):
    return AUGAN(ngf=args.ngf, ndf=args.ndf, n_d=args.n_d, n_scale=args.n_scale, upsample=args.upsample,
                 up_mode=args.up_mode, demod=args.demod)


def save_ckpt(path, model, opt_g, opt_d, epoch, counter, args):
    torch.save({"model": model.state_dict(), "opt_g": opt_g.state_dict(),
                "opt_d": [o.state_dict() for o in opt_d], "epoch": epoch, "counter": counter,
                "args": vars(args)}, path)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    out_dir = os.path.join(args.checkpoint_root, args.experiment_name)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)

    model = model_from_args(args).to(dev)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=dev))
    g_params = list(model.G_A2B.parameters()) + list(model.G_B2A.parameters())
    opt_g = torch.optim.Adam(g_params, lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = [torch.optim.Adam(list(da.parameters()) + list(db.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
             for da, db in zip(model.D_A, model.D_B)]
    d_params = [p for d in list(model.D_A) + list(model.D_B) for p in d.parameters()]

    start_epoch, counter = 0, 1
    last = os.path.join(out_dir, "last.pt")
    if args.continue_train and os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"])
        opt_g.load_state_dict(ck["opt_g"])
        for o, s in zip(opt_d, ck["opt_d"]):
            o.load_state_dict(s)
        start_epoch, counter = ck["epoch"], ck["counter"]
        print(f"[*] resumed from {last} (epoch {start_epoch}, iter {counter})")

    ds = UnpairedDataset(args.dataset_dir, "train", args.load_size, args.fine_size, args.aspect, train=True,
                         seed=args.seed)
    pool = ImagePool(args.max_size)
    model.train()
    t0 = time.time()
    for epoch in range(start_epoch, args.epoch):
        ds.reshuffle()
        n = min(len(ds), args.train_size) // args.batch_size
        lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)
        for o in [opt_g] + opt_d:
            for g in o.param_groups:
                g["lr"] = lr
        loader = DataLoader(torch.utils.data.Subset(ds, range(n * args.batch_size)), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, drop_last=True)
        for idx, (real_A, real_B) in enumerate(loader):
            real_A, real_B = real_A.to(dev), real_B.to(dev)

            # --- generator step (discriminators frozen) ---
            for p in d_params:
                p.requires_grad_(False)
            g_loss, parts, fakes = generator_forward(model, real_A, real_B, args.L1_lambda, args.conf_lambda,
                                                     args.use_uncertainty, args.use_lsgan)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            for p in d_params:
                p.requires_grad_(True)

            # --- discriminator steps ---
            fake_A, fake_B = pool(fakes["fake_A"], fakes["fake_B"])
            d_losses = []
            for i in range(args.n_d):
                d_loss = discriminator_loss(model.D_A[i], model.D_B[i], real_A, real_B, fake_A, fake_B, args.use_lsgan)
                opt_d[i].zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d[i].step()
                d_losses.append(d_loss.item())

            counter += 1
            print(f"Epoch [{epoch:2d}] [{idx:5d}/{n:5d}] t={time.time() - t0:8.1f}s g={g_loss.item():.4f} "
                  f"adv={parts['g_adv'].item():.4f} percep={parts['percep'].item():.4f} "
                  f"d={' '.join(f'{d:.4f}' for d in d_losses)}", flush=True)
            if counter % args.print_freq == 1:
                save_samples(model, real_A, real_B, fakes, os.path.join(out_dir, "samples", f"{epoch:02d}_{idx:05d}.jpg"))
            if counter % args.save_freq == 2:
                save_ckpt(last, model, opt_g, opt_d, epoch, counter, args)
            if args.max_iters and counter > args.max_iters:
                save_ckpt(last, model, opt_g, opt_d, epoch, counter, args)
                return model
        save_ckpt(last, model, opt_g, opt_d, epoch + 1, counter, args)
        torch.save(model.state_dict(), os.path.join(out_dir, f"weights_epoch{epoch:02d}.pt"))
    return model


def save_samples(model, real_A, real_B, fakes, path):
    from PIL import Image

    def img(t):
        return ((t[0].detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)

    grid = np.concatenate([np.concatenate([img(real_A), img(fakes["fake_B"])], 1),
                           np.concatenate([img(real_B), img(fakes["fake_A"])], 1)], 0)
    Image.fromarray(grid).save(path)


if __name__ == "__main__":
    train(parse_args())
