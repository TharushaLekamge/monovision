"""Convert an original AU-GAN TensorFlow checkpoint into a PyTorch state_dict for models.AUGAN.

  python -m monovision.enhance.convert_tf --tf_ckpt checkpoints/augan_tf/bdd100k_256 \
      --out checkpoints/augan_bdd100k_256_torch.pt

Needs TensorFlow 2.x (any recent version; only tf.train.load_checkpoint is used, so the old
TF 1.14 environment is NOT required). Only the original architecture converts
(upsample='deconv', demod=False, ngf=64, ndf=64, n_d=2, n_scale=2).

Weight layout: TF conv [kh, kw, in, out] and TF conv-transpose [kh, kw, out, in] both map to
PyTorch with permute(3, 2, 0, 1); no kernel flip is needed (verified in tests).
"""
import argparse

import numpy as np
import torch

from .models import AUGAN


def _norm(prefix, tf_scope):
    return {f"{prefix}.weight": f"{tf_scope}/scale", f"{prefix}.bias": f"{tf_scope}/offset"}


def _res_dilated(prefix, scope, tf_name):
    m = {f"{prefix}.c1.weight": f"{scope}/{tf_name}_c1/filter", f"{prefix}.c2.weight": f"{scope}/{tf_name}_c2/filter"}
    m.update(_norm(f"{prefix}.n1", f"{scope}/{tf_name}_bn1"))
    m.update(_norm(f"{prefix}.n2", f"{scope}/{tf_name}_bn2"))
    return m


def _res_tnet(prefix, scope, tf_name):
    s = f"{scope}/{tf_name}"
    m = {f"{prefix}.c1.weight": f"{s}/res1/{tf_name}_c1/Conv/weights",
         f"{prefix}.c2.weight": f"{s}/res2/{tf_name}_c2/Conv/weights"}
    m.update(_norm(f"{prefix}.n1", f"{s}/res1/{tf_name}_in1"))
    m.update(_norm(f"{prefix}.n2", f"{s}/res2/{tf_name}_in2"))
    return m


def generator_map(prefix, scope, transfer):
    m = {}
    for i in (1, 2, 3):
        m[f"{prefix}.e{i}.weight"] = f"{scope}/g_e{i}_c/Conv/weights"
        m.update(_norm(f"{prefix}.n_e{i}", f"{scope}/g_e{i}_bn"))
    for k in range(4):
        m.update(_res_dilated(f"{prefix}.enc_res.{k}", scope, f"g_r{k + 1}"))
        if transfer:
            m.update(_res_tnet(f"{prefix}.tnet.{k}", scope, f"g_t{k + 1}"))
        m.update(_res_dilated(f"{prefix}.dec.res.{k}", scope, f"g_r{k + 6}"))
        m.update(_res_dilated(f"{prefix}.dec_rec.res.{k}", scope, f"g_r{k + 6}_rec"))
    for dec, sfx in (("dec", ""), ("dec_rec", "_rec")):
        m[f"{prefix}.{dec}.d1.weight"] = f"{scope}/g_d1_dc{sfx}/Conv2d_transpose/weights"
        m[f"{prefix}.{dec}.d2.weight"] = f"{scope}/g_d2_dc{sfx}/Conv2d_transpose/weights"
        m.update(_norm(f"{prefix}.{dec}.n_d1", f"{scope}/g_d1_bn{sfx}"))
        m.update(_norm(f"{prefix}.{dec}.n_d2", f"{scope}/g_d2_bn{sfx}"))
        m[f"{prefix}.{dec}.pred.weight"] = f"{scope}/g_pred_c{sfx}/Conv/weights"
    if transfer:
        m[f"{prefix}.d_conf.weight"] = f"{scope}/g_d_dc_conf/Conv2d_transpose/weights"
        m.update(_norm(f"{prefix}.n_conf", f"{scope}/g_d_bn_conf"))
        m[f"{prefix}.pred_conf.weight"] = f"{scope}/g_pred_c_conf/Conv/weights"
    return m


def discriminator_map(prefix, scope, n_scale=2, n_levels=4):
    m = {}
    for lvl in range(n_levels):
        base = f"d_h{lvl}_conv_scale_"
        for i in range(n_scale):
            if i == n_scale - 1:
                for j in range(n_scale):
                    m[f"{prefix}.levels.{lvl}.{i}.{j}.0.weight"] = f"{scope}/{base}{i}_c_{j}/Conv/weights"
                    m.update(_norm(f"{prefix}.levels.{lvl}.{i}.{j}.1", f"{scope}/{base}{i}_bn_{j}"))
            else:
                m[f"{prefix}.levels.{lvl}.{i}.0.0.weight"] = f"{scope}/{base}{i}_first_c/filter"
                m.update(_norm(f"{prefix}.levels.{lvl}.{i}.0.1", f"{scope}/{base}{i}_first_bn"))
    for i in range(n_scale):
        m[f"{prefix}.final.{i}.weight"] = f"{scope}/d_pred_scale_{i}/Conv/weights"
    return m


def augan_map(n_d=2, n_scale=2):
    m = {}
    m.update(generator_map("G_A2B", "generatorA2B", True))
    m.update(generator_map("G_B2A", "generatorB2A", False))
    for i in range(n_d):
        m.update(discriminator_map(f"D_A.{i}", f"{i}_discriminatorA", n_scale))
        m.update(discriminator_map(f"D_B.{i}", f"{i}_discriminatorB", n_scale))
    return m


def tf_to_torch(arr):
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    return t.permute(3, 2, 0, 1).contiguous() if t.ndim == 4 else t


def convert(get_tensor, model=None, n_d=2, n_scale=2):
    """get_tensor(name) -> np.ndarray. Returns a state_dict; raises if anything is unmapped."""
    model = model or AUGAN(n_d=n_d, n_scale=n_scale)
    ref = model.state_dict()
    mapping = augan_map(n_d, n_scale)
    missing = set(ref) - set(mapping)
    if missing:
        raise KeyError(f"torch params without TF mapping: {sorted(missing)[:10]}")
    out = {}
    for k, name in mapping.items():
        t = tf_to_torch(get_tensor(name))
        if t.shape != ref[k].shape:
            raise ValueError(f"shape mismatch {k} {tuple(ref[k].shape)} vs {name} {tuple(t.shape)}")
        out[k] = t
    return out


def convert_checkpoint(tf_ckpt):
    import tensorflow as tf

    path = tf.train.latest_checkpoint(tf_ckpt) or tf_ckpt
    reader = tf.train.load_checkpoint(path)
    names = set(reader.get_variable_to_shape_map())
    sd = convert(reader.get_tensor)
    used = set(augan_map().values())
    unused = sorted(n for n in names - used if "Adam" not in n and "beta" not in n.lower())
    if unused:
        print(f"[!] {len(unused)} TF variables not used, e.g. {unused[:5]}")
    return sd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tf_ckpt", required=True, help="checkpoint dir or prefix (e.g. .../cyclegan.model-12345)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    sd = convert_checkpoint(a.tf_ckpt)
    torch.save(sd, a.out)
    print(f"[*] wrote {len(sd)} tensors to {a.out}")


if __name__ == "__main__":
    main()
