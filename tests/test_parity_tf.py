"""Numerical parity: PyTorch port vs the ORIGINAL AU-GAN TF graph (random weights).

Builds upstream AUGAN under TF2 v1-compat, fills every variable with random values (so norms,
offsets and all branches are exercised), saves a TF checkpoint, converts it with
monovision.enhance.convert_tf, and compares outputs and losses.

Requires: tensorflow (2.x), tf-slim, and third_party/AU-GAN (scripts/00_setup.sh).
These tests do not need the pretrained AU-GAN weights.
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
tf_mod = pytest.importorskip("tensorflow")
pytest.importorskip("tf_slim")

from monovision.enhance.convert_tf import convert_checkpoint  # noqa: E402
from monovision.enhance.layers import tf_legacy_resize_bicubic  # noqa: E402
from monovision.enhance.losses import discriminator_loss, generator_forward  # noqa: E402
from monovision.enhance.models import AUGAN, Generator  # noqa: E402

ATOL = 2e-4


def _ref(float64=False):
    import tf_reference

    try:
        return tf_reference.load(float64=float64)
    except FileNotFoundError as e:
        pytest.skip(str(e))


def _randomize(sess, tf, rng):
    for v in tf.global_variables():
        shape = v.shape.as_list()
        if v.name.endswith("scale:0"):
            val = 1 + 0.1 * rng.standard_normal(shape)
        elif v.name.endswith("offset:0"):
            val = 0.1 * rng.standard_normal(shape)
        else:
            fan_in = np.prod(shape[:-1]) if "transpose" not in v.name else shape[0] * shape[1] * shape[3]
            val = rng.standard_normal(shape) / np.sqrt(fan_in)
        v.load(val.astype(np.float32), sess)


def _nhwc(t):
    return t.detach().permute(0, 2, 3, 1).numpy()


@pytest.fixture(scope="module")
def full_model():
    return build_full_model()


def build_full_model(float64=False):
    """TF AUGAN at 32x64 with random weights + converted torch model."""
    ref = _ref(float64)
    tf = ref.ops.tf
    tf.reset_default_graph()
    args = SimpleNamespace(batch_size=1, fine_size=32, input_nc=3, output_nc=3, L1_lambda=10.0, conf_lambda=1.0,
                           dataset_dir="x", n_d=2, n_scale=2, ndf=64, load_size=36, ngf=64, use_lsgan=True,
                           use_uncertainty=True, phase="train", save_conf=False, max_size=50)
    rng = np.random.default_rng(0)
    sess = tf.Session()
    m = ref.AUGAN.AUGAN(sess, args)
    sess.run(tf.global_variables_initializer())
    _randomize(sess, tf, rng)
    d = tempfile.mkdtemp()
    tf.train.Saver().save(sess, os.path.join(d, "cyclegan.model"), global_step=1)
    sd = convert_checkpoint(d)
    tm = AUGAN(noise_std=0.0)
    tm.load_state_dict(sd)
    tm.eval()
    if float64:
        tm.double()
    return SimpleNamespace(tf=tf, sess=sess, m=m, tm=tm, rng=rng)


def test_generator_and_losses_match(full_model):
    f = full_model
    real = f.rng.uniform(-1, 1, (1, 32, 64, 6)).astype(np.float32)
    fakeA = f.rng.uniform(-1, 1, (1, 32, 64, 3)).astype(np.float32)
    fakeB = f.rng.uniform(-1, 1, (1, 32, 64, 3)).astype(np.float32)
    m = f.m
    tf_out = f.sess.run(
        dict(fake_B=m.fake_B, rec_realA=m.rec_realA, r4=m.realA_percep, t4=m.transA_percep, conf=m.pred_confA,
             fake_A=m.fake_A, fake_A_=m.fake_A_, fake_B_=m.fake_B_, g_loss=m.g_loss, percep=m.percep_loss,
             g_adv=m.g_adv_total, a_cycle=m.g_A_cycle_loss, d0=m.d_loss_item[0], d1=m.d_loss_item[1]),
        feed_dict={m.real_data: real, m.fake_A_sample: fakeA, m.fake_B_sample: fakeB})

    A = torch.from_numpy(real[..., :3]).permute(0, 3, 1, 2)
    B = torch.from_numpy(real[..., 3:]).permute(0, 3, 1, 2)
    with torch.no_grad():
        fake_B, rec_realA, r4, t4, conf = f.tm.G_A2B(A)
        g_loss, parts, fakes = generator_forward(f.tm, A, B)
        fa = torch.from_numpy(fakeA).permute(0, 3, 1, 2)
        fb = torch.from_numpy(fakeB).permute(0, 3, 1, 2)
        d0 = discriminator_loss(f.tm.D_A[0], f.tm.D_B[0], A, B, fa, fb)
        d1 = discriminator_loss(f.tm.D_A[1], f.tm.D_B[1], A, B, fa, fb)

    pairs = {"fake_B": fake_B, "rec_realA": rec_realA, "r4": r4, "t4": t4, "conf": conf, "fake_A": fakes["fake_A"]}
    for k, t in pairs.items():
        np.testing.assert_allclose(_nhwc(t), tf_out[k], atol=ATOL, rtol=1e-3, err_msg=k)
    for k, v in {"g_loss": g_loss, "percep": parts["percep"], "g_adv": parts["g_adv"], "a_cycle": parts["a_cycle"],
                 "d0": d0, "d1": d1}.items():
        np.testing.assert_allclose(float(v), tf_out[k], rtol=1e-4, atol=1e-4, err_msg=k)


@pytest.mark.parametrize("hw", [(36, 52), (30, 50), (33, 47)])
def test_generator_odd_sizes(hw):
    """TF 'SAME' asymmetric padding and transpose-conv cropping at non-multiple-of-4 sizes."""
    ref = _ref()
    tf = ref.ops.tf
    tf.reset_default_graph()
    from collections import namedtuple

    O = namedtuple("O", "batch_size image_size gf_dim df_dim output_c_dim is_training")
    o = O(1, hw[0], 16, 8, 3, False)
    x = tf.placeholder(tf.float32, [1, hw[0], hw[1], 3])
    outs = ref.models.generator_resnet(x, o, transfer=True, name="generatorA2B")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _randomize(sess, tf, np.random.default_rng(1))
        xin = np.random.default_rng(2).uniform(-1, 1, (1, hw[0], hw[1], 3)).astype(np.float32)
        tf_outs = sess.run(list(outs), {x: xin})
        reader_vals = {v.name[:-2]: sess.run(v) for v in tf.global_variables()}
    from monovision.enhance.convert_tf import generator_map, tf_to_torch

    g = Generator(ngf=16, transfer=True, noise_std=0.0).eval()
    g.load_state_dict({k[2:]: tf_to_torch(reader_vals[n]) for k, n in generator_map("G", "generatorA2B", True).items()})
    with torch.no_grad():
        t_outs = g(torch.from_numpy(xin).permute(0, 3, 1, 2))
    for name, a, b in zip(["pred", "pred_rec", "r4", "t4", "conf"], t_outs, tf_outs):
        assert _nhwc(a).shape == b.shape, name
        np.testing.assert_allclose(_nhwc(a), b, atol=ATOL, rtol=1e-3, err_msg=name)


def test_bicubic_2x_is_subsampling():
    """Justifies the float64 reference: TF legacy 2x bicubic == x[::2, ::2], forward and adjoint."""
    tf = _ref().ops.tf
    tf.reset_default_graph()
    rng = np.random.default_rng(4)
    x = tf.constant(rng.standard_normal((1, 16, 24, 3)).astype(np.float32))
    y = tf.image.resize_bicubic(x, [8, 12])
    w = rng.standard_normal((1, 8, 12, 3)).astype(np.float32)
    gx = tf.gradients(tf.reduce_sum(y * w), x)[0]
    with tf.Session() as sess:
        xv, yv, gv = sess.run([x, y, gx])
    np.testing.assert_array_equal(yv, xv[:, ::2, ::2])
    exact = np.zeros_like(xv)
    exact[:, ::2, ::2] = w
    np.testing.assert_array_equal(gv, exact)


@pytest.mark.parametrize("src,dst", [((32, 64), (16, 32)), ((33, 47), (16, 23)), ((20, 30), (13, 7))])
def test_legacy_bicubic(src, dst):
    tf = _ref().ops.tf  # v1-compat graph mode
    tf.reset_default_graph()
    x = np.random.default_rng(3).standard_normal((2, *src, 3)).astype(np.float32)
    with tf.Session() as sess:
        ref = sess.run(tf.image.resize_bicubic(x, dst, align_corners=False, half_pixel_centers=False))
    ours = tf_legacy_resize_bicubic(torch.from_numpy(x).permute(0, 3, 1, 2), *dst)
    np.testing.assert_allclose(_nhwc(ours), ref, atol=1e-5)


def test_gradients_match():
    """dL/dW for generator and discriminator weights, so training updates match too.

    Run in float64 on both sides: in float32 the first/last generator layers differ by ~1e-3
    relative purely from round-off (TF's oneDNN kernels sum in a different order), which
    would hide or fake a real mismatch. In float64 a formula difference cannot hide.
    """
    f = build_full_model(float64=True)
    m, tf = f.m, f.tf
    real = f.rng.uniform(-1, 1, (1, 32, 64, 6))
    fakeA = f.rng.uniform(-1, 1, (1, 32, 64, 3))
    fakeB = f.rng.uniform(-1, 1, (1, 32, 64, 3))
    g = f.sess.graph  # other tests reset the default graph; use the model's own graph
    var = {v.name[:-2]: v for v in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
    checks = [("g", "generatorA2B/g_e1_c/Conv/weights", "G_A2B.e1.weight"),
              ("g", "generatorA2B/g_d_dc_conf/Conv2d_transpose/weights", "G_A2B.d_conf.weight"),
              ("g", "generatorB2A/g_r6_rec_c1/filter", "G_B2A.dec_rec.res.0.c1.weight"),
              ("g", "generatorA2B/g_pred_c/Conv/weights", "G_A2B.dec.pred.weight"),
              ("g", "generatorA2B/g_t2/res1/g_t2_c1/Conv/weights", "G_A2B.tnet.1.c1.weight"),
              ("d0", "0_discriminatorB/d_h0_conv_scale_0_first_c/filter", "D_B.0.levels.0.0.0.0.weight"),
              ("d1", "1_discriminatorA/d_h2_conv_scale_1_c_1/Conv/weights", "D_A.1.levels.2.1.1.0.weight")]
    losses = {"g": m.g_loss, "d0": m.d_loss_item[0], "d1": m.d_loss_item[1]}
    with g.as_default():
        grad_ops = [tf.gradients(losses[l], var[n])[0] for l, n, _ in checks]
    tf_grads = f.sess.run(grad_ops, {m.real_data: real, m.fake_A_sample: fakeA, m.fake_B_sample: fakeB})
    A = torch.from_numpy(real[..., :3]).permute(0, 3, 1, 2)
    B = torch.from_numpy(real[..., 3:]).permute(0, 3, 1, 2)
    fa = torch.from_numpy(fakeA).permute(0, 3, 1, 2)
    fb = torch.from_numpy(fakeB).permute(0, 3, 1, 2)
    params = dict(f.tm.named_parameters())
    for (l, tfname, tname), tg in zip(checks, tf_grads):
        f.tm.zero_grad()
        if l == "g":
            loss = generator_forward(f.tm, A, B)[0]
        else:
            i = int(l[1])
            loss = discriminator_loss(f.tm.D_A[i], f.tm.D_B[i], A, B, fa, fb)
        loss.backward()
        ours = params[tname].grad.permute(2, 3, 1, 0).numpy()
        scale = np.abs(tg).max()
        np.testing.assert_allclose(ours / scale, tg / scale, atol=1e-9, err_msg=tname)
