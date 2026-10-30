"""TF-free tests: shapes, pt-AUGAN variants, train/infer smoke runs on tiny synthetic data (CPU)."""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from monovision.enhance import infer, train
from monovision.enhance.layers import Conv2dTF, ConvTranspose2dTF, UpsampleConv
from monovision.enhance.losses import generator_forward
from monovision.enhance.models import AUGAN, Discriminator, Generator


@pytest.mark.parametrize("upsample,demod", [("deconv", False), ("upconv", False), ("upconv", True)])
@pytest.mark.parametrize("hw", [(32, 64), (36, 52)])
def test_generator_shapes(upsample, demod, hw):
    g = Generator(ngf=8, transfer=True, upsample=upsample, demod=demod).eval()
    pred, rec, r4, t4, conf = g(torch.randn(2, 3, *hw))
    assert pred.shape == rec.shape == (2, 3, *hw)
    assert conf.shape == (2, 1, *hw) and (conf > 0).all()
    assert r4.shape == t4.shape == (2, 32, hw[0] // 4, hw[1] // 4)
    assert pred.abs().max() <= 1


def test_b2a_has_no_tnet_or_conf():
    g = Generator(ngf=8, transfer=False)
    assert not hasattr(g, "tnet") and not hasattr(g, "d_conf")
    assert g(torch.randn(1, 3, 32, 32))[4] is None


def test_demod_unit_norm():
    c = Conv2dTF(4, 6, 3, demod=True)
    w = c.effective_weight()
    torch.testing.assert_close(w.pow(2).sum([1, 2, 3]), torch.ones(6), atol=1e-5, rtol=0)


def test_deconv_requires_no_demod():
    with pytest.raises(ValueError):
        Generator(ngf=8, upsample="deconv", demod=True)


def test_up_layers_double_size():
    x = torch.randn(1, 4, 5, 7)
    assert ConvTranspose2dTF(4, 2)(x).shape == (1, 2, 10, 14)
    assert UpsampleConv(4, 2)(x).shape == (1, 2, 10, 14)


def test_discriminator_outputs():
    outs = Discriminator(ndf=8)(torch.randn(1, 3, 64, 128))
    # scale 0 ends at 1/16, scale 1 at 1/32 (upstream cross-scale wiring; loss parity checked vs TF)
    assert [tuple(o.shape) for o in outs] == [(1, 1, 4, 8), (1, 1, 2, 4)]


def test_noise_only_in_train_mode():
    g = Generator(ngf=8, transfer=False)
    x = torch.randn(1, 3, 32, 32)
    g.eval()
    a, b = g(x)[1], g(x)[1]
    torch.testing.assert_close(a, b)
    g.train()
    assert not torch.allclose(g(x)[1], g(x)[1])


def test_loss_backward_all_generator_params_get_grad():
    m = AUGAN(ngf=8, ndf=16, upsample="upconv", demod=True)
    loss, parts, _ = generator_forward(m, torch.rand(1, 3, 32, 64) * 2 - 1, torch.rand(1, 3, 32, 64) * 2 - 1)
    loss.backward()
    for name, p in list(m.G_A2B.named_parameters()) + list(m.G_B2A.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def _make_dataset(root, n=4, h=40, w=80):
    rng = np.random.default_rng(0)
    for split in ["trainA", "trainB", "testA", "testB"]:
        os.makedirs(os.path.join(root, split))
        for i in range(n):
            arr = (rng.random((h, w, 3)) * (60 if split.endswith("A") else 255)).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(root, split, f"{i:04d}.jpg"))


@pytest.mark.parametrize("variant", [[], ["--upsample", "upconv", "--demod", "true", "--aspect", "1"]])
def test_train_then_infer_smoke(tmp_path, variant):
    data = tmp_path / "data"
    _make_dataset(str(data))
    args = train.parse_args(["--dataset_dir", str(data), "--checkpoint_root", str(tmp_path / "ck"),
                             "--experiment_name", "smoke", "--fine_size", "32", "--load_size", "36",
                             "--ngf", "8", "--ndf", "16", "--epoch", "1", "--epoch_step", "1",
                             "--num_workers", "0", "--max_iters", "3", "--print_freq", "2",
                             "--device", "cpu", *variant])
    train.train(args)
    ck = tmp_path / "ck" / "smoke" / "last.pt"
    assert ck.exists()
    out = tmp_path / "out"
    infer.main(["--weights", str(ck), "--input", str(data / "testA"), "--output", str(out), "--device", "cpu"])
    outs = sorted(os.listdir(out))
    assert outs == sorted(os.listdir(data / "testA"))
    assert Image.open(out / outs[0]).size == (80, 40)


def test_config_file_and_bool_parsing(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("use_uncertainty: false\nupsample: upconv\n")
    a = train.parse_args(["--config", str(cfg)])
    assert a.use_uncertainty is False and a.upsample == "upconv"
    a = train.parse_args(["--config", str(cfg), "--use_uncertainty", "True"])
    assert a.use_uncertainty is True


def test_instance_norm_1x1_returns_offset():
    from monovision.enhance.layers import InstanceNormTF

    n = InstanceNormTF(3).train()
    torch.testing.assert_close(n(torch.randn(2, 3, 1, 1)), n.bias[None, :, None, None].expand(2, 3, 1, 1))
