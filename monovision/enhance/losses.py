"""AU-GAN losses, ported from third_party/AU-GAN/loss_utils.py and AUGAN._build_model.

Default is LSGAN (upstream use_lsgan=True; upstream calls it `mae_criterion` but it is MSE).
"""
import torch
import torch.nn.functional as F


def gan_list(preds, target_is_real, lsgan=True):
    """Mean over scales of the per-scale GAN loss (mae_criterion_list / sce_criterion_list)."""
    total = 0.0
    for p in preds:
        t = torch.ones_like(p) if target_is_real else torch.zeros_like(p)
        total = total + (F.mse_loss(p, t) if lsgan else F.binary_cross_entropy_with_logits(p, t))
    return total / len(preds)


def abs_criterion(a, b):
    return (a - b).abs().mean()


def conf_criterion(im1, im2, conf):
    """Uncertainty-aware cycle loss (upstream 'gaussian' variant): exp(-s)*5*|a-b| + s/2."""
    loss = (im1 - im2).abs()
    if conf is not None:
        loss = torch.exp(-conf) * 5 * loss + conf / 2
    return loss.mean()


def percep_loss(transA_percep, fakeB_percep, realB_percep, fakeA_percep):
    """Asymmetric feature-matching loss on channel-averaged encoder features."""
    return (transA_percep.mean(1) - fakeB_percep.mean(1)).abs().mean() + \
           (realB_percep.mean(1) - fakeA_percep.mean(1)).abs().mean()


def generator_forward(model, real_A, real_B, L1_lambda=10.0, conf_lambda=1.0, use_uncertainty=True, lsgan=True):
    """Full generator objective. Returns (g_loss, dict of parts, dict of tensors for the D step)."""
    fake_B, rec_realA, realA_percep, transA_percep, pred_confA = model.G_A2B(real_A)
    fake_A_, rec_fakeB, fakeB_percep, _, _ = model.G_B2A(fake_B)
    fake_A, rec_realB, realB_percep, _, _ = model.G_B2A(real_B)
    fake_B_, rec_fakeA, fakeA_percep, _, _ = model.G_A2B(fake_A)

    g_adv = 0.0
    for DA, DB in zip(model.D_A, model.D_B):
        g_adv = g_adv + gan_list(DA(fake_A), True, lsgan) + gan_list(DB(fake_B), True, lsgan)

    p = percep_loss(transA_percep, fakeB_percep, realB_percep, fakeA_percep)
    a_recon = L1_lambda * abs_criterion(rec_realA, real_A)
    b_recon = L1_lambda * abs_criterion(rec_realB, real_B)
    if use_uncertainty:
        a_cycle = conf_lambda * conf_criterion(real_A, fake_A_, pred_confA)
    else:
        a_cycle = L1_lambda * abs_criterion(real_A, fake_A_)
    b_cycle = L1_lambda * abs_criterion(real_B, fake_B_)
    g_loss = g_adv + a_recon + b_recon + a_cycle + b_cycle + p
    parts = dict(g_adv=g_adv, percep=p, a_recon=a_recon, b_recon=b_recon, a_cycle=a_cycle, b_cycle=b_cycle)
    return g_loss, parts, dict(fake_A=fake_A, fake_B=fake_B)


def discriminator_loss(D_A, D_B, real_A, real_B, fake_A_sample, fake_B_sample, lsgan=True):
    """Loss for ONE (D_A, D_B) pair, as in the upstream per-discriminator d_loss_item."""
    db = 0.5 * gan_list(D_B(real_B), True, lsgan) + 0.5 * gan_list(D_B(fake_B_sample), False, lsgan)
    da = 0.5 * gan_list(D_A(real_A), True, lsgan) + 0.5 * gan_list(D_A(fake_A_sample), False, lsgan)
    return da + db
