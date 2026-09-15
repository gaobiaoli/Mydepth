import torch

from loss import absrel_optimal_log_scale, priorbim_multiscale_loss


def make_batch():
    da3 = torch.full((1, 1, 12, 12), 2.0)
    pattern = torch.linspace(-0.2, 0.2, 12).view(1, 1, 1, 12)
    gt = da3 * pattern.exp()
    return {
        "da3_depth": da3,
        "bim_depth": torch.full_like(da3, 2.5),
        "bim_valid": torch.ones_like(da3),
        "gt_depth": gt,
        "gt_valid": torch.ones_like(da3),
    }


def test_each_stage_is_supervised_in_depth_space():
    batch = make_batch()
    scale, supported = absrel_optimal_log_scale(
        batch["da3_depth"],
        batch["gt_depth"],
        batch["gt_valid"],
    )
    assert bool(supported.all())
    residual = batch["gt_depth"].log() - batch["da3_depth"].log() - scale
    output = {
        "log_scale": scale,
        "log_residual_r36": residual,
        "log_residual_r72": torch.zeros_like(residual),
        "log_residual_r144": torch.zeros_like(residual),
    }
    losses = priorbim_multiscale_loss(output, batch)

    assert torch.allclose(losses["depth_r36"], torch.zeros(()), atol=1e-7)
    assert torch.allclose(losses["depth_r72"], torch.zeros(()), atol=1e-7)
    assert torch.allclose(losses["depth_r144"], torch.zeros(()), atol=1e-7)
    assert torch.allclose(losses["scale"], torch.zeros(()), atol=1e-7)


def test_stage_depth_losses_follow_the_detached_cumulative_chain():
    batch = make_batch()
    log_scale = torch.zeros(1, 1, 1, 1, requires_grad=True)
    residual36 = torch.zeros(1, 1, 3, 3, requires_grad=True)
    residual72 = torch.zeros(1, 1, 6, 6, requires_grad=True)
    residual144 = torch.zeros(1, 1, 12, 12, requires_grad=True)
    output = {
        "log_scale": log_scale,
        "log_residual_r36": residual36,
        "log_residual_r72": residual72,
        "log_residual_r144": residual144,
    }
    losses = priorbim_multiscale_loss(output, batch)
    assert torch.allclose(
        losses["depth"],
        0.2 * losses["depth_r36"]
        + 0.3 * losses["depth_r72"]
        + 0.5 * losses["depth_r144"],
    )

    gradients = torch.autograd.grad(
        losses["depth_r36"],
        (log_scale, residual36, residual72, residual144),
        allow_unused=True,
        retain_graph=True,
    )
    assert gradients[0] is None
    assert gradients[1] is not None
    assert gradients[2] is None
    assert gradients[3] is None

    gradients = torch.autograd.grad(
        losses["depth_r72"],
        (log_scale, residual36, residual72, residual144),
        allow_unused=True,
        retain_graph=True,
    )
    assert gradients[0] is None
    assert gradients[1] is not None
    assert gradients[2] is not None
    assert gradients[3] is None

    gradients = torch.autograd.grad(
        losses["depth_r144"],
        (log_scale, residual36, residual72, residual144),
        allow_unused=True,
    )
    assert gradients[0] is None
    assert gradients[1] is None
    assert gradients[2] is None
    assert gradients[3] is not None
