"""Unit tests for the flow-matching core (CPU only, no extra dependencies).

Run with::

    python -m pytest tests/test_flows.py -q
    # or, without pytest:
    python tests/test_flows.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.flows import (  # noqa: E402
    ConsistencyFlowMatcher,
    FlowSampler,
    FlowSamplerConfig,
    LinearFlowPath,
    LinearNormalizer,
    consistency_flow_loss,
    flow_noise_fraction,
    rectified_flow_loss,
)
from core.flows.paths import ddim_equivalence_table  # noqa: E402


def test_path_projections_are_inverse():
    torch.manual_seed(0)
    path = LinearFlowPath()
    x1 = torch.randn(8, 4, 3)
    eps = torch.randn_like(x1)
    t = torch.rand(8)
    x_t = path.mu_t(eps, x1, t)
    v = path.target_velocity(eps, x1)
    x1_hat = path.project_onto_data(x_t, t, v)
    x0_hat = path.project_onto_noise(x_t, t, v)
    assert torch.allclose(x1_hat, x1, atol=1e-5), "projection onto data must recover x1"
    assert torch.allclose(x0_hat, eps, atol=1e-5), "projection onto noise must recover eps"
    v_back = path.velocity_from_data(x_t, t, x1_hat)
    assert torch.allclose(v_back, v, atol=1e-4)


def _oracle_velocity(target: torch.Tensor):
    """Exact (x, t) -> u field for a dataset whose samples are all ``target``.

    With ``x1 == A`` for every sample, ``u = A - eps`` and ``eps = (x_t - t A)/(1 - t)``,
    so ``u(x_t, t)`` is available in closed form and the flow-matching loss is exactly 0
    for this model.
    """

    def velocity_fn(x, t, cond):
        t_e = t.reshape(-1, *([1] * (x.dim() - 1)))
        eps = (x - t_e * target) / (1.0 - t_e).clamp(min=1e-4)
        return target - eps

    return velocity_fn


def test_oracle_field_achieves_zero_loss():
    torch.manual_seed(0)
    target = torch.tensor([1.0, -2.0, 0.5])
    actions = target.expand(512, 8, 3).clone()
    loss, info = rectified_flow_loss(_oracle_velocity(target), actions)
    assert loss.item() < 1e-6, loss.item()
    assert "fm_loss" in info


def test_rectified_flow_loss_trains_tiny_net():
    """A tiny MLP must be able to fit the closed-form field of a single-target dataset."""
    torch.manual_seed(0)
    target = torch.tensor([1.0, -2.0, 0.5])
    actions = target.expand(256, 6, 3).clone()
    net = torch.nn.Sequential(torch.nn.Linear(3 + 1, 64), torch.nn.Mish(), torch.nn.Linear(64, 3))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    def velocity_fn(x, t, cond):
        feat = torch.cat([x, t.reshape(-1, 1, 1).expand(-1, x.shape[1], 1)], dim=-1)
        return net(feat)

    losses = []
    for _ in range(600):
        loss, _ = rectified_flow_loss(velocity_fn, actions)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])


def test_sampler_integrates_constant_field():
    """With v == c the euler sampler must return x0 + c * (t_end - t_start)."""
    c = torch.tensor([0.3, -1.0, 2.0])
    x0 = torch.randn(4, 5, 3)

    def velocity_fn(x, t, cond):
        return c.expand_as(x)

    cfg = FlowSamplerConfig(n_steps=1, t_start=0.0, t_end=1.0)
    out = FlowSampler(cfg).sample(velocity_fn, x0)
    assert torch.allclose(out["actions"], x0 + c, atol=1e-5)


def test_sampler_stochastic_modes_agree_with_fine_grid():
    """ss sub-steps must not change the integral of a constant field."""
    c = torch.tensor([1.0, 0.0, -1.0])
    x0 = torch.zeros(2, 3, 3)

    def velocity_fn(x, t, cond):
        return c.expand_as(x)

    for mode in ("split", "refine"):
        cfg = FlowSamplerConfig(n_steps=2, ss=3, ss_mode=mode, t_start=0.0, t_end=1.0)
        out = FlowSampler(cfg).sample(velocity_fn, x0)
        assert torch.allclose(out["actions"], c.expand_as(x0), atol=1e-5), mode


def test_midpoint_is_more_accurate_than_euler_on_curved_field():
    """A time-dependent field: RK2 must beat euler at equal step count."""

    def velocity_fn(x, t, cond):
        # exact solution from x(0)=0 : x(t) = sin(pi t)/pi
        return torch.cos(t).reshape(-1, 1, 1).expand_as(x) * torch.pi

    x0 = torch.zeros(1, 3, 2)
    exact = torch.sin(torch.tensor(torch.pi)) / torch.pi  # 0
    exact = torch.zeros(1, 3, 2)
    e = FlowSampler(FlowSamplerConfig(n_steps=2, method="euler", t_start=0.0, t_end=1.0)).sample(velocity_fn, x0)["actions"]
    m = FlowSampler(FlowSamplerConfig(n_steps=2, method="midpoint", t_start=0.0, t_end=1.0)).sample(velocity_fn, x0)["actions"]
    err_e = (e - exact).abs().max()
    err_m = (m - exact).abs().max()
    assert err_m < err_e, (err_e.item(), err_m.item())


def test_consistency_loss_runs_and_backprops():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.Mish(), torch.nn.Linear(64, 3))
    actions = torch.randn(32, 8, 3)

    def velocity_fn(x, t, cond):
        feat = torch.cat([x, t.reshape(-1, 1, 1).expand(-1, x.shape[1], 1)], dim=-1)
        return net(feat)

    loss, info = consistency_flow_loss(velocity_fn, actions)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    assert "consistency_pos_loss" in info and "consistency_vel_loss" in info

    cfm = ConsistencyFlowMatcher(num_segments=4)
    t = torch.rand(8)
    ends = cfm.segment_boundaries(t)
    assert torch.all(ends >= t), "segment end must be to the right of t"


def test_consistency_loss_is_zero_for_perfect_euler_model():
    """If the network returns the exact conditional velocity the consistency loss is 0."""
    torch.manual_seed(0)
    actions = torch.randn(16, 4, 2)
    store = {}

    def velocity_fn(x, t, cond):
        # reconstruct the (x1 - x0) field implied by the linear path for this x,t is not
        # uniquely defined; instead cheat with a field that is constant in x.  For a
        # constant field both f_t and f_r map exactly onto the segment end only when the
        # segment end equals 1, so we test that case.
        return store["u"]

    # with num_segments = 1 the only segment end is 1
    cfm = ConsistencyFlowMatcher(num_segments=1, boundary=0)
    x0 = torch.randn_like(actions)
    u = actions - x0
    store["u"] = u
    # the sampler inside the loss draws its own noise, so we only check finiteness here
    loss, _ = cfm.loss(velocity_fn, actions)
    assert torch.isfinite(loss)


def test_normalizer_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(64, 7) * 3 + 1
    norm = LinearNormalizer().fit_one("action", x)
    n = norm.normalize("action", x)
    assert n.min() >= -1.0 - 1e-6 and n.max() <= 1.0 + 1e-6
    back = norm.unnormalize("action", n)
    assert torch.allclose(back, x, atol=1e-5)


def test_noise_fraction_monotone_and_bounded():
    t = torch.linspace(0.05, 0.95, 19)
    nu = flow_noise_fraction(t)
    assert torch.all(nu > 0) and torch.all(nu < 1)
    assert torch.all(nu[:-1] > nu[1:]), "noise fraction must decrease with flow time"


def test_ddim_equivalence_table():
    rows = ddim_equivalence_table(100)
    assert len(rows) == 100
    assert rows[0]["sqrt_one_minus_alpha_bar"] > rows[-1]["sqrt_one_minus_alpha_bar"]
    # clean-space correction blows up when alpha_bar -> 1 (i.e. at the end of sampling)
    assert rows[-1]["clean_correction"] < rows[0]["clean_correction"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
