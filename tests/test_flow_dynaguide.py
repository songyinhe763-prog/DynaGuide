"""Tests of the flow DynaGuide guidance and of the guidance energies."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.flow_dynaguide import FlowDynaGuide, GuidanceWeightSpec, make_flow_guidance  # noqa: E402
from core.flow_energies import (  # noqa: E402
    CompositeEnergy,
    DifferentiableDisassemblyModel,
    LatentMatchEnergy,
    SatellitePhysicsEnergy,
    quat_matrix_t,
)
from core.flows.samplers import FlowSampler, FlowSamplerConfig  # noqa: E402
from core.satellite.env import SatelliteEnv, module_block  # noqa: E402
from core.satellite.spec import make_sc1_spec  # noqa: E402


# ---------------------------------------------------------------- schedules ----
def test_weight_schedules():
    t = torch.tensor([1e-3, 0.25, 0.5, 0.75, 1.0])
    fp = GuidanceWeightSpec("fp")(t)
    assert torch.allclose(fp, 0.5 * (1 - t) ** 2, atol=1e-3)
    assert float(fp[0]) > float(fp[-1])
    lin = GuidanceWeightSpec("linear")(t)
    assert torch.allclose(lin, 1 - t, atol=1e-4)
    ddim = GuidanceWeightSpec("ddim", clamp_max=50.0)(t)
    assert float(ddim[-1]) < 1e-3, "strict DDIM schedule must anneal to zero at data"
    assert float(ddim[0]) <= 50.0 + 1e-6, "must be clamped at t -> 0"
    const = GuidanceWeightSpec("const")(t)
    assert torch.allclose(const, torch.ones_like(t))


def test_guidance_reduces_quadratic_energy():
    """A quadratic energy must pull the generated chunk towards its minimiser."""
    target = torch.full((6, 3), 1.5)

    def energy(actions, proprio, ctx):
        return ((actions - target) ** 2).mean(), {}

    x0 = torch.zeros(1, 6, 3)

    def velocity_fn(x, t, cond):
        return torch.zeros_like(x)

    cfg = FlowSamplerConfig(n_steps=4, t_start=1e-2, t_end=1.0)
    base = FlowSampler(cfg).sample(velocity_fn, x0)["actions"]
    guide = make_flow_guidance(energy, scale=2.0, weight_mode="const")
    guided = FlowSampler(cfg).sample(velocity_fn, x0, guidance=guide, ctx={})["actions"]
    assert (guided - target).abs().mean() < (base - target).abs().mean(), (guided, base)


def test_endpoint_guidance_matches_analytic_gradient():
    """With stop-grad velocity the guidance must equal -s*w*grad_{x1} J."""
    A = torch.tensor([[0.5, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.1]])
    target = torch.ones(4, 3)

    def velocity_fn(x, t, cond):
        return x @ A.T

    def energy(actions, proprio, ctx):
        return ((actions - target) ** 2).sum(), {}

    guide = FlowDynaGuide(energy=energy, scale=1.0, weight=GuidanceWeightSpec("const"), mode="endpoint", clip_norm=None, clip_value=None)
    x = torch.randn(1, 4, 3)
    t_vec = torch.tensor([0.4])
    v = velocity_fn(x, t_vec, None)
    out = guide(x, t_vec, v, {})
    x1 = x + (1 - 0.4) * v  # detached estimate
    expected = -2.0 * (x1 - target)
    assert torch.allclose(out["delta_v"], expected, atol=1e-5), (out["delta_v"], expected)


def test_jacobian_flag_adds_velocity_sensitivity():
    """With ``jacobian=True`` the transport must include ``I + (1-t) dv/dx``."""
    A = torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]])
    target = torch.ones(4, 3)

    def velocity_fn(x, t, cond):
        return x @ A.T

    def energy(actions, proprio, ctx):
        return ((actions - target) ** 2).sum(), {}

    x = torch.randn(1, 4, 3)
    t_vec = torch.tensor([0.5])

    # stop-grad variant: the velocity needs no graph
    v_stop = velocity_fn(x, t_vec, None)
    g_stop = FlowDynaGuide(energy, 1.0, GuidanceWeightSpec("const"), jacobian=False, clip_norm=None, clip_value=None)(x, t_vec, v_stop, {})["delta_v"]

    # full Jacobian variant: the velocity must actually depend on x through the graph
    x_g = x.clone().requires_grad_(True)
    v_g = velocity_fn(x_g, t_vec, None)
    g_full = FlowDynaGuide(energy, 1.0, GuidanceWeightSpec("const"), jacobian=True, clip_norm=None, clip_value=None)(x_g, t_vec, v_g, {})["delta_v"]

    x1 = x + 0.5 * velocity_fn(x, t_vec, None)
    grad_x1 = 2.0 * (x1 - target)
    expected_stop = -grad_x1
    expected_full = -(grad_x1 + 0.5 * torch.einsum("ji,bkj->bki", A, grad_x1))
    assert torch.allclose(g_stop, expected_stop, atol=1e-5)
    assert torch.allclose(g_full, expected_full, atol=1e-5)


def test_gradient_clipping_bounds_the_correction():
    def energy(actions, proprio, ctx):
        return (actions.sum() * 1e6, {})

    guide = FlowDynaGuide(energy, 1.0, GuidanceWeightSpec("const"), clip_norm=0.5)
    x = torch.zeros(1, 4, 3)
    out = guide(x, torch.tensor([0.5]), torch.zeros_like(x), {})
    assert float(out["grad_norm"]) <= 0.5 + 1e-5


def test_velocity_mode_is_the_dynaguide_literal_variant():
    """``mode="velocity"`` must evaluate the energy on the noisy chunk itself."""

    def energy(actions, proprio, ctx):
        return (actions.sum(), {})

    guide = FlowDynaGuide(energy, 1.0, GuidanceWeightSpec("const"), mode="velocity", clip_norm=None, clip_value=None)
    x = torch.randn(1, 4, 3)
    out = guide(x, torch.tensor([0.3]), torch.zeros_like(x), {})
    assert torch.allclose(out["delta_v"], -torch.ones_like(x))


# ------------------------------------------------------------------ energies ----
def test_latent_energy_matches_dynaguide_formula():
    positives = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    negatives = torch.tensor([[5.0, 5.0]])
    predictor = lambda state, actions: actions.mean(dim=1)  # (B,2)
    energy = LatentMatchEnergy(predictor, positives, negatives, alpha=2.0)
    actions = torch.tensor([[[0.2, -0.1], [0.2, -0.1]]])
    cost, info = energy(actions, None, {})
    z = torch.tensor([[0.2, -0.1]])
    d_pos = torch.cdist(positives, z).squeeze(-1)
    d_neg = torch.cdist(negatives, z).squeeze(-1)
    expected = torch.logsumexp(-d_pos / 2.0, dim=0) - torch.logsumexp(-d_neg / 2.0, dim=0)
    assert torch.allclose(cost, expected, atol=1e-6)


def test_physics_energy_prefers_the_correct_protocol():
    """Twist-then-pull must be cheaper than yanking at a bolted module."""
    spec = make_sc1_spec()
    env = SatelliteEnv(spec=spec, seed=0, randomize=False)
    obs = env.reset(task="payload_box")["proprio"]
    model = DifferentiableDisassemblyModel(spec)
    idx = spec.module_names.index("payload_box")
    axis = torch.tensor(spec.module("payload_box").joint.axis_np, dtype=torch.float32)

    H = 16
    naive = torch.zeros(H, 7)
    naive[:, 0:3] = axis.expand(H, 3) * 0.5  # straight yank, no unscrewing
    naive[:, 6] = 1.0

    good = torch.zeros(H, 7)
    good[:8, 3:6] = axis.expand(8, 3) * 1.0  # quarter turn first (0.2 rad / step)
    good[8:, 0:3] = axis.expand(H - 8, 3) * 0.4
    good[:, 6] = 1.0

    r_naive = model.rollout(naive, torch.from_numpy(obs), idx)
    r_good = model.rollout(good, torch.from_numpy(obs), idx)
    assert float(r_good["pull_ratio"]) > float(r_naive["pull_ratio"])
    assert float(r_naive["force_max"]) >= float(r_good["force_max"])
    assert float(r_good["attitude_drift"]) < float(r_naive["attitude_drift"])

    energy = SatellitePhysicsEnergy(spec)
    e_naive, terms_naive = energy(naive, torch.from_numpy(obs), {"module_index": idx})
    e_good, terms_good = energy(good, torch.from_numpy(obs), {"module_index": idx})
    assert float(e_good) < float(e_naive), (float(e_good), float(e_naive))


def test_composite_energy_sums_weighted_terms():
    a = SatellitePhysicsEnergy(make_sc1_spec(), name="physics", weight=2.0)
    b = SatellitePhysicsEnergy(make_sc1_spec(), name="physics2", weight=0.5)
    comp = CompositeEnergy([a, b])
    spec = make_sc1_spec()
    env = SatelliteEnv(spec=spec, seed=1, randomize=False)
    obs = env.reset(task="solar_panel")["proprio"]
    actions = torch.zeros(16, 7)
    actions[:, 6] = 1.0
    total, info = comp(actions, torch.from_numpy(obs), {"module_index": 1})
    single, _ = a(actions, torch.from_numpy(obs), {"module_index": 1})
    assert abs(float(total) - 2.5 * float(single)) < 1e-4


def test_quat_matrix_matches_numpy():
    from core.satellite import math3d as m3

    rng = np.random.default_rng(0)
    rv = rng.normal(size=3) * 0.4
    q = m3.rotvec_quat(rv)
    R_np = m3.quat_matrix(q)
    R_t = quat_matrix_t(torch.from_numpy(q).unsqueeze(0))[0].numpy()
    assert np.allclose(R_np, R_t, atol=1e-6)


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

