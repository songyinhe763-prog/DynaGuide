"""Smoke test of the robomimic ``flow_policy`` integration (CPU only).

Verifies that the algorithm is registered, trains on a synthetic batch, samples actions,
and that the flow guidance hook is wired correctly (both the 4-argument flow signature and
the legacy 2-argument DynaGuide signature).

The test is skipped automatically if robomimic (or its dependencies) is not importable.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "robomimic"))

try:
    import robomimic  # noqa: F401
    from robomimic.algo import algo_factory
    from robomimic.config import config_factory
    import robomimic.utils.obs_utils as ObsUtils

    HAVE_ROBOMIMIC = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_ROBOMIMIC = False


def _make_algo(objective="fm"):
    config = config_factory("flow_policy")
    config.observation.modalities.obs.low_dim = ["proprio"]
    config.observation.modalities.obs.rgb = []
    config.observation.modalities.goal.low_dim = []
    config.observation.modalities.goal.rgb = []
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=config.observation.modalities)
    config.algo.flow.objective = objective
    config.algo.flow.num_inference_steps = 4
    config.train.seq_length = 16
    obs_shapes = OrderedDict([("proprio", (15,))])
    algo = algo_factory(algo_name="flow_policy", config=config, obs_key_shapes=obs_shapes, ac_dim=7, device="cpu")
    return algo, config


def test_registration_and_training():
    if not HAVE_ROBOMIMIC:
        print("SKIP robomimic not importable")
        return
    algo, _ = _make_algo("fm")
    sample = {
        "obs": {"proprio": torch.randn(4, 2, 15)},
        "actions": torch.rand(4, 16, 7) * 2 - 1,
    }
    algo.set_train()
    info = algo.train_on_batch(sample, epoch=0)
    loss = info["losses"]["bc_loss"]
    assert torch.isfinite(loss), loss
    assert loss.item() >= 0.0


def test_consistency_objective_trains():
    if not HAVE_ROBOMIMIC:
        return
    algo, _ = _make_algo("cfm")
    sample = {
        "obs": {"proprio": torch.randn(4, 2, 15)},
        "actions": torch.rand(4, 16, 7) * 2 - 1,
    }
    algo.set_train()
    info = algo.train_on_batch(sample, epoch=0)
    assert torch.isfinite(info["losses"]["bc_loss"])


def test_get_action_with_and_without_guidance():
    if not HAVE_ROBOMIMIC:
        return
    algo, _ = _make_algo("fm")
    algo.set_train()
    algo.train_on_batch(
        {"obs": {"proprio": torch.randn(4, 2, 15)}, "actions": torch.rand(4, 16, 7) * 2 - 1}, epoch=0
    )
    algo.set_eval()
    obs = {"proprio": torch.randn(1, 2, 15)}

    algo.reset()
    a0 = algo.get_action(obs)
    assert a0.shape == (1, 7), a0.shape

    calls = {"n": 0}

    def flow_guidance(state, actions, t, velocity):
        calls["n"] += 1
        # a simple quadratic energy in the clean-action space
        return {"delta_v": -0.05 * actions.detach(), "energy": actions.square().mean()}

    algo.reset()
    a1 = algo.get_action(obs, guidance_function=flow_guidance, guidance_type="flow", ss=2)
    assert a1.shape == (1, 7)
    assert calls["n"] > 0, "the guidance hook was never called"
    assert len(algo.guidance_list) > 0

    def legacy_guidance(state, actions):
        return -0.05 * actions.detach()

    algo.reset()
    a2 = algo.get_action(obs, guidance_function=legacy_guidance, guidance_type="flow", ss=1)
    assert a2.shape == (1, 7)


def test_guidance_type_is_validated():
    if not HAVE_ROBOMIMIC:
        return
    algo, _ = _make_algo("fm")
    algo.set_eval()
    algo.reset()
    try:
        algo.get_action({"proprio": torch.randn(1, 2, 15)}, guidance_function=lambda *a: None, guidance_type="diffusion")
    except ValueError as exc:
        assert "guidance_type" in str(exc)
        return
    raise AssertionError("expected a ValueError for guidance_type='diffusion' on a flow policy")


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

