"""Flow-matching (rectified flow / consistency flow matching) policy for robomimic.

This is the *flow* sibling of ``robomimic/algo/diffusion_policy.py`` and is what makes
DynaGuide applicable to a velocity-field policy:

* the network predicts a **velocity** ``v_theta(x_t, t)`` instead of an epsilon;
* training uses the rectified-flow objective (or FlowPolicy's consistency flow matching);
* inference integrates the probability-flow ODE with :class:`core.flows.FlowSampler`;
* the guidance hook receives ``(state, actions, t, velocity)`` and returns a **velocity
  correction** (see :func:`core.flow_dynaguide.calculate_flow_classifier_guidance`), which
  is the mathematically consistent port of DynaGuide's

  .. code-block:: python

      noise_pred = noise_pred - (1 - alphas_cumprod[k]).sqrt() * scaled_grad

  to a velocity field.  See ``docs/FLOW_DYNAGUIDE.md`` for the derivation of the
  ``lambda(t)`` weight schedule.

Config: see ``robomimic/config/flow_policy_config.py`` and
``configs/flow_policy_image_calvin_ddim.json``.
"""
from __future__ import annotations

import copy
import inspect
from collections import OrderedDict, deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import PolicyAlgo, register_algo_factory_func

from core.flows.consistency import ConsistencyFlowMatcher, rectified_flow_loss
from core.flows.nets import ConditionalUnet1D
from core.flows.samplers import FlowSampler, FlowSamplerConfig


def replace_bn_with_gn(root_module: nn.Module, features_per_group: int = 16) -> nn.Module:
    """Replace every ``BatchNorm2d`` by a ``GroupNorm`` (needed with EMA)."""
    bn_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if isinstance(m, nn.BatchNorm2d)]
    for *parent, key in bn_list:
        parent_module = root_module.get_submodule(".".join(parent)) if parent else root_module
        src = parent_module[int(key)] if isinstance(parent_module, nn.Sequential) else getattr(parent_module, key)
        tgt = nn.GroupNorm(num_groups=max(1, src.num_features // features_per_group), num_channels=src.num_features)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(key)] = tgt
        else:
            setattr(parent_module, key, tgt)
    return root_module


class FlowEMA:
    """Minimal power-law EMA (the same schedule as ``diffusers.training_utils.EMAModel``)."""

    def __init__(self, parameters, power: float = 0.75):
        self.power = power
        self.num_updates = 0
        self.shadow = {k: v.detach().clone() for k, v in parameters}

    @torch.no_grad()
    def step(self, parameters):
        self.num_updates += 1
        decay = min(1.0 - 1.0 / (self.num_updates + 1), 0.999) ** (1.0 + self.power)
        for k, v in parameters:
            self.shadow[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, parameters):
        for k, v in parameters:
            v.copy_(self.shadow[k])

    def state_dict(self):
        return {"shadow": self.shadow, "num_updates": self.num_updates, "power": self.power}

    def load_state_dict(self, state):
        self.shadow = state["shadow"]
        self.num_updates = state.get("num_updates", 0)
        self.power = state.get("power", self.power)


@register_algo_factory_func("flow_policy")
def algo_config_to_class(algo_config):
    """Map the ``flow_policy`` config to the algorithm class."""
    if algo_config.unet.enabled:
        return FlowPolicyUNet, {}
    raise RuntimeError("flow_policy currently only supports the UNet backbone")


class FlowPolicyUNet(PolicyAlgo):
    """Rectified-flow / consistency-flow-matching action-chunk policy with guidance."""

    # ------------------------------------------------------------------ setup ----
    def _create_networks(self):
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        if len(self.goal_shapes.keys()) > 0:
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        obs_encoder = replace_bn_with_gn(obs_encoder)  # IMPORTANT: EMA compatibility

        obs_dim = obs_encoder.output_shape()[0]
        velocity_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim * self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=tuple(self.algo_config.unet.down_dims),
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups,
        )

        nets = nn.ModuleDict({"policy": nn.ModuleDict({"obs_encoder": obs_encoder, "velocity_net": velocity_net})})
        nets = nets.float().to(self.device)

        ema = None
        if self.algo_config.ema.enabled:
            ema = FlowEMA(list(nets.named_parameters()), power=self.algo_config.ema.power)

        self.nets = nets
        self._shadow_nets = copy.deepcopy(self.nets).eval()
        self._shadow_nets.requires_grad_(False)
        self.ema = ema

        # flow machinery
        self.time_scale = float(self.algo_config.flow.get("time_scale", 1.0))
        self.objective = str(self.algo_config.flow.get("objective", "fm"))
        self.t_min = float(self.algo_config.flow.get("t_min", 1e-2))
        self.consistency = ConsistencyFlowMatcher(
            eps=float(self.algo_config.flow.get("cfm_eps", 1e-2)),
            num_segments=int(self.algo_config.flow.get("cfm_num_segments", 2)),
            boundary=float(self.algo_config.flow.get("cfm_boundary", 1.0)),
            delta=float(self.algo_config.flow.get("cfm_delta", 1e-2)),
            alpha=float(self.algo_config.flow.get("cfm_alpha", 1e-5)),
        )

        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.corrections_list = []
        self.guidance_list = []

    def process_batch_for_training(self, batch):
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            if not torch.all(in_range).item():
                raise ValueError('"actions" must be in range [-1,1] for flow policies! Check hdf5_normalize_action.')
            self.action_check_done = True
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    # --------------------------------------------------------------- training ----
    def train_on_batch(self, batch, epoch, validate=False):
        action_dim = self.ac_dim
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super().train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]

            inputs = {"obs": batch["obs"], "goal": batch.get("goal_obs", None)}
            for k in self.obs_shapes:
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            obs_features = TensorUtils.time_distributed(
                inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True
            )
            obs_cond = obs_features.flatten(start_dim=1)

            def velocity_fn(x_t, t, cond):
                return self.nets["policy"]["velocity_net"](x_t, t * self.time_scale, global_cond=cond)

            if self.objective == "cfm":
                loss, loss_info = self.consistency.loss(velocity_fn, actions, obs_cond)
            else:
                loss, loss_info = rectified_flow_loss(velocity_fn, actions, obs_cond, t_min=self.t_min)

            losses = {"bc_loss": loss}
            info["losses"] = TensorUtils.detach(losses)
            if not validate:
                grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets, optim=self.optimizers["policy"], loss=loss
                )
                if self.ema is not None:
                    self.ema.step(list(self.nets.named_parameters()))
                info["policy_grad_norms"] = grad_norms
        return info

    def log_info(self, info):
        log = super().log_info(info)
        log["Loss"] = info["losses"]["bc_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    # -------------------------------------------------------------- inference ----
    def reset(self):
        Ta = self.algo_config.horizon.action_horizon
        # a slightly longer chunk helps the guidance (same trick as DynaGuide)
        self.action_queue = deque(maxlen=max(Ta, 14))
        self.obs_queue = deque(maxlen=self.algo_config.horizon.observation_horizon)

    def _encode_obs(self, obs_dict, goal_dict=None, nets=None):
        nets = nets if nets is not None else self.nets
        inputs = {"obs": obs_dict, "goal": goal_dict}
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = torch.unsqueeze(inputs["obs"][k], dim=1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        return obs_features.flatten(start_dim=1)

    def _velocity_fn(self, obs_cond):
        def fn(x, t, cond=None):
            return self.nets["policy"]["velocity_net"](x, t * self.time_scale, global_cond=obs_cond)

        return fn

    def _wrap_guidance(self, guidance_function, state):
        """Adapt either the flow (4-arg) or the DDIM-style (2-arg) guidance signature.

        这是“向后兼容”的关键小函数。原 DynaGuide 的引导函数签名是
        (state, actions)，流匹配版多一个当前时刻 t 和当前速度 v；这里按参数个数分派，
        于是两代实现都能直接挂到同一个策略上。
        """
        try:
            n_params = len(
                [p for p in inspect.signature(guidance_function).parameters.values() if p.kind != p.VAR_KEYWORD]
            )
        except (TypeError, ValueError):  # builtins / wrapped callables
            n_params = 4

        if n_params >= 4:

            def fn(x, t, v, ctx=None):
                out = guidance_function(state, x, t, v)
                return out["delta_v"] if isinstance(out, dict) else out

            return fn

        def fn(x, t, v, ctx=None):  # DynaGuide-literal: cost on the noisy chunk
            out = guidance_function(state, x)
            return out["delta_v"] if isinstance(out, dict) else out

        return fn

    def _sample_chunk(self, obs_dict, goal_dict, guidance_function, ss, n_steps, method, t_start):
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        nets = self.nets
        if self.ema is not None:
            self.ema.copy_to(list(self._shadow_nets.named_parameters()))
            nets = self._shadow_nets

        obs_cond = self._encode_obs(obs_dict, goal_dict, nets=nets)
        B = obs_cond.shape[0]
        x0 = torch.randn(B, Tp, self.ac_dim, device=self.device)
        sampler = FlowSampler(
            FlowSamplerConfig(
                n_steps=n_steps,
                method=method,
                t_start=t_start,
                t_end=1.0,
                ss=ss,
                time_scale=self.time_scale,
            ).validate()
        )
        guidance = None
        if guidance_function is not None:
            guidance = self._wrap_guidance(guidance_function, obs_dict)
        out = sampler.sample(
            lambda x, t, cond: nets["policy"]["velocity_net"](x, t * self.time_scale, global_cond=obs_cond),
            x0,
            guidance=guidance,
            record=guidance is not None,
        )
        naction = out["actions"]
        if out.get("trace") is not None:
            trace = out["trace"].as_dict()
            self.corrections_list = trace["states"]
            self.guidance_list = trace["guidance"]
        self.last_nfe = out["nfe"]
        start = To - 1
        end = start + max(Ta, 14)
        return naction[:, start:end]

    def get_action(self, obs_dict, goal_dict=None, guidance_function=None, guidance_type=None, ss=1):
        """Return one action; ``guidance_type`` is ``"flow"`` for the flow guided sampler."""
        Ta = self.algo_config.horizon.action_horizon
        if not self.nets.training:
            pass
        if len(self.action_queue) == 0:
            n_steps = int(self.algo_config.flow.num_inference_steps)
            method = str(self.algo_config.flow.sampler)
            t_start = float(self.algo_config.flow.t_start)
            if guidance_function is not None and guidance_type not in (None, "flow"):
                raise ValueError(
                    f"flow_policy only supports guidance_type='flow' (got {guidance_type!r}); "
                    "use core.flow_dynaguide.calculate_flow_classifier_guidance"
                )
            sequence = self._sample_chunk(obs_dict, goal_dict, guidance_function, ss, n_steps, method, t_start)
            self.action_queue.extend(sequence[0])
        action = self.action_queue.popleft()
        return action.unsqueeze(0)

    # ------------------------------------------------------------- checkpoint ----
    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None and self.ema is not None:
            self.ema.load_state_dict(model_dict["ema"])
