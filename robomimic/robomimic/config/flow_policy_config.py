"""Config for the flow-matching (rectified flow / consistency flow matching) policy.

Mirrors :class:`robomimic.config.diffusion_policy_config.DiffusionPolicyConfig`, replacing
the DDPM/DDIM noise schedulers by the flow-matching knobs.
"""

from robomimic.config.base_config import BaseConfig


class FlowPolicyConfig(BaseConfig):
    ALGO_NAME = "flow_policy"

    def algo_config(self):
        """Populate ``config.algo`` -- see the diffusion policy config for the structure."""

        # optimization parameters
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.regularization.L2 = 0.00

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16

        # UNet parameters (same backbone as Diffusion Policy)
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256, 512, 1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8

        # EMA
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        # ---- flow matching -------------------------------------------------
        # objective: "fm"  = rectified flow matching (recommended default)
        #            "cfm" = FlowPolicy's consistency flow matching (few-step sampling)
        self.algo.flow.objective = "fm"
        # time rescaling fed to the sinusoidal embedding.  The consistency-FM reference
        # implementation uses 999 (its embedding assumes a 1000-step diffusion); the
        # FlowPolicy repository uses 99.  Keep training and inference consistent.
        self.algo.flow.time_scale = 99.0
        # lower bound of the sampled flow time (the ODE starts at t_start)
        self.algo.flow.t_min = 1e-2
        self.algo.flow.t_start = 1e-2
        # sampler used at inference: euler | midpoint | heun
        self.algo.flow.sampler = "euler"
        # number of ODE steps: 1 is FlowPolicy's selling point, 4-10 is safer with guidance
        self.algo.flow.num_inference_steps = 4
        # consistency flow matching hyperparameters (used when objective == "cfm")
        self.algo.flow.cfm_eps = 1e-2
        self.algo.flow.cfm_num_segments = 2
        self.algo.flow.cfm_boundary = 1.0
        self.algo.flow.cfm_delta = 1e-2
        self.algo.flow.cfm_alpha = 1e-5
