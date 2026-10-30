"""Flow-matching building blocks used by the DynaGuide-for-flow implementation.

The modules here are intentionally dependency-free (only ``torch``) so that they can be
unit-tested on CPU and reused from both the standalone satellite experiments and the
robomimic integration.

Layout
------
``paths``         probability paths (linear interpolant / rectified flow) and their
                  x1 / x0 projections, SNR based weight helpers.
``consistency``   conditional *consistency* flow matching losses (FlowPolicy, AAAI'25).
``samplers``      ODE samplers (euler / midpoint / heun) with a guidance hook and the
                  DynaGuide "stochastic sampling" (``ss``) inner refinement loop.
``normalizer``    simple per-key linear (min-max) normalizer for actions / observations.
"""

from core.flows.paths import (
    LinearFlowPath,
    pad_t_like_x,
    time_embedding_scale,
    flow_noise_fraction,
)
from core.flows.consistency import (
    ConsistencyFlowMatcher,
    rectified_flow_loss,
    consistency_flow_loss,
)
from core.flows.samplers import FlowSampler, FlowSamplerConfig, SamplerTrace
from core.flows.normalizer import LinearNormalizer

__all__ = [
    "LinearFlowPath",
    "pad_t_like_x",
    "time_embedding_scale",
    "flow_noise_fraction",
    "ConsistencyFlowMatcher",
    "rectified_flow_loss",
    "consistency_flow_loss",
    "FlowSampler",
    "FlowSamplerConfig",
    "SamplerTrace",
    "LinearNormalizer",
]
