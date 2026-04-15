"""
Real-Time Chunking (RTC) for LAP (π₀-based) inference.

JAX-compatible implementation ported from X-VLA's rtc_xvla.py for LAP's
flow-matching denoising loop.  Designed for use inside ``jax.lax.while_loop``.

Reference:
    https://www.physicalintelligence.company/download/real_time_chunking.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp


class RTCSchedule(str, Enum):
    LINEAR = "linear"
    EXP = "exp"
    ONES = "ones"
    ZEROS = "zeros"


@dataclass
class LAPRTCConfig:
    enabled: bool = False
    max_guidance_weight: float = 1.0
    execution_horizon: int = 10
    schedule: str = "linear"

    @classmethod
    def from_dict(cls, d: dict | None) -> "LAPRTCConfig":
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def get_prefix_weights(schedule: str, start: int, end: int, total: int) -> jnp.ndarray:
    """Compute prefix weights as a 1-D array of shape (total,).

    Pure-function version suitable for JAX tracing.
    """
    start = min(start, end)
    sched = RTCSchedule(schedule)

    if sched == RTCSchedule.ZEROS:
        w = jnp.zeros(total)
        w = w.at[:start].set(1.0)
    elif sched == RTCSchedule.ONES:
        w = jnp.ones(total)
        w = w.at[end:].set(0.0)
    elif sched == RTCSchedule.LINEAR:
        lin = _linweights(start, end, total)
        w = _add_trailing_zeros(lin, total, end)
        w = _add_leading_ones(w, start, total)
    elif sched == RTCSchedule.EXP:
        lin = _linweights(start, end, total)
        lin = lin * jnp.expm1(lin) / (math.e - 1)
        w = _add_trailing_zeros(lin, total, end)
        w = _add_leading_ones(w, start, total)
    else:
        raise ValueError(f"Unknown schedule: {sched}")
    return w


def _linweights(start: int, end: int, total: int) -> jnp.ndarray:
    skip = max(total - end, 0)
    n = total - skip - start
    if end <= start or n <= 0:
        return jnp.array([])
    return jnp.linspace(1, 0, n + 2)[1:-1]


def _add_trailing_zeros(weights: jnp.ndarray, total: int, end: int) -> jnp.ndarray:
    zlen = total - end
    if zlen <= 0:
        return weights
    return jnp.concatenate([weights, jnp.zeros(zlen)])


def _add_leading_ones(weights: jnp.ndarray, start: int, total: int) -> jnp.ndarray:
    olen = min(start, total)
    if olen <= 0:
        return weights
    return jnp.concatenate([jnp.ones(olen), weights])


def guide_prediction(
    action: jnp.ndarray,
    prev_chunk_left_over: jnp.ndarray,
    prev_chunk_valid: bool | jnp.ndarray,
    time_value: float | jnp.ndarray,
    max_guidance_weight: float,
    prefix_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Apply RTC guidance to the current action estimate.

    This is a pure function suitable for use inside ``jax.lax.while_loop``.
    The caller pre-computes ``prefix_weights`` once outside the loop.

    Args:
        action: Current action estimate, shape (B, T, A).
        prev_chunk_left_over: Zero-padded previous chunk tail, shape (B, T, A).
            Must be same shape as action (pre-padded by caller).
        prev_chunk_valid: Whether prev_chunk_left_over is valid (False on first chunk).
        time_value: Noise level in [0, 1] where 1 = noisy, 0 = clean.
            For LAP: this is the current ``time`` value directly (1→0).
        max_guidance_weight: Maximum guidance weight (clamped to 1.0 for direct blending).
        prefix_weights: Pre-computed weights, shape (T,).

    Returns:
        Guided action, same shape as *action*.
    """
    B, T, A = action.shape

    # Guidance weight: strong when noisy, weak when clean
    gw = jnp.minimum(max_guidance_weight * time_value, 1.0)

    # Broadcast prefix weights: (T,) → (1, T, 1)
    w = prefix_weights[None, :, None]

    # Direct guidance: blend toward previous tail
    err = (prev_chunk_left_over - action) * w
    guided = action + gw * err

    # If prev_chunk is not valid (first chunk), return original action
    return jnp.where(prev_chunk_valid, guided, action)
