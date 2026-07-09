# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SAC-SMA + Snow-17 Loss Functions and Gradient Utilities.

Provides differentiable loss functions (NSE, KGE) for model calibration
and gradient computation utilities for gradient-based optimization.

The loss functions run the full coupled Snow-17 + SAC-SMA simulation
(or standalone SAC-SMA when ``snow_module='none'``) via JAX's lax.scan
backend, enabling end-to-end autodifferentiation with ``jax.grad``.

All loss functions return negative values for minimization
(higher metric = lower loss).
"""

import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np

# Lazy JAX import
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


# =============================================================================
# LOSS FUNCTIONS (DIFFERENTIABLE)
# =============================================================================

def _mask_nan_jax(sim, obs):
    """Sanitize timesteps with missing observations for a differentiable loss.

    Streamflow observations routinely have gaps (and daily obs aligned to a
    sub-daily grid are mostly NaN). Feeding NaN into corrcoef / mean / std makes
    the whole loss — and every parameter gradient — NaN. Zeroing the masked
    positions in *both* arrays with ``where`` keeps the forward value correct and
    the backward pass finite (a NaN in an unused ``where`` branch still poisons
    the gradient, so both branches must be finite).

    Returns ``(sim_safe, obs_safe, mask, n_valid)`` where masked stats are
    ``sum(x_safe) / n_valid``.
    """
    mask = jnp.isfinite(obs) & jnp.isfinite(sim)
    n = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.where(mask, sim, 0.0), jnp.where(mask, obs, 0.0), mask, n


def nse_loss(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    use_jax: bool = True,
    day_of_year: Optional[Any] = None,
    latitude: float = 45.0,
    si: float = 100.0,
    snow_module: str = 'snow17',
) -> Any:
    """Compute negative NSE (Nash-Sutcliffe Efficiency) loss.

    Negative because optimization minimizes, and higher NSE is better.

    Args:
        params_dict: Combined parameter dictionary (Snow-17 + SAC-SMA).
        precip: Precipitation timeseries (mm/day).
        temp: Temperature timeseries (deg C).
        pet: PET timeseries (mm/day).
        obs: Observed streamflow timeseries (mm/day).
        warmup_days: Days to exclude from loss calculation.
        use_jax: Whether to use JAX backend.
        day_of_year: Julian day array (1-366). Required when
            ``snow_module='snow17'``.
        latitude: Latitude for Snow-17 melt factor seasonality.
        si: SWE threshold for areal depletion in Snow-17.
        snow_module: 'snow17' for coupled mode, 'none' for standalone.

    Returns:
        Negative NSE (loss to minimize).
    """
    from .model import simulate

    if use_jax and HAS_JAX:
        flow, _ = simulate(
            precip, temp, pet,
            params=params_dict,
            day_of_year=day_of_year,
            warmup_days=warmup_days,
            latitude=latitude,
            si=si,
            use_jax=True,
            snow_module=snow_module,
        )
        sim_eval = flow[warmup_days:]
        obs_eval = obs[warmup_days:]

        sim_s, obs_s, mask, n = _mask_nan_jax(sim_eval, obs_eval)
        obs_mean = jnp.sum(obs_s) / n
        ss_res = jnp.sum(jnp.where(mask, (sim_s - obs_s) ** 2, 0.0))
        ss_tot = jnp.sum(jnp.where(mask, (obs_s - obs_mean) ** 2, 0.0))
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse
    else:
        flow, _ = simulate(
            precip, temp, pet,
            params=params_dict,
            day_of_year=day_of_year,
            warmup_days=warmup_days,
            latitude=latitude,
            si=si,
            use_jax=False,
            snow_module=snow_module,
        )
        sim_eval = flow[warmup_days:]
        obs_eval = obs[warmup_days:]

        m = np.isfinite(sim_eval) & np.isfinite(obs_eval)
        sim_eval, obs_eval = sim_eval[m], obs_eval[m]
        ss_res = np.sum((sim_eval - obs_eval) ** 2)
        ss_tot = np.sum((obs_eval - np.mean(obs_eval)) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)
        return -nse


def kge_loss(
    params_dict: Dict[str, float],
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    use_jax: bool = True,
    day_of_year: Optional[Any] = None,
    latitude: float = 45.0,
    si: float = 100.0,
    snow_module: str = 'snow17',
) -> Any:
    """Compute negative KGE (Kling-Gupta Efficiency) loss.

    Args:
        params_dict: Combined parameter dictionary (Snow-17 + SAC-SMA).
        precip: Precipitation timeseries (mm/day).
        temp: Temperature timeseries (deg C).
        pet: PET timeseries (mm/day).
        obs: Observed streamflow timeseries (mm/day).
        warmup_days: Days to exclude from loss calculation.
        use_jax: Whether to use JAX backend.
        day_of_year: Julian day array (1-366). Required when
            ``snow_module='snow17'``.
        latitude: Latitude for Snow-17 melt factor seasonality.
        si: SWE threshold for areal depletion in Snow-17.
        snow_module: 'snow17' for coupled mode, 'none' for standalone.

    Returns:
        Negative KGE (loss to minimize).
    """
    from .model import simulate

    if use_jax and HAS_JAX:
        flow, _ = simulate(
            precip, temp, pet,
            params=params_dict,
            day_of_year=day_of_year,
            warmup_days=warmup_days,
            latitude=latitude,
            si=si,
            use_jax=True,
            snow_module=snow_module,
        )
        sim_eval = flow[warmup_days:]
        obs_eval = obs[warmup_days:]

        # NaN-safe KGE: mask missing obs, compute correlation / ratios from the
        # masked moments. The +1e-12 inside the sqrts guards the sqrt gradient
        # (1/(2*sqrt(x)) -> inf at x=0).
        sim_s, obs_s, mask, n = _mask_nan_jax(sim_eval, obs_eval)
        obs_mean = jnp.sum(obs_s) / n
        sim_mean = jnp.sum(sim_s) / n
        d_obs = jnp.where(mask, obs_s - obs_mean, 0.0)
        d_sim = jnp.where(mask, sim_s - sim_mean, 0.0)
        std_obs = jnp.sqrt(jnp.sum(d_obs ** 2) / n + 1e-12)
        std_sim = jnp.sqrt(jnp.sum(d_sim ** 2) / n + 1e-12)
        r = (jnp.sum(d_obs * d_sim) / n) / (std_obs * std_sim)
        alpha = std_sim / (std_obs + 1e-10)
        beta = sim_mean / (obs_mean + 1e-10)

        kge = 1.0 - jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2 + 1e-12)
        return -kge
    else:
        flow, _ = simulate(
            precip, temp, pet,
            params=params_dict,
            day_of_year=day_of_year,
            warmup_days=warmup_days,
            latitude=latitude,
            si=si,
            use_jax=False,
            snow_module=snow_module,
        )
        sim_eval = flow[warmup_days:]
        obs_eval = obs[warmup_days:]

        m = np.isfinite(sim_eval) & np.isfinite(obs_eval)
        sim_eval, obs_eval = sim_eval[m], obs_eval[m]
        r = np.corrcoef(sim_eval, obs_eval)[0, 1]
        alpha = np.std(sim_eval) / (np.std(obs_eval) + 1e-10)
        beta = np.mean(sim_eval) / (np.mean(obs_eval) + 1e-10)

        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return -kge


# =============================================================================
# GRADIENT FUNCTIONS
# =============================================================================

def get_nse_gradient_fn(
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    day_of_year: Optional[Any] = None,
    latitude: float = 45.0,
    si: float = 100.0,
    snow_module: str = 'snow17',
) -> Optional[Callable]:
    """Get gradient function for NSE loss.

    Returns a function that computes gradients w.r.t. parameters.

    Args:
        precip: Precipitation timeseries (fixed).
        temp: Temperature timeseries (fixed).
        pet: PET timeseries (fixed).
        obs: Observed streamflow (fixed).
        warmup_days: Warmup period.
        day_of_year: Julian day array (fixed).
        latitude: Latitude for Snow-17.
        si: SWE threshold for Snow-17.
        snow_module: 'snow17' or 'none'.

    Returns:
        Gradient function if JAX available, None otherwise.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        params_dict = dict(zip(param_names, params_array))
        return nse_loss(
            params_dict, precip, temp, pet, obs,
            warmup_days, use_jax=True,
            day_of_year=day_of_year,
            latitude=latitude, si=si,
            snow_module=snow_module,
        )

    return jax.grad(loss_fn)


def get_kge_gradient_fn(
    precip: Any,
    temp: Any,
    pet: Any,
    obs: Any,
    warmup_days: int = 365,
    day_of_year: Optional[Any] = None,
    latitude: float = 45.0,
    si: float = 100.0,
    snow_module: str = 'snow17',
) -> Optional[Callable]:
    """Get gradient function for KGE loss.

    Returns a function that computes gradients w.r.t. parameters.

    Args:
        precip: Precipitation timeseries (fixed).
        temp: Temperature timeseries (fixed).
        pet: PET timeseries (fixed).
        obs: Observed streamflow (fixed).
        warmup_days: Warmup period.
        day_of_year: Julian day array (fixed).
        latitude: Latitude for Snow-17.
        si: SWE threshold for Snow-17.
        snow_module: 'snow17' or 'none'.

    Returns:
        Gradient function if JAX available, None otherwise.
    """
    if not HAS_JAX:
        warnings.warn("JAX not available. Cannot compute gradients.")
        return None

    def loss_fn(params_array, param_names):
        params_dict = dict(zip(param_names, params_array))
        return kge_loss(
            params_dict, precip, temp, pet, obs,
            warmup_days, use_jax=True,
            day_of_year=day_of_year,
            latitude=latitude, si=si,
            snow_module=snow_module,
        )

    return jax.grad(loss_fn)
