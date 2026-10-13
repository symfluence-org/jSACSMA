# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Sacramento Soil Moisture Accounting (SAC-SMA) Model — Dual-Backend (JAX + NumPy).

Burnash (1995) dual-layer tension/free water conceptual model.
All branching uses ``xp.where()`` for JAX differentiability.

Key physics:
1. ET demand sequence: UZTWC -> UZFWC -> LZTWC -> ADIMC
2. Percolation with ZPERC/REXP demand curve
3. Surface runoff: direct (PCTIM), saturation-excess (ADIMP), UZ overflow
4. Interflow from upper zone free water
5. Primary + supplemental baseflow from lower zone
6. Deep recharge (SIDE = non-channel/channel baseflow ratio)

Two interfaces:
- Functional API: ``sacsma_step()`` with ``xp`` parameter
- ``lax.scan``-based: ``sacsma_simulate_jax()``

References:
    Burnash, R.J.C. (1995). The NWS River Forecast System - Catchment Modeling.
    Computer Models of Watershed Hydrology, 311-366.
"""

from typing import Any, NamedTuple, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
    jax = None
    lax = None

from .parameters import SacSmaParameters

__all__ = [
    'HAS_JAX',
    'SacSmaState',
    'sacsma_step',
    'sacsma_step_with_fluxes',
    'sacsma_simulate_jax',
    'sacsma_simulate_numpy',
    'sacsma_simulate',
    '_create_default_state',
]


class SacSmaState(NamedTuple):
    """SAC-SMA model state variables (all in mm). Fields use Any for JAX."""
    uztwc: Any   # Upper zone tension water content
    uzfwc: Any   # Upper zone free water content
    lztwc: Any   # Lower zone tension water content
    lzfpc: Any   # Lower zone primary free water content
    lzfsc: Any   # Lower zone supplemental free water content
    adimc: Any   # Additional impervious area content


def _create_default_state(
    params: SacSmaParameters,
    use_jax: bool = False,
) -> SacSmaState:
    """Create initial state at 50% capacity.

    Args:
        params: SAC-SMA parameters
        use_jax: Whether to use JAX arrays

    Returns:
        SacSmaState at 50% of storage capacities
    """
    uztwc = params.UZTWM * 0.5
    uzfwc = params.UZFWM * 0.5
    lztwc = params.LZTWM * 0.5
    lzfpc = params.LZFPM * 0.5
    lzfsc = params.LZFSM * 0.5
    adimc = (params.UZTWM + params.LZTWM) * 0.5

    if use_jax and HAS_JAX:
        return SacSmaState(
            uztwc=jnp.asarray(uztwc, dtype=float),
            uzfwc=jnp.asarray(uzfwc, dtype=float),
            lztwc=jnp.asarray(lztwc, dtype=float),
            lzfpc=jnp.asarray(lzfpc, dtype=float),
            lzfsc=jnp.asarray(lzfsc, dtype=float),
            adimc=jnp.asarray(adimc, dtype=float),
        )
    else:
        return SacSmaState(
            uztwc=np.float64(uztwc),
            uzfwc=np.float64(uzfwc),
            lztwc=np.float64(lztwc),
            lzfpc=np.float64(lzfpc),
            lzfsc=np.float64(lzfsc),
            adimc=np.float64(adimc),
        )


def sacsma_step(
    pxv: Any,
    pet: Any,
    dt: float,
    state: SacSmaState,
    params: SacSmaParameters,
    xp: Any = np,
) -> Tuple[SacSmaState, Any, Any, Any, Any]:
    """Execute one timestep of the SAC-SMA model (branch-free).

    All branching uses ``xp.where()`` for JAX differentiability.

    Args:
        pxv: Effective precipitation (rain + snowmelt, mm/dt)
        pet: Potential evapotranspiration (mm/dt)
        dt: Timestep in days (1.0 for daily)
        state: Current model state
        params: SAC-SMA parameters
        xp: Array backend (jnp or np)

    Returns:
        Tuple of (new_state, surface_runoff, interflow, baseflow, actual_et)
        All fluxes in mm/dt.
    """
    from .nws import step_with_fluxes
    values = step_with_fluxes(pxv, pet, dt, state, params, xp)
    return SacSmaState(*values[:6]), *values[6:10]


def sacsma_step_with_fluxes(pxv, pet, dt, state, params, xp=np):
    """SAC1 step including non-channel baseflow (deep recharge).

    SIDE is the ratio of non-channel to channel baseflow. Returned stores
    use pervious/ADIMP area depths; all five fluxes use basin depths.
    """
    from .nws import step_with_fluxes
    values = step_with_fluxes(pxv, pet, dt, state, params, xp)
    return SacSmaState(*values[:6]), tuple(values[6:])


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def sacsma_simulate_jax(
    pxv: Any,
    pet: Any,
    params: SacSmaParameters,
    initial_state: Optional[SacSmaState] = None,
    dt: float = 1.0,
) -> Tuple[Any, SacSmaState]:
    """Run SAC-SMA simulation using JAX lax.scan.

    Args:
        pxv: Effective precipitation array (rain + melt, mm/dt)
        pet: PET array (mm/dt)
        params: SAC-SMA parameters
        initial_state: Initial state (default: 50% capacity)
        dt: Timestep in days

    Returns:
        Tuple of (total_channel_inflow mm/dt, final state)
    """
    if not HAS_JAX:
        return sacsma_simulate_numpy(
            np.asarray(pxv), np.asarray(pet), params, initial_state, dt,
        )

    if initial_state is None:
        initial_state = _create_default_state(params, use_jax=True)

    forcing = jnp.stack([pxv, pet], axis=1)

    def scan_fn(carry, forcing_step):
        p, e = forcing_step
        new_state, surface, interflow, baseflow, _ = sacsma_step(
            p, e, dt, carry, params, xp=jnp,
        )
        return new_state, surface + interflow + baseflow

    final_state, total_flow = lax.scan(scan_fn, initial_state, forcing)
    return total_flow, final_state


def sacsma_simulate_numpy(
    pxv: np.ndarray,
    pet: np.ndarray,
    params: SacSmaParameters,
    initial_state: Optional[SacSmaState] = None,
    dt: float = 1.0,
) -> Tuple[np.ndarray, SacSmaState]:
    """Run SAC-SMA simulation using NumPy (Python loop fallback).

    Args:
        pxv: Effective precipitation array (rain + melt, mm/dt)
        pet: PET array (mm/dt)
        params: SAC-SMA parameters
        initial_state: Initial state (default: 50% capacity)
        dt: Timestep in days

    Returns:
        Tuple of (total_channel_inflow mm/dt, final state)
    """
    n = len(pxv)
    total_flow = np.zeros(n)

    if initial_state is None:
        initial_state = _create_default_state(params, use_jax=False)

    state = initial_state

    for i in range(n):
        state, surface, interflow, baseflow, _ = sacsma_step(
            np.float64(pxv[i]), np.float64(pet[i]), dt, state, params, xp=np,
        )
        total_flow[i] = float(surface + interflow + baseflow)

    return total_flow, state


def sacsma_simulate(
    pxv: Any,
    pet: Any,
    params: SacSmaParameters,
    initial_state: Optional[SacSmaState] = None,
    dt: float = 1.0,
    use_jax: bool = False,
) -> Tuple[Any, SacSmaState]:
    """Run SAC-SMA simulation with automatic backend selection.

    Args:
        pxv: Effective precipitation array (rain + melt, mm/dt)
        pet: PET array (mm/dt)
        params: SAC-SMA parameters
        initial_state: Initial state (default: 50% capacity)
        dt: Timestep in days
        use_jax: Whether to prefer JAX backend

    Returns:
        Tuple of (total_channel_inflow mm/dt, final state)
    """
    if use_jax and HAS_JAX:
        return sacsma_simulate_jax(pxv, pet, params, initial_state, dt)
    else:
        return sacsma_simulate_numpy(
            np.asarray(pxv), np.asarray(pet), params, initial_state, dt,
        )
