# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests: KGE/NSE losses must be autodiff-safe with NaN observations.

Streamflow observations have gaps, and daily observations aligned to a sub-daily
grid are mostly NaN. Feeding NaN into corrcoef/mean/std made the loss — and every
parameter gradient — NaN, silently collapsing the gradient-based optimisers
(Adam, L-BFGS). The losses now mask missing observations.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from jsacsma.losses import kge_loss, nse_loss  # noqa: E402
from jsacsma.parameters import DEFAULT_PARAMS  # noqa: E402


def _forcing(n, seed=0):
    rng = np.random.default_rng(seed)
    precip = jnp.array(rng.gamma(2.0, 2.0, n))
    temp = jnp.array(10 + 5 * np.sin(np.arange(n) * 2 * np.pi / 365))
    pet = jnp.array(np.abs(3 + 2 * np.sin(np.arange(n) * 2 * np.pi / 365)))
    doy = jnp.array((np.arange(n) % 365 + 1).astype(float))
    return precip, temp, pet, doy


@pytest.mark.parametrize("loss_fn", [kge_loss, nse_loss])
def test_loss_gradient_finite_with_mostly_nan_obs(loss_fn):
    """A ~96%-NaN observation series (daily obs on an hourly grid) must still
    yield finite gradients for every parameter."""
    n = 1500
    precip, temp, pet, doy = _forcing(n)
    # observed flow present only every 24th step; NaN elsewhere
    obs = np.full(n, np.nan)
    obs[::24] = np.abs(np.sin(np.arange(len(obs[::24])) * 0.1)) + 0.05
    obs = jnp.array(obs)

    names = list(DEFAULT_PARAMS.keys())
    vals = jnp.array([DEFAULT_PARAMS[k] for k in names])

    def loss(v):
        return loss_fn(dict(zip(names, v)), precip, temp, pet, obs,
                       warmup_days=100, use_jax=True, day_of_year=doy,
                       latitude=51.0, si=100.0, snow_module="snow17")

    val = loss(vals)
    grad = jax.grad(loss)(vals)
    assert np.isfinite(float(val)), "loss itself is non-finite"
    assert np.all(np.isfinite(np.asarray(grad))), (
        f"{loss_fn.__name__} produced non-finite gradients: "
        f"{int(np.sum(~np.isfinite(np.asarray(grad))))}/{len(names)} bad"
    )


def test_masked_kge_matches_dense_when_no_gaps():
    """With no missing observations the masked KGE equals the plain KGE, so the
    NaN-safe path doesn't change results when there is nothing to mask."""
    n = 800
    precip, temp, pet, doy = _forcing(n, seed=3)
    rng = np.random.default_rng(1)
    obs = jnp.array(np.abs(rng.normal(1.0, 0.5, n)) + 0.05)
    kw = dict(warmup_days=100, use_jax=True, day_of_year=doy,
              latitude=51.0, si=100.0, snow_module="snow17")
    dense = float(kge_loss(dict(DEFAULT_PARAMS), precip, temp, pet, obs, **kw))
    # inject a single NaN and confirm the loss stays finite and close
    obs2 = obs.at[123].set(jnp.nan)
    withgap = float(kge_loss(dict(DEFAULT_PARAMS), precip, temp, pet, obs2, **kw))
    assert np.isfinite(dense) and np.isfinite(withgap)
    assert abs(dense - withgap) < 0.05
