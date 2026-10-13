"""Independent NOAA Fortran fixtures plus basin-area water accounting."""

import json
from pathlib import Path

import numpy as np
import pytest

from jsacsma.parameters import SacSmaParameters
from jsacsma.sacsma import SacSmaState, sacsma_step_with_fluxes

DATA = json.loads((Path(__file__).parent / "reference/fixtures.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", DATA, ids=lambda case: case["name"])
def test_noaa_fortran(case):
    x = case["input"]
    state, flux = sacsma_step_with_fluxes(
        *x[:3], SacSmaState(*x[3:9]), SacSmaParameters(*x[9:])
    )
    np.testing.assert_allclose(
        [*state, *flux], case["expected"], rtol=1e-11, atol=1e-11
    )
    p = SacSmaParameters(*x[9:])
    weight = 1 - p.PCTIM - p.ADIMP
    store = lambda s: weight * sum(s[:5]) + p.ADIMP * s[5]
    assert x[0] == pytest.approx(sum(flux) + store(state) - store(x[3:9]), abs=1e-9)


@pytest.mark.parametrize("case", DATA, ids=lambda case: case["name"])
def test_jax_matches_reference(case):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jsacsma.nws import step_with_fluxes

    x = jnp.asarray(case["input"])
    result = jax.jit(
        lambda a: step_with_fluxes(
            *a[:3], SacSmaState(*a[3:9]), SacSmaParameters(*a[9:]), jnp
        )
    )(x)
    np.testing.assert_allclose(result, case["expected"], rtol=1e-10, atol=1e-10)


def test_reverse_gradient_matches_finite_difference():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jsacsma.nws import step_with_fluxes

    x = np.array(DATA[0]["input"], float)
    x[0] = 17.3

    def flow(a):
        y = step_with_fluxes(
            *a[:3], SacSmaState(*a[3:9]), SacSmaParameters(*a[9:]), jnp
        )
        return y[6:9].sum()

    gradient = np.asarray(jax.grad(flow)(jnp.asarray(x)))
    assert np.isfinite(gradient).all()
    for i in [0, 9, 11, 15]:
        lo = x.copy()
        hi = x.copy()
        lo[i] -= 1e-5
        hi[i] += 1e-5
        finite = (float(flow(jnp.asarray(hi))) - float(flow(jnp.asarray(lo)))) / 2e-5
        assert gradient[i] == pytest.approx(finite, rel=2e-3, abs=1e-5)
