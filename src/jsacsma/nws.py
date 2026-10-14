# SPDX-License-Identifier: Apache-2.0
"""Unfrozen SAC1 accounting, translated from NOAA-OWP/sac-sma.

Reference revision: 975902e3d44785f3b3503f29adfb5755120f5bf5,
src/sac/sac1.f90 (Apache-2.0). Frozen-ground operations are not implemented.
Stores are depths over their respective areas; output fluxes are basin depths.
"""

import numpy as np

from .parameters import SACSMA_PARAM_NAMES

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError:
    jax = jnp = lax = None


def _kernel(x, xp):
    rain, pet, dt = x[:3]
    ut, uf, lt, lp, ls, ad = x[3:9]
    p = dict(zip(SACSMA_PARAM_NAMES, x[9:]))
    utm, ufm, ltm, lpm, lsm = [
        p[k] for k in ["UZTWM", "UZFWM", "LZTWM", "LZFPM", "LZFSM"]
    ]
    area = 1 - p["PCTIM"] - p["ADIMP"]
    e1 = xp.minimum(pet * ut / utm, ut)
    exhausted = pet * ut / utm > ut
    ut = ut - e1
    red = pet - e1
    e2 = xp.where(exhausted, xp.minimum(red, uf), 0.0)
    bypass = exhausted & (uf < red)
    uf = uf - e2
    red = red - e2
    ratio = (ut + uf) / (utm + ufm)
    equalize = (~bypass) & (ut / utm < uf / ufm)
    ut, uf = xp.where(equalize, utm * ratio, ut), xp.where(equalize, ufm * ratio, uf)
    e3 = xp.minimum(red * lt / (utm + ltm), lt)
    lt = lt - e3
    saved = p["RSERV"] * (lpm + lsm)
    lower_ratio = (lt + lp + ls - saved) / (ltm + lpm + lsm - saved)
    transfer = xp.maximum((lower_ratio - lt / ltm) * ltm, 0.0)
    lt = lt + transfer
    ls = ls - transfer
    lp = lp + xp.minimum(ls, 0.0)
    ls = xp.maximum(ls, 0.0)
    e5 = xp.minimum(e1 + (red + e2) * (ad - e1 - ut) / (utm + ltm), ad)
    ad = ad - e5
    excess = xp.maximum(rain + ut - utm, 0.0)
    ut = xp.minimum(ut + rain, utm)
    ad = ad + rain - excess
    ninc = xp.asarray(1 + 0.2 * (uf + excess), dtype=int)
    dinc, pinc = dt / ninc, excess / ninc
    duz = 1 - (1 - p["UZK"]) ** dinc
    dlzp, dlzs = 1 - (1 - p["LZPK"]) ** dinc, 1 - (1 - p["LZSK"]) ** dinc

    def increment(i, carry):
        uf, lt, lp, ls, ad, ssur, sif, sbf, sdro = carry
        addro = pinc * xp.maximum((ad - ut) / ltm, 0.0) ** 2
        bf = lp * dlzp
        lp = lp - bf
        bf = bf + xp.where(lp <= 0.0001, lp, 0.0)
        lp = xp.where(lp <= 0.0001, 0.0, lp)
        sbf = sbf + bf
        bf = ls * dlzs
        ls = ls - bf
        bf = bf + xp.where(ls <= 0.0001, ls, 0.0)
        ls = xp.where(ls <= 0.0001, 0.0, ls)
        sbf = sbf + bf
        active = pinc + uf > 0.01
        percm = lpm * dlzp + lsm * dlzs
        deficiency = xp.maximum(1 - (lt + lp + ls) / (ltm + lpm + lsm), 0.0)
        perc = xp.minimum(
            percm
            * (uf / ufm)
            * (1 + p["ZPERC"] * xp.maximum(deficiency, 1e-30) ** p["REXP"]),
            uf,
        )
        perc = xp.minimum(perc, xp.maximum(ltm + lpm + lsm - lt - lp - ls, 0.0))
        perc = xp.where(active, perc, 0.0)
        uf = uf - perc
        inter = xp.where(active, uf * duz, 0.0)
        sif = sif + inter
        uf = uf - inter
        perct = perc * (1 - p["PFREE"])
        percf = xp.maximum(perct + lt - ltm, 0.0) + perc * p["PFREE"]
        lt = xp.minimum(lt + perct, ltm)
        hpl = lpm / (lpm + lsm)
        dprimary, dsupp = 1 - lp / lpm, 1 - ls / lsm
        denominator = dprimary + dsupp
        fracp = xp.minimum(hpl * 2 * dprimary / xp.maximum(denominator, 1e-30), 1.0)
        percs = xp.minimum(percf * (1 - fracp), xp.maximum(lsm - ls, 0.0))
        ls = ls + percs
        lp = lp + percf - percs
        lt = lt + xp.maximum(lp - lpm, 0.0)
        lp = xp.minimum(lp, lpm)
        sur = xp.where(active, xp.maximum(pinc + uf - ufm, 0.0), 0.0)
        adsur = sur * (1 - addro / xp.maximum(pinc, 1e-30))
        ssur = ssur + sur * area + adsur * p["ADIMP"]
        uf = uf + pinc - sur
        ad = ad + pinc - addro - adsur
        addro = addro + xp.maximum(ad - utm - ltm, 0.0)
        ad = xp.minimum(ad, utm + ltm)
        sdro = sdro + addro * p["ADIMP"]
        return (uf, lt, lp, ls, ad, ssur, sif, sbf, sdro)

    carry = (uf, lt, lp, ls, ad, rain * 0.0, rain * 0.0, rain * 0.0, rain * 0.0)
    if xp is np:
        for i in range(int(ninc)):
            carry = increment(i, carry)
    else:
        carry = lax.fori_loop(0, ninc, increment, carry)
    uf, lt, lp, ls, ad, ssur, sif, sbf, sdro = carry
    surface = rain * p["PCTIM"] + sdro + ssur
    inter = sif * area
    base = sbf * area / (1 + p["SIDE"])
    deep = sbf * area - base
    soil_et = e1 + e2 + e3
    gross = surface + inter + base
    riparian = xp.minimum(xp.maximum(pet - soil_et, 0.0) * p["RIVA"], gross)
    # SAC1 removes riparian ET from total channel inflow, not from soil.
    # Allocate that removal proportionally to returned channel components.
    factor = 1 - riparian / xp.maximum(gross, 1e-30)
    et = soil_et * area + e5 * p["ADIMP"] + riparian
    ad = xp.maximum(ad, ut)
    return xp.stack(
        [
            ut,
            uf,
            lt,
            lp,
            ls,
            ad,
            surface * factor,
            inter * factor,
            base * factor,
            et,
            deep,
        ]
    )


if jax is not None:
    # Adaptive SAC1 subincrements require a dynamic loop. Reverse-mode uses
    # its forward-mode Jacobian instead of transposing a dynamic while loop.
    @jax.custom_vjp
    def _jax_kernel(x):
        return _kernel(x, jnp)

    def _forward(x):
        return _kernel(x, jnp), x

    def _backward(x, cotangent):
        return (jax.jacfwd(lambda value: _kernel(value, jnp))(x).T @ cotangent,)

    _jax_kernel.defvjp(_forward, _backward)


def step_with_fluxes(rain, pet, dt, state, params, xp=np):
    """Return six stores and five basin fluxes, including deep recharge.

    NumPy validates scalar inputs. JAX callers should validate configurations
    before tracing; supports scalar steps and vmap over HRUs. Higher-order
    derivatives at adaptive-subincrement thresholds are not guaranteed.
    """
    x = xp.stack([rain, pet, dt, *state, *params])
    if xp is np:
        if not np.isfinite(x).all() or rain < 0 or pet < 0 or dt <= 0:
            raise ValueError("SAC-SMA inputs must be finite, nonnegative; dt positive")
        if (
            min(params.UZTWM, params.UZFWM, params.LZTWM, params.LZFPM, params.LZFSM)
            <= 0
        ):
            raise ValueError("Store capacities must be positive")
        if (
            any(
                not 0 <= getattr(params, k) <= 1
                for k in [
                    "UZK",
                    "LZPK",
                    "LZSK",
                    "PFREE",
                    "PCTIM",
                    "ADIMP",
                    "RSERV",
                    "RIVA",
                ]
            )
            or min(params.SIDE, params.ZPERC, params.REXP) < 0
        ):
            raise ValueError("Invalid SAC-SMA fractions or percolation parameters")
        if params.PCTIM + params.ADIMP > 1:
            raise ValueError("PCTIM + ADIMP must not exceed one")
        caps = np.array(
            [
                params.UZTWM,
                params.UZFWM,
                params.LZTWM,
                params.LZFPM,
                params.LZFSM,
                params.UZTWM + params.LZTWM,
            ]
        )
        if (
            (np.asarray(state) < 0).any()
            or (np.asarray(state) > caps + 1e-8).any()
            or state.adimc + 1e-8 < state.uztwc
        ):
            raise ValueError(
                "SAC-SMA carryover must satisfy store capacities and ADIMC >= UZTWC"
            )
        return _kernel(x, np)
    return _jax_kernel(x)
