# SPDX-License-Identifier: Apache-2.0
"""Explicit SAC1/Snow17/finite-ice uniform-HRU research variant.

Geometric glacier cover is a binary snow-presence mask, not Snow17 effective
melt area. Ice melt occurs only on substeps bare at the start, with no new
snowfall: snow-to-ice meltout cannot receive a full-step ice degree-day budget.
The unallocated remainder of a meltout substep is deliberately conservative;
its maximum timing loss is controlled by max_substep_days. No retreat/firn,
wind redistribution, or snow-liquid transmission lag is claimed.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from jsnow17.depletion import effective_snow_area
from jsnow17.model import snow17_step
from jsnow17.parameters import (
    create_initial_state,
)
from jsnow17.parameters import (
    params_dict_to_namedtuple as snow_params,
)

from .parameters import DEFAULT_PARAMS
from .parameters import params_dict_to_namedtuple as soil_params
from .sacsma import SacSmaState, _create_default_state, sacsma_step_with_fluxes

MODEL_ID = "SACSMA_CRYOSPHERE_UNIFORM_HRU_V1"
CONTROL_NAMES = (
    "CRYO_SI",
    "ICE_DDF",
    "ROUTING_TAU_DAYS",
    "PRECIP_SCALE",
    "TEMP_OFFSET_C",
)
PARAMETER_NAMES = tuple(DEFAULT_PARAMS) + CONTROL_NAMES
RECORD_NAMES = (
    "runoff_mm",
    "effective_precip_mm",
    "ET_mm",
    "deep_recharge_mm",
    "balance_residual_mm",
    "snow_storage_mm",
    "ice_storage_mm",
    "glacier_cryo_storage_mm",
    "effective_land_snow_area",
    "geometric_glacier_snow_fraction",
    "ice_melt_mm",
    "total_storage_mm",
    "routing_storage_mm",
    "geometric_land_snow_fraction",
)


@dataclass(frozen=True)
class CryosphereConfig:
    area_km2: float
    weights: tuple[float, ...]
    glacier: tuple[bool, ...]
    latitudes: tuple[float, ...]
    elevations_m: tuple[float, ...]
    initial_ice_mm: float | tuple[float, ...] = 100000.0
    si: float = 600.0
    ice_degree_day_factor: float = 5.0
    routing_tau_days: float = 1.0
    timestep_days: float = 1.0
    max_substep_days: float = 0.125
    precipitation_scale: float = 1.0
    temperature_offset_c: float = 0.0

    def __post_init__(self):
        for name in ["weights", "glacier", "latitudes", "elevations_m"]:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.weights)
        if n == 0 or any(
            len(v) != n for v in [self.glacier, self.latitudes, self.elevations_m]
        ):
            raise ValueError("HRU geometry vectors must have equal nonzero length")
        numbers = [
            self.area_km2,
            self.si,
            self.ice_degree_day_factor,
            self.routing_tau_days,
            self.timestep_days,
            self.max_substep_days,
            self.precipitation_scale,
        ]
        if not np.isfinite(numbers).all() or min(numbers) <= 0:
            raise ValueError(
                "Area, SI, melt/routing/time parameters and scale must be positive and finite"
            )
        if self.max_substep_days > 0.25:
            raise ValueError("Cryosphere substeps must be at most six hours")
        if not np.isfinite(self.temperature_offset_c):
            raise ValueError("Temperature offset must be finite")
        if (
            not np.isfinite(self.weights).all()
            or min(self.weights) <= 0
            or not np.isclose(sum(self.weights), 1.0, atol=1e-10, rtol=0)
        ):
            raise ValueError("Positive HRU weights must sum to one")
        if (
            not np.isfinite(self.latitudes).all()
            or max(abs(v) for v in self.latitudes) > 90
        ):
            raise ValueError("Latitudes must be finite and within +/-90")
        if (
            not np.isfinite(self.elevations_m).all()
            or min(self.elevations_m) < -500
            or max(self.elevations_m) > 9000
        ):
            raise ValueError("Elevation outside supported standard-atmosphere range")
        if any(type(v) not in [bool, np.bool_] for v in self.glacier):
            raise ValueError("Glacier flags must be booleans")
        ice = np.asarray(self.initial_ice_mm, float)
        if ice.ndim == 0:
            ice = np.full(n, float(ice))
        if ice.shape != (n,) or not np.isfinite(ice).all() or (ice < 0).any():
            raise ValueError(
                "Initial ice must be finite, nonnegative scalar or per-HRU vector"
            )
        object.__setattr__(self, "initial_ice_mm", tuple(ice))

    @property
    def substeps(self):
        return math.ceil(self.timestep_days / self.max_substep_days)


def simulate_cryosphere(precip, temp, pet, day_of_year, params, config, use_jax=False):
    """Replay depths in mm/input-step on (time, HRU) arrays.

    Returns records and final (Snow17State, SacSmaState, ice_mm, routing_mm).
    Initial ice and all geometry/time/correction values are honored explicitly.
    Every output depth is per input interval; discharge conversion uses that dt.
    Parameters must be finite and use known package names. Recalibration is
    required after replacing the previous experimental accounting kernels.
    """
    arrays = [np.asarray(v, float) for v in [precip, temp, pet]]
    shape = arrays[0].shape
    if (
        len(shape) != 2
        or shape[1] != len(config.weights)
        or shape[0] == 0
        or any(a.shape != shape for a in arrays)
    ):
        raise ValueError(
            "P/T/PET must share nonempty (time, HRU) shape matching geometry"
        )
    doy = np.asarray(day_of_year, float)
    if (
        doy.shape != (shape[0],)
        or not np.isfinite(doy).all()
        or (doy < 1).any()
        or (doy >= 367).any()
    ):
        raise ValueError("Invalid day-of-year axis")
    if (
        any(not np.isfinite(a).all() for a in arrays)
        or (arrays[0] < 0).any()
        or (arrays[2] < 0).any()
    ):
        raise ValueError("Forcing must be finite; P/PET nonnegative")
    if set(params) - set(PARAMETER_NAMES):
        raise ValueError(
            f"Unknown model parameters: {sorted(set(params) - set(PARAMETER_NAMES))}"
        )
    full = dict(
        DEFAULT_PARAMS,
        **dict(
            zip(
                CONTROL_NAMES,
                [
                    config.si,
                    config.ice_degree_day_factor,
                    config.routing_tau_days,
                    config.precipitation_scale,
                    config.temperature_offset_c,
                ],
            )
        ),
        **params,
    )
    if not np.isfinite(list(full.values())).all():
        raise ValueError("Model parameters must be finite")
    if (
        min(full[k] for k in ["UZTWM", "UZFWM", "LZTWM", "LZFPM", "LZFSM", "MFMAX"])
        <= 0
    ):
        raise ValueError("Store capacities and MFMAX must be positive")
    if (
        any(
            not 0 <= full[k] <= 1
            for k in [
                "UZK",
                "LZPK",
                "LZSK",
                "PFREE",
                "PCTIM",
                "ADIMP",
                "RSERV",
                "TIPM",
                "PLWHC",
                "RIVA",
            ]
        )
        or full["PCTIM"] + full["ADIMP"] > 1
        or full["SIDE"] < 0
        or full["SCF"] <= 0
        or full.get("PXADJ", 1) <= 0
    ):
        raise ValueError("Invalid fraction/correction/deep-recharge parameters")
    if min(full[k] for k in ["MFMIN", "NMF", "UADJ", "DAYGM", "ZPERC", "REXP"]) < 0:
        raise ValueError("Melt/heat/percolation coefficients must be nonnegative")
    if any(full[k] <= 0 for k in CONTROL_NAMES[:4]):
        raise ValueError("Cryosphere controls must be positive")
    if use_jax:
        import jax
        import jax.numpy as jnp

        if not jax.config.x64_enabled:
            raise ValueError(
                "Cryosphere replay requires JAX_ENABLE_X64=true for ice-storage balance accuracy"
            )
        records, state = compiled_simulator(config)(
            jnp.asarray([full[k] for k in PARAMETER_NAMES]),
            *[jnp.asarray(a) for a in [*arrays, doy]],
        )
    else:
        records, state = _run_core(arrays, doy, full, config, False)
    return dict(zip(RECORD_NAMES, np.asarray(records).T)), state


@lru_cache(maxsize=16)
def compiled_simulator(config):
    """Cached differentiable simulator: (parameter vector, P, T, PET, DOY).

    Parameter ordering is PARAMETER_NAMES. Geometry/time discretization is
    static; all soil/snow and five coupling controls remain dynamic.
    """
    import jax

    return jax.jit(
        lambda vector, p, t, e, d: _run_core(
            (p, t, e), d, dict(zip(PARAMETER_NAMES, vector)), config, True
        )
    )


def _run_core(arrays, doy, full, config, use_jax):
    shape = arrays[0].shape
    if use_jax:
        import jax
        import jax.numpy as xp
    else:
        xp = np
    weights = xp.asarray(config.weights)
    glacier = xp.asarray(config.glacier)
    land = ~glacier
    lat = xp.asarray(config.latitudes)
    elevation = xp.asarray(config.elevations_m)
    n = shape[1]
    dt = config.timestep_days / config.substeps
    sp = snow_params(full, use_jax=use_jax)
    ap = soil_params(full, use_jax=use_jax)
    si, ice_ddf, tau, pcorr, tcorr = [full[k] for k in CONTROL_NAMES]

    def broadcast(state):
        return type(state)(*[xp.full(n, v) for v in state])

    snow = broadcast(create_initial_state(use_jax=use_jax))
    soil = broadcast(_create_default_state(ap, use_jax=use_jax))
    soil = SacSmaState(*[v * land for v in soil])
    state = (snow, soil, xp.asarray(config.initial_ice_mm) * glacier, xp.asarray(0.0))
    parea = 1 - ap.PCTIM - ap.ADIMP
    gw = weights * glacier
    lw = weights * land

    def storage(state):
        sn, sa, ic, r = state
        return (
            xp.sum(
                weights
                * (
                    sn.w_i
                    + sn.w_q
                    + ic
                    + land * (parea * sum(sa[:5]) + ap.ADIMP * sa.adimc)
                )
            )
            + r
        )

    def soil_step(rpm, et, sa):
        if use_jax:
            return jax.vmap(
                lambda pr, e, s: sacsma_step_with_fluxes(pr, e, dt, s, ap, xp)
            )(rpm, et, sa)
        results = [
            sacsma_step_with_fluxes(
                rpm[i], et[i], dt, SacSmaState(*[v[i] for v in sa]), ap
            )
            for i in range(n)
        ]
        return SacSmaState(
            *[np.array([r[0][k] for r in results]) for k in range(6)]
        ), tuple(np.array([r[1][k] for r in results]) for k in range(5))

    def substep(state, forcing):
        sn, sa, ic, r = state
        pr, t, et, d = forcing
        corrected_t = t + tcorr
        pr = pr * pcorr
        frac = xp.clip((corrected_t - (sp.PXTEMP - 1)) / 2, 0, 1)
        snowfall = pr * sp.PXADJ * (1 - frac) * sp.SCF
        bare_at_start = (sn.w_i + sn.w_q <= 0) & (snowfall <= 0)
        new_sn, rpm = snow17_step(
            pr, corrected_t, dt, sn, sp, d, lat, si, xp=xp, elevation_m=elevation
        )
        new_sa, flux = soil_step(rpm * land, et * land, sa)
        soil_q = flux[0] + flux[1] + flux[2]
        ice_melt = (
            xp.minimum(ic, ice_ddf * xp.maximum(corrected_t, 0) * dt * bare_at_start)
            * glacier
        )
        new_ic = ic - ice_melt
        raw = xp.sum(weights * (land * soil_q + glacier * (rpm + ice_melt)))
        decay = xp.exp(-dt / tau)
        new_r = r * decay + raw / dt * tau * (-xp.expm1(-dt / tau))
        q = r + raw - new_r
        new_state = (new_sn, new_sa, new_ic, new_r)
        effective_p = xp.sum(weights * pr * sp.PXADJ * (frac + (1 - frac) * sp.SCF))
        aet = xp.sum(lw * flux[3])
        deep = xp.sum(lw * flux[4])
        total = storage(new_state)
        residual = effective_p - q - aet - deep - (total - storage(state))
        pack = new_sn.w_i + new_sn.w_q
        area = effective_snow_area(new_sn, si, xp=xp)
        record = xp.stack(
            [
                q,
                effective_p,
                aet,
                deep,
                residual,
                xp.sum(weights * pack),
                xp.sum(weights * new_ic),
                xp.sum(gw * (pack + new_ic)) / xp.maximum(gw.sum(), 1e-30),
                xp.sum(lw * area) / xp.maximum(lw.sum(), 1e-30),
                xp.sum(gw * (pack > 0)) / xp.maximum(gw.sum(), 1e-30),
                xp.sum(weights * ice_melt),
                total,
                new_r,
                xp.sum(lw * (pack > 0)) / xp.maximum(lw.sum(), 1e-30),
            ]
        )
        return new_state, record

    def outer(state, forcing):
        pr, t, et, d = forcing
        subforcing = (pr / config.substeps, t, et / config.substeps, d)
        if use_jax:
            state, records = jax.lax.scan(
                lambda state, _: substep(state, subforcing),
                state,
                None,
                length=config.substeps,
            )
        else:
            entries = []
            for _ in range(config.substeps):
                state, record = substep(state, subforcing)
                entries.append(record)
            records = np.array(entries)
        # Depths/residuals sum; states and area diagnostics use the interval end.
        record = records[-1]
        sums = records.sum(axis=0)
        if use_jax:
            record = record.at[:5].set(sums[:5]).at[10].set(sums[10])
        else:
            record = record.copy()
            record[:5] = sums[:5]
            record[10] = sums[10]
        return state, record

    forcings = tuple(xp.asarray(a) for a in [*arrays, doy])
    if use_jax:
        state, records = jax.lax.scan(outer, state, forcings)
    else:
        output = []
        for i in range(shape[0]):
            state, record = outer(state, tuple(a[i] for a in forcings))
            output.append(record)
        records = np.array(output)
    return records, state
