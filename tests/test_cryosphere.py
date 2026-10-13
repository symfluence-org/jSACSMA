from dataclasses import replace

import numpy as np
import pytest

from jsacsma.cryosphere import (
    CryosphereConfig,
    compiled_simulator,
    simulate_cryosphere,
)


def config(**kw):
    return CryosphereConfig(
        area_km2=1.0,
        weights=(0.8, 0.2),
        glacier=(False, True),
        latitudes=(64.0, 64.0),
        elevations_m=(500.0, 1000.0),
        **kw,
    )


def run(c, steps=80, use_jax=False):
    p = np.zeros((steps, 2))
    t = np.full((steps, 2), 10.0)
    e = np.zeros_like(p)
    d = np.full(steps, 172.0)
    return simulate_cryosphere(
        p, t, e, d, {"PCTIM": 0.1, "ADIMP": 0.2, "RIVA": 0.1, "SIDE": 0.3}, c, use_jax
    )


def test_finite_ice_and_all_area_branches_close():
    r, state = run(config(initial_ice_mm=10.0))
    assert np.max(np.abs(r["balance_residual_mm"])) < 1e-8
    assert (r["runoff_mm"] >= 0).all() and float(state[2][1]) == 0
    assert r["ice_melt_mm"].sum() == pytest.approx(2.0)


def test_config_honors_per_hru_ice_and_rejects_invalid_geometry():
    _r, state = run(config(initial_ice_mm=(999.0, 1000.0)), steps=1)
    assert state[2][0] == 0 and state[2][1] > 900
    with pytest.raises(ValueError):
        replace(config(), weights=(0.5, 0.6))
    with pytest.raises(ValueError):
        replace(config(), initial_ice_mm=(-1.0, 2.0))
    with pytest.raises(ValueError):
        replace(config(), max_substep_days=1.0)


def test_meltout_does_not_receive_full_day_ice_potential():
    c = CryosphereConfig(
        1.0,
        (1.0,),
        (True,),
        (45.0,),
        (0.0,),
        max_substep_days=0.125,
        si=1000.0,
        ice_degree_day_factor=5.0,
    )
    p = np.array([[10.0], [0.0]])
    t = np.array([[-5.0], [10.0]])
    e = np.zeros_like(p)
    r, _state = simulate_cryosphere(
        p,
        t,
        e,
        [171.0, 172.0],
        {"DAYGM": 0.0, "NMF": 0.0, "PLWHC": 0.0, "MFMAX": 1.0, "MFMIN": 1.0},
        c,
    )
    assert 0 < r["ice_melt_mm"][1] < 50.0
    # Binary geometric cover does not expose bare ice on partially melted snow.
    assert np.max(np.abs(r["balance_residual_mm"])) < 1e-8


def test_substep_convergence_and_jax_parity():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    c = config(initial_ice_mm=1000.0)
    p = np.zeros((12, 2))
    p[0] = 10.0
    t = np.full((12, 2), 5.0)
    t[0] = -5.0
    e = np.zeros_like(p)
    d = np.full(12, 172.0)
    params = {"DAYGM": 0.0, "NMF": 0.0, "PLWHC": 0.0, "MFMAX": 1.0, "MFMIN": 1.0}
    a, _ = simulate_cryosphere(p, t, e, d, params, c)
    b, _ = simulate_cryosphere(p, t, e, d, params, c, True)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-8, rtol=1e-9)
    finer, _ = simulate_cryosphere(
        p, t, e, d, params, replace(c, max_substep_days=1 / 16)
    )
    # Lost meltout exposure is bounded by one coarse substep's ice potential.
    assert (
        abs(a["ice_melt_mm"].sum() - finer["ice_melt_mm"].sum())
        <= 0.2 * 5 * 5 * 0.125 + 1e-8
    )
    assert compiled_simulator(c) is compiled_simulator(c)


def test_transient_tiny_snow_does_not_shield_whole_ice_interval():
    c = CryosphereConfig(
        1.0,
        (1.0,),
        (True,),
        (45.0,),
        (0.0,),
        initial_ice_mm=1000.0,
        timestep_days=0.125,
        max_substep_days=0.125,
    )
    params = {
        "PXTEMP": 2.0,
        "MFMAX": 1.0,
        "MFMIN": 1.0,
        "PLWHC": 0.0,
        "DAYGM": 0.0,
        "NMF": 0.0,
        "MBASE": 0.0,
        "SCF": 1.0,
    }
    records, _ = simulate_cryosphere([[0.1]], [[2.9]], [[0.0]], [172.0], params, c)
    # 0.005 mm snow melts almost immediately; almost the whole 3h interval
    # remains available to melt exposed ice, rather than being fully shielded.
    potential = 5.0 * 2.9 * 0.125
    assert 0.99 * potential < records["ice_melt_mm"][0] < potential
    assert records["snow_storage_mm"][0] == 0
    assert 0.99 < records["glacier_bare_time_fraction"][0] < 1
    finer, _ = simulate_cryosphere(
        [[0.1]], [[2.9]], [[0.0]], [172.0], params, replace(c, max_substep_days=0.0625)
    )
    assert records["ice_melt_mm"][0] == pytest.approx(
        finer["ice_melt_mm"][0], abs=1e-10
    )
    assert np.max(abs(records["balance_residual_mm"])) < 1e-8


def test_snow_warming_and_melting_consume_ice_exposure_time():
    c = CryosphereConfig(1.,(1.,),(True,),(45.,),(0.,),initial_ice_mm=1000.,timestep_days=.125,max_substep_days=.125)
    params={'MFMAX':1.,'MFMIN':1.,'PLWHC':0.,'DAYGM':0.,'NMF':0.,'MBASE':0.,'SCF':1.}
    r,_=simulate_cryosphere([[10.],[0.]],[[-5.],[40.]],[[0.],[0.]],[171.,172.],params,c)
    # Cold snow heat deficit = .00625 * 10 mm * 5 C = .3125 mm.
    # Warm-step gross melt equivalent = 1 * (3h/6h) * 40 = 20 mm.
    bare_fraction=1-(10.+.3125)/20.
    assert r['glacier_bare_time_fraction'][1] == pytest.approx(bare_fraction)
    assert r['ice_melt_mm'][1] == pytest.approx(5.*40.*.125*bare_fraction)
    assert r['snow_storage_mm'][1] == 0
