"""Regenerate frozen fixtures using the vendored NOAA Fortran, not Python code."""

import ctypes
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# SAC parameter order is explicitly fixed here, independent of the implementation.
parameters = [
    50.0,
    40.0,
    0.3,
    130.0,
    60.0,
    25.0,
    0.01,
    0.05,
    40.0,
    2.0,
    0.3,
    0.01,
    0.05,
    0.0,
    0.0,
    0.3,
]
with tempfile.TemporaryDirectory() as temp:
    lib = Path(temp) / "reference.so"
    subprocess.run(
        [
            "gfortran",
            "-shared",
            "-fPIC",
            "-ffree-line-length-none",
            str(ROOT / "sac_data_mod.f90"),
            str(ROOT / "sac1.f90"),
            str(ROOT / "wrapper.f90"),
            "-o",
            str(lib),
        ],
        cwd=temp,
        check=True,
    )
    dll = ctypes.CDLL(str(lib))
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    dll.reference_step.argtypes = [ptr, ptr]
    records = []
    for name, initial, p, forcing in [
        (
            "dry_storm",
            [0.0] * 6,
            parameters,
            [(100.0, 0.0, 1.0), (0.0, 2.0, 1.0), (10.0, 0.5, 0.25)],
        ),
        (
            "saturated",
            [50.0, 40.0, 130.0, 60.0, 25.0, 180.0],
            parameters,
            [(100.0, 0.0, 1.0), (0.0, 3.0, 1.0)],
        ),
        (
            "riparian_side",
            [25.0, 20.0, 65.0, 30.0, 12.5, 90.0],
            parameters[:13] + [0.2, 0.5, 0.3],
            [(0.0, 15.0, 1.0), (3.0, 2.0, 0.125), (80.0, 1.0, 1.0)],
        ),
        (
            "impervious_adimp",
            [25.0, 20.0, 65.0, 30.0, 12.5, 90.0],
            parameters[:11] + [0.1, 0.4, 0.1, 0.2, 0.3],
            [(180.0, 5.0, 1.0), (0.0, 20.0, 1.0)],
        ),
    ]:
        state = initial
        for k, (rain, pet, dt) in enumerate(forcing):
            x = np.array([rain, pet, dt, *state, *p], dtype=np.float64)
            y = np.empty(11)
            dll.reference_step(x, y)
            records.append(
                {"name": f"{name}_{k}", "input": x.tolist(), "expected": y.tolist()}
            )
            state = y[:6].tolist()
    (ROOT / "fixtures.json").write_text(
        json.dumps(
            {
                "source": "https://github.com/NOAA-OWP/sac-sma",
                "revision": "975902e3d44785f3b3503f29adfb5755120f5bf5",
                "frozen_ground": False,
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )
