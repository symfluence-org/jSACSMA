# Independent unfrozen SAC1 reference

`sac1.f90` and `sac_data_mod.f90` are unmodified copies from
[NOAA-OWP/sac-sma](https://github.com/NOAA-OWP/sac-sma/tree/975902e3d44785f3b3503f29adfb5755120f5bf5/src/sac),
revision `975902e3d44785f3b3503f29adfb5755120f5bf5`, distributed under the
included Apache-2.0 `NOAA-LICENSE`. Original source attribution is preserved.
`wrapper.f90` only sets the reference's state/parameters and exposes output;
`generate.py` compiles those sources with gfortran and saves `fixtures.json`.

Regenerate from the repository root: `python tests/reference/generate.py`.
The ordinary tests read frozen fixtures and do not require a Fortran compiler.
The fixtures cover unsaturated/saturated stores, permanent/additional
impervious area, riparian ET, SIDE deep recharge and daily/subdaily timesteps.
They verify total net channel flow, ET, deep recharge and all six stores.
Returned channel components allocate riparian removal proportionally; NOAA's
internal pre-riparian component diagnostics are not identical.

This establishes equivalence for these unfrozen reference cases, not complete
operational validation or frozen-ground equivalence. NumPy/JAX comparison and
area-weighted storage/flux closure are checked separately.
