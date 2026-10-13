# Corrected accounting and explicit cryosphere research variant

Version 0.3.0 replaces the earlier approximate soil step with the unfrozen
NOAA SAC1 accounting translation. All area branches remain active. Store
capacities constrain percolation before allocation; excess is not invented as
extra channel runoff. SIDE is the non-channel/channel baseflow ratio; deep
recharge is an explicit sink. Riparian ET removes available channel inflow.
`tests/reference/README.md` identifies the independent compiled reference.
These numerical changes require recalibration of existing parameter sets.

`jsacsma.cryosphere` identifies its extension as
`SACSMA_CRYOSPHERE_UNIFORM_HRU_V1`, not standard lumped SAC-SMA.
It uses elevation/land/glacier HRUs, native soil accounting on land, glacier
soil bypass, finite initial ice in mm water equivalent and an exact linear
routing reservoir. P and PET are mm per input interval; T is degrees C.
Weights sum to one; geometry, initial ice, interval and corrections are explicit
in `CryosphereConfig`. `simulate_cryosphere` validates host inputs; the cached
`compiled_simulator` accepts dynamic parameters in `PARAMETER_NAMES` order.
JAX requires 64-bit precision. Never trace unvalidated configurations.

Snow17 effective melt area is kept distinct from geometric snow presence.
Each HRU is spatially uniform: glacier ice exposure requires a bare HRU at the
start of a substep with no incoming snow. This binary geometric approximation
is not a measured fractional glacier-cover model. Ice only melts up to its
remaining inventory. Meltout substeps conservatively omit the remaining ice
melt opportunity, bounded by `max_substep_days` (default 3 hours, maximum 6).
Test convergence and use native subdaily forcing for diurnal effects; splitting
daily mean temperature cannot reconstruct those effects.

No glacier retreat, firn evolution, wind redistribution, frozen ground or full
Snow17 snow-liquid transmission lag is represented. Geometric land cover is
also binary per HRU and should not be equated to optical fractional snow cover
without an observation operator. The effective-area diagnostic is a separate
proxy. Glacier storage change is fixed-model-area specific, not dynamically
observed-area specific. Include snow and ice in glacier storage changes.

The ledger reports effective corrected precipitation (including PXADJ/SCF),
ET, channel runoff, deep recharge, all storage, and balance residual. Snow17
state checkpoints now include depletion-history fields. Existing six-field
constructors default these fields, but cannot reconstruct a partially depleted
pack's missing history; replay from a known snow-free state instead.

Maríufossar prototype parameters and reported scores predate these corrections.
Replay them as uncalibrated starting points, then recalibrate and assess
sensitivity to geometry, initial ice, substep, precipitation and temperature.
Inspected later periods are development-validation/transfer results, not
untouched holdouts. Alternative configurations share one SACSMA family vote.
