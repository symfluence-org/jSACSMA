# jSACSMA

[![PyPI version](https://img.shields.io/pypi/v/jsacsma.svg)](https://pypi.org/project/jsacsma/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Dual-backend (JAX + NumPy) implementation of the Sacramento Soil Moisture Accounting model (SAC-SMA) with optional Snow-17 coupling — usable standalone or as a SYMFLUENCE plugin.

Part of the SYMFLUENCE JAX-native model family — self-contained packages that
run standalone (NumPy fallback, no JAX required) and register automatically with
[SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE) when installed alongside it.

## Features

- **Differentiable**: automatic differentiation through the full simulation (JAX)
- **Fast**: JIT compilation via `lax.scan`; `vmap` for ensembles; GPU-capable
- **Dependency-light**: pure-NumPy fallback when JAX is not installed
- **Plugin architecture**: auto-registers with SYMFLUENCE via entry points

## Installation

```bash
pip install jsacsma          # NumPy backend
pip install 'jsacsma[jax]'    # with JAX (differentiable, JIT)
```

## Quickstart

```python
from jsacsma.model import simulate

# coupled Snow-17 + SAC-SMA (temperature drives the snow module)
flow, state = simulate(precip, temp, pet, day_of_year=doy, latitude=51.17)

# standalone SAC-SMA (no snow module)
flow, state = simulate(precip, temp, pet, snow_module="none")
```

## Gradient-based calibration

The JAX backend makes the full simulation differentiable end-to-end, so model
parameters can be calibrated with gradient descent:

```python
import jax
from jsacsma.losses import kge_loss, get_kge_gradient_fn

grad_fn = get_kge_gradient_fn(precip, temp, pet, observed)
value, grads = grad_fn(params)          # dKGE/dparam for every parameter
```

`nse_loss` / `kge_loss` and their gradient factories are JIT-compatible and work
with any `optax` optimizer. Within SYMFLUENCE the same interface powers the
ADAM and L-BFGS calibration options.

## Use with SYMFLUENCE

jsacsma registers with [SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE)
through the `symfluence.plugins` entry point — installation is the integration:

```bash
pip install symfluence jsacsma
```

```yaml
# config.yaml (excerpt)
model:
  hydrological_model: SACSMA
```

SYMFLUENCE then handles forcing preparation, calibration, evaluation, and
benchmarking for the model with no further wiring.

## Model structure

SAC-SMA (Burnash, 1995) partitions the soil into upper- and lower-zone tension
and free water storages, generating direct runoff, surface runoff, interflow,
and primary/supplemental baseflow. The optional Snow-17 coupling (Anderson, 2006)
converts precipitation to rain-plus-melt before the soil accounting.

16 calibration parameters (`jsacsma.PARAM_BOUNDS`); snow parameters are handled by
the embedded Snow-17 component.

## Testing

```bash
pip install -e '.[dev]'
pytest
```

## How to cite

If you use jSACSMA in your research, please cite the SYMFLUENCE companion papers,
which describe the design of the JAX-native model family (registry integration,
differentiability, and the calibration experiments they enable):

> Eythorsson, D., et al. (2026). The registry as social contract: Architectural patterns
> for community hydrological modeling. *Water Resources Research* (submitted).
>
> Eythorsson, D., et al. (2026). From configuration to prediction: Multi-model,
> multi-basin experiments with SYMFLUENCE. *Water Resources Research* (submitted).

Citation metadata for this package is provided in [`CITATION.cff`](CITATION.cff);
a version-specific DOI is minted via Zenodo for each GitHub release.
<!-- After the first Zenodo release, add the concept-DOI badge here. -->

## References

- Burnash, R. J. C. (1995). The NWS River Forecast System — catchment modeling. In V. P. Singh (Ed.), Computer Models of Watershed Hydrology (pp. 311–366). Water Resources Publications.
- Anderson, E. A. (2006). Snow Accumulation and Ablation Model — SNOW-17. NOAA Technical Report NWS HYDRO-17.

## License

Apache-2.0. See [LICENSE](LICENSE).
