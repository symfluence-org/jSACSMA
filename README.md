# jSACSMA

SAC-SMA (Sacramento Soil Moisture Accounting) + Snow-17 coupled hydrological model extracted from [SYMFLUENCE](https://github.com/DarriEy/SYMFLUENCE).

Burnash (1995) dual-backend (JAX + NumPy) implementation with Anderson (2006) Snow-17 coupling.

## Installation

```bash
pip install jsacsma
```

## Usage

```python
from jsacsma.model import simulate

# Coupled Snow-17 + SAC-SMA simulation
flow, state = simulate(precip, temp, pet, start_date='2004-01-01')

# Standalone SAC-SMA (no snow)
flow, state = simulate(precip, temp, pet, snow_module='none')
```

## License

Apache-2.0
