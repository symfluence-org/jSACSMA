"""Explicit geometric snow-area observation operator.

Area is the weighted share of modeled surface units above a specified snow
water threshold. This quantity is distinct from Snow17's effective melt area.
The user must match surface masks and sampling times to the observational
product. No spectral inversion, cloud correction, canopy correction or pixel
scale heterogeneity is inferred here.
"""

from __future__ import annotations

import numpy as np


def snow_covered_area(snow_water_mm, weights, include=None, threshold_mm=1.0):
    """Aggregate the final axis of nonnegative per-unit snow water storage.

    ``include`` selects the observed domain, e.g. non-glacier surface units.
    Weights are renormalized within that domain. Threshold uncertainty should
    be profiled instead of fitted on a validation period.
    """
    snow = np.asarray(snow_water_mm, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.ones(w.shape, dtype=bool) if include is None else np.asarray(include)
    if (
        snow.ndim < 1
        or w.ndim != 1
        or snow.shape[-1] != len(w)
        or mask.shape != w.shape
        or mask.dtype != bool
        or not np.isfinite(snow).all()
        or (snow < 0).any()
        or not np.isfinite(w).all()
        or (w <= 0).any()
        or not np.isfinite(threshold_mm)
        or threshold_mm < 0
        or not mask.any()
    ):
        raise ValueError("Invalid snow storage, area weights, domain mask or threshold")
    selected = w * mask
    return (snow > threshold_mm) @ selected / selected.sum()
