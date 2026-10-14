from __future__ import annotations

import numpy as np
import pytest
from jsacsma.observations import snow_covered_area


def test_observation_excludes_glacier_and_renormalizes_area():
    snow = [[2.0, 0.0, 30.0], [0.0, 2.0, 0.0]]
    result = snow_covered_area(snow, [0.2, 0.5, 0.3], [True, True, False])
    np.testing.assert_allclose(result, [2 / 7, 5 / 7])
    np.testing.assert_allclose(
        snow_covered_area([[1.0, 1.0, 1.0]], [0.2, 0.5, 0.3]), [0.0]
    )


@pytest.mark.parametrize(
    "snow,weights,mask,threshold",
    [
        ([-1.0], [1.0], None, 1.0),
        ([np.nan], [1.0], None, 1.0),
        ([1.0], [0.0], None, 1.0),
        ([1.0], [1.0], [False], 1.0),
        ([1.0], [1.0], None, -1.0),
        ([1.0], [1.0], [1], 1.0),
    ],
)
def test_invalid_observation_inputs_fail(snow, weights, mask, threshold):
    with pytest.raises(ValueError):
        snow_covered_area(snow, weights, mask, threshold)
