"""Regression tests for dtype-faithful snapshots and empty-parameter handling (0.5.2)."""
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_observatory import Observatory, ObservatoryConfig
from neural_observatory.collectors.base import BaseCollector
from neural_observatory.collectors.parameter_collector import ParameterCollector


def test_safe_numpy_preserves_float64_dtype():
    t = torch.ones(3, dtype=torch.float64)
    arr = BaseCollector._safe_numpy(t)
    assert arr is not None
    assert arr.dtype == np.float64, "float64 snapshot was silently downcast."


def test_safe_numpy_preserves_float16_dtype():
    t = torch.ones(3, dtype=torch.float16)
    arr = BaseCollector._safe_numpy(t)
    assert arr is not None
    assert arr.dtype == np.float16


def test_safe_numpy_widens_bfloat16_to_float32():
    t = torch.ones(3, dtype=torch.bfloat16)
    arr = BaseCollector._safe_numpy(t)
    assert arr is not None
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr, np.ones(3, dtype=np.float32))


def test_safe_numpy_returns_none_for_complex_tensors():
    t = torch.ones(3, dtype=torch.complex64)
    assert BaseCollector._safe_numpy(t) is None


def test_safe_numpy_copies_ndarray_input():
    src = np.ones(3, dtype=np.float32)
    arr = BaseCollector._safe_numpy(src)
    src += 100.0
    np.testing.assert_array_equal(arr, np.ones(3, dtype=np.float32))


def test_zero_numel_parameter_does_not_crash_collector():
    col = ParameterCollector(config=ObservatoryConfig(parameter_sample_rate=1))
    col.collect("empty", torch.zeros(0), step=0, epoch=0)

    stored = col.get("empty")
    assert len(stored) == 1
    assert stored[0].stats["drift_from_init"] == 0.0
    assert stored[0].stats["drift_from_init_norm"] == 0.0


def test_zero_numel_parameter_model_can_be_watched():
    class EmptyParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.empty = nn.Parameter(torch.zeros(0))

        def forward(self, x):
            return x

    obs = Observatory(EmptyParam(), config=ObservatoryConfig(max_observations=50))
    obs.watch()
    assert obs.parameter_collector.total_observations > 0


def test_store_on_cpu_false_emits_deprecation_warning():
    with pytest.warns(UserWarning, match="store_on_cpu"):
        ObservatoryConfig(store_on_cpu=False)


def test_default_config_is_silent_about_store_on_cpu():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ObservatoryConfig()