"""Sanity tests for single-vector chunk aggregation methods."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunk_aggregation import aggregate


def _unit_rows(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def test_single_window_returns_that_window():
    rng = np.random.default_rng(0)
    e = _unit_rows(rng, 1, 16)
    for method in ("mean", "gem", "coherence_mean", "lse"):
        c, R = aggregate(e, method=method)
        assert np.allclose(c, e[0], atol=1e-6), method
        assert R == 1.0


def test_identical_windows_return_same_direction():
    rng = np.random.default_rng(1)
    v = _unit_rows(rng, 1, 32)[0]
    e = np.tile(v, (7, 1))
    for method in ("mean", "gem", "coherence_mean", "lse"):
        c, R = aggregate(e, method=method)
        assert np.allclose(c, v, atol=1e-6), method
        assert abs(R - 1.0) < 1e-9


def test_all_methods_return_unit_norm():
    rng = np.random.default_rng(2)
    e = _unit_rows(rng, 8, 64)
    for method in ("mean", "gem", "coherence_mean", "lse"):
        c, R = aggregate(e, method=method)
        assert abs(np.linalg.norm(c) - 1.0) < 1e-5, method
        assert 0.0 <= R <= 1.0


def test_gem_p_one_matches_mean():
    rng = np.random.default_rng(3)
    e = _unit_rows(rng, 6, 20)
    c_mean, _ = aggregate(e, method="mean")
    c_gem, _ = aggregate(e, method="gem", gem_p=1.0)
    assert np.allclose(c_mean, c_gem, atol=1e-6)


def test_coherence_mean_tau_zero_matches_mean():
    rng = np.random.default_rng(4)
    e = _unit_rows(rng, 5, 24)
    c_mean, _ = aggregate(e, method="mean")
    c_coh, _ = aggregate(e, method="coherence_mean", coherence_tau=0.0)
    assert np.allclose(c_mean, c_coh, atol=1e-6)


def test_coherence_R_is_mean_independent():
    rng = np.random.default_rng(5)
    e = _unit_rows(rng, 10, 32)
    R_values = [aggregate(e, method=m)[1] for m in
                ("mean", "gem", "coherence_mean", "lse")]
    assert max(R_values) - min(R_values) < 1e-12


def test_orthogonal_windows_have_low_coherence():
    e = np.eye(5, 64).astype(np.float64)
    _, R = aggregate(e, method="mean")
    assert R < 1.0 / np.sqrt(5) + 1e-6


def test_unknown_method_raises():
    rng = np.random.default_rng(6)
    e = _unit_rows(rng, 3, 8)
    try:
        aggregate(e, method="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")


def test_empty_input_raises():
    try:
        aggregate(np.zeros((0, 4)), method="mean")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty input")
