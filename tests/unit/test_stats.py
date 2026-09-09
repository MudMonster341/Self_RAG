"""Tests for selfrag.eval.stats.

Covers: paired_bootstrap on identical inputs gives delta 0 and verdict
"indistinguishable"; a real, consistent improvement is detected and
verdicted "b_better"; the minimum-practical-effect gate makes a "winner"
verdict impossible below the pre-registered threshold even when the CI
excludes zero; the randomization test's p-value behaves sensibly; alignment
drops unpaired/None queries; and minimum_detectable_effect / required_n are
consistent inverses, cross-checked against known standard-normal quantiles.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from selfrag.eval.stats import (
    align_by_query_id,
    compare_paired,
    minimum_detectable_effect,
    paired_bootstrap,
    randomization_test,
    required_n,
)

# --------------------------------------------------------------------------
# paired_bootstrap / compare_paired on identical inputs
# --------------------------------------------------------------------------


def test_paired_bootstrap_identical_inputs_yields_zero_delta():
    values = np.array([0.5, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4])
    result = paired_bootstrap(values, values, n_resamples=2000, seed=0)
    assert result.mean_delta == 0.0
    assert result.ci_low <= 0.0 <= result.ci_high
    assert result.n == 7


def test_compare_paired_identical_inputs_is_indistinguishable():
    values = np.array([0.5, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4])
    cmp = compare_paired(values, values, min_effect=0.02, n_resamples=2000, n_permutations=2000, seed=0)
    assert cmp.delta == 0.0
    assert cmp.verdict == "indistinguishable"
    assert cmp.p_value == pytest.approx(1.0)


# --------------------------------------------------------------------------
# A real, consistent difference is detected
# --------------------------------------------------------------------------


def test_compare_paired_detects_consistent_large_improvement():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.3, 0.5, size=50)
    b = a + 0.3  # a large, consistent improvement on every single query
    cmp = compare_paired(a, b, min_effect=0.02, n_resamples=2000, n_permutations=2000, seed=0)
    assert cmp.verdict == "b_better"
    assert cmp.delta == pytest.approx(0.3, abs=1e-9)
    assert cmp.ci_low > 0.0
    assert cmp.p_value < 0.01


def test_compare_paired_detects_consistent_large_regression():
    rng = np.random.default_rng(2)
    a = rng.uniform(0.5, 0.7, size=50)
    b = a - 0.3
    cmp = compare_paired(a, b, min_effect=0.02, n_resamples=2000, n_permutations=2000, seed=0)
    assert cmp.verdict == "a_better"
    assert cmp.ci_high < 0.0


# --------------------------------------------------------------------------
# The minimum-practical-effect gate: no winner below threshold, even if "significant"
# --------------------------------------------------------------------------


def test_min_effect_gate_blocks_winner_despite_zero_variance_significance():
    """A tiny but perfectly consistent delta (every query improves by exactly
    0.005) would have a bootstrap CI that is a single point excluding zero --
    "significant" in the naive sense. It must still be reported as
    indistinguishable because 0.005 is below the pre-registered min_effect
    of 0.02. This is the guard rail the whole PairedComparison dataclass
    exists to enforce."""
    a = np.array([0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47])
    b = a + 0.005
    cmp = compare_paired(a, b, min_effect=0.02, n_resamples=2000, n_permutations=2000, seed=0)
    assert cmp.ci_low > 0.0  # CI would call this "significant" on its own
    assert cmp.delta == pytest.approx(0.005, abs=1e-9)
    assert cmp.verdict == "indistinguishable"  # but the effect gate overrides it


def test_min_effect_gate_allows_winner_once_above_threshold():
    a = np.array([0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47])
    b = a + 0.05  # comfortably above min_effect=0.02, still perfectly consistent
    cmp = compare_paired(a, b, min_effect=0.02, n_resamples=2000, n_permutations=2000, seed=0)
    assert cmp.verdict == "b_better"


def test_ci_containing_zero_is_indistinguishable_even_above_min_effect():
    """The other half of the gate: a large point-estimate delta with high
    per-query variance (CI still spans zero) must also be withheld, even
    though |delta| clears min_effect."""
    rng = np.random.default_rng(3)
    a = rng.uniform(0.0, 1.0, size=6)
    b = rng.uniform(0.0, 1.0, size=6)  # totally unrelated to a -- noisy, not a real paired improvement
    cmp = compare_paired(a, b, min_effect=0.02, n_resamples=2000, n_permutations=2000, seed=0)
    if cmp.ci_low <= 0.0 <= cmp.ci_high:
        assert cmp.verdict == "indistinguishable"


# --------------------------------------------------------------------------
# randomization_test
# --------------------------------------------------------------------------


def test_randomization_test_p_value_in_valid_range():
    rng = np.random.default_rng(4)
    a = rng.uniform(size=20)
    b = rng.uniform(size=20)
    result = randomization_test(a, b, n_permutations=1000, seed=0)
    assert 0.0 < result.p_value <= 1.0


def test_randomization_test_zero_delta_gives_p_value_one():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    result = randomization_test(values, values, n_permutations=500, seed=0)
    assert result.mean_delta == 0.0
    assert result.p_value == 1.0


def test_randomization_test_large_consistent_delta_gives_small_p_value():
    a = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    b = a + 0.5
    result = randomization_test(a, b, n_permutations=2000, seed=0)
    # With 8 queries all shifted the same direction, only 2 of 2^8=256 sign
    # patterns are as extreme (all +1 or all -1) -- p should be small.
    assert result.p_value < 0.05


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_paired_functions_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        randomization_test(np.array([1.0, 2.0]), np.array([1.0]))


def test_paired_functions_reject_empty_input():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([]), np.array([]))


def test_paired_functions_reject_nan():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([1.0, float("nan")]), np.array([1.0, 2.0]))


# --------------------------------------------------------------------------
# align_by_query_id
# --------------------------------------------------------------------------


def test_align_by_query_id_drops_unmatched_and_none():
    qa = ["q1", "q2", "q3", "q4"]
    va = [0.5, 0.6, None, 0.9]
    qb = ["q1", "q2", "q3", "q5"]
    vb = [0.4, None, 0.7, 0.2]
    a, b, kept = align_by_query_id(qa, va, qb, vb)
    # q1: both defined -> kept. q2: b is None -> dropped. q3: a is None -> dropped.
    # q4: missing from b -> dropped. q5: missing from a -> dropped.
    assert kept == ["q1"]
    assert a.tolist() == [0.5]
    assert b.tolist() == [0.4]


def test_align_by_query_id_empty_intersection():
    a, b, kept = align_by_query_id(["q1"], [0.5], ["q2"], [0.5])
    assert kept == []
    assert a.shape == (0,)
    assert b.shape == (0,)


# --------------------------------------------------------------------------
# minimum_detectable_effect / required_n
# --------------------------------------------------------------------------


def test_minimum_detectable_effect_matches_known_normal_quantiles():
    n = 100
    sd = 0.2
    mde = minimum_detectable_effect(n, sd, power=0.8, alpha=0.05)
    # Well-known standard normal quantiles: z_0.975 ~= 1.9599639845400545,
    # z_0.8 ~= 0.8416212335729143 (independent of this module's own
    # rational-approximation implementation).
    expected = (1.9599639845400545 + 0.8416212335729143) * sd / math.sqrt(n)
    assert mde == pytest.approx(expected, rel=1e-6)


def test_minimum_detectable_effect_shrinks_with_more_queries():
    sd = 0.25
    mde_small = minimum_detectable_effect(50, sd)
    mde_large = minimum_detectable_effect(500, sd)
    assert mde_large < mde_small


def test_required_n_inverts_minimum_detectable_effect():
    n = 200
    sd = 0.15
    mde = minimum_detectable_effect(n, sd)
    n_req = required_n(mde, sd)
    assert n_req == pytest.approx(n, abs=1)


def test_required_n_grows_as_effect_shrinks():
    sd = 0.2
    n_for_big_effect = required_n(0.1, sd)
    n_for_small_effect = required_n(0.01, sd)
    assert n_for_small_effect > n_for_big_effect


def test_minimum_detectable_effect_rejects_bad_input():
    with pytest.raises(ValueError):
        minimum_detectable_effect(0, 0.2)
    with pytest.raises(ValueError):
        minimum_detectable_effect(10, -0.1)


def test_required_n_rejects_nonpositive_effect():
    with pytest.raises(ValueError):
        required_n(0.0, 0.2)
    with pytest.raises(ValueError):
        required_n(-0.01, 0.2)


def test_required_n_zero_sd_returns_one():
    # With zero variance in the deltas, a single paired observation already
    # resolves any nonzero effect exactly.
    assert required_n(0.05, 0.0) == 1
