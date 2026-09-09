"""Paired significance testing for comparing two retrieval configurations.

Comparisons in this system are ALWAYS paired on the same queries: config A
and config B are run over the identical query set, and every test here
consumes per-query metric arrays aligned by ``query_id``, never independent
samples. Pairing is what lets a small eval set detect a real effect at all --
per-query variance (some queries are just harder than others, for both
configs) is enormous compared to between-config variance, and an unpaired
test would spend its entire power budget on the wrong source of noise.
Comparing per-query *deltas* cancels that shared variance out.

``numpy`` only, deliberately: this module has no scipy/sklearn dependency.
The inverse-normal-CDF helper below (``_norm_ppf``) exists only because of
that constraint -- it is a well-known closed-form rational approximation
(Acklam), not a reimplementation of anything novel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Verdict = str  # "b_better" | "a_better" | "indistinguishable"


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a paired bootstrap over per-query deltas (b - a)."""

    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    frac_favoring_b: float
    n_resamples: int
    seed: int
    ci: float


@dataclass(frozen=True)
class RandomizationResult:
    """Result of a sign-flip randomization test over per-query deltas (b - a)."""

    n: int
    mean_delta: float
    p_value: float
    n_permutations: int
    seed: int


@dataclass(frozen=True)
class PairedComparison:
    """A full paired comparison, gated by a pre-registered minimum effect size.

    ``verdict`` can only be ``"b_better"`` or ``"a_better"`` when BOTH the
    bootstrap CI excludes zero AND the magnitude of the delta clears
    ``min_effect``. That gate is the entire point of this dataclass: with
    enough queries, arbitrarily tiny and operationally meaningless deltas
    become "statistically significant" (CI excludes zero). Requiring the
    magnitude to also clear a pre-registered minimum practical effect --
    chosen and written down *before* the comparison is run, not tuned
    afterward to fit whatever delta happened to appear -- is what keeps a
    huge eval set from manufacturing winners out of noise-floor differences.
    """

    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    min_effect: float
    verdict: Verdict


def align_by_query_id(
    query_ids_a: list[str],
    values_a: list[float | None],
    query_ids_b: list[str],
    values_b: list[float | None],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pair two per-query metric series by query_id, keeping only common, non-null queries.

    Queries missing from either side, or where either side's metric
    guard-railed to ``None`` (see ``metrics.py`` -- typically "no relevant
    gold spans for this query"), are dropped rather than imputed. Comparing
    on a query where one side has no defined value would silently change
    what queries the test is about between A and B, which is exactly the
    unpaired-comparison mistake this module exists to prevent.

    Returns ``(a, b, query_ids)`` where ``a``/``b`` are aligned float arrays
    and ``query_ids`` is the sorted list of query_ids actually used, in the
    same order as the arrays.
    """
    map_a = dict(zip(query_ids_a, values_a, strict=True))
    map_b = dict(zip(query_ids_b, values_b, strict=True))
    common = sorted(set(map_a) & set(map_b))
    kept: list[str] = []
    out_a: list[float] = []
    out_b: list[float] = []
    for qid in common:
        va, vb = map_a[qid], map_b[qid]
        if va is None or vb is None:
            continue
        out_a.append(va)
        out_b.append(vb)
        kept.append(qid)
    return np.asarray(out_a, dtype=float), np.asarray(out_b, dtype=float), kept


def _check_paired(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"a and b must be the same length (paired), got {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"a and b must be 1-D per-query arrays, got shape {a.shape}")
    if a.shape[0] == 0:
        raise ValueError("cannot compare zero paired observations")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("a and b must not contain NaN -- filter with align_by_query_id first")
    return a, b


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapResult:
    """Paired bootstrap over per-query deltas ``b - a``.

    Resamples query indices with replacement (the standard paired-bootstrap
    move: resample *pairs*, never a and b independently, or pairing is
    destroyed) ``n_resamples`` times, and reports the mean delta, a
    percentile confidence interval, and the fraction of resamples in which b
    beats a.
    """
    a, b = _check_paired(a, b)
    n = a.shape[0]
    deltas = b - a
    mean_delta = float(deltas.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = deltas[idx].mean(axis=1)

    alpha = 1.0 - ci
    ci_low = float(np.quantile(resample_means, alpha / 2))
    ci_high = float(np.quantile(resample_means, 1 - alpha / 2))
    frac_favoring_b = float(np.mean(resample_means > 0))

    return BootstrapResult(
        n=n,
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        frac_favoring_b=frac_favoring_b,
        n_resamples=n_resamples,
        seed=seed,
        ci=ci,
    )


def randomization_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 0,
) -> RandomizationResult:
    """Two-sided paired randomization (sign-flip) test on ``b - a``.

    Under the null hypothesis "A and B are exchangeable per query", flipping
    the sign of each query's delta independently and at random should not
    change the distribution of the mean delta. The p-value is the fraction
    of random sign-flips whose |mean delta| is at least as extreme as the
    one actually observed.

    Uses the ``(count + 1) / (n_permutations + 1)`` correction (Davison &
    Hinkley) rather than a raw fraction, so a p-value of exactly 0.0 is
    never reported from a finite number of permutations -- that would
    overstate certainty no finite resampling procedure can actually provide.
    """
    a, b = _check_paired(a, b)
    n = a.shape[0]
    deltas = b - a
    mean_delta = float(deltas.mean())
    observed = abs(mean_delta)

    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_permutations, n)) * 2 - 1  # -1 or +1
    perm_means = (signs * deltas).mean(axis=1)

    count_as_extreme = int(np.sum(np.abs(perm_means) >= observed))
    p_value = (count_as_extreme + 1) / (n_permutations + 1)

    return RandomizationResult(
        n=n, mean_delta=mean_delta, p_value=p_value, n_permutations=n_permutations, seed=seed
    )


def compare_paired(
    a: np.ndarray,
    b: np.ndarray,
    min_effect: float,
    n_resamples: int = 10000,
    n_permutations: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> PairedComparison:
    """Run the full paired comparison (bootstrap CI + randomization p-value) and gate the verdict.

    ``min_effect`` has no default: it must be chosen and stated at the call
    site, before the comparison is run, for the same reason the overlap
    rule in ``metrics.py`` has no default -- a threshold picked after seeing
    the delta is not pre-registered, it is post-hoc rationalization.

    The verdict is ``"indistinguishable"`` whenever the bootstrap CI
    contains zero OR ``|delta| < min_effect`` -- either condition alone is
    enough to withhold a winner. Only when both a) the sign is consistent
    enough that zero is outside the CI and b) the magnitude clears the
    pre-registered bar does the verdict become ``"b_better"`` or
    ``"a_better"``.
    """
    boot = paired_bootstrap(a, b, n_resamples=n_resamples, seed=seed, ci=ci)
    rand = randomization_test(a, b, n_permutations=n_permutations, seed=seed)

    a_arr, b_arr = _check_paired(a, b)
    mean_a = float(a_arr.mean())
    mean_b = float(b_arr.mean())
    delta = boot.mean_delta

    ci_contains_zero = boot.ci_low <= 0.0 <= boot.ci_high
    below_min_effect = abs(delta) < min_effect

    verdict: Verdict
    if ci_contains_zero or below_min_effect:
        verdict = "indistinguishable"
    elif delta > 0:
        verdict = "b_better"
    else:
        verdict = "a_better"

    return PairedComparison(
        n=boot.n,
        mean_a=mean_a,
        mean_b=mean_b,
        delta=delta,
        ci_low=boot.ci_low,
        ci_high=boot.ci_high,
        p_value=rand.p_value,
        min_effect=min_effect,
        verdict=verdict,
    )


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function).

    Peter Acklam's rational approximation (max relative error ~1.15e-9),
    used only because this module is restricted to numpy and must not pull
    in scipy for a single function. Not a general-purpose statistics
    library substitute -- just this one closed-form approximation.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def minimum_detectable_effect(
    n: int, sd_of_deltas: float, power: float = 0.8, alpha: float = 0.05
) -> float:
    """Smallest true mean delta this eval set's size can reliably distinguish from zero.

    Normal approximation to the power function of a paired-difference test:
    ``mde = (z_(1-alpha/2) + z_power) * sd / sqrt(n)``. This is exact only in
    the large-n limit; per-query IR metrics are bounded (e.g. nDCG in
    [0, 1]) so their deltas are not exactly normal, especially for small n.
    Treat the result as an eval-set-sizing heuristic to decide whether an
    experiment is even worth running, not as an exact guarantee -- it tells
    the CLI when to warn "this eval set is too small to resolve an effect
    this small," not to certify a specific p-value in advance.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if sd_of_deltas < 0:
        raise ValueError(f"sd_of_deltas must be non-negative, got {sd_of_deltas}")
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_power = _norm_ppf(power)
    return (z_alpha + z_power) * sd_of_deltas / math.sqrt(n)


def required_n(effect: float, sd_of_deltas: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Smallest number of paired queries needed to detect ``effect`` at the given power/alpha.

    The algebraic inverse of ``minimum_detectable_effect``; same normal
    approximation and same caveat about exactness for small n.
    """
    if effect <= 0:
        raise ValueError(f"effect must be positive, got {effect}")
    if sd_of_deltas < 0:
        raise ValueError(f"sd_of_deltas must be non-negative, got {sd_of_deltas}")
    if sd_of_deltas == 0:
        return 1
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_power = _norm_ppf(power)
    n = ((z_alpha + z_power) * sd_of_deltas / effect) ** 2
    return max(1, math.ceil(n))
