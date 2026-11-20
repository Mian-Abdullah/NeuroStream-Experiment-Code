"""Uncertainty helpers used by every experiment: Wilson intervals, bootstrap intervals, Poisson upper bounds."""
import math
import numpy as np

Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan"), float("nan")]
    if np.all(x == x[0]):
        return [float(x[0]), float(x[0])]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def poisson_ucb(count, exposure, level=0.95):
    """One-sided upper confidence bound on a Poisson rate (exact, via chi-square quantile)."""
    from scipy.stats import chi2
    return float(chi2.ppf(level, 2 * (count + 1)) / 2 / exposure)


def summarize_rate(per_seed_values, seed=0):
    x = np.asarray(per_seed_values, float)
    return dict(mean=float(np.nanmean(x)), ci95=boot_ci(x, seed=seed), n=int(np.sum(~np.isnan(x))))


def summarize_prop(bools):
    b = np.asarray(bools, bool)
    k, n = int(b.sum()), len(b)
    return dict(rate=k / n if n else float("nan"), k=k, n=n, ci95=wilson(k, n))


def summarize_counts_per_1000(counts, exposure_blocks):
    """counts: per-seed event counts; exposure_blocks: per-seed exposure. Returns mean rate, bootstrap CI and,
    when no event was seen, the exact Poisson 95% upper bound."""
    counts = np.asarray(counts, float)
    exp_ = np.asarray(exposure_blocks, float)
    rate = counts / exp_ * 1000
    out = summarize_rate(rate)
    tot, texp = counts.sum(), exp_.sum() / 1000
    out["total_events"] = int(tot)
    out["exposure_thousand_blocks"] = float(texp)
    out["poisson_ucb95"] = poisson_ucb(tot, texp)
    return out
