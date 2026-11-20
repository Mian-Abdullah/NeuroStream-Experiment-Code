"""Privacy accounting: Rényi/zCDP curves for Gaussian releases and pure-DP Sparse-Vector instances, converted to
(eps, delta) with the Canonne-Kamath-Steinke conversion; cross-checked against Google's dp_accounting.
Record-level DP of each node's local data with respect to everything the node releases."""
import math
import numpy as np

ALPHAS = np.concatenate([np.linspace(1.01, 10, 600), np.linspace(10, 2000, 4000)])


def rdp_total(n_gauss=None, rho=0.0):
    """n_gauss: dict sigma -> number of Gaussian releases (noise multiplier sigma, L2 sensitivity 1).
    rho: total zCDP from pure-DP mechanisms (eps-DP => eps^2/2 zCDP, Bun & Steinke 2016)."""
    r = rho * ALPHAS
    for s, cnt in (n_gauss or {}).items():
        r = r + cnt * ALPHAS / (2 * s * s)
    return r


def eps_of(rdp, delta=1e-5):
    a = ALPHAS
    return float(np.min(rdp + np.log1p(-1 / a) - (np.log(delta) + np.log(a)) / (a - 1)))


def eps_dp_accounting(sigma, n, delta=1e-5, kind="rdp"):
    import dp_accounting as dpa
    from dp_accounting import rdp as rdpm, pld as pldm
    acc = rdpm.RdpAccountant() if kind == "rdp" else pldm.PLDAccountant()
    acc.compose(dpa.GaussianDpEvent(sigma), int(n))
    return float(acc.get_epsilon(delta))


def max_units(per_unit_gauss, per_unit_rho, eps_max=4.0, delta=1e-5, hi=10 ** 7):
    """Largest T such that T units (each with the given Gaussian releases and zCDP) stay within eps_max."""
    def eps_T(T):
        return eps_of(rdp_total({s: c * T for s, c in per_unit_gauss.items()}, per_unit_rho * T), delta)
    lo, h = 0, hi
    if eps_T(1) > eps_max:
        return 0
    while lo < h:
        mid = (lo + h + 1) // 2
        if eps_T(mid) <= eps_max:
            lo = mid
        else:
            h = mid - 1
    return lo
