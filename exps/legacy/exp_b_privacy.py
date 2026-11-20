"""Experiment B: privacy accounting with a real accountant (Google dp_accounting), cross-checked by closed forms.

Threat model: record-level (eps,delta)-DP of each node's released updates. Each release = sum of per-record-clipped
gradients (L2 sensitivity C) + N(0, (sigma*C)^2 I)  -> Gaussian mechanism with noise multiplier sigma (no subsampling
amplification claimed; this is conservative).
Compared accountants: (a) basic composition, (b) advanced composition, (c) RDP, (d) PLD.
Drift gating: node releases only in "active" rounds (fraction f).  The gate is a data-dependent decision, so it is
implemented as AboveThreshold/Sparse-Vector with cutoff c and per-block budget eps_svt (pure eps-DP => rho=eps^2/2 zCDP),
and that cost is charged.
"""
import json, math, numpy as np
import dp_accounting as dpa
from dp_accounting import rdp as rdpm, pld as pldm

EPS_MAX, DELTA = 4.0, 1e-5
SIGMA = 25.0

def eps_rdp(T, sigma=SIGMA, delta=DELTA, extra_rho=0.0):
    """epsilon after T Gaussian releases (+ optional zCDP-style extra cost rho) via dp_accounting RDP accountant."""
    acc = rdpm.RdpAccountant()
    if T > 0: acc.compose(dpa.GaussianDpEvent(sigma), T)
    eps = acc.get_epsilon(delta) if T > 0 else 0.0
    return eps

def eps_rdp_closed(T, rho_extra=0.0, sigma=SIGMA, delta=DELTA):
    """Mironov conversion, minimise over real alpha>1: eps = T*a/(2s^2) + rho_extra*a + ln(1/d)/(a-1)."""
    r = T / (2 * sigma ** 2) + rho_extra
    al = np.linspace(1.001, 2000, 400000)
    return float(np.min(r * al + math.log(1 / delta) / (al - 1)))

def eps_rdp_total(T_active, c, eps_svt, sigma=SIGMA, delta=DELTA):
    """Gaussian releases + SVT blocks, RDP of eps-DP bounded by alpha*eps^2/2 (Bun-Steinke zCDP)."""
    rho_svt = 0.0 if eps_svt is None else math.ceil(T_active / c) * eps_svt ** 2 / 2
    # use dp_accounting for Gaussian part, add the SVT RDP curve through closed form with fine alpha grid
    alphas = np.concatenate([np.linspace(1.01, 10, 400), np.linspace(10, 1000, 2000)])
    r = T_active / (2 * sigma ** 2) + rho_svt
    # Canonne-Kamath-Steinke style (tighter) conversion for pure RDP-of-form rho*alpha
    eps = np.min(r * alphas + np.log1p(-1 / alphas) - (np.log(delta) + np.log(alphas)) / (alphas - 1))
    return float(eps)

def max_rounds(fn, hi=200000):
    lo, h = 0, hi
    while lo < h:
        mid = (lo + h + 1) // 2
        if fn(mid) <= EPS_MAX: lo = mid
        else: h = mid - 1
    return lo

out = {"sigma": SIGMA, "eps_max": EPS_MAX, "delta": DELTA}

# (a) basic and (b) advanced composition of per-round classical Gaussian (eps0, delta0)
def basic(T):
    d0 = DELTA / T
    eps0 = math.sqrt(2 * math.log(1.25 / d0)) / SIGMA
    return T * eps0
def advanced(T):
    d_prime = DELTA / 2; d0 = DELTA / (2 * T)
    eps0 = math.sqrt(2 * math.log(1.25 / d0)) / SIGMA
    return math.sqrt(2 * T * math.log(1 / d_prime)) * eps0 + T * eps0 * (math.exp(eps0) - 1)
def pld(T):
    acc = pldm.PLDAccountant(); acc.compose(dpa.GaussianDpEvent(SIGMA), T); return acc.get_epsilon(DELTA)

res = {}
res["basic"] = max_rounds(basic)
res["advanced"] = max_rounds(advanced)
res["rdp_dp_accounting"] = max_rounds(lambda T: eps_rdp(T))
res["rdp_closed_form_mironov"] = max_rounds(lambda T: eps_rdp_closed(T))
res["pld"] = max_rounds(pld, hi=20000)
out["rounds_until_eps4_all_nodes_active"] = res

# curves for figure
Ts = list(range(0, 1201, 20))
out["curves"] = {"T": Ts,
                 "basic": [basic(t) if t else 0 for t in Ts],
                 "advanced": [advanced(t) if t else 0 for t in Ts],
                 "rdp": [eps_rdp(t) for t in Ts],
                 "pld": [pld(t) if t else 0 for t in Ts]}

# (d) gating with the correct SVT noise coupling (Lyu-Su-Li 2017): threshold noise Lap(D/eps1), query noise Lap(2cD/eps2),
# eps1=eps2=eps_svt/2, sensitivity D = 1/m for a window-mean difference of [0,1]-bounded data.
DELTA_PRIME = 0.002
gate = []
base = res["rdp_dp_accounting"]
for m in [5_000, 50_000]:
    cut = math.sqrt(math.log(4 / DELTA_PRIME) / (2 * m))
    D = 1.0 / m
    for eps_svt in [0.2, 0.5, 1.0]:
        for c in [1, 5, 10]:
            scale_q = 2 * c * D / (eps_svt / 2)
            useful = scale_q <= cut / 5
            t_act = max_rounds(lambda T: eps_rdp_total(T, c, eps_svt))
            row = dict(m=m, eps_svt=eps_svt, c=c, cut=round(cut, 5), query_noise_scale=round(scale_q, 5),
                       noise_over_cut=round(scale_q / cut, 3), useful=bool(useful), T_active_max=t_act,
                       breakeven_f=round(t_act / base, 3))
            for f in [0.2, 0.1, 0.05]:
                row[f"horizon_f{f}"] = int(t_act / f)
                row[f"gain_f{f}"] = round(t_act / f / base, 2)
            gate.append(row)
t_free = max_rounds(lambda T: eps_rdp_total(T, 10, None))
out["gating_assumed_free"] = {f"f{f}": int(t_free / f) for f in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]}
out["gating"] = gate
json.dump(out, open("results/exp_b.json", "w"), indent=1)

print("sigma =", SIGMA, " eps_max =", EPS_MAX, " delta =", DELTA)
print("rounds until eps>4 (all nodes active every round):")
for k, v in res.items(): print(f"  {k:28s} {v}")
print("\nGate assumed FREE (original claim):", out["gating_assumed_free"])
print("\nGate CHARGED (SVT):  baseline no-gating =", base)
print(f"{'m':>6} {'eps':>4} {'c':>3} {'noise/cut':>9} {'useful':>6} {'T_act':>6} {'brk-f':>6} {'gain@.2':>8} {'gain@.1':>8} {'gain@.05':>8}")
for r in gate:
    print(f"{r['m']:>6} {r['eps_svt']:>4} {r['c']:>3} {r['noise_over_cut']:>9} {str(r['useful']):>6} {r['T_active_max']:>6} {r['breakeven_f']:>6} {r['gain_f0.2']:>8} {r['gain_f0.1']:>8} {r['gain_f0.05']:>8}")
