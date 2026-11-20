"""Experiment A: Monte-Carlo validation of the federated-drift-consensus false-positive bound.

Claim in the original draft: P(FP) <= exp(-2K(theta-delta')^2)   [equal-weight, independent votes].
Corrected claim: P(FP) <= exp(-2 K_eff (theta-p)^2),  K_eff = (sum a_i)^2 / sum a_i^2,
where p >= max_i P(d_i = 1) is the per-node false-alarm probability inside one decision window.
Also: common-cause (correlated) failure, where no K-dependent bound can hold.
"""
import json, numpy as np
rng = np.random.default_rng(7)
THETA = 0.35
N = 400_000

def mc_fp(alpha, p, theta=THETA, n=N):
    d = rng.random((n, len(alpha))) < p
    return float(((d * alpha).sum(1) >= theta * alpha.sum()).mean())

rows = []
for wname, sigma in [("equal", 0.0), ("lognormal_s1.0", 1.0), ("lognormal_s2.0", 2.0)]:
    for K in [10, 25, 50, 100, 200]:
        a = np.ones(K) if sigma == 0 else rng.lognormal(0, sigma, K)
        keff = a.sum() ** 2 / (a ** 2).sum()
        for p in [0.10, 0.20, 0.25]:
            emp = mc_fp(a, p)
            orig = float(np.exp(-2 * K * (THETA - p) ** 2))
            corr = float(np.exp(-2 * keff * (THETA - p) ** 2))
            rows.append(dict(weights=wname, K=K, K_eff=round(keff, 2), p=p, empirical=emp,
                             bound_original=orig, bound_corrected=corr,
                             original_violated=bool(emp > orig), corrected_violated=bool(emp > corr)))

# common-cause failure: one group G (fraction gamma of nodes) alarms w.p. q_G when outage Z occurs (P(Z)=pz)
cc = []
pz, qG, p0 = 0.05, 0.9, 0.05
for gamma in [0.2, 0.5]:
    for K in [10, 25, 50, 100, 200, 400]:
        nG = int(round(gamma * K)); a = np.ones(K)
        n = 200_000
        z = rng.random(n) < pz
        d = rng.random((n, K)) < p0
        hit = rng.random((n, nG)) < qG
        d[:, :nG] = np.where(z[:, None], hit, d[:, :nG])
        fp = float(((d.sum(1)) >= THETA * K).mean())
        cc.append(dict(gamma=gamma, K=K, fp=fp, P_Z=pz, expected_mean_vote_fraction_given_Z=gamma*qG+(1-gamma)*p0))
json.dump(dict(theta=THETA, trials=N, rows=rows, common_cause=cc), open("results/exp_a.json", "w"), indent=1)

viol = [r for r in rows if r["original_violated"]]
print("rows:", len(rows), "| original bound violated in", len(viol), "rows | corrected violated in",
      sum(r["corrected_violated"] for r in rows))
for r in viol[:8]: print(r)
print("\ncommon-cause:")
for r in cc: print(r)
