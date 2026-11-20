"""Experiment C: controlled simulation of the NeuroStream decision layer (ECL detectors + FDC + causal alarm filter).

This is a SIMULATION OF THE DECISION LOGIC on synthetic streams with known ground truth. It is not a hardware or
real-dataset evaluation. It answers: (i) does reliability-weighted consensus suppress local faults, (ii) does a vote
threshold alone suffice against a common-cause outage, (iii) does conditioning on an observed common cause fix it,
(iv) what does a differentially-private (Sparse-Vector) alarm gate cost in detection quality?
"""
import json, math, sys, numpy as np

C, NB, MB, SX = 3, 5000, 10, 0.25          # channels, samples/block, blocks/half-window, per-sample std
M = NB * MB                                 # samples per detector sub-window  (m = 50,000)
BSTD = SX / math.sqrt(NB)                   # std of a block mean
DPRIME = 0.002
CUT = math.sqrt(math.log(4 / DPRIME) / (2 * M))      # Eq. (1)
MAG = 0.03                                  # mean shift for faults / outages / drift  (~3.4 x CUT)
W = 10                                      # DRR window (blocks)
LF, LO = 20, 20                             # fault / outage duration (blocks)
EPS_SVT, C_SVT = 0.2, 5                     # private gate: eps per SVT block, cutoff c
D = 1.0 / M

def gen(rng, K, B, t0, drift, outage, lam_f, lam_o=0.004, frac_out=0.5, hit_p=0.9, frac_drift=0.6):
    sh = np.zeros((K, B, C))
    for i, b in zip(*np.nonzero(rng.random((K, B)) < lam_f)):
        sh[i, b:b + LF, rng.integers(C)] += MAG
    hit = np.zeros((K, B), bool)
    if outage:
        Go = np.zeros(K, bool); Go[rng.permutation(K)[:int(frac_out * K)]] = True
        for b in np.nonzero(rng.random(B) < lam_o)[0]:
            h = Go & (rng.random(K) < hit_p)
            hit[h, b:b + LO] = True
        sh[:, :, 1] += MAG * hit
    Gd = np.zeros(K, bool)
    if drift:
        Gd[rng.permutation(K)[:int(frac_drift * K)]] = True
        sh[Gd, t0:, 0] += MAG
    Y = 0.5 + rng.normal(0, 0.05, (K, 1, C)) + sh + rng.normal(0, BSTD, (K, B, C))
    Z = 1.0 - 0.5 * hit + rng.normal(0, 0.03, (K, B))
    n_i = rng.lognormal(0, 0.5, K); rho = rng.uniform(0.6, 1.0, K)
    return Y, Z, n_i * rho, Gd

def window_diff(Y):
    K, B, _ = Y.shape
    cs = np.concatenate([np.zeros((K, 1, C)), np.cumsum(Y, 1)], 1)
    diff = np.full((K, B), np.nan)
    b = np.arange(2 * MB - 1, B)
    new = (cs[:, b + 1] - cs[:, b + 1 - MB]) / MB
    old = (cs[:, b + 1 - MB] - cs[:, b + 1 - 2 * MB]) / MB
    diff[:, b] = np.abs(new - old).max(2)
    return diff

def roll_any(a, w):  # d[:, b] = any(a[:, b-w+1..b])
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, 1)], 1)
    idx = np.arange(a.shape[1])
    lo = np.maximum(idx - w + 1, 0)
    return (cs[:, idx + 1] - cs[:, lo]) > 0

def last_alarm_index(alarm):
    idx = np.where(alarm, np.arange(alarm.shape[1])[None, :], -1)
    return np.maximum.accumulate(idx, 1)

def adj_alarm(Y, Z, nodes, ends):
    """backdoor adjustment: remove component of Y linearly explained by the observed common-cause covariate Z, re-test."""
    off = np.arange(2 * MB - 1, -1, -1)
    cols = ends[:, None] - off[None, :]
    Yw = Y[nodes[:, None], cols]                # (n, 2MB, C)
    Zw = Z[nodes[:, None], cols]                # (n, 2MB)
    zc = Zw - Zw.mean(1, keepdims=True)
    beta = (zc[:, :, None] * (Yw - Yw.mean(1, keepdims=True))).sum(1) / ((zc ** 2).sum(1)[:, None] + 1e-12)
    Ya = Yw - beta[:, None, :] * zc[:, :, None]
    d = np.abs(Ya[:, MB:].mean(1) - Ya[:, :MB].mean(1)).max(1)
    return d > CUT

def filter_votes(Y, Z, alpha, nodes, la_nodes, b):
    """node keeps its vote if the covariate-adjusted test fires at ANY block between its alarm and now."""
    keep = np.zeros(len(nodes), bool)
    for j in range(W + 1):
        ends = np.minimum(np.maximum(la_nodes, 2 * MB - 1) + j, b)
        keep |= adj_alarm(Y, Z, nodes, ends)
    return alpha[nodes[keep]].sum() / alpha.sum()

def episodes(flag, refractory=W):
    ep, last = [], -10 ** 9
    for b in np.nonzero(flag)[0]:
        if b - last > refractory: ep.append(b)
        last = b
    return ep

def run(seed, K=50, B=1200, t0=600, drift=True, outage=True, lam_f=0.001, thetas=(0.35, 0.45, 0.55), private=(), frac_drift=0.5):
    rng = np.random.default_rng(seed)
    Y, Z, alpha, Gd = gen(rng, K, B, t0, drift, outage, lam_f, frac_drift=frac_drift)
    diff = window_diff(Y)
    raw = np.nan_to_num(diff, nan=0.0) > CUT
    d = roll_any(raw, W)
    la = last_alarm_index(raw)
    pre = slice(2 * MB, t0 if drift else B)
    flags = {}
    sa = alpha.sum()
    v_raw = (alpha[:, None] * d).sum(0) / sa
    for th in thetas: flags[f"FDC(th={th})"] = v_raw >= th
    th0 = 0.35
    # + causal filter
    v_f = np.zeros(B)
    for b in np.nonzero(v_raw >= th0)[0]:
        nodes = np.nonzero(d[:, b])[0]
        v_f[b] = filter_votes(Y, Z, alpha, nodes, la[nodes, b], b)
    flags["FDC+filter(th=0.35)"] = v_f >= th0
    # + filter + private (sparse-vector) gate, several (eps, c) settings
    priv = {}
    for (eps_svt, c_svt) in private:
        e1 = e2 = eps_svt / 2
        sthr, sq = D / e1, 2 * c_svt * D / e2
        lat = np.full(K, -1); thr = rng.laplace(0, sthr, K); la_p = np.full(K, -1)
        dp = np.zeros((K, B), bool); above_pre = 0
        for b in range(2 * MB - 1, B):
            active = b >= lat
            q = diff[:, b] + rng.laplace(0, sq, K)
            ab = active & (q >= CUT + thr)
            if ab.any():
                lat[ab] = b + W; la_p[ab] = b; thr[ab] = rng.laplace(0, sthr, ab.sum())
                if b < (t0 if drift else B): above_pre += ab.sum()
            dp[:, b] = b < lat
        vp = (alpha[:, None] * dp).sum(0) / sa
        vpf = np.zeros(B)
        for b in np.nonzero(vp >= th0)[0]:
            nodes = np.nonzero(dp[:, b])[0]
            vpf[b] = filter_votes(Y, Z, alpha, nodes, la_p[nodes], b)
        nm = f"FDC+filter+SVT(eps={eps_svt},c={c_svt})"
        flags[nm] = vpf >= th0
        flags[nm + " [no filter]"] = vp >= th0
        priv[nm] = dict(p_hat=float(dp[:, pre].mean()), above_pre=float(above_pre / K / (pre.stop - pre.start) * 1000))
    out = {"p_hat": float(d[:, pre].mean()), "K_eff": float(sa ** 2 / (alpha ** 2).sum()),
           "node_episodes_pre_per_1000": float(sum(len(episodes(raw[i, pre.start:pre.stop], 1)) for i in range(K)) / (pre.stop - pre.start) * 1000),
           "methods": {}}
    out["priv"] = priv
    for name, fl in flags.items():
        r = {"fp_block_rate": float(fl[pre].mean()),
             "false_episodes_per_1000": len(episodes(fl[pre.start:pre.stop])) / (pre.stop - pre.start) * 1000}
        if drift:
            post = fl[t0:t0 + 60]
            r["detected"] = bool(post.any()); r["delay"] = int(np.argmax(post)) if post.any() else None
        out["methods"][name] = r
    # M1 node-level detection of drift (drifted nodes)
    if drift:
        pn = raw[Gd][:, t0:t0 + 60]
        out["node_detect_frac"] = float(pn.any(1).mean())
        out["node_delay_median"] = float(np.median(np.argmax(pn, 1)[pn.any(1)])) if pn.any() else None
    return out

def agg(runs, K):
    names = list(runs[0]["methods"].keys())
    res = {"K": K, "n": len(runs), "p_hat": float(np.mean([r["p_hat"] for r in runs])), "K_eff": float(np.mean([r["K_eff"] for r in runs])),
           "node_episodes_pre_per_1000": float(np.mean([r["node_episodes_pre_per_1000"] for r in runs])), "methods": {}}
    res["priv"] = {k: {"p_hat": float(np.mean([r["priv"][k]["p_hat"] for r in runs])), "above_pre_per_node_per_1000": float(np.mean([r["priv"][k]["above_pre"] for r in runs]))} for k in runs[0]["priv"]}
    for n in names:
        ms = [r["methods"][n] for r in runs]
        a = {"fp_block_rate": float(np.mean([m["fp_block_rate"] for m in ms])),
             "false_episodes_per_1000": float(np.mean([m["false_episodes_per_1000"] for m in ms]))}
        if "detected" in ms[0]:
            dl = [m["delay"] for m in ms if m["detected"]]
            a["detect_rate"] = float(np.mean([m["detected"] for m in ms])); a["delay_median"] = float(np.median(dl)) if dl else None
        a["wasted_node_retrains_per_1000"] = a["false_episodes_per_1000"] * K
        res["methods"][n] = a
    if "node_detect_frac" in runs[0]:
        res["node_detect_frac"] = float(np.mean([r["node_detect_frac"] for r in runs]))
        res["node_delay_median"] = float(np.median([r["node_delay_median"] for r in runs if r["node_delay_median"] is not None]))
    return res

if __name__ == "__main__":
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    PRIV = [(0.2, 1), (0.2, 5), (0.2, 10), (0.5, 5)]
    out = {"params": dict(CUT=CUT, MAG=MAG, M=M, NB=NB, MB=MB, W=W, seeds=NS, priv_cfgs=PRIV)}
    out["main_outage_drift"] = agg([run(s, outage=True, drift=True, private=PRIV) for s in range(NS)], 50)
    out["local_faults_only"] = agg([run(1000 + s, outage=False, drift=True, private=PRIV) for s in range(NS)], 50)
    out["outage_no_drift"] = agg([run(2000 + s, outage=True, drift=False, private=PRIV) for s in range(NS)], 50)
    out["drift_fraction_sweep"] = []
    for fd in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        r = agg([run(4000 + s, outage=True, drift=True, frac_drift=fd) for s in range(NS)], 50); r["frac_drift"] = fd
        out["drift_fraction_sweep"].append(r)
    sweep = []
    for lam, tag in [(0.0025, "p~0.1"), (0.0055, "p~0.2")]:
        for K in [10, 25, 50, 100, 200]:
            r = agg([run(3000 + s, K=K, drift=False, outage=False, lam_f=lam, thetas=(0.35,)) for s in range(NS)], K)
            r["tag"] = tag; sweep.append(r)
    out["K_sweep"] = sweep
    json.dump(out, open("results/exp_c.json", "w"), indent=1)
    def show(title, r):
        print("\n==", title, "| p_hat=%.3f K_eff=%.1f node-episodes/1000blk=%.1f" % (r["p_hat"], r["K_eff"], r["node_episodes_pre_per_1000"]))
        for k, v in r.get("priv", {}).items(): print("   private gate", k, "p_hat=%.3f aboves/node/1000blk=%.1f" % (v["p_hat"], v["above_pre_per_node_per_1000"]))
        if "node_detect_frac" in r: print("   M1 local-only: node detect frac %.2f, median node delay %.1f blocks" % (r["node_detect_frac"], r["node_delay_median"]))
        for n, a in r["methods"].items():
            print("  %-40s fp_block=%.4f fp_ep/1000=%.2f wasted/1000=%.1f detect=%s delay=%s" %
                  (n, a["fp_block_rate"], a["false_episodes_per_1000"], a["wasted_node_retrains_per_1000"], a.get("detect_rate", "-"), a.get("delay_median", "-")))
    show("MAIN: outage(50% nodes) + drift(50% nodes)", out["main_outage_drift"])
    show("local faults only + drift", out["local_faults_only"])
    show("outage, no drift", out["outage_no_drift"])
    print("\nDrift-fraction sweep (detect rate / fp episodes per 1000 blocks):")
    for r in out["drift_fraction_sweep"]:
        print("frac_drift=%.1f " % r["frac_drift"] + " | ".join("%s det=%.2f fp=%.2f" % (n.replace("FDC", "F"), a["detect_rate"], a["false_episodes_per_1000"]) for n, a in r["methods"].items()))
    print("\nK sweep (FDC th=0.35, local faults only):")
    for r in out["K_sweep"]:
        a = r["methods"]["FDC(th=0.35)"]
        bound = math.exp(-2 * r["K_eff"] * max(0.35 - r["p_hat"], 0) ** 2)
        print(r["tag"], "K=%3d p_hat=%.3f K_eff=%.1f  emp_fp_block=%.5f  corrected_bound=%.4f" % (r["K"], r["p_hat"], r["K_eff"], a["fp_block_rate"], bound))
