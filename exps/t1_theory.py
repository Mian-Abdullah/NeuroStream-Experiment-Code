"""Monte-Carlo validation of the anytime-valid guarantee (Theorem 3), the threshold-rule ARL corollary, the
Byzantine-voter corollary, and the long-horizon behaviour of the e-detector inside the Track S simulator.
Usage: python exps/t1_theory.py -> results/T/theory.json
"""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import sim, stats, runner

LAM = [5, 10, 20, 40, 80]


def weights(K, rng, s=1.0):
    a = rng.lognormal(0, s, K)
    return a, a.sum() ** 2 / (a ** 2).sum()


def null_votes(a, p_i, n_ep, rng, cc=None):
    """Epoch vote fractions. cc = (pi, gamma_idx, q) common cause: with prob pi per epoch, nodes in gamma_idx alarm w.p. q."""
    K = len(a)
    D = rng.random((n_ep, K)) < p_i
    if cc is not None:
        pi, gi, q = cc
        z = rng.random(n_ep) < pi
        D[np.ix_(z, gi)] = rng.random((z.sum(), len(gi))) < q
    return (D * a).sum(1) / a.sum()


def sr_alarm_count(v, p, keff, c):
    al = sim.sr_edetector(v, p, keff, LAM, c)
    return int(al.sum())


def main():
    t = time.time()
    rng = np.random.default_rng(2026)
    out = {"lambdas": LAM}
    # (a) ARL0 of the SR e-detector under the null (independent alarms, heterogeneous weights)
    arl = []
    for K in [25, 50, 100]:
        a, keff = weights(K, rng)
        p_i = np.clip(rng.uniform(0.05, 0.15, K), 0, 1)
        p = float((a * p_i).sum() / a.sum())
        for scen in ["independent", "common_cause"]:
            gi = np.argsort(-a)[:int(0.5 * K)]  # heaviest half
            cc = (0.01, gi, 0.9) if scen == "common_cause" else None
            n_ep, reps = 20000, 25
            for c in [20, 50, 100, 200, 500, 1000]:
                cnt = sum(sr_alarm_count(null_votes(a, p_i, n_ep, rng, cc), p, keff, c) for _ in range(reps))
                tot = n_ep * reps
                arl.append(dict(K=K, k_eff=float(keff), p=p, scenario=scen, c=c, alarms=cnt, epochs=tot,
                                arl0_hat=(tot / cnt) if cnt else None,
                                arl0_lower95=1.0 / stats.poisson_ucb(cnt, tot),
                                guarantee_holds=(cnt == 0) or (tot / cnt >= c)))
    out["edetector_arl0"] = arl
    # (b) threshold rule: per-epoch false-alarm probability and ARL0 vs the corollary bound exp(2 K_eff (theta-p)^2)
    thr = []
    for K in [25, 50, 100]:
        a, keff = weights(K, rng)
        p_i = np.full(K, 0.1)
        v = null_votes(a, p_i, 400000, rng)
        for th in [0.25, 0.3, 0.35, 0.4]:
            q = float((v >= th).mean())
            thr.append(dict(K=K, k_eff=float(keff), theta=th, p=0.1, per_epoch_fp=q,
                            arl0_hat=(1 / q) if q > 0 else None, arl0_bound=float(math.exp(2 * keff * (th - 0.1) ** 2))))
    out["threshold_arl0"] = thr
    # (c) detection delay after a change: fraction of weight starts alarming with prob 0.9
    dl = []
    for K in [50]:
        a, keff = weights(K, rng)
        p_i = np.full(K, 0.1)
        for gam in [0.3, 0.5, 0.7]:
            for c in [100, 1000]:
                ds = []
                for r in range(400):
                    pre = null_votes(a, p_i, 50, rng)
                    sel = rng.permutation(K)[:int(gam * K)]
                    pp = p_i.copy(); pp[sel] = 0.9
                    post = null_votes(a, pp, 100, rng)
                    al = sim.sr_edetector(np.concatenate([pre, post]), 0.1, keff, LAM, c)
                    hits = np.nonzero(al[50:])[0]
                    ds.append(int(hits[0]) if len(hits) else None)
                ok = [x for x in ds if x is not None]
                dl.append(dict(K=K, gamma=gam, c=c, detected_within_100=len(ok) / len(ds),
                               delay_median=float(np.median(ok)) if ok else None))
    out["edetector_delay"] = dl
    # (d) Byzantine-voter corollary: attackers hold weight fraction b_w and always vote
    byz = []
    for K in [50, 100]:
        a, _ = weights(K, rng)
        for bw in [0.0, 0.05, 0.1, 0.15]:
            order = rng.permutation(K)
            cum = np.cumsum(a[order]) / a.sum()
            bad = order[:int(np.searchsorted(cum, bw))] if bw > 0 else np.array([], int)
            bwr = float(a[bad].sum() / a.sum())
            hon = np.setdiff1d(np.arange(K), bad)
            kh = a[hon].sum() ** 2 / (a[hon] ** 2).sum()
            p, th = 0.1, 0.35
            D = rng.random((400000, K)) < p
            D[:, bad] = True
            emp = float(((D * a).sum(1) >= th * a.sum()).mean())
            dev = th - bwr - p * (1 - bwr)
            bound = float(math.exp(-2 * kh * dev ** 2 / (1 - bwr) ** 2)) if dev > 0 else 1.0
            byz.append(dict(K=K, byz_weight=bwr, k_eff_honest=float(kh), theta=th, p=p, empirical=emp, bound=bound,
                            violated=emp > bound))
    out["byzantine_corollary"] = byz
    # (e) long-horizon Track S runs: e-detector false alarms with outages, filtered vs raw votes
    cal = json.load(open(os.path.join(runner.RESULTS, "S", "edet_calibration.json")))
    cfg = dict(sim.DEFAULT, B=20000, t0=20000)
    lg = []
    for kind in ["raw", "filtered"]:
        for c in [20, 100, 1000]:
            fa, ep = 0, 0
            for s in range(5000, 5010):
                g = sim.generate(cfg, s, drift=False, outage=True)
                sg = sim.decision_signals(cfg, g, (0.35,))
                al = sg["raw_alarm"] if kind == "raw" else sg["raw_alarm"] & sg["adj_alarm"]
                v = sim.epoch_votes(al, sg["alpha"], cfg["W"], 2 * cfg["MB"])
                fa += sr_alarm_count(v, cal["p"][kind], sg["k_eff"], c)
                ep += len(v)
            lg.append(dict(kind=kind, c=c, false_alarms=fa, epochs=ep, arl0_hat=(ep / fa) if fa else None,
                           guarantee_holds=(fa == 0) or (ep / fa >= c)))
    out["tracks_long_horizon"] = lg
    out["runtime_s"] = round(time.time() - t, 1)
    os.makedirs(os.path.join(runner.RESULTS, "T"), exist_ok=True)
    json.dump(out, open(os.path.join(runner.RESULTS, "T", "theory.json"), "w"), indent=1)
    for k in ["edetector_arl0", "threshold_arl0", "edetector_delay", "byzantine_corollary", "tracks_long_horizon"]:
        print("==", k)
        for r in out[k]:
            print(r)


if __name__ == "__main__":
    main()
