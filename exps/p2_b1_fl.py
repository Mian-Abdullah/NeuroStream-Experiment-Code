"""(P2) Utility of DP federated training vs noise multiplier, and (B1) robust aggregation under adaptive attacks,
on real data (Covertype, area partition, K=50; Elec2 Dirichlet partition, K=10 for P2).
Training data: first 30% of the stream; test data: the next 10%.  10 seeds.
Usage: python exps/p2_b1_fl.py [p2|b1]  -> results/P/dp_utility.json, results/B/byzantine.json
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import data, fl, stats, runner, accounting as A

_C = {}


def prep(name, K, seed, n_per_node=None):
    if name not in _C:
        _C[name] = data.load(name)
    X, y, group, meta = _C[name]
    rng = np.random.default_rng([seed, 7])
    n = len(y)
    tr, te = int(0.3 * n), int(0.4 * n)
    Xs = data.standardize_from_prefix(X, tr)
    node = data.assign_nodes(name, y, group, K, rng)
    parts = []
    for i in range(K):
        ix = np.nonzero(node[:tr] == i)[0]
        if n_per_node and len(ix) > n_per_node:
            ix = rng.choice(ix, n_per_node, replace=False)
        parts.append((Xs[ix], y[ix]))
    test = (Xs[tr:te], y[tr:te])
    root = rng.choice(tr, 100, replace=False)
    return parts, test, (Xs[root], y[root]), fl.Softmax(Xs.shape[1], int(y.max() + 1)), rng


def dp_run(a):
    """mode: 'noclip' (non-private reference, lr 1.0), 'clip' (clipping only, sigma=0), 'dp' (clipping + noise)."""
    name, K, seed, mode, sigma, lr, rounds = a
    parts, (Xt, yt), _, model, rng = prep(name, K, seed)
    w = np.zeros(model.size())
    for _ in range(rounds):
        U, nn = [], []
        for X, y in parts:
            if len(y) == 0:
                continue
            dp = None if mode == "noclip" else (1.0, sigma)
            U.append(fl.local_update(model, w, X, y, steps=1, lr=lr, dp=dp, rng=rng))
            nn.append(len(y))
        w = w + (np.array(nn)[:, None] * np.array(U)).sum(0) / sum(nn)
    return dict(seed=seed, mode=mode, sigma=sigma, lr=lr, acc=float((model.predict(w, Xt) == yt).mean()))


def byz_run(a):
    name, K, seed, aggn, attk, frac, rounds = a
    parts, (Xt, yt), (Xr, yr), model, rng = prep(name, K, seed, n_per_node=500)
    # IID re-partition of the same training records: under the wilderness-area partition, unweighted robust
    # aggregation of a linear model stays near the majority-class rate (results/B/byzantine_area_partition.json)
    Xa = np.concatenate([p_[0] for p_ in parts]); ya = np.concatenate([p_[1] for p_ in parts])
    parts = [(Xa[c], ya[c]) for c in np.array_split(rng.permutation(len(ya)), K)]
    nb = int(round(frac * K))
    bad = set(rng.permutation(K)[:nb].tolist())
    w = np.zeros(model.size())
    prev = None
    for _ in range(rounds):
        U = np.array([fl.local_update(model, w, X, y, steps=5, lr=0.5) for X, y in parts])
        good = np.array([U[i] for i in range(K) if i not in bad])
        if nb:
            M = fl.attack(good, nb, attk)
            U = np.concatenate([good, M])
        kw = dict(f=nb, beta=0.1, prev=prev, tau=float(np.median(np.linalg.norm(U, axis=1))))
        if aggn == "fltrust":
            kw["root_update"] = fl.local_update(model, w, Xr, yr, steps=5, lr=0.5)
        if aggn == "trimmed20":
            g = fl.AGG["trimmed"](U, beta=0.2)
        else:
            g = fl.AGG[aggn](U, **kw)
        prev = g
        w = w + g
    return dict(seed=seed, agg=aggn, attack=attk, frac=frac, acc=float((model.predict(w, Xt) == yt).mean()))


def p2():
    t = time.time()
    DS = [("covtype", 50), ("elec", 10)]
    # learning rate for clipped training, chosen on validation seed 100 (clip only)
    lrs = {}
    with Pool(2) as p:
        for name, K in DS:
            cand = [1.0, 5.0, 20.0, 50.0]
            res = p.map(dp_run, [(name, K, 100, "clip", 0.0, lr, 60) for lr in cand])
            lrs[name] = cand[int(np.argmax([r["acc"] for r in res]))]
        jobs = []
        for name, K in DS:
            for s_ in range(10):
                jobs.append((name, K, s_, "noclip", 0.0, 1.0, 60))
                jobs.append((name, K, s_, "clip", 0.0, lrs[name], 60))
                for sig in [1.0, 10.0, 25.0]:
                    jobs.append((name, K, s_, "dp", sig, lrs[name], 60))
        rows = p.map(dp_run, jobs)
    summ = {}
    for (name, K) in DS:
        for mode, sig in [("noclip", 0.0), ("clip", 0.0), ("dp", 1.0), ("dp", 10.0), ("dp", 25.0)]:
            acc = [r["acc"] for r, j in zip(rows, jobs) if j[0] == name and j[3] == mode and j[4] == sig]
            eps = A.eps_of(A.rdp_total({sig: 60}), 1e-5) if mode == "dp" else None
            summ[f"{name}_K{K}_{mode}_sigma{sig:g}"] = dict(acc=stats.summarize_rate(acc), eps_after_60_releases=eps)
    out = dict(protocol="per-example clipping C=1 and Gaussian noise sigma*C on the summed gradient, one full-batch step per "
                        "round, 60 rounds, no subsampling; 'noclip' = non-private reference (lr 1.0); 'clip' = clipping only; "
                        "learning rate for clipped runs chosen on validation seed 100 from {1,5,20,50}; record-level eps per "
                        "node at delta=1e-5; test = next 10% of the stream; 10 seeds", lr_clipped=lrs,
               summary=summ, per_run=rows, versions=runner.versions(), runtime_s=round(time.time() - t, 1))
    os.makedirs(os.path.join(runner.RESULTS, "P"), exist_ok=True)
    json.dump(out, open(os.path.join(runner.RESULTS, "P", "dp_utility.json"), "w"), indent=1)
    print("lr", lrs)
    for k, v in summ.items():
        print(k, round(v["acc"]["mean"], 4), [round(x, 4) for x in v["acc"]["ci95"]], v["eps_after_60_releases"])


def b1():
    AGGS = ["mean", "median", "trimmed", "trimmed20", "krum", "multikrum", "cclip", "fltrust"]
    ATT = ["signflip", "alie", "ipm", "minmax"]
    jobs = []
    for s in range(10):
        for ag in AGGS:
            jobs.append(("covtype", 50, s, ag, "none", 0.0, 40))
            for at in ATT:
                for fr in [0.1, 0.2, 0.3]:
                    jobs.append(("covtype", 50, s, ag, at, fr, 40))
    t = time.time()
    with Pool(2) as p:
        rows = p.map(byz_run, jobs, chunksize=8)
    summ = {}
    for ag in AGGS:
        base = [r["acc"] for r in rows if r["agg"] == ag and r["attack"] == "none"]
        summ[f"{ag}|none|0.0"] = stats.summarize_rate(base)
        for at in ATT:
            for fr in [0.1, 0.2, 0.3]:
                summ[f"{ag}|{at}|{fr}"] = stats.summarize_rate([r["acc"] for r in rows if r["agg"] == ag and r["attack"] == at and r["frac"] == fr])
    out = dict(protocol="Covertype, K=50, the same 500 records/node re-partitioned IID, 40 rounds x 5 local GD steps; "
                        "omniscient attackers replace their updates; krum/multikrum told f; FLTrust root set = 100 records; "
                        "centered clipping tau = median norm of all received updates", summary=summ, per_run=rows,
               versions=runner.versions(), runtime_s=round(time.time() - t, 1))
    os.makedirs(os.path.join(runner.RESULTS, "B"), exist_ok=True)
    json.dump(out, open(os.path.join(runner.RESULTS, "B", "byzantine.json"), "w"), indent=1)
    for ag in AGGS:
        print(ag, "none %.3f" % summ[f"{ag}|none|0.0"]["mean"], " ".join(
            f"{at}@{fr}:{summ[f'{ag}|{at}|{fr}']['mean']:.3f}" for at in ATT for fr in [0.1, 0.3]))


if __name__ == "__main__":
    {"p2": p2, "b1": b1}[sys.argv[1]]()
