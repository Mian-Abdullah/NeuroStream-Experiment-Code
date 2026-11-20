"""Experiment D: Byzantine robustness of the FCP aggregator (trimmed mean, beta=0.1) under sign-flip attack,
with and without DP noise.  Synthetic federated logistic regression (non-IID label skew), K=50 nodes.
"""
import json, numpy as np
K, D_, N_LOC, ROUNDS, LR, LOCAL = 50, 20, 200, 60, 0.5, 5
CLIP = 1.0

def make(rng):
    w_true = rng.normal(size=D_)
    data = []
    for i in range(K):
        shift = rng.normal(0, 0.5, D_)              # feature shift => heterogeneity
        X = rng.normal(size=(N_LOC, D_)) + shift
        p = 1 / (1 + np.exp(-(X @ w_true)))
        y = (rng.random(N_LOC) < p).astype(float)
        data.append((X, y))
    Xt = rng.normal(size=(5000, D_)); yt = (rng.random(5000) < 1 / (1 + np.exp(-(Xt @ w_true)))).astype(float)
    return data, (Xt, yt)

def local_update(w, X, y):
    w0 = w.copy()
    for _ in range(LOCAL):
        g = X.T @ (1 / (1 + np.exp(-(X @ w))) - y) / len(y)
        w = w - LR * g
    return w - w0

def agg(U, kind, beta=0.1):
    if kind == "mean": return U.mean(0)
    if kind == "median": return np.median(U, 0)
    s = np.sort(U, 0); t = int(np.floor(beta * len(U)))
    return s[t:len(U) - t].mean(0)

def run(seed, n_bad, kind, noise_sd):
    rng = np.random.default_rng(seed)
    data, (Xt, yt) = make(rng)
    bad = set(rng.permutation(K)[:n_bad].tolist())
    w = np.zeros(D_)
    for _ in range(ROUNDS):
        U = np.stack([local_update(w, *data[i]) for i in range(K)])
        nrm = np.linalg.norm(U, axis=1, keepdims=True); U = U * np.minimum(1, CLIP / (nrm + 1e-12))   # clip
        for i in bad: U[i] = -3.0 * U[i] * 1.0                                                          # sign-flip x3 (after clipping => stronger than clip allows)
        U = U + rng.normal(0, noise_sd, U.shape)                                                          # local Gaussian noise
        w = w + agg(U, kind)
    pred = (1 / (1 + np.exp(-(Xt @ w))) > 0.5).astype(float)
    return float((pred == yt).mean())

out = {"K": K, "rounds": ROUNDS, "attack": "sign-flip x3 applied after clipping", "rows": []}
SEEDS = 10
print("n_bad  frac | " + " | ".join(f"{k}/sd{s}" for s in (0.0, 0.05) for k in ("mean", "trim0.1", "median")))
for n_bad in [0, 2, 5, 7, 10, 15]:
    row = {"n_bad": n_bad, "frac": n_bad / K}
    for sd in (0.0, 0.05):
        for kind in ("mean", "trimmed", "median"):
            accs = [run(s, n_bad, kind, sd) for s in range(SEEDS)]
            row[f"{kind}_sd{sd}"] = (float(np.mean(accs)), float(np.std(accs)))
    out["rows"].append(row)
    print(f"{n_bad:5d} {n_bad/K:5.2f} | " + " | ".join(f"{row[f'{k}_sd{s}'][0]:.3f}" for s in (0.0, 0.05) for k in ("mean", "trimmed", "median")))
json.dump(out, open("results/exp_d.json", "w"), indent=1)
