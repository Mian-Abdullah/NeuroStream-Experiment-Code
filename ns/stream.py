"""Track A: real data streams (Elec2, Covertype, Airlines) partitioned across K nodes, with labelled injected events.

Injected (semi-synthetic, always reported as such):
  * common-cause outage: a fixed half of the nodes; outage starts with rate lam_o per block, lasts LO blocks, hits each
    group node w.p. 0.9; during an outage, a fixed random half of the (standardised) features of the node's records get
    an offset of +3 (sensor offset fault); a synthetic supply covariate Z = 1 - 0.5*outage + N(0, z_sd^2) is observed.
  * regional concept drift at block t0 on a fraction gamma of nodes: labels swapped (classes 0<->1) for records whose
    drift feature is above its warm-up median.
Natural drift in the streams is untouched and unlabelled.

Open loop: the model trained on the warm-up blocks is frozen and three paired variants are scored
(R0 natural, R1 +outages, R2 +outages+drift); differences between them isolate the injected effects.
Closed loop: the decision rule triggers federated retraining and prequential accuracy is measured.
"""
import math
import numpy as np
from . import data, fl, sim

DRIFT_FEATURE = {"elec": 0, "covtype": 0, "airlines": -2}
_CACHE = {}


def _load(name):
    if name not in _CACHE:
        _CACHE[name] = data.load(name)
    return _CACHE[name]


def setup(name, K, B, seed, warm_frac=0.1, t0_frac=0.67, lam_o=0.01, LO=10, hit_p=0.9, gamma=0.5, z_sd=0.03,
          frac_out=0.5, R0=30):
    X, y, group, meta = _load(name)
    ss = np.random.SeedSequence([int(seed), K, B])
    g = {n: np.random.default_rng(c) for n, c in zip(["part", "warm", "out", "drift", "z", "feat"], ss.spawn(6))}
    node = data.assign_nodes(name, y, group, K, g["part"])
    blk = data.blocks_of(len(y), B)
    Bw = int(warm_frac * B)
    n_prefix = int(np.searchsorted(blk, Bw))
    Xs = data.standardize_from_prefix(X, n_prefix)
    keep = node >= 0
    k = int(y.max() + 1)
    model = fl.Softmax(Xs.shape[1], k)
    # warm-up federated training (FedAvg, R0 rounds) on warm-up blocks
    w = np.zeros(model.size())
    wmask = keep & (blk < Bw)
    idx_by_node = [np.nonzero(wmask & (node == i))[0] for i in range(K)]
    nvol = np.array([len(ix) for ix in idx_by_node], float)
    for _ in range(R0):
        U, nn = [], []
        for i in range(K):
            ix = idx_by_node[i]
            if len(ix) == 0:
                continue
            U.append(fl.local_update(model, w, Xs[ix], y[ix]))
            nn.append(len(ix))
        w = w + (np.array(nn)[:, None] * np.array(U)).sum(0) / sum(nn)
    # injected events
    Go = np.zeros(K, bool)
    Go[g["out"].permutation(K)[:int(frac_out * K)]] = True
    hit = np.zeros((K, B), bool)
    for b in np.nonzero(g["out"].random(B) < lam_o)[0]:
        if b < Bw:
            continue
        h = Go & (g["out"].random(K) < hit_p)
        hit[h, b:b + LO] = True
    Gd = np.zeros(K, bool)
    Gd[g["drift"].permutation(K)[:int(round(gamma * K))]] = True
    t0 = int(t0_frac * B)
    Z = 1.0 - 0.5 * hit + g["z"].normal(0, z_sd, (K, B))
    feat = g["feat"].permutation(Xs.shape[1])[:max(1, Xs.shape[1] // 2)]
    df = DRIFT_FEATURE[name]
    med = np.median(Xs[:n_prefix, df])
    return dict(name=name, K=K, B=B, Bw=Bw, t0=t0, Xs=Xs, y=y, node=node, blk=blk, keep=keep, model=model, w=w,
                hit=hit, Gd=Gd, Go=Go, Z=Z, feat=feat, df=df, med=med, nvol=np.maximum(nvol, 1.0), k=k, LO=LO)


def variant_xy(S, rows, outage=True, drift=True):
    """Features and labels of the given rows under the injected events."""
    X = S["Xs"][rows].copy()
    yy = S["y"][rows].copy()
    nd, bk = S["node"][rows], S["blk"][rows]
    if outage:
        hm = S["hit"][nd, bk]
        if hm.any():
            X[np.ix_(hm, S["feat"])] += 3.0
    if drift:
        dm = S["Gd"][nd] & (bk >= S["t0"]) & (S["Xs"][rows, S["df"]] > S["med"]) & (yy <= 1)
        yy[dm] = 1 - yy[dm]
    return X, yy


def error_counts(S, w, outage, drift):
    rows = np.nonzero(S["keep"])[0]
    X, yy = variant_xy(S, rows, outage, drift)
    err = (S["model"].predict(w, X) != yy).astype(float)
    K, B = S["K"], S["B"]
    n = np.zeros((K, B))
    e = np.zeros((K, B))
    np.add.at(n, (S["node"][rows], S["blk"][rows]), 1.0)
    np.add.at(e, (S["node"][rows], S["blk"][rows]), err)
    return n, e, rows, err


def _cs(X):
    return np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X, 1)], 1)


def detector(n, e, Z, MB=10, dprime=0.002, start=0):
    """One-sided two-window test on pooled error rates (ADWIN-style harmonic-mean cut) and its covariate-adjusted
    version (count-weighted regression of block error rate on Z inside the window). Windows must start >= start."""
    K, B = n.shape
    cn, ce = _cs(n), _cs(e)
    cnz, cnzz, cez = _cs(n * Z), _cs(n * Z * Z), _cs(e * Z)
    b = np.arange(max(2 * MB - 1, start + 2 * MB - 1), B)
    raw = np.zeros((K, B), bool)
    adj = np.zeros((K, B), bool)
    if len(b) == 0:
        return raw, adj
    E, M, S0 = b + 1, b + 1 - MB, b + 1 - 2 * MB
    nN, nO = cn[:, E] - cn[:, M], cn[:, M] - cn[:, S0]
    eN, eO = ce[:, E] - ce[:, M], ce[:, M] - ce[:, S0]
    zN, zO = cnz[:, E] - cnz[:, M], cnz[:, M] - cnz[:, S0]
    zz, ez = cnzz[:, E] - cnzz[:, S0], cez[:, E] - cez[:, S0]
    ok = (nN > 0) & (nO > 0)
    nNs, nOs = np.maximum(nN, 1), np.maximum(nO, 1)
    mh = 1.0 / (1.0 / nNs + 1.0 / nOs)
    cut = np.sqrt(np.log(4 / dprime) / (2 * mh))
    d_raw = eN / nNs - eO / nOs
    ntot = nN + nO
    zbar = (zN + zO) / np.maximum(ntot, 1)
    ebar = (eN + eO) / np.maximum(ntot, 1)
    sxx = zz - ntot * zbar ** 2
    sxy = ez - ntot * zbar * ebar
    beta = sxy / (sxx + 1e-9)
    d_adj = d_raw - beta * (zN / nNs - zO / nOs)
    raw[:, b] = ok & (d_raw > cut)
    adj[:, b] = ok & (d_adj > cut)
    return raw, adj


def flags_from(raw, adj, alpha, thetas, W=10):
    d = sim.roll_any(raw, W)
    dfl = d & sim.roll_any(adj, W)
    A = alpha.sum()
    vr = (alpha[:, None] * d).sum(0) / A
    vf = (alpha[:, None] * dfl).sum(0) / A
    out = {}
    for th in thetas:
        out[f"FDC@{th:.2f}"] = vr >= th
        out[f"FDC+filter@{th:.2f}"] = vf >= th
    return out


def river_local(S, rows, err, kind):
    """Per-node river detector on the per-record error stream; returns alarm counts per (node, block)."""
    from river import drift
    K, B = S["K"], S["B"]
    mk = {"adwin": lambda: drift.ADWIN(delta=0.002), "ddm": lambda: drift.binary.DDM(),
          "ph": lambda: drift.PageHinkley()}[kind]
    det = [mk() for _ in range(K)]
    al = np.zeros((K, B), bool)
    nd, bk = S["node"][rows], S["blk"][rows]
    for i, b, x in zip(nd.tolist(), bk.tolist(), err.tolist()):
        if b < S["Bw"]:
            continue
        dd = det[i]
        dd.update(x)
        if dd.drift_detected:
            al[i, b] = True
    return al


def open_loop(name, K, B, seed, thetas, H=30, W=10, river_kinds=(), **kw):
    S = setup(name, K, B, seed, **kw)
    alpha = S["nvol"]
    res = {}
    for tag, (o, dr) in dict(R0=(False, False), R1=(True, False), R2=(True, True)).items():
        n, e, rows, err = error_counts(S, S["w"], o, dr)
        Zv = S["Z"] if o else S["Z"] + 0.5 * S["hit"]          # same covariate noise, no outage signal
        raw, adj = detector(n, e, Zv, start=S["Bw"])
        res[tag] = dict(raw=raw, adj=adj, flags=flags_from(raw, adj, alpha, thetas, W), n=n, e=e)
        for rk in river_kinds:
            res[tag][rk] = river_local(S, rows, err, rk)
    t0, Bw = S["t0"], S["Bw"]
    pre = slice(Bw + 2 * 10, t0)
    npre = pre.stop - pre.start
    out = dict(seed=int(seed), npre=npre, k_eff=float(alpha.sum() ** 2 / (alpha ** 2).sum()), methods={},
               mean_error_R0=float(res["R0"]["e"][:, Bw:].sum() / res["R0"]["n"][:, Bw:].sum()),
               outage_block_frac=float(S["hit"][S["Go"], Bw:t0].mean()))
    for m in res["R0"]["flags"]:
        f0, f1, f2 = (res[t]["flags"][m] for t in ("R0", "R1", "R2"))
        induced = f1[pre] & ~f0[pre]
        att = f2[t0:t0 + H] & ~f1[t0:t0 + H]
        out["methods"][m] = dict(false_episodes=len(sim.episodes(induced, W)),
                                 natural_episodes=len(sim.episodes(f0[pre], W)),
                                 detected=bool(att.any()), delay=int(np.argmax(att)) if att.any() else None)
    for loc in ["raw"] + list(river_kinds):
        a0, a1, a2 = res["R0"][loc], res["R1"][loc], res["R2"][loc]
        ind = sum(len(sim.episodes((a1[i, pre] & ~a0[i, pre]), 1)) for i in range(K))
        Gd = S["Gd"]
        det = (a2[Gd][:, t0:t0 + H] & ~a1[Gd][:, t0:t0 + H]).any(1).mean()
        out["methods"][f"local-{'twowindow' if loc == 'raw' else loc}"] = dict(
            wasted_node_retrains=int(ind), natural_node_episodes=int(sum(len(sim.episodes(a0[i, pre], 1)) for i in range(K))),
            node_detect_frac=float(det))
    return out


# ================================================================================================ closed loop
def _fedavg_round(model, w, parts, steps=5, lr=0.5):
    U, nn = [], []
    for X, y in parts:
        if len(y) == 0:
            continue
        U.append(fl.local_update(model, w, X, y, steps=steps, lr=lr))
        nn.append(len(y))
    if not U:
        return w
    return w + (np.array(nn)[:, None] * np.array(U)).sum(0) / sum(nn)


def _window_alarm(n, e, Z, b, reset, MB, dprime):
    """Raw and adjusted one-sided alarms at window end b for all nodes (windows must start at or after reset)."""
    K = n.shape[0]
    if b - 2 * MB + 1 < reset:
        return np.zeros(K, bool), np.zeros(K, bool)
    sl_n, sl_o = slice(b - MB + 1, b + 1), slice(b - 2 * MB + 1, b - MB + 1)
    nN, nO = n[:, sl_n].sum(1), n[:, sl_o].sum(1)
    eN, eO = e[:, sl_n].sum(1), e[:, sl_o].sum(1)
    zN, zO = (n[:, sl_n] * Z[:, sl_n]).sum(1), (n[:, sl_o] * Z[:, sl_o]).sum(1)
    sl = slice(b - 2 * MB + 1, b + 1)
    zz, ez = (n[:, sl] * Z[:, sl] ** 2).sum(1), (e[:, sl] * Z[:, sl]).sum(1)
    ok = (nN > 0) & (nO > 0)
    nNs, nOs = np.maximum(nN, 1), np.maximum(nO, 1)
    cut = np.sqrt(np.log(4 / dprime) / (2 / (1 / nNs + 1 / nOs)))
    d_raw = eN / nNs - eO / nOs
    nt = nN + nO
    zb, eb = (zN + zO) / np.maximum(nt, 1), (eN + eO) / np.maximum(nt, 1)
    beta = (ez - nt * zb * eb) / (zz - nt * zb ** 2 + 1e-9)
    d_adj = d_raw - beta * (zN / nNs - zO / nOs)
    return ok & (d_raw > cut), ok & (d_adj > cut)


def closed_loop(name, K, B, seed, policy, theta=0.35, R_tr=10, L=20, P=50, W=10, MB=10, dprime=0.002,
                fd_delta=0.06, fd_deltap=0.06, **kw):
    """Prequential accuracy and cost of a retraining policy on a real stream with injected outages and drift."""
    from scipy.cluster import hierarchy as sch
    from scipy.spatial.distance import squareform
    S = setup(name, K, B, seed, **kw)
    model, Bw, t0 = S["model"], S["Bw"], S["t0"]
    rows = np.nonzero(S["keep"])[0]
    Xv, yv = variant_xy(S, rows, True, True)
    nd, bk = S["node"][rows], S["blk"][rows]
    starts = np.searchsorted(bk, np.arange(B + 1))
    blkrows = [(starts[b], starts[b + 1]) for b in range(B)]
    alpha = S["nvol"]
    n = np.zeros((K, B)); e = np.zeros((K, B)); corr = np.zeros((K, B))
    raw_hist = np.zeros((K, B), bool); adj_hist = np.zeros((K, B), bool)
    w = S["w"].copy()
    Wn = np.repeat(w[None, :], K, 0)                      # per-node models (local / feddrift assignment)
    retrains, retrain_blocks, node_retrains, releases = 0, [], 0, np.zeros(K)
    reset, next_ok = Bw, Bw
    if policy == "local_adwin":
        from river import drift
        dets = [drift.ADWIN(delta=0.002) for _ in range(K)]
    if policy == "feddrift":
        models = [w.copy() for _ in range(K)]            # FedDrift-C 'F': every client starts on its own model
        assign = np.arange(K)
        prev_acc = np.full(K, np.nan)
        marked = {}
        fd_isolations, fd_isolations_pre = 0, 0

    def node_parts(b_lo, b_hi, i_sel=None):
        a, z = blkrows[max(b_lo, 0)][0], blkrows[b_hi][1]
        idx = np.arange(a, z)
        out = []
        for i in (range(K) if i_sel is None else i_sel):
            ii = idx[nd[a:z] == i]
            out.append((Xv[ii], yv[ii]))
        return out

    for b in range(Bw, B):
        a, z = blkrows[b]
        if z == a:
            continue
        Xb, yb, nb = Xv[a:z], yv[a:z], nd[a:z]
        if policy in ("local_adwin",):
            pred = np.empty(z - a, int)
            for i in np.unique(nb):
                m = nb == i
                pred[m] = model.predict(Wn[i], Xb[m])
        elif policy == "feddrift":
            pred = np.empty(z - a, int)
            for i in np.unique(nb):
                m = nb == i
                pred[m] = model.predict(models[assign[i]], Xb[m])
        else:
            pred = model.predict(w, Xb)
        err = (pred != yb).astype(float)
        np.add.at(n[:, b], nb, 1.0)
        np.add.at(e[:, b], nb, err)
        corr[:, b] = n[:, b] - e[:, b]
        if policy in ("fdc", "fdc_filter"):
            r_, a_ = _window_alarm(n, e, S["Z"], b, reset, MB, dprime)
            raw_hist[:, b], adj_hist[:, b] = r_, a_
            lo = max(b - W + 1, reset)
            d = raw_hist[:, lo:b + 1].any(1)
            if policy == "fdc_filter":
                d = d & adj_hist[:, lo:b + 1].any(1)
            if (alpha * d).sum() / alpha.sum() >= theta and b >= next_ok:
                trig = True
            else:
                trig = False
        elif policy == "periodic":
            trig = (b - Bw) % P == 0 and b > Bw
        elif policy == "oracle":
            trig = b == t0 + 5
        else:
            trig = False
        if trig:
            parts = node_parts(b - L + 1, b)
            for _ in range(R_tr):
                w = _fedavg_round(model, w, parts)
            retrains += 1
            retrain_blocks.append(b)
            releases += R_tr
            reset, next_ok = b + 1, b + W
        if policy == "local_adwin":
            for i, x in zip(nb.tolist(), err.tolist()):
                dets[i].update(x)
                if dets[i].drift_detected:
                    X_i, y_i = node_parts(b - L + 1, b, [i])[0]
                    for _ in range(R_tr):
                        Wn[i] = Wn[i] + fl.local_update(model, Wn[i], X_i, y_i)
                    node_retrains += 1
                    dets[i] = drift.ADWIN(delta=0.002)
        if policy == "feddrift":
            parts = node_parts(b, b)
            for c_, (until) in list(marked.items()):
                if until <= b:
                    del marked[c_]
            in_use = sorted(set(assign[[i for i in range(K) if i not in marked]].tolist()))
            accm = np.full((len(in_use), K), np.nan)
            for mi, m in enumerate(in_use):
                for i, (X_i, y_i) in enumerate(parts):
                    if len(y_i):
                        accm[mi, i] = (model.predict(models[m], X_i) == y_i).mean()
            for i in range(K):
                if i in marked or np.all(np.isnan(accm[:, i])):
                    continue
                bi = int(np.nanargmax(accm[:, i]))
                best, acc_b = in_use[bi], accm[bi, i]
                assign[i] = best
                if not np.isnan(prev_acc[i]) and prev_acc[i] - acc_b > fd_delta:
                    models.append(models[best].copy())
                    assign[i] = len(models) - 1
                    marked[i] = b + 1
                    fd_isolations += 1
                    fd_isolations_pre += int(b < t0)
                prev_acc[i] = acc_b
            # hierarchical merging among models in use (complete linkage, distance A, threshold delta')
            in_use = sorted(set(assign[[i for i in range(K) if i not in marked]].tolist()))
            if len(in_use) > 1:
                hist = node_parts(b - L + 1, b)
                cdata = {}
                for m in in_use:
                    Xs_ = [hist[i][0] for i in range(K) if assign[i] == m and i not in marked]
                    ys_ = [hist[i][1] for i in range(K) if assign[i] == m and i not in marked]
                    cdata[m] = (np.concatenate(Xs_), np.concatenate(ys_)) if Xs_ else (None, None)
                M = len(in_use)
                ca = np.zeros((M, M))
                for p_, mp in enumerate(in_use):
                    for q_, mq in enumerate(in_use):
                        Xq, yq = cdata[mq]
                        ca[p_, q_] = (model.predict(models[mp], Xq) == yq).mean() if Xq is not None and len(yq) else 0.0
                D = np.maximum(np.maximum(np.diag(ca)[:, None] - ca, np.diag(ca)[None, :] - ca.T), 0.0)
                np.fill_diagonal(D, 0.0)
                T = sch.fcluster(sch.linkage(squareform(D, checks=False), method="complete"), t=fd_deltap, criterion="distance")
                for cl in np.unique(T):
                    grp = [in_use[j] for j in np.nonzero(T == cl)[0]]
                    if len(grp) > 1:
                        wts = np.array([max(1, sum(len(hist[i][1]) for i in range(K) if assign[i] == m_)) for m_ in grp], float)
                        merged = (wts[:, None] * np.array([models[m_] for m_ in grp])).sum(0) / wts.sum()
                        models[grp[0]] = merged
                        for m_ in grp[1:]:
                            assign[assign == m_] = grp[0]
            # one FedAvg round per model on its clients' current block (continuous training)
            for m in sorted(set(assign.tolist())):
                cl = [i for i in range(K) if assign[i] == m]
                models[m] = _fedavg_round(model, models[m], [parts[i] for i in cl])
            releases += 1
    post = slice(Bw, B)
    acc = lambda sl: float(corr[:, sl].sum() / max(n[:, sl].sum(), 1))
    hit_any = S["hit"].any(0)
    out = dict(seed=int(seed), policy=policy, acc_all=acc(post), acc_pre=acc(slice(Bw, t0)), acc_post=acc(slice(t0, B)),
               acc_adapt=acc(slice(t0, min(B, t0 + 50))), retrains=retrains,
               retrains_pre_t0=int(sum(rb < t0 for rb in retrain_blocks)),
               retrains_during_outage=int(sum(bool(hit_any[rb]) for rb in retrain_blocks)),
               node_retrains=node_retrains, releases_per_node=float(releases.mean()), t0=t0, Bw=Bw)
    if policy == "feddrift":
        out.update(fd_isolations=fd_isolations, fd_isolations_pre_t0=fd_isolations_pre, fd_models_final=len(set(assign.tolist())))
    return out
