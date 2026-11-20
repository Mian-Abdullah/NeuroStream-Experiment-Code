"""Track S: controlled simulation of the federated decision layer with known ground truth.

Corrected and extended version of the earlier simulator in exps/legacy/exp_c_sim.py:
  * independent RNG streams per event type (SeedSequence.spawn) -> exact paired counterfactual runs;
  * attributable detection = a declaration in [t0, t0+horizon) that is absent from the paired no-drift run;
  * covariate realism knobs: covariate noise, partial observability (hidden confounder at 0), non-linear effect,
    compromised shared covariate feed;
  * Algorithm 1 as written (LOCAL scope by persistence, weights from data volume x calibrated reliability that
    never uses ground truth);
  * vectorised covariate-adjusted test, optional private (Gaussian-noised) sufficient statistics;
  * Sparse-Vector gate that follows Lyu-Su-Li Alg. 1 (threshold noise redrawn only when an instance halts);
  * Byzantine voters (flooding / suppression);
  * e-detector (Shiryaev-Roberts with Hoeffding e-values) on disjoint epochs.
"""
import math
import numpy as np

DEFAULT = dict(K=50, B=1200, t0=600, C=3, NB=5000, MB=10, SX=0.25, MAG=0.03, W=10, LF=20, LO=20,
               lam_f=0.001, lam_o=0.004, frac_out=0.5, hit_p=0.9, frac_drift=0.5, drift_mag=1.0,
               dprime=0.002, z_sd=0.03, z_obs=1.0, nonlinear=False, spoof_feed=False,
               weights="volume", r_local=3, calib_end=300, horizon=60, zmax=1.5)

STREAMS = ["weights", "faults", "outages", "drift", "noise", "zcov", "gate", "byz", "gate2"]


def cut_value(m, dprime):
    return math.sqrt(math.log(4 / dprime) / (2 * m))


def rng_streams(seed):
    ss = np.random.SeedSequence(int(seed))
    return {n: np.random.default_rng(c) for n, c in zip(STREAMS, ss.spawn(len(STREAMS)))}


# ----------------------------------------------------------------------------------------------- generator
def generate(cfg, seed, drift=True, outage=True):
    c = cfg
    g = rng_streams(seed)
    K, B, C, LF, LO, MAG = c["K"], c["B"], c["C"], c["LF"], c["LO"], c["MAG"]
    bstd = c["SX"] / math.sqrt(c["NB"])
    sh = np.zeros((K, B, C))
    # local sensor faults (always drawn)
    fpos = np.argwhere(g["faults"].random((K, B)) < c["lam_f"])
    fch = g["faults"].integers(C, size=len(fpos))
    for (i, b), ch in zip(fpos, fch):
        sh[i, b:b + LF, ch] += MAG
    # common-cause outages (always drawn, applied only if outage=True)
    Go = np.zeros(K, bool)
    Go[g["outages"].permutation(K)[:int(c["frac_out"] * K)]] = True
    starts = np.nonzero(g["outages"].random(B) < c["lam_o"])[0]
    hit = np.zeros((K, B), bool)
    sev = np.zeros((K, B))
    for b in starts:
        h = Go & (g["outages"].random(K) < c["hit_p"])
        s = g["outages"].random(K)
        hit[h, b:b + LO] = True
        sev[h, b:b + LO] = np.maximum(sev[h, b:b + LO], s[h, None])
    if not outage:
        hit[:] = False
        sev[:] = 0.0
    if c["nonlinear"]:
        eff = hit & (sev > 0.5)               # monitored shift only for severe sags
        zsig = sev * hit                      # covariate tracks severity linearly
    else:
        eff = hit
        zsig = hit.astype(float)
    sh[:, :, 1] += MAG * eff
    # genuine drift (group always drawn)
    Gd = np.zeros(K, bool)
    Gd[g["drift"].permutation(K)[:int(round(c["frac_drift"] * K))]] = True
    if drift:
        sh[Gd, c["t0"]:, 0] += MAG * c["drift_mag"]
    # observation noise
    Y = 0.5 + g["noise"].normal(0, 0.05, (K, 1, C)) + sh + g["noise"].normal(0, bstd, (K, B, C))
    # covariate (supply voltage); partial observability and spoofed shared feed
    obs = g["zcov"].random(K) < c["z_obs"]
    Z = 1.0 - 0.5 * zsig * obs[:, None] + g["zcov"].normal(0, c["z_sd"], (K, B))
    if c["spoof_feed"] and drift:
        Z[:, c["t0"]:c["t0"] + c["horizon"]] -= 0.5   # compromised shared feed reports an outage during the drift
    Z = np.clip(Z, 0.0, c["zmax"])
    n_i = g["weights"].lognormal(0, 0.5, K)
    rho_rand = g["weights"].uniform(0.6, 1.0, K)
    return dict(Y=Y, Z=Z, n=n_i, rho_rand=rho_rand, Gd=Gd, Go=Go, hit=hit, g=g)


# ----------------------------------------------------------------------------------------------- detector
def _cs(X):
    return np.concatenate([np.zeros((X.shape[0], 1) + X.shape[2:]), np.cumsum(X, 1)], 1)


def window_stats(Y, Z, MB, private_sd=0.0, rng=None, sxx_floor=0.0):
    """Raw and covariate-adjusted two-window mean differences for every node and window end.
    Returns arrays of shape (K, B) (max over channels of |difference|), NaN before the first full window.
    If private_sd > 0, Gaussian noise of that standard deviation is added to the private sums
    (sum_new Y, sum_old Y, sum Z*Y per channel) before the adjusted statistic is formed."""
    K, B, C = Y.shape
    cY, cZ, cZZ, cZY = _cs(Y), _cs(Z), _cs(Z * Z), _cs(Z[:, :, None] * Y)
    b = np.arange(2 * MB - 1, B)
    e, m, s = b + 1, b + 1 - MB, b + 1 - 2 * MB
    n = 2 * MB
    yN, yO = cY[:, e] - cY[:, m], cY[:, m] - cY[:, s]
    zN, zO = cZ[:, e] - cZ[:, m], cZ[:, m] - cZ[:, s]
    zz, zy = cZZ[:, e] - cZZ[:, s], cZY[:, e] - cZY[:, s]
    raw = np.full((K, B), np.nan)
    raw[:, b] = np.abs((yN - yO) / MB).max(2)
    if private_sd > 0:
        yN = yN + rng.normal(0, private_sd, yN.shape)
        yO = yO + rng.normal(0, private_sd, yO.shape)
        zy = zy + rng.normal(0, private_sd, zy.shape)
    zbar = (zN + zO) / n
    ybar = (yN + yO) / n
    sxx = zz - n * zbar ** 2
    sxy = zy - n * zbar[:, :, None] * ybar
    beta = sxy / (np.maximum(sxx, sxx_floor)[:, :, None] + 1e-12)   # variance floor stabilises noisy slopes
    adjd = (yN - yO) / MB - beta * ((zN - zO) / MB)[:, :, None]
    adj = np.full((K, B), np.nan)
    adj[:, b] = np.abs(adjd).max(2)
    return raw, adj


def roll_any(a, w):
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, 1)], 1)
    idx = np.arange(a.shape[1])
    lo = np.maximum(idx - w + 1, 0)
    return (cs[:, idx + 1] - cs[:, lo]) > 0


def roll_all(a, w):
    return ~roll_any(~a, w)


def episodes(flag, refractory):
    ep, last = [], -10 ** 9
    for b in np.nonzero(flag)[0]:
        if b - last > refractory:
            ep.append(int(b))
        last = b
    return ep


# ----------------------------------------------------------------------------------------------- SVT gate
def svt_gate(diff, cut, W, eps_svt, c_svt, NB_m, rng):
    """Sparse-Vector gate (Lyu, Su & Li 2017, Alg. 1): eps1 = eps2 = eps_svt/2, threshold noise Lap(D/eps1) drawn
    once per instance, query noise Lap(2cD/eps2), instance halts after c positives and a fresh instance starts.
    D = 1/m (sensitivity of a two-window mean difference of [0,1]-bounded records).
    A positive answer puts the node in the alarm state for W blocks; the gate is not queried meanwhile.
    Returns (alarm_state (K,B) bool, positives per node, instances per node)."""
    K, B = diff.shape
    D = 1.0 / NB_m
    e1 = e2 = eps_svt / 2
    s_thr, s_q = D / e1, 2 * c_svt * D / e2
    thr = rng.laplace(0, s_thr, K)
    count = np.zeros(K, int)
    instances = np.ones(K, int)
    positives = np.zeros(K, int)
    until = np.full(K, -1)
    state = np.zeros((K, B), bool)
    start = int(np.argmax(~np.isnan(diff[0])))
    for b in range(start, B):
        q = diff[:, b] + rng.laplace(0, s_q, K)
        ask = b >= until
        pos = ask & (q >= cut + thr)
        if pos.any():
            until[pos] = b + W
            positives[pos] += 1
            count[pos] += 1
            done = pos & (count >= c_svt)
            if done.any():
                thr[done] = rng.laplace(0, s_thr, done.sum())
                count[done] = 0
                instances[done] += 1
        state[:, b] = b < until
    return state, positives, instances


# ----------------------------------------------------------------------------------------------- e-detector
def sr_edetector(vote_frac, p, k_eff, lambdas, c_thr):
    """Shiryaev-Roberts e-detector on a sequence of epoch vote fractions.
    e_t = mean_j exp(lambda_j (v_t - p) - lambda_j^2 / (8 K_eff)); R_t = (1 + R_{t-1}) e_t; alarm when R_t >= c_thr,
    then reset R to 0. Returns boolean alarm array over epochs."""
    lam = np.asarray(lambdas, float)
    e = np.exp(np.outer(vote_frac - p, lam) - lam ** 2 / (8 * k_eff)).mean(1)
    R, out = 0.0, np.zeros(len(vote_frac), bool)
    for t, et in enumerate(e):
        R = (1.0 + R) * et
        if R >= c_thr:
            out[t] = True
            R = 0.0
    return out


# ----------------------------------------------------------------------------------------------- one run
def weights_of(gen, cfg, raw_alarm, adj_alarm):
    w = cfg["weights"]
    n = gen["n"]
    if w == "equal":
        return np.ones_like(n)
    if w == "volume":
        return n
    if w == "random_rho":                       # what exp_c_sim.py did
        return n * gen["rho_rand"]
    if w == "volume_x_calibrated":
        # reliability = share of the node's calibration-period alarm episodes that survive covariate adjustment
        lo, hi = 2 * cfg["MB"], cfg["calib_end"]
        rho = np.empty(len(n))
        for i in range(len(n)):
            ep = episodes(raw_alarm[i, lo:hi], 1)
            kept = sum(bool(adj_alarm[i, lo + b: lo + b + cfg["W"]].any()) for b in ep)
            rho[i] = (1 + kept) / (2 + len(ep))
        return n * rho
    raise ValueError(w)


def decision_signals(cfg, gen, thetas, private_sd=0.0, svt=None, byz=None):
    """Compute per-block global flags for every method from one generated run."""
    c = cfg
    W, MB = c["W"], c["MB"]
    CUT = cut_value(c["NB"] * MB, c["dprime"])
    floor = private_sd * 3 * math.sqrt(2.0 / MB) * c["z_sd"] / (0.1 * cut_value(c["NB"] * MB, c["dprime"])) if private_sd > 0 else 0.0
    raw, adj = window_stats(gen["Y"], gen["Z"], MB, private_sd, gen["g"]["zcov"] if private_sd > 0 else None, floor)
    raw_alarm = np.nan_to_num(raw, nan=0.0) > CUT
    adj_alarm = np.nan_to_num(adj, nan=0.0) > CUT
    adj_recent = roll_any(adj_alarm, W)
    if svt is not None:
        # private votes: each node runs its own Sparse-Vector gate; the plain-FDC baseline gates the raw statistic,
        # the filtered system gates the locally covariate-adjusted statistic (one release stream per deployment)
        d, pos, inst = svt_gate(np.nan_to_num(raw, nan=0.0), CUT, W, svt[0], svt[1], c["NB"] * MB, gen["g"]["gate"])
        df, pos_f, inst_f = svt_gate(np.nan_to_num(adj, nan=0.0), CUT, W, svt[0], svt[1], c["NB"] * MB, gen["g"]["gate2"])
    else:
        d = roll_any(raw_alarm, W)
        df = d & adj_recent
        pos = inst = pos_f = inst_f = None
    alpha = weights_of(gen, c, raw_alarm, adj_alarm)
    K = len(alpha)
    if byz is not None:                          # Byzantine voters
        mode, frac = byz
        bad = np.zeros(K, bool)
        bad[gen["g"]["byz"].permutation(K)[:int(round(frac * K))]] = True
        if mode == "flood":
            d = d | bad[:, None]
            df = df | bad[:, None]
        elif mode == "suppress":
            d = d & ~bad[:, None]
            df = df & ~bad[:, None]
    A = alpha.sum()
    v_raw = (alpha[:, None] * d).sum(0) / A
    v_f = (alpha[:, None] * df).sum(0) / A
    flags = {}
    for th in thetas:
        flags[f"FDC@{th:.2f}"] = v_raw >= th
        flags[f"FDC+filter@{th:.2f}"] = v_f >= th
    # LOCAL scope of Algorithm 1: alarm persisting over r consecutive decision windows, no global declaration
    persist = roll_all(roll_any(raw_alarm, W), c["r_local"] * W)
    persist_f = roll_all(roll_any(raw_alarm & adj_alarm, W), c["r_local"] * W)
    # releases of the private filter: window ends consulted by some decision (theta = min of thetas)
    th0 = min(thetas)
    consult = d & (v_raw >= th0)[None, :]
    consulted_ends = np.zeros_like(consult)
    for k in range(W):
        consulted_ends[:, :consult.shape[1] - k] |= consult[:, k:]
    keff = A ** 2 / (alpha ** 2).sum()
    return dict(flags=flags, raw_alarm=raw_alarm, adj_alarm=adj_alarm, d=d, df=df, alpha=alpha, k_eff=keff,
                v_raw=v_raw, v_f=v_f, persist=persist, persist_f=persist_f, filter_releases=consulted_ends.sum(1),
                svt_pos=pos, svt_inst=inst, svt_pos_f=pos_f, svt_inst_f=inst_f, CUT=CUT)


def epoch_votes(alarm, alpha, W, start):
    """Disjoint epochs [start + jW, start + (j+1)W): node votes if it alarmed in the epoch."""
    K, B = alarm.shape
    nE = (B - start) // W
    a = alarm[:, start:start + nE * W].reshape(K, nE, W).any(2)
    return (alpha[:, None] * a).sum(0) / alpha.sum()


def run_seed(cfg, seed, thetas=(0.35, 0.45, 0.55), scenario="outage_drift", private_sd=0.0, svt=None, byz=None,
             edet=None):
    """Paired runs for one seed. scenario in {outage_drift, faults_drift, outage_nodrift}.
    Returns per-seed metrics per method."""
    c = cfg
    W, MB, t0, H = c["W"], c["MB"], c["t0"], c["horizon"]
    outage = scenario != "faults_drift"
    has_drift = scenario != "outage_nodrift"
    g_cf = generate(c, seed, drift=False, outage=outage)
    s_cf = decision_signals(c, g_cf, thetas, private_sd, svt, byz)
    if has_drift:
        g_d = generate(c, seed, drift=True, outage=outage)
        s_d = decision_signals(c, g_d, thetas, private_sd, svt, byz)
    pre = slice(2 * MB, t0 if has_drift else c["B"])
    npre = pre.stop - pre.start
    out = dict(seed=int(seed), k_eff=float(s_cf["k_eff"]), npre=npre, methods={})
    out["p_hat_raw"] = float(s_cf["d"][:, pre].mean())
    out["p_hat_filtered"] = float(s_cf["df"][:, pre].mean())
    # node-level baselines: local retrain-on-alarm, and filtered local retrain (filter without consensus)
    for name, al in [("local", s_cf["raw_alarm"]), ("local+filter", s_cf["raw_alarm"] & roll_any(s_cf["adj_alarm"], W))]:
        nep = sum(len(episodes(al[i, pre], 1)) for i in range(al.shape[0]))
        r = dict(false_episodes=None, wasted_node_retrains=nep)
        if has_drift:
            ald = s_d["raw_alarm"] if name == "local" else s_d["raw_alarm"] & roll_any(s_d["adj_alarm"], W)
            Gd = g_d["Gd"]
            win = ald[Gd][:, t0:t0 + H] & ~(al[Gd][:, t0:t0 + H])
            r["node_detect_frac"] = float(win.any(1).mean()) if Gd.any() else float("nan")
        out["methods"][name] = r
    # global methods
    for name, fl in s_cf["flags"].items():
        ep = episodes(fl[pre], W)
        r = dict(false_episodes=len(ep), wasted_node_retrains=len(ep) * c["K"])
        if has_drift:
            fd = s_d["flags"][name]
            att = fd[t0:t0 + H] & ~fl[t0:t0 + H]
            r["detected"] = bool(att.any())
            r["delay"] = int(np.argmax(att)) if att.any() else None
            r["detected_raw"] = bool(fd[t0:t0 + H].any())
        out["methods"][name] = r
    # LOCAL scope counts (Algorithm 1 line 6), pre-drift, only when no global declaration at that block
    loc = s_cf["persist"][:, pre] & ~s_cf["flags"][f"FDC@{min(thetas):.2f}"][pre][None, :]
    out["local_scope_episodes"] = int(sum(len(episodes(loc[i], c["r_local"] * W)) for i in range(loc.shape[0])))
    locf = s_cf["persist_f"][:, pre] & ~s_cf["flags"][f"FDC+filter@{min(thetas):.2f}"][pre][None, :]
    out["local_scope_episodes_filtered"] = int(sum(len(episodes(locf[i], c["r_local"] * W)) for i in range(locf.shape[0])))
    out["filter_releases_mean"] = float(s_cf["filter_releases"].mean())
    if svt is not None:
        out["svt_active_frac"] = float(s_cf["d"][:, pre].mean())
        out["svt_positives_per_node_per_1000"] = float(s_cf["svt_pos"].mean() / c["B"] * 1000)
        out["svt_instances_mean"] = float(s_cf["svt_inst"].mean())
        out["svt_active_frac_filtered"] = float(s_cf["df"][:, pre].mean())
        out["svt_positives_filtered_per_node_per_1000"] = float(s_cf["svt_pos_f"].mean() / c["B"] * 1000)
        out["svt_instances_filtered_mean"] = float(s_cf["svt_inst_f"].mean())
    if edet is not None:                         # anytime-valid e-detector on disjoint epochs
        start = 2 * MB
        for kind, key in [("raw", "d_alarm"), ("filtered", "df_alarm")]:
            alarm_cf = s_cf["raw_alarm"] if kind == "raw" else (s_cf["raw_alarm"] & s_cf["adj_alarm"])
            v = epoch_votes(alarm_cf, s_cf["alpha"], W, start)
            for cthr in edet["c"]:
                al = sr_edetector(v, edet["p"][kind], s_cf["k_eff"], edet["lambdas"], cthr)
                nE_pre = (t0 - start) // W if has_drift else len(v)
                r = dict(false_alarms=int(al[:nE_pre].sum()), epochs_pre=int(nE_pre))
                if has_drift:
                    alarm_d = s_d["raw_alarm"] if kind == "raw" else (s_d["raw_alarm"] & s_d["adj_alarm"])
                    vd = epoch_votes(alarm_d, s_d["alpha"], W, start)
                    ald = sr_edetector(vd, edet["p"][kind], s_d["k_eff"], edet["lambdas"], cthr)
                    e0 = (t0 - start) // W
                    eH = e0 + max(1, H // W)
                    att = ald[e0:eH] & ~al[e0:eH]
                    r["detected"] = bool(att.any())
                    r["delay_epochs"] = int(np.argmax(att)) if att.any() else None
                out["methods"][f"edet-{kind}@c={cthr}"] = r
    return out
