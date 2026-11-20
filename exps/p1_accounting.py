"""Privacy accounting that charges every data-dependent release (votes, filter statistics, model updates).
Uses measured event rates from Track S result files; deterministic.  Output: results/P/accounting.json
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import accounting as A, runner

R = runner.RESULTS
EPS_MAX, DELTA = 4.0, 1e-5


def load(sub, name):
    return json.load(open(os.path.join(R, sub, f"{name}.json")))["summary"]


def main():
    out = {"eps_max": EPS_MAX, "delta": DELTA}
    # 1. continuity with the revised manuscript: sigma = 25, every round
    s = 25.0
    def basic(T):
        d0 = DELTA / T; e0 = math.sqrt(2 * math.log(1.25 / d0)) / s; return T * e0
    def advanced(T):
        d0 = DELTA / (2 * T); e0 = math.sqrt(2 * math.log(1.25 / d0)) / s
        return math.sqrt(2 * T * math.log(2 / DELTA)) * e0 + T * e0 * (math.exp(e0) - 1)
    def bs(f, hi=20000):
        lo, h = 0, hi
        while lo < h:
            m = (lo + h + 1) // 2
            lo, h = (m, h) if f(m) <= EPS_MAX else (lo, m - 1)
        return lo
    out["sigma25_rounds"] = dict(basic=bs(basic), advanced=bs(advanced),
                                 rdp_dp_accounting=bs(lambda T: A.eps_dp_accounting(s, T, DELTA, "rdp"), 3000),
                                 pld_dp_accounting=bs(lambda T: A.eps_dp_accounting(s, T, DELTA, "pld"), 3000),
                                 rdp_cks_conversion=A.max_units({s: 1.0}, 0.0))
    # 2. decision-layer privacy: node-side Sparse-Vector votes (eps_svt = 0.2, c = 1), measured positive rates
    svt = load("S_privacy", "svt_0.2_1")
    pos_raw = svt["svt_positives_per_node_per_1000"]["mean"] / 1000          # per block
    pos_f = svt["svt_positives_filtered_per_node_per_1000"]["mean"] / 1000
    rho_inst = 0.2 ** 2 / 2
    fe = {m: svt["methods"][m]["false_episodes_per_1000"]["mean"] / 1000 for m in ["FDC@0.35", "FDC@0.45", "FDC+filter@0.35"]}
    out["svt_measured"] = dict(positives_raw_per_node_per_block=pos_raw, positives_filtered_per_node_per_block=pos_f,
                               false_global_episodes_per_block=fe, source="results/S_privacy/svt_0.2_1.json")
    # 3. horizon (blocks until eps > 4) of retraining policies; R updates per retraining, d genuine drifts / 1000 blocks
    pol = {}
    for sig in [5.0, 10.0, 25.0]:
        for Rtr in [5, 10, 20]:
            for d in [0.5, 1.0, 2.0]:
                key = f"sigma={sig},R={Rtr},drifts_per_1000={d}"
                g = d / 1000
                pol[key] = dict(
                    continuous_every_block=A.max_units({sig: 1.0}, 0.0),
                    periodic_every_100=A.max_units({sig: Rtr / 100}, 0.0),
                    fdc035_private_votes=A.max_units({sig: Rtr * (fe["FDC@0.35"] + g)}, (pos_raw) * rho_inst),
                    fdc045_private_votes=A.max_units({sig: Rtr * (fe["FDC@0.45"] + g)}, (pos_raw) * rho_inst),
                    filter035_private_votes=A.max_units({sig: Rtr * (fe["FDC+filter@0.35"] + g)}, (pos_f) * rho_inst),
                    oracle_retrain_only_on_drift=A.max_units({sig: Rtr * g}, 0.0))
    out["retraining_horizon_blocks"] = pol
    # 4. node-level drift gating (the earlier drafts' idea) with releases-per-activation R made explicit
    gate = []
    for (eps_svt, c) in [(0.2, 1), (0.5, 1), (0.5, 5), (1.0, 1)]:
        sv = load("S_privacy", f"svt_{eps_svt}_{c}")
        f = sv["svt_active_frac"]["mean"]
        posr = sv["svt_positives_per_node_per_1000"]["mean"] / 1000
        for sig in [5.0, 25.0]:
            base = A.max_units({sig: 1.0}, 0.0)          # release every round, no gate
            for Rr in [1, 5, 10, 20]:
                # per round: positives posr; each positive authorises Rr releases; instances = positives / c
                rel = posr * Rr
                rho = posr / c * eps_svt ** 2 / 2
                T = A.max_units({sig: rel}, rho)
                gate.append(dict(eps_svt=eps_svt, c=c, sigma=sig, R=Rr, simulated_active_frac=f,
                                 positives_per_round=posr, releases_per_round=rel, horizon_rounds=T,
                                 horizon_no_gate=base, gain=round(T / base, 3) if base else None))
    out["node_gating"] = gate
    # 5. server-side private filter (Gaussian-noised window sums) vs node-side SVT filter
    pf = []
    for mult in [1, 3, 10, 30]:
        sv = load("S_privacy", f"privfilter_{mult}")
        rel = sv["filter_releases_per_node"]["mean"]
        pf.append(dict(noise_multiplier=mult, releases_per_node_per_run=rel, blocks_per_run=1200,
                       eps_per_run=A.eps_of(A.rdp_total({float(mult): rel}), DELTA),
                       false_episodes_filter035=sv["methods"]["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"],
                       detection_filter035=sv["methods"]["FDC+filter@0.35"]["attributable_detection"]["rate"]))
    out["server_side_private_filter"] = pf
    out["node_side_svt_filter_eps_per_1200_blocks"] = A.eps_of(A.rdp_total({}, pos_f * 1200 * rho_inst), DELTA)
    os.makedirs(os.path.join(R, "P"), exist_ok=True)
    json.dump(out, open(os.path.join(R, "P", "accounting.json"), "w"), indent=1)
    print(json.dumps(out["sigma25_rounds"]), json.dumps(out["svt_measured"]))
    k = "sigma=10.0,R=10,drifts_per_1000=1.0"
    print(k, out["retraining_horizon_blocks"][k])
    for g in gate:
        if g["sigma"] == 25.0: print(g)
    for x in pf: print(x)
    print("node-side svt eps/1200 blocks", out["node_side_svt_filter_eps_per_1200_blocks"])


if __name__ == "__main__":
    main()
