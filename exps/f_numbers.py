"""Generates paper/numbers.tex (one \\newcommand per number quoted in the text) and paper/tables/*.tex from the
result files. Nothing in the paper's text or tables is typed by hand.  Usage: python exps/f_numbers.py
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import runner

R = runner.RESULTS
P = os.path.join(os.path.dirname(R), "paper")
os.makedirs(os.path.join(P, "tables"), exist_ok=True)
N = {}
MISSING = r"\textbf{[TO BE MEASURED]}"


def J(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def f2(x, d=2):
    return MISSING if x is None else f"{x:.{d}f}"


def put(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    N[name] = val


def ci(c, d=2):
    return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def fe(m):          # false episodes cell: mean [CI] (or 0 with upper bound)
    x = m["false_episodes_per_1000"]
    if x["total_events"] == 0:
        return rf"0 ($\le${x['poisson_ucb95']:.3f})"
    return rf"{x['mean']:.2f} {ci(x['ci95'])}"


def det(m):
    x = m["attributable_detection"]
    return rf"{x['rate']:.2f} {ci(x['ci95'])}"


def main():
    for fn in os.listdir(os.path.join(P, "tables")):      # never leave a stale table behind
        os.remove(os.path.join(P, "tables", fn))
    # ------------------------------------------------------------------ Track S main
    S = J("S", "main_outage_drift.json")["summary"]
    tune = J("S", "tuning.json")
    tf, tc = tune["theta_fdc"], tune["theta_filter"]
    m = S["methods"]
    put("SthetaFdc", f2(tf)); put("SthetaFil", f2(tc))
    put("SfdcFE", f2(m["FDC@0.35"]["false_episodes_per_1000"]["mean"])); put("SfdcFEci", ci(m["FDC@0.35"]["false_episodes_per_1000"]["ci95"]))
    put("SfdcDet", f2(m["FDC@0.35"]["attributable_detection"]["rate"], 3)); put("SfdcRawDet", f2(m["FDC@0.35"]["raw_detection"]["rate"], 3))
    put("SfdctFE", f2(m[f"FDC@{tf:.2f}"]["false_episodes_per_1000"]["mean"])); put("SfdctDet", f2(m[f"FDC@{tf:.2f}"]["attributable_detection"]["rate"], 3))
    put("SfdcHighFE", f2(m["FDC@0.55"]["false_episodes_per_1000"]["mean"])); put("SfdcHighDet", f2(m["FDC@0.55"]["attributable_detection"]["rate"], 3))
    put("SfilUCB", f2(m["FDC+filter@0.35"]["false_episodes_per_1000"]["poisson_ucb95"], 3))
    put("SfilDet", f2(m["FDC+filter@0.35"]["attributable_detection"]["rate"], 3)); put("SfilDetci", ci(m["FDC+filter@0.35"]["attributable_detection"]["ci95"], 3))
    put("SfiltDet", f2(m[f"FDC+filter@{tc:.2f}"]["attributable_detection"]["rate"], 3))
    put("SlocalW", f2(m["local"]["wasted_node_retrains_per_1000"]["mean"], 1)); put("SlocfilW", f2(m["local+filter"]["wasted_node_retrains_per_1000"]["mean"], 1))
    put("SfdcW", f2(m["FDC@0.35"]["wasted_node_retrains_per_1000"]["mean"], 1))
    put("SpRaw", f2(S["p_hat_raw"]["mean"], 3)); put("SpFil", f2(S["p_hat_filtered"]["mean"], 3)); put("SKeff", f2(S["k_eff"]["mean"], 1))
    put("SedRawFA", f2(m["edet-raw@c=100"]["false_alarms_per_1000_epochs"]["mean"], 1))
    put("SedFilDet", f2(m["edet-filtered@c=100"]["attributable_detection"]["rate"], 3))
    put("SedFilUCB", f2(m["edet-filtered@c=100"]["false_alarms_per_1000_epochs"]["poisson_ucb95"], 2))
    rows = [("Local retrain on every alarm", None, "local"), ("Local retrain, filtered alarms", None, "local+filter"),
            (r"FDC, $\theta=0.35$ (prior work)", "FDC@0.35", None), (rf"FDC, tuned $\theta^\ast={tf:.2f}$", f"FDC@{tf:.2f}", None),
            (r"FDC, $\theta=0.55$", "FDC@0.55", None), (r"FDC + filter, $\theta=0.35$", "FDC+filter@0.35", None),
            (rf"FDC + filter, tuned $\theta^\ast={tc:.2f}$", f"FDC+filter@{tc:.2f}", None),
            (r"e-detector, raw votes ($c=100$)", "edet-raw@c=100", "ed"), (r"e-detector, filtered votes ($c=100$)", "edet-filtered@c=100", "ed")]
    L = []
    for lab, key, kind in rows:
        if kind in ("local", "local+filter"):
            a = m[kind]
            L.append(rf"{lab} & -- & {a['node_detect_frac']['mean']:.3f}$^\dagger$ & {a['wasted_node_retrains_per_1000']['mean']:.1f} \\")
        elif kind == "ed":
            a = m[key]; x = a["false_alarms_per_1000_epochs"]
            fa = rf"0 ($\le${x['poisson_ucb95']:.2f})" if x["total_events"] == 0 else rf"{x['mean']:.1f} {ci(x['ci95'],1)}"
            L.append(rf"{lab} & {fa}$^\ddagger$ & {det(a)} & -- \\")
        else:
            a = m[key]
            L.append(rf"{lab} & {fe(a)} & {det(a)} & {a['wasted_node_retrains_per_1000']['mean']:.1f} \\")
    open(os.path.join(P, "tables", "tab_sim_main.tex"), "w").write("\n".join(L) + "\n")
    # weights ablation + scenarios (appendix)
    L = []
    for w, lab in [("weights_equal", "equal"), (None, "data volume (default)"), ("weights_random_rho", r"volume $\times$ random $\rho$ (prior code)"),
                   ("weights_volume_x_calibrated", r"volume $\times$ calibrated $\rho$")]:
        mm = (J("S", f"{w}.json")["summary"]["methods"] if w else m)
        L.append(rf"{lab} & {fe(mm['FDC@0.35'])} & {det(mm['FDC@0.35'])} & {fe(mm['FDC+filter@0.35'])} & {det(mm['FDC+filter@0.35'])} \\")
    open(os.path.join(P, "tables", "tab_sim_weights.tex"), "w").write("\n".join(L) + "\n")
    # sweeps (appendix)
    L = []
    for grp, vals, fmt in [("K", [10, 25, 50, 100, 200, 400], "K_{}"), ("affected fraction", [0.3, 0.4, 0.5, 0.6, 0.8, 1.0], "frac_{}"),
                           ("drift magnitude", [0.33, 0.5, 0.67, 1.0], "mag_{}")]:
        for v in vals:
            mm = J("S_sweeps", fmt.format(v) + ".json")["summary"]["methods"]
            L.append(rf"{grp} = {v} & {fe(mm['FDC@0.35'])} & {det(mm['FDC@0.35'])} & {fe(mm['FDC+filter@0.35'])} & {det(mm['FDC+filter@0.35'])} \\")
    open(os.path.join(P, "tables", "tab_sim_sweeps.tex"), "w").write("\n".join(L) + "\n")
    # failure modes + attacks (main text table)
    L = []
    def fm(sub, name, lab):
        mm = J(sub, name + ".json")["summary"]["methods"]
        L.append(rf"{lab} & {fe(mm['FDC@0.35'])} & {det(mm['FDC@0.35'])} & {fe(mm['FDC+filter@0.35'])} & {det(mm['FDC+filter@0.35'])} \\")
        return mm
    fm("S_fail", "zobs_1.0", "covariate observed at all affected nodes (default)")
    a5 = fm("S_fail", "zobs_0.5", "covariate observed at 50\\% of affected nodes")
    a0 = fm("S_fail", "zobs_0.0", "hidden common cause (no covariate)")
    z2 = fm("S_fail", "zsd_0.2", "noisy covariate (s.d.\\ 0.2 vs.\\ effect 0.5)")
    fm("S_fail", "nonlinear", "non-linear covariate effect")
    sp = fm("S_fail", "spoofed_feed", "compromised shared covariate feed")
    L.append(r"\midrule")
    for mode in ["flood", "suppress"]:
        for fr in [0.1, 0.2, 0.3]:
            fm("S_attack", f"byz_{mode}_{fr}", rf"Byzantine vote {'flooding' if mode=='flood' else 'suppression'}, {int(fr*100)}\% of nodes")
    open(os.path.join(P, "tables", "tab_sim_failure.tex"), "w").write("\n".join(L) + "\n")
    put("SfailHalfFE", f2(a5["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"]))
    put("SfailHidFE", f2(a0["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"]))
    put("SfailNoiseFE", f2(z2["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"]))
    put("SspoofDet", f2(sp["FDC+filter@0.35"]["attributable_detection"]["rate"], 3))
    fl2 = J("S_attack", "byz_flood_0.2.json")["summary"]["methods"]; fl3 = J("S_attack", "byz_flood_0.3.json")["summary"]["methods"]
    put("SfloodTwoFE", f2(fl2["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"])); put("SfloodThreeFE", f2(fl3["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"]))
    su3 = J("S_attack", "byz_suppress_0.3.json")["summary"]["methods"]
    put("SsuppThreeDet", f2(su3["FDC+filter@0.35"]["attributable_detection"]["rate"], 3))
    # SVT / private filter
    sv = J("S_privacy", "svt_0.2_1.json")["summary"]
    put("SvtFilDet", f2(sv["methods"]["FDC+filter@0.35"]["attributable_detection"]["rate"], 3))
    put("SvtPosRaw", f2(sv["svt_positives_per_node_per_1000"]["mean"], 1)); put("SvtPosFil", f2(sv["svt_positives_filtered_per_node_per_1000"]["mean"], 1))
    put("SvtActRaw", f2(sv["svt_active_frac"]["mean"], 3))
    # ablations / sensitivity quoted in the text
    wc = J("S", "weights_volume_x_calibrated.json")["summary"]["methods"]; we = J("S", "weights_equal.json")["summary"]["methods"]
    put("WcalFdcFE", f2(wc["FDC@0.35"]["false_episodes_per_1000"]["mean"])); put("WcalFdcFEh", f2(wc["FDC@0.45"]["false_episodes_per_1000"]["mean"]))
    put("WcalFdcDeth", f2(wc["FDC@0.45"]["attributable_detection"]["rate"], 3))
    put("WeqFdcFE", f2(we["FDC@0.35"]["false_episodes_per_1000"]["mean"]))
    ml = J("S_sweeps", "mag_0.33.json")["summary"]["methods"]
    put("MagLowFilDet", f2(ml["FDC+filter@0.35"]["attributable_detection"]["rate"], 2)); put("MagLowFdcDet", f2(ml["FDC@0.35"]["attributable_detection"]["rate"], 2))
    put("MagLowFiltTDet", f2(ml[f"FDC+filter@{tc:.2f}"]["attributable_detection"]["rate"], 2))
    k4 = J("S_sweeps", "K_400.json")["summary"]["methods"]
    put("KfourFdcFE", f2(k4["FDC@0.35"]["false_episodes_per_1000"]["mean"])); put("KfourFilDet", f2(k4["FDC+filter@0.35"]["attributable_detection"]["rate"], 2))
    nl = J("S_fail", "nonlinear.json")["summary"]["methods"]
    put("NonlinFdcFE", f2(nl["FDC@0.35"]["false_episodes_per_1000"]["mean"])); put("NonlinFilFE", f2(nl["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"]))
    put("SlocScope", f2(S["local_scope_per_1000"]["mean"], 1)); put("SlocScopeF", f2(S["local_scope_filtered_per_1000"]["mean"], 1))
    put("SedFilDelay", f2(m["edet-filtered@c=100"]["delay_median"], 0)); put("SfilDelay", f2(m["FDC+filter@0.35"]["delay_median"], 0))
    s10 = J("S_privacy", "svt_1.0_1.json")["summary"]; s25 = J("S_privacy", "svt_0.2_5.json")["summary"]
    put("SvtCfiveAct", f2(s25["svt_active_frac"]["mean"], 3)); put("SvtCfiveFE", f2(s25["methods"]["FDC@0.35"]["false_episodes_per_1000"]["mean"]))
    # theta curve and small-K sweep (quoted in the text)
    tc = J("S_sweeps", "theta_curve.json")["summary"]["methods"]
    fil = [v for k, v in tc.items() if k.startswith("FDC+filter@")]
    put("CurveFilFEmax", f2(max(v["false_episodes_per_1000"]["mean"] for v in fil)))
    put("CurveFilDetMin", f2(min(v["attributable_detection"]["rate"] for v in fil), 2))
    put("CurveFilDetMax", f2(max(v["attributable_detection"]["rate"] for v in fil), 2))
    put("KtenFilFE", f2(J("S_sweeps", "K_10.json")["summary"]["methods"]["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"]))
    put("FloodOneFE", f2(J("S_attack", "byz_flood_0.1.json")["summary"]["methods"]["FDC+filter@0.35"]["false_episodes_per_1000"]["mean"]))
    lg_a = json.load(open(os.path.join(R, "legacy", "exp_a.json")))
    put("LegViol", str(sum(r["original_violated"] for r in lg_a["rows"]))); put("LegN", str(len(lg_a["rows"])))
    put("LegCorrViol", str(sum(r["corrected_violated"] for r in lg_a["rows"])))
    ccl = [r["fp"] for r in lg_a["common_cause"] if r["gamma"] == 0.5]
    put("LegCCmin", f2(min(ccl), 3)); put("LegCCmax", f2(max(ccl), 3))
    # ------------------------------------------------------------------ theory
    T = J("T", "theory.json")
    lg = {(r["kind"], r["c"]): r for r in T["tracks_long_horizon"]}
    put("TlongRawArlLo", f2(min(lg[("raw", c)]["arl0_hat"] for c in [20, 100, 1000]), 1))
    put("TlongRawArlHi", f2(max(lg[("raw", c)]["arl0_hat"] for c in [20, 100, 1000]), 1))
    put("TlongEpochs", f"{lg[('filtered', 100)]['epochs']:,}")
    put("TlongFilFAc", str(lg[("filtered", 20)]["false_alarms"]))
    put("TlongFilFAhundred", str(lg[("filtered", 100)]["false_alarms"]))
    cc = [r for r in T["edetector_arl0"] if r["scenario"] == "common_cause"]
    put("TccArlLo", f2(min(r["arl0_hat"] for r in cc), 0)); put("TccArlHi", f2(max(r["arl0_hat"] for r in cc), 0))
    ind = [r for r in T["edetector_arl0"] if r["scenario"] == "independent"]
    put("TindViol", str(sum(not r["guarantee_holds"] for r in ind))); put("TindN", str(len(ind)))
    put("TbyzViol", str(sum(r["violated"] for r in T["byzantine_corollary"]))); put("TbyzN", str(len(T["byzantine_corollary"])))
    dl = {(r["gamma"], r["c"]): r for r in T["edetector_delay"]}
    put("TdelayHalf", f2(dl[(0.5, 1000)]["delay_median"], 0)); put("TdelayThirty", f2(dl[(0.3, 1000)]["delay_median"], 0))
    put("TdetThirty", f2(dl[(0.3, 1000)]["detected_within_100"], 3))
    # ------------------------------------------------------------------ accounting
    A = J("P", "accounting.json")
    s25 = A["sigma25_rounds"]
    for k, nm in [("basic", "AccBasic"), ("advanced", "AccAdv"), ("rdp_dp_accounting", "AccRdp"), ("pld_dp_accounting", "AccPld")]:
        put(nm, str(s25[k]))
    h = A["retraining_horizon_blocks"]["sigma=10.0,R=10,drifts_per_1000=1.0"]
    for k, nm in [("continuous_every_block", "HorCont"), ("periodic_every_100", "HorPer"), ("fdc035_private_votes", "HorFdc"),
                  ("fdc045_private_votes", "HorFdct"), ("filter035_private_votes", "HorFil"), ("oracle_retrain_only_on_drift", "HorOra")]:
        put(nm, f"{h[k]:,}")
    put("HorRatio", f2(h["filter035_private_votes"] / h["fdc035_private_votes"], 1))
    L = []
    for sig in [5.0, 10.0, 25.0]:
        for Rr in [10]:
            for d in [0.5, 1.0, 2.0]:
                hh = A["retraining_horizon_blocks"][f"sigma={sig},R={Rr},drifts_per_1000={d}"]
                L.append(rf"{sig:g} & {Rr} & {d:g} & {hh['continuous_every_block']:,} & {hh['periodic_every_100']:,} & {hh['fdc035_private_votes']:,} & {hh['fdc045_private_votes']:,} & {hh['filter035_private_votes']:,} \\")
    open(os.path.join(P, "tables", "tab_horizon.tex"), "w").write("\n".join(L) + "\n")
    L = []
    for g in A["node_gating"]:
        if g["sigma"] == 25.0:
            L.append(rf"{g['eps_svt']:g} & {g['c']} & {g['R']} & {g['simulated_active_frac']:.3f} & {g['positives_per_round']*1000:.1f} & {g['horizon_rounds']:,} & {g['gain']:.2f} \\")
    open(os.path.join(P, "tables", "tab_gating.tex"), "w").write("\n".join(L) + "\n")
    g0 = [g for g in A["node_gating"] if g["sigma"] == 25.0 and g["eps_svt"] == 0.2 and g["c"] == 1]
    put("GateGainRone", f2(g0[0]["gain"])); put("GateGainRtwenty", f2(g0[-1]["gain"]))
    put("GateHorNoGate", str(g0[0]["horizon_no_gate"]))
    pf = {x["noise_multiplier"]: x for x in A["server_side_private_filter"]}
    put("PfEpsTen", f2(pf[10]["eps_per_run"], 1)); put("PfEpsThirty", f2(pf[30]["eps_per_run"], 1))
    put("PfFEThirty", f2(pf[30]["false_episodes_filter035"])); put("PfRel", f2(pf[10]["releases_per_node_per_run"], 0))
    put("SvtEpsRun", f2(A["node_side_svt_filter_eps_per_1200_blocks"], 2))
    # ------------------------------------------------------------------ DP utility
    D = J("P", "dp_utility.json")
    if D and "lr_clipped" in D:
        L = []
        keys = [("noclip", "0"), ("clip", "0"), ("dp", "1"), ("dp", "10"), ("dp", "25")]
        for nm, lab in [("covtype_K50", "Covertype, $K=50$"), ("elec_K10", "Elec2, $K=10$")]:
            cells = [f"{D['summary'][f'{nm}_{m}_sigma{sg}']['acc']['mean']:.3f}" for m, sg in keys]
            L.append(lab + " & " + " & ".join(cells) + r" \\")
        eps = [D["summary"][f"covtype_K50_dp_sigma{sg}"]["eps_after_60_releases"] for sg in ["1", "10", "25"]]
        L.append(r"$\varepsilon$ after 60 rounds & -- & -- & " + " & ".join(f"{e:.1f}" for e in eps) + r" \\")
        open(os.path.join(P, "tables", "tab_dputil.tex"), "w").write("\n".join(L) + "\n")
        g = lambda k: D["summary"][k]["acc"]["mean"]
        put("DpCovZero", f2(g("covtype_K50_noclip_sigma0"), 3)); put("DpCovClip", f2(g("covtype_K50_clip_sigma0"), 3))
        put("DpCovTen", f2(g("covtype_K50_dp_sigma10"), 3)); put("DpCovTwentyFive", f2(g("covtype_K50_dp_sigma25"), 3))
        put("DpEpsTen", f2(D["summary"]["covtype_K50_dp_sigma10"]["eps_after_60_releases"], 1))
        put("DpElecZero", f2(g("elec_K10_noclip_sigma0"), 3)); put("DpElecClip", f2(g("elec_K10_clip_sigma0"), 3)); put("DpElecTen", f2(g("elec_K10_dp_sigma10"), 3))
        put("DpLrCov", f2(D["lr_clipped"]["covtype"], 0))
    # ------------------------------------------------------------------ C-MAPSS
    C = J("B_real", "cmapss.json")
    L = []
    for fd in ["FD002", "FD004"]:
        o = C[fd]
        for kind, lab in [("raw", "raw"), ("adjusted", "per-condition")]:
            s_ = o["summary"][kind]
            h_ = s_["healthy_alarm_rate"]
            L.append(rf"{fd} & {lab} & {100*h_['mean']:.2f} [{100*h_['ci95'][0]:.2f}, {100*h_['ci95'][1]:.2f}] & {s_['degraded_detection']['rate']:.3f} {ci(s_['degraded_detection']['ci95'],3)} \\")
    open(os.path.join(P, "tables", "tab_cmapss.tex"), "w").write("\n".join(L) + "\n")
    put("CmRawDetTwo", f2(C["FD002"]["summary"]["raw"]["degraded_detection"]["rate"], 3)); put("CmAdjDetTwo", f2(C["FD002"]["summary"]["adjusted"]["degraded_detection"]["rate"], 3))
    put("CmRawDetFour", f2(C["FD004"]["summary"]["raw"]["degraded_detection"]["rate"], 3)); put("CmAdjDetFour", f2(C["FD004"]["summary"]["adjusted"]["degraded_detection"]["rate"], 3))
    put("CmAdjFATwo", f2(100 * C["FD002"]["summary"]["adjusted"]["healthy_alarm_rate"]["mean"], 2)); put("CmAdjFAFour", f2(100 * C["FD004"]["summary"]["adjusted"]["healthy_alarm_rate"]["mean"], 2))
    put("CmRowsTwo", f"{C['FD002']['rows']:,}"); put("CmRowsFour", f"{C['FD004']['rows']:,}")
    # ------------------------------------------------------------------ Track A open loop
    L, red = [], []
    for d, K in [("covtype", 10), ("covtype", 50), ("covtype", 100), ("airlines", 10), ("airlines", 50), ("airlines", 100), ("elec", 5), ("elec", 10)]:
        o = J("A", f"open_{d}_K{K}.json")
        if o is None:
            L.append(rf"{d} & {K} & \multicolumn{{6}}{{c}}{{{MISSING}}} \\"); continue
        s_, tu = o["summary"]["methods"], o["tuned"]
        a, b = s_[f"FDC@{tu['theta_fdc']:.2f}"], s_[f"FDC+filter@{tu['theta_filter']:.2f}"]
        lo = s_["local-twowindow"]["wasted_node_retrains_per_1000"]["mean"]
        nm = {"covtype": "Covertype", "airlines": "Airlines", "elec": "Elec2"}[d]
        L.append(rf"{nm} & {K} & {tu['theta_fdc']:.2f} / {tu['theta_filter']:.2f} & {fe(a)} & {fe(b)} & {det(a)} & {det(b)} & {lo:.1f} \\")
        red.append((a["false_episodes_per_1000"]["mean"], b["false_episodes_per_1000"]["mean"], a["attributable_detection"]["rate"], b["attributable_detection"]["rate"], nm, K))
    open(os.path.join(P, "tables", "tab_real_open.tex"), "w").write("\n".join(L) + "\n")
    if red:
        put("AnSettings", str(len(red)))
        put("AnLower", str(sum(r[1] < r[0] for r in red)))
        put("AnDetLower", str(sum(r[3] < r[2] - 1e-9 for r in red)))
        worst = min(red, key=lambda r: r[3] - r[2])
        put("AworstName", f"{worst[4]}, $K={worst[5]}$"); put("AworstDetF", f2(worst[2], 2)); put("AworstDetC", f2(worst[3], 2))
        put("AfdcFEmin", f2(min(r[0] for r in red))); put("AfdcFEmax", f2(max(r[0] for r in red)))
        put("AfilFEmin", f2(min(r[1] for r in red))); put("AfilFEmax", f2(max(r[1] for r in red)))
        put("AnDetGeq", str(sum(r[3] >= r[2] - 1e-9 for r in red)))
        edge = 0
        for d, K in [("covtype", 10), ("covtype", 50), ("covtype", 100), ("airlines", 10), ("airlines", 50), ("airlines", 100), ("elec", 5), ("elec", 10)]:
            o = J("A", f"open_{d}_K{K}.json")
            edge += int(abs(o["tuned"]["theta_filter"] - min(o["grid"])) < 1e-9) + int(abs(o["tuned"]["theta_fdc"] - min(o["grid"])) < 1e-9)
        put("AgridEdge", str(edge)); put("AgridMin", f2(min(o["grid"])))
        Dp_ = J("data_provenance.json")
        put("ElecRecTen", str(int(Dp_["datasets"]["elec"]["rows"] / 150 / 10))); put("ElecRecFive", str(int(Dp_["datasets"]["elec"]["rows"] / 150 / 5)))
    # ------------------------------------------------------------------ closed loop
    L = []
    for d in ["covtype", "airlines"]:
        o = J("A", f"closed_{d}_K50.json")
        if o is None:
            continue
        for pol, lab in [("static", "static"), ("periodic", "periodic ($P=50$)"), ("oracle", "oracle (retrain at $t_0{+}5$)"),
                         ("local_adwin", "local ADWIN + local retrain"), ("fdc", "FDC (tuned $\\theta$)"), ("fdc_filter", "FDC + filter (tuned $\\theta$)"),
                         ("feddrift", "FedDrift (re-impl.)")]:
            s_ = o["summary"][pol]
            extra = rf"{s_['fd_isolations_pre_t0']['mean']:.1f}$^\S$" if pol == "feddrift" else f"{s_['retrains_pre_t0']['mean']:.1f}"
            L.append(rf"{'Covertype' if d=='covtype' else 'Airlines'} & {lab} & {s_['acc_pre']['mean']:.3f} & {s_['acc_post']['mean']:.3f} & {s_['acc_all']['mean']:.3f} {ci(s_['acc_all']['ci95'],3)} & {extra} & {s_['retrains_during_outage']['mean']:.1f} & {s_['releases_per_node']['mean']:.0f} \\")
        L.append(r"\midrule")
        S_ = o["summary"]
        put(("Cov" if d == "covtype" else "Air") + "ClFdcPre", f2(S_["fdc"]["retrains_pre_t0"]["mean"], 1))
        put(("Cov" if d == "covtype" else "Air") + "ClFilPre", f2(S_["fdc_filter"]["retrains_pre_t0"]["mean"], 1))
        put(("Cov" if d == "covtype" else "Air") + "ClFdcAcc", f2(S_["fdc"]["acc_all"]["mean"], 3))
        put(("Cov" if d == "covtype" else "Air") + "ClFilAcc", f2(S_["fdc_filter"]["acc_all"]["mean"], 3))
        put(("Cov" if d == "covtype" else "Air") + "ClFdAcc", f2(S_["feddrift"]["acc_all"]["mean"], 3))
        put(("Cov" if d == "covtype" else "Air") + "ClFdRel", f2(S_["feddrift"]["releases_per_node"]["mean"], 0))
        put(("Cov" if d == "covtype" else "Air") + "ClFilRel", f2(S_["fdc_filter"]["releases_per_node"]["mean"], 0))
        put(("Cov" if d == "covtype" else "Air") + "ClFdIso", f2(S_["feddrift"]["fd_isolations_pre_t0"]["mean"], 1))
        pre_ = ("Cov" if d == "covtype" else "Air")
        put(pre_ + "ClFdcOut", f2(S_["fdc"]["retrains_during_outage"]["mean"], 1)); put(pre_ + "ClFilOut", f2(S_["fdc_filter"]["retrains_during_outage"]["mean"], 1))
        put(pre_ + "ClFdcTot", f2(S_["fdc"]["retrains"]["mean"], 1)); put(pre_ + "ClFilTot", f2(S_["fdc_filter"]["retrains"]["mean"], 1))
        put(pre_ + "ClPerAcc", f2(S_["periodic"]["acc_all"]["mean"], 3)); put(pre_ + "ClLocAcc", f2(S_["local_adwin"]["acc_all"]["mean"], 3))
        put(pre_ + "ClStaAcc", f2(S_["static"]["acc_all"]["mean"], 3)); put(pre_ + "ClFdcRel", f2(S_["fdc"]["releases_per_node"]["mean"], 0))
        r0 = o["per_run"][0]
        put(pre_ + "ClFdIsoPct", f2(100 * S_["feddrift"]["fd_isolations_pre_t0"]["mean"] / (o["K"] * (r0["t0"] - r0["Bw"])), 0))
    cl = [J("A", f"closed_{d}_K50.json") for d in ["covtype", "airlines"]]
    if all(cl):
        put("ClAccDiffMax", f2(max(abs(o["summary"]["fdc"]["acc_all"]["mean"] - o["summary"]["fdc_filter"]["acc_all"]["mean"]) for o in cl), 3))
    if L:
        open(os.path.join(P, "tables", "tab_real_closed.tex"), "w").write("\n".join(L[:-1]) + "\n")
    elif os.path.exists(os.path.join(P, "tables", "tab_real_closed.tex")):
        os.remove(os.path.join(P, "tables", "tab_real_closed.tex"))
    # ------------------------------------------------------------------ Byzantine
    Bz = J("B", "byzantine.json")
    if Bz:
        s_ = Bz["summary"]
        L = []
        for ag, lab in [("mean", "mean"), ("median", "coord.\\ median"), ("trimmed", r"trimmed mean $\beta{=}0.1$"), ("trimmed20", r"trimmed mean $\beta{=}0.2$"),
                        ("krum", "Krum"), ("multikrum", "Multi-Krum"), ("cclip", "centered clipping"), ("fltrust", "FLTrust")]:
            cells = [f"{s_[f'{ag}|none|0.0']['mean']:.3f}"] + [f"{s_[f'{ag}|{at}|{fr}']['mean']:.3f}" for at in ["signflip", "alie", "ipm", "minmax"] for fr in [0.2, 0.3]]
            L.append(lab + " & " + " & ".join(cells) + r" \\")
        open(os.path.join(P, "tables", "tab_byz.tex"), "w").write("\n".join(L) + "\n")
        put("ByzMeanNone", f2(s_["mean|none|0.0"]["mean"], 3))
        worst = {ag: min(s_[f"{ag}|{at}|0.3"]["mean"] for at in ["signflip", "alie", "ipm", "minmax"]) for ag in ["mean", "median", "trimmed", "trimmed20", "krum", "multikrum", "cclip", "fltrust"]}
        best = max(worst, key=worst.get)
        put("ByzBestAgg", {"mean": "mean", "median": "coordinate-wise median", "trimmed": "trimmed mean", "trimmed20": "trimmed mean ($\\beta=0.2$)",
                            "krum": "Krum", "multikrum": "Multi-Krum", "cclip": "centered clipping", "fltrust": "FLTrust"}[best])
        put("ByzBestWorst", f2(worst[best], 3)); put("ByzTrimWorst", f2(worst["trimmed"], 3)); put("ByzMedWorst", f2(worst["median"], 3))
    # ------------------------------------------------------------------ data provenance
    Dp = J("data_provenance.json")
    for k, nm in [("elec", "Elec"), ("covtype", "Cov"), ("airlines", "Air")]:
        put(f"Rows{nm}", f"{Dp['datasets'][k]['rows']:,}"); put(f"Feat{nm}", str(Dp["datasets"][k]["features_used"]))
        put(f"Bytes{nm}", f"{Dp['files'][k + '.csv']['bytes']/1e6:.1f}")
    put("BytesCm", f"{Dp['cmapss_all_12_files_bytes']/1e6:.1f}")
    put("EngTwo", str(Dp["datasets"]["cmapss_FD002"]["engines"])); put("EngFour", str(Dp["datasets"]["cmapss_FD004"]["engines"]))
    import glob
    tot = 0.0
    for f in glob.glob(os.path.join(R, "**", "*.json"), recursive=True):
        try:
            d_ = json.load(open(f))
            tot += d_.get("runtime_s", 0) if isinstance(d_, dict) else 0
        except Exception:
            pass
    put("RuntimeHours", f2(tot / 3600, 1))
    # placeholders for tables whose results are not yet available
    for fn, ncol in [("tab_real_closed.tex", 8), ("tab_dputil.tex", 6), ("tab_byz.tex", 10)]:
        fp = os.path.join(P, "tables", fn)
        if not os.path.exists(fp):
            open(fp, "w").write(rf"\multicolumn{{{ncol}}}{{c}}{{{MISSING}}} \\" + "\n")
    # ------------------------------------------------------------------ write
    with open(os.path.join(P, "numbers.tex"), "w") as f:
        f.write("% AUTO-GENERATED by exps/f_numbers.py from results/*.json -- do not edit by hand\n")
        for k, v in sorted(N.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(len(N), "numbers;", sorted(os.listdir(os.path.join(P, "tables"))))


if __name__ == "__main__":
    main()
