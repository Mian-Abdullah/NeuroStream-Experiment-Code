"""All paper figures, generated only from result files (and, for Fig. 1, from a seeded re-run of the simulator whose
series are saved to results/figs_data/fig1.json).  Usage: python exps/f_figures.py -> figs/*.pdf, figs/*.png
Colour: validated categorical slots (blue, orange, aqua, yellow) + grey; identity also carried by markers/labels.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ns import sim, runner

R = runner.RESULTS
FIG = os.path.join(os.path.dirname(R), "figs")
os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.join(R, "figs_data"), exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85"
INK, MUTED = "#1f1f1e", "#6b6b66"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "font.family": "serif", "pdf.fonttype": 42, "savefig.dpi": 300})


def J(*p):
    return json.load(open(os.path.join(R, *p)))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)


def fig1():
    cfg = dict(sim.DEFAULT)
    seed = 3
    g = sim.generate(cfg, seed, drift=True, outage=True)
    s = sim.decision_signals(cfg, g, (0.35,))
    outfrac = g["hit"].mean(0)
    b = np.arange(cfg["B"])
    json.dump(dict(seed=seed, v_raw=s["v_raw"].tolist(), v_filtered=s["v_f"].tolist(), outage_frac=outfrac.tolist(),
                   t0=cfg["t0"], theta=0.35), open(os.path.join(R, "figs_data", "fig1.json"), "w"))
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    ax.fill_between(b, 0, outfrac, color=GREY, alpha=0.25, lw=0, step="mid", label="nodes in outage (fraction)")
    ax.plot(b, s["v_raw"], color=ORANGE, lw=1.1, label="vote, raw alarms")
    ax.plot(b, s["v_f"], color=BLUE, lw=1.4, label="vote, covariate-adjusted")
    ax.axhline(0.35, color=INK, lw=0.8, ls="--")
    ax.text(15, 0.37, r"$\theta=0.35$", color=INK, fontsize=7)
    ax.axvline(cfg["t0"], color=INK, lw=0.8, ls=":")
    ax.text(cfg["t0"] + 8, 0.62, "true drift", fontsize=7, color=INK)
    ax.set_xlim(0, cfg["B"]); ax.set_ylim(0, 1.0)
    ax.set_xlabel("block"); ax.set_ylabel("weighted vote fraction")
    ax.legend(loc="upper left", frameon=False, ncol=1, bbox_to_anchor=(0.0, 1.04), fontsize=6.5)
    save(fig, "fig1_motivation")


def fig2():
    a = json.load(open(os.path.join(os.path.dirname(R), "results", "legacy", "exp_a.json")))
    t = J("T", "theory.json")
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.0))
    ax = axs[0]
    emp = np.array([r["empirical"] for r in a["rows"]])
    bo = np.array([r["bound_original"] for r in a["rows"]])
    bc = np.array([r["bound_corrected"] for r in a["rows"]])
    m = emp > 0
    ax.loglog(bo[m], emp[m], "o", ms=3.5, color=ORANGE, mfc="none", label="equal-weight bound")
    ax.loglog(bc[m], emp[m], "s", ms=3.5, color=BLUE, label=r"$K_{\mathrm{eff}}$ bound (Thm. 1)")
    lim = [1e-7, 1.5]
    ax.plot(lim, lim, color=INK, lw=0.8)
    ax.text(2e-6, 3e-2, "above line:\nbound violated", fontsize=6.5, color=MUTED)
    ax.set_xlim(lim); ax.set_ylim(1e-6, 1.5)
    ax.set_xlabel("bound value"); ax.set_ylabel("empirical false-positive prob.")
    ax.set_title("(a) per-window bound, 45 configs")
    ax.legend(frameon=False, loc="lower right")
    ax = axs[1]
    for gam, col, mk in [(0.2, AQUA, "^"), (0.5, ORANGE, "o")]:
        rr = [r for r in a["common_cause"] if r["gamma"] == gam]
        ax.plot([r["K"] for r in rr], [max(r["fp"], 1e-6) for r in rr], marker=mk, ms=4, color=col,
                label=rf"$\gamma={gam}$" + (r" ($\gamma q<\theta$)" if gam == 0.2 else r" ($\gamma q>\theta$)"))
    ax.axhline(0.05, color=INK, lw=0.8, ls="--"); ax.text(12, 0.06, r"$\pi=P(Z)=0.05$", fontsize=6.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-7, 0.3)
    ax.set_xlabel("federation size $K$"); ax.set_ylabel("false-positive prob.")
    ax.set_title("(b) common cause (Prop. 2)")
    ax.legend(frameon=False, loc="center right")
    ax = axs[2]
    for scen, col, mk, lab in [("independent", BLUE, "s", "independent alarms"), ("common_cause", ORANGE, "o", "common cause")]:
        rr = [r for r in t["edetector_arl0"] if r["K"] == 50 and r["scenario"] == scen]
        cs = [r["c"] for r in rr]
        ys = [r["arl0_hat"] if r["arl0_hat"] else r["arl0_lower95"] for r in rr]
        ax.plot(cs, ys, marker=mk, ms=4, color=col, label=lab)
        for r, y in zip(rr, ys):
            if not r["arl0_hat"]:
                ax.annotate("", xy=(r["c"], y * 3), xytext=(r["c"], y), arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    c = np.array([10, 2000])
    ax.plot(c, c, color=INK, lw=0.8, ls="--"); ax.text(60, 2500, r"guarantee $\mathrm{ARL}_0=c$ (dashed)", fontsize=6.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5, 3e7)
    ax.set_xlabel("alarm threshold $c$"); ax.set_ylabel(r"$\widehat{\mathrm{ARL}}_0$ (epochs)")
    ax.set_title(r"(c) e-detector (Thm.~4), K=50".replace("~", " "))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig2_theory")


def fig3():
    tc = J("S_sweeps", "theta_curve.json")["summary"]["methods"]
    ths = sorted({float(k.split("@")[1]) for k in tc if "@" in k})
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.0))
    ax = axs[0]
    for pre, col, mk, lab in [("FDC", ORANGE, "o", "FDC (vote count)"), ("FDC+filter", BLUE, "s", "FDC + causal filter")]:
        x = [tc[f"{pre}@{t:.2f}"]["false_episodes_per_1000"]["mean"] for t in ths]
        y = [tc[f"{pre}@{t:.2f}"]["attributable_detection"]["rate"] for t in ths]
        ax.plot(x, y, marker=mk, ms=3.5, color=col, label=lab)
    for t, off in [(0.35, (-6, -12)), (0.55, (8, -2))]:
        k = f"FDC@{t:.2f}"
        ax.annotate(rf"$\theta$={t}", (tc[k]["false_episodes_per_1000"]["mean"], tc[k]["attributable_detection"]["rate"]),
                    xytext=off, textcoords="offset points", fontsize=6.5, color=MUTED)
    ax.set_xlabel("false global retrains / 1,000 blocks"); ax.set_ylabel("attributable detection")
    ax.set_title(r"(a) threshold sweep $\theta\in[0.25,0.65]$")
    ax.legend(frameon=False, loc="lower right")
    for ax, key, vals, xl, ttl in [(axs[1], "zobs", [1.0, 0.75, 0.5, 0.25, 0.0], "share of nodes observing the cause", "(b) partially hidden common cause"),
                                   (axs[2], "zsd", [0.03, 0.1, 0.2, 0.3, 0.5], "covariate noise s.d. (effect 0.5)", "(c) noisy covariate")]:
        for m, col, mk, lab in [("FDC@0.35", ORANGE, "o", r"FDC $\theta$=0.35"), ("FDC+filter@0.35", BLUE, "s", r"FDC+filter $\theta$=0.35")]:
            y = [J("S_fail", f"{key}_{v}.json")["summary"]["methods"][m]["false_episodes_per_1000"]["mean"] for v in vals]
            lo = [J("S_fail", f"{key}_{v}.json")["summary"]["methods"][m]["false_episodes_per_1000"]["ci95"][0] for v in vals]
            hi = [J("S_fail", f"{key}_{v}.json")["summary"]["methods"][m]["false_episodes_per_1000"]["ci95"][1] for v in vals]
            ax.errorbar(vals, y, yerr=[np.array(y) - lo, np.array(hi) - y], marker=mk, ms=3.5, color=col, capsize=2, lw=1.4, label=lab)
        ax.set_xlabel(xl); ax.set_ylabel("false retrains / 1,000 blk"); ax.set_title(ttl)
        ax.set_ylim(-0.2, 6.9)
        if key == "zobs":
            ax.invert_xaxis()
        ax.legend(frameon=False, loc="upper left", ncol=1)
    axs[0].set_xlabel("false retrains / 1,000 blocks")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig3_tradeoff_failure")


def fig4():
    cfgs = [("covtype", 10), ("covtype", 50), ("covtype", 100), ("airlines", 10), ("airlines", 50), ("airlines", 100), ("elec", 5), ("elec", 10)]
    labs, fe_f, fe_c, de_f, de_c, ci = [], [], [], [], [], []
    for d, K in cfgs:
        o = J("A", f"open_{d}_K{K}.json")
        s, tu = o["summary"]["methods"], o["tuned"]
        a, b = s[f"FDC@{tu['theta_fdc']:.2f}"], s[f"FDC+filter@{tu['theta_filter']:.2f}"]
        labs.append(f"{ {'covtype': 'Cov', 'airlines': 'Air', 'elec': 'Elec'}[d] }\n{K}")
        fe_f.append(a["false_episodes_per_1000"]); fe_c.append(b["false_episodes_per_1000"])
        de_f.append(a["attributable_detection"]); de_c.append(b["attributable_detection"])
    x = np.arange(len(cfgs)); w = 0.38
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.3), sharex=True)
    for ax, A_, B_, key, yl in [(axs[0], fe_f, fe_c, "mean", "outage-induced false\nretrains / 1,000 blocks"),
                                (axs[1], de_f, de_c, "rate", "attributable detection")]:
        for off, S_, col, lab in [(-w / 2, A_, ORANGE, "tuned FDC"), (w / 2, B_, BLUE, "tuned FDC + filter")]:
            y = np.array([s_[key] for s_ in S_])
            lo = np.array([s_["ci95"][0] for s_ in S_]); hi = np.array([s_["ci95"][1] for s_ in S_])
            ax.bar(x + off, y, w - 0.04, color=col, label=lab, edgecolor="white", linewidth=0.6)
            ax.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
        ax.set_ylabel(yl)
    axs[1].set_ylim(0, 1.05)
    axs[0].legend(frameon=False, ncol=2, loc="upper right")
    axs[1].set_xticks(x); axs[1].set_xticklabels(labs, fontsize=6.5)
    axs[1].set_xlabel("dataset and federation size $K$ (Cov = Covertype, Air = Airlines, Elec = Elec2)", fontsize=6.5)
    fig.tight_layout(h_pad=0.4)
    save(fig, "fig4_real_streams")


def fig5():
    acc = J("P", "accounting.json")
    byz = J("B", "byzantine.json")["summary"]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.0), gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax = axs[0]
    k = "sigma=10.0,R=10,drifts_per_1000=1.0"
    pol = acc["retraining_horizon_blocks"][k]
    names = [("continuous_every_block", "continuous FL"), ("periodic_every_100", "periodic (P=100)"),
             ("fdc035_private_votes", r"FDC $\theta$=0.35"), ("fdc045_private_votes", r"FDC $\theta$=0.45"),
             ("filter035_private_votes", "FDC + filter"), ("oracle_retrain_only_on_drift", "oracle")]
    vals = [pol[n] for n, _ in names]
    cols = [GREY, GREY, ORANGE, ORANGE, BLUE, AQUA]
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, vals, color=cols, height=0.6, edgecolor="white")
    for yv, v in zip(yy, vals):
        ax.text(v * 1.08, yv, f"{v:,}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([l for _, l in names])
    ax.set_xscale("log"); ax.set_xlim(30, 4e4)
    ax.set_xlabel(r"blocks until $\varepsilon>4$ ($\delta=10^{-5}$)")
    ax.set_title(r"(a) private horizon ($\sigma$=10, R=10)")
    ax = axs[1]
    aggs = [("mean", "mean", GREY, "x"), ("median", "median", AQUA, "^"), ("trimmed", r"trimmed $\beta$=0.1", ORANGE, "o"),
            ("multikrum", "Multi-Krum", YELLOW, "D"), ("cclip", "centered clipping", BLUE, "s")]
    ATT = ["signflip", "alie", "ipm", "minmax"]
    for ag, lab, col, mk in aggs:
        xs = [0.0, 0.1, 0.2, 0.3]
        ys = [byz[f"{ag}|none|0.0"]["mean"]] + [min(byz[f"{ag}|{a}|{f}"]["mean"] for a in ATT) for f in xs[1:]]
        ax.plot(xs, ys, marker=mk, ms=3.5, color=col, label=lab, lw=1.3)
    ax.set_ylim(0, 0.72)
    ax.set_xlabel("fraction of Byzantine nodes"); ax.set_ylabel("worst-case test accuracy")
    ax.set_title("(b) worst case over 4 attacks, Covertype")
    ax.legend(frameon=False, ncol=2, fontsize=6.3, loc="lower left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig5_privacy_robustness")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        globals()[f"fig{w}"]()
        print("fig", w, "ok")
