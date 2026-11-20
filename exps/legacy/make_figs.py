import json, math, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 200})
A = json.load(open("results/exp_a.json")); B = json.load(open("results/exp_b.json"))
Cc = json.load(open("results/exp_c.json")); Dd = json.load(open("results/exp_d.json"))
COL = dict(orig="#b5523b", corr="#1f3b6e", ok="#2e7d5b", grey="#777777", gold="#c28b1e")

# ---------- Fig 1: architecture
fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2)
def box(x, y, w, h, c, title, body, fs=7.6, tfs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec="#222", lw=1))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", color="w", fontsize=tfs, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.15, body, ha="center", va="center", color="w", fontsize=fs, linespacing=1.4)
box(0.5, 4.5, 8.6, 1.5, "#1f3b6e", "Tier 3 - Meta-Reasoning Engine (MRE)",
    "Causal alarm filter (backdoor adjustment on observed common causes)\nAdaptation policy store  \u2022  Federation health monitor\n[optional advisory policy module - design only, not evaluated]")
box(0.5, 2.55, 8.6, 1.5, "#2d5d8a", "Tier 2 - Federated Coordination Protocol (FCP)",
    "Reliability-weighted Federated Drift Consensus (Alg. 1)  \u2022  Trimmed-mean aggregator\nRenyi-DP accountant (gate cost charged)  \u2022  Topology manager")
for i, x in enumerate([0.5, 2.7, 4.9, 7.1]):
    box(x, 0.45, 2.0, 1.45, "#4a7c68", f"ECL node {i+1}", "streaming pipeline\nsimplified ADWIN-X\nlocal model + rollback", fs=6.6, tfs=8.5)
    ax.annotate("", xy=(x + 1.0, 2.55), xytext=(x + 1.0, 1.9), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#333"))
ax.annotate("", xy=(4.8, 4.5), xytext=(4.8, 4.05), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#333"))
ax.text(5.0, 4.27, "consensus events \u2193 / filtered decisions \u2191", fontsize=7.2, style="italic", va="center", ha="left")
ax.text(4.8, 2.22, "DP-noised updates and drift resolution requests (DRR)", fontsize=6.8, style="italic", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1.5), zorder=10)
ax.text(4.8, 0.12, "Tier 1 - Edge Cognition Layer (IoT gateways, micro-data centres, base stations)", ha="center", fontsize=7.6, style="italic", color="#2d4a3e")
fig.savefig("figs/fig1_arch.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2: bounds
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.2))
rows = A["rows"]
e = np.array([r["empirical"] for r in rows]); o = np.array([r["bound_original"] for r in rows]); c = np.array([r["bound_corrected"] for r in rows])
m = e > 0
ax[0].loglog(o[m], e[m], "o", c=COL["orig"], ms=4, alpha=.75, label="original bound, $K$")
ax[0].loglog(c[m], e[m], "s", c=COL["corr"], ms=4, alpha=.75, label="corrected bound, $K_{eff}$")
ax[0].plot([1e-6, 1], [1e-6, 1], "k--", lw=1); ax[0].set_xlim(1e-6, 1.2); ax[0].set_ylim(1e-6, 1.2)
ax[0].set_xlabel("bound"); ax[0].set_ylabel("empirical false-positive prob."); ax[0].legend(fontsize=7, loc="lower right")
ax[0].set_title("(a) Points above dashed line violate the bound", fontsize=8)
sw = Cc["K_sweep"]
for tag, mk in [("p~0.1", "o"), ("p~0.2", "s")]:
    rr = [r for r in sw if r["tag"] == tag]
    Ks = [r["K"] for r in rr]
    emp = [max(r["methods"]["FDC(th=0.35)"]["fp_block_rate"], 1e-5) for r in rr]
    bnd = [math.exp(-2 * r["K_eff"] * max(.35 - r["p_hat"], 0) ** 2) for r in rr]
    ax[1].semilogy(Ks, emp, mk + "-", c=COL["ok"] if tag == "p~0.1" else COL["gold"], label=f"simulated, {tag}")
    ax[1].semilogy(Ks, bnd, mk + ":", c=COL["ok"] if tag == "p~0.1" else COL["gold"], alpha=.6, label=f"corrected bound, {tag}")
ax[1].set_xlabel("federation size $K$"); ax[1].set_ylabel("false-positive rate per decision"); ax[1].legend(fontsize=6.5); ax[1].set_ylim(5e-6, 1.5)
ax[1].set_title("(b) Decay with $K$ (simulated detectors)", fontsize=8)
cc = A["common_cause"]
for g, col in [(0.2, COL["ok"]), (0.5, COL["orig"])]:
    rr = [r for r in cc if r["gamma"] == g]
    ax[2].semilogy([r["K"] for r in rr], [max(r["fp"], 1e-6) for r in rr], "o-", c=col, label=rf"outage hits {int(g*100)}% of nodes")
ax[2].axhline(0.05, c="k", ls="--", lw=1); ax[2].text(230, 0.02, "$P(Z)=0.05$", fontsize=7)
ax[2].set_xlabel("federation size $K$"); ax[2].set_ylabel("false-positive rate"); ax[2].set_title("(c) Common cause: no decay with $K$", fontsize=8); ax[2].legend(fontsize=7)
plt.tight_layout(); fig.savefig("figs/fig2_bounds.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: privacy
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3))
cv = B["curves"]; T = cv["T"]
ax[0].semilogy(T[1:], cv["basic"][1:], c=COL["grey"], label="basic composition")
ax[0].semilogy(T[1:], cv["advanced"][1:], c=COL["gold"], label="advanced composition")
ax[0].semilogy(T[1:], cv["rdp"][1:], c=COL["corr"], label="R\u00e9nyi DP")
ax[0].semilogy(T[1:], cv["pld"][1:], c=COL["ok"], ls="--", label="PLD (numerical)")
ax[0].axhline(4, c=COL["orig"], ls=":"); ax[0].text(25, 4.5, r"$\varepsilon_{max}=4$", fontsize=7, color=COL["orig"])
ax[0].set_xlim(0, 700); ax[0].set_ylim(0.3, 300); ax[0].set_xlabel("rounds (all nodes release every round)"); ax[0].set_ylabel(r"cumulative $\varepsilon$ ($\delta=10^{-5}$, $\sigma=25$)")
ax[0].legend(fontsize=7); ax[0].set_title("(a) Accountant choice", fontsize=8)
base = B["rounds_until_eps4_all_nodes_active"]["rdp_dp_accounting"]
fs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
ax[1].plot(fs, 1 / fs, c=COL["grey"], ls="--", label="gate assumed free ($1/f$)")
for (m_, e_, c_), col in zip([(50000, 0.2, 1), (50000, 0.5, 5), (50000, 0.2, 5)], [COL["ok"], COL["orig"], COL["gold"]]):
    r = [g for g in B["gating"] if g["m"] == m_ and g["eps_svt"] == e_ and g["c"] == c_][0]
    ax[1].plot(fs, r["T_active_max"] / fs / base, "o-", c=col, ms=3, label=rf"gate charged: $\varepsilon_{{svt}}$={e_}, c={c_} ({'noisy' if not r['useful'] and c_>1 and e_==0.2 else 'usable'})")
ax[1].axhline(1, c="k", lw=1); ax[1].set_yscale("log"); ax[1].set_xscale("log")
ax[1].set_xlabel("fraction of rounds in which a node is active, $f$"); ax[1].set_ylabel("training-horizon gain vs. no gating"); ax[1].legend(fontsize=6.3, loc="upper right")
ax[1].set_title("(b) Net effect of drift gating", fontsize=8)
plt.tight_layout(); fig.savefig("figs/fig3_privacy.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: FDC simulation
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
sw = Cc["drift_fraction_sweep"]
fr = [r["frac_drift"] for r in sw]
for name, col, lab, ls in [("FDC(th=0.35)", COL["orig"], r"FDC $\theta$=0.35", "-"), ("FDC(th=0.45)", COL["gold"], r"FDC $\theta$=0.45", "-"),
                           ("FDC(th=0.55)", COL["grey"], r"FDC $\theta$=0.55", "-"), ("FDC+filter(th=0.35)", COL["corr"], r"FDC+filter $\theta$=0.35", "-")]:
    ax[0].plot(fr, [r["methods"][name]["detect_rate"] for r in sw], "o" + ls, c=col, ms=4, label=lab, lw=2 if "filter" in name else 1.2)
ax[0].axvline(0.35, c="k", ls=":", lw=.8); ax[0].text(0.355, 0.05, r"$\theta$", fontsize=8)
ax[0].set_xlabel("fraction of nodes affected by true drift"); ax[0].set_ylabel("drift detection rate (60 blocks)"); ax[0].legend(fontsize=6.8, loc="lower right"); ax[0].set_ylim(0, 1.05)
ax[0].set_title("(a) Detection under regional drift", fontsize=8)
mo = Cc["outage_no_drift"]
labels = ["local\nretrain", r"FDC" "\n" r"$\theta$=.35", r"FDC" "\n" r"$\theta$=.45", r"FDC" "\n" r"$\theta$=.55", "FDC\n+filter"]
vals = [mo["node_episodes_pre_per_1000"], mo["methods"]["FDC(th=0.35)"]["wasted_node_retrains_per_1000"], mo["methods"]["FDC(th=0.45)"]["wasted_node_retrains_per_1000"],
        mo["methods"]["FDC(th=0.55)"]["wasted_node_retrains_per_1000"], mo["methods"]["FDC+filter(th=0.35)"]["wasted_node_retrains_per_1000"]]
bars = ax[1].bar(range(5), vals, color=[COL["grey"], COL["orig"], COL["gold"], "#999", COL["corr"]])
for i, v in enumerate(vals): ax[1].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=7)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(labels, fontsize=7); ax[1].set_ylabel("wasted node-retrains per 1,000 blocks")
ax[1].set_title("(b) Cost of false alarms (common-cause outages, no drift)", fontsize=8)
plt.tight_layout(); fig.savefig("figs/fig4_fdc.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 5: Byzantine
fig, ax = plt.subplots(figsize=(4.6, 3.2))
fr = [r["frac"] for r in Dd["rows"]]
for k, col, lab in [("mean", COL["orig"], "mean"), ("trimmed", COL["corr"], r"trimmed mean ($\beta$=0.1)"), ("median", COL["ok"], "coordinate median")]:
    ax.errorbar(fr, [r[f"{k}_sd0.0"][0] for r in Dd["rows"]], yerr=[r[f"{k}_sd0.0"][1] for r in Dd["rows"]], fmt="o-", c=col, ms=4, capsize=2, label=lab)
ax.axvline(0.1, c="k", ls=":", lw=.8); ax.text(0.102, 0.35, r"$\beta K/K$", fontsize=7)
ax.set_xlabel("fraction of sign-flip attackers"); ax.set_ylabel("test accuracy"); ax.legend(fontsize=7); ax.set_ylim(0.25, 0.92)
plt.tight_layout(); fig.savefig("figs/fig5_byz.png", bbox_inches="tight"); plt.close(fig)
print("figures ok")
