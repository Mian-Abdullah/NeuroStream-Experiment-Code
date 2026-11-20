# NeuroStream code guide

This guide explains every module in `ns/` and every script in `exps/`. Put it at `docs/CODE_GUIDE.md` in the repository.

## 1. The big idea

A federation of `K` nodes each watches a stream. Each node raises an **alarm** when its recent data differs from its
older data. A coordinator adds up the weights of alarming nodes (a **vote**) and starts a global retrain when the
weighted share reaches a threshold `θ`.

The failure: a **common cause** (power sag, uplink outage) can make many nodes alarm together, so the vote crosses `θ`
even though the modelled process did not change. NeuroStream fixes this by asking each node, before it votes, "does my
alarm survive after I account for the observed common cause?"

Data flow for one decision step:

```
node data Y, covariate Z
   -> two-window test (raw)                 -> raw alarm
   -> same test after regressing out Z      -> adjusted alarm
   -> vote = raw alarm AND adjusted alarm   (within the last W blocks)
coordinator: v = sum(alpha_i * vote_i) / sum(alpha_i)
   -> v >= theta        : GLOBAL retrain
   -> one node persists : LOCAL retrain
   -> otherwise         : NONE (log the suppressed alarm)
```

## 2. The detector (shared by `sim.py` and `stream.py`)

**Two-window test.** For each node and block, compare the mean of the newest `MB` blocks with the mean of the previous
`MB` blocks. The alarm fires when the difference exceeds a Hoeffding-style cut
`sqrt(log(4/d') / (2m))` (`cut_value` in `sim.py`). `d'` is the per-test error level.

**Covariate adjustment.** Inside the same `2·MB` window, regress `Y` on the covariate `Z` and subtract the part of the
difference explained by the change in `Z`:

```
adjusted difference = (mean_new - mean_old of Y) - beta * (mean_new - mean_old of Z)
beta = Sxy / Sxx   (slope of Y on Z inside the window)
```

This is a backdoor adjustment. It is computed with running sums (`_cs` = cumulative sums), so all nodes and all
windows are processed at once with NumPy arrays and no Python loops over time.

**Vote rule.** A node votes only if it had a raw alarm **and** an adjusted alarm within the last `W` blocks
(`d & adj_recent`).

## 3. Library: `ns/`

### `ns/sim.py`: Track S, the controlled simulation

| Function | What it does |
|---|---|
| `DEFAULT` | All simulation settings: `K=50` nodes, `B=1200` blocks, drift at `t0=600`, outage rate `lam_o`, fraction of nodes exposed to outages `frac_out=0.5`, and so on. |
| `rng_streams(seed)` | One independent random generator per event type (weights, faults, outages, drift, noise, covariate, gate, Byzantine). Because each event type has its own stream, the run **with** drift and the run **without** drift see identical outages and faults. This is what makes the paired counterfactual exact. |
| `generate(cfg, seed, drift, outage)` | Builds one synthetic run: local sensor faults, common-cause outages, genuine drift on a subset of nodes, observation noise, the supply-voltage covariate `Z`, and per-node data volumes. Knobs for covariate realism: noise (`z_sd`), partial observability (`z_obs`), non-linear effect, and a spoofed shared feed. |
| `window_stats` | Computes the raw and the covariate-adjusted statistic for every node and block. Optionally adds Gaussian noise to the sufficient statistics for the private version. |
| `roll_any`, `roll_all`, `episodes` | Helpers: "any alarm in the last w blocks", "all of the last w blocks", and grouping consecutive flags into episodes with a refractory period. |
| `svt_gate` | Sparse-Vector mechanism (Lyu, Su & Li 2017, Alg. 1) for private voting. Threshold noise is drawn once per instance and redrawn only when an instance halts after `c` positives. |
| `sr_edetector` | Shiryaev–Roberts e-detector. For epoch vote fractions `v_t`, `e_t = mean_j exp(λ_j (v_t − p) − λ_j² / (8 K_eff))` and `R_t = (1 + R_{t-1}) · e_t`. It alarms when `R_t ≥ c` and then resets. |
| `weights_of` | Voting weights: `equal`, `volume`, `random_rho`, or `volume_x_calibrated` (data volume times a reliability estimated from the calibration period, never from ground truth). |
| `decision_signals` | For one generated run, computes the per-block global flags for every method (plain FDC and FDC+filter at each θ), the local-scope persistence flags, private-release counts and `K_eff`. Also applies Byzantine voters (flooding or suppression). |
| `epoch_votes` | Collapses alarms into disjoint epochs of length `W` for the e-detector. |
| `run_seed` | The entry point. For one seed it runs the paired counterfactual runs and returns, per method, false-retrain episodes before `t0`, whether the drift was **attributably** detected in `[t0, t0+horizon)` (a declaration that is absent from the paired no-drift run), and the delay. |

Scenarios in `run_seed`: `outage_drift` (main), `faults_drift` (no outages) and `outage_nodrift`.

### `ns/stream.py`: Track A, real streams with injected events

- `setup(...)` loads a dataset, assigns rows to `K` nodes (`data.assign_nodes`), standardises using only the warm-up
  prefix, warm-up trains a softmax model with FedAvg, then injects labelled events:
  - **outage**: for a fixed half of the nodes, a random half of the features get an offset of +3, and a synthetic
    supply covariate `Z = 1 − 0.5·outage + noise` is observed;
  - **drift**: at `t0`, a fraction of nodes get their class labels 0 and 1 swapped for records whose drift feature
    is above its warm-up median.
- `variant_xy` and `error_counts` produce the data and per-block error counts under the injected events.
- `detector` is the stream version of the two-window test, applied to pooled error rates, with a count-weighted
  covariate regression.
- `open_loop` freezes the warm-up model and scores three paired variants: R0 (natural), R1 (+outages),
  R2 (+outages +drift). The difference R1−R0 isolates outage-induced false alarms, and R2−R1 isolates detection of the
  injected drift. It also runs per-node `river` detectors (ADWIN, DDM, Page–Hinkley) as baselines.
- `closed_loop` lets the policy actually trigger FedAvg retraining and measures prequential accuracy. Policies:
  `static`, `periodic`, `oracle`, `local_adwin`, `fdc`, `fdc_filter`, `feddrift` (a re-implementation of FedDrift-C
  with per-client models, isolation and hierarchical merging).

### `ns/fl.py`: federated learning, DP and robustness

- `Softmax`: multinomial logistic regression in NumPy (parameters flattened into one vector).
- `local_update`: full-batch local gradient descent. With `dp=(C, sigma)` it clips every per-example gradient to norm
  `C` and adds Gaussian noise `N(0, (σC)²)`.
- Aggregators: `agg_mean`, `agg_median`, `agg_trimmed`, `agg_krum`, `agg_multikrum`, `agg_cclip` (centered clipping)
  and `agg_fltrust`.
- `attack`: omniscient attackers that see the benign updates: `signflip`, `alie` ("a little is enough"), `ipm`
  (inner-product manipulation) and `minmax`.

### `ns/accounting.py`: privacy accounting

Rényi-DP and zCDP curves for Gaussian releases and pure-DP Sparse-Vector instances, converted to `(ε, δ)`.
`max_units` finds the largest number of decision units (for example retrains) a node can afford before `ε` exceeds
a budget. The functions are cross-checked against Google's `dp_accounting` in `eps_dp_accounting`.
This is what supports the claim that fewer false retrains let the privacy budget last longer.

### `ns/data.py`: data loading

- `load(name)` reads `elec`, `covtype` or `airlines` and checks the SHA-256 of the file first.
- `standardize_from_prefix` standardises with statistics from the warm-up prefix only, so there is no look-ahead.
- `assign_nodes` partitions rows non-IID: Dirichlet label skew (Elec2), one wilderness area per node group
  (Covertype), or the busiest departure airports (Airlines).
- `blocks_of` cuts the stream into `B` equal blocks in time order.

### `ns/stats.py`: uncertainty

Wilson intervals for proportions, bootstrap intervals for means, and an exact Poisson upper confidence bound. The
Poisson bound is what turns "0 false retrains observed" into "95% upper bound 0.026 per 1,000 blocks".

### `ns/runner.py`: experiment runner

`run(name, overrides, seeds, kw, procs, subdir)` runs `sim.run_seed` over many seeds in parallel, summarises the
results with confidence intervals, and writes a JSON file under `results/<subdir>/` that also records the full
configuration, its hash, the seeds, the runtime and the library versions.

## 4. Experiment scripts: `exps/`

| Script | Purpose | Output |
|---|---|---|
| `s1_main.py` | Tunes θ on validation seeds 1000 to 1099, then runs the main comparison on 200 test seeds, the scenarios, the weight ablations and the e-detector. | `results/S/` |
| `s2_sweeps.py` | Threshold curve, federation size, affected fraction, drift magnitude, covariate failure modes, Byzantine voters, the private filter and the Sparse-Vector gate. | `results/S_sweeps/`, `S_fail/`, `S_attack/`, `S_privacy/` |
| `a1_open_loop.py` | Open-loop decision quality on Elec2, Covertype and Airlines. | `results/A/open_*.json` |
| `a2_closed_loop.py` | Closed-loop accuracy and retraining cost of the seven policies. | `results/A/closed_*.json` |
| `c1_cmapss.py` | NASA C-MAPSS with the real operating-condition covariate. | `results/B_real/cmapss.json` |
| `p1_accounting.py` | Deterministic privacy accounting using the event rates measured in Track S. | `results/P/accounting.json` |
| `p2_b1_fl.py` | `p2`: utility of DP training versus noise. `b1`: robust aggregators under attack. | `results/P/`, `results/B/` |
| `t1_theory.py` | Monte-Carlo checks of the theorems, propositions and corollaries. | `results/T/theory.json` |
| `d0_provenance.py` | Row counts, shapes and checksums of the datasets. | `data/PROVENANCE.md` |
| `f_figures.py` | All figures, from result files only. | `figs/` |
| `f_numbers.py` | Every number macro and LaTeX table in the paper, from result files. | `paper/numbers.tex`, `paper/tables/` |
| `f_claims.py` | Maps every number in the paper to its result file and script. | `docs/CLAIMS_TABLE.md` |
| `v_verify.py` | Recomputes headline numbers from per-seed records and compares them with the paper. Exit code 1 on mismatch. | console |
| `legacy/` | The earlier manuscript's scripts, kept for continuity. Only Fig. 2(a) uses its output. | `results/legacy/` |

## 5. Reproducibility design

- **Separate seeds for tuning and testing.** Thresholds are tuned on validation seeds and reported on different test
  seeds, so reported numbers are not tuned on themselves.
- **Paired counterfactual runs.** Per-event random streams make "with drift" and "without drift" runs share every
  other random event.
- **Provenance.** Each result file stores its configuration hash and library versions. Datasets are verified by
  SHA-256.
- **No hand-typed numbers.** The paper's numbers and tables are generated from `results/`, and `v_verify.py` checks them.

## 6. Where to start changing things

| I want to... | Change |
|---|---|
| Try other simulator settings | Pass `overrides` to `runner.run` or edit `sim.DEFAULT` |
| Use my own covariate | Replace `Z` in `generate` (simulation) or in `setup` (real streams) |
| Add a new aggregator | Add a function to `ns/fl.py` and register it in `AGG` |
| Add a new dataset | Add a branch in `data.load` and `data.assign_nodes`, and a SHA-256 entry in `data.SHA` |
| Add a new retraining policy | Add a branch in `stream.closed_loop` and its name to `POLICIES` in `a2_closed_loop.py` |
