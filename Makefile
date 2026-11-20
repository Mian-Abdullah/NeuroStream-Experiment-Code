# Regenerates every result, table, figure and number in the paper.  Approximate times on 2 vCPUs.
PY ?= python3
export NS_DATA ?= $(abspath data/raw)

all: fetch data sim real privacy robust theory paper

fetch:           ## download the four public datasets (pinned mirror commits) into $(NS_DATA) and verify SHA-256
	$(PY) data/fetch_data.py

data:            ## dataset provenance: row counts and checksums -> results/data_provenance.json, data/PROVENANCE.md
	$(PY) exps/d0_provenance.py

sim:             ## Track S: main study, tuning, scenarios, weights (~1 min); sweeps, failure modes, attacks, privacy (~4 min)
	$(PY) exps/s1_main.py
	$(PY) exps/s2_sweeps.py

real:            ## Track A open loop (~15 min), closed loop (~40 min), Track B C-MAPSS (<1 min)
	$(PY) exps/a1_open_loop.py
	$(PY) exps/a2_closed_loop.py
	$(PY) exps/c1_cmapss.py

privacy:         ## accounting (deterministic, <1 min) and DP utility (~30 min)
	$(PY) exps/p1_accounting.py
	$(PY) exps/p2_b1_fl.py p2

robust:          ## robust aggregation under adaptive attacks (~15 min)
	$(PY) exps/p2_b1_fl.py b1

theory:          ## Monte-Carlo validation of Theorems 1 and 4, Proposition 2 and Corollaries 3 and 5 (~1 min)
	$(PY) exps/t1_theory.py

legacy:          ## the revised manuscript's original scripts, unchanged (for continuity)
	cd exps/legacy && mkdir -p results figs && $(PY) exp_a_bound.py && $(PY) exp_b_privacy.py && $(PY) exp_c_sim.py 40 && $(PY) exp_d_byzantine.py

paper:           ## figures, number macros, tables, PDF
	$(PY) exps/f_figures.py
	$(PY) exps/f_numbers.py
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null

.PHONY: all fetch data sim real privacy robust theory legacy paper
