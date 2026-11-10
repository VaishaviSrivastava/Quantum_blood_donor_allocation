# Hybrid Quantum-Classical Blood Donor-Recipient Matching (Grover + QAOA)

Python implementation of the MTE 2026 project deck. Synthetic data and quantum circuits are simulated with Qiskit and Qiskit Aer; nothing here is clinical software.

## Run it
```bash
pip install -r requirements.txt
python validate.py          # QUBO/Ising and Qiskit circuit checks
python experiments.py       # generates data/ + all results/*.csv  (~7 min; --skip-sweep for ~1 min)
python make_figures.py      # results/figures/*.png
python test_dynamic.py      # 8 tests for the time-dynamic engine
streamlit run app.py        # interactive console (opens in the browser)
```

## Slide -> code map
| Slide | Code |
|---|---|
| 9 five datasets | `bloodq/data_gen.py` -> `data/` donors, compatibility, requests, inventory, hospitals + distances |
| 10-11 preprocessing, 5 location tiers | `bloodq/preprocess.py` (`preprocess_tier`), `pipeline.tiered_search` |
| 12-14 Grover, oracle = compatibility AND eligibility AND availability AND deadline | `preprocess.oracle_truth`, `bloodq/grover.py`, `bloodq/circuits.py` |
| 15-20 objective, constraints, QUBO, binary encoding | `bloodq/qubo.py` |
| 21 QAOA | `bloodq/qaoa.py`, `bloodq/circuits.py` (Qiskit Statevector + Aer circuit sampling) |
| 22, 24 hybrid loop, real-time updates | `bloodq/pipeline.py` |
| 23 metrics | `pipeline.summarize` -> `results/system_comparison.csv` |
| classical comparison | `bloodq/classical.py` (greedy, ILP, brute-force QUBO, simulated annealing) |

## Quantum implementation boundary
- Grover amplitude amplification is now executed as a Qiskit `QuantumCircuit` and sampled with Qiskit Aer. The current data pipeline still computes the eligible-index mask classically (`preprocess.oracle_truth`) and uses those indices to construct the phase oracle. This is a circuit-based Grover simulation, but not yet a reversible circuit for live blood-group/stock/distance arithmetic and does not claim end-to-end quantum speedup.
- QAOA layers are constructed as Qiskit circuits. Qiskit `Statevector` evaluates ideal expectation values during classical COBYLA parameter optimisation, and Qiskit Aer performs finite-shot sampling. The COBYLA optimiser remains classical, as in standard hybrid QAOA.
- QUBO remains a classical mathematical formulation in `bloodq/qubo.py`; its binary variables, penalties and matrix are passed to the Qiskit QAOA circuit through the QUBO-to-Ising mapping. The exact/SA/ILP baselines and feasibility repair remain classical.

## Pipeline
Requests arrive in order and are solved in batches of 2. Per round (<=3 units per recipient, <=4 rounds):
1. **Tiered search** (local bank -> nearby banks -> nearby donors -> distant banks -> distant donors), expanding only
   while feasible supply is insufficient. Each tier: classical pruning, then **Grover** finds all oracle-marked pairs.
2. A **sub-problem** is cut to a qubit budget (14), the **QUBO** is built (slack bits for stock, binary-encoded unmet units).
3. **QAOA** (p=2, COBYLA, 3 restarts) is evaluated with Qiskit Statevector during ideal expectation optimisation, then sampled with Qiskit Aer for 4096 shots.
4. **Classical validation**: best *feasible* sampled bit-string is chosen (repair fallback), inventory/donors are updated.

## Previously bundled baseline results (seed 42, 36 requests, 95 units)

The CSVs and figures included in `results/` are retained from the original NumPy-simulator version for reference. They are **not** results from the upgraded Qiskit execution path. Run `python experiments.py` to regenerate all result files with the Qiskit implementation.


| Method | Units met | Urgency-wtd | High-urg. unmet | Delivery time (min) | Objective H |
|---|---|---|---|---|---|
| Hybrid Grover + QAOA | 95 | 1.000 | 0 | 1389 | 280.1 |
| QUBO exact (brute force) | 95 | 1.000 | 0 | 1603 | 278.7 |
| QUBO simulated annealing | 95 | 1.000 | 0 | 1986 | 277.8 |
| ILP (all candidates) | 95 | 1.000 | 0 | 1207 | **282.0** |
| Greedy urgency-first | 94 | 0.983 | 1 | 2288 | 262.1 |

Zero deadline violations, zero constraint-violating outputs, penalty sizing kept the QUBO optimum feasible in every
sub-problem. A 12-run scarcity sweep (4 stock levels x 3 data seeds, `results/scarcity_sweep.csv`) shows the same pattern.
Rows are not strictly comparable on H because each method's inventory evolves differently.

## What the experiments honestly show
- **Solution quality**: QAOA matched the exact QUBO optimum in 100% of sub-problems, but sub-problems are small
  (mean 7.7, max 14 qubits). At that size random sampling also often hits the optimum (mean 92%, worst case 22% vs
  QAOA mean 99.8%, worst 96% within 4096 shots). QAOA concentrates probability on the optimum, but p>2 gave no
  clear gain (`fig5`) and a classical ILP is as good or better and ~200x faster in simulation.
- **Grover**: query gap is quadratic for sparse matches (M=1, N=1024: ~26 vs ~512 queries) but vanishes as M/N grows.
  In the real run it cost 738 queries vs 901 classical scans overall: worse for the dense local tier (191 vs 157),
  better for the donor tier (302 vs 493). Assumes M known (counting cost not charged); unknown M costs more (BBHT, `fig3`).
- **Hardware realism**: the transpiled Grover circuit has depth 627 at only 5 qubits (depth 3940 at 8); with
  depolarising noise its success falls from ~1.0 to ~0.3 at 5e-4 one-qubit error and to chance by 2e-3. QAOA
  (depth 58, 10 qubits) degrades more gracefully (`fig6`). Both are far below today's hardware needs.
- Runtime numbers are classical-simulation times, not quantum-hardware times; no speedup is claimed.

## Notes / corrections to the slides
- **Slide 18 arithmetic**: for lambda=0.1, c=10, y=2 the cost is 0.1x10x2 = **2** (slide: 1); for c=40 it is **8** (slide: 4).
  The slide values are per-unit costs. The code uses the correct formula.
- c_sj is delivery time in **hours** (not the slide's 10/40 toy values); donor sources add 120 min collection/testing.
- w_sj = urgency weight (5/3/1) x 0.9 for non-identical compatible groups (conserves O-); p_j = urgency weight, mu=1, lambda=0.1.
- Penalties P_D = P_A = 1.2 (w_max + mu p_max).
- Simplifications: donor eligibility rules are simplified; donors count as single units; the tier eligibility mask is still computed classically and supplied as marked indices to the Qiskit phase-oracle circuit; QAOA uses ideal Qiskit Statevector expectations during optimisation.

## Extension ideas
Aggregate donors into pools to cut qubits, CVaR objective (`cvar_alpha`), Grover `enough` early-stop mode
(`grover_mode`), warm-started QAOA, larger batches with a sparse/MPS simulator.

## Time dynamics and the Streamlit console (new)

`bloodq/dynamic.py` adds a simulation clock on top of the same search -> QUBO -> QAOA -> validation machinery:

| Feature | How it works |
|---|---|
| Restocking | every `restock_every_min` each hospital receives a random delivery of new units |
| Expiry | stock is a list of lots with expiry times, issued FIFO (oldest first); expired units are discarded and counted |
| Donors online later | a share of the offline eligible donors becomes available at random times |
| Failed deliveries | donor no-show or shipment failure; units go back to the request (and to the bank if the shipment failed), a no-show donor is dropped, and the request is **re-planned** with the time left |
| Live requests | submit a request at any moment; deadlines run on the simulation clock |
| Urgent-first batching | the most urgent waiting requests are batched first (addresses the limitation of arrival-order batches) |

The environment (restocks, shelf life, donor arrivals, failure outcomes) is fixed by the seed and independent of the method, so all
methods face the identical world. `check_invariants()` verifies that units, stock, donors and deadlines balance; `test_dynamic.py` tests
expiry, restock, failure re-planning, priority batching, determinism and the QAOA path.

`app.py` (Streamlit) tabs: **Live console** (step the clock, submit requests, force a failure, restock, stock heatmap, event log, and a
*decision trace* showing the Grover tier search, candidate sources, QUBO size and the chosen allocation), **Full run** (whole-horizon analysis,
request timeline, response times, CSV downloads), **Compare methods** (all solvers on the same worlds, mean +- sd over seeds),
**Datasets**, **Static results**, **How it works**.

Remaining limits: simulated data and circuits; failure/expiry rates are assumptions you set in the sidebar, not measured values; donors give one unit;
a request that finds no supply is retried only when supply changes (restock, donor online, returned units).
