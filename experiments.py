"""Run every experiment and write CSVs to results/.   Usage:  python experiments.py [--skip-sweep]"""
import argparse
import math
import os
import time
import numpy as np
import pandas as pd

from bloodq.config import Params
from bloodq.data_gen import generate_all
from bloodq.preprocess import World
from bloodq.pipeline import run_method, METHODS
from bloodq.grover import optimal_iterations, grover_probs, bbht
from bloodq.qaoa import QAOA
from bloodq.qubo import to_ising
from bloodq.circuits import grover_circuit, qaoa_circuit, resource_metrics, noisy_counts

RES = "results"
os.makedirs(RES, exist_ok=True)


def main_comparison(params):
    world = World("data", params)
    rows, kept = [], []
    for m in METHODS:
        t = time.time()
        r = run_method(world, params, m, keep_subs=40 if m == "qaoa" else 0)
        rows.append(r["summary"])
        r["alloc"].to_csv(f"{RES}/alloc_{m}.csv", index=False)
        if len(r["subs"]):
            r["subs"].to_csv(f"{RES}/subproblems_{m}.csv", index=False)
        r["search"].to_csv(f"{RES}/search_{m}.csv", index=False)
        if m == "qaoa":
            kept = r["kept"]
            subs = r["subs"].copy()
            subs["hit_prob_qaoa"] = 1 - (1 - subs.p_opt) ** params.shots
            subs["hit_prob_random"] = 1 - (1 - subs.n_opt_states / 2.0 ** subs.n_qubits) ** params.shots
            subs["p_opt_uniform"] = subs.n_opt_states / 2.0 ** subs.n_qubits
            subs.to_csv(f"{RES}/subproblems_qaoa.csv", index=False)
        print(f"  {m:11s} done in {time.time() - t:5.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/system_comparison.csv", index=False)
    return kept


def scarcity_sweep(base_params):
    out = []
    for base in (4.5, 3.0, 2.0, 1.2):
        for seed in (1, 2, 3):
            p = Params(**{**base_params.__dict__, "inventory_base": base, "seed": seed})
            d = f"{RES}/tmp_data/b{base}_s{seed}"
            generate_all(p, d)
            w = World(d, p)
            for m in METHODS:
                s = run_method(w, p, m, circuit_metrics=False)["summary"]
                s.update(inventory_base=base, seed=seed)
                out.append(s)
            print(f"  scarcity base={base} seed={seed} done", flush=True)
    pd.DataFrame(out).to_csv(f"{RES}/scarcity_sweep.csv", index=False)


def grover_scaling(rng):
    rows = []
    for n in range(4, 17):                                   # M = 1
        N = 1 << n
        r = optimal_iterations(N, 1)
        mask = np.zeros(N, bool)
        mask[N // 3] = True
        rows.append(dict(kind="n_scaling", n=n, N=N, M=1, iterations=r, grover_queries=r + 1,
                         classical_expected=(N + 1) / 2, classical_worst=N,
                         p_success=float(grover_probs(n, mask, r)[mask].sum())))
    n = 10
    N = 1 << n
    for M in (1, 2, 4, 8, 16, 32, 64, 128, 256):            # fixed N, growing M: where does Grover stop paying off?
        mask = np.zeros(N, bool)
        mask[rng.choice(N, M, replace=False)] = True
        r = optimal_iterations(N, M)
        q_bbht = [bbht(mask, rng)[1] for _ in range(60)]
        rows.append(dict(kind="M_scaling", n=n, N=N, M=M, iterations=r, grover_queries=r + 1,
                         classical_expected=(N + 1) / (M + 1), classical_worst=N - M + 1,
                         p_success=float(grover_probs(n, mask, r)[mask].sum()), bbht_mean_queries=float(np.mean(q_bbht))))
    pd.DataFrame(rows).to_csv(f"{RES}/grover_scaling.csv", index=False)

    res = []
    for n in range(3, 9):                                    # transpiled circuit resources, M = 1
        r = optimal_iterations(1 << n, 1)
        res.append(dict(n=n, iterations=r, **resource_metrics(grover_circuit(n, [1], r))))
    pd.DataFrame(res).to_csv(f"{RES}/grover_resources.csv", index=False)


def qaoa_depth_sweep(kept, params, rng):
    rows = []
    cands = sorted(kept, key=lambda k: -k["q"].n)
    seen, chosen = set(), []
    for k in cands:                                          # three largest distinct-size instances
        if k["q"].n not in seen:
            seen.add(k["q"].n)
            chosen.append(k)
        if len(chosen) == 3:
            break
    for k in chosen:
        q, E, F = k["q"], k["E"], k["F"]
        E_opt = float(E[F].min())
        n_opt = int((F & (E <= E_opt + 1e-9)).sum())
        h, J, _ = to_ising(q)
        for p in range(1, 6):
            qa = QAOA(E, q.n, p, Q=q.Q)
            th, _ = qa.optimize(params.restarts, params.maxiter, rng)
            pr = qa.probs(th)
            res = resource_metrics(qaoa_circuit(h, J, th[:p], th[p:], qa.scale))
            rows.append(dict(n_qubits=q.n, p=p, p_opt=float(pr[F & (E <= E_opt + 1e-9)].sum()),
                             p_opt_uniform=n_opt / 2.0 ** q.n, p_feasible=float(pr[F].sum()),
                             improvement=float((E.mean() - pr @ E) / (E.mean() - E_opt)),
                             evals=qa.evals, **res))
        print(f"  depth sweep n={q.n} done", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/qaoa_depth_sweep.csv", index=False)


def noise_study(kept, params):
    pl = [0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    rows = []
    n, marked = 5, 11                                        # Grover: N=32, M=1
    r = optimal_iterations(1 << n, 1)
    qc = grover_circuit(n, [marked], r)
    for p1 in pl:
        c = noisy_counts(qc, p1, 2000)
        rows.append(dict(algo="Grover", n_qubits=n, p1=p1, success=c.get(format(marked, f"0{n}b"), 0) / 2000,
                         **resource_metrics(qc)))
        print(f"  noise grover p={p1}", flush=True)
    pick = [k for k in kept if 8 <= k["q"].n <= 10]
    if pick:
        k = max(pick, key=lambda k: k["q"].n)
        q, E, F, th = k["q"], k["E"], k["F"], k["theta"]
        E_opt = float(E[F].min())
        opt_set = {i for i in np.nonzero(F & (E <= E_opt + 1e-9))[0]}
        h, J, _ = to_ising(q)
        qa = QAOA(E, q.n, params.p_layers, Q=q.Q)
        qc = qaoa_circuit(h, J, th[:params.p_layers], th[params.p_layers:], qa.scale)
        met = resource_metrics(qc)
        for p1 in pl:
            c = noisy_counts(qc, p1, 2000)
            tot = sum(c.values())
            idx = np.array([int(b, 2) for b in c])
            cnt = np.array(list(c.values()))
            rows.append(dict(algo="QAOA", n_qubits=q.n, p1=p1,
                             success=sum(v for b, v in c.items() if int(b, 2) in opt_set) / tot,
                             p_feasible=float(cnt[F[idx]].sum() / tot),
                             improvement=float((E.mean() - (cnt * E[idx]).sum() / tot) / (E.mean() - E_opt)),
                             uniform_success=len(opt_set) / 2.0 ** q.n, **met))
            print(f"  noise qaoa p={p1}", flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/noise_study.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweep", action="store_true")
    a = ap.parse_args()
    params = Params()
    generate_all(params, "data")
    rng = np.random.default_rng(params.seed)
    print("[1/5] main comparison", flush=True)
    kept = main_comparison(params)
    print("[2/5] grover scaling", flush=True)
    grover_scaling(rng)
    print("[3/5] qaoa depth sweep", flush=True)
    qaoa_depth_sweep(kept, params, rng)
    print("[4/5] noise study", flush=True)
    noise_study(kept, params)
    if not a.skip_sweep:
        print("[5/5] scarcity sweep", flush=True)
        scarcity_sweep(params)
    print("done")
