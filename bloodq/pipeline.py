"""End-to-end allocation pipeline (slides 10 and 24).

request batch -> tiered location search (classical preprocessing + Grover oracle search) -> candidate set ->
sub-problem within qubit budget -> QUBO -> QAOA (or a classical solver) -> classical validation/repair ->
commit to inventory -> next round / next batch.   Each method runs on its own copy of the inventory state.
"""
import time
import numpy as np
import pandas as pd

from .config import Params, BLOOD_GROUPS
from .preprocess import World, State, PairSet, preprocess_tier, oracle_truth, DONOR_KEY0, G
from .grover import grover_find
from .qubo import (Subproblem, build_qubo, count_qubits, all_bits, energies, feasible_mask, decode, repair, to_ising)
from .qaoa import QAOA
from .circuits import qaoa_circuit, resource_metrics
from . import classical

METHODS = ["qaoa", "qubo_exact", "qubo_sa", "ilp", "greedy"]
LABELS = {"qaoa": "Hybrid Grover + QAOA", "qubo_exact": "QUBO brute-force (exact)", "qubo_sa": "QUBO simulated annealing",
          "ilp": "ILP (full candidates)", "greedy": "Greedy urgency-first"}


def _cat(sets):
    sets = [s for s in sets if len(s)]
    if not sets:
        return PairSet(*(np.zeros(0) for _ in range(7)))
    return PairSet(*(np.concatenate([getattr(s, f) for s in sets]) for f in
                     ("kind", "skey", "grp", "cap", "time", "j", "tier")))


def _sufficient(feas: PairSet, recips) -> bool:
    for j, r in enumerate(recips):
        m = feas.j == j
        if len(feas) == 0 or feas.cap[m].sum() < r["D"]:
            return False
    return True


# --------------------------------------------------------------------------------------------- search
def tiered_search(world, state, recips, params, engine, rng):
    """Expand through the five location tiers until feasible supply covers the remaining demand (slide 11)."""
    found, stats = [], []
    for tier in (1, 2, 3, 4, 5):
        pairs = preprocess_tier(world, state, recips, tier)
        N = len(pairs)
        if N == 0:
            continue
        truth = oracle_truth(world, state, pairs, recips)
        t0 = time.perf_counter()
        if engine == "grover":
            enough = None
            if params.grover_mode == "enough":
                enough = lambda f, pairs=pairs: _sufficient(_cat(found + [pairs.take(f)]), recips)
            res = grover_find(truth, rng, enough=enough)
            sel, queries, nq = np.array(res["found"], int), res["queries"], res["n_qubits"]
            p_min = min(res["p_success"]) if res["p_success"] else 1.0
        else:
            sel, queries, nq, p_min = np.nonzero(truth)[0], N, 0, 1.0
        stats.append(dict(tier=tier, N=N, M=int(truth.sum()), found=len(sel), oracle_queries=queries,
                          classical_evals=N, grover_qubits=nq, min_p_success=p_min,
                          complete=bool(len(sel) == truth.sum()), seconds=time.perf_counter() - t0))
        if len(sel):
            found.append(pairs.take(sel))
        if _sufficient(_cat(found), recips):
            break
    return _cat(found), stats


# --------------------------------------------------------------------------------------------- sub-problem
def _cand_list(world, feas: PairSet, recips, params):
    out = []
    for k in range(len(feas)):
        j = int(feas.j[k])
        exact = bool(world.exact[feas.grp[k], recips[j]["g"]])
        out.append(dict(skey=int(feas.skey[k]), j=j, cap=int(feas.cap[k]), time=float(feas.time[k]),
                        tier=int(feas.tier[k]), kind=int(feas.kind[k]), exact=exact,
                        w=recips[j]["w"] * (1.0 if exact else params.mismatch_factor)))
    return out


def _make_sub(cands, recips, skeys):
    chosen = {k: i for i, k in enumerate(skeys)}
    used = sorted({c["j"] for c in cands if c["skey"] in chosen})
    jmap = {j: i for i, j in enumerate(used)}
    sources = []
    for k in skeys:
        c = next(c for c in cands if c["skey"] == k)
        sources.append(dict(key=k, cap=c["cap"], tier=c["tier"]))
    pairs = {(chosen[c["skey"]], jmap[c["j"]]): dict(w=c["w"], time=c["time"], exact=c["exact"])
             for c in cands if c["skey"] in chosen}
    rs = [dict(rid=recips[j]["rid"], D=recips[j]["D"], w=recips[j]["w"], p=recips[j]["p"], orig=j) for j in used]
    return Subproblem(rs, sources, pairs)


def build_subproblem(cands, recips, params):
    """Pick sources (round-robin over recipients by urgency, nearest/lowest tier first) until the qubit budget is used."""
    per = {}
    for j in sorted(range(len(recips)), key=lambda j: -recips[j]["w"]):
        lst = sorted([c for c in cands if c["j"] == j], key=lambda c: (c["tier"], c["time"]))
        if lst:
            per[j] = [c["skey"] for c in lst]
    order, seen, ptr = [], set(), {j: 0 for j in per}
    while any(ptr[j] < len(per[j]) for j in per):
        for j in per:
            while ptr[j] < len(per[j]) and per[j][ptr[j]] in seen:
                ptr[j] += 1
            if ptr[j] < len(per[j]):
                order.append(per[j][ptr[j]])
                seen.add(per[j][ptr[j]])
    chosen = []
    cap_of = {c["skey"]: c["cap"] for c in cands}
    demand = sum(recips[j]["D"] for j in per)
    for k in order:
        trial = _make_sub(cands, recips, chosen + [k])
        if chosen and count_qubits(trial) > params.max_qubits:
            continue
        chosen.append(k)
        if sum(cap_of[s] for s in chosen) >= demand and len(chosen) >= params.min_sources:
            break
    return _make_sub(cands, recips, chosen)


# --------------------------------------------------------------------------------------------- solvers
def solve_qubo_method(method, sub, params, rng, circuit_metrics=True):
    q = build_qubo(sub, params.lam, params.mu, params.penalty_scale)
    X = all_bits(q.n)
    E = energies(q, X)
    F = feasible_mask(q, X)
    ex = classical.exact_qubo(E, F)
    E_opt = float(E[ex])
    info = dict(n_qubits=q.n, n_sources=len(sub.sources), n_recips=len(sub.recips), E_opt=E_opt,
                n_opt_states=int((F & (E <= E_opt + 1e-9)).sum()),
                penalty_ok=bool(E.min() >= E_opt - 1e-9), P=q.P)
    if method == "qubo_exact":
        x = X[ex]
        info.update(E_best=E_opt, top_infeasible=False, shot_violation=0.0)
    elif method == "qubo_sa":
        x, Eb = classical.simulated_annealing(q, lambda v: bool(np.all(q.A_eq @ v + q.b_eq == 0)), rng,
                                              params.sa_restarts, params.sa_sweeps)
        if x is None:
            x, Eb = np.zeros(q.n), np.inf
        info.update(E_best=Eb, top_infeasible=False, shot_violation=0.0)
    else:   # qaoa
        t0 = time.perf_counter()
        qa = QAOA(E, q.n, params.p_layers, params.cvar_alpha, Q=q.Q)
        th, fun = qa.optimize(params.restarts, params.maxiter, rng)
        idx, counts = qa.sample(th, params.shots, rng)
        fs = F[idx]
        pr = qa.probs(th)
        info.update(
            qaoa_seconds=time.perf_counter() - t0, qaoa_evals=qa.evals,
            p_opt=float(pr[F & (E <= E_opt + 1e-9)].sum()), p_feasible_exact=float(pr[F].sum()),
            shot_violation=float(1 - counts[fs].sum() / counts.sum()),
            top_infeasible=bool(not F[idx[np.argmin(E[idx])]]),
            exp_E=float(pr @ E), E_uniform=float(E.mean()),
            improvement=float((E.mean() - pr @ E) / (E.mean() - E_opt)))
        if fs.any():
            sel = idx[fs][np.argmin(E[idx][fs])]          # best feasible sampled bit-string (slide 21 step 7)
        else:
            sel = idx[np.argmin(E[idx])]
        x = X[sel]
        info["E_best"] = float(E[sel]) if fs.any() else np.inf
        if circuit_metrics:
            h, J, _ = to_ising(q)
            info.update(resource_metrics(qaoa_circuit(h, J, th[:params.p_layers], th[params.p_layers:], qa.scale)))
        info["_theta"] = th
        info["_samples"] = (idx.copy(), counts.copy())
    info["_x"] = np.asarray(x).copy()
    y = decode(q, x)
    feas = bool(np.all(q.A_eq @ x + q.b_eq == 0)) if len(x) else False
    if not feas:
        y = repair(q, sub, y)
    info["repaired"] = not feas
    info["gap_to_opt"] = (info["E_best"] - E_opt) if np.isfinite(info.get("E_best", np.inf)) else np.nan
    info["_q"], info["_E"], info["_F"] = q, E, F
    alloc = {(sub.sources[s]["key"], sub.recips[j]["orig"]): u for (s, j), u in y.items() if u > 0}
    return alloc, info


def solve_classical(method, recips, cands, params):
    if method == "greedy":
        return classical.greedy(recips, cands)
    return classical.ilp(recips, cands, params.lam, params.mu)


# --------------------------------------------------------------------------------------------- main loop
def commit(state, alloc):
    for (skey, _), u in alloc.items():
        if skey < DONOR_KEY0:
            h, g = divmod(skey, G)
            assert state.inv[h, g] >= u, "over-allocation of stock"
            state.inv[h, g] -= u
        else:
            assert u == 1 and not state.donor_used[skey - DONOR_KEY0], "donor reused"
            state.donor_used[skey - DONOR_KEY0] = True


def run_method(world: World, params: Params, method: str, keep_subs: int = 0, circuit_metrics=True, verbose=False):
    rng = np.random.default_rng(params.seed + 7)
    state = State(world)
    engine = "grover" if method == "qaoa" else "classical"
    req = world.requests
    alloc_rows, sub_rows, search_rows, kept = [], [], [], []
    t_start = time.perf_counter()
    t_search = t_solve = 0.0
    B = params.batch_size
    for b0 in range(0, len(req), B):
        idxs = list(range(b0, min(b0 + B, len(req))))
        recs = [dict(rid=req.request_id[i], h=int(world.req_h[i]), g=int(world.req_g[i]),
                     deadline=float(req.deadline_min[i]), w=int(req.urgency_weight[i]), p=int(req.urgency_weight[i]),
                     Dtot=int(req.units_requested[i]), rem=int(req.units_requested[i]), dead=False) for i in idxs]
        for rnd in range(params.max_rounds):
            act = [r for r in recs if r["rem"] > 0 and not r["dead"]]
            if not act:
                break
            recips = [dict(r, D=min(r["rem"], params.max_units_per_round)) for r in act]
            t0 = time.perf_counter()
            feas, stats = tiered_search(world, state, recips, params, engine, rng)
            t_search += time.perf_counter() - t0
            for s in stats:
                search_rows.append(dict(method=method, batch=b0 // B, round=rnd, **s))
            if len(feas) == 0:
                for r in act:
                    r["dead"] = True
                break
            cands = _cand_list(world, feas, recips, params)
            for j, r in enumerate(act):
                if not any(c["j"] == j for c in cands):
                    r["dead"] = True
            t0 = time.perf_counter()
            if method in ("greedy", "ilp"):
                a = solve_classical(method, recips, cands, params)
                info = {}
            else:
                sub = build_subproblem(cands, recips, params)
                a, info = solve_qubo_method(method, sub, params, rng, circuit_metrics)
                if keep_subs and info["n_qubits"] <= 12 and len(kept) < keep_subs:
                    kept.append(dict(sub=sub, info={k: v for k, v in info.items() if not k.startswith("_")},
                                     q=info["_q"], E=info["_E"], F=info["_F"], theta=info.get("_theta")))
            dt = time.perf_counter() - t0
            t_solve += dt
            # translate local alloc {(skey, j_local_in_act)} -> request ids, commit
            by_cand = {(c["skey"], c["j"]): c for c in cands}
            full = {}
            for (skey, j), u in a.items():
                full[(skey, j)] = u
            commit(state, {(k, act[j]["rid"]): u for (k, j), u in full.items()})
            got = 0
            for (skey, j), u in full.items():
                c = by_cand[(skey, j)]
                act[j]["rem"] -= u
                got += u
                alloc_rows.append(dict(method=method, rid=act[j]["rid"], skey=skey, kind=c["kind"], tier=c["tier"],
                                       units=u, time_min=c["time"], w_sj=c["w"], round=rnd, deadline=act[j]["deadline"]))
            if info:
                sub_rows.append(dict(method=method, batch=b0 // B, round=rnd, solve_seconds=dt,
                                     **{k: v for k, v in info.items() if not k.startswith("_")}))
            if got == 0:
                break
            if verbose:
                print(f"  [{method}] batch {b0 // B} round {rnd}: +{got} units", flush=True)
    total = time.perf_counter() - t_start
    alloc = pd.DataFrame(alloc_rows)
    subs = pd.DataFrame(sub_rows)
    search = pd.DataFrame(search_rows)
    summary = summarize(world, params, method, alloc, subs, search, total, t_search, t_solve)
    return dict(summary=summary, alloc=alloc, subs=subs, search=search, kept=kept)


def summarize(world, params, method, alloc, subs, search, total, t_search, t_solve):
    req = world.requests
    if len(alloc):
        ful = alloc.groupby("rid").units.sum()
    else:
        ful = pd.Series(dtype=int)
    got = req.request_id.map(ful).fillna(0).astype(int)
    unmet = req.units_requested - got
    w = req.urgency_weight
    H_obj = float(((alloc.w_sj * alloc.units).sum() - params.lam * (alloc.time_min / 60 * alloc.units).sum()) if len(alloc) else 0.0)
    H_obj -= params.mu * float((w * unmet).sum())
    s = dict(method=method, label=LABELS[method],
             units_requested=int(req.units_requested.sum()), units_fulfilled=int(got.sum()),
             fulfillment_rate=float(got.sum() / req.units_requested.sum()),
             urgency_weighted_fulfillment=float((w * got).sum() / (w * req.units_requested).sum()),
             high_urgency_unmet_units=int(unmet[req.urgency == "High"].sum()),
             unmet_units=int(unmet.sum()),
             requests_fully_met=int((unmet == 0).sum()),
             total_delivery_time_min=float((alloc.time_min * alloc.units).sum()) if len(alloc) else 0.0,
             mean_delivery_time_per_unit_min=float((alloc.time_min * alloc.units).sum() / max(alloc.units.sum(), 1)) if len(alloc) else 0.0,
             deadline_violations=int((alloc.time_min > alloc.deadline).sum()) if len(alloc) else 0,
             objective_H=H_obj, runtime_s=total, search_s=t_search, solve_s=t_solve)
    if len(subs):
        s.update(n_subproblems=len(subs), mean_qubits=float(subs.n_qubits.mean()), max_qubits=int(subs.n_qubits.max()),
                 penalty_ok_rate=float(subs.penalty_ok.mean()),
                 constraint_violation_rate=float(subs.top_infeasible.mean()),
                 repaired_rate=float(subs.repaired.mean()),
                 mean_gap_to_qubo_opt=float(subs.gap_to_opt.mean(skipna=True)),
                 optimal_hit_rate=float((subs.gap_to_opt.abs() < 1e-6).mean()))
        if "depth" in subs:
            s.update(mean_depth=float(subs.depth.mean()), max_depth=int(subs.depth.max()),
                     mean_cx=float(subs.cx.mean()), mean_p_opt=float(subs.p_opt.mean()),
                     mean_shot_violation=float(subs.shot_violation.mean()),
                     mean_improvement_over_uniform=float(subs.improvement.mean()))
    else:
        s.update(constraint_violation_rate=0.0)
    if len(search):
        s.update(oracle_queries=int(search.oracle_queries.sum()), classical_evals=int(search.classical_evals.sum()),
                 search_complete_rate=float(search.complete.mean()))
    return s
