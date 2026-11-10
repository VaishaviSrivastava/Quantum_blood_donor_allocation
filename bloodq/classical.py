"""Classical baselines used to benchmark the hybrid quantum-classical framework (slides 3, 10, 23).

All solvers take a list of recipients `recips` (dict: D, w, p, deadline ...) and candidate (source, recipient)
pairs `cands` (dict: skey, j, cap, time, w) and return {(skey, j): units}.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def greedy(recips, cands):
    """Rule-based greedy: serve recipients by urgency (then deadline), each from its fastest feasible source."""
    rem = {j: r["D"] for j, r in enumerate(recips)}
    cap = {}
    for c in cands:
        cap[c["skey"]] = c["cap"]
    alloc = {}
    for j in sorted(range(len(recips)), key=lambda j: (-recips[j]["w"], recips[j]["deadline"], -recips[j]["D"])):
        for c in sorted([c for c in cands if c["j"] == j], key=lambda c: c["time"]):
            if rem[j] <= 0:
                break
            u = min(rem[j], cap[c["skey"]])
            if u > 0:
                alloc[(c["skey"], j)] = u
                rem[j] -= u
                cap[c["skey"]] -= u
    return alloc


def ilp(recips, cands, lam, mu):
    """Exact optimum of  max sum (w_sj - lam c_sj) y_sj - mu sum p_j u_j  s.t. demand and stock constraints (MILP)."""
    nY, R = len(cands), len(recips)
    nv = nY + R
    c = np.zeros(nv)
    for k, cd in enumerate(cands):
        c[k] = -(cd["w"] - lam * cd["time"] / 60.0)
    for j, r in enumerate(recips):
        c[nY + j] = mu * r["p"]
    A, lo, hi = [], [], []
    for j, r in enumerate(recips):                       # sum_s y_sj + u_j = D_j
        row = np.zeros(nv)
        for k, cd in enumerate(cands):
            if cd["j"] == j:
                row[k] = 1
        row[nY + j] = 1
        A.append(row); lo.append(r["D"]); hi.append(r["D"])
    for s in sorted({cd["skey"] for cd in cands}):       # sum_j y_sj <= A_s
        row = np.zeros(nv)
        cap = 0
        for k, cd in enumerate(cands):
            if cd["skey"] == s:
                row[k] = 1
                cap = cd["cap"]
        A.append(row); lo.append(0); hi.append(cap)
    ub = np.array([min(recips[cd["j"]]["D"], cd["cap"]) for cd in cands] + [r["D"] for r in recips], float)
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(np.zeros(nv), ub))
    if res.x is None:
        return {}
    return {(cd["skey"], cd["j"]): int(round(res.x[k])) for k, cd in enumerate(cands) if round(res.x[k]) > 0}


def exact_qubo(E, F):
    """Brute-force global optimum of the QUBO restricted to constraint-satisfying bit-strings."""
    idx = np.nonzero(F)[0]
    return int(idx[np.argmin(E[idx])])


def simulated_annealing(q, F_fn, rng, restarts=8, sweeps=150):
    """Classical QUBO heuristic: single-bit-flip simulated annealing; returns the best feasible bit-string found."""
    n, Q = q.n, q.Q
    P = q.P
    best_x, best_E = None, np.inf
    T0, T1 = P, 0.02
    for _ in range(restarts):
        x = rng.integers(0, 2, n).astype(float)
        f = Q @ x
        E = float(x @ f) + q.const
        for sw in range(sweeps):
            T = T0 * (T1 / T0) ** (sw / max(sweeps - 1, 1))
            for i in rng.permutation(n):
                dx = 1 - 2 * x[i]
                dE = dx * (Q[i, i] + 2 * (f[i] - Q[i, i] * x[i]))
                if dE <= 0 or rng.random() < np.exp(-dE / T):
                    x[i] += dx
                    f += Q[:, i] * dx
                    E += dE
                    if E < best_E - 1e-9 and F_fn(x):
                        best_E, best_x = E, x.copy()
    return best_x, best_E
