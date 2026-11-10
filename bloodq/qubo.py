"""QUBO formulation (slides 15-20).

    min Q = - sum w_sj y_sj + lam sum c_sj y_sj + mu sum p_j u_j
            + P_D sum_j ( sum_s y_sj + u_j - D_j )^2 + P_A sum_s ( sum_j y_sj + z_s - A_s )^2

Integer variables are binary-encoded exactly as on slide 20:  y = b1 + 2 b2 + 4 b3 ...
c_sj is the delivery time in HOURS.  Equality constraints are kept as a matrix (A_eq x + b_eq = 0) so that every
bit-string can be tested for feasibility, and so the penalty terms and the feasibility check cannot disagree.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Subproblem:
    recips: list                 # dicts: rid, D (tranche), w (urgency weight), p (priority)
    sources: list                # dicts: key, cap, tier
    pairs: dict                  # (s, j) -> dict(w, time, exact)
    meta: dict = field(default_factory=dict)


@dataclass
class QUBO:
    Q: np.ndarray                # symmetric, E(x) = x^T Q x + const
    const: float
    A_eq: np.ndarray             # rows: constraint vectors
    b_eq: np.ndarray             # constraint constants  (A_eq x + b_eq == 0 when satisfied)
    ylist: dict                  # (s, j) -> [(bit_index, weight)]
    ulist: dict
    zlist: dict
    P: float
    n: int


def _layout(sub: Subproblem):
    idx = 0
    ylist, ulist, zlist = {}, {}, {}
    for (s, j) in sorted(sub.pairs):
        cap = min(sub.recips[j]["D"], sub.sources[s]["cap"])
        nb = int(cap).bit_length()
        ylist[(s, j)] = [(idx + k, 2 ** k) for k in range(nb)]
        idx += nb
    for j, r in enumerate(sub.recips):
        nb = int(r["D"]).bit_length()
        ulist[j] = [(idx + k, 2 ** k) for k in range(nb)]
        idx += nb
    for s, src in enumerate(sub.sources):
        max_rep = sum(sum(w for _, w in ylist[(s, j)]) for j in range(len(sub.recips)) if (s, j) in ylist)
        if max_rep > src["cap"]:                       # supply constraint can actually bind -> slack bits
            nb = int(src["cap"]).bit_length()
            zlist[s] = [(idx + k, 2 ** k) for k in range(nb)]
            idx += nb
    return ylist, ulist, zlist, idx


def count_qubits(sub: Subproblem) -> int:
    return _layout(sub)[3]


def build_qubo(sub: Subproblem, lam: float, mu: float, penalty_scale: float) -> QUBO:
    ylist, ulist, zlist, n = _layout(sub)
    R, S = len(sub.recips), len(sub.sources)
    w_max = max(v["w"] for v in sub.pairs.values())
    p_max = max(r["p"] for r in sub.recips)
    P = penalty_scale * (w_max + mu * p_max)        # large enough that violating a constraint never pays

    lin = np.zeros(n)
    for (s, j), bits in ylist.items():
        pr = sub.pairs[(s, j)]
        coef = -pr["w"] + lam * pr["time"] / 60.0   # urgency reward (neg.) + travel cost (hours)
        for i, wt in bits:
            lin[i] += coef * wt
    for j, bits in ulist.items():
        for i, wt in bits:
            lin[i] += mu * sub.recips[j]["p"] * wt

    rows, consts = [], []
    for j in range(R):                               # demand constraint
        a = np.zeros(n)
        for s in range(S):
            for i, wt in ylist.get((s, j), []):
                a[i] += wt
        for i, wt in ulist[j]:
            a[i] += wt
        rows.append(a)
        consts.append(-float(sub.recips[j]["D"]))
    for s, bits in zlist.items():                    # supply constraint (only where it can bind)
        a = np.zeros(n)
        for j in range(R):
            for i, wt in ylist.get((s, j), []):
                a[i] += wt
        for i, wt in bits:
            a[i] += wt
        rows.append(a)
        consts.append(-float(sub.sources[s]["cap"]))
    A, b = np.array(rows), np.array(consts)

    Q = np.zeros((n, n))
    const = 0.0
    for a, k in zip(A, b):                            # P * (a.x + k)^2  with x_i^2 = x_i
        Q += P * np.outer(a, a)
        lin += P * 2.0 * k * a
        const += P * k * k
    Q[np.diag_indices(n)] += lin
    return QUBO(Q, const, A, b, ylist, ulist, zlist, P, n)


def all_bits(n: int) -> np.ndarray:
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int8)


def energies(q: QUBO, X: np.ndarray) -> np.ndarray:
    Xf = X.astype(float)
    return np.einsum("ij,ij->i", Xf @ q.Q, Xf) + q.const


def feasible_mask(q: QUBO, X: np.ndarray) -> np.ndarray:
    return np.all(X.astype(float) @ q.A_eq.T + q.b_eq == 0, axis=1)


def decode(q: QUBO, x: np.ndarray) -> dict:
    """(s, j) -> units allocated"""
    return {sj: int(sum(wt * int(x[i]) for i, wt in bits)) for sj, bits in q.ylist.items()}


def to_ising(q: QUBO):
    """x_i = (1 - Z_i)/2  ->  H = sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j + offset."""
    n, Q = q.n, q.Q
    h = -np.diag(Q) / 2.0
    J = {}
    off = q.const + np.trace(Q) / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            qij = 2 * Q[i, j]                       # coefficient of x_i x_j
            if qij != 0.0:
                J[(i, j)] = qij / 4.0
                h[i] -= qij / 4.0
                h[j] -= qij / 4.0
                off += qij / 4.0
    return h, J, off


def repair(q: QUBO, sub: Subproblem, y: dict) -> dict:
    """Classical post-processing: clip an infeasible decoded allocation to a feasible one."""
    y = dict(y)
    for j, r in enumerate(sub.recips):                       # never exceed the recipient's demand
        tot = sum(v for (s, jj), v in y.items() if jj == j)
        for (s, jj) in sorted([k for k in y if k[1] == j], key=lambda k: sub.pairs[k]["w"]):
            if tot <= r["D"]:
                break
            cut = min(y[(s, jj)], tot - r["D"])
            y[(s, jj)] -= cut
            tot -= cut
    for s, src in enumerate(sub.sources):                    # never exceed source stock
        tot = sum(v for (ss, j), v in y.items() if ss == s)
        for (ss, j) in sorted([k for k in y if k[0] == s], key=lambda k: sub.pairs[k]["w"]):
            if tot <= src["cap"]:
                break
            cut = min(y[(ss, j)], tot - src["cap"])
            y[(ss, j)] -= cut
            tot -= cut
    return y
