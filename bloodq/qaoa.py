"""QAOA executed as Qiskit circuits, with classical COBYLA parameter optimisation.

The objective and constraints are still constructed by the project's QUBO builder.
Qiskit Statevector evaluates the ideal expectation during optimisation; Qiskit Aer
performs the final finite-shot measurement sampling.
"""
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit_aer import AerSimulator

from .circuits import qaoa_circuit


class QAOA:
    def __init__(self, E: np.ndarray, n: int, p: int, cvar_alpha: float = 1.0, Q=None):
        self.n, self.p, self.alpha = int(n), int(p), float(cvar_alpha)
        self.E = np.asarray(E, dtype=float)
        self.mean = float(self.E.mean())
        self.scale = float(self.E.std()) or 1.0
        self.Es = (self.E - self.mean) / self.scale
        self.evals = 0
        if Q is None:
            raise ValueError("QAOA now requires the QUBO matrix Q so it can construct the Qiskit cost circuit")
        self.h, self.J, self.offset = self._qubo_to_ising(np.asarray(Q, dtype=float))

    def _qubo_to_ising(self, Q):
        h = -np.diag(Q) / 2.0
        J = {}
        offset = float(np.trace(Q) / 2.0)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                qij = 2.0 * Q[i, j]
                if abs(qij) > 1e-12:
                    J[(i, j)] = qij / 4.0
                    h[i] -= qij / 4.0
                    h[j] -= qij / 4.0
                    offset += qij / 4.0
        return h, J, offset

    def circuit(self, params, measure=False):
        gammas, betas = params[:self.p], params[self.p:]
        qc = qaoa_circuit(self.h, self.J, gammas, betas, scale=self.scale)
        if measure:
            qc.measure_all()
        return qc

    def probs(self, params):
        qc = self.circuit(params, measure=False)
        pr = Statevector.from_instruction(qc).probabilities()
        return pr / pr.sum()

    def cost(self, params):
        self.evals += 1
        pr = self.probs(params)
        if self.alpha >= 1.0:
            return float(pr @ self.Es)
        order = np.argsort(self.Es)
        cum = np.cumsum(pr[order])
        k = np.searchsorted(cum, self.alpha) + 1
        w = pr[order][:k].copy()
        w[-1] -= cum[k - 1] - self.alpha
        return float((w * self.Es[order][:k]).sum() / self.alpha)

    def optimize(self, restarts, maxiter, rng):
        best = None
        for r in range(restarts):
            if r == 0:
                ramp = (np.arange(self.p) + 0.5) / self.p
                x0 = np.concatenate([0.6 * ramp, 0.6 * (1 - ramp)])
            else:
                x0 = np.concatenate([rng.uniform(0.05, 1.2, self.p), rng.uniform(0.05, 0.9, self.p)])
            res = minimize(self.cost, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
            if best is None or res.fun < best.fun:
                best = res
        return best.x, float(best.fun)

    def sample(self, params, shots, rng):
        qc = self.circuit(params, measure=True)
        seed = int(rng.integers(0, 2**31 - 1))
        backend = AerSimulator()
        tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=seed)
        counts = backend.run(tqc, shots=int(shots), seed_simulator=seed).result().get_counts()
        # Qiskit count strings are displayed most-significant bit first; integer indices
        # match the little-endian qubit convention used by all_bits() and the QUBO decoder.
        by_index = {}
        for bitstring, count in counts.items():
            idx = int(bitstring.replace(" ", ""), 2)
            by_index[idx] = by_index.get(idx, 0) + int(count)
        idx = np.array(sorted(by_index), dtype=int)
        return idx, np.array([by_index[i] for i in idx], dtype=int)
