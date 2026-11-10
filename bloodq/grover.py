"""Grover search executed with Qiskit circuits and Qiskit Aer.

Important modelling note: the current BloodQ preprocessing stage supplies the marked
indices (via its classical eligibility mask). This module runs the Grover circuit for
those indices; it does not yet encode the live donor/inventory predicate as reversible
quantum arithmetic. See README for this boundary.
"""
import math
import numpy as np

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from .circuits import grover_circuit


def optimal_iterations(N: int, M: int) -> int:
    if M <= 0:
        return 0
    theta = math.asin(math.sqrt(min(M / N, 1.0)))
    return max(0, int(round(math.pi / (4 * theta) - 0.5)))


def pad_to_qubits(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    N0 = len(mask)
    n = max(2, math.ceil(math.log2(max(N0, 2))))
    if mask.sum() / (1 << n) > 0.25:
        n += 1
    full = np.zeros(1 << n, dtype=bool)
    full[:N0] = mask
    return n, full


def grover_probs(n: int, marked: np.ndarray, r: int) -> np.ndarray:
    """Return ideal probabilities from a Qiskit Grover circuit's statevector."""
    marked = np.asarray(marked, dtype=bool)
    indices = np.flatnonzero(marked).tolist()
    if len(marked) != (1 << n):
        raise ValueError(f"marked mask must have length 2**n ({1 << n})")
    if not indices:
        return np.full(1 << n, 1 / (1 << n))
    qc = grover_circuit(n, indices, r, measure=False)
    return Statevector.from_instruction(qc).probabilities()


def _sample_circuit(n: int, marked_indices: list[int], iterations: int, shots: int, seed: int):
    qc = grover_circuit(n, marked_indices, iterations, measure=True)
    backend = AerSimulator()
    tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=seed)
    result = backend.run(tqc, shots=shots, seed_simulator=seed).result()
    return result.get_counts()


def grover_find(mask: np.ndarray, rng, enough=None, max_attempts: int = 6) -> dict:
    """Repeated Qiskit/Aer Grover search, removing each successfully found marked index."""
    mask = np.asarray(mask, dtype=bool)
    M_total = int(mask.sum())
    if len(mask) == 0 or M_total == 0:
        return dict(found=[], queries=0, n_qubits=0, N=len(mask), M=M_total, iterations=[], p_success=[])
    n, remaining = pad_to_qubits(mask)
    N = 1 << n
    found, queries, iters, psucc = [], 0, [], []
    backend = AerSimulator()
    while remaining.any():
        if enough is not None and found and enough(found):
            break
        marked_indices = np.flatnonzero(remaining).tolist()
        M = len(marked_indices)
        r = optimal_iterations(N, M)
        probs = grover_probs(n, remaining, r)
        psucc.append(float(probs[remaining].sum()))
        iters.append(r)
        qc = grover_circuit(n, marked_indices, r, measure=True)
        seed = int(rng.integers(0, 2**31 - 1))
        tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=seed)
        counts = backend.run(tqc, shots=max_attempts, seed_simulator=seed).result().get_counts()
        queries += max_attempts * (r + 1)  # each shot represents a fresh search/verification attempt
        sampled = []
        for bitstring, count in counts.items():
            x = int(bitstring.replace(" ", ""), 2)
            if x < N and remaining[x]:
                sampled.extend([x] * int(count))
        if not sampled:
            break
        # Pick one of the successfully measured marked states; leave other samples unused.
        x = int(sampled[int(rng.integers(0, len(sampled)))])
        found.append(x)
        remaining[x] = False
    return dict(found=found, queries=queries, n_qubits=n, N=N, M=M_total, iterations=iters, p_success=psucc)


def bbht(mask: np.ndarray, rng, lam: float = 6 / 5, cap_factor: float = 12.0):
    """BBHT-style unknown-M search using Qiskit circuits; mask is still a supplied oracle table."""
    n, full = pad_to_qubits(np.asarray(mask, bool))
    N = 1 << n
    m, queries, cap = 1.0, 0, cap_factor * math.sqrt(N)
    backend = AerSimulator()
    while queries < cap:
        j = int(rng.integers(0, max(1, int(math.ceil(m)))))
        marked_indices = np.flatnonzero(full).tolist()
        qc = grover_circuit(n, marked_indices, j, measure=True)
        seed = int(rng.integers(0, 2**31 - 1))
        tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=seed)
        counts = backend.run(tqc, shots=1, seed_simulator=seed).result().get_counts()
        x = int(next(iter(counts)).replace(" ", ""), 2)
        queries += j + 1
        if full[x]:
            return x, queries
        m = min(lam * m, math.sqrt(N))
    return None, queries
