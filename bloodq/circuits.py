"""Qiskit circuits for Grover and QAOA: resource metrics (qubits, depth, CX count) and noisy simulation."""
import numpy as np
from qiskit import QuantumCircuit, transpile

BASIS = ["rz", "sx", "x", "cx"]


def grover_circuit(n: int, marked: list, r: int, measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(r):
        for m in marked:                                   # phase oracle: flip sign of each marked basis state
            zeros = [q for q in range(n) if not (m >> q) & 1]
            if zeros:
                qc.x(zeros)
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
            if zeros:
                qc.x(zeros)
        qc.h(range(n))                                     # diffusion: inversion about the mean
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
    if measure:
        qc.measure_all()
    return qc


def qaoa_circuit(h, J, gammas, betas, scale: float = 1.0) -> QuantumCircuit:
    n = len(h)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * g * h[i] / scale, i)             # exp(-i g h Z)
        for (i, j), v in J.items():
            qc.rzz(2 * g * v / scale, i, j)                # exp(-i g J Z Z)
        qc.rx(2 * b, range(n))                             # mixer exp(-i b X)
    return qc


def resource_metrics(qc: QuantumCircuit) -> dict:
    t = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)
    ops = t.count_ops()
    return dict(qubits=qc.num_qubits, depth=t.depth(), gates=int(t.size()), cx=int(ops.get("cx", 0)))


def noisy_counts(qc: QuantumCircuit, p1: float, shots: int, seed: int = 1) -> dict:
    """Run on Aer with depolarising noise: 1-qubit error p1 on sx/x, 2-qubit error min(10*p1, 0.75) on cx."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    qm = qc.copy()
    qm.measure_all()
    t = transpile(qm, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)
    if p1 > 0:
        nm = NoiseModel(basis_gates=BASIS)
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(min(10 * p1, 0.75), 2), ["cx"])
        sim = AerSimulator(noise_model=nm, seed_simulator=seed)
    else:
        sim = AerSimulator(seed_simulator=seed)
    return sim.run(t, shots=shots).result().get_counts()
