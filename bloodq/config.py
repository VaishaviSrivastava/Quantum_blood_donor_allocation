"""Global constants and tunable parameters for the hybrid quantum-classical blood allocation project."""
from dataclasses import dataclass

BLOOD_GROUPS = ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]

# Approximate ABO/Rh distribution for an Indian population (synthetic data only).
GROUP_FREQ = {"O+": 0.36, "B+": 0.31, "A+": 0.21, "AB+": 0.07,
              "O-": 0.02, "B-": 0.015, "A-": 0.01, "AB-": 0.005}

# Urgency priority weights: w_H > w_M > w_L > 0 (slide 15/17).
URGENCY_WEIGHT = {"High": 5, "Medium": 3, "Low": 1}

TIER_NAMES = {
    1: "Local hospital blood bank",
    2: "Nearby hospitals' blood banks",
    3: "Nearby eligible donors",
    4: "Distant hospitals' blood banks",
    5: "Distant eligible donors",
}


@dataclass
class Params:
    seed: int = 42
    # ---- synthetic data sizes ----
    n_donors: int = 400
    n_requests: int = 36
    inventory_base: float = 4.5        # scarcity knob for hospital stock
    # ---- tiered location search (slide 11) ----
    near_km: float = 15.0              # "nearby" radius
    donor_lead_min: float = 120.0      # collection + testing time before a donor unit can ship
    # ---- objective / QUBO (slides 15-19) ----
    lam: float = 0.1                   # travel-cost weight  (c_sj is in HOURS)
    mu: float = 1.0                    # unmet-demand penalty weight
    mismatch_factor: float = 0.9       # w_sj multiplier for non-identical (compatible) group
    penalty_scale: float = 1.2         # P_D = P_A = penalty_scale * (w_max + mu * p_max)
    max_units_per_round: int = 3       # tranche of units per recipient per round
    max_rounds: int = 4
    batch_size: int = 2                # recipients solved jointly in one QUBO
    max_qubits: int = 14               # qubit budget per QUBO sub-problem
    min_sources: int = 3               # try to give the optimiser a real choice
    # ---- QAOA ----
    p_layers: int = 2
    shots: int = 4096
    restarts: int = 3
    maxiter: int = 120
    cvar_alpha: float = 1.0            # 1.0 = plain expectation (slide 21)
    # ---- Grover ----
    grover_mode: str = "all"           # 'all' = list every feasible pair in a tier, 'enough' = stop when supply suffices
    # ---- SA baseline ----
    sa_restarts: int = 8
    sa_sweeps: int = 150
