"""Synthetic generation of the five datasets listed on slide 9.

1. donors.csv          - Donor Registry Dataset
2. compatibility.csv   - Blood Compatibility Dataset (RBC, ABO + Rh)
3. requests.csv        - Recipient Request Dataset
4. inventory.csv       - Hospital Blood Bank Inventory Dataset
5. hospitals.csv + distances.csv - Location and Distance Dataset

Everything is simulated; no real patient or donor data is used.
"""
import os
import numpy as np
import pandas as pd

from .config import BLOOD_GROUPS, GROUP_FREQ, URGENCY_WEIGHT, Params

# (id, area, lat, lon, relative size).  Hospital names are fictional placeholders around Delhi-NCR.
HOSPITALS = [
    ("H01", "Faridabad Central", 28.4089, 77.3178, 1.5),
    ("H02", "Ballabgarh", 28.3380, 77.3230, 0.7),
    ("H03", "Noida Sector-62", 28.6270, 77.3649, 1.6),
    ("H04", "Greater Noida", 28.4744, 77.5040, 1.0),
    ("H05", "South Delhi", 28.5245, 77.2066, 2.0),
    ("H06", "East Delhi", 28.6280, 77.2770, 1.3),
    ("H07", "Central Delhi", 28.6328, 77.2197, 2.0),
    ("H08", "Gurugram", 28.4595, 77.0266, 1.5),
    ("H09", "Palwal", 28.1487, 77.3320, 0.6),
    ("H10", "Ghaziabad", 28.6692, 77.4538, 1.2),
]
ROAD_FACTOR = 1.3        # road distance = 1.3 x great-circle distance
SPEED_KMPH = 30.0        # average emergency-transport speed in city traffic
DISPATCH_MIN = 5.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def abo_rh_compatible(donor_group: str, recipient_group: str) -> bool:
    """Red-cell compatibility (donor -> recipient)."""
    d_abo, d_rh = donor_group[:-1], donor_group[-1]
    r_abo, r_rh = recipient_group[:-1], recipient_group[-1]
    abo_ok = d_abo == "O" or d_abo == r_abo or r_abo == "AB"
    rh_ok = d_rh == "-" or r_rh == "+"
    return abo_ok and rh_ok


def donor_eligibility(age, sex, weight, hb, days_since, deferral):
    """Simplified donor eligibility rules (vectorised)."""
    min_gap = np.where(np.asarray(sex) == "M", 90, 120)
    return ((age >= 18) & (age <= 65) & (weight >= 50) & (hb >= 12.5)
            & (days_since >= min_gap) & (deferral == 0))


def _travel(dist_km):
    return DISPATCH_MIN + dist_km / SPEED_KMPH * 60.0


def generate_all(params: Params, out_dir: str):
    rng = np.random.default_rng(params.seed)
    os.makedirs(out_dir, exist_ok=True)
    g_names = list(GROUP_FREQ.keys())
    g_p = np.array([GROUP_FREQ[g] for g in g_names])
    g_p = g_p / g_p.sum()

    hosp = pd.DataFrame(HOSPITALS, columns=["hospital_id", "name", "lat", "lon", "size"])
    H = len(hosp)

    # ---------------- 5a. hospitals / distances ----------------
    hh_km = np.zeros((H, H))
    for a in range(H):
        for b in range(H):
            hh_km[a, b] = ROAD_FACTOR * haversine_km(hosp.lat[a], hosp.lon[a], hosp.lat[b], hosp.lon[b])

    # ---------------- 1. donors ----------------
    n = params.n_donors
    home = rng.integers(0, H, n)
    spread = np.where(rng.random(n) < 0.85, 0.045, 0.15)          # most donors live near a hospital
    lat = hosp.lat.values[home] + rng.normal(0, 1, n) * spread
    lon = hosp.lon.values[home] + rng.normal(0, 1, n) * spread
    sex = rng.choice(["M", "F"], n, p=[0.7, 0.3])
    age = np.clip(rng.normal(32, 10, n), 17, 68).round().astype(int)
    weight = np.clip(np.where(sex == "M", rng.normal(70, 10, n), rng.normal(58, 8, n)), 42, 105).round(1)
    hb = np.clip(np.where(sex == "M", rng.normal(14.5, 1.1, n), rng.normal(13.2, 1.0, n)), 10.0, 18.0).round(1)
    never = rng.random(n) < 0.25
    days_since = np.where(never, 9999, rng.integers(10, 500, n))
    total_don = np.where(never, 0, rng.integers(1, 30, n))
    deferral = (rng.random(n) < 0.04).astype(int)
    available = (rng.random(n) < 0.75).astype(int)
    donors = pd.DataFrame({
        "donor_id": [f"D{i:04d}" for i in range(n)],
        "blood_group": rng.choice(g_names, n, p=g_p),
        "age": age, "sex": sex, "weight_kg": weight, "hemoglobin_g_dl": hb,
        "days_since_last_donation": days_since, "total_donations": total_don,
        "medical_deferral": deferral, "available_now": available,
        "lat": lat.round(5), "lon": lon.round(5),
    })
    donors["nearest_hospital_id"] = [hosp.hospital_id[int(np.argmin(haversine_km(
        la, lo, hosp.lat.values, hosp.lon.values)))] for la, lo in zip(lat, lon)]
    donors["eligible"] = donor_eligibility(age, sex, weight, hb, days_since, deferral).astype(int)
    donors.to_csv(os.path.join(out_dir, "donors.csv"), index=False)

    # ---------------- 5b. distances (hospital->hospital and donor->hospital) ----------------
    rows = []
    for a in range(H):
        for b in range(H):
            rows.append((hosp.hospital_id[a], "hospital", hosp.hospital_id[b],
                         round(hh_km[a, b], 2), round(_travel(hh_km[a, b]) if a != b else 0.0, 1)))
    for i in range(n):
        d = ROAD_FACTOR * haversine_km(lat[i], lon[i], hosp.lat.values, hosp.lon.values)
        for b in range(H):
            rows.append((donors.donor_id[i], "donor", hosp.hospital_id[b], round(d[b], 2), round(_travel(d[b]), 1)))
    pd.DataFrame(rows, columns=["source_id", "source_type", "hospital_id", "distance_km", "travel_time_min"]).to_csv(
        os.path.join(out_dir, "distances.csv"), index=False)
    hosp[["hospital_id", "name", "lat", "lon"]].to_csv(os.path.join(out_dir, "hospitals.csv"), index=False)

    # ---------------- 2. compatibility ----------------
    comp = [(d, r, int(abo_rh_compatible(d, r)), int(d == r)) for d in BLOOD_GROUPS for r in BLOOD_GROUPS]
    pd.DataFrame(comp, columns=["donor_group", "recipient_group", "compatible", "exact_match"]).to_csv(
        os.path.join(out_dir, "compatibility.csv"), index=False)

    # ---------------- 4. hospital inventory ----------------
    inv = []
    for a in range(H):
        for g in BLOOD_GROUPS:
            lam = params.inventory_base * GROUP_FREQ[g] * hosp["size"][a] * 2.5
            inv.append((hosp.hospital_id[a], g, int(rng.poisson(lam))))
    pd.DataFrame(inv, columns=["hospital_id", "blood_group", "units_available"]).to_csv(
        os.path.join(out_dir, "inventory.csv"), index=False)

    # ---------------- 3. recipient requests ----------------
    m = params.n_requests
    size_p = hosp["size"].values / hosp["size"].sum()
    urg = rng.choice(["High", "Medium", "Low"], m, p=[0.30, 0.40, 0.30])
    deadline = np.array([rng.integers(90, 241) if u == "High" else
                         rng.integers(240, 601) if u == "Medium" else rng.integers(600, 1441) for u in urg])
    units = rng.choice([1, 2, 3, 4, 5, 6], m, p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05])
    arrival = np.sort(rng.integers(0, 241, m))
    req = pd.DataFrame({
        "request_id": [f"R{i:03d}" for i in range(m)],
        "patient_id": [f"P{i:04d}" for i in rng.permutation(9000)[:m] + 1000],
        "hospital_id": hosp.hospital_id.values[rng.choice(H, m, p=size_p)],
        "blood_group": rng.choice(g_names, m, p=g_p),
        "units_requested": units,
        "urgency": urg,
        "urgency_weight": [URGENCY_WEIGHT[u] for u in urg],
        "deadline_min": deadline,
        "arrival_min": arrival,
    })
    req.to_csv(os.path.join(out_dir, "requests.csv"), index=False)
    return out_dir
