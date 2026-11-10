"""Classical preprocessing: dataset integration, location tiers, candidate (source, recipient) pairs and the
feasibility predicate that the Grover oracle evaluates (slides 10-14)."""
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import BLOOD_GROUPS, Params
from .data_gen import donor_eligibility

G = len(BLOOD_GROUPS)
DONOR_KEY0 = 10_000          # source keys: bank = h*G+g ; donor = DONOR_KEY0 + donor_index


class World:
    """Static (read-only) view of the five integrated datasets."""

    def __init__(self, data_dir: str, params: Params):
        self.p = params
        self.hosp = pd.read_csv(os.path.join(data_dir, "hospitals.csv"))
        self.H = len(self.hosp)
        self.h_idx = {h: i for i, h in enumerate(self.hosp.hospital_id)}
        self.g_idx = {g: i for i, g in enumerate(BLOOD_GROUPS)}
        self.donors = pd.read_csv(os.path.join(data_dir, "donors.csv"))
        self.requests = pd.read_csv(os.path.join(data_dir, "requests.csv")).sort_values(
            ["arrival_min", "request_id"]).reset_index(drop=True)

        comp = pd.read_csv(os.path.join(data_dir, "compatibility.csv"))
        self.compat = np.zeros((G, G), bool)
        self.exact = np.zeros((G, G), bool)
        for r in comp.itertuples():
            self.compat[self.g_idx[r.donor_group], self.g_idx[r.recipient_group]] = bool(r.compatible)
            self.exact[self.g_idx[r.donor_group], self.g_idx[r.recipient_group]] = bool(r.exact_match)

        dist = pd.read_csv(os.path.join(data_dir, "distances.csv"))
        hh = dist[dist.source_type == "hospital"]
        self.hh_km = np.zeros((self.H, self.H))
        self.hh_min = np.zeros((self.H, self.H))
        for r in hh.itertuples():
            a, b = self.h_idx[r.source_id], self.h_idx[r.hospital_id]
            self.hh_km[a, b], self.hh_min[a, b] = r.distance_km, r.travel_time_min
        dd = dist[dist.source_type == "donor"]
        did = {d: i for i, d in enumerate(self.donors.donor_id)}
        self.dh_km = np.zeros((len(self.donors), self.H))
        self.dh_min = np.zeros((len(self.donors), self.H))
        for r in dd.itertuples():
            self.dh_km[did[r.source_id], self.h_idx[r.hospital_id]] = r.distance_km
            self.dh_min[did[r.source_id], self.h_idx[r.hospital_id]] = r.travel_time_min

        # donor arrays; eligibility is re-validated from raw attributes (not trusted from the file)
        d = self.donors
        self.d_grp = d.blood_group.map(self.g_idx).values
        self.d_elig = donor_eligibility(d.age.values, d.sex.values, d.weight_kg.values,
                                        d.hemoglobin_g_dl.values, d.days_since_last_donation.values,
                                        d.medical_deferral.values)
        assert (self.d_elig.astype(int) == d.eligible.values).all(), "eligibility rule mismatch with registry"
        self.d_avail0 = d.available_now.values.astype(bool)

        inv = pd.read_csv(os.path.join(data_dir, "inventory.csv"))
        self.inv0 = np.zeros((self.H, G), int)
        for r in inv.itertuples():
            self.inv0[self.h_idx[r.hospital_id], self.g_idx[r.blood_group]] = r.units_available

        rq = self.requests
        self.req_h = rq.hospital_id.map(self.h_idx).values
        self.req_g = rq.blood_group.map(self.g_idx).values


class State:
    """Mutable inventory / donor availability (one copy per method so runs stay independent)."""

    def __init__(self, world: World):
        self.inv = world.inv0.copy()
        self.donor_used = np.zeros(len(world.donors), bool)
        self.donor_avail = world.d_avail0.copy()     # mutable: the dynamic simulation brings donors online/offline


@dataclass
class PairSet:
    kind: np.ndarray     # 0 = blood bank, 1 = donor
    skey: np.ndarray     # unique source key
    grp: np.ndarray      # source blood-group index
    cap: np.ndarray      # units the source can supply (stock, or 1 for a donor)
    time: np.ndarray     # delivery time in minutes (donors include collection/testing lead time)
    j: np.ndarray        # local recipient index
    tier: np.ndarray

    def __len__(self):
        return len(self.skey)

    def take(self, idx):
        idx = np.asarray(idx, dtype=int)
        return PairSet(*(getattr(self, f)[idx] for f in ("kind", "skey", "grp", "cap", "time", "j", "tier")))


def _empty():
    z = np.zeros(0)
    return PairSet(z.astype(int), z.astype(int), z.astype(int), z.astype(int), z.astype(float), z.astype(int), z.astype(int))


def tier_of_bank(world: World, h: int, h_r: int) -> int:
    if h == h_r:
        return 1
    return 2 if world.hh_km[h, h_r] <= world.p.near_km else 4


def preprocess_tier(world: World, state: State, recips: list, tier: int) -> PairSet:
    """Cheap classical pruning: keep only sources of this location tier that still hold stock / are not
    already committed. The *full* feasibility test (compatibility, eligibility, deadline ...) is left to the oracle."""
    cols = {k: [] for k in ("kind", "skey", "grp", "cap", "time", "j", "tier")}
    for jl, r in enumerate(recips):
        h_r = r["h"]
        if tier in (1, 2, 4):
            for h in range(world.H):
                if tier_of_bank(world, h, h_r) != tier:
                    continue
                for g in range(G):
                    if state.inv[h, g] > 0:
                        for k, v in (("kind", 0), ("skey", h * G + g), ("grp", g), ("cap", int(state.inv[h, g])),
                                     ("time", float(world.hh_min[h, h_r])), ("j", jl), ("tier", tier)):
                            cols[k].append(v)
        else:
            near = world.dh_km[:, h_r] <= world.p.near_km
            sel = np.where((near if tier == 3 else ~near) & state.donor_avail & ~state.donor_used)[0]
            for d in sel:
                for k, v in (("kind", 1), ("skey", DONOR_KEY0 + int(d)), ("grp", int(world.d_grp[d])), ("cap", 1),
                             ("time", float(world.dh_min[d, h_r] + world.p.donor_lead_min)), ("j", jl), ("tier", tier)):
                    cols[k].append(v)
    if not cols["skey"]:
        return _empty()
    return PairSet(np.array(cols["kind"]), np.array(cols["skey"]), np.array(cols["grp"]), np.array(cols["cap"]),
                   np.array(cols["time"], float), np.array(cols["j"]), np.array(cols["tier"]))


def oracle_truth(world: World, state: State, pairs: PairSet, recips: list) -> np.ndarray:
    """f(i,j) = Compatibility AND Eligibility AND Availability AND DeadlineFeasible  (slide 14).
    Returns the truth table the Grover oracle encodes (one bool per candidate)."""
    if len(pairs) == 0:
        return np.zeros(0, bool)
    r_grp = np.array([r["g"] for r in recips])[pairs.j]
    r_dl = np.array([r["deadline"] for r in recips])[pairs.j]
    compat = world.compat[pairs.grp, r_grp]
    deadline_ok = pairs.time <= r_dl
    didx = np.where(pairs.kind == 1, pairs.skey - DONOR_KEY0, 0)
    donor_ok = world.d_elig[didx] & state.donor_avail[didx] & ~state.donor_used[didx]
    bh = np.where(pairs.kind == 0, pairs.skey // G, 0)
    bank_ok = state.inv[bh, np.where(pairs.kind == 0, pairs.skey % G, 0)] >= 1
    avail = np.where(pairs.kind == 1, donor_ok, bank_ok)
    return compat & deadline_ok & avail
