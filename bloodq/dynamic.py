"""Time-dynamic simulation engine (adds what the batch pipeline lacked).

What changes over simulated time
  * Restocking      - every `restock_every_min` each hospital receives a random delivery of new units.
  * Expiry          - every unit is a lot with an expiry time; stock is FIFO (oldest first) and expired units are discarded.
  * Donor arrivals  - donors that are unavailable at t=0 come online at random later times.
  * Failed delivery - a donor may not show up, a shipment may fail; units go back to the bank (or expire),
                      the request gets its units back and is RE-PLANNED with the time that is left.
  * Live requests   - new requests can be submitted at any moment (UI), deadlines run on the simulation clock.

Every planning round re-uses the same machinery as the static pipeline (tiered Grover search -> QUBO -> QAOA or a
classical solver -> classical validation), so results are comparable. Randomness of the ENVIRONMENT (restock sizes,
unit shelf life, donor arrival times, delivery failures) is fixed by the seed and does not depend on the method,
so different methods face exactly the same world.
"""
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

from .config import Params, BLOOD_GROUPS, GROUP_FREQ, URGENCY_WEIGHT
from .data_gen import HOSPITALS
from .preprocess import World, State, DONOR_KEY0, G
from . import pipeline as P


@dataclass
class DynParams:
    step_min: float = 5.0               # simulation tick
    horizon_min: float = 1500.0
    restock_every_min: float = 360.0    # 0 = never restock
    restock_scale: float = 0.5          # restock size relative to the initial stock level
    near_expiry_frac: float = 0.25      # share of initial units that expire within `near_expiry_max_h` hours
    near_expiry_max_h: float = 18.0
    donor_online_frac: float = 0.5      # share of initially unavailable (eligible) donors that come online later
    donor_noshow_prob: float = 0.15
    shipment_fail_prob: float = 0.05    # bank-to-bank transport failure (own-hospital hand-over never fails)
    max_rounds_per_step: int = 8
    priority_batching: bool = True      # most urgent pending requests are batched first (else arrival order)


@dataclass
class Req:
    idx: int
    rid: str
    h: int
    g: int
    units: int
    urgency: str
    w: int
    arrival: float
    deadline_rel: float
    rem: int                            # units still to be assigned (not delivered, not in transit)
    inflight: int = 0
    delivered: int = 0
    status: str = "scheduled"           # scheduled -> pending -> fulfilled | lapsed
    replans: int = 0
    tried_version: int = -1
    closed_t: float = None
    source: str = "dataset"

    @property
    def abs_deadline(self):
        return self.arrival + self.deadline_rel


@dataclass
class Delivery:
    did: int
    rid: str
    req_idx: int
    skey: int
    kind: int
    tier: int
    units: int
    depart: float
    arrive: float
    lots: list
    status: str = "transit"             # transit | delivered | failed
    force_fail: bool = False


def _py(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class Simulation:
    def __init__(self, world: World, params: Params, dyn: DynParams, method: str = "qaoa", circuit_metrics: bool = False):
        self.world, self.p, self.dyn, self.method = world, params, dyn, method
        self.circuit_metrics = circuit_metrics
        self.t = 0.0
        self.state = State(world)
        self.rng = np.random.default_rng(params.seed + 7)                   # solver randomness
        env = np.random.default_rng(params.seed + 101)                       # environment randomness (method independent)
        self.sizes = np.array([h[4] for h in HOSPITALS], float)
        self.freq = np.array([GROUP_FREQ[g] for g in BLOOD_GROUPS])

        # ---- stock as lots with expiry times ----
        self.lots = {(h, g): [] for h in range(world.H) for g in range(G)}
        for h in range(world.H):
            for g in range(G):
                self.lots[(h, g)] = sorted(self._draw_life(env, 0.0) for _ in range(int(world.inv0[h, g])))
        self.state.inv[:] = 0
        for hg in self.lots:
            self._sync(*hg)
        self.initial_units = int(self.state.inv.sum())

        # ---- pre-generated environment events ----
        self.restock_events = []
        if dyn.restock_every_min > 0:
            k = 1
            while k * dyn.restock_every_min <= dyn.horizon_min:
                t = k * dyn.restock_every_min
                for h in range(world.H):
                    by_g = {}
                    for g in range(G):
                        lam = params.inventory_base * self.freq[g] * self.sizes[h] * 2.5 * dyn.restock_scale
                        n = int(env.poisson(lam))
                        if n:
                            by_g[g] = [self._draw_life(env, t) for _ in range(n)]
                    if by_g:
                        self.restock_events.append((t, h, by_g))
                k += 1
        self.donor_events = sorted(
            (float(env.uniform(0, 0.7 * dyn.horizon_min)), int(d)) for d in range(len(world.donors))
            if world.d_elig[d] and not world.d_avail0[d] and env.random() < dyn.donor_online_frac)
        self._ri = 0
        self._di = 0

        # ---- requests ----
        self.reqs = []
        rq = world.requests
        for i in range(len(rq)):
            self._add_req(rq.request_id[i], int(world.req_h[i]), int(world.req_g[i]), int(rq.units_requested[i]),
                          rq.urgency[i], float(rq.arrival_min[i]), float(rq.deadline_min[i]), "dataset")
        self.deliveries = []
        self.events = []
        self.trace = []
        self.timeline = []
        self.attempts = {}
        self.supply_version = 0
        self.used_donors = set()
        # accounting
        self.restocked = 0
        self.expired_stock = 0
        self.expired_transit = 0
        self.returned = 0
        self.dispatched_bank = 0
        self.noshows = 0
        self.ship_fails = 0
        self.solver_seconds = 0.0
        self._record()

    # ------------------------------------------------------------------ helpers
    def _draw_life(self, env, t0):
        if env.random() < self.dyn.near_expiry_frac:
            return float(t0 + env.uniform(30, self.dyn.near_expiry_max_h * 60))
        return float(t0 + env.uniform(7 * 1440, 42 * 1440))

    def _sync(self, h, g):
        self.state.inv[h, g] = len(self.lots[(h, g)])

    def _add_req(self, rid, h, g, units, urgency, arrival, deadline_rel, source):
        r = Req(len(self.reqs), rid, h, g, units, urgency, URGENCY_WEIGHT[urgency], arrival, deadline_rel, units, source=source)
        self.reqs.append(r)
        return r

    def _log(self, kind, text, rid="", t=None):
        self.events.append(dict(t=round(self.t if t is None else t, 1), type=kind, request=rid, detail=text))

    def describe_source(self, skey):
        if skey >= DONOR_KEY0:
            d = skey - DONOR_KEY0
            return f"Donor {self.world.donors.donor_id[d]} ({BLOOD_GROUPS[self.world.d_grp[d]]})"
        h, g = divmod(skey, G)
        return f"{self.world.hosp.hospital_id[h]} bank ({BLOOD_GROUPS[g]})"

    def hosp_label(self, h):
        return f"{self.world.hosp.hospital_id[h]} {self.world.hosp.name[h]}"

    # ------------------------------------------------------------------ environment events
    def _process_restock(self):
        while self._ri < len(self.restock_events) and self.restock_events[self._ri][0] <= self.t:
            t, h, by_g = self.restock_events[self._ri]
            self._ri += 1
            self._add_stock(h, by_g, t)

    def _add_stock(self, h, by_g, t):
        for g, lots in by_g.items():
            self.lots[(h, g)] = sorted(self.lots[(h, g)] + list(lots))
            self._sync(h, g)
            self.restocked += len(lots)
        self.supply_version += 1
        parts = ", ".join(f"{BLOOD_GROUPS[g]} x{len(l)}" for g, l in by_g.items())
        self._log("RESTOCK", f"{self.world.hosp.hospital_id[h]} received {sum(len(l) for l in by_g.values())} unit(s): {parts}", t=t)

    def restock_now(self, hospital_idx, group_idx, n):
        """Manual restock from the UI (fresh units, ~30 day life)."""
        self._add_stock(int(hospital_idx), {int(group_idx): [self.t + 30 * 1440] * int(n)}, self.t)
        self._plan()
        self._record()

    def _process_donors(self):
        while self._di < len(self.donor_events) and self.donor_events[self._di][0] <= self.t:
            t, d = self.donor_events[self._di]
            self._di += 1
            if not self.state.donor_used[d]:
                self.state.donor_avail[d] = True
                self.supply_version += 1
                self._log("DONOR_ONLINE", f"{self.world.donors.donor_id[d]} ({BLOOD_GROUPS[self.world.d_grp[d]]}) is now available", t=t)

    def _process_expiry(self):
        for (h, g), lots in self.lots.items():
            n = 0
            while lots and lots[0] <= self.t:
                lots.pop(0)
                n += 1
            if n:
                self.expired_stock += n
                self._sync(h, g)
                self._log("EXPIRED", f"{n} unit(s) of {BLOOD_GROUPS[g]} expired at {self.world.hosp.hospital_id[h]} (discarded)")

    # ------------------------------------------------------------------ deliveries
    def _fail_prob(self, d: Delivery):
        if d.kind == 1:
            return self.dyn.donor_noshow_prob
        return 0.0 if d.tier == 1 else self.dyn.shipment_fail_prob

    def _roll_fail(self, d: Delivery):
        if d.force_fail:
            return True
        key = (d.rid, d.skey)
        a = self.attempts.get(key, 0)
        self.attempts[key] = a + 1
        return bool(np.random.default_rng([self.p.seed, 9001 + d.req_idx, d.skey, a]).random() < self._fail_prob(d))

    def _process_deliveries(self):
        for d in sorted([d for d in self.deliveries if d.status == "transit" and (d.arrive <= self.t or d.force_fail)],
                        key=lambda d: d.arrive):
            r = self.reqs[d.req_idx]
            if self._roll_fail(d):
                self._fail(d, r)
            else:
                d.status = "delivered"
                r.inflight -= d.units
                r.delivered += d.units
                self._log("DELIVERED", f"{d.units} unit(s) from {self.describe_source(d.skey)} reached {self.hosp_label(r.h)}",
                          r.rid, t=d.arrive)
                if r.rem == 0 and r.inflight == 0:
                    r.status, r.closed_t = "fulfilled", d.arrive
                    self._log("FULFILLED", f"{r.rid} fully served", r.rid, t=d.arrive)

    def _fail(self, d: Delivery, r: Req):
        d.status = "failed"
        r.inflight -= d.units
        r.rem += d.units
        r.replans += 1
        r.tried_version = -1
        r.status = "pending"
        if d.kind == 1:
            didx = d.skey - DONOR_KEY0
            self.state.donor_avail[didx] = False
            self.noshows += 1
            why = "donor did not show up"
        else:
            h, g = divmod(d.skey, G)
            back = [e for e in d.lots if e > self.t]
            self.expired_transit += len(d.lots) - len(back)
            self.lots[(h, g)] = sorted(self.lots[(h, g)] + back)
            self._sync(h, g)
            self.returned += len(back)
            self.ship_fails += 1
            self.supply_version += 1
            why = "shipment failed, units returned to bank" + (" (some expired)" if len(back) < len(d.lots) else "")
        left = r.abs_deadline - self.t
        self._log("FAILED", f"{d.units} unit(s) from {self.describe_source(d.skey)}: {why}", r.rid)
        self._log("REPLAN", f"{r.rid} re-queued, {max(left, 0):.0f} min left before deadline", r.rid)

    def fail_delivery(self, did: int):
        """UI: force an in-transit delivery to fail right now."""
        for d in self.deliveries:
            if d.did == did and d.status == "transit":
                d.force_fail = True
                self._process_deliveries()
                self._plan()
                self._record()
                return True
        return False

    # ------------------------------------------------------------------ requests
    def submit_request(self, hospital_idx, group_idx, units, urgency, deadline_min, plan=True):
        n = sum(1 for r in self.reqs if r.source == "live") + 1
        r = self._add_req(f"LIVE{n:03d}", int(hospital_idx), int(group_idx), int(units), urgency, self.t, float(deadline_min), "live")
        r.status = "pending"
        self._log("ARRIVAL", f"{r.units} unit(s) {BLOOD_GROUPS[r.g]} for {self.hosp_label(r.h)} [{urgency}, deadline {deadline_min:.0f} min] (live)", r.rid)
        if plan:
            self._plan()
            self._record()
        return r

    def _arrivals(self):
        for r in self.reqs:
            if r.status == "scheduled" and r.arrival <= self.t:
                r.status = "pending"
                self._log("ARRIVAL", f"{r.units} unit(s) {BLOOD_GROUPS[r.g]} for {self.hosp_label(r.h)} [{r.urgency}, deadline {r.deadline_rel:.0f} min]",
                          r.rid, t=r.arrival)

    def _lapse(self):
        for r in self.reqs:
            if r.status == "pending" and r.rem > 0 and self.t >= r.abs_deadline and r.inflight == 0:
                r.status, r.closed_t = "lapsed", self.t
                self._log("LAPSED", f"{r.rid} deadline passed: {r.delivered}/{r.units} units delivered", r.rid)

    # ------------------------------------------------------------------ planning (same machinery as the static pipeline)
    def _plan(self):
        rounds = 0
        while rounds < self.dyn.max_rounds_per_step:
            pend = [r for r in self.reqs if r.status == "pending" and r.rem > 0 and self.t < r.abs_deadline
                    and r.tried_version != self.supply_version]
            if not pend:
                break
            key = (lambda r: (-r.w, r.abs_deadline, r.arrival)) if self.dyn.priority_batching else (lambda r: (r.arrival, r.idx))
            self._plan_round(sorted(pend, key=key)[: self.p.batch_size])
            rounds += 1

    def _plan_round(self, batch):
        import time as _t
        t0 = _t.perf_counter()
        recips = [dict(rid=r.rid, h=r.h, g=r.g, deadline=r.abs_deadline - self.t, w=r.w, p=r.w,
                       D=min(r.rem, self.p.max_units_per_round)) for r in batch]
        engine = "grover" if self.method == "qaoa" else "classical"
        feas, stats = P.tiered_search(self.world, self.state, recips, self.p, engine, self.rng)
        tr = dict(t=self.t, method=self.method, requests=[r.rid for r in batch], search=[{k: _py(v) for k, v in s.items()} for s in stats],
                  n_candidates=int(len(feas)), candidates=[], info={}, allocations=[])
        if len(feas) == 0:
            for r in batch:
                r.tried_version = self.supply_version
            self._log("NO_SUPPLY", "no feasible source (compatibility/stock/eligibility/deadline) for " + ", ".join(r.rid for r in batch),
                      batch[0].rid)
            self.trace.append(tr)
            return
        cands = P._cand_list(self.world, feas, recips, self.p)
        for j, r in enumerate(batch):
            if not any(c["j"] == j for c in cands):
                r.tried_version = self.supply_version
        if self.method in ("greedy", "ilp"):
            a = P.solve_classical(self.method, recips, cands, self.p)
            info = {}
        else:
            sub = P.build_subproblem(cands, recips, self.p)
            a, info = P.solve_qubo_method(self.method, sub, self.p, self.rng, self.circuit_metrics)
            info = {k: _py(v) for k, v in info.items() if not k.startswith("_")}
        by = {(c["skey"], c["j"]): c for c in cands}
        got = {j: 0 for j in range(len(batch))}
        for (skey, j), u in a.items():
            if u > 0:
                self._dispatch(batch[j], by[(skey, j)], int(u))
                got[j] += int(u)
        for j, r in enumerate(batch):
            if got[j] == 0:
                r.tried_version = self.supply_version
        tr["candidates"] = [dict(source=self.describe_source(c["skey"]), request=recips[c["j"]]["rid"], tier=c["tier"],
                                 minutes=round(c["time"], 1), reward_w=round(c["w"], 2), group_match="identical" if c["exact"] else "compatible",
                                 stock=c["cap"], units_chosen=int(a.get((c["skey"], c["j"]), 0))) for c in cands]
        tr["info"] = info
        tr["allocations"] = [dict(request=batch[j].rid, source=self.describe_source(s), units=int(u)) for (s, j), u in a.items() if u > 0]
        tr["seconds"] = _t.perf_counter() - t0
        self.solver_seconds += tr["seconds"]
        self.trace.append(tr)

    def _dispatch(self, r: Req, c: dict, u: int):
        assert 0 < u <= r.rem, "allocation exceeds remaining demand"
        taken = []
        if c["kind"] == 0:
            h, g = divmod(c["skey"], G)
            assert len(self.lots[(h, g)]) >= u, "over-allocation of stock"
            taken = [self.lots[(h, g)].pop(0) for _ in range(u)]          # FIFO: oldest units first
            self._sync(h, g)
            self.dispatched_bank += u
        else:
            d = c["skey"] - DONOR_KEY0
            assert u == 1 and not self.state.donor_used[d] and d not in self.used_donors, "donor reused"
            self.state.donor_used[d] = True
            self.used_donors.add(d)
        r.rem -= u
        r.inflight += u
        dl = Delivery(len(self.deliveries), r.rid, r.idx, c["skey"], c["kind"], c["tier"], u, self.t, self.t + c["time"], taken)
        self.deliveries.append(dl)
        self._log("ALLOCATION", f"{u} unit(s) from {self.describe_source(c['skey'])} (tier {c['tier']}, ETA {c['time']:.0f} min) -> {self.hosp_label(r.h)}", r.rid)

    # ------------------------------------------------------------------ clock
    def tick(self):
        self.t = min(self.t + self.dyn.step_min, self.dyn.horizon_min)
        self._process_restock()
        self._process_donors()
        self._process_expiry()
        self._process_deliveries()
        self._arrivals()
        self._lapse()
        self._plan()
        self._record()

    def advance(self, minutes, progress=None):
        target = min(self.t + minutes, self.dyn.horizon_min)
        n = int(np.ceil((target - self.t) / self.dyn.step_min - 1e-9))
        for i in range(max(n, 0)):
            if self.t >= self.dyn.horizon_min:
                break
            self.tick()
            if progress:
                progress((i + 1) / max(n, 1))

    def run_to_end(self, progress=None):
        self.advance(self.dyn.horizon_min - self.t, progress)
        # close anything still open at the horizon
        for r in self.reqs:
            if r.status == "pending" and r.inflight == 0 and r.rem > 0:
                r.status, r.closed_t = "lapsed", self.t

    @property
    def finished(self):
        return self.t >= self.dyn.horizon_min - 1e-9

    # ------------------------------------------------------------------ views / metrics
    def _record(self):
        arrived = [r for r in self.reqs if r.status != "scheduled"]
        self.timeline.append(dict(
            t=self.t, stock=int(self.state.inv.sum()), in_transit=sum(r.inflight for r in self.reqs),
            delivered=sum(r.delivered for r in self.reqs), expired=self.expired_stock + self.expired_transit,
            restocked=self.restocked, waiting_requests=sum(1 for r in arrived if r.status == "pending"),
            donors_online=int((self.state.donor_avail & self.world.d_elig & ~self.state.donor_used).sum())))

    def stock_matrix(self):
        return pd.DataFrame(self.state.inv, index=[f"{a} {b}" for a, b in zip(self.world.hosp.hospital_id, self.world.hosp.name)],
                            columns=BLOOD_GROUPS)

    def requests_df(self):
        rows = []
        for r in self.reqs:
            rows.append(dict(request=r.rid, hospital=self.world.hosp.hospital_id[r.h], group=BLOOD_GROUPS[r.g], urgency=r.urgency,
                             units=r.units, delivered=r.delivered, in_transit=r.inflight, waiting=r.rem, status=r.status,
                             arrival=r.arrival, deadline_at=r.abs_deadline, closed_at=r.closed_t, replans=r.replans, source=r.source))
        return pd.DataFrame(rows)

    def deliveries_df(self):
        rows = [dict(id=d.did, request=d.rid, source=self.describe_source(d.skey), kind="donor" if d.kind else "bank", tier=d.tier,
                     units=d.units, departed=d.depart, arrives=round(d.arrive, 1), status=d.status) for d in self.deliveries]
        return pd.DataFrame(rows, columns=["id", "request", "source", "kind", "tier", "units", "departed", "arrives", "status"])

    def events_df(self):
        return pd.DataFrame(self.events, columns=["t", "type", "request", "detail"])

    def timeline_df(self):
        return pd.DataFrame(self.timeline)

    def metrics(self):
        arrived = [r for r in self.reqs if r.status != "scheduled"]
        req_units = sum(r.units for r in arrived)
        deliv = sum(r.delivered for r in arrived)
        wreq = sum(r.w * r.units for r in arrived)
        wdel = sum(r.w * r.delivered for r in arrived)
        ok = [d for d in self.deliveries if d.status == "delivered"]
        resp = [(d.arrive - self.reqs[d.req_idx].arrival, d.units) for d in ok]
        closed = [r for r in arrived if r.status in ("fulfilled", "lapsed")]
        high_unmet = sum(r.units - r.delivered for r in closed if r.urgency == "High" and r.status == "lapsed")
        return dict(
            time_min=self.t, requests_arrived=len(arrived), requests_fulfilled=sum(r.status == "fulfilled" for r in arrived),
            requests_lapsed=sum(r.status == "lapsed" for r in arrived), requests_open=sum(r.status == "pending" for r in arrived),
            units_requested=req_units, units_delivered=deliv, units_in_transit=sum(r.inflight for r in arrived),
            fulfillment_rate=deliv / req_units if req_units else 1.0,
            urgency_weighted_fulfillment=wdel / wreq if wreq else 1.0, high_urgency_units_lapsed=high_unmet,
            mean_response_min=(sum(a * u for a, u in resp) / sum(u for _, u in resp)) if resp else 0.0,
            failed_deliveries=self.noshows + self.ship_fails, donor_noshows=self.noshows, shipment_failures=self.ship_fails,
            replans=sum(r.replans for r in self.reqs), units_expired=self.expired_stock + self.expired_transit,
            units_restocked=self.restocked, stock_now=int(self.state.inv.sum()),
            deadline_violations=sum(1 for d in ok if d.arrive > self.reqs[d.req_idx].abs_deadline + 1e-9),
            planning_rounds=len(self.trace), solver_seconds=self.solver_seconds)

    def check_invariants(self):
        """Raises AssertionError if the books do not balance."""
        for r in self.reqs:
            assert r.delivered + r.inflight + r.rem == r.units, f"unit leak in {r.rid}"
            assert r.rem >= 0 and r.inflight >= 0
        stock = int(self.state.inv.sum())
        assert stock == sum(len(v) for v in self.lots.values()), "inv out of sync with lots"
        assert self.initial_units + self.restocked + self.returned == stock + self.dispatched_bank + self.expired_stock, \
            "stock does not balance"
        for d in self.deliveries:
            if d.status == "delivered":
                assert d.arrive <= self.reqs[d.req_idx].abs_deadline + 1e-9, "late delivery"
        assert len(self.used_donors) == len({d.skey for d in self.deliveries if d.kind == 1}), "donor used twice"
        return True


def run_full(world, params, dyn, method, circuit_metrics=False, progress=None):
    sim = Simulation(world, params, dyn, method, circuit_metrics)
    sim.run_to_end(progress)
    return sim
