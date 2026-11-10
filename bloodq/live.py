"""Time-dynamic ("live") version of the allocation system.

Adds what the batch simulation (pipeline.run_method) does not have:

* a simulation clock that advances in ticks (minutes)
* blood units with an expiry time (lots, first-expiry-first-out) and periodic RESTOCKING
* donors who become available / unavailable over time (churn) and who are used up once they donate
* dispatched deliveries that take time and can FAIL (donor no-show, transport failure, quality reject)
* RE-PLANNING: a failed delivery puts the units back on the request, which is searched again with the
  deadline that is *left* (absolute deadline - now)
* requests that arrive over time (from the dataset or typed in by the user), queue policy, retry of requests
  that currently have no feasible source, and expiry of requests whose deadline passes
* a decision log (tier search, QUBO size, bit-string, QAOA samples) so every allocation can be explained

The solving machinery is reused unchanged from pipeline.py (tiered_search -> build_subproblem -> QUBO ->
QAOA / classical solver)."""
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Params, BLOOD_GROUPS, GROUP_FREQ, URGENCY_WEIGHT
from .preprocess import World, State, DONOR_KEY0, G
from . import pipeline as pl

METHOD_KEYS = pl.METHODS
METHOD_LABELS = pl.LABELS


@dataclass
class LiveParams:
    method: str = "qaoa"
    tick_min: int = 10
    horizon_min: int = 1440
    queue_policy: str = "arrival"          # 'arrival' (first come) or 'urgency' (most urgent first)
    retry_min: int = 30                    # re-try interval for a request with no feasible source
    restock_every_min: int = 240
    restock_units_per_hosp: float = 6.0    # mean units delivered to each hospital per restock
    shelf_life_min: int = 720              # remaining life of restocked units (compressed for demonstration)
    initial_shelf_lo: int = 120            # initial stock: remaining life drawn uniformly in [lo, hi]
    initial_shelf_hi: int = 1500
    donor_churn_per_hour: float = 0.20     # prob. per hour that a donor flips available <-> unavailable
    p_donor_noshow: float = 0.15
    p_transport_fail: float = 0.05
    p_quality_reject: float = 0.02
    noshow_detect_min: int = 30
    circuit_metrics: bool = True


class LiveSystem:
    def __init__(self, world: World, params: Params, live: LiveParams = None, seed: int = None):
        self.P = params
        self.L = live or LiveParams()
        self.seed = params.seed if seed is None else seed
        self.world = copy.copy(world)                       # shallow copy: only d_avail0 is mutated
        self.world.d_avail0 = world.d_avail0.copy()
        self.state = State(self.world)
        self.env = np.random.default_rng([self.seed, 101])  # environment randomness (same for every method)
        self.rng = np.random.default_rng([self.seed, 202])  # solver randomness (Grover / QAOA sampling)
        self.now = 0
        self.engine = "grover" if self.L.method == "qaoa" else "classical"

        self.lots, self._lot_id = {}, 0
        self.reqs, self.deliveries = {}, []
        self.events, self.decisions, self.series = [], [], []
        self.restocked_units = self.expired_units = self.discarded_units = 0
        self.init_stock = int(world.inv0.sum())
        self.oracle_queries = self.classical_evals = self.n_solves = 0
        self.solve_seconds = 0.0
        self._did = 0

        L = self.L
        for h in range(world.H):
            for g in range(G):
                n = int(world.inv0[h, g])
                while n > 0:
                    c = min(n, int(self.env.integers(1, 4)))
                    self._add_lot(h, g, c, int(self.env.integers(L.initial_shelf_lo, L.initial_shelf_hi + 1)))
                    n -= c
        self._sync()
        rq = world.requests
        self.schedule = [(int(rq.arrival_min[i]), i) for i in range(len(rq))]   # already sorted by arrival
        self._next_sched = 0
        self.next_restock = L.restock_every_min
        self._ucount = 0
        self._weight_h = world.inv0.sum(axis=1) / max(world.inv0.sum(), 1) * world.H
        self._freq = np.array([GROUP_FREQ[g] for g in BLOOD_GROUPS])
        self._arrivals()
        self._snapshot()

    # ------------------------------------------------------------------ helpers
    def source_label(self, skey):
        if skey >= DONOR_KEY0:
            d = skey - DONOR_KEY0
            return f"Donor {self.world.donors.donor_id[d]} ({BLOOD_GROUPS[self.world.d_grp[d]]})"
        h, g = divmod(int(skey), G)
        return f"Bank {self.world.hosp.hospital_id[h]} · {BLOOD_GROUPS[g]}"

    def _log(self, typ, text, rid=None):
        self.events.append(dict(time=self.now, type=typ, request=rid, detail=text))

    def _add_lot(self, h, g, units, expires):
        self._lot_id += 1
        self.lots[self._lot_id] = dict(id=self._lot_id, h=h, g=g, units=units, expires=expires)

    def _sync(self):
        inv = np.zeros_like(self.state.inv)
        for l in self.lots.values():
            inv[l["h"], l["g"]] += l["units"]
        self.state.inv[:] = inv

    def _consume(self, h, g, units):
        """First-expiry-first-out. Returns [(lot_id, units)]."""
        used = []
        for l in sorted((l for l in self.lots.values() if l["h"] == h and l["g"] == g and l["units"] > 0),
                        key=lambda l: l["expires"]):
            if units <= 0:
                break
            t = min(units, l["units"])
            l["units"] -= t
            units -= t
            used.append((l["id"], t))
        assert units == 0, "stock inconsistency"
        return used

    # ------------------------------------------------------------------ requests
    def _new_request(self, rid, hidx, gidx, units, urgency, deadline_min, source, num):
        w = URGENCY_WEIGHT[urgency]
        r = dict(rid=rid, num=num, h=int(hidx), hospital=self.world.hosp.hospital_id[hidx], g=int(gidx),
                 group=BLOOD_GROUPS[gidx], units=int(units), urgency=urgency, w=w, p=w,
                 deadline_min=int(deadline_min), arrival=self.now, abs_deadline=self.now + int(deadline_min),
                 rem=int(units), in_transit=0, delivered=0, status="Waiting", attempts=0, replans=0,
                 next_try=self.now, expired=False, first_delivery=None, done_at=None, source=source)
        self.reqs[rid] = r
        self._log("arrival", f"{units}× {BLOOD_GROUPS[gidx]} at {r['hospital']} · {urgency} urgency · "
                             f"deadline {deadline_min} min", rid)
        return r

    def add_request(self, hospital_id, group, units, urgency, deadline_min):
        self._ucount += 1
        rid = f"U{self._ucount:03d}"
        return self._new_request(rid, self.world.h_idx[hospital_id], self.world.g_idx[group], units, urgency,
                                 deadline_min, "user", 10_000 + self._ucount)

    def _arrivals(self):
        rq = self.world.requests
        while self._next_sched < len(self.schedule) and self.schedule[self._next_sched][0] <= self.now:
            _, i = self.schedule[self._next_sched]
            self._next_sched += 1
            self._new_request(rq.request_id[i], self.world.req_h[i], self.world.req_g[i], rq.units_requested[i],
                              rq.urgency[i], rq.deadline_min[i], "dataset", int(rq.request_id[i][1:]))
            self.reqs[rq.request_id[i]]["arrival"] = int(rq.arrival_min[i])      # true arrival (may be < now)
            r = self.reqs[rq.request_id[i]]
            r["abs_deadline"] = r["arrival"] + r["deadline_min"]
            r["next_try"] = self.now

    def _refresh_status(self, r):
        if r["delivered"] >= r["units"]:
            r["status"] = "Fulfilled"
            if r["done_at"] is None:
                r["done_at"] = self.now
        elif r["expired"]:
            r["status"] = "Partially fulfilled (expired)" if r["delivered"] > 0 else "Unmet (expired)"
        elif r["in_transit"] > 0 and r["rem"] == 0:
            r["status"] = "In transit"
        elif r["in_transit"] + r["delivered"] > 0:
            r["status"] = "Partially allocated"
        else:
            r["status"] = "Waiting"

    # ------------------------------------------------------------------ environment processes
    def _restock(self):
        if self.now < self.next_restock:
            return
        self.next_restock += self.L.restock_every_min
        lam = self.L.restock_units_per_hosp * self._weight_h[:, None] * self._freq[None, :]
        add = self.env.poisson(lam)
        tot = 0
        for h in range(self.world.H):
            for g in range(G):
                if add[h, g] > 0:
                    self._add_lot(h, g, int(add[h, g]), self.now + self.L.shelf_life_min)
                    tot += int(add[h, g])
        self.restocked_units += tot
        self._sync()
        self._log("restock", f"{tot} units delivered to blood banks (shelf life {self.L.shelf_life_min} min)")

    def add_stock(self, hospital_id, group, units, shelf_min=None):
        h, g = self.world.h_idx[hospital_id], self.world.g_idx[group]
        self._add_lot(h, g, int(units), self.now + int(shelf_min or self.L.shelf_life_min))
        self.restocked_units += int(units)
        self._sync()
        self._log("restock", f"manual: +{units}× {group} at {hospital_id}")

    def remove_stock(self, hospital_id, group, units):
        """Manually write off units (e.g. fridge failure) -> counted as discarded."""
        h, g = self.world.h_idx[hospital_id], self.world.g_idx[group]
        n = min(int(units), int(self.state.inv[h, g]))
        if n > 0:
            self._consume(h, g, n)
            self.discarded_units += n
            self._sync()
            self._log("discard", f"manual write-off: {n}× {group} at {hospital_id}")
        return n

    def _expire_stock(self):
        gone = [l for l in self.lots.values() if l["expires"] <= self.now]
        for l in gone:
            if l["units"] > 0:
                self.expired_units += l["units"]
                self._log("expiry", f"{l['units']}× {BLOOD_GROUPS[l['g']]} expired at "
                                    f"{self.world.hosp.hospital_id[l['h']]}")
            del self.lots[l["id"]]
        for l in [l for l in self.lots.values() if l["units"] == 0 and l["expires"] <= self.now + 1]:
            del self.lots[l["id"]]
        if gone:
            self._sync()

    def _donor_churn(self):
        u = self.env.random(len(self.world.donors))          # fixed-shape draw -> identical across methods
        p = min(1.0, self.L.donor_churn_per_hour * self.L.tick_min / 60.0)
        free = ~self.state.donor_used
        av = self.world.d_avail0
        flip = (u < p) & free & self.world.d_elig
        self.world.d_avail0 = np.where(flip, ~av, av)

    # ------------------------------------------------------------------ deliveries
    def _reserve(self, r, c, units, decision_id):
        skey, time = int(c["skey"]), float(c["time"])
        rr = np.random.default_rng([self.seed, r["num"], skey, r["replans"]])     # same fate for same (request, source, attempt)
        u1, u2, u3 = rr.random(3)
        outcome, resolve_at, lots = "ok", self.now + time, []
        if c["kind"] == 1:
            self.state.donor_used[skey - DONOR_KEY0] = True
            if u1 < self.L.p_donor_noshow:
                outcome, resolve_at = "donor_noshow", self.now + min(self.L.noshow_detect_min, time)
        else:
            h, g = divmod(skey, G)
            lots = self._consume(h, g, units)
            self._sync()
            if time > 0 and u1 < self.L.p_transport_fail:
                outcome, resolve_at = "transport_fail", self.now + time * (0.3 + 0.6 * u3)
            elif u2 < self.L.p_quality_reject:
                outcome = "quality_reject"
        self._did += 1
        d = dict(id=self._did, rid=r["rid"], skey=skey, source=self.source_label(skey), kind=int(c["kind"]),
                 tier=int(c["tier"]), units=int(units), created=self.now, eta=self.now + time,
                 resolve_at=resolve_at, outcome=outcome, status="pending", lots=lots, decision=decision_id,
                 exact=bool(c["exact"]))
        self.deliveries.append(d)
        r["rem"] -= units
        r["in_transit"] += units
        self._log("dispatch", f"{units}× from {d['source']} (tier {d['tier']}, ETA +{time:.0f} min)", r["rid"])

    def force_failure(self, delivery_id):
        for d in self.deliveries:
            if d["id"] == delivery_id and d["status"] == "pending":
                d["outcome"] = "donor_noshow" if d["kind"] == 1 else "transport_fail"
                d["resolve_at"] = self.now
                self._resolve()
                self._plan()
                self._resolve()
                return True
        return False

    def _resolve(self):
        for d in self.deliveries:
            if d["status"] != "pending" or d["resolve_at"] > self.now:
                continue
            r = self.reqs[d["rid"]]
            r["in_transit"] -= d["units"]
            if d["outcome"] == "ok":
                d["status"] = "delivered"
                d["done"] = d["eta"]
                r["delivered"] += d["units"]
                if r["first_delivery"] is None:
                    r["first_delivery"] = d["eta"]
                self._log("delivered", f"{d['units']}× from {d['source']}", r["rid"])
            else:
                d["status"] = "failed"
                if d["outcome"] == "donor_noshow":
                    k = d["skey"] - DONOR_KEY0
                    self.state.donor_used[k] = False
                    self.world.d_avail0[k] = False
                    why = "donor did not show up"
                elif d["outcome"] == "transport_fail":
                    for lid, t in d["lots"]:
                        if lid in self.lots:
                            self.lots[lid]["units"] += t
                        else:
                            self.expired_units += t
                    self._sync()
                    why = "transport failed - units returned to stock"
                else:
                    self.discarded_units += d["units"]
                    why = "unit rejected on quality check - discarded"
                r["rem"] += d["units"]
                r["replans"] += 1
                r["next_try"] = self.now
                self._log("failure", f"{d['units']}× from {d['source']}: {why} -> re-planning", r["rid"])
            self._refresh_status(r)

    def _expire_requests(self):
        for r in self.reqs.values():
            if r["rem"] > 0 and not r["expired"] and self.now > r["abs_deadline"]:
                r["expired"] = True
                self._log("deadline", f"deadline passed with {r['rem']} unit(s) still unallocated", r["rid"])
            self._refresh_status(r)

    # ------------------------------------------------------------------ planning
    def _plan(self):
        P, L = self.P, self.L
        todo = [r for r in self.reqs.values() if r["rem"] > 0 and not r["expired"]
                and r["next_try"] <= self.now and r["arrival"] <= self.now]
        if L.queue_policy == "urgency":
            todo.sort(key=lambda r: (-r["w"], r["abs_deadline"], r["arrival"]))
        else:
            todo.sort(key=lambda r: (r["arrival"], r["rid"]))
        for i in range(0, len(todo), P.batch_size):
            self._plan_batch(todo[i:i + P.batch_size])

    def _plan_batch(self, batch):
        import time as _t
        P, L, W = self.P, self.L, self.world
        blocked = set()
        for rnd in range(P.max_rounds):
            act = [r for r in batch if r["rem"] > 0 and r["rid"] not in blocked]
            if not act:
                break
            recips = [dict(rid=r["rid"], h=r["h"], g=r["g"], deadline=float(max(r["abs_deadline"] - self.now, 0)),
                           w=r["w"], p=r["p"], D=min(r["rem"], P.max_units_per_round)) for r in act]
            t0 = _t.perf_counter()
            feas, stats = pl.tiered_search(W, self.state, recips, P, self.engine, self.rng)
            self.oracle_queries += sum(s["oracle_queries"] for s in stats)
            self.classical_evals += sum(s["classical_evals"] for s in stats)
            cands = pl._cand_list(W, feas, recips, P) if len(feas) else []
            for j, r in enumerate(act):
                if not any(c["j"] == j for c in cands):
                    blocked.add(r["rid"])
                    r["attempts"] += 1
                    r["next_try"] = self.now + L.retry_min
                    self._log("wait", f"no feasible source before the deadline right now - retry in {L.retry_min} min", r["rid"])
            if not cands:
                break
            sub, info = None, {}
            if L.method in ("greedy", "ilp"):
                a = pl.solve_classical(L.method, recips, cands, P)
            else:
                sub = pl.build_subproblem(cands, recips, P)
                a, info = pl.solve_qubo_method(L.method, sub, P, self.rng, L.circuit_metrics)
            self.n_solves += 1
            self.solve_seconds += _t.perf_counter() - t0
            did = len(self.decisions) + 1
            by = {(c["skey"], c["j"]): c for c in cands}
            got = 0
            alloc_rows = []
            for (skey, j), u in a.items():
                c = by[(skey, j)]
                if c["kind"] == 0:
                    h, g = divmod(skey, G)
                    if self.state.inv[h, g] < u:
                        continue
                self._reserve(act[j], c, int(u), did)
                got += int(u)
                alloc_rows.append(dict(request=act[j]["rid"], source=self.source_label(skey), units=int(u),
                                       tier=int(c["tier"]), minutes=round(float(c["time"]), 1)))
            self._record_decision(did, act, recips, stats, cands, sub, info, alloc_rows, rnd)
            if got == 0:
                break

    def _record_decision(self, did, act, recips, stats, cands, sub, info, alloc_rows, rnd):
        P = self.P
        in_q = set()
        if sub is not None:
            for (s, j) in sub.pairs:
                in_q.add((sub.sources[s]["key"], sub.recips[j]["orig"]))
        got = {(row["request"], row["source"]): row["units"] for row in alloc_rows}
        ct = []
        for c in cands:
            rid = recips[c["j"]]["rid"]
            lab = self.source_label(c["skey"])
            ct.append(dict(request=rid, source=lab, tier=c["tier"], minutes=round(c["time"], 1),
                           group_match="identical" if c["exact"] else "compatible", w_sj=round(c["w"], 2),
                           net_gain=round(c["w"] - P.lam * c["time"] / 60.0, 3),
                           in_qubo=(c["skey"], c["j"]) in in_q, units_allocated=got.get((rid, lab), 0)))
        dec = dict(id=did, time=self.now, round=rnd, method=self.L.method, requests=[r["rid"] for r in act],
                   search=pd.DataFrame(stats), candidates=pd.DataFrame(ct), allocations=pd.DataFrame(alloc_rows),
                   n_qubits=info.get("n_qubits"), n_sources=info.get("n_sources"), bits=None, samples=None,
                   info={k: v for k, v in info.items() if not k.startswith("_")})
        if sub is not None and "_x" in info:
            q, x = info["_q"], info["_x"]
            rows = []
            for (s, j), bits in sorted(q.ylist.items()):
                rows.append(dict(variable=f"y  {self.source_label(sub.sources[s]['key'])} → {sub.recips[j]['rid']}",
                                 bits="".join(str(int(x[i])) for i, _ in bits),
                                 bit_weights=" + ".join(str(w) for _, w in bits),
                                 value=int(sum(w * int(x[i]) for i, w in bits))))
            for j, bits in q.ulist.items():
                rows.append(dict(variable=f"u  unmet units of {sub.recips[j]['rid']}",
                                 bits="".join(str(int(x[i])) for i, _ in bits),
                                 bit_weights=" + ".join(str(w) for _, w in bits),
                                 value=int(sum(w * int(x[i]) for i, w in bits))))
            for s, bits in q.zlist.items():
                rows.append(dict(variable=f"z  slack of {self.source_label(sub.sources[s]['key'])}",
                                 bits="".join(str(int(x[i])) for i, _ in bits),
                                 bit_weights=" + ".join(str(w) for _, w in bits),
                                 value=int(sum(w * int(x[i]) for i, w in bits))))
            dec["bits"] = pd.DataFrame(rows)
            dec["bitstring"] = "".join(str(int(v)) for v in x)
            E, F, n = info["_E"], info["_F"], q.n
            if "_samples" in info:
                idx, cnt = info["_samples"]
                order = np.argsort(-cnt)[:10]
                dec["samples"] = pd.DataFrame([dict(
                    bitstring="".join(str((int(idx[o]) >> k) & 1) for k in range(n)), shots=int(cnt[o]),
                    energy=round(float(E[idx[o]]), 3), feasible=bool(F[idx[o]])) for o in order])
            dec["E_opt"] = float(E[F].min())
        self.decisions.append(dec)
        if len(self.decisions) > 300:
            self.decisions.pop(0)

    # ------------------------------------------------------------------ clock
    def tick(self):
        self.now += self.L.tick_min
        self._arrivals()
        self._restock()
        self._expire_stock()
        self._donor_churn()
        self._resolve()
        self._expire_requests()
        self._plan()
        self._resolve()
        for r in self.reqs.values():
            self._refresh_status(r)
        self._snapshot()

    def advance(self, minutes, progress=None):
        target = min(self.now + int(minutes), self.L.horizon_min)
        steps = max(0, (target - self.now + self.L.tick_min - 1) // self.L.tick_min)
        for k in range(steps):
            self.tick()
            if progress:
                progress((k + 1) / steps)
        return steps

    @property
    def finished(self):
        return self.now >= self.L.horizon_min

    # ------------------------------------------------------------------ views
    def _snapshot(self):
        reqs = list(self.reqs.values())
        self.series.append(dict(
            time=self.now, stock=int(self.state.inv.sum()),
            donors_available=int((self.world.d_avail0 & self.world.d_elig & ~self.state.donor_used).sum()),
            open_requests=sum(1 for r in reqs if r["rem"] + r["in_transit"] > 0 and not r["expired"]),
            in_transit_units=sum(r["in_transit"] for r in reqs),
            delivered_units=sum(r["delivered"] for r in reqs),
            expired_units=self.expired_units,
            failed_deliveries=sum(1 for d in self.deliveries if d["status"] == "failed")))

    def kpis(self):
        reqs = list(self.reqs.values())
        ur = sum(r["units"] for r in reqs)
        dl = sum(r["delivered"] for r in reqs)
        wsum = sum(r["w"] * r["units"] for r in reqs)
        ok = [d for d in self.deliveries if d["status"] == "delivered"]
        late = sum(1 for d in ok if d["done"] > self.reqs[d["rid"]]["abs_deadline"] + 1e-9)
        wait = [(d["done"] - self.reqs[d["rid"]]["arrival"]) for d in ok for _ in range(d["units"])]
        return dict(
            time=self.now, requests=len(reqs),
            fulfilled=sum(1 for r in reqs if r["status"] == "Fulfilled"),
            unmet=sum(1 for r in reqs if r["expired"]),
            open=sum(1 for r in reqs if not r["expired"] and r["delivered"] < r["units"]),
            units_requested=ur, units_delivered=dl,
            fulfillment_rate=dl / ur if ur else 1.0,
            urgency_weighted_fulfillment=sum(r["w"] * r["delivered"] for r in reqs) / wsum if wsum else 1.0,
            high_urgency_unmet_units=sum(r["units"] - r["delivered"] for r in reqs
                                         if r["urgency"] == "High" and r["expired"]),
            failed_deliveries=sum(1 for d in self.deliveries if d["status"] == "failed"),
            replans=sum(r["replans"] for r in reqs), late_deliveries=late,
            mean_minutes_to_delivery=float(np.mean(wait)) if wait else 0.0,
            stock=int(self.state.inv.sum()), expired_units=self.expired_units,
            discarded_units=self.discarded_units, restocked_units=self.restocked_units,
            oracle_queries=self.oracle_queries, classical_evals=self.classical_evals,
            solves=self.n_solves, solve_seconds=self.solve_seconds)

    def requests_df(self):
        rows = [dict(request=r["rid"], hospital=r["hospital"], group=r["group"], urgency=r["urgency"],
                     units=r["units"], delivered=r["delivered"], in_transit=r["in_transit"], waiting=r["rem"],
                     arrived=r["arrival"], deadline_at=r["abs_deadline"], replans=r["replans"],
                     status=r["status"], source=r["source"]) for r in self.reqs.values()]
        return pd.DataFrame(rows)

    def deliveries_df(self):
        return pd.DataFrame([dict(id=d["id"], request=d["rid"], source=d["source"], tier=d["tier"], units=d["units"],
                                  dispatched=d["created"], eta=round(d["eta"], 1), status=d["status"],
                                  outcome=d["outcome"] if d["status"] == "failed" else "") for d in self.deliveries])

    def events_df(self):
        return pd.DataFrame(self.events)

    def series_df(self):
        return pd.DataFrame(self.series)

    def inventory_matrix(self):
        return pd.DataFrame(self.state.inv, index=self.world.hosp.hospital_id, columns=BLOOD_GROUPS)

    def lots_df(self):
        return pd.DataFrame([dict(hospital=self.world.hosp.hospital_id[l["h"]], group=BLOOD_GROUPS[l["g"]],
                                  units=l["units"], expires_in_min=l["expires"] - self.now)
                             for l in self.lots.values() if l["units"] > 0]).sort_values("expires_in_min") \
            if self.lots else pd.DataFrame(columns=["hospital", "group", "units", "expires_in_min"])

    def donors_df(self):
        d = self.world.donors[["donor_id", "blood_group", "lat", "lon", "nearest_hospital_id"]].copy()
        d["eligible"] = self.world.d_elig
        d["available_now"] = self.world.d_avail0
        d["used"] = self.state.donor_used
        d["state"] = np.where(d.used, "donated / reserved",
                      np.where(~d.eligible, "not eligible", np.where(d.available_now, "available", "unavailable")))
        return d

    def check_invariants(self):
        """Stock conservation and sanity checks used by the tests."""
        assert (self.state.inv >= 0).all(), "negative stock"
        in_stock = int(self.state.inv.sum())
        reserved_bank = sum(d["units"] for d in self.deliveries if d["status"] == "pending" and d["kind"] == 0)
        delivered_bank = sum(d["units"] for d in self.deliveries if d["status"] == "delivered" and d["kind"] == 0)
        discarded_q = sum(d["units"] for d in self.deliveries if d["status"] == "failed" and d["outcome"] == "quality_reject")
        lhs = self.init_stock + self.restocked_units
        rhs = in_stock + self.expired_units + self.discarded_units + reserved_bank + delivered_bank
        assert lhs == rhs, f"stock not conserved: {lhs} != {rhs}"
        for r in self.reqs.values():
            assert r["rem"] >= 0 and r["in_transit"] >= 0
            assert r["rem"] + r["in_transit"] + r["delivered"] == r["units"], f"{r['rid']} unit accounting"
        assert self.kpis()["late_deliveries"] == 0, "late delivery"
        return True
