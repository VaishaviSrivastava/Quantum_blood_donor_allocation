"""Tests for the time-dynamic engine.  Run: `python test_dynamic.py`  (or `pytest test_dynamic.py`)."""
import os
import tempfile
from dataclasses import replace

import numpy as np

from bloodq.config import Params, BLOOD_GROUPS
from bloodq.data_gen import generate_all
from bloodq.preprocess import World
from bloodq.dynamic import Simulation, DynParams, run_full

_CACHE = {}


def world(base=1.2, seed=42, **kw):
    key = (base, seed, tuple(sorted(kw.items())))
    if key not in _CACHE:
        p = Params(inventory_base=base, seed=seed, **kw)
        d = tempfile.mkdtemp()
        generate_all(p, d)
        _CACHE[key] = (World(d, p), p)
    return _CACHE[key]


def test_invariants_all_fast_methods():
    for base in (4.5, 1.2):
        w, p = world(base)
        for m in ("ilp", "greedy", "qubo_exact", "qubo_sa"):
            s = run_full(w, p, DynParams(), m)
            s.check_invariants()
            assert s.metrics()["deadline_violations"] == 0
    print("OK  books balance (units, stock, donors, deadlines) for 4 methods x 2 stock levels")


def test_expiry_and_fifo():
    w, p = world(4.5)
    dyn = DynParams(near_expiry_frac=1.0, near_expiry_max_h=1.0, restock_every_min=0, donor_online_frac=0,
                    horizon_min=600, donor_noshow_prob=0, shipment_fail_prob=0)
    s = run_full(w, p, dyn, "ilp")
    s.check_invariants()
    m = s.metrics()
    assert s.state.inv.sum() == 0, "all units were set to expire within 1 hour"
    assert m["units_expired"] > 0 and s.expired_stock + s.dispatched_bank == s.initial_units
    # FIFO: a unit is never dispatched while an older one stays in the same lot list
    s2 = Simulation(w, p, DynParams(restock_every_min=0), "ilp")
    h, g = np.unravel_index(np.argmax(s2.state.inv), s2.state.inv.shape)
    before = list(s2.lots[(int(h), int(g))])
    r = s2.submit_request(h, g, 1, "High", 600, plan=True)
    after = s2.lots[(int(h), int(g))]
    if r.rem == 0 and s2.deliveries and s2.deliveries[0].kind == 0 and s2.deliveries[0].skey == int(h) * 8 + int(g):
        assert s2.deliveries[0].lots == before[:1], "oldest unit must leave first"
        assert after == before[1:]
    print(f"OK  expiry discards units ({m['units_expired']} expired) and stock leaves FIFO")


def test_restock():
    w, p = world(1.2)
    s = run_full(w, p, DynParams(restock_every_min=120, restock_scale=1.0), "ilp")
    s.check_invariants()
    assert s.restocked > 0 and s.events_df().type.eq("RESTOCK").any()
    s2 = Simulation(w, p, DynParams(restock_every_min=0), "ilp")
    n0 = int(s2.state.inv.sum())
    s2.restock_now(0, 1, 5)
    assert s2.state.inv.sum() == n0 + 5 and s2.restocked == 5
    s2.check_invariants()
    print(f"OK  restocking adds units ({s.restocked} over the run); manual restock +5 works")


def test_failures_force_replanning():
    w, p = world(1.2)
    dyn = DynParams(shipment_fail_prob=1.0, donor_noshow_prob=1.0)
    s = run_full(w, p, dyn, "ilp")
    s.check_invariants()
    ok_remote = [d for d in s.deliveries if d.status == "delivered" and (d.kind == 1 or d.tier != 1)]
    assert not ok_remote, "with 100% failure only own-hospital hand-overs can succeed"
    failed = [d for d in s.deliveries if d.status == "failed"]
    assert failed and s.metrics()["replans"] >= len(failed)
    for d in failed:   # a no-show donor is never used again
        if d.kind == 1:
            assert not s.state.donor_avail[d.skey - 10_000]
    # with failures OFF there must be no failures at all
    s0 = run_full(w, p, DynParams(shipment_fail_prob=0, donor_noshow_prob=0), "ilp")
    assert s0.metrics()["failed_deliveries"] == 0
    print(f"OK  failures trigger re-planning ({len(failed)} failed deliveries, none reused donors); 0 failures when disabled")


def test_live_request_and_forced_failure():
    w, p = world(1.2)
    s = Simulation(w, p, DynParams(shipment_fail_prob=0, donor_noshow_prob=0), "ilp")
    # find a (hospital, group) with no local stock so the request must be served remotely
    h, g = next((h, g) for h in range(w.H) for g in range(8) if s.state.inv[h, g] == 0 and g in (1, 3, 5))
    r = s.submit_request(h, g, 2, "High", 900)
    s.check_invariants()
    assert r.delivered + r.inflight + r.rem == 2 and r.inflight > 0, "live request must be planned immediately"
    d = next(d for d in s.deliveries if d.status == "transit")
    n_before = len(s.deliveries)
    s.fail_delivery(d.did)
    s.check_invariants()
    ev = s.events_df()
    assert ev.type.eq("FAILED").any() and ev.type.eq("REPLAN").any()
    assert len(s.deliveries) > n_before or r.status in ("lapsed",) or r.rem > 0, "failed units must be re-planned"
    print("OK  live request is planned instantly; forced failure -> FAILED + REPLAN + new dispatch")


def test_priority_batching():
    w, p = world(1.2, batch_size=1)
    s = Simulation(w, p, DynParams(restock_every_min=0, donor_online_frac=0), "ilp")
    s.state.donor_avail[:] = False
    for k in s.lots:
        s.lots[k] = []
        s._sync(*k)
    h = 6
    s.lots[(h, 1)] = [s.t + 99999]        # exactly one O+ unit in the whole system
    s._sync(h, 1)
    s.initial_units = 1                    # we edited the opening stock by hand, so re-base the books
    low = s.submit_request(h, 1, 1, "Low", 600, plan=False)
    high = s.submit_request(h, 1, 1, "High", 600, plan=False)
    s._plan()
    assert high.inflight == 1 and low.inflight == 0, "the High request must get the only unit"
    s.check_invariants()
    print("OK  priority batching gives the only unit to the High request even though Low arrived first")


def test_determinism_and_same_environment():
    w, p = world(1.2)
    a = run_full(w, p, DynParams(), "ilp").metrics()
    b = run_full(w, p, DynParams(), "ilp").metrics()
    assert {k: v for k, v in a.items() if k != "solver_seconds"} == {k: v for k, v in b.items() if k != "solver_seconds"}
    s1, s2 = Simulation(w, p, DynParams(), "ilp"), Simulation(w, p, DynParams(), "greedy")
    assert s1.restock_events == s2.restock_events and s1.donor_events == s2.donor_events and s1.lots == s2.lots
    print("OK  same seed -> identical results; different methods face the identical environment")


def test_qaoa_dynamic_smoke():
    w, p = world(1.2, restarts=1, maxiter=30)
    s = run_full(w, p, DynParams(horizon_min=600), "qaoa")
    s.check_invariants()
    assert any(t["info"].get("n_qubits") for t in s.trace), "QAOA must have been used in at least one round"
    print(f"OK  hybrid Grover+QAOA runs inside the dynamic loop ({len(s.trace)} planning rounds)")


if __name__ == "__main__":
    test_invariants_all_fast_methods()
    test_expiry_and_fifo()
    test_restock()
    test_failures_force_replanning()
    test_live_request_and_forced_failure()
    test_priority_batching()
    test_determinism_and_same_environment()
    test_qaoa_dynamic_smoke()
    print("ALL DYNAMIC TESTS PASSED")
