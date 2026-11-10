"""BloodQ console - Streamlit interface for the hybrid Grover + QAOA blood allocation system.

Run:  streamlit run app.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from bloodq.config import Params, BLOOD_GROUPS, TIER_NAMES, URGENCY_WEIGHT        # noqa: E402
from bloodq.data_gen import generate_all                                          # noqa: E402
from bloodq.preprocess import World                                               # noqa: E402
from bloodq.pipeline import LABELS                                                # noqa: E402
from bloodq.dynamic import Simulation, DynParams                                  # noqa: E402

st.set_page_config(page_title="BloodQ Console", page_icon="🩸", layout="wide")

INK2, GRID = "#52514e", "#e4e3df"
URG_COL = {"High": "#a3200f", "Medium": "#e0603a", "Low": "#f0b49c"}            # one hue, light -> dark = urgency
LINE = {"stock": "#2a78d6", "in_transit": "#eda100", "delivered": "#1baf7a", "expired": "#eb6834"}
STATUS_BG = {"fulfilled": "#dff3e8", "lapsed": "#f9dcd8", "pending": "#fdf1d0", "scheduled": "#eeeeee"}
SCARCITY = {"Abundant (4.5)": 4.5, "Normal (3.0)": 3.0, "Tight (2.0)": 2.0, "Scarce (1.2)": 1.2}
METHOD_ORDER = ["qaoa", "qubo_exact", "qubo_sa", "ilp", "greedy"]


def hm(t):
    t = int(round(t))
    return f"{t // 60}h {t % 60:02d}m"


def style_fig(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK2), legend=dict(orientation="h", y=1.12, x=0), title_font=dict(size=14, color="#0b0b0b"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# --------------------------------------------------------------------------------------------- data / simulation plumbing
@st.cache_resource(show_spinner="Generating datasets...")
def get_world(seed: int, base: float):
    p = Params(seed=seed, inventory_base=base)
    d = ROOT / "data" if (seed == 42 and base == 4.5 and (ROOT / "data" / "requests.csv").exists()) else ROOT / "data_cache" / f"s{seed}_b{base}"
    if not (d / "requests.csv").exists():
        generate_all(p, str(d))
    return World(str(d), p)


def objs(cfg, method=None, seed=None):
    seed = cfg["seed"] if seed is None else seed
    p = Params(seed=seed, inventory_base=cfg["base"], **({"restarts": 2, "maxiter": 60} if cfg["fast"] else {}))
    dyn = DynParams(step_min=cfg["step"], restock_every_min=cfg["restock_every"], restock_scale=cfg["restock_scale"],
                    near_expiry_frac=cfg["near_expiry"], donor_online_frac=cfg["donor_online"], donor_noshow_prob=cfg["noshow"],
                    shipment_fail_prob=cfg["shipfail"], priority_batching=cfg["priority"])
    return get_world(seed, cfg["base"]), p, dyn, (method or cfg["method"])


def make_sim(cfg):
    w, p, dyn, m = objs(cfg)
    return Simulation(w, p, dyn, m)


def response_df(sim):
    rows = []
    for d in sim.deliveries:
        if d.status == "delivered":
            r = sim.reqs[d.req_idx]
            rows.append(dict(request=r.rid, urgency=r.urgency, response_min=d.arrive - r.arrival, units=d.units,
                             tier=d.tier, kind="donor" if d.kind else "bank"))
    return pd.DataFrame(rows, columns=["request", "urgency", "response_min", "units", "tier", "kind"])


@st.cache_data(show_spinner=False)
def cached_full_run(cfg, method, seed):
    w, p, dyn, m = objs(cfg, method, seed)
    s = Simulation(w, p, dyn, m)
    s.run_to_end()
    s.check_invariants()
    return dict(metrics=s.metrics(), timeline=s.timeline_df(), requests=s.requests_df(), deliveries=s.deliveries_df(),
                events=s.events_df(), response=response_df(s), horizon=dyn.horizon_min)


@st.cache_data(show_spinner=False)
def cached_compare(cfg, methods, n_seeds):
    rows = []
    for k in range(n_seeds):
        for m in methods:
            r = cached_full_run(cfg, m, cfg["seed"] + k)
            rows.append(dict(method=m, seed=cfg["seed"] + k, **r["metrics"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- sidebar: scenario
def sidebar():
    sb = st.sidebar
    sb.title("🩸 BloodQ console")
    sb.caption("Hybrid Grover + QAOA blood allocation, now with a live clock.")
    sb.subheader("Allocation method")
    method = sb.selectbox("Solver", METHOD_ORDER, format_func=lambda m: LABELS[m], key="w_method")
    fast = sb.checkbox("Fast QAOA settings (2 restarts, 60 iterations)", value=True, key="w_fast",
                       help="Full settings (3 restarts, 120 iterations) are ~4x slower with the same optimum on these small problems.")
    sb.subheader("Scenario")
    scar = sb.select_slider("Blood-bank stock level", list(SCARCITY), value="Tight (2.0)", key="w_scar")
    seed = int(sb.number_input("Random seed", 0, 9999, 42, key="w_seed"))
    sb.subheader("Dynamics")
    restock_every = sb.select_slider("Restock every (min)", [0, 120, 240, 360, 480, 720], value=360, key="w_rs",
                                     help="0 = never restock")
    restock_scale = sb.slider("Restock size (x initial stock)", 0.0, 1.5, 0.5, 0.1, key="w_rsc")
    near_expiry = sb.slider("Units that expire within 18 h", 0, 100, 25, 5, format="%d%%", key="w_ne") / 100
    donor_online = sb.slider("Offline donors that come online later", 0, 100, 50, 10, format="%d%%", key="w_do") / 100
    noshow = sb.slider("Donor no-show probability", 0, 60, 15, 5, format="%d%%", key="w_ns") / 100
    shipfail = sb.slider("Shipment failure probability", 0, 40, 5, 5, format="%d%%", key="w_sf") / 100
    priority = sb.checkbox("Urgent requests are batched first", value=True, key="w_pr",
                           help="Off = strict arrival order (as in the static batch pipeline).")
    step = sb.select_slider("Clock tick (min)", [5, 10, 15], value=5, key="w_step")
    cfg = dict(method=method, fast=fast, base=SCARCITY[scar], seed=seed, restock_every=float(restock_every),
               restock_scale=float(restock_scale), near_expiry=float(near_expiry), donor_online=float(donor_online),
               noshow=float(noshow), shipfail=float(shipfail), priority=bool(priority), step=float(step))
    apply = sb.button("Apply settings and restart", type="primary", key="b_apply", width="stretch")
    if apply or "sim" not in st.session_state:
        with st.spinner("Building simulation..."):
            st.session_state.sim = make_sim(cfg)
            st.session_state.cfg = cfg
            st.session_state.last_msg = None
    elif st.session_state.cfg != cfg:
        sb.warning("Settings changed. Press **Apply settings and restart** to use them in the live console.")
    sb.divider()
    sb.caption("Everything here is simulated. No real donor, patient or hospital data; not clinical software.")
    return cfg


# --------------------------------------------------------------------------------------------- shared views
def kpis(m, horizon):
    a = st.columns(4)
    a[0].metric("Clock", hm(m["time_min"]), f"of {hm(horizon)}", delta_color="off", delta_arrow="off")
    a[1].metric("Requests open", m["requests_open"], f"{m['requests_fulfilled']} fulfilled | {m['requests_lapsed']} lapsed", delta_color="off", delta_arrow="off")
    a[2].metric("Units delivered", f"{m['units_delivered']} / {m['units_requested']}", f"{m['units_in_transit']} in transit", delta_color="off", delta_arrow="off")
    a[3].metric("Urgency-weighted fulfilment", f"{m['urgency_weighted_fulfillment']:.1%}",
                f"{m['high_urgency_units_lapsed']} high-urgency units lapsed", delta_color="off", delta_arrow="off")
    b = st.columns(4)
    b[0].metric("Mean response", f"{m['mean_response_min']:.0f} min", "request -> delivery", delta_color="off", delta_arrow="off")
    b[1].metric("Failed deliveries", m["failed_deliveries"], f"{m['donor_noshows']} no-shows | {m['replans']} re-plans", delta_color="off", delta_arrow="off")
    b[2].metric("Units expired", m["units_expired"], f"{m['units_restocked']} restocked", delta_color="off", delta_arrow="off")
    b[3].metric("Units in blood banks", m["stock_now"], f"{m['planning_rounds']} planning rounds", delta_color="off", delta_arrow="off")


def stock_heatmap(df):
    fig = px.imshow(df.values, x=list(df.columns), y=list(df.index), text_auto=True, aspect="auto",
                    color_continuous_scale="Reds", labels=dict(color="units"))
    fig.update_layout(coloraxis_showscale=False, title="Blood-bank stock (units)")
    fig.update_xaxes(side="top")
    return style_fig(fig, 380)


def units_chart(tl):
    fig = go.Figure()
    for col, name in (("stock", "In blood banks"), ("in_transit", "In transit"), ("delivered", "Delivered (cumulative)"),
                      ("expired", "Expired (cumulative)")):
        fig.add_trace(go.Scatter(x=tl.t, y=tl[col], name=name, mode="lines", line=dict(color=LINE[col], width=2.2),
                                 hovertemplate="%{y} units<extra>" + name + "</extra>"))
    fig.update_layout(title="Blood units over time", hovermode="x unified")
    fig.update_xaxes(title="simulation minutes")
    return style_fig(fig, 320)


def waiting_chart(tl):
    fig = go.Figure(go.Scatter(x=tl.t, y=tl.waiting_requests, mode="lines", line=dict(color="#a3200f", width=2.2, shape="hv"),
                               hovertemplate="%{y} waiting<extra></extra>"))
    fig.update_layout(title="Requests waiting for supply", hovermode="x unified", showlegend=False)
    fig.update_xaxes(title="simulation minutes")
    return style_fig(fig, 300)


def donors_chart(tl):
    fig = go.Figure(go.Scatter(x=tl.t, y=tl.donors_online, mode="lines", line=dict(color="#2a78d6", width=2.2, shape="hv"),
                               hovertemplate="%{y} donors<extra></extra>"))
    fig.update_layout(title="Eligible donors online", hovermode="x unified", showlegend=False)
    fig.update_xaxes(title="simulation minutes")
    fig.update_yaxes(tickformat="d", rangemode="tozero")
    return style_fig(fig, 300)


def gantt(req, t_now):
    d = req[req.status != "scheduled"].copy()
    if d.empty:
        return None
    d["end"] = d.closed_at.fillna(t_now)
    d = d.sort_values("arrival")
    fig = go.Figure()
    for u in ("High", "Medium", "Low"):
        s = d[d.urgency == u]
        fig.add_trace(go.Bar(y=s.request, x=(s.end - s.arrival).clip(lower=1), base=s.arrival, orientation="h", name=u, marker_color=URG_COL[u],
                             customdata=np.stack([s.status, s.delivered, s.units, s.replans], axis=1),
                             hovertemplate="%{y} | %{customdata[0]}<br>delivered %{customdata[1]}/%{customdata[2]} units, %{customdata[3]} re-plans<extra>" + u + "</extra>"))
    fig.add_trace(go.Scatter(y=d.request, x=d.deadline_at, mode="markers", name="Deadline",
                             marker=dict(symbol="line-ns", size=11, color="#0b0b0b", line=dict(width=2, color="#0b0b0b"))))
    fig.update_layout(title="Request lifetimes (arrival -> closed, | = deadline)", barmode="overlay", bargap=0.25)
    fig.update_yaxes(categoryorder="array", categoryarray=list(d.request)[::-1], title=None, dtick=1, tickfont=dict(size=9))
    fig.update_xaxes(title="simulation minutes")
    return style_fig(fig, max(320, 14 * len(d) + 120))


def requests_table(req):
    d = req.copy()
    d["served"] = np.where(d.units > 0, d.delivered / d.units * 100, 0)
    d = d[["request", "hospital", "group", "urgency", "units", "delivered", "in_transit", "waiting", "served", "status", "arrival",
           "deadline_at", "closed_at", "replans", "source"]]
    sty = d.style.map(lambda v: f"background-color: {STATUS_BG.get(v, '')}", subset=["status"])
    st.dataframe(sty, hide_index=True, width="stretch", height=380, column_config={
        "served": st.column_config.ProgressColumn("served", min_value=0, max_value=100, format="%.0f%%"),
        "arrival": st.column_config.NumberColumn("arrival (min)", format="%.0f"),
        "deadline_at": st.column_config.NumberColumn("deadline at (min)", format="%.0f"),
        "closed_at": st.column_config.NumberColumn("closed at (min)", format="%.0f")})


def events_table(ev, key):
    types = sorted(ev.type.unique())
    pick = st.multiselect("Event types", types, default=types, key=key)
    d = ev[ev.type.isin(pick)].iloc[::-1].head(300)
    st.dataframe(d, hide_index=True, width="stretch", height=380)


def trace_view(sim, key):
    if not sim.trace:
        st.info("No planning round has happened yet. Advance the clock or submit a request.")
        return
    labels = [f"#{i}  t={hm(t['t'])}  {', '.join(t['requests'])}" + ("  (no supply)" if not t["allocations"] else "") for i, t in enumerate(sim.trace)]
    i = st.selectbox("Planning round", range(len(labels)), index=len(labels) - 1, format_func=lambda k: labels[k], key=key)
    tr = sim.trace[i]
    st.markdown("**1. Tiered search** - Grover looks for sources that pass compatibility, eligibility, availability and the deadline. "
                "The search stops expanding as soon as supply covers demand.")
    if tr["search"]:
        s = pd.DataFrame(tr["search"])
        s.insert(1, "location tier", s.tier.map(TIER_NAMES))
        st.dataframe(s[["tier", "location tier", "N", "M", "found", "oracle_queries", "classical_evals", "grover_qubits"]],
                     hide_index=True, width="stretch")
    else:
        st.caption("No candidate sources existed for this batch.")
    if tr["candidates"]:
        st.markdown(f"**2. Candidates** - {tr['n_candidates']} feasible (source, request) pairs. Reward = urgency weight (x0.9 if the group is only compatible); "
                    f"travel time is charged at lambda = {sim.p.lam} per hour.")
        c = pd.DataFrame(tr["candidates"]).sort_values(["request", "units_chosen", "tier"], ascending=[True, False, True])
        st.dataframe(c, hide_index=True, width="stretch", height=260)
    st.markdown("**3. Decision**")
    info = tr["info"]
    if info:
        cols = st.columns(5)
        cols[0].metric("QUBO qubits", info.get("n_qubits", "-"))
        cols[1].metric("Penalty P", f"{info.get('P', 0):.1f}")
        cols[2].metric("Optimal QUBO energy", f"{info.get('E_opt', 0):.2f}")
        if "p_opt" in info:
            cols[3].metric("QAOA P(optimal)", f"{info['p_opt']:.1%}", f"vs {info['n_opt_states'] / 2 ** info['n_qubits']:.2%} random", delta_color="off", delta_arrow="off")
            cols[4].metric("Infeasible shots", f"{info['shot_violation']:.0%}", "filtered classically", delta_color="off", delta_arrow="off")
        else:
            cols[3].metric("Solver", LABELS[tr["method"]].split(" (")[0])
            cols[4].metric("Repaired", "yes" if info.get("repaired") else "no")
    elif tr["allocations"]:
        st.caption(f"{LABELS[tr['method']]} works on the original objective directly, so no QUBO is built.")
    if tr["allocations"]:
        st.dataframe(pd.DataFrame(tr["allocations"]), hide_index=True, width="stretch")
    else:
        st.warning("Nothing could be allocated. The request stays queued and is retried when supply changes (restock, donor online, returned units).")


# --------------------------------------------------------------------------------------------- tab: live console
def live_tab():
    sim = st.session_state.sim
    cfg = st.session_state.cfg
    hz = sim.dyn.horizon_min
    st.markdown(f"**{LABELS[sim.method]}**  |  stock level {cfg['base']}  |  seed {cfg['seed']}")

    def adv(mins):
        with st.spinner(f"Simulating {mins:.0f} minutes..."):
            sim.advance(mins)
            if sim.finished:
                sim.run_to_end()
        st.session_state.last_msg = None

    bc = st.columns(6)
    for col, (lbl, mins) in zip(bc[:5], (("+5 min", 5), ("+30 min", 30), ("+2 h", 120), ("+6 h", 360), ("Run to end", hz))):
        if col.button(lbl, key=f"adv_{lbl}", width="stretch", disabled=sim.finished):
            adv(mins)
    if bc[5].button("Restart", key="b_restart", width="stretch"):
        st.session_state.sim = make_sim(cfg)
        st.session_state.last_msg = None
        st.rerun()
    sim = st.session_state.sim
    if sim.finished:
        st.info("Simulation reached the end of its horizon. Press Restart for a fresh run.")

    kpis(sim.metrics(), hz)
    if st.session_state.get("last_msg"):
        kind, msg = st.session_state.last_msg
        (st.success if kind == "ok" else st.warning)(msg)

    left, right = st.columns([3, 2])
    tl = sim.timeline_df()
    with left:
        st.plotly_chart(stock_heatmap(sim.stock_matrix()), width="stretch", key="live_heat")
        st.plotly_chart(units_chart(tl), width="stretch", key="live_units")
    with right:
        st.subheader("Actions")
        hosp = [f"{a} {b}" for a, b in zip(sim.world.hosp.hospital_id, sim.world.hosp.name)]
        with st.form("f_req"):
            st.markdown("**Submit a request now**")
            c1, c2 = st.columns(2)
            h = c1.selectbox("Hospital", range(len(hosp)), format_func=lambda i: hosp[i], key="f_h")
            g = c2.selectbox("Blood group", range(8), format_func=lambda i: BLOOD_GROUPS[i], index=1, key="f_g")
            c3, c4 = st.columns(2)
            u = c3.number_input("Units", 1, 6, 2, key="f_u")
            urg = c4.selectbox("Urgency", ["High", "Medium", "Low"], key="f_urg")
            dl = st.slider("Deadline (minutes from now)", 30, 1440, 240, 15, key="f_dl")
            if st.form_submit_button("Submit and plan", type="primary", disabled=sim.finished):
                n_before = len(sim.trace)
                r = sim.submit_request(h, g, u, urg, dl)
                if r.inflight + r.delivered > 0:
                    st.session_state.last_msg = ("ok", f"{r.rid}: {r.inflight + r.delivered} of {u} unit(s) dispatched" + (
                        f", {r.rem} still waiting for supply." if r.rem else ". Fully allocated."))
                else:
                    st.session_state.last_msg = ("warn", f"{r.rid}: no feasible source right now. It stays queued until supply changes or the deadline passes.")
                st.rerun()
        transit = [d for d in sim.deliveries if d.status == "transit"]
        with st.form("f_fail"):
            st.markdown("**Make a delivery fail** (tests re-planning)")
            did = st.selectbox("In-transit delivery", [d.did for d in transit], key="f_did",
                               format_func=lambda i: next(f"#{d.did} {d.rid} <- {sim.describe_source(d.skey)} ({d.units}u, ETA {d.arrive:.0f})" for d in transit if d.did == i),
                               placeholder="Nothing in transit") if transit else None
            if st.form_submit_button("Fail it", disabled=not transit):
                sim.fail_delivery(did)
                st.session_state.last_msg = ("warn", f"Delivery #{did} failed. Its units went back to the request and were re-planned.")
                st.rerun()
        with st.form("f_rs"):
            st.markdown("**Restock a blood bank**")
            c1, c2, c3 = st.columns(3)
            rh = c1.selectbox("Hospital", range(len(hosp)), format_func=lambda i: hosp[i].split(" ")[0], key="rs_h")
            rg = c2.selectbox("Group", range(8), format_func=lambda i: BLOOD_GROUPS[i], key="rs_g")
            rn = c3.number_input("Units", 1, 30, 5, key="rs_n")
            if st.form_submit_button("Add stock", disabled=sim.finished):
                sim.restock_now(rh, rg, rn)
                st.session_state.last_msg = ("ok", f"Added {rn} unit(s) of {BLOOD_GROUPS[rg]} at {hosp[rh].split(' ')[0]}. Waiting requests were re-planned.")
                st.rerun()

    c1, c2 = st.columns(2)
    c1.plotly_chart(waiting_chart(tl), width="stretch", key="live_wait")
    c2.plotly_chart(donors_chart(tl), width="stretch", key="live_donors")

    t1, t2, t3, t4 = st.tabs(["Requests", "In transit and deliveries", "Event log", "Decision trace (why this source?)"])
    with t1:
        requests_table(sim.requests_df())
    with t2:
        dd = sim.deliveries_df()
        st.dataframe(dd.iloc[::-1], hide_index=True, width="stretch", height=380)
    with t3:
        events_table(sim.events_df(), "live_ev")
    with t4:
        trace_view(sim, "live_trace")


# --------------------------------------------------------------------------------------------- tab: full run
def full_tab(cfg):
    st.markdown("Runs the whole horizon in one go with the sidebar settings and shows the analysis. Results are cached per setting.")
    c = st.columns([2, 1, 3])
    method = c[0].selectbox("Method", METHOD_ORDER, index=METHOD_ORDER.index(cfg["method"]), format_func=lambda m: LABELS[m], key="full_m")
    go_run = c[1].button("Run full simulation", type="primary", key="b_full", width="stretch")
    if go_run:
        st.session_state.full_key = (method,)
    if "full_key" not in st.session_state:
        st.info("Press **Run full simulation**.")
        return
    m = st.session_state.full_key[0]
    with st.spinner(f"Running {LABELS[m]} over the full horizon..."):
        res = cached_full_run(cfg, m, cfg["seed"])
    st.success(f"{LABELS[m]}  |  {res['metrics']['planning_rounds']} planning rounds  |  {res['metrics']['solver_seconds']:.1f} s of solver time  |  all stock, unit and deadline checks passed")
    kpis(res["metrics"], res["horizon"])
    c1, c2, c3_ = st.columns([2, 1, 1])
    c1.plotly_chart(units_chart(res["timeline"]), width="stretch", key="full_units")
    c2.plotly_chart(waiting_chart(res["timeline"]), width="stretch", key="full_wait")
    c3_.plotly_chart(donors_chart(res["timeline"]), width="stretch", key="full_donors")
    g = gantt(res["requests"], res["horizon"])
    if g:
        st.plotly_chart(g, width="stretch", key="full_gantt")
    c3, c4 = st.columns(2)
    dv = res["deliveries"]
    if len(dv):
        t = dv.groupby(["tier", "status"]).units.sum().reset_index()
        t["tier"] = "Tier " + t.tier.astype(str)
        fig = px.bar(t, x="tier", y="units", color="status", barmode="stack", title="Units dispatched by location tier",
                     color_discrete_map={"delivered": "#1baf7a", "failed": "#eb6834", "transit": "#eda100"})
        fig.update_layout(legend_title_text="")
        c3.plotly_chart(style_fig(fig, 320), width="stretch", key="full_tier")
    rs = res["response"]
    if len(rs):
        fig = px.box(rs, x="urgency", y="response_min", color="urgency", color_discrete_map=URG_COL, points="all",
                     category_orders={"urgency": ["High", "Medium", "Low"]}, title="Response time by urgency (min)")
        fig.update_layout(showlegend=False)
        c4.plotly_chart(style_fig(fig, 320), width="stretch", key="full_resp")
    a, b, c_ = st.tabs(["Requests", "Event log", "Downloads"])
    with a:
        requests_table(res["requests"])
    with b:
        events_table(res["events"], "full_ev")
    with c_:
        st.download_button("Events (CSV)", res["events"].to_csv(index=False), f"events_{m}.csv", "text/csv")
        st.download_button("Requests (CSV)", res["requests"].to_csv(index=False), f"requests_{m}.csv", "text/csv")
        st.download_button("Deliveries (CSV)", res["deliveries"].to_csv(index=False), f"deliveries_{m}.csv", "text/csv")


# --------------------------------------------------------------------------------------------- tab: compare
def compare_tab(cfg):
    st.markdown("Every method faces the **same simulated world** (same restocks, expiries, donor arrivals and failure outcomes). "
                "Single runs are noisy, so average over several seeds.")
    c = st.columns([3, 1, 1])
    methods = c[0].multiselect("Methods", METHOD_ORDER, default=METHOD_ORDER, format_func=lambda m: LABELS[m], key="cmp_m")
    n = c[1].number_input("Seeds", 1, 8, 3, key="cmp_n", help="Seeds run from the sidebar seed upwards.")
    if c[2].button("Compare", type="primary", key="b_cmp", width="stretch") and methods:
        st.session_state.cmp_key = (tuple(methods), int(n))
    if "cmp_key" not in st.session_state:
        st.info("Press **Compare**. The hybrid QAOA row takes the longest.")
        return
    methods, n = st.session_state.cmp_key
    with st.spinner("Running simulations..."):
        df = cached_compare(cfg, methods, n)
    cols = ["urgency_weighted_fulfillment", "fulfillment_rate", "high_urgency_units_lapsed", "mean_response_min", "failed_deliveries",
            "units_expired", "requests_lapsed", "solver_seconds"]
    agg = df.groupby("method")[cols].agg(["mean", "std"]).reindex(list(methods))
    show = pd.DataFrame({"Method": [LABELS[m] for m in agg.index]})
    names = {"urgency_weighted_fulfillment": "Urgency-weighted fulfilment", "fulfillment_rate": "Unit fulfilment",
             "high_urgency_units_lapsed": "High-urgency units lapsed", "mean_response_min": "Mean response (min)",
             "failed_deliveries": "Failed deliveries", "units_expired": "Units expired", "requests_lapsed": "Requests lapsed",
             "solver_seconds": "Solver time (s)"}
    for k, v in names.items():
        show[v] = [f"{a:.3f} +- {0 if pd.isna(b) else b:.3f}" if "fulfil" in v else f"{a:.1f} +- {0 if pd.isna(b) else b:.1f}"
                   for a, b in zip(agg[(k, "mean")], agg[(k, "std")])]
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption(f"Mean +- standard deviation over {n} seed(s). Lower is better except for the two fulfilment columns.")
    grid = st.columns(2)
    for slot, (k, title) in zip((grid[0], grid[1], grid[0], grid[1]),
                                (("urgency_weighted_fulfillment", "Urgency-weighted fulfilment (higher is better)"),
                                 ("mean_response_min", "Mean response time, min (lower is better)"),
                                 ("high_urgency_units_lapsed", "High-urgency units lapsed (lower is better)"),
                                 ("solver_seconds", "Solver time, s (lower is better)"))):
        d = pd.DataFrame({"method": [LABELS[m] for m in agg.index], "mean": agg[(k, "mean")].values, "sd": agg[(k, "std")].fillna(0).values})
        fig = px.bar(d, x="method", y="mean", error_y="sd", title=title, color="method",
                     color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"])
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        slot.plotly_chart(style_fig(fig, 300), width="stretch", key=f"cmp_{k}")
    st.warning("Read this honestly: at this problem size a classical ILP is as good or better than the hybrid method and much faster in simulation. "
               "Differences between methods are small compared with the seed-to-seed spread.")
    with st.expander("Per-run results"):
        st.dataframe(df, hide_index=True, width="stretch")


# --------------------------------------------------------------------------------------------- tab: datasets
def data_tab(cfg):
    w = get_world(cfg["seed"], cfg["base"])
    desc = {"Donor registry": (w.donors, "400 donors: group, age, sex, weight, haemoglobin, days since last donation, location, eligibility."),
            "Recipient requests": (w.requests, "Requests with group, units, urgency, arrival time and deadline (minutes after arrival)."),
            "Hospital blood-bank inventory": (pd.DataFrame(w.inv0, index=w.hosp.hospital_id, columns=BLOOD_GROUPS).reset_index(names="hospital"),
                                              "Starting stock per hospital and blood group."),
            "Compatibility (red cells)": (pd.DataFrame(w.compat.astype(int), index=BLOOD_GROUPS, columns=BLOOD_GROUPS).rename_axis("donor group ->  recipient group").reset_index(),
                                          "1 = the donor group can give red cells to the recipient group."),
            "Hospitals": (w.hosp, "Fictional hospital locations around Delhi-NCR."),
            "Hospital-to-hospital travel time (min)": (pd.DataFrame(w.hh_min.round(0), index=w.hosp.hospital_id, columns=w.hosp.hospital_id).reset_index(names="from"),
                                                       "Road time including 5 min dispatch.")}
    pick = st.selectbox("Dataset", list(desc), key="ds_pick")
    df, text = desc[pick]
    st.caption(text)
    st.dataframe(df, hide_index=True, width="stretch", height=420)
    if pick in ("Hospitals", "Donor registry"):
        pts = w.hosp[["lat", "lon"]] if pick == "Hospitals" else w.donors[["lat", "lon"]]
        st.map(pts, size=60)


# --------------------------------------------------------------------------------------------- tab: static results
def paper_tab():
    res = ROOT / "results"
    f = res / "system_comparison.csv"
    if not f.exists():
        st.info("Run `python experiments.py` to produce the static-pipeline results.")
        return
    st.markdown("Results of the original **static batch pipeline** (no clock): 36 requests, one stock level.")
    st.info("Bundled CSVs and figures are baseline snapshots from the previous simulator version. Run `python experiments.py` to regenerate them with the Qiskit circuit implementation.")
    d = pd.read_csv(f)
    keep = [c for c in ["label", "units_fulfilled", "units_requested", "urgency_weighted_fulfillment", "high_urgency_unmet_units",
                        "total_delivery_time_min", "deadline_violations", "objective_H", "runtime_s"] if c in d]
    st.dataframe(d[keep], hide_index=True, width="stretch")
    figs = sorted((res / "figures").glob("*.png"))
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, p in zip(cols, figs[i:i + 2]):
            col.image(str(p), caption=p.stem.replace("_", " "))


def about_tab():
    st.markdown("""
### How a request travels through the system
1. **Arrival** - a request (blood group, units, urgency, deadline) joins the queue. Deadlines run on the simulation clock.
2. **Batching** - each tick the most urgent waiting requests are grouped (two at a time, up to 3 units each per round).
3. **Tiered search** - own bank, nearby banks, nearby donors, distant banks, distant donors. **Grover** marks feasible sources
   (compatible, eligible, available, reachable before the *remaining* deadline).
4. **QUBO** - urgency reward, travel cost, unmet-demand penalty, plus penalties enforcing demand and stock limits; units are binary-encoded.
5. **QAOA** uses Qiskit circuits/Statevector for parameter optimisation and Qiskit Aer for bit-string sampling; the best *feasible* one is kept (classical validation).
6. **Dispatch** - stock is reserved FIFO (oldest units first); donors are booked.
7. **Clock events** - restocks, expiries, donors coming online, and failed deliveries. A failed delivery returns its units to the request
   (and to the bank if the shipment failed), the donor is dropped if they did not show up, and the request is **re-planned** with the time left.

### What is new compared with the static pipeline
Restocking, FIFO expiry, donors that come online later, delivery failures with re-planning, live requests, and urgent-first batching.

### Honest limits
Simulated data and Qiskit-simulated quantum circuits; the tier eligibility mask is currently computed classically before Grover circuit construction; QAOA sub-problems are capped at 14 qubits;
donors give one unit; red-cell compatibility only. This is a research prototype, not clinical software.
""")


def main():
    cfg = sidebar()
    st.title("Blood donor-recipient matching console")
    tabs = st.tabs(["Live console", "Full run", "Compare methods", "Datasets", "Static results", "How it works"])
    with tabs[0]:
        live_tab()
    with tabs[1]:
        full_tab(cfg)
    with tabs[2]:
        compare_tab(cfg)
    with tabs[3]:
        data_tab(cfg)
    with tabs[4]:
        paper_tab()
    with tabs[5]:
        about_tab()


main()
