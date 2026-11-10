"""Build figures from results/*.csv.  Palette: validated reference categorical slots 1-5 (fixed order)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "results", "results/figures"
os.makedirs(F, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"qaoa": "#2a78d6", "qubo_exact": "#eb6834", "qubo_sa": "#1baf7a", "ilp": "#eda100", "greedy": "#e87ba4"}
MARK = {"qaoa": "o", "qubo_exact": "s", "qubo_sa": "^", "ilp": "D", "greedy": "v"}
LAB = {"qaoa": "Hybrid Grover + QAOA", "qubo_exact": "QUBO exact (brute force)", "qubo_sa": "QUBO simulated annealing",
       "ilp": "ILP (all candidates)", "greedy": "Greedy urgency-first"}
ORDER = ["qaoa", "qubo_exact", "qubo_sa", "ilp", "greedy"]

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{F}/{name}", dpi=160)
    plt.close(fig)


def fig_comparison():
    s = pd.read_csv(f"{R}/system_comparison.csv").set_index("method").loc[ORDER]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col, title, fmt in ((ax[0], "objective_H", "Objective value H (higher is better)", "{:.1f}"),
                               (ax[1], "total_delivery_time_min", "Total delivery time, min (lower is better)", "{:.0f}"),
                               (ax[2], "urgency_weighted_fulfillment", "Urgency-weighted fulfilment", "{:.3f}")):
        y = np.arange(len(ORDER))[::-1]
        a.barh(y, s[col].values, color=[COL[m] for m in ORDER], height=0.62)
        for yy, v in zip(y, s[col].values):
            a.text(v, yy, " " + fmt.format(v), va="center", fontsize=9, color=INK)
        a.set_yticks(y)
        a.set_yticklabels([LAB[m] for m in ORDER] if a is ax[0] else [])
        a.set_title(title)
        a.grid(axis="y", visible=False)
        a.set_xlim(0, s[col].max() * 1.18)
    save(fig, "fig1_method_comparison.png")


def fig_scarcity():
    d = pd.read_csv(f"{R}/scarcity_sweep.csv")
    levels = [4.5, 3.0, 2.0, 1.2]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col, title in ((ax[0], "urgency_weighted_fulfillment", "Urgency-weighted fulfilment"),
                          (ax[1], "high_urgency_unmet_units", "High-urgency units left unmet"),
                          (ax[2], "objective_H", "Objective value H")):
        for m in ORDER:
            g = d[d.method == m].groupby("inventory_base")[col].mean().reindex(levels)
            a.plot(range(len(levels)), g.values, color=COL[m], marker=MARK[m], lw=2, ms=6, label=LAB[m])
        a.set_xticks(range(len(levels)))
        a.set_xticklabels(["abundant", "", "", "scarce"])
        a.set_xlabel("blood-bank stock level (mean of 3 data seeds)")
        a.set_title(title)
    ax[0].legend(fontsize=8, loc="lower left")
    save(fig, "fig2_scarcity_sweep.png")


def fig_grover():
    g = pd.read_csv(f"{R}/grover_scaling.csv")
    a, b = g[g.kind == "n_scaling"], g[g.kind == "M_scaling"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].loglog(a.N, a.classical_expected, color=COL["greedy"], marker="s", lw=2, label="classical scan (expected)")
    ax[0].loglog(a.N, a.grover_queries, color=COL["qaoa"], marker="o", lw=2, label="Grover  (r+1 queries)")
    ax[0].set_xlabel("search-space size N = 2ⁿ (one marked item)")
    ax[0].set_ylabel("oracle queries")
    ax[0].set_title("Quadratic query gap: Grover vs classical")
    ax[0].legend()
    ax[1].loglog(b.M, b.classical_expected, color=COL["greedy"], marker="s", lw=2, label="classical scan (expected)")
    ax[1].loglog(b.M, b.bbht_mean_queries, color=COL["yellow"] if "yellow" in COL else COL["ilp"], marker="D", lw=2,
                 label="Grover, M unknown (BBHT)")
    ax[1].loglog(b.M, b.grover_queries, color=COL["qaoa"], marker="o", lw=2, label="Grover, M known")
    ax[1].set_xlabel("number of marked items M  (N = 1024)")
    ax[1].set_title("The advantage shrinks as M/N grows")
    ax[1].legend()
    save(fig, "fig3_grover_scaling.png")


def fig_tiers():
    s = pd.read_csv(f"{R}/search_qaoa.csv")
    t = s.groupby("tier")[["N", "M", "oracle_queries", "classical_evals"]].sum()
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.classical_evals, 0.38, color=COL["greedy"], label="classical scan (N evaluations)")
    ax.bar(x + 0.2, t.oracle_queries, 0.38, color=COL["qaoa"], label="Grover oracle queries")
    for xi, (N, M) in zip(x, zip(t.N, t.M)):
        ax.text(xi, max(t.loc[t.index[xi], ["classical_evals", "oracle_queries"]]) * 1.03, f"M/N={M / N:.2f}",
                ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tier {i}" for i in t.index])
    ax.set_ylabel("oracle evaluations (summed over run)")
    ax.set_title("Search cost per location tier (hybrid run)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, "fig4_grover_by_tier.png")


def fig_qaoa():
    d = pd.read_csv(f"{R}/qaoa_depth_sweep.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for (n, g), c in zip(d.groupby("n_qubits"), [COL["qaoa"], COL["qubo_exact"], COL["qubo_sa"]]):
        ax[0].plot(g.p, g.p_opt, color=c, marker="o", lw=2, label=f"{n} qubits")
        ax[0].axhline(g.p_opt_uniform.iloc[0], color=c, ls=":", lw=1.2)
        ax[1].plot(g.p, g.improvement, color=c, marker="o", lw=2, label=f"{n} qubits")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("QAOA depth p")
    ax[0].set_ylabel("P(sample = optimal allocation)")
    ax[0].set_title("Optimum probability (dotted = uniform guessing)")
    ax[0].legend()
    ax[1].set_xlabel("QAOA depth p")
    ax[1].set_ylabel("energy improvement over uniform (1 = optimal)")
    ax[1].set_title("Expected-energy improvement")
    save(fig, "fig5_qaoa_depth.png")


def fig_noise():
    n = pd.read_csv(f"{R}/noise_study.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    xs = lambda p: np.where(p == 0, 3e-5, p)
    g = n[n.algo == "Grover"]
    ax[0].semilogx(xs(g.p1.values), g.success, color=COL["qaoa"], marker="o", lw=2)
    ax[0].axhline(1 / 32, color=INK2, ls=":", lw=1.2)
    ax[0].text(3e-5, 1 / 32 + 0.02, "random guess (1/32)", fontsize=8.5, color=INK2)
    ax[0].set_title(f"Grover (5 qubits, depth {int(g.depth.iloc[0])}): success vs gate noise")
    ax[0].set_ylabel("P(marked item)")
    q = n[n.algo == "QAOA"]
    if len(q):
        ax[1].semilogx(xs(q.p1.values), q.improvement, color=COL["qubo_exact"], marker="s", lw=2)
        ax[1].set_title(f"QAOA ({int(q.n_qubits.iloc[0])} qubits, depth {int(q.depth.iloc[0])}): quality vs noise")
        ax[1].set_ylabel("energy improvement over uniform")
    for a in ax:
        a.set_xlabel("1-qubit depolarising error  (0 plotted at far left; 2-qubit error = 10x)")
    save(fig, "fig6_noise_sensitivity.png")


if __name__ == "__main__":
    fig_comparison(); fig_grover(); fig_tiers(); fig_qaoa(); fig_noise()
    if os.path.exists(f"{R}/scarcity_sweep.csv"):
        fig_scarcity()
    print("figures written to", F)
