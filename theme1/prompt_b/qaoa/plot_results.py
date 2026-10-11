"""Figures for the scaling study (reads the JSON written by qaoa_scaling.py).

Produces three PNGs:
  fig_depth.png   P(optimal), P(best 1%), expected ratio, and enhancement vs depth p,
                  one line per candidate-pool size n (mean +/- std over runs).
  fig_scaling.png Shots to reach 90% success vs N = C(n,k) on log-log axes, QAOA (median,
                  min-max over runs) against random sampling, with fitted slope.
  fig_cost.png    Total cost to reach ~90% success: QAOA (training + readout) vs simulated
                  annealing vs random sampling vs brute force.

Several JSON files can be passed (e.g. a small-n run with deep p and a larger-n run);
runs are pooled by n.

Run:  python plot_results.py --inputs results_small.json results_large.json \
          --outdir figures --depths 3 5 --cost-depth 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(paths):
    instances, meta = [], None
    for path in paths:
        with open(path) as fh:
            blob = json.load(fh)
        meta = meta or blob["meta"]
        instances.extend(blob["instances"])
    return instances, meta


def group_by_n(instances):
    by_n = defaultdict(lambda: {"N": None, "runs": [], "sa": [], "random_ratio": []})
    for ins in instances:
        g = by_n[ins["n"]]
        g["N"] = ins["N"]
        g["runs"].extend(ins["runs"])
        g["sa"].append(ins["sa"])
        g["random_ratio"].append(ins["random_ratio"])
    return dict(sorted(by_n.items()))


def random_shots(N: int, q: float = 0.9) -> float:
    return math.log(1 - q) / math.log(1 - 1.0 / N)


# ---------------------------------------------------------------------------

def fig_depth(by_n, top_frac, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(by_n)))
    panels = [("P(optimal subset)", "p_opt", True),
              (f"P(best {top_frac:.0%} of subsets)", "top", False),
              ("Expected approximation ratio (feasible-only)", "ratio", False),
              ("Enhancement over random = P(optimal) x N", "enh", True)]
    all_p = sorted({x["p"] for g in by_n.values() for x in g["runs"]})
    mean_random_ratio = float(np.mean([r for g in by_n.values() for r in g["random_ratio"]]))

    for ax, (title, key, log) in zip(axes.ravel(), panels):
        for color, (n, g) in zip(colors, by_n.items()):
            ps = sorted({x["p"] for x in g["runs"]})
            mean, std = [], []
            for p in ps:
                vals = [(x["p_opt"] * g["N"] if key == "enh" else x[key])
                        for x in g["runs"] if x["p"] == p]
                mean.append(np.mean(vals))
                std.append(np.std(vals))
            ax.errorbar(ps, mean, yerr=std, marker="o", capsize=3, color=color,
                        label=f"n={n}  (N={g['N']:,})")
            if key == "p_opt":
                ax.axhline(1.0 / g["N"], color=color, ls=":", alpha=0.45)
        if key == "top":
            ax.axhline(top_frac, color="gray", ls="--", label="random sampling")
        if key == "ratio":
            ax.axhline(mean_random_ratio, color="gray", ls="--", label="random subset")
        if key == "enh":
            ax.axhline(1.0, color="gray", ls="--", label="random sampling")
        if key == "p_opt":
            ax.text(0.98, 0.04, "dotted = random 1/N", transform=ax.transAxes,
                    ha="right", fontsize=8, color="gray")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("QAOA depth p")
        ax.set_xticks(all_p)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[1, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Exact QAOA (XY mixer, weight-5 subspace): quality vs depth", y=0.995)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_depth.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_scaling(by_n, depths, outdir):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    Ns = np.array([g["N"] for g in by_n.values()], dtype=float)
    grid = np.geomspace(Ns.min() * 0.8, Ns.max() * 1.3, 60)
    ax.plot(grid, [random_shots(int(N)) for N in grid], "k--", label="random sampling (slope 1)")
    report = []
    for d in depths:
        pts = []
        for n, g in by_n.items():
            s = [x["shots90"] for x in g["runs"] if x["p"] == d and np.isfinite(x["shots90"])]
            if s:
                pts.append((g["N"], float(np.median(s)), float(min(s)), float(max(s))))
        if not pts:
            continue
        N_, med, lo, hi = (np.array(v) for v in zip(*pts))
        label = f"QAOA p={d}"
        if len(pts) >= 3:
            slope = float(np.polyfit(np.log(N_), np.log(med), 1)[0])
            label += f"  (fitted slope {slope:.2f})"
            report.append((d, slope))
        ax.errorbar(N_, med, yerr=[med - lo, hi - med], marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("search-space size N = C(n, 5)")
    ax.set_ylabel("shots to see the optimum with 90% probability")
    ax.set_title("Shots-to-solution vs problem size")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_scaling.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, report


def sa_evals_to_90(curves):
    """Budget at which SA reaches 90% success, log-linearly interpolated.

    Returns (value, note): note is '' (interpolated), '<=' (already >= 90% at the
    smallest budget), or '>' (never reached 90% within the budgets tried).
    """
    budgets = sorted({int(b) for c in curves for b in c})
    succ = [float(np.mean([c[str(b)] for c in curves if str(b) in c])) for b in budgets]
    if succ[0] >= 0.9:
        return float(budgets[0]), "<="
    for i in range(1, len(budgets)):
        if succ[i] >= 0.9 > succ[i - 1]:
            t = (0.9 - succ[i - 1]) / (succ[i] - succ[i - 1])
            lb = math.log(budgets[i - 1]) + t * (math.log(budgets[i]) - math.log(budgets[i - 1]))
            return math.exp(lb), ""
    return float(budgets[-1]), ">"


def fig_cost(by_n, depth, outdir):
    rows = []
    for n, g in by_n.items():
        rp = [x for x in g["runs"] if x["p"] == depth]
        if not rp:
            continue
        sa, note = sa_evals_to_90(g["sa"])
        qaoa = float(np.mean([x["fd_equiv"] for x in rp]) + np.median([x["shots90"] for x in rp]))
        rows.append((n, g["N"], qaoa, sa, note, random_shots(g["N"])))
    if not rows:
        return None, []

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(rows))
    w = 0.2
    series = [("QAOA (training + readout)", [r[2] for r in rows], "tab:blue"),
              ("simulated annealing", [r[3] for r in rows], "tab:orange"),
              ("random sampling", [r[5] for r in rows], "tab:gray"),
              ("brute force (all N subsets)", [r[1] for r in rows], "tab:green")]
    for j, (name, vals, color) in enumerate(series):
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=name, color=color)
        for b, v, r in zip(bars, vals, rows):
            tag = f"{v:,.0f}"
            if name == "simulated annealing" and r[4]:
                tag = r[4] + tag
            ax.text(b.get_x() + b.get_width() / 2, v * 1.08, tag, ha="center",
                    fontsize=7, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={r[0]}\nN={r[1]:,}" for r in rows])
    ax.set_ylabel("evaluations to reach ~90% success (log scale)")
    ax.set_title(f"Total cost to find the optimum (QAOA at p={depth})")
    ax.set_ylim(top=ax.get_ylim()[1] * 4)
    ax.legend(fontsize=8, loc="upper left")
    fig.text(0.01, 0.005,
             "QAOA = training circuit runs (finite-difference equivalent) + readout shots; each "
             "circuit run needs many shots on hardware.\nSA is untuned; its cost is classical "
             "objective evaluations (microseconds each). '<=' / '>' mark bounds from the budgets tried.",
             fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(outdir, "fig_cost.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="JSON files from qaoa_scaling.py")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--depths", type=int, nargs="+", default=[3],
                    help="depths to show in the scaling plot")
    ap.add_argument("--cost-depth", type=int, default=3, help="QAOA depth in the cost figure")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    instances, meta = load(args.inputs)
    by_n = group_by_n(instances)
    print(f"loaded {len(instances)} instance(s) covering n = {list(by_n)}")

    print("saved", fig_depth(by_n, meta["top_frac"], args.outdir))
    path, slopes = fig_scaling(by_n, args.depths, args.outdir)
    print("saved", path)
    for d, s in slopes:
        print(f"  fitted log-log slope of QAOA shots-to-90% vs N at p={d}: {s:.2f} "
              f"(random sampling = 1.00)")
    path, rows = fig_cost(by_n, args.cost_depth, args.outdir)
    if path:
        print("saved", path)
        print(f"  cost table at p={args.cost_depth}  "
              f"(n, N, QAOA total, SA to 90%, random, brute force):")
        for n, N, q, sa, note, rnd in rows:
            print(f"    n={n:>2d} N={N:>6d}  QAOA {q:>9,.0f}  SA {note}{sa:>7,.0f}  "
                  f"random {rnd:>9,.0f}  brute force {N:>6d}")
    else:
        print(f"no runs at depth {args.cost_depth}; skipped fig_cost.png")


if __name__ == "__main__":
    main()