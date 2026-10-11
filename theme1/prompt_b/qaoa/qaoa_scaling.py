"""Scaling study v2: exact sparse QAOA vs classical baselines, with multiple seeds.

Changes from v1 (marked [NEW] / [CHANGED] in comments):
  [NEW]     --seeds: several independent TRAINING seeds per instance; results are
            reported as mean +/- std (plus min/max) so a trend is not a single draw.
  [NEW]     --mi-seeds: several INSTANCE draws (different mutual-information seeds),
            to show results do not hinge on one estimator draw. MI is also averaged
            over --mi-repeats draws inside each instance (see data_expanded.py).
  [NEW]     --out results.json: every raw record is saved for plot_results.py.
  [CHANGED] Per-depth output is aggregated across seeds instead of one run's numbers.
  [CHANGED] Default SA budgets are finer so the cost comparison can interpolate the
            budget at which SA reaches 90% success.

Run (quick):   python qaoa_scaling.py --n 13 16 --pmax 3 --restarts 4 --seeds 0 1 2
Run (fuller):  python qaoa_scaling.py --n 13 16 20 --pmax 5 --seeds 0 1 2 3 4 \
                   --mi-seeds 0 1 --out results_small.json
Then:          python plot_results.py --inputs results_small.json --outdir figures
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np

from data_expanded import load_expanded
from sparse_qaoa import (SparseSubspaceQAOA, all_subsets, johnson_sparse,
                         subset_energies_fast, train_sparse)


# ---------------------------------------------------------------------------
# baselines and helper metrics (unchanged from v1)
# ---------------------------------------------------------------------------

def shots_to_success(p_hit: float, q: float) -> float:
    """Samples needed to see the target at least once with probability q."""
    if p_hit <= 0:
        return float("inf")
    if p_hit >= 1:
        return 1.0
    return math.ceil(math.log(1 - q) / math.log(1 - p_hit))


def top_fraction_probability(probs: np.ndarray, E: np.ndarray, frac: float) -> float:
    """Probability mass on the best ``frac`` of subsets (at least one subset)."""
    m = max(1, math.ceil(frac * len(E)))
    return float(probs[np.argpartition(E, m - 1)[:m]].sum())


def greedy_forward(r, c, lam, n, k) -> tuple[int, ...]:
    chosen: list[int] = []
    for _ in range(k):
        best_j, best_e = None, np.inf
        for j in range(n):
            if j in chosen:
                continue
            T = np.array(chosen + [j])
            e = -r[T].sum() + lam * c[np.ix_(T, T)].sum() / 2.0
            if e < best_e:
                best_j, best_e = j, e
        chosen.append(best_j)
    return tuple(sorted(chosen))


def simulated_annealing(E_hat, n, k, tab_list, budget, rnd, t0=0.1, t1=0.002) -> int:
    """Swap-move SA over size-k subsets using ``budget`` objective evaluations.

    Returns the rank (index) of the best subset seen. Temperatures are in units of
    the [0, 1]-rescaled energy; t0/t1 are untuned defaults.
    """
    def rank(S):
        return sum(tab_list[s][i + 1] for i, s in enumerate(S))

    cur = tuple(sorted(rnd.sample(range(n), k)))
    cur_idx = rank(cur)
    cur_e = E_hat[cur_idx]
    best_idx, best_e = cur_idx, cur_e
    for step in range(1, budget):
        T = t0 * (t1 / t0) ** (step / max(budget - 1, 1))
        pos = rnd.randrange(k)
        while True:
            j = rnd.randrange(n)
            if j not in cur:
                break
        cand = tuple(sorted(cur[:pos] + (j,) + cur[pos + 1:]))
        idx = rank(cand)
        e = E_hat[idx]
        d = e - cur_e
        if d <= 0 or rnd.random() < math.exp(-d / T):
            cur, cur_idx, cur_e = cand, idx, e
            if e < best_e:
                best_idx, best_e = idx, e
    return best_idx


# ---------------------------------------------------------------------------
# one instance (fixed n and MI seed) x several training seeds
# ---------------------------------------------------------------------------

def ms(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.4f} +/- {v.std():.4f}"


def run_instance(n: int, mi_seed: int, args) -> dict:
    k = args.k
    pool = load_expanded(n, order_seed=args.pool_seed, mi_seed=mi_seed,
                         mi_repeats=args.mi_repeats)
    r, c = pool.relevance, pool.redundancy
    subsets, tab = all_subsets(n, k)
    N = len(subsets)
    tab_list = tab.tolist()

    E = subset_energies_fast(subsets, r, c, args.lam)
    Emin, span = float(E.min()), float(E.max() - E.min())
    E_hat = (E - Emin) / span
    opt = int(np.argmin(E))
    opt_S = tuple(int(i) for i in subsets[opt])
    order = np.sort(E_hat)
    random_ratio = 1.0 - float(E_hat.mean())

    topk_S = tuple(sorted(int(i) for i in np.argsort(-r)[:k]))
    topk_idx = int(np.flatnonzero((subsets == np.array(topk_S)).all(axis=1))[0])
    topk_pct = float((E_hat < E_hat[topk_idx]).mean()) * 100.0

    print(f"\n{'=' * 78}\nn = {n}   N = C({n},{k}) = {N}   lambda = {args.lam}   "
          f"MI seed = {mi_seed}\n{'=' * 78}")
    print(f"  QUBO optimum    : {[pool.names[i] for i in opt_S]}  "
          f"({pool.distinct_measurements(opt_S)} distinct measurements)")
    print(f"  top-{k} MI subset : {[pool.names[i] for i in topk_S]}  "
          f"({pool.distinct_measurements(topk_S)} distinct measurements; "
          f"{'same as optimum' if topk_idx == opt else f'beats {topk_pct:.1f}% of subsets on the QUBO'})")
    print(f"  next-5 normalized gaps: " + ", ".join(f"{g:.4f}" for g in order[1:6]))
    print(f"  random ratio {random_ratio:.4f};  random P(optimal) = 1/{N}")

    B = johnson_sparse(subsets, n, k, tab)
    sim = SparseSubspaceQAOA(E_hat, B, n, k)

    runs = []
    for seed in args.seeds:                                   # [NEW] loop over training seeds
        rng = np.random.default_rng(1000 * seed + n)
        warm = None
        for p in range(1, args.pmax + 1):
            t0 = time.time()
            res = train_sparse(sim, p, rng, args.restarts, warm=warm)
            res.sort(key=lambda t: t[0])
            f_best, x_best, _ = res[0]
            warm = x_best
            nfev = sum(t[2] for t in res)
            probs = sim.probs(x_best)
            p_opt = float(probs[opt])
            runs.append(dict(
                seed=seed, p=p, p_opt=p_opt,
                top=top_fraction_probability(probs, E, args.top_frac),
                ratio=1.0 - f_best,
                restart_mean_ratio=float(np.mean([1.0 - t[0] for t in res])),
                nfev=nfev, fd_equiv=nfev * (2 * p + 1),
                shots90=shots_to_success(p_opt, 0.90),
                shots99=shots_to_success(p_opt, 0.99),
                seconds=time.time() - t0))
            print(f"    [seed {seed}] p={p}: P(opt) {p_opt:.4f}  "
                  f"top{args.top_frac:.0%} {runs[-1]['top']:.3f}  ratio {1.0 - f_best:.4f}  "
                  f"({runs[-1]['seconds']:.0f}s)", flush=True)

    # [CHANGED] aggregate across seeds instead of printing one run's numbers
    print(f"\n  QAOA, mean +/- std over {len(args.seeds)} training seed(s) "
          f"(random ratio {random_ratio:.4f}):")
    for p in range(1, args.pmax + 1):
        rp = [x for x in runs if x["p"] == p]
        po = [x["p_opt"] for x in rp]
        print(f"    p={p}: P(opt) {ms(po)} ({np.mean(po) * N:.0f}x random)   "
              f"top{args.top_frac:.0%} {ms([x['top'] for x in rp])}   "
              f"ratio {ms([x['ratio'] for x in rp])}   "
              f"shots90 median {np.median([x['shots90'] for x in rp]):.0f} "
              f"(random {shots_to_success(1 / N, .9):.0f})   "
              f"training ~{np.mean([x['fd_equiv'] for x in rp]):,.0f} circuit runs")

    # -- classical baselines (once per instance) ---------------------------------
    g = greedy_forward(r, c, args.lam, n, k)
    g_idx = int(np.flatnonzero((subsets == np.array(g)).all(axis=1))[0])
    greedy_evals = sum(n - i for i in range(k))
    print(f"  greedy forward: {'found the optimum' if g_idx == opt else 'missed the optimum'} "
          f"(ratio {1.0 - E_hat[g_idx]:.4f}, {greedy_evals} evaluations)")

    rnd = random.Random(args.seed_sa)
    Eh_list = E_hat.tolist()
    sa = {}
    for budget in args.budgets:
        hits = sum(simulated_annealing(Eh_list, n, k, tab_list, budget, rnd) == opt
                   for _ in range(args.sa_trials))
        sa[str(budget)] = hits / args.sa_trials
    print("  SA success at budget: " + "  ".join(f"B={b}: {v:.2f}" for b, v in sa.items()))

    return dict(
        n=n, N=N, mi_seed=mi_seed, lam=args.lam,
        opt_names=[pool.names[i] for i in opt_S], topk_names=[pool.names[i] for i in topk_S],
        distinct_opt=pool.distinct_measurements(opt_S),
        distinct_topk=pool.distinct_measurements(topk_S),
        gaps=[float(g_) for g_ in order[1:6]], random_ratio=random_ratio,
        greedy_found=bool(g_idx == opt), greedy_ratio=float(1.0 - E_hat[g_idx]),
        greedy_evals=greedy_evals, sa=sa, runs=runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[13, 16, 20])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.75)
    ap.add_argument("--pmax", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=4, help="random starts per depth")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="training seeds (independent QAOA trainings)")             # [NEW]
    ap.add_argument("--mi-seeds", type=int, nargs="+", default=[0],
                    help="instance draws: seed of the mutual-information estimate")  # [NEW]
    ap.add_argument("--mi-repeats", type=int, default=5,
                    help="MI draws averaged inside each instance")                    # [NEW]
    ap.add_argument("--pool-seed", type=int, default=0,
                    help="order in which one-hot columns are added")
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--sa-trials", type=int, default=100)
    ap.add_argument("--budgets", type=int, nargs="+",                                # [CHANGED]
                    default=[25, 50, 100, 200, 400, 800, 1600, 3200])
    ap.add_argument("--seed-sa", type=int, default=0)
    ap.add_argument("--out", type=str, default="results.json")                       # [NEW]
    args = ap.parse_args()

    instances = [run_instance(n, ms_, args) for n in args.n for ms_ in args.mi_seeds]

    meta = {k_: v for k_, v in vars(args).items()}
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "instances": instances}, fh, indent=1)
    print(f"\nSaved {len(instances)} instance(s) to {args.out}")

    print(f"\n{'=' * 78}\nSUMMARY at the deepest p = {args.pmax} (mean over seeds)\n{'=' * 78}")
    print(f"{'n':>3s} {'mi':>3s} {'N':>7s} {'P(opt)':>8s} {'xrandom':>8s} {'top1%':>7s} "
          f"{'ratio':>7s} {'shots90':>8s} {'greedy':>7s}")
    for ins in instances:
        rp = [x for x in ins["runs"] if x["p"] == args.pmax]
        po = np.mean([x["p_opt"] for x in rp])
        print(f"{ins['n']:>3d} {ins['mi_seed']:>3d} {ins['N']:>7d} {po:>8.4f} "
              f"{po * ins['N']:>8.0f} {np.mean([x['top'] for x in rp]):>7.3f} "
              f"{np.mean([x['ratio'] for x in rp]):>7.3f} "
              f"{np.median([x['shots90'] for x in rp]):>8.0f} "
              f"{'yes' if ins['greedy_found'] else 'no':>7s}")


if __name__ == "__main__":
    main()