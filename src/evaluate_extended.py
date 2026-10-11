"""Does redundancy-aware selection beat top-k mutual information on the expanded pool?

Same protocol as evaluate.py, but over the nested expanded pools from
data_expanded.py. On each of ``--splits`` stratified train/test splits:

  * relevance and redundancy are computed on the TRAINING split only (no leakage),
  * the best size-k subset is found EXACTLY by scanning all C(n, k) subsets
    (vectorized; ~50k subsets at n = 25 is instant), for each lambda,
  * a logistic regression is fit on the training split and scored (AUC) on the test split.

lambda = 0 is the naive top-k mutual-information baseline.

Run:  uv run python src/evaluate_extended.py --n 13 20 25 --lam 0.75 1.5 3.0
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_expanded import load_expanded, relevance_and_redundancy
from sparse_qaoa import all_subsets, relevance_and_redundancy_sums


def test_auc(cols, Xtr, ytr, Xte, yte) -> float:
    cols = list(cols)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(Xtr[:, cols], ytr)
    return roc_auc_score(yte, model.predict_proba(Xte[:, cols])[:, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[13, 20, 25])
    ap.add_argument("--lam", type=float, nargs="+", default=[0.75, 1.5, 3.0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-seed", type=int, default=0,
                    help="seed for the order in which one-hot columns are added")
    ap.add_argument("--mi-repeats", type=int, default=5,
                    help="MI draws averaged per split (reduces estimator noise)")
    args = ap.parse_args()
    k = args.k
    lams = [0.0] + list(args.lam)               # 0.0 = top-k MI baseline
    keys = [f"lam={l:g}" for l in lams]

    for n in args.n:
        pool = load_expanded(n, order_seed=args.pool_seed, mi_repeats=args.mi_repeats)
        X, y = pool.X, pool.y
        subsets, _ = all_subsets(n, k)
        rng = np.random.default_rng(args.seed)

        auc = {key: [] for key in keys}
        auc["all"], auc["random"] = [], []
        picked = {key: Counter() for key in keys}
        n_meas = {key: [] for key in keys}
        same_as_topk = {key: 0 for key in keys}

        splitter = StratifiedShuffleSplit(n_splits=args.splits, test_size=0.3,
                                          random_state=args.seed)
        for tr, te in splitter.split(X, y):
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            r, c = relevance_and_redundancy(Xtr, ytr, pool.discrete, seed=args.seed,
                                            n_repeats=args.mi_repeats)
            rel, red = relevance_and_redundancy_sums(subsets, r, c)   # once per split

            topk = None
            for key, lam in zip(keys, lams):
                best = subsets[int(np.argmin(-rel + lam * red))]
                S = tuple(int(i) for i in best)
                if lam == 0.0:
                    topk = S
                picked[key][S] += 1
                n_meas[key].append(pool.distinct_measurements(S))
                same_as_topk[key] += int(S == topk)
                auc[key].append(test_auc(S, Xtr, ytr, Xte, yte))

            auc["all"].append(test_auc(range(n), Xtr, ytr, Xte, yte))
            draws = rng.choice(len(subsets), 20, replace=False)
            auc["random"].append(np.mean([test_auc(subsets[i], Xtr, ytr, Xte, yte)
                                          for i in draws]))

        print(f"\n=== n = {n} candidate features  (C({n},{k}) = {len(subsets)} subsets, "
              f"{args.splits} splits) ===")
        print("Test AUC, mean +/- std")
        for name in keys + ["all", "random"]:
            label = {"lam=0": "top-k MI (lam=0)", "all": f"all {n} features",
                     "random": f"random {k}"}.get(name, name)
            print(f"  {label:18s} {np.mean(auc[name]):.3f} +/- {np.std(auc[name]):.3f}")

        base = np.array(auc["lam=0"])
        print("Paired difference vs top-k MI (positive = QUBO better; s.e. is optimistic,"
              " splits overlap)")
        for key in keys[1:]:
            d = np.array(auc[key]) - base
            print(f"  {key:10s} {d.mean():+.4f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.4f}"
                  f"   subset == top-k in {same_as_topk[key]}/{args.splits} splits")
        print("Distinct underlying measurements in the selected subset (mean; max is "
              f"{k}) and most frequent subset")
        for key in keys:
            S, cnt = picked[key].most_common(1)[0]
            print(f"  {key:10s} {np.mean(n_meas[key]):.2f}   "
                  f"{[pool.names[i] for i in S]}  ({cnt}/{args.splits})")


if __name__ == "__main__":
    main()