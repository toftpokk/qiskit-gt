import numpy as np
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from collections import Counter
from data import load_heart_disease

data = load_heart_disease()
X, y, names = data.X, data.y, data.feature_names
n, k = X.shape[1], 5
DISCRETE = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
mask = [nm in DISCRETE for nm in names]
subsets = list(combinations(range(n), k))
idx_arr = [np.array(S) for S in subsets]
lams = [0.0, 0.75, 1.0, 1.5, 2.0, 3.0]          # 0.0 is the top-5 MI baseline
rng = np.random.default_rng(0)

def best_subset(r, c, lam):
    vals = [-r[ix].sum() + lam * c[np.ix_(ix, ix)].sum() / 2 for ix in idx_arr]
    return subsets[int(np.argmin(vals))]

def test_auc(S, Xtr, ytr, Xte, yte):
    S = list(S)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(Xtr[:, S], ytr)
    return roc_auc_score(yte, model.predict_proba(Xte[:, S])[:, 1])

results = {f"lam={l}": [] for l in lams}
results["all 13"], results["random 5"] = [], []
chosen = {f"lam={l}": Counter() for l in lams}

splitter = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=0)
for tr, te in splitter.split(X, y):
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    r = mutual_info_classif(Xtr, ytr, discrete_features=mask, random_state=0)
    r = r / r.max()
    c = np.abs(np.corrcoef(Xtr, rowvar=False))
    np.fill_diagonal(c, 0.0)
    for l in lams:
        S = best_subset(r, c, l)
        chosen[f"lam={l}"][S] += 1
        results[f"lam={l}"].append(test_auc(S, Xtr, ytr, Xte, yte))
    results["all 13"].append(test_auc(range(n), Xtr, ytr, Xte, yte))
    results["random 5"].append(np.mean([
        test_auc(subsets[i], Xtr, ytr, Xte, yte)
        for i in rng.choice(len(subsets), 20, replace=False)]))

print("Test AUC (mean +/- std over 30 splits)")
for name, v in results.items():
    print(f"  {name:10s} {np.mean(v):.3f} +/- {np.std(v):.3f}")

base = np.array(results["lam=0.0"])
print("\nPaired difference vs top-5 MI (positive = QUBO better)")
for l in lams[1:]:
    d = np.array(results[f"lam={l}"]) - base
    print(f"  lam={l}: {d.mean():+.4f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.4f} (s.e.)")

print("\nMost frequently selected subset per method")
for name, cnt in chosen.items():
    S, num = cnt.most_common(1)[0]
    print(f"  {name}: {[names[i] for i in S]}  ({num}/30 splits)")