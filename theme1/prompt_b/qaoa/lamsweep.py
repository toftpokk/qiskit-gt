import numpy as np
from qaoa import feature_selection_qubo, feasible_table
from data import load_heart_disease

data = load_heart_disease()
n, k = data.n_features, 5
r = data.relevance / data.relevance.max()
top5 = tuple(sorted(np.argsort(r)[::-1][:k]))

for lam in (0, 0.1, 0.25, 0.5, 1.0, 2.0):
    Q = feature_selection_qubo(r, data.redundancy, lam, penalty=0.0, k=k)
    table = feasible_table(Q, n, k)
    best = min(table, key=table.get)
    names = [data.feature_names[i] for i in best]
    print(f"lambda={lam:<4} best={names}  same as top-5 MI: {best == top5}")