"""QAOA for feature selection using the library ``qiskit_algorithms.QAOA``.

Delegates the p-layer QAOA ansatz, the COBYLA optimizer, and the final sampling
to IBM's ``qiskit-algorithms`` package.  The QUBO construction, the
QUBO -> Ising -> ``SparsePauliOp`` translation, and the brute-force reference
are all done here (``qiskit-optimization`` could also do the conversion, but it
pulls a much larger dependency tree for no extra benefit).

Note: the library QAOA builds a *non-flattened* ``QAOAAnsatz`` whose cost and
mixer evolutions are nested ``PauliEvolutionGate``\\ s.  Left as-is, the sampler
evaluates ``PauliEvolutionGate.to_matrix()``, i.e. ``scipy.sparse.linalg.expm``
of the full 2**n x 2**n operator -- impractically slow for n = 13.  We pass a
small transpiler that decomposes the ansatz down to RZ / RZZ / RX gates before
sampling, which sidesteps that matrix-exponential fallback.

Run with::

    uv run python theme1/prompt_b/qaoa/qaoa.py
"""

from __future__ import annotations

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from data import load_heart_disease

from itertools import combinations 

# ---------------------------------------------------------------------------
# QUBO construction and brute-force reference
# ---------------------------------------------------------------------------

def feature_selection_qubo(
    relevance: np.ndarray,
    redundancy: np.ndarray,
    lam: float,
    penalty: float,
    k: int,
) -> dict[tuple[int, int], float]:
    """Build the feature-selection QUBO.

    Minimize, over ``x_i in {0, 1}`` ("keep feature i" = 1):

        -sum_i r_i x_i                          (maximize relevance)
        + lam * sum_{i<j} c_ij x_i x_j          (penalize redundancy)
        + penalty * (sum_i x_i - k)^2           (force exactly k features)

    ``relevance`` is ``r`` (n,), ``redundancy`` is the symmetric matrix ``c``
    (n, n).  Expanding the penalty gives

        Q_ii = -r_i + penalty * (1 - 2k)
        Q_ij = lam * c_ij + 2 * penalty        (i < j)
    """
    n = len(relevance)
    Q: dict[tuple[int, int], float] = {}
    for i in range(n):
        Q[(i, i)] = -float(relevance[i]) + penalty * (1.0 - 2.0 * k)
    for i in range(n):
        for j in range(i + 1, n):
            Q[(i, j)] = lam * float(redundancy[i, j]) + 2.0 * penalty
    return Q


def qubo_to_ising(
    Q: dict[tuple[int, int], float],
) -> tuple[float, np.ndarray, dict]:
    """Translate a QUBO into an Ising Hamiltonian.

    Substituting ``x_i = (1 - z_i) / 2`` with ``z_i in {+1, -1}`` (the eigenvalue
    of ``Z_i``) yields ``H_C = offset*I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j``
    with ``h_i = -Q_ii/2 - sum_{j!=i} Q_ij/4``, ``J_ij = Q_ij/4``, and
    ``offset = sum_i Q_ii/2 + sum_{i<j} Q_ij/4``.  Returns ``(offset, h, J)``.
    """
    n = max(max(i, j) for (i, j) in Q) + 1
    offset = 0.0
    h = np.zeros(n)
    J: dict[tuple[int, int], float] = {}

    for (i, j), q in Q.items():
        if i == j:
            offset += q / 2.0
            h[i] -= q / 2.0
        else:
            offset += q / 4.0
            h[i] -= q / 4.0
            h[j] -= q / 4.0
            J[(i, j)] = J.get((i, j), 0.0) + q / 4.0

    return offset, h, J


def qubo_value(Q: dict[tuple[int, int], float], x: np.ndarray | list[int]) -> float:
    """Evaluate a QUBO on a binary assignment ``x``."""
    return sum(coeff * x[i] * x[j] for (i, j), coeff in Q.items())


def brute_force(
    Q: dict[tuple[int, int], float], n: int
) -> tuple[float, list[int], float]:
    """Exhaustive minimum and maximum of a QUBO over 2**n assignments.

    Returns ``(best_value, best_x, worst_value)``; the range feeds the
    normalized approximation ratio.
    """
    best_val = float("inf")
    worst_val = float("-inf")
    best_x: list[int] = []
    for m in range(1 << n):
        x = [(m >> i) & 1 for i in range(n)]
        v = qubo_value(Q, x)
        if v < best_val:
            best_val, best_x = v, x
        if v > worst_val:
            worst_val = v
    return best_val, best_x, worst_val


# ---------------------------------------------------------------------------
# Ising operator + QAOA driver
# ---------------------------------------------------------------------------

def _ising_operator(
    n: int, offset: float, h: np.ndarray, J: dict[tuple[int, int], float]
) -> SparsePauliOp:
    """Build ``H_C`` as a ``SparsePauliOp`` with explicit qubit indices."""
    terms: list[tuple[str, list[int], float]] = []
    if not np.isclose(offset, 0.0):
        terms.append(("I", [0], offset))
    for i in range(n):
        if not np.isclose(h[i], 0.0):
            terms.append(("Z", [i], h[i]))
    for (i, j), jij in J.items():
        if not np.isclose(jij, 0.0):
            terms.append(("ZZ", [i, j], jij))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def _bitstring_to_features(bitstring: str, n: int) -> list[int]:
    """Qiskit bitstrings are big-endian (leftmost = q_{n-1}); map to x_0..x_{n-1}."""
    return [int(bitstring[n - 1 - i]) for i in range(n)]


class _DecomposeTranspiler:
    """Flatten nested ``PauliEvolutionGate``\\ s into RZ / RZZ / RX gates.

    ``qiskit_algorithms.QAOA`` keeps the cost/mixer evolutions as parameterized
    ``PauliEvolutionGate``\\ s; without decomposition the sampler computes their
    matrices via ``scipy.sparse.linalg.expm`` of the full 2**n operator.
    """

    def run(self, circuits, **options):
        if isinstance(circuits, list):
            return [self._decompose(c) for c in circuits]
        return self._decompose(circuits)

    @staticmethod
    def _decompose(circuit):
        return circuit.decompose(reps=2)


# ===========================================================================
# [NEW] BLOCK 1: correctness check + feasible-subspace reference
# ===========================================================================
 
def verify_hamiltonian(Q, operator: SparsePauliOp, n: int) -> None:   # [NEW]
    """Assert the Ising operator's diagonal equals the QUBO on every bitstring.
 
    H_C is diagonal in the computational basis, so we read its diagonal from a
    sparse matrix instead of building a dense 2**n x 2**n array. Basis index m
    has qubit i = bit i of m, matching x[i] = (m >> i) & 1.
    """
    diag = np.real(operator.to_matrix(sparse=True).diagonal())
    for m in range(1 << n):
        x = [(m >> i) & 1 for i in range(n)]
        assert np.isclose(diag[m], qubo_value(Q, x)), (
            f"Ising/QUBO mismatch at bitstring {m:0{n}b}"
        )
 
 
def feasible_table(Q, n: int, k: int) -> dict[tuple[int, ...], float]:  # [NEW]
    """QUBO value of every size-k subset (C(n, k) entries; 1287 for 13 choose 5)."""
    table: dict[tuple[int, ...], float] = {}
    for S in combinations(range(n), k):
        x = [1 if i in S else 0 for i in range(n)]
        table[S] = qubo_value(Q, x)
    return table

# ===========================================================================
# [NEW] BLOCK 2: distribution-based metrics (replaces the per-run metric code)
# ===========================================================================
 
def evaluate_distribution(                                  # [NEW]
    dist: dict[str, float],
    Q,
    n: int,
    k: int,
    f_min: float,
    f_max: float,
    opt_full: float,
    worst_full: float,
) -> dict:
    """Metrics from the final measured distribution (bitstring -> probability).
 
    Returns:
      p_feasible   total probability on bitstrings with exactly k ones
      p_optimal    probability on the best feasible subset
      cond_ratio   expected feasible-only ratio, conditioned on being feasible
      best_ratio   feasible-only ratio of the best sampled feasible subset
      best_subset  that subset (tuple of feature indices)
      legacy_ratio the ORIGINAL full-space ratio, now from the exact sampled
                   distribution rather than result.eigenvalue
    """
    f_span = f_max - f_min
    total = sum(dist.values())
 
    p_feas = 0.0
    p_opt = 0.0
    ratio_mass = 0.0
    exp_energy = 0.0
    best_v = float("inf")
    best_S: tuple[int, ...] = ()
 
    for bitstring, prob in dist.items():
        prob = prob / total
        x = _bitstring_to_features(bitstring, n)
        v = qubo_value(Q, x)
        exp_energy += prob * v
        if sum(x) == k:
            S = tuple(i for i in range(n) if x[i])
            p_feas += prob
            ratio_mass += prob * (f_max - v) / f_span
            if np.isclose(v, f_min):
                p_opt += prob
            if v < best_v:
                best_v, best_S = v, S
 
    return {
        "p_feasible": p_feas,
        "p_optimal": p_opt,
        "cond_ratio": ratio_mass / p_feas if p_feas > 0 else float("nan"),
        "best_ratio": (f_max - best_v) / f_span if best_S else float("nan"),
        "best_subset": best_S,
        "legacy_ratio": (worst_full - exp_energy) / (worst_full - opt_full),
    }
 
 
def _mean_std(rows: list[dict], key: str) -> tuple[float, float]:   # [NEW]
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))
 
 
def main() -> None:
    data = load_heart_disease()
    n = data.n_features
    k = 5                      # the clinic can collect 5 measurements
    lam = 0.75                  # redundancy penalty weight
    penalty = 2.0              # force exactly k features
    n_restarts = 5             # [NEW] independent QAOA runs per depth p
 
    Q = feature_selection_qubo(data.relevance, data.redundancy, lam, penalty, k)
    opt, opt_x, worst = brute_force(Q, n)
 
    offset, h, J = qubo_to_ising(Q)
    operator = _ising_operator(n, offset, h, J)
 
    # ---- [NEW] sanity checks and feasible-only reference --------------------
    verify_hamiltonian(Q, operator, n)
 
    table = feasible_table(Q, n, k)
    f_min, f_max = min(table.values()), max(table.values())
    f_span = f_max - f_min
    opt_subset = min(table, key=table.get)
    assert sum(opt_x) == k and np.isclose(opt, f_min), (
        "Penalty too small: the unconstrained optimum is not a size-k subset. "
        "Increase `penalty`."
    )
    random_feasible_ratio = float(np.mean([(f_max - v) / f_span for v in table.values()]))
    random_p_opt = 1.0 / len(table)
    p0_legacy_ratio = (worst - offset) / (worst - opt)   # untrained uniform state

    # ---- end NEW ------------------------------------------------------------
 
    print(f"UCI Heart Disease feature selection via qiskit_algorithms.QAOA "
          f"(n={n}, k={k}, lambda={lam}, P={penalty})")
    print(f"  {data.n_samples} patients, {n} features")
    print("  relevance (mutual information with label), descending:")
    for name, r in data.relevance_ranking():
        print(f"    {name:9s} {r:.4f}")
    print(f"  brute-force optimum = {opt:.6f}")
    print(f"    selected = {[data.feature_names[i] for i in range(n) if opt_x[i]]}")
    # ---- [NEW] reference numbers so the results have a floor and a ceiling --
    print(f"  full-space span (worst - opt)      : {worst - opt:.3f}")
    print(f"  feasible-only span (max - min)     : {f_span:.4f}")
    print(f"  untrained (p=0) legacy ratio       : {p0_legacy_ratio:.4f}   <- already high")
    print(f"  random feasible subset ratio       : {random_feasible_ratio:.4f}")
    print(f"  random P(optimal) = 1/{len(table)}          : {random_p_opt:.5f}")
    # ---- end NEW ------------------------------------------------------------
    print()

    # [CHANGED] The original created one sampler and ran one QAOA per p.
    # Now: n_restarts independent runs per p (own seeds), metrics from the
    # final distribution, then mean +/- std across restarts.
    for p in (1, 2, 3):
        rows: list[dict] = []
        total_evals = 0
        for r in range(n_restarts):
            seed = 100 + 10 * p + r
            algorithm_globals.random_seed = seed
            sampler = StatevectorSampler(default_shots=10000, seed=seed)
            qaoa = QAOA(
                sampler,
                COBYLA(maxiter=300),
                reps=p,
                transpiler=_DecomposeTranspiler(),
            )
            result = qaoa.compute_minimum_eigenvalue(operator)
            total_evals += result.cost_function_evals

            # [CHANGED] was: energy = float(np.real(result.eigenvalue)) (a
            # shot-noisy estimate) and a scan of result.eigenstate for the best
            # value. Now every metric comes from the sampled distribution.
            dist = {b: float(q) for b, q in result.eigenstate.items()}
            rows.append(evaluate_distribution(
                dist, Q, n, k, f_min, f_max, opt, worst
            ))
 
        # ---- [NEW] aggregate and report ------------------------------------
        best_row = max(rows, key=lambda row: row["best_ratio"])
        names = [data.feature_names[i] for i in best_row["best_subset"]]
        found_opt = sum(row["best_subset"] == opt_subset for row in rows)
 
        pf, pf_s = _mean_std(rows, "p_feasible")
        po, po_s = _mean_std(rows, "p_optimal")
        cr, cr_s = _mean_std(rows, "cond_ratio")
        br, br_s = _mean_std(rows, "best_ratio")
        lr, lr_s = _mean_std(rows, "legacy_ratio")
 
        print(f"QAOA p={p} on {n} qubits, {n_restarts} restarts")
        print(f"  P(feasible, weight={k})        : {pf:.3f} +/- {pf_s:.3f}")
        print(f"  P(optimal subset)            : {po:.4f} +/- {po_s:.4f}"
              f"   (random: {random_p_opt:.5f})")
        print(f"  feasible-only ratio, E[.|feasible]: {cr:.4f} +/- {cr_s:.4f}"
              f"   (random: {random_feasible_ratio:.4f})")
        print(f"  best sampled feasible ratio  : {br:.4f} +/- {br_s:.4f}")
        print(f"  restarts that sampled the optimum: {found_opt}/{n_restarts}")
        print(f"  best subset across restarts  : {names}")
        print(f"  legacy full-space ratio      : {lr:.4f} +/- {lr_s:.4f}"
              f"   (p=0 baseline: {p0_legacy_ratio:.4f})")
        print(f"  mean cost function evals     : {total_evals / n_restarts:.0f}")
        print()
        # ---- end NEW -------------------------------------------------------

'''
def main() -> None:
    data = load_heart_disease()
    n = data.n_features
    k = 5                      # the clinic can collect 5 measurements
    lam = 0.5                  # redundancy penalty weight
    penalty = 2.0              # force exactly k features

    Q = feature_selection_qubo(data.relevance, data.redundancy, lam, penalty, k)
    opt, opt_x, worst = brute_force(Q, n)

    offset, h, J = qubo_to_ising(Q)
    operator = _ising_operator(n, offset, h, J)

    print(f"UCI Heart Disease feature selection via qiskit_algorithms.QAOA "
          f"(n={n}, k={k}, lambda={lam}, P={penalty})")
    print(f"  {data.n_samples} patients, {n} features")
    print("  relevance (mutual information with label), descending:")
    for name, r in data.relevance_ranking():
        print(f"    {name:9s} {r:.4f}")
    print(f"  brute-force optimum = {opt:.6f}")
    print(f"    selected = {[data.feature_names[i] for i in range(n) if opt_x[i]]}")
    span = worst - opt
    print()

    sampler = StatevectorSampler(default_shots=10000, seed=42)
    for p in (1, 2, 3):
        algorithm_globals.random_seed = 100 + p   # seed QAOA's initial point
        qaoa = QAOA(
            sampler,
            COBYLA(maxiter=300),
            reps=p,
            transpiler=_DecomposeTranspiler(),
        )
        result = qaoa.compute_minimum_eigenvalue(operator)

        # <H_C> at the optimized parameters (the variational energy).
        energy = float(np.real(result.eigenvalue))

        # Cheapest QUBO assignment among the sampled bitstrings.
        best_val = float("inf")
        best_x: list[int] = []
        for bitstring in result.eigenstate:
            x = _bitstring_to_features(bitstring, n)
            v = qubo_value(Q, x)
            if v < best_val:
                best_val, best_x = v, x

        ar = (worst - energy) / span
        sampled_ratio = (worst - best_val) / span
        bm = result.best_measurement

        print(f"QAOA p={p} on {n} qubits (library)")
        print(f"  <H_C> at optimum     : {energy:.6f}")
        print(f"  approximation ratio  : {ar:.4f}")
        print(f"  best sampled value   : {best_val:.6f}  "
              f"selected={[data.feature_names[i] for i in range(n) if best_x[i]]}")
        print(f"  sampled ratio        : {sampled_ratio:.4f}")
        print(f"  most-likely bitstring: {bm['bitstring']}  (p={bm['probability']:.3f})")
        print(f"  cost function evals  : {result.cost_function_evals}")
        print()
'''

if __name__ == "__main__":
    main()
