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


if __name__ == "__main__":
    main()
