"""QAOA for QUBO feature selection, implemented with IBM Qiskit primitives.

This module implements the Quantum Approximate Optimization Algorithm (QAOA)
from first principles on top of Qiskit's EstimatorV2 / SamplerV2 primitives.

The concrete problem targeted here is the feature-selection QUBO described in
``theme1/prompt_b/README.md``: choose exactly ``k`` features out of ``n`` so that
we maximize summed relevance while penalizing pairwise redundancy.

Notable design choices
----------------------
* No external solver: the QUBO -> Ising translation, the QAOA ansatz, the
  COBYLA outer loop, and the final sampling are all built here.
* ``StatevectorEstimator`` / ``StatevectorSampler`` give exact, noise-free
  simulation, which is the right ground truth to measure the *algorithm's*
  performance (as opposed to hardware noise).
* The relevance/redundancy terms are computed from the real UCI Heart Disease
  dataset in ``data.py`` (``load_heart_disease``); the algorithm itself is
  dataset-agnostic.

Run the demo with::

    uv run python theme1/prompt_b/qaoa/qaoa.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from data import load_heart_disease


# ---------------------------------------------------------------------------
# QUBO -> Ising translation
# ---------------------------------------------------------------------------

def qubo_to_ising(Q: dict[tuple[int, int], float]) -> tuple[float, np.ndarray, dict]:
    """Translate a QUBO into an Ising Hamiltonian.

    ``Q`` maps ``(i, j)`` with ``i <= j`` to the coefficient of ``x_i x_j``
    (``x_i in {0, 1}``).  Substituting ``x_i = (1 - z_i) / 2`` with
    ``z_i in {+1, -1}`` (the eigenvalue of ``Z_i``) yields

        H_C = offset * I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

    where

        h_i   = -Q_ii/2 - sum_{j != i} Q_ij/4
        J_ij  =  Q_ij/4          (i < j)
        offset = sum_i Q_ii/2 + sum_{i<j} Q_ij/4

    The offset is a constant energy shift that drops out of the ansatz but is
    kept so expectation values are directly comparable to QUBO costs.

    Returns ``(offset, h, J)``.
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
    (n, n).  Expanding the penalty term ``(sum x_i - k)^2`` gives the QUBO
    coefficients

        Q_ii  = -r_i + penalty * (1 - 2k)
        Q_ij  = lam * c_ij + 2 * penalty      (i < j)

    and an additive constant ``penalty * k^2`` that we drop (it does not
    change which assignment is optimal).
    """
    n = len(relevance)
    Q: dict[tuple[int, int], float] = {}
    for i in range(n):
        Q[(i, i)] = -float(relevance[i]) + penalty * (1.0 - 2.0 * k)
    for i in range(n):
        for j in range(i + 1, n):
            Q[(i, j)] = lam * float(redundancy[i, j]) + 2.0 * penalty
    return Q


def qubo_value(Q: dict[tuple[int, int], float], x: np.ndarray | list[int]) -> float:
    """Evaluate a QUBO on a binary assignment ``x``."""
    return sum(coeff * x[i] * x[j] for (i, j), coeff in Q.items())


def brute_force(
    Q: dict[tuple[int, int], float], n: int
) -> tuple[float, list[int], float]:
    """Exhaustive minimum and maximum of a QUBO over 2**n assignments.

    Only practical for small ``n``; here it is the reference used to compute
    the approximation ratio.  Returns ``(best_value, best_x, worst_value)``.
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
# QAOA ansatz
# ---------------------------------------------------------------------------

def _ising_observable(
    n: int, offset: float, h: np.ndarray, J: dict[tuple[int, int], float]
) -> SparsePauliOp:
    """Build the Pauli observable for H_C (including the constant offset).

    Uses ``SparsePauliOp.from_sparse_list`` with explicit qubit indices so the
    mapping to ``QuantumCircuit`` qubits is unambiguous (Qiskit's ``from_list``
    string convention is little-endian, i.e. rightmost character = qubit 0, and
    is easy to get wrong).
    """
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


def build_ansatz(
    n: int, h: np.ndarray, J: dict[tuple[int, int], float], p: int
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Build the p-layer QAOA circuit.

    Starting from ``|+>^n``, each layer applies the cost unitary
    ``exp(-i gamma H_C)`` followed by the mixer ``exp(-i beta sum_i X_i)``.
    Because ``H_C`` is diagonal in ``Z`` these exponentials decompose into
    single-qubit ``RZ`` and two-qubit ``RZZ`` gates:

        exp(-i gamma h_i Z_i)   = RZ(2 gamma h_i)
        exp(-i gamma J_ij Z_i Z_j) = RZZ(2 gamma J_ij)
        exp(-i beta X_i)        = RX(2 beta)

    Returns ``(circuit, gammas, betas)`` where ``gammas``/``betas`` are the
    ``Parameter`` objects in *logical* order (all gammas, then all betas).
    """
    gammas = [Parameter(f"gamma_{l}") for l in range(p)]
    betas = [Parameter(f"beta_{l}") for l in range(p)]

    qc = QuantumCircuit(n)
    qc.h(range(n))

    for l in range(p):
        # Cost layer: exp(-i gamma_l H_C)
        for i in range(n):
            if not np.isclose(h[i], 0.0):
                qc.rz(2.0 * gammas[l] * h[i], i)
        for (i, j), jij in J.items():
            if not np.isclose(jij, 0.0):
                qc.rzz(2.0 * gammas[l] * jij, i, j)
        # Mixer layer: exp(-i beta_l sum_i X_i)
        qc.rx(2.0 * betas[l], range(n))

    return qc, gammas, betas


# ---------------------------------------------------------------------------
# QAOA driver
# ---------------------------------------------------------------------------

@dataclass
class QaoaResult:
    """Outcome of a QAOA run."""
    p: int
    n: int
    optimal_value: float                       # brute-force minimum
    worst_value: float                         # brute-force maximum
    optimal_assignment: list[int]              # brute-force optimum bitstring
    final_value: float                         # <H_C> at the optimum (== QUBO cost)
    final_params: np.ndarray                   # optimized (gamma, beta)
    best_sampled_value: float                  # cheapest sampled assignment
    best_sampled_assignment: list[int]
    approximation_ratio: float                 # normalized, 1.0 = optimal
    sampled_ratio: float                       # normalized, 1.0 = optimal
    counts: dict[str, int] = field(default_factory=dict)
    nfev: int = 0

    def summary(self) -> str:
        lines = [
            f"QAOA p={self.p} on {self.n} qubits",
            f"  brute-force optimum   : {self.optimal_value:.6f}  {self.optimal_assignment}",
            f"  <H_C> at optimum     : {self.final_value:.6f}",
            f"  approximation ratio  : {self.approximation_ratio:.4f}",
            f"  best sampled value   : {self.best_sampled_value:.6f}  {self.best_sampled_assignment}",
            f"  sampled ratio        : {self.sampled_ratio:.4f}",
            f"  estimator calls      : {self.nfev}",
        ]
        return "\n".join(lines)


class QAOA:
    """QAOA for a QUBO, optimized with COBYLA and sampled with SamplerV2."""

    def __init__(self, Q: dict[tuple[int, int], float], p: int = 1):
        self.Q = Q
        self.p = p
        self.n = max(max(i, j) for (i, j) in Q) + 1
        self.offset, self.h, self.J = qubo_to_ising(Q)
        self._qc, self._gammas, self._betas = build_ansatz(self.n, self.h, self.J, p)
        self._observable = _ising_observable(self.n, self.offset, self.h, self.J)
        self._estimator = StatevectorEstimator()
        self._sampler = StatevectorSampler()

    # -- binding ---------------------------------------------------------
    def _bind(self, theta: np.ndarray) -> QuantumCircuit:
        """Bind logical ``[gammas..., betas...]`` to the ansatz (order-safe)."""
        mapping = {}
        for l in range(self.p):
            mapping[self._gammas[l]] = theta[l]
            mapping[self._betas[l]] = theta[self.p + l]
        return self._qc.assign_parameters(mapping)

    def _energy(self, theta: np.ndarray) -> float:
        bound = self._bind(theta)
        job = self._estimator.run([(bound, self._observable)])
        return float(np.ravel(job.result()[0].data.evs)[0])

    def _initial_point(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        gammas = rng.uniform(0.0, 2.0 * np.pi, self.p)
        betas = rng.uniform(0.0, np.pi, self.p)
        return np.concatenate([gammas, betas])

    # -- public API ------------------------------------------------------
    def optimize(self, seed: int = 0, maxiter: int = 200) -> tuple[np.ndarray, float, int]:
        """Minimize ``<H_C>`` with COBYLA; returns ``(theta, energy, nfev)``."""
        x0 = self._initial_point(seed)
        result = minimize(
            self._energy,
            x0,
            method="COBYLA",
            options={"maxiter": maxiter, "disp": False},
        )
        return np.asarray(result.x), float(result.fun), int(result.nfev)

    def sample(self, theta: np.ndarray, shots: int = 1024) -> dict[str, int]:
        """Sample the QAOA output state and return measured counts."""
        measured = self._bind(theta)
        measured.measure_all()
        job = self._sampler.run([(measured,)], shots=shots)
        return job.result()[0].data.meas.get_counts()

    def run(self, seed: int = 0, maxiter: int = 200, shots: int = 1024) -> QaoaResult:
        """Full pipeline: optimize, sample, and score against brute force."""
        optimal_value, optimal_x, worst_value = brute_force(self.Q, self.n)
        theta, energy, nfev = self.optimize(seed=seed, maxiter=maxiter)

        counts = self.sample(theta, shots=shots)
        best_val = float("inf")
        best_x: list[int] = []
        for bitstring, _ in counts.items():
            x = _bitstring_to_features(bitstring, self.n)
            v = qubo_value(self.Q, x)
            if v < best_val:
                best_val, best_x = v, x

        # Minimization: normalize to [0, 1] over [worst, optimal]; 1.0 = optimal.
        # This is well-defined even when QUBO values are negative.
        span = worst_value - optimal_value

        def ratio(value: float) -> float:
            if np.isclose(span, 0.0):
                return float("nan")
            return (worst_value - value) / span

        return QaoaResult(
            p=self.p,
            n=self.n,
            optimal_value=optimal_value,
            worst_value=worst_value,
            optimal_assignment=optimal_x,
            final_value=energy,
            final_params=theta,
            best_sampled_value=best_val,
            best_sampled_assignment=best_x,
            approximation_ratio=ratio(energy),
            sampled_ratio=ratio(best_val),
            counts=counts,
            nfev=nfev,
        )


def _bitstring_to_features(bitstring: str, n: int) -> list[int]:
    """Qiskit counts are big-endian (leftmost = q_{n-1}); return x_0..x_{n-1}."""
    return [int(bitstring[n - 1 - i]) for i in range(n)]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    # Load the UCI Heart Disease dataset and compute r / c from real data.
    data = load_heart_disease()
    n = data.n_features
    k = 5                      # the clinic can collect 5 measurements
    lam = 0.5                  # redundancy penalty weight
    penalty = 2.0              # force exactly k features

    relevance = data.relevance
    redundancy = data.redundancy
    Q = feature_selection_qubo(relevance, redundancy, lam, penalty, k)
    opt, opt_x, _ = brute_force(Q, n)

    print(f"UCI Heart Disease feature selection (n={n}, k={k}, "
          f"lambda={lam}, P={penalty})")
    print(f"  {data.n_samples} patients, {n} features")
    print("  relevance (mutual information with label), descending:")
    for name, r in data.relevance_ranking():
        print(f"    {name:9s} {r:.4f}")
    print(f"  brute-force optimum = {opt:.6f}")
    print(f"    selected = {[data.feature_names[i] for i in range(n) if opt_x[i]]}")
    print()

    for p in (1, 2, 3):
        solver = QAOA(Q, p=p)
        res = solver.run(seed=0, maxiter=300, shots=2048)
        print(res.summary())
        top = sorted(res.counts.items(), key=lambda kv: -kv[1])[:3]
        print("  top sampled bitstrings:", top)
        print()


if __name__ == "__main__":
    main()
