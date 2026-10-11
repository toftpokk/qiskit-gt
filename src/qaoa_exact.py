"""Exact QAOA for feature selection in the weight-k subspace (NumPy), plus baselines.

Change markers used in this file (relative to qaoa_revised.py):
    [NEW]      added; nothing like it existed before
    [CHANGED]  replaces something in qaoa_revised.py (a comment says what)
    [REUSED]   same idea/formula as qaoa_revised.py, re-implemented here
No marker on a line means it belongs to the block whose banner is above it.

What this script does
  1. Builds the feature-selection energies for every size-k subset (C(13,5) = 1287).
  2. Simulates QAOA *exactly* inside that 1287-dimensional subspace with an XY
     (constraint-preserving) mixer and a Dicke-state start. No shots, no penalty.
  3. Trains it with many restarts + INTERP warm starts, reports exact metrics.
  4. Compares against random search, greedy, and simulated annealing.
  5. (If Qiskit is installed) cross-checks the result against an independent
     full 2**n-dimensional computation built from Qiskit's SparsePauliOp.

Run with::

    python qaoa_exact.py
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from data import load_heart_disease


# ===========================================================================
# [NEW] BLOCK 1: energies of all feasible subsets (no penalty term)
# ===========================================================================

def build_subspace(n: int, k: int):                                   # [NEW]
    """All size-k subsets (the Dicke subspace) and an index lookup."""
    subsets = list(combinations(range(n), k))
    index = {S: i for i, S in enumerate(subsets)}
    return subsets, index


def subset_energies(subsets, r, c, lam) -> np.ndarray:                # [CHANGED]
    """QUBO energy of every size-k subset.

    [CHANGED] was: a QUBO matrix that included P*(sum x - k)^2. Inside the
    constraint-preserving subspace that term is identically zero, so it is
    dropped: E(S) = -sum_{i in S} r_i + lam * sum_{i<j in S} c_ij.
    """
    E = np.empty(len(subsets))
    for a, S in enumerate(subsets):
        ix = np.array(S)
        E[a] = -r[ix].sum() + lam * c[np.ix_(ix, ix)].sum() / 2.0   # c has zero diagonal
    return E


# ===========================================================================
# [NEW] BLOCK 2: XY mixer in the subspace + exact QAOA state evolution
# ===========================================================================

def johnson_adjacency(subsets, index, n: int) -> np.ndarray:           # [NEW]
    """Complete-graph XY mixer restricted to weight-k states.

    (X_i X_j + Y_i Y_j)/2 swaps the bits of qubits i and j when they differ, so
    inside the weight-k subspace it connects two subsets that differ by exactly
    one swap (drop feature i, add feature j). That adjacency matrix is B.
    """
    N = len(subsets)
    B = np.zeros((N, N))
    for a, S in enumerate(subsets):
        inside = set(S)
        outside = [j for j in range(n) if j not in inside]
        for i in S:
            for j in outside:
                T = tuple(sorted((inside - {i}) | {j}))
                B[a, index[T]] = 1.0
    return B


class SubspaceQAOA:                                                    # [NEW]
    """Exact p-layer QAOA on the weight-k subspace.

    Cost layer  : psi <- exp(-i gamma E_hat) * psi      (diagonal, elementwise)
    Mixer layer : psi <- exp(-i beta B) psi              (B diagonalized once)
    Start state : uniform over all size-k subsets (the Dicke state).
    E_hat is the energy rescaled to [0, 1] over the feasible subsets.
    """

    def __init__(self, E_hat: np.ndarray, B: np.ndarray):
        self.E = E_hat
        self.N = len(E_hat)
        self.mix_evals, V = np.linalg.eigh(B)
        self.V = V                      # real orthogonal; used on real/imag parts
        self.Vt = np.ascontiguousarray(V.T)

    def _mix(self, psi: np.ndarray, beta: float) -> np.ndarray:
        z = self.Vt @ np.stack([psi.real, psi.imag], axis=1)          # to eigenbasis
        z = (z[:, 0] + 1j * z[:, 1]) * np.exp(-1j * beta * self.mix_evals)
        y = self.V @ np.stack([z.real, z.imag], axis=1)               # back
        return y[:, 0] + 1j * y[:, 1]

    def state(self, params: np.ndarray) -> np.ndarray:
        p = len(params) // 2
        gammas, betas = params[:p], params[p:]
        psi = np.full(self.N, 1.0 / math.sqrt(self.N), dtype=complex)
        for g, b in zip(gammas, betas):
            psi = np.exp(-1j * g * self.E) * psi
            psi = self._mix(psi, b)
        return psi

    def probs(self, params: np.ndarray) -> np.ndarray:
        return np.abs(self.state(params)) ** 2

    def expectation(self, params: np.ndarray) -> float:
        """Exact <H>; replaces the 10,000-shot estimate used before."""   # [CHANGED]
        return float(self.probs(params) @ self.E)


# ===========================================================================
# [NEW] BLOCK 3: training (multi-start + INTERP warm start)
# ===========================================================================

def interp(params: np.ndarray) -> np.ndarray:                          # [NEW]
    """INTERP warm start (Zhou et al. 2018): stretch depth-p angles to depth p+1."""
    p = len(params) // 2

    def stretch(a):
        ext = np.concatenate([[0.0], a, [0.0]])
        i = np.arange(1, p + 2)
        return (i - 1) / p * ext[i - 1] + (p - i + 1) / p * ext[i]

    return np.concatenate([stretch(params[:p]), stretch(params[p:])])


def train(sim: SubspaceQAOA, p: int, rng, n_random: int, warm=None):   # [CHANGED]
    """Run L-BFGS-B from random starts (and an INTERP start if ``warm`` given).

    [CHANGED] was: one COBYLA run on a shot-noisy cost. Now the cost is exact,
    so a gradient-based optimizer converges fully; many cheap restarts are used.
    Returns a list of (expectation, params, n_function_evals).
    """
    starts = []
    if warm is not None:
        starts.append(interp(warm))
    for _ in range(n_random):
        starts.append(np.concatenate([rng.uniform(0, 2 * np.pi, p),
                                      rng.uniform(0, 2 * np.pi, p)]))
    out = []
    for x0 in starts:
        res = minimize(sim.expectation, x0, method="L-BFGS-B",
                       options={"maxiter": 200})
        out.append((float(res.fun), res.x, int(res.nfev)))
    return out


# ===========================================================================
# [NEW] BLOCK 4: classical baselines on the same energies
# ===========================================================================

def greedy_forward(r, c, lam, n: int, k: int) -> tuple[int, ...]:     # [NEW]
    """Add, one at a time, the feature that lowers the objective most."""
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


def simulated_annealing(E_hat, subsets, index, n, k, budget, rng,      # [NEW]
                        t0=0.1, t1=0.002) -> int:
    """Swap-move SA over size-k subsets; ``budget`` objective evaluations.

    Returns the index of the best subset seen.
    """
    cur = int(rng.integers(len(subsets)))
    cur_e = E_hat[cur]
    best, best_e = cur, cur_e
    for step in range(1, budget):
        T = t0 * (t1 / t0) ** (step / max(budget - 1, 1))
        S = set(subsets[cur])
        i = int(rng.choice(list(S)))
        j = int(rng.choice([x for x in range(n) if x not in S]))
        cand = index[tuple(sorted((S - {i}) | {j}))]
        d = E_hat[cand] - cur_e
        if d <= 0 or rng.random() < math.exp(-d / T):
            cur, cur_e = cand, E_hat[cand]
            if cur_e < best_e:
                best, best_e = cur, cur_e
    return best


def shots_to_success(p_hit: float, q: float) -> float:                 # [NEW]
    """Samples needed to see the target at least once with probability q."""
    if p_hit <= 0:
        return float("inf")
    if p_hit >= 1:
        return 1.0
    return math.ceil(math.log(1 - q) / math.log(1 - p_hit))


# ===========================================================================
# [NEW] BLOCK 5: independent cross-check against Qiskit operators (full 2**n)
# ===========================================================================

def qiskit_crosscheck(r, c, lam, n, k, subsets, E_hat, Emin, span,    # [NEW]
                      sim: SubspaceQAOA, params: np.ndarray) -> None:
    """Re-run the same QAOA in the FULL 2**n space with Qiskit-built operators.

    Independent pieces: (a) the cost operator comes from the QUBO -> Ising
    conversion in qaoa_revised.py (checks that mapping with P = 0); (b) the XY
    mixer is built from Pauli XX/YY terms on all 2**n states (checks the
    subspace reduction and shows zero probability leaks out of weight k).
    This uses Qiskit *operators*, not gate-level circuits.
    """
    from scipy.sparse.linalg import expm_multiply
    from qiskit.quantum_info import SparsePauliOp
    from qaoa import feature_selection_qubo, qubo_to_ising, _ising_operator

    Q = feature_selection_qubo(r, c, lam, penalty=0.0, k=k)
    offset, h, J = qubo_to_ising(Q)
    H_C = _ising_operator(n, offset, h, J)
    diag = np.real(H_C.to_matrix(sparse=True).diagonal())
    E_full = (diag - Emin) / span

    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            terms += [("XX", [i, j], 0.5), ("YY", [i, j], 0.5)]
    H_M = SparsePauliOp.from_sparse_list(terms, num_qubits=n).to_matrix(sparse=True)

    members = np.array([sum(1 << i for i in S) for S in subsets])   # bit i = qubit i
    psi = np.zeros(2 ** n, dtype=complex)
    psi[members] = 1.0 / math.sqrt(len(subsets))

    p = len(params) // 2
    for g, b in zip(params[:p], params[p:]):
        psi = np.exp(-1j * g * E_full) * psi
        psi = expm_multiply(-1j * b * H_M, psi)

    probs_full = np.abs(psi) ** 2
    probs_sub = sim.probs(params)
    print("  Qiskit-operator cross-check (full 2**n space):")
    print(f"    max |E_full - E_hat| on feasible states : "
          f"{np.max(np.abs(E_full[members] - E_hat)):.2e}")
    print(f"    probability leaked outside weight-{k}     : "
          f"{1.0 - probs_full[members].sum():.2e}")
    print(f"    max |prob_full - prob_subspace|         : "
          f"{np.max(np.abs(probs_full[members] - probs_sub)):.2e}")


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    data = load_heart_disease()
    n = data.n_features
    k = 5
    lam = 0.75                 # [CHANGED] was 0.5; chosen from the AUC comparison
    # [CHANGED] `penalty` no longer exists: the XY mixer keeps every state at weight k.
    p_max = 3
    n_random = 10              # [NEW] random restarts per depth
    p_max_note = f"p = 1..{p_max}"
    rng = np.random.default_rng(0)

    r, c = data.relevance, data.redundancy

    subsets, index = build_subspace(n, k)
    N = len(subsets)
    E = subset_energies(subsets, r, c, lam)
    Emin, Emax = float(E.min()), float(E.max())
    span = Emax - Emin
    E_hat = (E - Emin) / span                      # [NEW] rescaled to [0, 1]
    opt = int(np.argmin(E))
    opt_names = [data.feature_names[i] for i in subsets[opt]]

    order = np.argsort(E)                          # [NEW] how hard is the needle?
    gaps = [(E[order[i]] - Emin) / span for i in range(1, 6)]
    random_ratio = 1.0 - float(E_hat.mean())

    print(f"Exact QAOA, XY mixer, weight-{k} subspace (n={n}, N={N} subsets, "
          f"lambda={lam}, {p_max_note})")
    print(f"  optimum subset      : {opt_names}")
    print(f"  normalized gaps of the next 5 best subsets: "
          + ", ".join(f"{g:.4f}" for g in gaps))
    print(f"  random subset ratio : {random_ratio:.4f}    "
          f"random P(optimal) = 1/{N} = {1 / N:.5f}")
    print()

    B = johnson_adjacency(subsets, index, n)
    sim = SubspaceQAOA(E_hat, B)

    results: dict[int, dict] = {}
    warm = None
    for p in range(1, p_max + 1):
        runs = train(sim, p, rng, n_random, warm=warm)
        runs.sort(key=lambda t: t[0])
        best_f, best_x, _ = runs[0]
        total_evals = sum(t[2] for t in runs)
        warm = best_x

        probs = sim.probs(best_x)
        p_opt = float(probs[opt])
        ratios = [1.0 - f for f, _, _ in runs]
        results[p] = {"x": best_x, "p_opt": p_opt, "evals": total_evals,
                      "ratio": 1.0 - best_f}

        print(f"QAOA p={p}  ({len(runs)} restarts, {total_evals} cost evaluations)")
        print(f"  expected feasible ratio  : {1.0 - best_f:.4f}   "
              f"(random: {random_ratio:.4f}; restarts mean {np.mean(ratios):.4f}"
              f" +/- {np.std(ratios):.4f})")
        print(f"  P(optimal subset)        : {p_opt:.4f}   "
              f"({p_opt * N:.1f}x random)")
        print(f"  shots for 90% / 99% success: {shots_to_success(p_opt, 0.90):.0f} / "
              f"{shots_to_success(p_opt, 0.99):.0f}   "
              f"(random: {shots_to_success(1 / N, 0.90):.0f} / "
              f"{shots_to_success(1 / N, 0.99):.0f})")
        print()

    # ---- baselines -----------------------------------------------------
    greedy = greedy_forward(r, c, lam, n, k)
    g_idx = index[greedy]
    print("Classical baselines (same objective)")
    print(f"  greedy forward : {[data.feature_names[i] for i in greedy]}   "
          f"ratio {1.0 - E_hat[g_idx]:.4f}   "
          f"{'= optimum' if g_idx == opt else 'not optimal'}")

    budgets = [25, 100, 400, 1000]
    trials = 300
    p_best = max(results, key=lambda q: results[q]["p_opt"])
    p_hit = results[p_best]["p_opt"]
    print(f"\n  P(find optimum) at matched sample budgets "
          f"(QAOA = p={p_best} readout only; SA over {trials} trials)")
    print(f"  {'budget':>8s} {'random':>8s} {'SA':>8s} {'QAOA':>8s}")
    for B_ in budgets:
        sa_hits = sum(simulated_annealing(E_hat, subsets, index, n, k, B_, rng) == opt
                      for _ in range(trials))
        print(f"  {B_:>8d} {1 - (1 - 1 / N) ** B_:>8.3f} {sa_hits / trials:>8.3f} "
              f"{1 - (1 - p_hit) ** B_:>8.3f}")
    print(f"  NOTE: QAOA column excludes training. Training used "
          f"{results[p_best]['evals']} cost evaluations at this depth\n"
          f"        (each costs many shots on real hardware); a fair total-cost "
          f"comparison must add that.")
    print()

    # ---- Qiskit cross-check -------------------------------------------
    try:
        qiskit_crosscheck(r, c, lam, n, k, subsets, E_hat, Emin, span,
                          sim, results[p_best]["x"])
    except ImportError as exc:
        print(f"  (Qiskit cross-check skipped: {exc})")


if __name__ == "__main__":
    main()