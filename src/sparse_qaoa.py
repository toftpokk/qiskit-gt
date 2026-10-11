"""Sparse, exact QAOA in the weight-k subspace (scales to n ~ 24-26 on a laptop).

Replaces the dense simulator in qaoa_exact.py, which diagonalized an N x N matrix
(N = C(n, k)); that needs ~2 GB at n = 20 and far more beyond. Two ideas make
this version scale:

  1. The XY-mixer matrix B (subsets that differ by one swap) is stored SPARSE:
     each subset has only k*(n-k) neighbours.
  2. B is the adjacency matrix of the Johnson graph J(n, k), whose spectrum is
     known in closed form: theta_j = (k-j)(n-k-j) - j for j = 0..min(k, n-k).
     Only k+1 distinct eigenvalues means exp(-i beta B) is EXACTLY a polynomial
     in B of degree k. We apply it with k sparse matrix-vector products, using a
     Chebyshev basis for numerical stability. No eigendecomposition, no
     approximation.

Also added: exact analytic (adjoint) gradients, so training costs ~2-3 circuit
evaluations per step instead of 2p+1 finite-difference evaluations.

Run `uv run python src/sparse_qaoa.py` to execute the built-in self-test.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Subsets, ranking, energies, sparse mixer matrix
# ---------------------------------------------------------------------------

def _comb_table(n: int, k: int) -> np.ndarray:
    """tab[s, t] = C(s, t); used to rank subsets (combinatorial number system)."""
    tab = np.zeros((n + 1, k + 2), dtype=np.int64)
    for s in range(n + 1):
        for t in range(k + 2):
            tab[s, t] = math.comb(s, t)
    return tab


def rank_rows(rows: np.ndarray, tab: np.ndarray) -> np.ndarray:
    """Colex rank of each sorted row (a size-k subset) -> integer in [0, C(n,k))."""
    k = rows.shape[1]
    out = np.zeros(len(rows), dtype=np.int64)
    for i in range(k):
        out += tab[rows[:, i], i + 1]
    return out


def all_subsets(n: int, k: int):
    """All size-k subsets as an (N, k) array in colex order, so rank(row a) == a."""
    arr = np.array(list(combinations(range(n), k)), dtype=np.int64)
    tab = _comb_table(n, k)
    arr = arr[np.argsort(rank_rows(arr, tab), kind="stable")]
    assert np.array_equal(rank_rows(arr, tab), np.arange(len(arr)))
    return arr, tab


def subset_energies_fast(subsets: np.ndarray, r: np.ndarray, c: np.ndarray,
                         lam: float) -> np.ndarray:
    """E(S) = -sum_{i in S} r_i + lam * sum_{i<j in S} c_ij, vectorized over all S."""
    rel, red = relevance_and_redundancy_sums(subsets, r, c)
    return -rel + lam * red


def relevance_and_redundancy_sums(subsets, r, c):
    """Per-subset relevance sum and redundancy sum (so lambda can be swept cheaply)."""
    k = subsets.shape[1]
    rel = r[subsets].sum(axis=1)
    red = np.zeros(len(subsets))
    for a in range(k):
        for b in range(a + 1, k):
            red += c[subsets[:, a], subsets[:, b]]
    return rel, red


def johnson_sparse(subsets: np.ndarray, n: int, k: int, tab: np.ndarray):
    """Sparse adjacency of the Johnson graph: S ~ T iff they differ by one swap.

    This is the complete-graph XY mixer (sum_{i<j} (X_i X_j + Y_i Y_j)/2)
    restricted to weight-k states.
    """
    N = len(subsets)
    inside = np.zeros((N, n), dtype=bool)
    inside[np.arange(N)[:, None], subsets] = True
    allrows = np.arange(N)
    rows, cols = [], []
    for pos in range(k):
        for j in range(n):
            idx = allrows[~inside[:, j]]
            if idx.size == 0:
                continue
            new = subsets[idx].copy()
            new[:, pos] = j                 # replace one member by outsider j
            new.sort(axis=1)
            rows.append(idx)
            cols.append(rank_rows(new, tab))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))


# ---------------------------------------------------------------------------
# Exact QAOA simulator
# ---------------------------------------------------------------------------

class SparseSubspaceQAOA:
    """p-layer QAOA on the weight-k subspace. Start state: uniform (Dicke).

    Cost layer  : psi <- exp(-i gamma E_hat) * psi
    Mixer layer : psi <- exp(-i beta B) psi, via an exact degree-k polynomial in B.
    E_hat is the subset energy rescaled to [0, 1]; the expected "feasible-only
    approximation ratio" is therefore 1 - <E_hat>.
    """

    def __init__(self, E_hat: np.ndarray, B: sp.csr_matrix, n: int, k: int):
        self.E = E_hat
        self.N = len(E_hat)
        self.B = B.astype(np.complex128).tocsr()
        jmax = min(k, n - k)
        self.theta = np.array([(k - j) * (n - k - j) - j for j in range(jmax + 1)],
                              dtype=float)                     # Johnson-graph spectrum
        lo, hi = self.theta.min(), self.theta.max()
        self.c, self.s = (hi + lo) / 2.0, (hi - lo) / 2.0
        x = np.clip((self.theta - self.c) / self.s, -1.0, 1.0)  # nodes in [-1, 1]
        T = np.cos(np.outer(np.arccos(x), np.arange(jmax + 1)))  # T[j, m] = T_m(x_j)
        self.Tinv = np.linalg.inv(T)
        self.deg = jmax

    # -- mixer: exp(-i beta B) psi = sum_m d_m(beta) T_m(B') psi ---------------
    def _chebyshev_vectors(self, psi: np.ndarray):
        c, s = self.c, self.s
        vs = [psi, (self.B @ psi - c * psi) / s]
        for _ in range(2, self.deg + 1):
            vs.append(2.0 * (self.B @ vs[-1] - c * vs[-1]) / s - vs[-2])
        return vs

    def _mix(self, psi: np.ndarray, beta: float) -> np.ndarray:
        vs = self._chebyshev_vectors(psi)
        d = self.Tinv @ np.exp(-1j * beta * self.theta)
        out = d[0] * vs[0]
        for m in range(1, self.deg + 1):
            out = out + d[m] * vs[m]
        return out

    # -- forward ---------------------------------------------------------------
    def state(self, params: np.ndarray) -> np.ndarray:
        p = len(params) // 2
        psi = np.full(self.N, 1.0 / math.sqrt(self.N), dtype=complex)
        for g, b in zip(params[:p], params[p:]):
            psi = self._mix(np.exp(-1j * g * self.E) * psi, b)
        return psi

    def probs(self, params: np.ndarray) -> np.ndarray:
        return np.abs(self.state(params)) ** 2

    def expectation(self, params: np.ndarray) -> float:
        return float(self.probs(params) @ self.E)

    # -- exact analytic gradient (adjoint method) --------------------------------
    def value_and_grad(self, params: np.ndarray):
        """<H> and its exact gradient w.r.t. (gamma_1..p, beta_1..p).

        Forward pass stores the state after each cost layer; the backward pass
        propagates the co-state H|psi_p> through the layers in reverse. Cost is
        roughly 2-3x one forward evaluation, independent of the number of
        parameters. (A simulation shortcut: hardware cannot read gradients out
        this way.)
        """
        p = len(params) // 2
        gam, bet = params[:p], params[p:]
        E = self.E
        psi = np.full(self.N, 1.0 / math.sqrt(self.N), dtype=complex)
        mids = []                                   # state after each cost layer
        for l in range(p):
            psi = np.exp(-1j * gam[l] * E) * psi
            mids.append(psi)
            psi = self._mix(psi, bet[l])

        value = float(np.real(np.vdot(psi, E * psi)))
        lam = E * psi                               # H |psi_p>
        out = psi                                   # state after mixer l
        g_gam, g_bet = np.zeros(p), np.zeros(p)
        for l in reversed(range(p)):
            g_bet[l] = 2.0 * np.real(np.vdot(lam, -1j * (self.B @ out)))
            lam_mid = self._mix(lam, -bet[l])       # M_l^dagger applied to co-state
            g_gam[l] = 2.0 * np.real(np.vdot(lam_mid, -1j * E * mids[l]))
            phase = np.exp(1j * gam[l] * E)         # C_l^dagger
            lam = phase * lam_mid
            out = phase * mids[l]
        return value, np.concatenate([g_gam, g_bet])


# ---------------------------------------------------------------------------
# Training (multi-start + INTERP warm start), analytic gradients
# ---------------------------------------------------------------------------

def interp(params: np.ndarray) -> np.ndarray:
    """INTERP warm start (Zhou et al. 2018): stretch depth-p angles to depth p+1."""
    p = len(params) // 2

    def stretch(a):
        ext = np.concatenate([[0.0], a, [0.0]])
        i = np.arange(1, p + 2)
        return (i - 1) / p * ext[i - 1] + (p - i + 1) / p * ext[i]

    return np.concatenate([stretch(params[:p]), stretch(params[p:])])


def train_sparse(sim: SparseSubspaceQAOA, p: int, rng, n_random: int,
                 warm=None, maxiter: int = 200):
    """L-BFGS-B with exact gradients. Returns [(expectation, params, n_evals)]."""
    starts = []
    if warm is not None:
        starts.append(interp(warm))
    for _ in range(n_random):
        starts.append(np.concatenate([rng.uniform(0, 2 * np.pi, p),
                                      rng.uniform(0, 2 * np.pi, p)]))
    out = []
    for x0 in starts:
        res = minimize(sim.value_and_grad, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        out.append((float(res.fun), res.x, int(res.nfev)))
    return out


# ---------------------------------------------------------------------------
# Self-test: verifies the sparse simulator against independent references
# ---------------------------------------------------------------------------

def self_test(n: int = 13, k: int = 5, seed: int = 0) -> None:
    from scipy.sparse.linalg import expm_multiply

    rng = np.random.default_rng(seed)
    subsets, tab = all_subsets(n, k)
    N = len(subsets)
    B = johnson_sparse(subsets, n, k, tab)
    print(f"self-test: n={n}, k={k}, N={N}")

    # 1. structure
    assert B.nnz == N * k * (n - k), "wrong neighbour count"
    assert abs(B - B.T).sum() == 0, "mixer matrix not symmetric"
    print(f"  mixer: {B.nnz} nonzeros = N*k*(n-k), symmetric")

    # 2. closed-form Johnson spectrum vs dense eigendecomposition (small N only)
    if N <= 3000:
        evals = np.unique(np.round(np.linalg.eigvalsh(B.toarray()), 6))
        theta = np.sort([(k - j) * (n - k - j) - j for j in range(min(k, n - k) + 1)])
        assert np.allclose(evals, theta), f"spectrum mismatch {evals} vs {theta}"
        print(f"  spectrum matches closed form: {theta.astype(int).tolist()}")

    # 3. polynomial mixer vs scipy's sparse matrix exponential
    E = rng.random(N)
    sim = SparseSubspaceQAOA(E, B, n, k)
    psi = rng.normal(size=N) + 1j * rng.normal(size=N)
    psi /= np.linalg.norm(psi)
    worst = 0.0
    for beta in (0.1, 1.0, 3.0, 6.0):
        ref = expm_multiply(-1j * beta * B.astype(complex), psi)
        worst = max(worst, np.max(np.abs(sim._mix(psi, beta) - ref)))
    print(f"  mixer vs expm_multiply, max abs diff     : {worst:.2e}")
    assert worst < 1e-9

    # 4. norm and analytic gradient vs finite differences
    x = rng.uniform(0, 2 * np.pi, 6)
    assert abs(np.linalg.norm(sim.state(x)) - 1) < 1e-10
    val, grad = sim.value_and_grad(x)
    assert abs(val - sim.expectation(x)) < 1e-10
    fd = np.array([(sim.expectation(x + 1e-6 * e) - sim.expectation(x - 1e-6 * e)) / 2e-6
                   for e in np.eye(len(x))])
    print(f"  analytic vs finite-difference gradient   : {np.max(np.abs(grad - fd)):.2e}")
    assert np.max(np.abs(grad - fd)) < 1e-6
    print("  all checks passed")


if __name__ == "__main__":
    self_test()