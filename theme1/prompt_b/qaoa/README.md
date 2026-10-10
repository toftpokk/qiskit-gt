# QAOA for Feature Selection

An implementation of the Quantum Approximate Optimization Algorithm (QAOA) for
the feature-selection QUBO from `theme1/prompt_b/README.md`, built directly on
IBM Qiskit's `EstimatorV2` / `SamplerV2` primitives.

> The dataset is intentionally **not** loaded. Relevance `r` and redundancy
> `c` are produced synthetically so the algorithm itself can be studied in
> isolation. Swap `feature_selection_qubo(…)`'s inputs for real data when ready.

## The problem

Pick exactly `k` of `n` features. One qubit per feature (`|1⟩` = keep,
`|0⟩` = drop). We minimize the binary cost

```
C(x) = -Σᵢ rᵢ xᵢ              maximize relevance
       + λ Σᵢ<ⱼ cᵢⱼ xᵢ xⱼ      penalize pairwise redundancy
       + P (Σᵢ xᵢ - k)²        force exactly k features
```

where `rᵢ` is the relevance of feature `i`, `cᵢⱼ` its redundancy with feature
`j`, `λ` the redundancy penalty, and `P` a large penalty. Expanding the
quadratic penalty gives the QUBO coefficients (up to a constant we drop):

```
Qᵢᵢ = -rᵢ + P (1 - 2k)
Qᵢⱼ = λ cᵢⱼ + 2P        (i < j)
```

## How QAOA solves it

**1. QUBO → Ising.** Substituting `xᵢ = (1 - zᵢ)/2` with `zᵢ ∈ {+1, -1}` (the
eigenvalue of `Zᵢ`) turns the QUBO into a cost Hamiltonian

```
H_C = offset·I + Σᵢ hᵢ Zᵢ + Σᵢ<ⱼ Jᵢⱼ Zᵢ Zⱼ

hᵢ   = -Qᵢᵢ/2 - Σⱼ≠ᵢ Qᵢⱼ/4
Jᵢⱼ  = Qᵢⱼ/4
offset = Σᵢ Qᵢᵢ/2 + Σᵢ<ⱼ Qᵢⱼ/4
```

The ground state of `H_C` is the optimal assignment.

**2. The ansatz.** Start from `|+⟩⊗ⁿ` and apply `p` layers of

```
U_C(γₗ) = exp(-i γₗ H_C)          cost unitary
U_M(βₗ) = exp(-i βₗ Σᵢ Xᵢ)        mixer unitary
```

Because `H_C` is diagonal in `Z`, the cost unitary decomposes into single-qubit
`RZ` and two-qubit `RZZ` gates (the mixer into `RX`):

```
exp(-i γ hᵢ Zᵢ)     = RZ(2 γ hᵢ)
exp(-i γ Jᵢⱼ Zᵢ Zⱼ) = RZZ(2 γ Jᵢⱼ)
exp(-i β Xᵢ)        = RX(2 β)
```

**3. Optimize.** COBYLA varies the `2p` parameters `(γ, β)` to minimize
`⟨H_C⟩`, evaluated exactly with `StatevectorEstimator` (noiseless ground truth).

**4. Sample.** `StatevectorSampler` measures the optimized state; each shot is a
candidate feature subset. The cheapest observed bitstring is reported.

**Approximation ratio.** Minimization with possibly-negative costs is normalized
to `[0, 1]` over the brute-force range:

```
AR(value) = (worst - value) / (worst - optimal)     1.0 = optimal
```

## Run

```bash
uv sync                                   # provision the venv
uv run python theme1/prompt_b/qaoa/qaoa.py
```

The demo builds a synthetic `n = 6, k = 3` instance and runs `p = 1, 2, 3`,
printing each layer count's approximation ratio, the best sampled assignment,
and the top measured bitstrings against the brute-force optimum.

## Code layout

| Symbol | Purpose |
|---|---|
| `feature_selection_qubo` | Build `Q` from `r`, `c`, `λ`, `P`, `k` |
| `qubo_to_ising` | QUBO → `(offset, h, J)` |
| `_ising_observable` | `H_C` as a `SparsePauliOp` (explicit qubit indices) |
| `build_ansatz` | p-layer QAOA circuit |
| `QAOA` | Optimize (COBYLA) + sample + score pipeline |
| `QaoaResult` | Result record with normalized approximation ratios |

## Notes

- `StatevectorEstimator`/`StatevectorSampler` are exact; this measures the
  algorithm's inherent performance, isolated from hardware noise.
- Qiskit's `SparsePauliOp.from_list` is little-endian (rightmost character =
  qubit 0); the observable uses `from_sparse_list` with explicit indices to
  avoid that footgun.
- Qiskit counts are big-endian (leftmost bit = highest qubit); `_bitstring_to_features`
  reverses them before scoring.
