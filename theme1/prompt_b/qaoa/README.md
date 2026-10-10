# QAOA for Feature Selection

An implementation of the Quantum Approximate Optimization Algorithm (QAOA) for
the feature-selection QUBO from `theme1/prompt_b/README.md`, driven by IBM's
`qiskit_algorithms.QAOA` (on the Qiskit `SamplerV2` primitive).

> Data comes from the UCI Heart Disease dataset (`ucimlrepo`, id 45). Relevance
> `r` (mutual information) and redundancy `c` (`|Pearson correlation|`) are
> computed from the real 297-patient Cleveland subset in `data.py`.

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
`⟨H_C⟩`, estimated by sampling the ansatz with `StatevectorSampler`
(`SamplingVQE`). The sampler is noiseless (exact statevector); the only
uncertainty is finite-shot statistics.

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

The demo loads the UCI Heart Disease dataset, computes `r`/`c`, and runs
`p = 1, 2, 3` on the `n = 13, k = 5` problem, printing each layer count's
approximation ratio and the best sampled assignment against the brute-force
optimum.

One non-obvious step: `qiskit_algorithms.QAOA` builds a non-flattened ansatz
whose cost/mixer evolutions are nested `PauliEvolutionGate`s; left as-is the
sampler calls `PauliEvolutionGate.to_matrix()` -> `scipy.sparse.linalg.expm` of
the full 2ⁿ×2ⁿ operator, which hangs on 13 qubits. `qaoa.py` passes a small
`transpiler` that decomposes the ansatz to `RZ`/`RZZ`/`RX` first (see the
module docstring).

## Data

`data.py` -> `load_heart_disease()`:

1. Fetch `ucimlrepo` id 45 (303 patients, 13 features).
2. Drop rows with missing entries (Cleveland subset -> 297 patients).
3. Binarize the 0-4 target: `0` = no disease, `1-4` = disease.
4. `relevance` = `mutual_info_classif` (KNN estimator, `random_state=42`).
5. `redundancy` = `|corr|` (`np.corrcoef`), diagonal zeroed.

The QUBO needs only `r` and `c`, so no feature scaling is required.

## Code layout

| Symbol | Purpose |
|---|---|
| `load_heart_disease` | Fetch, clean, binarize; compute `r` and `c` (`data.py`) |
| `feature_selection_qubo` | Build `Q` from `r`, `c`, `λ`, `P`, `k` |
| `qubo_to_ising` | QUBO → `(offset, h, J)` |
| `qubo_value` / `brute_force` | QUBO cost on an assignment / exhaustive optimum + worst |
| `_ising_operator` | `H_C` as a `SparsePauliOp` (explicit qubit indices) |
| `_DecomposeTranspiler` | Flatten `PauliEvolutionGate`s to `RZ`/`RZZ`/`RX` |
| `main` | Run library `qiskit_algorithms.QAOA` and score vs brute force |

## Notes

- `StatevectorSampler` samples the exact statevector (no hardware noise); the
  energy is a finite-shot estimate, so the approximation ratio is slightly
  noisy while the `best sampled` metric is robust.
- Qiskit's `SparsePauliOp.from_list` is little-endian (rightmost character =
  qubit 0); the operator uses `from_sparse_list` with explicit indices to avoid
  that footgun.
- Qiskit counts are big-endian (leftmost bit = highest qubit);
  `_bitstring_to_features` reverses them before scoring.
