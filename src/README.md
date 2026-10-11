# QAOA for Feature Selection

Two QAOA implementations for the feature-selection QUBO from the assignment in
[`PROMPT.md`](../PROMPT.md), plus a data loader and two analysis scripts:

- `qaoa.py` — drives IBM's `qiskit_algorithms.QAOA` (on the Qiskit `SamplerV2`
  primitive) over the full 2ⁿ space, using a penalty term to force "exactly k
  features".
- `qaoa_exact.py` — simulates QAOA *exactly* inside the weight-k subspace (the
  1287 subsets of 5 features) with a constraint-preserving XY mixer, and
  compares it against greedy and simulated-annealing baselines.

> Data comes from the UCI Heart Disease dataset (`ucimlrepo`, id 45). Relevance
> `r` (mutual information, normalized to `[0, 1]`) and redundancy `c`
> (`|Pearson correlation|`) are computed from the real 297-patient Cleveland
> subset in `data.py`.

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

## How QAOA solves it (`qaoa.py`)

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
candidate feature subset. `P(feasible)`, `P(optimal)`, and the feasible-only
approximation ratio are computed from the measured distribution.

**Approximation ratio.** Minimization with possibly-negative costs is normalized
to `[0, 1]` over the brute-force range:

```
AR(value) = (worst - value) / (worst - optimal)     1.0 = optimal
```

## The exact subspace approach (`qaoa_exact.py`)

Instead of penalizing weight-k violations on the full 2ⁿ space, this script
enumerates the `C(n, k) = 1287` feasible subsets once, rescales their energies
to `[0, 1]`, and simulates QAOA *exactly* in that subspace (no shots, no
penalty, no sampling noise):

- **State** is a `1287`-vector over weight-k subsets, started from the uniform
  (Dicke) state.
- **Mixer** is the XY model restricted to the Johnson graph of adjacent subsets
  (swap one feature); it preserves the weight-k constraint.
- **Training** uses L-BFGS-B from many random starts plus an INTERP warm start
  that carries the best depth-p angles into depth p+1.
- **Baselines** are greedy forward selection and swap-move simulated annealing
  on the same objective, compared at matched sample budgets.
- **Cross-check** rebuilds the cost operator in the full 2ⁿ space with Qiskit's
  `SparsePauliOp` and verifies the subspace reduction and probabilities agree.

## Run

```bash
uv sync                                   # provision the venv

uv run python src/qaoa.py                 # library QAOA, p = 1..3 (full space)
uv run python src/qaoa_exact.py           # exact subspace QAOA + baselines
uv run python src/lamsweep.py             # argmin over λ vs top-5-MI baseline
uv run python src/evaluate.py             # downstream AUC across λ (30 splits)
```

`qaoa.py` loads the UCI Heart Disease dataset, computes `r`/`c`, and runs
`p = 1, 2, 3` on the `n = 13, k = 5` problem over several restarts, printing
per-depth probabilities and approximation ratios against the brute-force
optimum. `qaoa_exact.py` does the same in the subspace and adds the classical
baselines. `src/test.py` is a one-line `fetch_ucirepo` connectivity check.

One non-obvious step: `qiskit_algorithms.QAOA` builds a non-flattened ansatz
whose cost/mixer evolutions are nested `PauliEvolutionGate`s; left as-is the
sampler calls `PauliEvolutionGate.to_matrix()` -> `scipy.sparse.linalg.expm` of
the full 2ⁿ×2ⁿ operator, which hangs on 13 qubits. `qaoa.py` passes a small
`transpiler` that decomposes the ansatz to `RZ`/`RZZ`/`RX` first (see the
module docstring).

## Data

`data.py` -> `load_heart_disease()` returns a `HeartData` dataclass:

1. Fetch `ucimlrepo` id 45 (303 patients, 13 features).
2. Drop rows with missing entries (Cleveland subset -> 297 patients).
3. Binarize the 0-4 target: `0` = no disease, `1-4` = disease.
4. `relevance` = `mutual_info_classif` with a discrete-feature mask for the 8
   categorical columns, normalized by its maximum to `[0, 1]`.
5. `redundancy` = `|corr|` (`np.corrcoef`), diagonal zeroed.

The QUBO needs only `r` and `c`, so no feature scaling is required.

## Code layout

`qaoa.py`:

| Symbol | Purpose |
|---|---|
| `feature_selection_qubo` | Build `Q` from `r`, `c`, `λ`, `P`, `k` |
| `qubo_to_ising` | QUBO → `(offset, h, J)` |
| `qubo_value` / `brute_force` | QUBO cost on an assignment / exhaustive optimum + worst |
| `verify_hamiltonian` | Assert the Ising diagonal matches the QUBO on every bitstring |
| `feasible_table` | QUBO value of every size-k subset |
| `evaluate_distribution` | Metrics from the final measured distribution |
| `_ising_operator` | `H_C` as a `SparsePauliOp` (explicit qubit indices) |
| `_DecomposeTranspiler` | Flatten `PauliEvolutionGate`s to `RZ`/`RZZ`/`RX` |
| `main` | Run library `qiskit_algorithms.QAOA` and score vs brute force |

`qaoa_exact.py`:

| Symbol | Purpose |
|---|---|
| `build_subspace` / `subset_energies` | Enumerate weight-k subsets and their QUBO energies |
| `johnson_adjacency` / `SubspaceQAOA` | XY mixer and exact p-layer evolution in the subspace |
| `train` / `interp` | Multi-start L-BFGS-B with INTERP warm starts |
| `greedy_forward` / `simulated_annealing` | Classical baselines on the same objective |
| `qiskit_crosscheck` | Verify the subspace reduction against full 2ⁿ Qiskit operators |

## Notes

- `StatevectorSampler` samples the exact statevector (no hardware noise); the
  energy is a finite-shot estimate, so the approximation ratio is slightly
  noisy while the `best sampled` metric is robust.
- Qiskit's `SparsePauliOp.from_list` is little-endian (rightmost character =
  qubit 0); the operator uses `from_sparse_list` with explicit indices to avoid
  that footgun.
- Qiskit counts are big-endian (leftmost bit = highest qubit);
  `_bitstring_to_features` reverses them before scoring.
