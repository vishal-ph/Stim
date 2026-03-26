# Sliding Window Decoding — Lattice Surgery Experiments

Extensions to the sliding window decoder that support **computation
circuits** — specifically lattice surgery merge/split operations on
two rotated surface code patches with multiple logical observables.

## Overview

Lattice surgery circuits differ from memory experiments in several ways:

- **Multiple code patches** with independent logical observables (L0, L1, ...).
- **Dynamic stabilizer sets** — the merge phase activates boundary stabilizers
  connecting patches; the split phase deactivates them.
- **Non-uniform detector counts** — merge/split transition rounds have
  different detector counts than steady-state rounds.
- **Spacetime observable-carrying edges** — in surface code circuits, hook
  errors during stabilizer extraction produce error mechanisms with detectors
  at different time rounds that also carry observable labels.  Empirically,
  the majority of L-carrying 2-detector mechanisms in a surface code DEM
  have detectors at different times.

The extended decoder handles all of these correctly.

## What's new (vs memory experiment decoder)

| Feature | Memory decoder | Extended decoder |
|---------|---------------|-----------------|
| Observables | Single (L0) | Multiple (L0..Ln) |
| Circuit input | `stim.Circuit.generated()` only | Any `stim.Circuit` |
| Cross-boundary obs | Committed (with closed-past anti-double-counting) | Same rule, validated for multi-patch circuits |
| Surgery circuits | Not supported | `build_merge_split_circuit` |

### Cross-boundary observable commitment

When a MWPM correction chain crosses from the commit region into the buffer
(a C×B edge), the decoder commits the observable contributions and creates
an artificial defect at the buffer-side endpoint.  The closed-past-boundary
rule in subsequent windows prevents double-counting: the same error
mechanism is dropped because its commit-side detector falls in the past
region.

This is important for surface codes generally (not just lattice surgery),
because hook errors produce spacetime L-carrying edges whose detectors
span multiple rounds.  Such edges can straddle the commit/buffer boundary.

## Quick start

### Build a merge-split circuit

```python
from glue.window_decoding.lattice_surgery_circuits import build_merge_split_circuit

circuit = build_merge_split_circuit(
    distance=3,     # code distance per patch
    r_pre=4,        # pre-merge rounds
    r_merge=4,      # merged rounds
    r_post=4,       # post-split rounds
    noise=0.003,    # depolarizing noise rate
)
print(f"Qubits: {circuit.num_qubits}")
print(f"Detectors: {circuit.num_detectors}")
print(f"Observables: {circuit.num_observables}")  # 2
```

### Run a surgery experiment

```python
from glue.window_decoding.experiment import run_surgery_experiment

result = run_surgery_experiment(
    distance=3,
    r_pre=4,
    r_merge=4,
    r_post=4,
    noise=0.003,
    shots=20_000,
    n_com=3,       # commit rounds per window
    n_buf=4,       # buffer rounds per window
)
# Experiment: d=3, rounds=13, window=3+4=7, p=0.003, shots=20000
#   Full-circuit error rate:   0.015450
#   Sliding window error rate: 0.015550
```

### Decode an arbitrary circuit

```python
from glue.window_decoding.experiment import run_experiment

# Pass any stim.Circuit — not just stim.Circuit.generated() outputs
result = run_experiment(
    distance=3,
    circuit=my_custom_circuit,  # your circuit here
    shots=10_000,
    n_com=3,
    n_buf=4,
)
```

### Multi-observable windowed decode (low-level)

```python
from glue.window_decoding.experiment import run_full_decode, run_sliding_window_decode

dem = circuit.detector_error_model(decompose_errors=True)
det_events, obs_actual, full_pred = run_full_decode(dem, shots=10_000, seed=42)

sliding_pred = run_sliding_window_decode(
    dem, det_events, distance=3, n_com=3, n_buf=4,
    num_observables=circuit.num_observables,  # tracks L0 and L1
)
# sliding_pred.shape == (10000, 2)  -- one column per observable
```

## Circuit structure

The `build_merge_split_circuit` function constructs a two-patch **rotated
surface code** circuit.  Two distance-d patches are placed side-by-side
and merged along their rough (Z-type) boundaries.

### Qubit layout (distance d)

- **Patch A data**: `(2i+1, 2j+1)` for `i, j ∈ [0, d)`.
- **Patch B data**: `(2d+2i+1, 2j+1)` for `i, j ∈ [0, d)`.
- **Merge boundary** at `x = 2d` — Z-type stabilizers connecting both patches.
- Ancilla positions follow the standard rotated surface code layout:
  X-type on smooth (top/bottom) boundaries, Z-type on rough (left/right)
  boundaries.

### Circuit phases

```
Phase 1 (pre-merge):   r_pre rounds — independent syndrome extraction
Phase 2 (merge):       r_merge rounds — boundary Z stabilizers grow to weight-4
Phase 3 (post-split):  r_post rounds — boundary stabilizers shrink to weight-2
Final:                 data qubit measurements + observable declarations
```

### Stabilizer transitions

At the merge boundary:

- **Pre-merge**: two weight-2 Z stabilizers per position (one per patch).
- **Merge**: one weight-4 Z stabilizer spanning both patches.
  `S_merge = S_A_bnd × S_B_bnd`, enabling a transition detector.
- **Post-split**: weight-2 stabilizers re-activated.  A split-transition
  detector checks `S_A_bnd ⊕ S_B_bnd ⊕ prev_S_merge`.

### Observables

- **L0**: Z-logical of patch A (bottom-row data qubits).
- **L1**: Z-logical of patch B (bottom-row data qubits).

### Noise model

Matches `stim.Circuit.generated` conventions:

- `DEPOLARIZE1(p)` after each H gate layer.
- `DEPOLARIZE2(p)` after each CX gate layer.

## API

### New functions

| Function | Description |
|----------|-------------|
| `build_merge_split_circuit(distance, r_pre, r_merge, r_post, noise)` | Build a two-patch merge-split circuit |
| `run_surgery_experiment(distance, r_pre, r_merge, r_post, noise, ...)` | One-call surgery experiment (build + decode + compare) |

### Extended functions

| Function | What changed |
|----------|-------------|
| `run_experiment(..., circuit=)` | New `circuit` parameter accepts arbitrary `stim.Circuit` |
| `run_sliding_window_decode(..., num_observables=)` | New `num_observables` parameter; returns `[shots, num_obs]` array |
| `_build_edge_obs_lookup(matcher)` | Returns `dict[edge, set[int]]` (set of observable indices) instead of `dict[edge, bool]` |

## Tutorial

See **`lattice_surgery_tutorial.ipynb`** for an interactive walkthrough,
including:

- Circuit construction and inspection
- Space-time detector visualization with merge/split phase coloring
- Observable label analysis in the DEM
- Zero-noise sanity checks
- Per-observable error rate comparison (L0 and L1)
- Window layout visualization over the surgery circuit
- Robustness sweep across window configurations
- Distance and noise scaling

## Files

| File | Description |
|------|-------------|
| `lattice_surgery_circuits.py` | Circuit constructor: `build_merge_split_circuit` |
| `experiment.py` | Decoder, experiment runners (extended for multi-observable + arbitrary circuits) |
| `window_decomposition.py` | Core utilities: `group_detectors_by_round`, `build_forward_window_dem` |
| `lattice_surgery_tutorial.ipynb` | Interactive tutorial for lattice surgery experiments |
| `tutorial.ipynb` | Interactive tutorial for memory experiments |

## Requirements

- Python >= 3.11
- [Stim](https://github.com/quantumlib/Stim) — circuit simulation and DEM extraction
- [PyMatching](https://github.com/oscarhiggott/PyMatching) 2.x — MWPM decoding
- NumPy
- Matplotlib (for tutorial visualizations)
