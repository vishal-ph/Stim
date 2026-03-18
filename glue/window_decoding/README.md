# Sliding Window Decoding v0.1.0

A Python implementation of the **forward sliding window decoder** for rotated
surface codes, following:

> Skoric et al., "Parallel window decoding enables scalable fault tolerant
> quantum computation", Nature Communications 14, 7040 (2023).
> [arXiv:2209.08552](https://arxiv.org/abs/2209.08552).

## Features

- **Forward sliding window decoder** with configurable commit (`n_com`) and
  buffer (`n_buf`) region sizes.
- **Artificial defect propagation** across window boundaries — correction
  chains crossing the commit/buffer boundary are cut, producing artificial
  defects that are inherited by the next window.
- **Asymmetric time boundaries**: closed past (no double-counting of committed
  observables) and open future (boundary edges connected to PyMatching's virtual
  node).
- **Edge-accurate observable lookup** built from PyMatching's internal graph,
  correctly handling parallel-edge combination rules.
- **Built-in benchmarking**: compare sliding window error rates against
  full-circuit (global) MWPM decoding.
- **Parameter sweep utilities** for systematic evaluation across distances,
  window counts, and noise levels.
- **Interactive tutorial notebook** with visualizations and step-by-step
  walkthrough.

## Installation

```bash
pip install stim pymatching numpy matplotlib
```

## Quick start

```python
from glue.window_decoding.experiment import run_experiment

# Compare sliding window vs full-circuit decoding
result = run_experiment(distance=3, num_windows=3, noise=0.005, shots=10_000)
# Experiment: d=3, windows=3, rounds=12, window=3+3=6, p=0.005, shots=10000
#   Full-circuit error rate:   0.023400
#   Sliding window error rate: 0.023400
```

Run a parameter sweep:

```python
from glue.window_decoding.experiment import run_sweep, print_results_table

results = run_sweep(
    distances=[3, 5],
    num_windows_list=[2, 4, 8],
    noise_levels=[0.001, 0.005],
    shots=10_000,
)
print_results_table(results)
#   d  wins rounds   window    noise   shots   full_err  sliding_err    ratio
# -------------------------------------------------------------------------
#   3     2      9      3+3   0.0010   10000   0.001000     0.001000     1.00
#   3     2      9      3+3   0.0050   10000   0.017800     0.017800     1.00
#   ...
```

Custom window sizes:

```python
result = run_experiment(distance=5, num_windows=4, n_com=5, n_buf=10, noise=0.003, shots=10_000)
# Window covers 5 commit + 10 buffer = 15 rounds per step
# Total rounds = 4 * 5 + 10 = 30
```

Or run the experiment module directly:

```bash
python -m glue.window_decoding.experiment
```

## Algorithm

The decoder processes syndrome data through a sequence of overlapping
windows. Each window covers `n_com + n_buf` syndrome rounds (by default,
`d + d = 2d`):

```
Rounds:   0  1  2  3  4  5  6  7  8  9  10  11
Window 0: [commit ] [buffer ]
Window 1:          [commit ] [buffer ]
Window 2:                   [  commit (last)  ]
```

- **Commit region** (first `n_com` rounds): corrections are finalized and
  their logical observable contributions are accumulated.
- **Buffer region** (next `n_buf` rounds): provides lookahead context;
  corrections here are discarded and re-decoded by the next window.
- The **last window** has no buffer and commits everything remaining.
- **Window count** determines the total rounds: `total_rounds = num_windows * n_com + n_buf`.

### Time boundary rules

| Boundary | Treatment |
|----------|-----------|
| Past     | **Closed** — error mechanisms touching earlier rounds are dropped (already committed by the previous window) |
| Future   | **Open** — mechanisms reaching beyond the window become boundary edges connected to PyMatching's virtual node |
| Spatial  | **Open** — handled by PyMatching's virtual boundary node |

### Edge classification

After MWPM decoding within a window, each matched edge is classified:

| Edge type | Endpoints | Action |
|-----------|-----------|--------|
| Commit-internal | C × C or C × boundary | Commit L0 contribution |
| Cross-boundary | C × B | Do **not** commit L0; create artificial defect at buffer-side endpoint |
| Buffer-only | B × B or B × boundary | Discard entirely (next window re-decodes) |

### Artificial defects

When MWPM produces a correction chain that crosses from the commit region
into the buffer, only the commit-internal portion is committed. The
cross-boundary edge is "cut", producing an **artificial defect** at its
buffer-side endpoint.

**Example**: consider a chain `C₁ → C₂ → B₁ → B₂` where `C₁` and `B₂`
are fired detectors and `C₂`, `B₁` are unfired intermediates:

```
   commit region        buffer region
C₁(fired) ─── C₂(unfired) ─── B₁(unfired) ─── B₂(fired)
               │                │
          committed          artificial
          (with L0)           defect
```

1. The commit-internal edge `(C₁, C₂)` is committed with its L0.
2. The cross-boundary edge `(C₂, B₁)` is **not** committed. Its
   buffer-side endpoint `B₁` is recorded as an artificial defect.
3. The buffer edge `(B₁, B₂)` is discarded.
4. In the next window, `B₁` appears as an artificial defect
   (XOR-flipped 0 → 1) and `B₂` remains as a real defect. The next
   window's MWPM freely matches them — possibly pairing `B₁` with `B₂`,
   or matching each to other defects.

## API reference

### Core functions

| Function | Description |
|----------|-------------|
| `run_experiment(distance, num_windows, noise, shots, n_com, n_buf)` | Run a single sliding window vs full-circuit comparison |
| `run_sweep(distances, num_windows_list, noise_levels, n_com, n_buf, shots)` | Run a parameter sweep |
| `run_sliding_window_decode(circuit, det_events, distance, n_com, n_buf)` | Decode detection events using the sliding window algorithm |
| `run_full_decode(circuit, shots)` | Baseline full-circuit MWPM decoding |
| `print_results_table(results)` | Print a formatted results table |

### Window decomposition utilities

| Function | Description |
|----------|-------------|
| `group_detectors_by_round(circuit)` | Map detector indices to syndrome rounds |
| `build_forward_window_dem(full_dem, window_det_indices, past_det_indices)` | Build a sub-DEM for one window with asymmetric boundaries |

### Data structures

| Class | Fields |
|-------|--------|
| `ExperimentResult` | `distance`, `num_windows`, `total_rounds`, `noise`, `shots`, `n_com`, `n_buf`, `full_error_rate`, `sliding_window_error_rate`, `elapsed_seconds` |

## Files

| File | Description |
|------|-------------|
| `window_decomposition.py` | Core utilities: `group_detectors_by_round`, `build_forward_window_dem` |
| `experiment.py` | Decoder implementation, experiment runner, parameter sweep |
| `tutorial.ipynb` | Interactive tutorial with visualizations and step-by-step walkthrough |

## Requirements

- Python ≥ 3.11
- [Stim](https://github.com/quantumlib/Stim) — circuit simulation and DEM extraction
- [PyMatching](https://github.com/oscarhiggott/PyMatching) 2.x — MWPM decoding (`decode_to_edges_array`, `Matching.edges()`)
- NumPy
- Matplotlib (for tutorial visualizations)
