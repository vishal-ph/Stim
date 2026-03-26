# Sliding Window Decoding — Memory Experiments

A Python implementation of the **forward sliding window decoder** for rotated
surface code **memory experiments**, following:

> Skoric et al., "Parallel window decoding enables scalable fault tolerant
> quantum computation", Nature Communications 14, 7040 (2023).
> [arXiv:2209.08552](https://arxiv.org/abs/2209.08552).

## Overview

Memory experiments run repeated syndrome extraction on a single surface code
patch with a single logical observable (L0). The sliding window decoder
processes syndrome data through overlapping time windows instead of decoding
all rounds at once, enabling:

- **Bounded memory** — each window uses a fixed-size matching graph.
- **Online decoding** — windows can be decoded as syndrome data arrives.
- **Near-optimal accuracy** — error rates match full-circuit MWPM (ratio ~1.0).

## Quick start

```python
from glue.window_decoding.experiment import run_experiment

# Compare sliding window vs full-circuit decoding
result = run_experiment(distance=3, num_windows=3, noise=0.005, shots=10_000)
# Experiment: d=3, rounds=12, window=3+3=6, p=0.005, shots=10000
#   Full-circuit error rate:   0.023400
#   Sliding window error rate: 0.023400
```

Parameter sweep:

```python
from glue.window_decoding.experiment import run_sweep, print_results_table

results = run_sweep(
    distances=[3, 5],
    num_windows_list=[2, 4, 8],
    noise_levels=[0.001, 0.005],
    shots=10_000,
)
print_results_table(results)
```

Custom window sizes:

```python
result = run_experiment(distance=5, num_windows=4, n_com=5, n_buf=10, noise=0.003, shots=10_000)
# Window covers 5 commit + 10 buffer = 15 rounds per step
# Total rounds = 4 * 5 + 10 = 30
```

Command-line:

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
| Commit-internal | C x C or C x boundary | Commit observable contributions |
| Cross-boundary | C x B | Commit observable contributions; create artificial defect at buffer endpoint |
| Buffer-only | B x B or B x boundary | Discard entirely (next window re-decodes) |

### Artificial defects

When MWPM produces a correction chain that crosses from the commit region
into the buffer, the cross-boundary edge is "cut", producing an **artificial
defect** at its buffer-side endpoint. This defect is XOR-flipped into the
next window's syndrome, where MWPM freely matches it.

The closed-past-boundary rule prevents double-counting of observable
contributions across windows.

## API

### Core functions

| Function | Description |
|----------|-------------|
| `run_experiment(distance, num_windows, noise, shots, ...)` | Run a single sliding window vs full-circuit comparison |
| `run_sweep(distances, num_windows_list, noise_levels, ...)` | Run a parameter sweep |
| `run_sliding_window_decode(full_dem, det_events, distance, ...)` | Decode detection events using the sliding window algorithm |
| `run_full_decode(full_dem, shots, seed)` | Baseline full-circuit MWPM decoding |
| `print_results_table(results)` | Print a formatted results table |

### Window decomposition utilities

| Function | Description |
|----------|-------------|
| `group_detectors_by_round(circuit_or_dem)` | Map detector indices to syndrome rounds |
| `build_forward_window_dem(full_dem, window_det_indices, past_det_indices)` | Build a sub-DEM for one window with asymmetric boundaries |

## Tutorial

See **`tutorial.ipynb`** for an interactive walkthrough with visualizations,
including:

- Detector-to-round mapping and window layout visualization
- Step-by-step manual window decode of a single shot
- Scaling plots (error rate vs rounds, ratio vs window size)

## Requirements

- Python >= 3.11
- [Stim](https://github.com/quantumlib/Stim) — circuit simulation and DEM extraction
- [PyMatching](https://github.com/oscarhiggott/PyMatching) 2.x — MWPM decoding
- NumPy
- Matplotlib (for tutorial visualizations)
