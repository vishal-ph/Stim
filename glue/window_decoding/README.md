# Sliding Window Decoding v0.2.0

A Python implementation of the **forward sliding window decoder** for quantum
error correcting codes, following:

> Skoric et al., "Parallel window decoding enables scalable fault tolerant
> quantum computation", Nature Communications 14, 7040 (2023).
> [arXiv:2209.08552](https://arxiv.org/abs/2209.08552).

## Features

- **Forward sliding window decoder** with configurable commit and buffer sizes.
- **Multi-observable support** — tracks L0..Ln for circuits with multiple logical observables.
- **Arbitrary circuit input** — decode any `stim.Circuit`, not just generated ones.
- **Lattice surgery circuits** — built-in merge-split circuit constructor.
- **Built-in benchmarking** against full-circuit MWPM decoding.
- **Parameter sweep utilities** and interactive tutorial notebooks.

## Installation

```bash
pip install stim pymatching numpy matplotlib
```

## Documentation

This module supports two categories of experiments, each with its own README
and tutorial notebook:

### Memory experiments

Single surface code patch, single logical observable, identity logical evolution.

- **[README_memory.md](README_memory.md)** — algorithm details, API reference, quick start
- **[tutorial.ipynb](tutorial.ipynb)** — interactive walkthrough with visualizations

```python
from glue.window_decoding.experiment import run_experiment

result = run_experiment(distance=3, num_windows=3, noise=0.005, shots=10_000)
```

### Lattice surgery experiments

Multiple code patches, multiple logical observables, merge/split operations.

- **[README_lattice_surgery.md](README_lattice_surgery.md)** — circuit structure, extensions, API
- **[lattice_surgery_tutorial.ipynb](lattice_surgery_tutorial.ipynb)** — interactive walkthrough

```python
from glue.window_decoding.experiment import run_surgery_experiment

result = run_surgery_experiment(distance=3, r_pre=4, r_merge=4, r_post=4, noise=0.003, shots=10_000)
```

## Files

| File | Description |
|------|-------------|
| `window_decomposition.py` | Core utilities: `group_detectors_by_round`, `build_forward_window_dem` |
| `experiment.py` | Decoder, experiment runners, parameter sweeps |
| `lattice_surgery_circuits.py` | Merge-split circuit constructor |
| `tutorial.ipynb` | Memory experiment tutorial |
| `lattice_surgery_tutorial.ipynb` | Lattice surgery tutorial |

## Requirements

- Python >= 3.11
- [Stim](https://github.com/quantumlib/Stim) — circuit simulation and DEM extraction
- [PyMatching](https://github.com/oscarhiggott/PyMatching) 2.x — MWPM decoding
- NumPy
- Matplotlib (for tutorial visualizations)
