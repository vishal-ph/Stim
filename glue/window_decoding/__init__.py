"""Sliding window decoding for rotated surface codes.

A Python implementation of the forward sliding window decoder from:

    Skoric et al., "Parallel window decoding enables scalable fault
    tolerant quantum computation", Nature Communications 14, 7040
    (2023).  arXiv:2209.08552.

See the README for algorithm details and the tutorial notebook for
interactive examples.
"""

__version__ = "0.1.0"

from glue.window_decoding.window_decomposition import (
    build_forward_window_dem,
    group_detectors_by_round,
)
from glue.window_decoding.experiment import (
    ExperimentResult,
    run_full_decode,
    run_sliding_window_decode,
    run_experiment,
    run_surgery_experiment,
    run_sweep,
    print_results_table,
)
from glue.window_decoding.lattice_surgery_circuits import (
    build_merge_split_circuit,
)
