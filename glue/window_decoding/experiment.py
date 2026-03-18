"""Sliding window decoding experiment for rotated surface codes.

Implements and benchmarks the forward sliding window decoder from:

    Skoric et al., "Parallel window decoding enables scalable fault
    tolerant quantum computation", Nature Communications 14, 7040
    (2023).  arXiv:2209.08552.

The experiment compares two decoding strategies on the same sampled
detection events:

    1. **Full-circuit decoding** (baseline): build one MWPM matcher for
       the entire circuit and decode all detectors at once.

    2. **Sliding window decoding**: slide a window of ``n_com + n_buf``
       rounds (``n_com`` commit + ``n_buf`` buffer) through the circuit,
       advancing by ``n_com`` rounds per step.  Each window decodes its
       local sub-DEM, commits only edges fully within the commit region,
       and creates artificial defects at the buffer-side endpoint of any
       correction chain that crosses the commit/buffer boundary.

The two strategies should produce nearly identical logical error rates
(ratio ~1.0) when the algorithm is implemented correctly.

Usage:
    python -m glue.window_decoding.experiment
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import stim

from glue.window_decoding.window_decomposition import (
    build_forward_window_dem,
    group_detectors_by_round,
)


@dataclass
class ExperimentResult:
    """Results from one experiment configuration."""

    distance: int
    num_windows: int
    total_rounds: int
    noise: float
    shots: int
    n_com: int
    n_buf: int
    full_error_rate: float
    sliding_window_error_rate: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Baseline: full-circuit MWPM decoding
# ---------------------------------------------------------------------------


def run_full_decode(
    full_dem: stim.DetectorErrorModel, shots: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample detection events and decode using the full circuit's DEM.

    This is the gold-standard baseline: a single MWPM decoder sees all
    detectors simultaneously, so it can find globally optimal matchings.

    Args:
        full_dem: The full noisy surface code detector error model.
        shots: Number of shots to sample.
        seed: Optional random seed for sampling.

    Returns:
        A tuple ``(det_events, obs_actual, predictions)`` where:

        - ``det_events``: bool array [shots x num_detectors]
        - ``obs_actual``: bool array [shots x num_observables]
        - ``predictions``: bool array [shots x num_observables] from MWPM
    """
    import pymatching

    matcher = pymatching.Matching.from_detector_error_model(full_dem)

    sampler = full_dem.compile_sampler(seed=seed)
    det_events, obs_actual, _ = sampler.sample(shots=shots)

    predictions = matcher.decode_batch(det_events)
    return det_events, obs_actual, predictions


# ---------------------------------------------------------------------------
# Forward sliding window decoder (Skoric et al. 2023)
# ---------------------------------------------------------------------------


def _build_edge_obs_lookup(
    matcher: "pymatching.Matching",
) -> dict[tuple[int, int], bool]:
    """Build edge -> L0 observable lookup from PyMatching's internal graph.

    PyMatching may combine parallel edges using probability-weighted rules
    that differ from naive XOR of the DEM's observable labels.  To ensure
    the per-edge L0 labels match what the decoder actually uses, we read
    them directly from ``matcher.edges()``.

    The virtual boundary node is represented as ``-1``.  Both orderings
    ``(u, v)`` and ``(v, u)`` are stored for convenient lookup.

    Args:
        matcher: A constructed PyMatching ``Matching`` object.

    Returns:
        Dict mapping ``(u, v)`` detector-index pairs to ``True`` if the
        edge carries the L0 (logical observable 0) label.
    """

    edge_obs: dict[tuple[int, int], bool] = {}
    for u, v, data in matcher.edges():
        has_l0 = 0 in data["fault_ids"]
        v_key = -1 if v is None else v
        edge_obs[(u, v_key)] = has_l0
        edge_obs[(v_key, u)] = has_l0
    return edge_obs


def run_sliding_window_decode(
    full_dem: stim.DetectorErrorModel,
    det_events: np.ndarray,
    distance: int,
    n_com: int | None = None,
    n_buf: int | None = None,
) -> np.ndarray:
    """Decode detection events using the forward sliding-window algorithm.

    Implements the forward decoder from Skoric et al. (2023), Section I.B.
    The algorithm processes the circuit's detection events through a
    sequence of overlapping windows, each covering ``n_com + n_buf``
    syndrome rounds, advancing by ``n_com`` rounds per step.

    **Algorithm overview** (per window):

    1. Build a sub-DEM with ``build_forward_window_dem``:
       - Past time boundary is **closed** (mechanisms touching earlier
         rounds are dropped to prevent observable double-counting).
       - Future time boundary is **open** (mechanisms reaching beyond the
         window become boundary edges in PyMatching).

    2. Construct the syndrome for this window from ``det_events``, then
       XOR-flip any **artificial defects** inherited from the previous
       window (see step 5).

    3. Run MWPM (via PyMatching) on the window's syndrome to obtain a
       set of correction edges.

    4. **Edge classification and commitment**:

       - *Commit-internal edges* (C x C): both non-boundary endpoints
         are in the commit region ``[0, n_commit_dets)``.  Their L0
         contributions are XOR'd into the running prediction.

       - *Commit-to-boundary edges* (C x boundary): the non-boundary
         endpoint is in the commit region.  Their L0 contributions are
         committed.

       - *Cross-boundary edges* (C x B): one endpoint in commit, one
         in buffer.  These are **not** committed for L0.  Instead, an
         artificial defect is created at the buffer-side endpoint (see
         step 5).

       - *Buffer edges* (B x B or B x boundary): both non-boundary
         endpoints are in the buffer region.  **Discarded** -- the next
         window will re-decode these detectors.

    5. **Artificial defects**: when a correction chain crosses the
       commit/buffer boundary, only the commit-internal portion is
       committed.  The cross-boundary edge is "cut", creating an
       artificial defect at the buffer-side endpoint.  This defect is
       XOR-flipped into the next window's syndrome, where the next
       window's MWPM freely matches it (possibly pairing it with the
       chain's original buffer-side endpoint, or with other defects).

    6. The **last window** commits all remaining rounds (no buffer),
       so every edge is a commit-internal edge and no artificial
       defects are generated.

    Args:
        full_dem: The full noisy surface code detector error model
            (used to extract detector coordinates and the DEM).
        det_events: Bool array ``[shots x num_detectors]`` of detection
            events sampled from the circuit.
        distance: Code distance ``d``.  Used as the default for
            ``n_com`` and ``n_buf`` if they are not specified.
        n_com: Number of commit rounds per window.  Defaults to ``d``.
        n_buf: Number of buffer rounds per window.  Defaults to ``d``.
        seed: Optional random seed for detection events
            simulation reproducibility.

    Returns:
        Bool array ``[shots x 1]`` of predicted logical observable flips.
    """
    import pymatching

    shots = det_events.shape[0]

    # set n_com = n_buf = distance if not specified
    if n_com is None:
        n_com = distance
    if n_buf is None:
        n_buf = distance

    # Map each detector to its syndrome round
    by_round = group_detectors_by_round(full_dem)
    if not by_round:
        return np.zeros((shots, 1), dtype=bool)

    total_rounds = max(by_round.keys()) + 1

    # Window count: the last window's buffer extends to the end of the
    # circuit, so we need (num_windows - 1) * d + 2d >= total_rounds.
    num_windows = max(1, math.ceil((total_rounds - n_buf) / n_com))

    final_prediction = np.zeros(shots, dtype=bool)

    # Per-shot list of GLOBAL detector indices inherited as artificial
    # defects from the previous window's committed C x B edges.
    artificial_defects: list[list[int]] = [[] for _ in range(shots)]

    for w in range(num_windows):
        # --- Determine this window's round ranges ---
        commit_start = w * n_com
        is_last_window = w == num_windows - 1

        if is_last_window:
            # Last window: commit everything remaining, no buffer.
            # All edges become core edges; no artificial defects out.
            commit_end = total_rounds
            buf_end = total_rounds
        else:
            commit_end = commit_start + n_com
            buf_end = min(commit_start + n_com + n_buf, total_rounds)

        # --- Collect detector indices for commit and buffer regions ---
        com_dets: list[int] = []
        for t in range(commit_start, commit_end):
            com_dets.extend(by_round.get(t, []))

        buf_dets: list[int] = []
        for t in range(commit_end, buf_end):
            buf_dets.extend(by_round.get(t, []))

        # Commit detectors first so local indices [0, n_com_dets) = commit
        window_dets = com_dets + buf_dets
        n_com_dets = len(com_dets)
        n_window_dets = len(window_dets)

        if not window_dets:
            artificial_defects = [[] for _ in range(shots)]
            continue

        # --- Build the window's sub-DEM (closed past, open future) ---
        past_det_indices: set[int] = set()
        for t in range(0, commit_start):
            past_det_indices.update(by_round.get(t, []))

        window_dem = build_forward_window_dem(
            full_dem,
            window_dets,
            past_det_indices,
        )

        if len(window_dem) == 0:
            artificial_defects = [[] for _ in range(shots)]
            continue

        # --- Prepare per-shot syndrome for this window ---
        global_to_local = {g: i for i, g in enumerate(window_dets)}
        w_det_data = det_events[:, window_dets].copy().astype(bool)

        # Apply artificial defects from the previous window: XOR-flip
        # buffer-side endpoints of cross-boundary edges.  These may be
        # unfired intermediate nodes (flipped 0 -> 1 to become artificial
        # defects) or fired chain endpoints (flipped 1 -> 0 to cancel).
        for shot in range(shots):
            for global_idx in artificial_defects[shot]:
                local = global_to_local.get(global_idx)
                if local is not None:
                    w_det_data[shot, local] ^= True

        # --- Decode with PyMatching ---
        matcher = pymatching.Matching.from_detector_error_model(window_dem)
        edge_obs = _build_edge_obs_lookup(matcher)

        new_artificial: list[list[int]] = [[] for _ in range(shots)]

        for shot in range(shots):
            syndrome = w_det_data[shot]
            if not np.any(syndrome):
                continue

            edges = matcher.decode_to_edges_array(syndrome)
            obs_flip = False

            for edge in edges:
                u, v = int(edge[0]), int(edge[1])

                # --- Edge classification ---
                # Virtual boundary (-1) is neither commit nor buffer.
                u_in_commit = 0 <= u < n_com_dets
                v_in_commit = 0 <= v < n_com_dets
                u_in_buffer = n_com_dets <= u < n_window_dets
                v_in_buffer = n_com_dets <= v < n_window_dets

                if not u_in_commit and not v_in_commit:
                    # Buffer-only edge (B x B or B x boundary): discard.
                    # The next window will re-decode these detectors.
                    continue

                if u_in_buffer or v_in_buffer:
                    # Cross-boundary edge (C x B): do NOT commit L0.
                    # Create artificial defect at the buffer-side
                    # endpoint.  The next window will see this defect
                    # and freely match it.
                    if u_in_buffer:
                        new_artificial[shot].append(window_dets[u])
                    if v_in_buffer:
                        new_artificial[shot].append(window_dets[v])
                    continue

                # Commit-internal edge (C x C or C x boundary):
                # commit its L0 contribution.
                if edge_obs.get((u, v), False):
                    obs_flip = not obs_flip

            if obs_flip:
                final_prediction[shot] = not final_prediction[shot]

        artificial_defects = new_artificial

    return final_prediction.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Experiment runner and parameter sweep
# ---------------------------------------------------------------------------


def run_experiment(
    distance: int = 3,
    num_windows: int = 2,
    noise: float = 0.001,
    shots: int = 10_000,
    n_com: int | None = None,
    n_buf: int | None = None,
    code: str = "surface_code:rotated_memory_z",
    verbose: bool = True,
    seed: int = 1234,
) -> ExperimentResult:
    """Run a single sliding-window vs full-circuit decoding comparison.

    Generates a rotated surface code memory-Z circuit, samples detection
    events, decodes with both the full-circuit baseline and the sliding
    window decoder, and reports the logical error rates.

    The total number of syndrome rounds is computed automatically::

        total_rounds = num_windows * n_com + n_buf

    This ensures the circuit length is always compatible with the window
    layout, removing a common source of user error.

    Args:
        distance: Code distance ``d``.
        num_windows: Number of decoding windows.
        noise: Depolarization rate (``after_clifford_depolarization``).
        shots: Number of Monte Carlo shots.
        n_com: Number of commit rounds per window.  Defaults to ``d``.
        n_buf: Number of buffer rounds per window.  Defaults to ``d``.
        code: Stim code task string.
        verbose: If True, print progress to stdout.
        seed: Optional random seed for sampling reproducibility.

    Returns:
        An ``ExperimentResult`` with both error rates and timing.
    """
    start = time.time()

    if n_com is None:
        n_com = distance
    if n_buf is None:
        n_buf = distance

    total_rounds = num_windows * n_com + n_buf

    if verbose:
        print(
            f"Experiment: d={distance}, windows={num_windows}, "
            f"rounds={total_rounds}, window={n_com}+{n_buf}={n_com + n_buf}, "
            f"p={noise}, shots={shots}"
        )

    full_circuit = stim.Circuit.generated(
        code,
        distance=distance,
        rounds=total_rounds,
        after_clifford_depolarization=noise,
    )
    full_dem = full_circuit.detector_error_model(decompose_errors=True)

    # --- Full-circuit baseline ---
    det_events, obs_actual, full_predictions = run_full_decode(full_dem, shots, seed)
    full_error_rate = float(np.mean(np.any(full_predictions != obs_actual, axis=1)))

    if verbose:
        print(f"  Full-circuit error rate:   {full_error_rate:.6f}")

    # --- Sliding window decoder ---
    sliding_predictions = run_sliding_window_decode(
        full_dem,
        det_events,
        distance=distance,
        n_com=n_com,
        n_buf=n_buf,
    )
    sliding_error_rate = float(
        np.mean(np.any(sliding_predictions != obs_actual, axis=1))
    )

    if verbose:
        print(f"  Sliding window error rate: {sliding_error_rate:.6f}")

    elapsed = time.time() - start
    if verbose:
        print(f"  Elapsed: {elapsed:.1f}s")

    return ExperimentResult(
        distance=distance,
        num_windows=num_windows,
        total_rounds=total_rounds,
        noise=noise,
        shots=shots,
        n_com=n_com,
        n_buf=n_buf,
        full_error_rate=full_error_rate,
        sliding_window_error_rate=sliding_error_rate,
        elapsed_seconds=elapsed,
    )


def run_sweep(
    distances: list[int] | None = None,
    num_windows_list: list[int] | None = None,
    noise_levels: list[float] | None = None,
    com_buf_sizes_list: list[(int, int)] | None = None,
    shots: int = 10_000,
    code: str = "surface_code:rotated_memory_z",
) -> list[ExperimentResult]:
    """Run a parameter sweep over (distance, num_windows, noise).

    Args:
        distances: List of code distances to test.
        num_windows_list: List of window counts to test.
        noise_levels: List of depolarization rates.
        com_buf_sizes_list: List of (commit, buffer) sizes to test.  Defaults to ``[(d, d)]``.
        shots: Shots per experiment.
        code: Stim code task string.

    Returns:
        List of ``ExperimentResult`` objects.
    """
    if distances is None:
        distances = [3, 5]
    if num_windows_list is None:
        num_windows_list = [2, 4, 8]
    if noise_levels is None:
        noise_levels = [0.001, 0.005]

    results = []
    for d in distances:
        for nw in num_windows_list:
            for p in noise_levels:
                if com_buf_sizes_list is None:
                    com_buf_sizes_list = [(d, d)]  # use defaults in run_experiment
                for n_com, n_buf in com_buf_sizes_list:
                    if com_buf_sizes_list is None:
                        com_buf_sizes_list = [(d, d)]  # use defaults in run_experiment
                    result = run_experiment(
                        distance=d,
                        num_windows=nw,
                        noise=p,
                        shots=shots,
                        n_com=n_com,
                        n_buf=n_buf,
                        code=code,
                    )
                    results.append(result)
                print()
    return results


def run_sweep_fixed_rounds(
    distances: list[int] | None = None,
    rounds: int | None = None,
    noise_levels: list[float] | None = None,
    com_buf_sizes_list: list[(int, int)] | None = None,
    shots: int = 10_000,
    code: str = "surface_code:rotated_memory_z",
) -> list[ExperimentResult]:
    """Run a parameter sweep over (distance, num_windows, noise).

    Args:
        distances: List of code distances to test.
        rounds: Total number of rounds to run.
        noise_levels: List of depolarization rates.
        com_buf_sizes_list: List of (commit, buffer) sizes to test.  Defaults to ``[(d, d)]``.
        shots: Shots per experiment.
        code: Stim code task string.

    Returns:
        List of ``ExperimentResult`` objects.
    """
    if distances is None:
        distances = [3, 5]
    if noise_levels is None:
        noise_levels = [0.001, 0.005]

    results = []
    for d in distances:
        for p in noise_levels:
            if com_buf_sizes_list is None:
                com_buf_sizes_list = [(d, d)]  # use defaults in run_experiment
            for n_com, n_buf in com_buf_sizes_list:
                if rounds is None:
                    rounds = np.lcm.reduce(
                        [3 * n_com + n_buf for n_com, n_buf in (com_buf_sizes_list)]
                    )  # ensure there's a minimum of 3 windows for each distance
                num_windows = (rounds - n_buf) // n_com
                result = run_experiment(
                    distance=d,
                    num_windows=num_windows,
                    noise=p,
                    shots=shots,
                    n_com=n_com,
                    n_buf=n_buf,
                    code=code,
                )
                results.append(result)
            print()
    return results


def print_results_table(results: list[ExperimentResult]) -> None:
    """Print a summary table of experiment results."""
    header = (
        f"{'d':>3} {'wins':>5} {'rounds':>6} {'window':>8} "
        f"{'noise':>8} {'shots':>7} "
        f"{'full_err':>10} {'sliding_err':>12} {'ratio':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        ratio = (
            r.sliding_window_error_rate / r.full_error_rate
            if r.full_error_rate > 0
            else float("inf")
        )
        window_str = f"{r.n_com}+{r.n_buf}"
        print(
            f"{r.distance:>3} {r.num_windows:>5} {r.total_rounds:>6} "
            f"{window_str:>8} "
            f"{r.noise:>8.4f} {r.shots:>7} "
            f"{r.full_error_rate:>10.6f} {r.sliding_window_error_rate:>12.6f} "
            f"{ratio:>8.2f}"
        )


if __name__ == "__main__":
    print("=== Sliding Window Decoding Experiment ===")
    print("    Skoric et al., arXiv:2209.08552 (2023)\n")

    # Sanity check: zero noise must produce zero errors
    print("--- Sanity check (zero noise) ---")
    result = run_experiment(distance=3, num_windows=2, noise=0.0, shots=1000)
    assert result.full_error_rate == 0.0
    assert result.sliding_window_error_rate == 0.0
    print("  PASSED\n")

    # Main sweep
    print("--- Parameter sweep ---")
    results = run_sweep(
        distances=[3, 5],
        num_windows_list=[2, 4, 8],
        noise_levels=[0.001, 0.005],
        shots=10_000,
    )

    print("\n=== Results Summary ===")
    print_results_table(results)
