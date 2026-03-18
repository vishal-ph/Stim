"""Window decomposition utilities for sliding window decoding.

Provides the core building blocks for the forward sliding window decoder
from Skoric et al., "Parallel window decoding enables scalable fault
tolerant quantum computation", Nature Communications 14, 7040 (2023).
arXiv:2209.08552.

The key function is ``build_forward_window_dem``, which extracts a
sub-DEM (detector error model) for a single decoding window from the
full circuit's DEM.  It implements the forward decoder's asymmetric
time boundaries:

    - **Closed past boundary**: error mechanisms involving any detector
      from rounds before the current window are dropped entirely.  This
      prevents double-counting of logical observable contributions that
      were already committed by the previous window.

    - **Open future boundary**: error mechanisms with one detector inside
      the window and another beyond the window's end become single-detector
      "boundary" edges (the future-side detector is dropped).  PyMatching
      automatically connects these to its virtual boundary node.

The helper ``group_detectors_by_round`` maps the full circuit's detector
indices to their syndrome extraction round, merging the thin final
measurement layer into the last round.
"""

import stim


def group_detectors_by_round(
    circuit_or_dem: stim.DetectorErrorModel | stim.Circuit,
) -> dict[int, list[int]]:
    """Group a DEM's detector indices by syndrome extraction round.

    Stim's generated surface code circuits assign each detector a
    coordinate ``(x, y, t)`` where ``t`` is the syndrome round index.
    The final data-qubit measurement produces a thin layer of detectors
    at ``t = total_rounds`` with roughly half the normal count; these are
    merged into round ``t - 1`` so that every round has a uniform
    detector count.

    Args:
        circuit_or_dem: A Stim detector error model or circuit.
        This works because both have a ``get_detector_coordinates()``
        method that returns a dict mapping global detector index
        to coordinate tuple.

    Returns:
        Dict mapping integer round index to a sorted list of global
        detector indices belonging to that round.

    Example:
        >>> import stim
        >>> c = stim.Circuit.generated(
        ...     "surface_code:rotated_memory_z", distance=3, rounds=6,
        ...     after_clifford_depolarization=0.001)
        >>> dem = c.detector_error_model(decompose_errors=True)
        >>> by_round = group_detectors_by_round(dem)
        >>> sorted(by_round.keys())
        [0, 1, 2, 3, 4, 5]
    """
    coords = circuit_or_dem.get_detector_coordinates()
    by_round: dict[int, list[int]] = {}
    for det_idx, coord in coords.items():
        t = int(coord[2]) if len(coord) > 2 else 0
        by_round.setdefault(t, []).append(det_idx)

    if not by_round:
        return by_round

    # Merge the thin final-measurement boundary layer into the last
    # syndrome round.  The final layer has <= half the detectors of a
    # normal round (it compares final data measurements against the last
    # syndrome measurements, producing only X- or Z-type detectors, not
    # both).
    max_round = max(by_round.keys())
    if max_round > 0:
        normal_round_size = max(len(v) for k, v in by_round.items() if k < max_round)
        if len(by_round.get(max_round, [])) <= normal_round_size // 2:
            boundary_dets = by_round.pop(max_round)
            by_round.setdefault(max_round - 1, []).extend(boundary_dets)

    for k in by_round:
        by_round[k].sort()

    return by_round


def build_forward_window_dem(
    full_dem: stim.DetectorErrorModel,
    window_det_indices: list[int],
    past_det_indices: set[int],
) -> stim.DetectorErrorModel:
    """Build a sub-DEM for one forward-decoder window.

    Extracts error mechanisms from the full circuit's DEM that are
    relevant to the current window, applying the forward decoder's
    asymmetric time-boundary rules:

    **Closed past boundary** -- Any mechanism that involves a detector
    from ``past_det_indices`` (rounds before this window) is dropped.
    The previous window already committed the logical observable (L0)
    contribution of these mechanisms, so including them again would
    cause double-counting.

    **Open future boundary** -- A mechanism with one detector inside the
    window and another beyond the window's end (not in window, not in
    past) has the future-side detector dropped, turning it into a
    single-detector boundary edge.  PyMatching connects such edges to
    its virtual boundary node automatically.

    **Interior mechanisms** -- Mechanisms whose detectors are all within
    the window are kept as-is (with detector IDs re-indexed to local
    0..N-1).

    The ordering of ``window_det_indices`` defines the local index
    mapping: the first element gets local index 0, the second gets 1,
    etc.  Callers should pass commit detectors first, then buffer
    detectors, so that the commit/buffer boundary aligns with a
    contiguous index range.

    Args:
        full_dem: The full circuit's detector error model, obtained via
            ``circuit.detector_error_model(decompose_errors=True)``.
        window_det_indices: Ordered list of global detector indices for
            this window.  Commit-region detectors should come first,
            followed by buffer-region detectors.
        past_det_indices: Set of global detector indices from rounds
            temporally before this window.  Empty for the first window.

    Returns:
        A ``stim.DetectorErrorModel`` with locally-indexed detectors
        (0..len(window_det_indices)-1), ready to be passed to
        ``pymatching.Matching.from_detector_error_model()``.
    """
    window_set = set(window_det_indices)

    # Map global detector index -> local index (0..N-1)
    det_local: dict[int, int] = {}
    for local_idx, global_idx in enumerate(window_det_indices):
        det_local[global_idx] = local_idx

    result = stim.DetectorErrorModel()

    for inst in full_dem.flattened():
        if inst.type != "error":
            continue

        targets = inst.targets_copy()
        args = inst.args_copy()

        # Classify each detector target
        dets_in_window: list[int] = []
        touches_past = False
        has_det_targets = False

        for t in targets:
            if t.is_relative_detector_id():
                has_det_targets = True
                d = t.val
                if d in window_set:
                    dets_in_window.append(d)
                elif d in past_det_indices:
                    touches_past = True
                # else: future-outside detector (open boundary)

        # Rule 1: closed past boundary -- drop entirely
        if touches_past:
            continue

        # Pure observable-only mechanism (no detector targets at all):
        # include unchanged so that PyMatching sees the L0 label.
        if not has_det_targets:
            result.append("error", args, targets)
            continue

        # No detectors from this window (all future-outside): drop
        if not dets_in_window:
            continue

        # Re-index targets: keep window detectors (remapped to local),
        # keep L0/separator targets, drop future-outside detectors
        new_targets: list[stim.DemTarget] = []
        for t in targets:
            if t.is_relative_detector_id():
                d = t.val
                if d in det_local:
                    new_targets.append(
                        stim.DemTarget.relative_detector_id(det_local[d])
                    )
            elif t.is_logical_observable_id():
                new_targets.append(t)
            elif t.is_separator():
                new_targets.append(t)

        # Clean up stale separators (from dropped sub-edges in
        # decomposed error instructions that use ^ delimiters)
        while new_targets and new_targets[0].is_separator():
            new_targets.pop(0)
        while new_targets and new_targets[-1].is_separator():
            new_targets.pop()
        deduped: list[stim.DemTarget] = []
        for t in new_targets:
            if t.is_separator() and deduped and deduped[-1].is_separator():
                continue
            deduped.append(t)

        if deduped:
            result.append("error", args, deduped)

    return result
