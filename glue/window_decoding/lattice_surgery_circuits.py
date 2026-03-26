"""Lattice surgery circuit constructors for windowed decoding experiments.

Provides helpers that build stim circuits for lattice surgery operations
(merge/split) on two rotated surface code patches.  These circuits serve
as validation targets for the multi-observable sliding window decoder.

The circuit uses two distance-d rotated surface code patches placed
side-by-side, merged along their rough (Z-type) boundaries:

- **Pre-merge**: independent syndrome extraction on both patches.
- **Merge**: Z-type boundary stabilizers connecting the patches are
  activated (growing from weight-2 to weight-4).
- **Post-split**: boundary stabilizers are deactivated (shrinking back
  to weight-2).

Detector coordinates are ``(x, y, t)`` compatible with the windowed
decoder's round-grouping logic.
"""

import stim


# ---------------------------------------------------------------------------
# Qubit layout helpers
# ---------------------------------------------------------------------------


def _stab_type(ax: int, ay: int) -> str:
    """Return 'X' or 'Z' for the stabilizer at even coordinates (ax, ay)."""
    return "X" if (ax // 2 + ay // 2) % 2 == 1 else "Z"


def _cx_layer(stype: str, dx: int, dy: int) -> int:
    """Which CX layer (0-3) does the (dx, dy) direction belong to?

    CX schedule matching ``stim.Circuit.generated`` for rotated surface codes:
        Layer 0: SE (+1, +1) — all types
        Layer 1: X→SW (−1, +1), Z→NE (+1, −1)
        Layer 2: X→NE (+1, −1), Z→SW (−1, +1)
        Layer 3: NW (−1, −1) — all types
    """
    if (dx, dy) == (+1, +1):
        return 0
    if (dx, dy) == (-1, -1):
        return 3
    if stype == "X":
        return 1 if (dx, dy) == (-1, +1) else 2
    else:
        return 1 if (dx, dy) == (+1, -1) else 2


# ---------------------------------------------------------------------------
# Main circuit constructor
# ---------------------------------------------------------------------------


def build_merge_split_circuit(
    distance: int = 3,
    r_pre: int = 3,
    r_merge: int = 3,
    r_post: int = 3,
    noise: float = 0.001,
) -> stim.Circuit:
    """Build a two-patch rotated surface code merge-split circuit.

    Constructs a circuit with two distance-``d`` rotated surface code
    patches that undergo a merge-split lattice surgery operation along
    their rough (Z-type) boundaries.

    **Circuit phases**:

    1. **Pre-merge** (``r_pre`` rounds): independent syndrome extraction
       on both patches.  Each patch has its own boundary stabilizers.
    2. **Merge** (``r_merge`` rounds): Z-type boundary stabilizers grow
       from weight-2 (per-patch) to weight-4 (spanning both patches).
       The first merge-round detector uses the identity
       ``S_merge = S_A_bnd × S_B_bnd``.
    3. **Post-split** (``r_post`` rounds): merge stabilizers are
       deactivated; per-patch boundary stabilizers are re-activated.
       A split-transition detector checks ``S_A_bnd ⊕ S_B_bnd ⊕ S_merge``.
    4. **Final data measurement** with detectors comparing Z-type
       stabilizer parities against last syndrome measurements.

    **Qubit layout** (distance ``d``):

    - Patch A data qubits: ``(2i+1, 2j+1)`` for ``i,j ∈ [0, d)``.
    - Patch B data qubits: ``(2d+2i+1, 2j+1)`` for ``i,j ∈ [0, d)``.
    - Merge boundary at ``x = 2d``.

    **Observables**:

    - ``L0``: Z-logical of patch A (bottom row data qubits).
    - ``L1``: Z-logical of patch B (bottom row data qubits).

    **Noise model** (matching ``stim.Circuit.generated``):

    - ``DEPOLARIZE1(p)`` after each H layer.
    - ``DEPOLARIZE2(p)`` after each CX layer.

    Args:
        distance: Code distance ``d`` per patch.  Must be ≥ 2.
        r_pre: Pre-merge syndrome extraction rounds.  Must be ≥ 1.
        r_merge: Merged syndrome extraction rounds.  Must be ≥ 1.
        r_post: Post-split syndrome extraction rounds.  Must be ≥ 1.
        noise: Depolarization noise rate.  Set to 0 for noiseless.

    Returns:
        A ``stim.Circuit`` suitable for DEM extraction via
        ``circuit.detector_error_model(decompose_errors=True)``.
    """
    d = distance
    if d < 2:
        raise ValueError(f"Distance must be >= 2, got {d}")
    if r_pre < 1 or r_merge < 1 or r_post < 1:
        raise ValueError("r_pre, r_merge, r_post must each be >= 1")

    # ================================================================
    # 1. Enumerate qubits
    # ================================================================

    # Coordinate -> qubit index mapping
    _next_q = 0
    _coord_to_q: dict[tuple, int] = {}
    _q_to_coord: dict[int, tuple[float, float]] = {}

    def alloc(x: float, y: float, label: str = "") -> int:
        nonlocal _next_q
        key = (x, y, label)
        if key in _coord_to_q:
            return _coord_to_q[key]
        idx = _next_q
        _next_q += 1
        _coord_to_q[key] = idx
        _q_to_coord[idx] = (float(x), float(y))
        return idx

    # --- Data qubits ---
    data_a_set: set[int] = set()
    data_b_set: set[int] = set()
    data_at: dict[tuple[int, int], int] = {}  # (x,y) -> qubit

    for i in range(d):
        for j in range(d):
            x, y = 2 * i + 1, 2 * j + 1
            q = alloc(x, y, "data_a")
            data_a_set.add(q)
            data_at[(x, y)] = q

    for i in range(d):
        for j in range(d):
            x, y = 2 * d + 2 * i + 1, 2 * j + 1
            q = alloc(x, y, "data_b")
            data_b_set.add(q)
            data_at[(x, y)] = q

    all_data = sorted(data_a_set | data_b_set)

    # --- Ancilla enumeration ---
    # Each ancilla dict: q, x, y, type, role, data_neighbors, cx_by_layer
    # role: 'regular', 'a_bnd', 'b_bnd', 'merge'
    ancillas: list[dict] = []

    def _get_neighbors(ax: int, ay: int, allowed: set[int] | None = None):
        """Get data qubit neighbors of ancilla at (ax, ay)."""
        nbrs = []
        for dx, dy in [(-1, -1), (+1, -1), (-1, +1), (+1, +1)]:
            q = data_at.get((ax + dx, ay + dy))
            if q is not None and (allowed is None or q in allowed):
                nbrs.append((q, dx, dy))
        return nbrs

    def _make_ancilla(ax, ay, stype, role, neighbors, label):
        q = alloc(ax, ay, label)
        cx = {0: [], 1: [], 2: [], 3: []}
        data_qs = []
        for dq, dx, dy in neighbors:
            layer = _cx_layer(stype, dx, dy)
            if stype == "X":
                cx[layer].append((q, dq))  # ancilla controls
            else:
                cx[layer].append((dq, q))  # data controls
            data_qs.append(dq)
        return {
            "q": q, "x": ax, "y": ay, "type": stype, "role": role,
            "cx": cx, "data_qs": data_qs, "label": label,
        }

    def _boundary_ok(ax, ay, stype, x_lo, x_hi, y_lo, y_hi):
        """Check boundary type rules for the rotated surface code."""
        on_left = ax <= x_lo
        on_right = ax >= x_hi
        on_top = ay <= y_lo
        on_bottom = ay >= y_hi
        if on_left or on_right:
            return stype == "Z"
        if on_top or on_bottom:
            return stype == "X"
        return True

    # Patch A ancillas (x < 2d, i.e. NOT at merge column)
    for ax in range(0, 2 * d, 2):
        for ay in range(0, 2 * d + 1, 2):
            stype = _stab_type(ax, ay)
            nbrs = _get_neighbors(ax, ay, data_a_set)
            if len(nbrs) < 2:
                continue
            if len(nbrs) == 2 and not _boundary_ok(
                ax, ay, stype, 0, 2 * d, 0, 2 * (d - 1) + 2
            ):
                continue
            ancillas.append(
                _make_ancilla(ax, ay, stype, "regular", nbrs, f"a_{ax}_{ay}")
            )

    # Patch B ancillas (x > 2d)
    for ax in range(2 * d + 2, 4 * d + 1, 2):
        for ay in range(0, 2 * d + 1, 2):
            stype = _stab_type(ax, ay)
            nbrs = _get_neighbors(ax, ay, data_b_set)
            if len(nbrs) < 2:
                continue
            if len(nbrs) == 2 and not _boundary_ok(
                ax, ay, stype, 2 * d, 4 * d, 0, 2 * (d - 1) + 2
            ):
                continue
            ancillas.append(
                _make_ancilla(ax, ay, stype, "regular", nbrs, f"b_{ax}_{ay}")
            )

    # Merge column (x = 2d): A boundary, B boundary, and merge ancillas
    # Only Z-type positions exist on the rough boundary.
    merge_positions: list[int] = []  # ay values
    for ay in range(0, 2 * d + 1, 2):
        stype = _stab_type(2 * d, ay)
        if stype != "Z":
            continue
        nbrs_a = _get_neighbors(2 * d, ay, data_a_set)
        nbrs_b = _get_neighbors(2 * d, ay, data_b_set)
        if len(nbrs_a) < 2 and len(nbrs_b) < 2:
            continue
        if len(nbrs_a) >= 2:
            ancillas.append(
                _make_ancilla(2 * d, ay, "Z", "a_bnd", nbrs_a, f"a_bnd_{ay}")
            )
        if len(nbrs_b) >= 2:
            ancillas.append(
                _make_ancilla(2 * d, ay, "Z", "b_bnd", nbrs_b, f"b_bnd_{ay}")
            )
        if len(nbrs_a) + len(nbrs_b) >= 4:
            ancillas.append(
                _make_ancilla(
                    2 * d, ay, "Z", "merge", nbrs_a + nbrs_b, f"merge_{ay}"
                )
            )
            merge_positions.append(ay)

    # Build lookup: (role, ay) -> ancilla dict, for merge column cross-refs
    _merge_col: dict[tuple[str, int], dict] = {}
    for anc in ancillas:
        if anc["role"] in ("a_bnd", "b_bnd", "merge"):
            _merge_col[(anc["role"], anc["y"])] = anc

    # ================================================================
    # 2. Build the circuit
    # ================================================================

    circuit = stim.Circuit()

    # QUBIT_COORDS
    for q in sorted(_q_to_coord):
        x, y = _q_to_coord[q]
        circuit.append("QUBIT_COORDS", [q], [x, y])

    # Initialize data qubits
    circuit.append("R", all_data)
    circuit.append("TICK")

    # --- Measurement tracking ---
    num_meas = 0
    prev_meas: dict[int, int] = {}  # ancilla qubit -> abs meas index

    def rec(abs_idx: int) -> stim.GateTarget:
        return stim.target_rec(abs_idx - num_meas)

    # ================================================================
    # 3. Syndrome extraction round
    # ================================================================

    def do_round(phase: str, t: int, transition_from: str | None = None):
        """Run one syndrome extraction round.

        Args:
            phase: 'pre', 'merge', or 'post'.
            t: Round number (for detector time coordinate).
            transition_from: If this is the first round of a new phase,
                the previous phase name (for transition detectors).
        """
        nonlocal num_meas

        # Determine active ancillas for this phase
        active: list[dict] = []
        for anc in ancillas:
            role = anc["role"]
            if role == "regular":
                active.append(anc)
            elif role == "a_bnd" and phase in ("pre", "post"):
                active.append(anc)
            elif role == "b_bnd" and phase in ("pre", "post"):
                active.append(anc)
            elif role == "merge" and phase == "merge":
                active.append(anc)

        # --- H gates on X-type ancillas ---
        x_qs = sorted({anc["q"] for anc in active if anc["type"] == "X"})
        if x_qs:
            circuit.append("H", x_qs)
            if noise > 0:
                circuit.append("DEPOLARIZE1", x_qs, [noise])
        circuit.append("TICK")

        # --- CX layers ---
        for layer in range(4):
            pairs: list[int] = []
            for anc in active:
                for ctrl, tgt in anc["cx"][layer]:
                    pairs.extend([ctrl, tgt])
            if pairs:
                circuit.append("CX", pairs)
                if noise > 0:
                    circuit.append("DEPOLARIZE2", pairs, [noise])
            circuit.append("TICK")

        # --- H gates on X-type ancillas ---
        if x_qs:
            circuit.append("H", x_qs)
            if noise > 0:
                circuit.append("DEPOLARIZE1", x_qs, [noise])
        circuit.append("TICK")

        # --- MR on active ancillas ---
        active_qs = sorted({anc["q"] for anc in active})
        circuit.append("MR", active_qs)

        # Record measurement indices
        curr: dict[int, int] = {}
        for q in active_qs:
            curr[q] = num_meas
            num_meas += 1

        # --- Detector declarations ---
        for anc in active:
            q = anc["q"]
            role = anc["role"]
            ax, ay = anc["x"], anc["y"]
            coord = [float(ax), float(ay), float(t)]

            if transition_from == "pre" and role == "merge":
                # Merge transition: S_merge = S_A_bnd × S_B_bnd
                targets = [rec(curr[q])]
                a_bnd = _merge_col.get(("a_bnd", ay))
                b_bnd = _merge_col.get(("b_bnd", ay))
                if a_bnd and a_bnd["q"] in prev_meas:
                    targets.append(rec(prev_meas[a_bnd["q"]]))
                if b_bnd and b_bnd["q"] in prev_meas:
                    targets.append(rec(prev_meas[b_bnd["q"]]))
                circuit.append("DETECTOR", targets, coord)

            elif transition_from == "merge" and role == "a_bnd":
                # Split transition: S_A_bnd ⊕ S_B_bnd ⊕ prev_S_merge = 0
                targets = [rec(curr[q])]
                b_bnd = _merge_col.get(("b_bnd", ay))
                merge_anc = _merge_col.get(("merge", ay))
                if b_bnd and b_bnd["q"] in curr:
                    targets.append(rec(curr[b_bnd["q"]]))
                if merge_anc and merge_anc["q"] in prev_meas:
                    targets.append(rec(prev_meas[merge_anc["q"]]))
                circuit.append("DETECTOR", targets, coord)

            elif transition_from == "merge" and role == "b_bnd":
                # Already accounted for in the a_bnd split detector
                pass

            elif q in prev_meas:
                # Normal: compare current with previous measurement
                targets = [rec(curr[q]), rec(prev_meas[q])]
                circuit.append("DETECTOR", targets, coord)

            elif t == 0 and anc["type"] == "Z":
                # Very first round, Z-type: deterministic from |0⟩ init
                targets = [rec(curr[q])]
                circuit.append("DETECTOR", targets, coord)

        # Update prev_meas
        for q, m in curr.items():
            prev_meas[q] = m

        circuit.append("TICK")

    # ================================================================
    # 4. Execute rounds
    # ================================================================

    t = 0

    # Pre-merge
    for r in range(r_pre):
        do_round("pre", t, transition_from=None)
        t += 1

    # Merge
    for r in range(r_merge):
        do_round(
            "merge", t, transition_from="pre" if r == 0 else None
        )
        t += 1

    # Post-split
    for r in range(r_post):
        do_round(
            "post", t, transition_from="merge" if r == 0 else None
        )
        t += 1

    # ================================================================
    # 5. Final data qubit measurements
    # ================================================================

    circuit.append("M", all_data)
    final: dict[int, int] = {}
    for q in all_data:
        final[q] = num_meas
        num_meas += 1

    final_t = float(t)

    # Final-layer detectors: Z-type ancillas only (since final M is in Z basis)
    for anc in ancillas:
        role = anc["role"]
        # Only ancillas active in the last phase (post-split)
        if role not in ("regular", "a_bnd", "b_bnd"):
            continue
        if anc["type"] != "Z":
            continue
        q = anc["q"]
        if q not in prev_meas:
            continue

        targets = [rec(prev_meas[q])]
        for dq in anc["data_qs"]:
            targets.append(rec(final[dq]))
        coord = [float(anc["x"]), float(anc["y"]), final_t]
        circuit.append("DETECTOR", targets, coord)

    # ================================================================
    # 6. Observable declarations
    # ================================================================

    # L0: Z-logical of patch A — bottom row data qubits (y = 2d-1)
    bottom_a = sorted(
        [data_at[(2 * i + 1, 2 * d - 1)] for i in range(d)],
    )
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [rec(final[q]) for q in bottom_a],
        [0],
    )

    # L1: Z-logical of patch B — bottom row data qubits (y = 2d-1)
    bottom_b = sorted(
        [data_at[(2 * d + 2 * i + 1, 2 * d - 1)] for i in range(d)],
    )
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [rec(final[q]) for q in bottom_b],
        [1],
    )

    return circuit
