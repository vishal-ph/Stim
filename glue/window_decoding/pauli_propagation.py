"""Pauli error propagation through Clifford bridge circuits.

NOTE: This module is not currently used by the sliding window decoder.
It is retained for potential future use.

Provides utilities to propagate decoded logical errors from one
decoding window through the Clifford circuit to determine their
effect on the logical operator at a future window.
"""

import stim


def build_bridge_tableau(
    code: str,
    distance: int,
    bridge_rounds: int = 1,
) -> stim.Tableau:
    """Build the Tableau for the Clifford bridge between decoding windows.

    Generates a noiseless circuit for the bridge rounds and converts
    its unitary part to a Tableau.

    Args:
        code: Code task string, e.g. "surface_code:rotated_memory_z".
        distance: Code distance.
        bridge_rounds: Number of syndrome extraction rounds in the bridge.

    Returns:
        Tableau representing the unitary Clifford operation.
    """
    bridge_circuit = stim.Circuit.generated(
        code, distance=distance, rounds=bridge_rounds,
    )
    return bridge_circuit.to_tableau(
        ignore_noise=True,
        ignore_measurement=True,
        ignore_reset=True,
    )


def propagate_error(
    error_pauli: stim.PauliString,
    bridge_tableau: stim.Tableau,
) -> stim.PauliString:
    """Propagate a Pauli error through a bridge circuit.

    Computes the conjugation T * P * T^-1, where T is the bridge
    Clifford and P is the error Pauli operator.

    Args:
        error_pauli: Pauli string representing the error. Should be
            defined on the full qubit space (data + ancilla), with
            identity on ancilla qubits.
        bridge_tableau: Tableau of the bridge circuit.

    Returns:
        The conjugated Pauli string after propagation.
    """
    return bridge_tableau(error_pauli)


def extract_data_qubit_pauli(
    full_pauli: stim.PauliString,
    data_qubit_indices: list[int],
) -> stim.PauliString:
    """Extract the data-qubit portion of a Pauli string.

    After propagation, the result may have non-trivial Pauli on
    ancilla qubits. This extracts just the data qubit part.

    Args:
        full_pauli: Pauli string on all qubits.
        data_qubit_indices: Indices of data qubits in the full circuit.

    Returns:
        PauliString containing only the data-qubit Paulis.
    """
    n = len(data_qubit_indices)
    result = stim.PauliString(n)
    sign = full_pauli.sign
    for i, q in enumerate(data_qubit_indices):
        p = full_pauli[q]  # 0=I, 1=X, 2=Y, 3=Z
        if p != 0:
            result[i] = p
    result.sign = sign
    return result


def check_logical_effect(
    propagated_error: stim.PauliString,
    logical_observable: stim.PauliString,
) -> bool:
    """Check if a propagated error flips the logical observable.

    An error flips the logical if and only if it anticommutes with
    the logical observable.

    Args:
        propagated_error: The Pauli error after propagation.
        logical_observable: The logical observable as a PauliString.

    Returns:
        True if the error flips the logical (anticommutes), False otherwise.
    """
    return not propagated_error.commutes(logical_observable)


def verify_propagation_with_flows(
    circuit: stim.Circuit,
    input_logical: stim.PauliString,
    output_logical: stim.PauliString,
) -> bool:
    """Verify that a circuit preserves a logical operator using flow analysis.

    Checks whether the circuit has a flow from input_logical to
    output_logical, confirming that the Clifford circuit maps one
    logical to the other.

    Args:
        circuit: The circuit to verify (typically the bridge circuit).
        input_logical: The logical operator at the input.
        output_logical: The expected logical operator at the output.

    Returns:
        True if the flow exists (logical is preserved/mapped correctly).
    """
    flow = stim.Flow(
        input=input_logical,
        output=output_logical,
    )
    return circuit.has_flow(flow, unsigned=True)


def get_data_qubit_indices(circuit: stim.Circuit) -> list[int]:
    """Identify data qubit indices from a generated surface code circuit.

    Data qubits are those that are measured (M) at the end of the
    circuit but not reset-measured (MR) during syndrome extraction.
    For generated circuits, data qubits are the ones included in
    the final M instruction (not MR).

    This heuristic works by finding qubits targeted by the final
    measurement layer (the last M instruction block) in the circuit.

    Args:
        circuit: A generated surface code circuit.

    Returns:
        Sorted list of data qubit indices.
    """
    flat = circuit.flattened()
    data_qubits = set()

    # Walk backwards to find the final M instructions
    for inst in reversed(list(flat)):
        if inst.name == "M":
            for t in inst.targets_copy():
                data_qubits.add(t.qubit_value if hasattr(t, 'qubit_value') else t.value)
        elif inst.name in ("MR", "MX", "MY", "MZ"):
            break  # Past the final measurement layer

    if not data_qubits:
        # Fallback: walk forward and collect all M targets
        for inst in flat:
            if inst.name == "M":
                for t in inst.targets_copy():
                    data_qubits.add(t.qubit_value if hasattr(t, 'qubit_value') else t.value)

    return sorted(data_qubits)
