from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class StabiliserShape:
    """Local stabiliser shape for a code.

    Attributes:
        pauli_type: The Pauli type of the stabiliser ('X' or 'Z').
        connectivity_subgraph: A graph representing the local connectivity of qubits
            in this stabiliser.
        sec_cycle_length: The nominal SEC cycle length to use for
            schedules later.
        qubit_map: A mapping from template indices to code qubit IDs.
        label: A label for the stabiliser shape.
        preferred_roots: Optional list of preferred local roots; if None,
            root not considered for preference.
        preferred_edges: Optional dictionary mapping edges to required timesteps.
        schedule_hint: Optional pre-defined schedule used as a hint only.
        layer_hint: Optional layer index at which to hint this stabiliser schedule.
        redundant_edges: Optional set of redundant edges; maintained for builder compatibility.
    """

    pauli_type: str  # 'X' or 'Z'
    connectivity_subgraph: nx.Graph  # nodes 0..k-1
    sec_cycle_length: int  # nominal SEC cycle length to use for schedules later
    qubit_map: list[int]  # len = k; template index -> code qubit id
    label: str
    # Preference inputs (all optional)
    preferred_roots: list[int] | None = (
        None  # list of preferred local roots; if None, root not considered for preference
    )
    preferred_edges: dict[tuple[int, int], list[int] | None] | None = (
        None  # edge -> optional list of required timesteps
    )
    # Solver hints (optional and soft)
    schedule_hint: list[list[tuple[int, int]]] | None = (
        None  # pre-defined schedule used as a hint only
    )
    layer_hint: int | None = (
        None  # layer index at which to hint this stabiliser schedule
    )
    # Accepted but unused; maintained for builder compatibility
    redundant_edges: set[tuple[int, int]] | None = None


@dataclass
class BaseCode:
    """Base code representation for a stabiliser code.

    Attributes:
        num_qubits: Total number of qubits in the code.
        connectivity_graph: Graph representing the connectivity of qubits.
        shapes: List of stabiliser shapes in the code.
        connection_classes: Optional list of labelled connection classes for visualisation.
    """

    num_qubits: int
    connectivity_graph: nx.Graph
    shapes: list[StabiliserShape]
    # Optional: labelled connection classes for visualisation (undirected)
    connection_classes: list[tuple[int, int, str]] | None = None

    def validate_local_connectivity(self) -> None:
        G = self.connectivity_graph
        for sh in self.shapes:
            S = sh.connectivity_subgraph
            qmap = sh.qubit_map
            for u, v in S.edges():
                a = qmap[u]
                b = qmap[v]
                if not G.has_edge(a, b):
                    raise RuntimeError(
                        f"Connectivity mismatch for {sh.label}: edge {(u, v)} -> {(a, b)} not in device"
                    )
