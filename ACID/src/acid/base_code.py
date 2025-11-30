from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import networkx as nx


@dataclass(frozen=True)
class StabiliserShape:
    pauli_type: str                 # 'X' or 'Z'
    connectivity_subgraph: nx.Graph # nodes 0..k-1
    sec_cycle_length: int           # nominal SEC cycle length to use for schedules later
    qubit_map: List[int]            # len = k; template index -> code qubit id
    label: str
    # Preference inputs (all optional)
    preferred_roots: Optional[List[int]] = None  # list of preferred local roots; if None, root not considered for preference
    preferred_edges: Optional[dict[tuple[int, int], Optional[List[int]]]] = None  # edge -> optional list of required timesteps
    # Solver hints (optional and soft)
    schedule_hint: Optional[List[List[tuple[int, int]]]] = None  # pre-defined schedule used as a hint only
    layer_hint: Optional[int] = None       # layer index at which to hint this stabiliser schedule
    # Accepted but unused; maintained for builder compatibility
    redundant_edges: Optional[set[tuple[int, int]]] = None


@dataclass
class BaseCode:
    num_qubits: int
    connectivity_graph: nx.Graph
    shapes: List[StabiliserShape]
    # Optional: labelled connection classes for visualisation (undirected)
    connection_classes: Optional[List[Tuple[int, int, str]]] = None

    def validate_local_connectivity(self) -> None:
        G = self.connectivity_graph
        for sh in self.shapes:
            S = sh.connectivity_subgraph
            qmap = sh.qubit_map
            for (u, v) in S.edges():
                a = qmap[u]; b = qmap[v]
                if not G.has_edge(a, b):
                    raise RuntimeError(
                        f"Connectivity mismatch for {sh.label}: edge {(u, v)} -> {(a, b)} not in device"
                    )
