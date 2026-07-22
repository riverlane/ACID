from __future__ import annotations

"""Per-build deduplication of StabiliserTemplate instances."""

from typing import List, Tuple, Optional
import networkx as nx

from .types import StabiliserTemplate


class TemplateFactory:
    """
    Per-build deduplication of StabiliserTemplate instances.

    Keyed by (pauli_type, n_qubits, SEC_len, sorted undirected edge list).
    """

    def __init__(self) -> None:
        self._cache: dict[Tuple, StabiliserTemplate] = {}

    @staticmethod
    def _key(
        pauli_type: str,
        n_qubits: int,
        connectivity_subgraph: nx.Graph,
        SEC_cycle_length: int,
    ) -> Tuple:
        edges = tuple(
            sorted((min(u, v), max(u, v)) for (u, v) in connectivity_subgraph.edges())
        )
        return (pauli_type, int(n_qubits), int(SEC_cycle_length), edges)

    def get_or_create(
        self,
        pauli_type: str,
        n_qubits: int,
        connectivity_subgraph: nx.Graph,
        SEC_cycle_length: int,
        name: str = "",
        preferred_roots: Optional[List[int]] = None,
        preferred_edges: Optional[dict[Tuple[int, int], Optional[List[int]]]] = None,
        schedule_hint: List[List[Tuple[int, int]]] | None = None,
        layer_hint: int | None = None,
    ) -> StabiliserTemplate:
        k = self._key(pauli_type, n_qubits, connectivity_subgraph, SEC_cycle_length)
        obj = self._cache.get(k)
        if obj is not None:
            return obj
        obj = StabiliserTemplate(
            pauli_type,
            n_qubits,
            connectivity_subgraph,
            SEC_cycle_length,
            name=name,
            preferred_roots=preferred_roots,
            preferred_edges=preferred_edges,
            schedule_hint=schedule_hint,
            layer_hint=layer_hint,
        )
        self._cache[k] = obj
        return obj

    def stats(self) -> dict:
        return {"unique": len(self._cache)}
