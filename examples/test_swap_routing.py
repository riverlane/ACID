#!/usr/bin/env python3
"""Test script for the CP-SAT swap routing solver.

Constructs a small periodic square-grid embedding, builds nearest-neighbour
connections on the torus, and exercises solve_swap_routing + verify_swap_routing
under several scenarios.
"""

from __future__ import annotations

from acid.codes.bb.algebra import GroupRing
from acid.embedding import SquareGridEmbedding
from acid.solver.swap import solve_swap_routing, verify_swap_routing


def build_torus_connections(embedding: SquareGridEmbedding) -> list[tuple[int, int]]:
    """Build nearest-neighbour connections on the periodic square grid.

    Each qubit (a, b, c) is connected to:
      - its L/R partner within the same cell: (a, b, 1-c)
      - the same-type qubit one step in a: (a+1, b, c)
      - the same-type qubit one step in b: (a, b+1, c)
    """
    ring = embedding.ring
    connections: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(i: int, j: int) -> None:
        e = (min(i, j), max(i, j))
        if e not in seen:
            seen.add(e)
            connections.append(e)

    for a in range(ring.l):
        for b in range(ring.m):
            for c in range(2):
                q = embedding.qubit_coords_to_index(a, b, c)
                # L-R intra-cell partner
                add(q, embedding.qubit_coords_to_index(a, b, 1 - c))
                # neighbour in a direction
                add(q, embedding.qubit_coords_to_index(a + 1, b, c))
                # neighbour in b direction
                add(q, embedding.qubit_coords_to_index(a, b + 1, c))

    return connections


def run_case(
    label: str,
    ring: GroupRing,
    offset: tuple[int, int],
    dead_positions: set[int],
    dead_connections: set[tuple[int, int]],
    dont_care: set[int] | None = None,
    max_layers: int = 8,
    time_limit_s: float = 30.0,
) -> None:
    embedding = SquareGridEmbedding(ring=ring)
    connections = build_torus_connections(embedding)

    print(f"\n{'=' * 60}")
    print(f"Case: {label}")
    print(f"  Ring: Z_{ring.l} x Z_{ring.m}  ({embedding.num_qubits} qubits)")
    print(f"  Offset: {offset}")
    print(f"  Dead positions: {sorted(dead_positions)}")
    print(f"  Dead connections: {sorted(dead_connections)}")
    if dont_care:
        print(f"  Don't-care positions: {len(dont_care)} qubits")

    layers = solve_swap_routing(
        embedding=embedding,
        connections=connections,
        dead_positions=dead_positions,
        dead_connections=dead_connections,
        offset=offset,
        root_qubits=dont_care,
        max_layers=max_layers,
        time_limit_s=time_limit_s,
    )

    if layers is None:
        print("  Result: NO SOLUTION FOUND")
        return

    print(f"  Result: {len(layers)} swap layer(s)")
    for t, layer in enumerate(layers):
        print(f"    Layer {t}: {layer if layer else '(no swaps)'}")

    ok = verify_swap_routing(
        layers,
        embedding.num_qubits,
        _compute_target(embedding, offset, dead_positions, dont_care or set()),
    )
    print(f"  Verification: {'PASS' if ok else 'FAIL'}")


def _compute_target(
    embedding: SquareGridEmbedding,
    offset: tuple[int, int],
    dead_positions: set[int],
    dont_care: set[int],
) -> dict[int, int]:
    """Reproduce the target dict the solver uses internally (for verification)."""
    N = embedding.num_qubits
    da, db = offset
    live = sorted(set(range(N)) - dead_positions - dont_care)
    shifted = embedding.get_shifted_positions(live, da, db)
    target: dict[int, int] = {}
    for q, dest in zip(live, shifted):
        assert dest not in dead_positions, f"Qubit {q} is shifted to position {dest} which is dead"
        target[dest] = q
    return target


def main() -> None:
    # -----------------------------------------------------------------------
    # Case 1: trivial — offset (0, 0) should need zero effective swaps
    # -----------------------------------------------------------------------
    run_case(
        label="Zero offset (identity)",
        ring=GroupRing(3, 3),
        offset=(0, 0),
        dead_positions=set(),
        dead_connections=set(),
        max_layers=4,
    )

    # -----------------------------------------------------------------------
    # Case 2: small ring, shift by (1, 0), no dropouts
    # -----------------------------------------------------------------------
    run_case(
        label="3x3 ring, shift (1,0), no dropouts",
        ring=GroupRing(3, 3),
        offset=(1, 0),
        dead_positions=set(),
        dead_connections=set(),
        max_layers=6,
    )

    # -----------------------------------------------------------------------
    # Case 3: small ring, shift by (0, 1), no dropouts
    # -----------------------------------------------------------------------
    run_case(
        label="3x3 ring, shift (0,1), no dropouts",
        ring=GroupRing(3, 3),
        offset=(0, 1),
        dead_positions=set(),
        dead_connections=set(),
        max_layers=6,
    )

    # -----------------------------------------------------------------------
    # Case 4: shift (1, 0) with one dead qubit (qubit 0)
    # -----------------------------------------------------------------------
    run_case(
        label="3x3 ring, shift (1,0), qubit 0 dead",
        ring=GroupRing(3, 3),
        offset=(1, 0),
        dead_positions={0},
        dead_connections=set(),
        max_layers=8,
        dont_care={12},
    )

    # -----------------------------------------------------------------------
    # Case 5: 2x2 ring (8 qubits), shift (1, 1)
    # -----------------------------------------------------------------------
    run_case(
        label="2x2 ring, shift (1,1), no dropouts",
        ring=GroupRing(2, 2),
        offset=(1, 1),
        dead_positions=set(),
        dead_connections=set(),
        max_layers=6,
    )

    # -----------------------------------------------------------------------
    # Case 6: 10x10 ring, shift (1,0), 2 dead qubits, 3 dead connections,
    #         97 don't-care qubits (every 2nd qubit starting at qubit 4,
    #          excluding dead)
    #   dead qubits: 0=(0,0,L), 50=(2,5,L)
    #   dead connections: (2,3)=(0,1,L)-(0,1,R), (4,6)=(0,2,L)-(0,3,L),
    #                     (30,32)=(1,0,L)-(1,1,L)
    # -----------------------------------------------------------------------
    dead = {0, 50}
    dont_care = {q for q in range(4, 200, 2) if q not in dead}
    run_case(
        label="10x10 ring, shift (1,0), 2 dead qubits, 3 dead connections, 97 dont-care",
        ring=GroupRing(10, 10),
        offset=(1, 0),
        dead_positions=dead,
        dead_connections={(2, 3), (4, 6), (30, 32)},
        dont_care=dont_care,
        max_layers=10,
        time_limit_s=120.0,
    )


if __name__ == "__main__":
    main()
