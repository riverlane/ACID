from __future__ import annotations

from typing import Dict, List, Tuple
import networkx as nx

from acid.base_code import BaseCode, StabiliserShape


def _cycle4() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(2, 3); G.add_edge(3, 0)
    return G


def build_unrotated_surface_grid_square_edges_code(d: int) -> Tuple[BaseCode, Dict[Tuple[int,int], int]]:
    """
    Distance-d unrotated surface (grid connectivity with square edges to the boundary).

    - Coordinate range: 0..2d (inclusive). We place qubits at (even,even) and (odd,odd),
      except there are no qubits at the four extreme corners (0,0), (0,2d), (2d,0), (2d,2d).
    - X stabilisers at coords (x odd in 1..2d-1, y even in 2..2d-2), using four qubits at
      (x,y+1), (x-1,y), (x,y-1), (x+1,y) in that order; local connectivity is a 4-cycle.
      Device connectivity graph includes the four edges around each square.
    - Z stabilisers at coords (x even in 2..2d-2, y odd in 1..2d-1), using the same order and
      local connectivity.
    - Boundary single-qubit stabilisers (no new edges added):
      * Z on x = 2,4,..,2d-2 at y = 0 and y = 2d (single-qubit Z).
      * X on y = 2,4,..,2d-2 at x = 0 and x = 2d (single-qubit X).

    Returns (G, stabs, coord_to_qid).
    """
    W = 2 * d + 1
    H = 2 * d + 1
    coord_to_qid: Dict[Tuple[int, int], int] = {}
    qid = 0
    # Place qubits at (even,even) and (odd,odd) within [0..2d], excluding corners
    for x in range(W):
        for y in range(H):
            if (x % 2) == (y % 2):
                if (x in (0, 2 * d)) and (y in (0, 2 * d)):
                    continue
                coord_to_qid[(x, y)] = qid
                qid += 1

    def hasq(x: int, y: int) -> bool:
        return (x, y) in coord_to_qid

    def idq(x: int, y: int) -> int:
        return coord_to_qid[(x, y)]

    G = nx.Graph()
    G.add_nodes_from(range(qid))

    shapes: List[StabiliserShape] = []

    cyc4 = _cycle4()

    # X stabilisers: x odd, y even (bulk rows)
    for x in range(1, 2 * d, 2):
        for y in range(2, 2 * d, 2):
            # four compass qubits in order: up, left, down, right
            u = (x, y + 1)
            l = (x - 1, y)
            dwn = (x, y - 1)
            r = (x + 1, y)
            if not (hasq(*u) and hasq(*l) and hasq(*dwn) and hasq(*r)):
                continue
            qmap = [idq(*u), idq(*l), idq(*dwn), idq(*r)]
            shapes.append(StabiliserShape('X', cyc4, 2, qmap, f"X({x},{y})"))
            # device edges around the square
            G.add_edge(qmap[0], qmap[1])
            G.add_edge(qmap[1], qmap[2])
            G.add_edge(qmap[2], qmap[3])
            G.add_edge(qmap[3], qmap[0])

    # Z stabilisers: x even, y odd (bulk columns)
    for x in range(2, 2 * d, 2):
        for y in range(1, 2 * d, 2):
            u = (x, y + 1)
            l = (x - 1, y)
            dwn = (x, y - 1)
            r = (x + 1, y)
            if not (hasq(*u) and hasq(*l) and hasq(*dwn) and hasq(*r)):
                continue
            qmap = [idq(*u), idq(*l), idq(*dwn), idq(*r)]
            shapes.append(StabiliserShape('Z', cyc4, 2, qmap, f"Z({x},{y})"))
            G.add_edge(qmap[0], qmap[1])
            G.add_edge(qmap[1], qmap[2])
            G.add_edge(qmap[2], qmap[3])
            G.add_edge(qmap[3], qmap[0])

    # Boundary single-qubit stabilisers (no additional edges)
    G1 = nx.Graph(); G1.add_nodes_from([0])
    for x in range(2, 2 * d, 2):
        for y in [0, 2 * d]:
            if hasq(x, y):
                shapes.append(StabiliserShape('Z', G1, 2, [idq(x, y)], f"Z({x},{y})"))

    for y in range(2, 2 * d, 2):
        for x in [0, 2 * d]:
            if hasq(x, y):
                shapes.append(StabiliserShape('X', G1, 2, [idq(x, y)], f"X({x},{y})"))

    base = BaseCode(num_qubits=qid, connectivity_graph=G, shapes=shapes)
    base.validate_local_connectivity()
    return base, coord_to_qid
