from __future__ import annotations

import itertools

import networkx as nx

from acid.base_code import BaseCode, StabiliserShape


def _path_template(n: int = 4) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    [G.add_edge(i, i + 1) for i in range(n - 1)]
    return G


def build_unrotated_surface_hex_code(
    d: int,
) -> tuple[BaseCode, dict[tuple[int, int], int]]:
    """
    Distance-d unrotated surface (hex connectivity):
      - Grid size: W = H = 2d; coordinates in [0..2d-1]. One corner (W-1,H-1) is unused.
      - Qubits at (even,even) and (odd,odd), excluding the unused corner.
      - Connectivity edges (undirected) between qubits:
        E1: (2i,2j) -- (2i+1,2j+1)
        E2: (2i+1,2j+1) -- (2i+2,2j)
        E3: (2i+1,2j+1) -- (2i,2j+2)
      - Stabiliser placement (local 3- or 4-node paths, with connectivity matching device edges):
        * X at (even,odd): i in [0..d-1], j in [0..d-2] (y = 1,3,..,2d-3), using 3 (boundary) or 4 (bulk) qubits.
        * Z at (odd,even): i in [0..d-2], j in [0..d-1] (x = 1,3,..,2d-3), similarly.
      - Additional boundary single-qubit stabilisers (no new edges):
        * Z singles at virtual coords (x=1,3,..,2d-1, y=2d+1), implemented as single-qubit Z on boundary qubits (x, H-1).
        * X singles at virtual coords (x=2d+1, y=1,3,..,2d-1), implemented as single-qubit X on boundary qubits (W-1, y).
      Returns (G, stabs, coord_to_qid) without constructing a Code object.
    """
    W = 2 * d
    H = 2 * d
    coord_to_qid: dict[tuple[int, int], int] = {}
    qlist: list[tuple[int, int]] = []
    qid = 0
    for x in range(W):
        for y in range(H):
            if x == W - 1 and y == H - 1:
                continue
            if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
                coord_to_qid[(x, y)] = qid
                qlist.append((x, y))
                qid += 1

    def hasq(x: int, y: int) -> bool:
        return (x, y) in coord_to_qid

    def idq(x: int, y: int) -> int:
        return coord_to_qid[(x, y)]

    G = nx.Graph()
    G.add_nodes_from(range(len(qlist)))
    # E1 edges
    for i in range(d):
        for j in range(d):
            if i == d - 1 and j == d - 1:
                continue
            x1, y1 = 2 * i, 2 * j
            x2, y2 = 2 * i + 1, 2 * j + 1
            if hasq(x1, y1) and hasq(x2, y2):
                G.add_edge(idq(x1, y1), idq(x2, y2))
    # E2 edges
    for i in range(d):
        for j in range(d):
            x1, y1 = 2 * i + 1, 2 * j + 1
            x2, y2 = 2 * i + 2, 2 * j
            if hasq(x1, y1) and hasq(x2, y2):
                G.add_edge(idq(x1, y1), idq(x2, y2))

    # E3 edges
    for i in range(d):
        for j in range(d):
            x1, y1 = 2 * i + 1, 2 * j + 1
            x2, y2 = 2 * i, 2 * j + 2
            if hasq(x1, y1) and hasq(x2, y2):
                G.add_edge(idq(x1, y1), idq(x2, y2))

    shapes: list[StabiliserShape] = []
    # X stabs (even, odd) — local path (k=3 boundary, k=4 bulk)
    for i in range(d):
        x = 2 * i
        for j in range(d - 1):
            y = 2 * j + 1
            u = hasq(x, y + 1)
            r = hasq(x + 1, y)
            down = hasq(x, y - 1)
            l = hasq(x - 1, y)
            if u and r and down and not l:
                order = [(x, y + 1), (x + 1, y), (x, y - 1)]
            elif u and r and not down and l:
                order = [(x - 1, y), (x, y + 1), (x + 1, y)]
            elif u and not r and down and l:
                order = [(x, y + 1), (x - 1, y), (x, y - 1)]
            elif not u and r and down and l:
                order = [(x + 1, y), (x, y - 1), (x - 1, y)]
            else:
                order = [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]

            for (x1, y1), (x2, y2) in itertools.pairwise(order):
                if not G.has_edge(idq(x1, y1), idq(x2, y2)):
                    raise RuntimeError(
                        f"Local stabiliser connectivity missing for X({x},{y}) between {(x1, y1)} and {(x2, y2)}"
                    )
            path = _path_template(_k := len(order))
            qmap = [idq(px, py) for (px, py) in order]
            shapes.append(StabiliserShape("X", path, 2, qmap, f"X({x},{y})"))

    # Z stabs (odd, even) — local path (k=3 or 4)
    for i in range(d - 1):
        x = 2 * i + 1
        for j in range(d):
            y = 2 * j
            u = hasq(x, y + 1)
            r = hasq(x + 1, y)
            down = hasq(x, y - 1)
            l = hasq(x - 1, y)
            if u and r and down and not l:
                order = [(x, y + 1), (x + 1, y), (x, y - 1)]
            elif u and r and not down and l:
                order = [(x - 1, y), (x, y + 1), (x + 1, y)]
            elif u and not r and down and l:
                order = [(x, y + 1), (x - 1, y), (x, y - 1)]
            elif not u and r and down and l:
                order = [(x + 1, y), (x, y - 1), (x - 1, y)]
            else:
                order = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
            for (x1, y1), (x2, y2) in zip(order[1:], order[:1]):
                if not G.has_edge(idq(x1, y1), idq(x2, y2)):
                    raise RuntimeError(
                        f"Local stabiliser connectivity missing for Z({x},{y}) between {(x1, y1)} and {(x2, y2)}"
                    )
            path = _path_template(_k := len(order))
            qmap = [idq(px, py) for (px, py) in order]
            shapes.append(StabiliserShape("Z", path, 2, qmap, f"Z({x},{y})"))

    # Boundary single-qubit stabilisers (no new edges)
    G1 = nx.Graph()
    G1.add_nodes_from([0])
    # Z singles along the top boundary: virtual y = H+1 (= 2d+1), odd x in [1..W-1]
    for x in range(1, W, 2):
        if hasq(x, H - 1):
            q = idq(x, H - 1)
            shapes.append(StabiliserShape("Z", G1, 2, [q], f"Z({x},{H + 1})"))
    # X singles along the right boundary: virtual x = W+1 (= 2d+1), odd y in [1..H-1]
    for y in range(1, H, 2):
        if hasq(W - 1, y):
            q = idq(W - 1, y)
            shapes.append(StabiliserShape("X", G1, 2, [q], f"X({W + 1},{y})"))

    base = BaseCode(num_qubits=len(qlist), connectivity_graph=G, shapes=shapes)
    base.validate_local_connectivity()
    return base, coord_to_qid
