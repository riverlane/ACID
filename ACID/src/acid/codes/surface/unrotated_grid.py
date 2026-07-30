from __future__ import annotations

import networkx as nx

from acid.base_code import BaseCode, StabiliserShape


def _path_template(n: int = 4) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    [G.add_edge(i, i + 1) for i in range(n - 1)]
    return G


def build_unrotated_surface_grid_code(
    d: int,
) -> tuple[BaseCode, dict[tuple[int, int], int]]:
    """
    Distance-d unrotated surface (grid connectivity):
      - Grid size: (2d-1) x (2d-1), coords in [0..2d-2]
      - Qubits at (even,even) and (odd,odd)
      - X stabs at (even,odd), Z stabs at (odd,even); 4 neighbors (up,right,down,left) where in-bounds
      - Connectivity: diagonal links between qubits (±1,±1)
    Returns (G, stabs, coord_to_qid) without constructing a Code object.
    """
    W = 2 * d - 1
    H = 2 * d - 1
    coord_to_qid: dict[tuple[int, int], int] = {}
    qlist: list[tuple[int, int]] = []
    qid = 0
    for x in range(W):
        for y in range(H):
            if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
                coord_to_qid[(x, y)] = qid
                qlist.append((x, y))
                qid += 1

    def hasq(x: int, y: int) -> bool:
        return (x, y) in coord_to_qid

    def idq(x: int, y: int) -> int:
        return coord_to_qid[(x, y)]

    # connectivity: diagonals (undirected)
    G = nx.Graph()
    G.add_nodes_from(range(len(qlist)))
    for (x, y), q in coord_to_qid.items():
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            xn, yn = x + dx, y + dy
            if hasq(xn, yn):
                G.add_edge(q, idq(xn, yn))

    # Add boundary reinforcement edges along the four borders to ensure
    # local stabiliser connectivity is present at boundaries.
    # Bottom and top rows (horizontal): (2j, y) - (2j+2, y)
    y_bottom = 0
    y_top = H - 1
    for j in range((W - 1) // 2):
        x1 = 2 * j
        x2 = x1 + 2
        if hasq(x1, y_bottom) and hasq(x2, y_bottom):
            G.add_edge(idq(x1, y_bottom), idq(x2, y_bottom))
        if hasq(x1, y_top) and hasq(x2, y_top):
            G.add_edge(idq(x1, y_top), idq(x2, y_top))
    # Left and right columns (vertical): (x, 2j) - (x, 2j+2)
    x_left = 0
    x_right = W - 1
    for j in range((H - 1) // 2):
        y1 = 2 * j
        y2 = y1 + 2
        if hasq(x_left, y1) and hasq(x_left, y2):
            G.add_edge(idq(x_left, y1), idq(x_left, y2))
        if hasq(x_right, y1) and hasq(x_right, y2):
            G.add_edge(idq(x_right, y1), idq(x_right, y2))

    shapes: list[StabiliserShape] = []

    # Build stabilisers using cyclic local connectivity.
    # - X at (even,odd)
    for x in range(0, W, 2):
        for y in range(1, H, 2):
            # Order: up, right, down, left (drop missing for boundaries)
            # Presence flags
            u = hasq(x, y + 1)
            r = hasq(x + 1, y)
            d = hasq(x, y - 1)
            l = hasq(x - 1, y)
            # Neighbor list in an order that ensures consecutive pairs are connected
            if u and r and d and not l:
                order = [(x, y + 1), (x + 1, y), (x, y - 1)]
            elif u and r and not d and l:
                order = [(x - 1, y), (x, y + 1), (x + 1, y)]
            elif u and not r and d and l:
                order = [(x, y + 1), (x, y - 1), (x - 1, y)]
            elif not u and r and d and l:
                order = [(x + 1, y), (x, y - 1), (x - 1, y)]
            else:
                # Bulk or generic: up, right, down, left filtered
                order = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
            neigh_ids = [idq(px, py) for (px, py) in order if hasq(px, py)]
            if len(neigh_ids) >= 3:
                k = len(neigh_ids)
                # Local connectivity: C4 for bulk; for boundary k=3 use a path (P3) to match device edges
                if k == 4:
                    local = nx.cycle_graph(4)
                else:
                    local = nx.path_graph(3)
                shapes.append(StabiliserShape("X", local, 2, neigh_ids, f"X({x},{y})"))

    # - Z at (odd,even)
    for x in range(1, W, 2):
        for y in range(0, H, 2):
            u = hasq(x, y + 1)
            r = hasq(x + 1, y)
            d = hasq(x, y - 1)
            l = hasq(x - 1, y)
            if u and r and d and not l:
                order = [(x, y + 1), (x + 1, y), (x, y - 1)]
            elif u and r and not d and l:
                order = [(x - 1, y), (x, y + 1), (x + 1, y)]
            elif u and not r and d and l:
                order = [(x, y + 1), (x, y - 1), (x - 1, y)]
            elif not u and r and d and l:
                order = [(x + 1, y), (x, y - 1), (x - 1, y)]
            else:
                order = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
            neigh_ids = [idq(px, py) for (px, py) in order if hasq(px, py)]
            if len(neigh_ids) >= 3:
                k = len(neigh_ids)
                if k == 4:
                    local = nx.cycle_graph(4)
                else:
                    local = nx.path_graph(3)
                shapes.append(StabiliserShape("Z", local, 2, neigh_ids, f"Z({x},{y})"))

    base = BaseCode(num_qubits=len(qlist), connectivity_graph=G, shapes=shapes)
    base.validate_local_connectivity()
    return base, coord_to_qid
