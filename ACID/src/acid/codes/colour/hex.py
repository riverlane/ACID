from __future__ import annotations

import networkx as nx

from acid.base_code import BaseCode, StabiliserShape


def _assert_odd_distance(d: int) -> None:
    if d < 3 or (d % 2) == 0:
        raise ValueError("d must be odd and >= 3")


def in_bounds_hex(d: int, x: int, y: int) -> bool:
    """Legacy helper (not used by the new builder). Kept for compatibility."""
    H = (3 * (d - 1)) // 2
    return 0 <= x <= d - 1 and 0 <= y <= H


def _path_graph(n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def _cycle_graph(n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
    return G


def build_colour_hex_code(
    d: int, *, deg4: bool = False
) -> tuple[BaseCode, dict[tuple[int, int], int]]:
    """Colour code on a d x (3/2)(d-1) lattice built from explicit stabiliser shapes.

    Shapes and placement follow the specification:
      - Basic hex rectangles (6-node cyclic), labelled at (x,y)=(2i,2j) and (2i-1,2j+1)
        with 0<=x<=d-1 and 0<=y<=min(3x-2, -3x + (3d-7)).
        Extra rectangles at (2i+1, 6i+3) and (d-2i-2, 6i+1) for i=0..(d-3)//4.
        If deg4, add extra edge between (x,y+1) and (x+1,y+1).
      - Left triangles (weight 4 path) centred at (2i, 6i) for i=0..(2d+1)//4.
      - Right triangles (weight 4 path) centred at (d-2i-2, 6i) for i=0..(2d-3)//4.
      - Spurs (weight 2 path) at (2i+1, 6i+4) for i=0..(d-3)//4 and at (d-2i-1, 6i+2) for i=0..(d+1)//4.
      - Squares (weight 4 cyclic) at (2i+1, 0) for i=0..(d-1)//2.

    The global device connectivity is inferred as the union of edges used by all stabilisers.
    SEC cycle length is 3 for all shapes. For each placement, both X and Z variants are added.
    """
    # Map requested logical distance d to effective construction distance d_eff = 2*d - 3
    # and proceed exactly as before using d_eff in all placement formulae.
    d = int(2 * int(d) - 3)
    _assert_odd_distance(d)

    H = (3 * (d - 1)) // 2

    # Coordinate registry built on demand while placing shapes
    coords: dict[tuple[int, int], int] = {}

    def in_dom(x: int, y: int) -> bool:
        return 0 <= x <= d - 1 and 0 <= y <= H

    def ensure_qid(x: int, y: int) -> int:
        key = (x, y)
        if key not in coords:
            coords[key] = len(coords)
        return coords[key]

    # Global connectivity inferred from shapes
    G = nx.Graph()

    shapes: list[StabiliserShape] = []

    def add_shape(
        label: str,
        nodes_xy: list[tuple[int, int]],
        edges_pairs: list[tuple[int, int]],
        sec_len: int = 3,
    ) -> None:
        # Skip if any node is out of domain
        if any(not in_dom(x, y) for (x, y) in nodes_xy):
            return
        # Local indexing
        # {xy: i for i, xy in enumerate(nodes_xy)}
        # Ensure qubits and add edges to global graph
        qmap = [ensure_qid(x, y) for (x, y) in nodes_xy]
        for ai, bi in edges_pairs:
            qa = qmap[ai]
            qb = qmap[bi]
            if qa == qb:
                continue
            G.add_edge(qa, qb)
        # Build local connectivity graph
        LG = nx.Graph()
        LG.add_nodes_from(range(len(nodes_xy)))
        for ai, bi in edges_pairs:
            LG.add_edge(ai, bi)
        # Preferences:
        # - Preferred edges are exactly those not added by the 'deg4' rung (edge between local nodes 1 and 4).
        #   No timestep constraints (values None).
        # - Preferred roots only for full 6-node rectangles: the 4 corners [0,2,3,5].
        pref_edges = None
        pref_roots = None
        if len(nodes_xy) == 6:
            # Determine if the deg4 rung (1,4) is present; exclude it from preferred edges if so.
            rung_present = any(
                ((a == 1 and b == 4) or (a == 4 and b == 1)) for (a, b) in edges_pairs
            )
            pe: dict[tuple[int, int], None] = {}
            for ai, bi in edges_pairs:
                u, v = (ai, bi) if ai <= bi else (bi, ai)
                if rung_present and (u, v) == (1, 4):
                    continue
                pe[(u, v)] = None
            pref_edges = pe
            # Preferred root depends on rectangle family:
            # - y odd (2i-1, 2j+1) family (including extras): top-left corner (local 2)
            # - y even (2i, 2j) family: bottom-right corner (local 5)
            y0 = nodes_xy[0][1]
            pref_roots = [2] if (y0 % 2 == 1) else [5]
        # Emit both X and Z stabilisers with preferences (if any)
        shapes.append(
            StabiliserShape(
                "X",
                LG,
                sec_len,
                qmap,
                f"{label}X",
                preferred_roots=pref_roots,
                preferred_edges=pref_edges,
            )
        )
        shapes.append(
            StabiliserShape(
                "Z",
                LG,
                sec_len,
                qmap,
                f"{label}Z",
                preferred_roots=pref_roots,
                preferred_edges=pref_edges,
            )
        )

    # 1) Basic hex rectangles
    # Base placements constrained by y <= min(3x-2, -3x + (3d-7)) with x in [0..d-1]
    for x in range(d):
        y_max = min(3 * x - 2, -3 * x + (3 * d - 7))
        if y_max < 0:
            continue
        for y in range(y_max + 1):
            # Families: (2i,2j) i.e. x even, y even; and (2i-1,2j+1) i.e. x odd, y odd
            if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
                # Node order around the rectangle perimeter
                pts = [
                    (x, y),
                    (x, y + 1),
                    (x, y + 2),
                    (x + 1, y + 2),
                    (x + 1, y + 1),
                    (x + 1, y),
                ]
                # 6-cycle edges
                cyc_edges = [(i, (i + 1) % 6) for i in range(6)]
                # Optional deg4 rung: (x,y+1)-(x+1,y+1) corresponds to indices 1 and 4
                edges = list(cyc_edges)
                if deg4:
                    edges.append((1, 4))
                add_shape(label=f"hex({x},{y})/", nodes_xy=pts, edges_pairs=edges)

    # Extra rectangle stabs
    r_extra_max = (d - 3) // 4
    parity_adjust = d % 4 == 1
    for i in range(max(r_extra_max, -1) + 1):
        # Left extras: (2i+1, 6i+3)
        x, y = 2 * i + 1, 6 * i + 3
        pts = [
            (x, y),
            (x, y + 1),
            (x, y + 2),
            (x + 1, y + 2),
            (x + 1, y + 1),
            (x + 1, y),
        ]
        edges = [(i2, (i2 + 1) % 6) for i2 in range(6)]
        if deg4:
            edges.append((1, 4))
        add_shape(label=f"hex_extraA({x},{y})/", nodes_xy=pts, edges_pairs=edges)
        # Right extras
        if not parity_adjust:
            # Default placement: (d - 2i - 2, 6i + 1)
            x2, y2 = d - 2 * i - 2, 6 * i + 1
            pts2 = [
                (x2, y2),
                (x2, y2 + 1),
                (x2, y2 + 2),
                (x2 + 1, y2 + 2),
                (x2 + 1, y2 + 1),
                (x2 + 1, y2),
            ]
            edges2 = [(i2, (i2 + 1) % 6) for i2 in range(6)]
            if deg4:
                edges2.append((1, 4))
            add_shape(label=f"hex_extraB({x2},{y2})/", nodes_xy=pts2, edges_pairs=edges2)
        else:
            # Adjusted placement for d % 4 == 1: (d - 2i - 3, 6i + 4) for i = 0 .. d//4 - 1
            if i <= (d // 4 - 1):
                x2, y2 = d - 2 * i - 3, 6 * i + 4
                pts2 = [
                    (x2, y2),
                    (x2, y2 + 1),
                    (x2, y2 + 2),
                    (x2 + 1, y2 + 2),
                    (x2 + 1, y2 + 1),
                    (x2 + 1, y2),
                ]
                edges2 = [(i2, (i2 + 1) % 6) for i2 in range(6)]
                if deg4:
                    edges2.append((1, 4))
                add_shape(label=f"hex_extraB({x2},{y2})/", nodes_xy=pts2, edges_pairs=edges2)

    # 2) Left triangles: 4-node path (x,y)->(x+1,y)->(x+1,y+1)->(x+1,y+2)
    lt_max = (d + 1) // 4
    for i in range(lt_max + 1):
        x, y = 2 * i, 6 * i
        nodes = [(x, y), (x + 1, y), (x + 1, y + 1), (x + 1, y + 2)]
        edges = [(0, 1), (1, 2), (2, 3)]
        add_shape(label=f"triL({x},{y})/", nodes_xy=nodes, edges_pairs=edges)

    # 3) Right triangles: 4-node path (x+1,y)->(x,y)->(x,y+1)->(x,y+2)
    # d % 4 == 1 (e.g., 9,13): (d-2i-2, 6i+1)
    # d % 4 == 3 (e.g., 7,11): (d-2i-3, 6i+4)
    rt_max = (2 * d - 3) // 4
    if parity_adjust:
        for i in range(rt_max + 1):
            x, y = d - 2 * i - 2, 6 * i + 1
            nodes = [(x + 1, y), (x, y), (x, y + 1), (x, y + 2)]
            edges = [(0, 1), (1, 2), (2, 3)]
            add_shape(label=f"triR({x},{y})/", nodes_xy=nodes, edges_pairs=edges)
    else:
        for i in range(rt_max + 1):
            x, y = d - 2 * i - 3, 6 * i + 4
            nodes = [(x + 1, y), (x, y), (x, y + 1), (x, y + 2)]
            edges = [(0, 1), (1, 2), (2, 3)]
            add_shape(label=f"triR({x},{y})/", nodes_xy=nodes, edges_pairs=edges)

    # 4) Spurs: 2-node vertical path at the listed positions
    sp1_max = (d - 3) // 4
    for i in range(sp1_max + 1):
        x, y = 2 * i + 1, 6 * i + 4
        nodes = [(x, y), (x, y + 1)]
        add_shape(label=f"spurA({x},{y})/", nodes_xy=nodes, edges_pairs=[(0, 1)])
    sp2_max = (d + 1) // 4
    if not parity_adjust:
        for i in range(sp2_max + 1):
            x, y = d - 2 * i - 1, 6 * i + 2
            nodes = [(x, y), (x, y + 1)]
            add_shape(label=f"spurB({x},{y})/", nodes_xy=nodes, edges_pairs=[(0, 1)])
    else:
        # Adjusted right spurs: sit on adjusted right extras at (d-2i-2, 6i+5)
        for i in range(max(d // 4 - 1, -1) + 1):
            x, y = d - 2 * i - 2, 6 * i + 5
            nodes = [(x, y), (x, y + 1)]
            add_shape(label=f"spurB({x},{y})/", nodes_xy=nodes, edges_pairs=[(0, 1)])

    # 5) Squares: 4-cycle at (2i+1, 0)
    # 5) Squares: 4-cycle at (2i+1, 0), i = 0 .. (d-1)//2 (no parity adjustment)
    sq_max = (d - 1) // 2
    for i in range(sq_max + 1):
        x, y = 2 * i + 1, 0
        nodes = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        add_shape(label=f"square({x},{y})/", nodes_xy=nodes, edges_pairs=edges)

    # Finalise global graph: add all discovered nodes
    G.add_nodes_from(range(len(coords)))

    base = BaseCode(num_qubits=len(coords), connectivity_graph=G, shapes=shapes)
    base.validate_local_connectivity()
    return base, coords


__all__ = ["build_colour_hex_code", "in_bounds_hex"]
