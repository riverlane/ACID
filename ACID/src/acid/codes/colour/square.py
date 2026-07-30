from __future__ import annotations

import networkx as nx

from acid.base_code import BaseCode, StabiliserShape


def _assert_odd_distance(d: int) -> None:
    if d < 3 or (d % 2) == 0:
        raise ValueError("d must be odd and >= 3")


def in_bounds_square(d: int, x: int, y: int) -> bool:
    """Bounds for square-grid colour code (type 1).

    Inequalities:
        y <= x + floor(x/2) + 2
        y <= -x - ceil((x+1)/2) + 3d - 3
        y >= 0
    And domain: -1 <= x <= 2d-3 and 0 <= y <= (3d-3)//2.
    """
    if x < -1 or x > 2 * d - 3:
        return False
    y_max = (3 * d - 3) // 2
    if y < 0 or y > y_max:
        return False
    lhs1 = y
    rhs1 = x + (x // 2) + 2
    # ceil((x+1)/2) for integer x is (x + 2) // 2
    rhs2 = (-x) - ((x + 2) // 2) + 3 * d - 3
    return (lhs1 <= rhs1) and (lhs1 <= rhs2) and (y >= 0)


def _add_cartesian_connectivity(
    G: nx.Graph, coords: dict[tuple[int, int], int], d: int
) -> None:
    # Four-neighbour connectivity (N,S,E,W) within bounds
    for (x, y), q in coords.items():
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            xn, yn = x + dx, y + dy
            if (xn, yn) in coords:
                # Networkx handles duplicate edges gracefully
                G.add_edge(q, coords[(xn, yn)])


def _path_graph(n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def _local_graph_from_coords(
    order_edges: list[tuple[tuple[int, int], tuple[int, int]]],
    qmap_list: list[tuple[int, int]],
) -> tuple[nx.Graph, list[int]]:
    """Build a local graph and qubit map from coordinate pairs and desired edges.

    - qmap_list: list of global coordinate points included in the stabiliser.
    - order_edges: list of coordinate-pair edges to include if both endpoints are present.
    Returns (local_graph, qmap) where qmap maps local node index to device qubit id.
    """
    idx_of: dict[tuple[int, int], int] = {xy: i for i, xy in enumerate(qmap_list)}
    G = nx.Graph()
    G.add_nodes_from(range(len(qmap_list)))
    for a, b in order_edges:
        ia = idx_of.get(a)
        ib = idx_of.get(b)
        if ia is not None and ib is not None:
            G.add_edge(int(ia), int(ib))
    return G, list(range(len(qmap_list)))


def build_colour_square_code(d: int) -> tuple[BaseCode, dict[tuple[int, int], int]]:
    _assert_odd_distance(d)

    offset_x = 1
    offset_y = 1

    # Qubit placement: all integer (x,y) within domain and bounds
    coords: dict[tuple[int, int], int] = {}
    qid = 0
    for x in range(-1, 2 * d - 2):  # inclusive upper bound 2d-3
        for y in range((3 * d - 3) // 2 + 1):
            if in_bounds_square(d, x, y):
                coords[(x + offset_x, y + offset_y)] = qid
                qid += 1

    G = nx.Graph()
    G.add_nodes_from(range(qid))
    _add_cartesian_connectivity(G, coords, d)

    shapes: list[StabiliserShape] = []

    # Helper closures
    def hasq(x: int, y: int) -> bool:
        return (x + offset_x, y + offset_y) in coords

    def idq(x: int, y: int) -> int:
        return coords[(x + offset_x, y + offset_y)]

    half = (d - 1) // 2

    def add_inner_outer(color: str, L_of: callable, R_of: callable) -> None:
        SEC_length = 4
        for i in range(half + 1):
            for j in range(half + 1):
                Lx, Ly = L_of(i, j)
                Rx, Ry = R_of(i, j)
                if not (hasq(Lx, Ly) and hasq(Rx, Ry)):
                    continue
                # Inner (2-qubit path)
                path2 = _path_graph(2)
                qmap2 = [idq(Lx, Ly), idq(Rx, Ry)]

                # Schedule hint (example): gather X onto L, Z onto R in 4 steps
                # schedule_hint = [[], [] , [] , [(1,0)]]  # (1,0) means CNOT controlled on 0 targeting 1
                # layer_hint 1 for X, layer_hint 0 for Z
                # Inner stabs: preferred roots as before, plus a preferred edge (L-R) at timestep 3
                pref_edges_inner: dict[tuple[int, int], list[int] | None] = {
                    (0, 1): [3]
                }
                shapes.append(
                    StabiliserShape(
                        "X",
                        path2,
                        SEC_length,
                        qmap2,
                        f"{color}inX({i},{j})",
                        preferred_roots=[0],
                        preferred_edges=pref_edges_inner,
                    )
                )
                shapes.append(
                    StabiliserShape(
                        "Z",
                        path2,
                        SEC_length,
                        qmap2,
                        f"{color}inZ({i},{j})",
                        preferred_roots=[1],
                        preferred_edges=pref_edges_inner,
                    )
                )

                # Outer: include neighbors up/down/left/right around L and R if present
                nb_coords: list[tuple[int, int]] = []
                # Base points first to stabilise indexing order
                base_points = [(Lx, Ly), (Rx, Ry)]
                add_points = [
                    (Lx, Ly + 1),
                    (Rx, Ry + 1),  # above
                    (Lx, Ly - 1),
                    (Rx, Ry - 1),  # below
                    (Lx - 1, Ly),
                    (Rx + 1, Ry),  # left of L, right of R
                ]
                for xy in base_points + [p for p in add_points if hasq(*p)]:
                    nb_coords.append(xy)

                if len(nb_coords) % 2 != 0:
                    raise AssertionError(
                        f"Outer stabiliser does not have even weight at i={i},j={j}, color={color}"
                    )

                # Local connectivity per spec (include only edges whose endpoints exist)
                L = (Lx, Ly)
                R = (Rx, Ry)
                UL = (Lx, Ly + 1)
                UR = (Rx, Ry + 1)
                DL = (Lx, Ly - 1)
                DR = (Rx, Ry - 1)
                LL = (Lx - 1, Ly)
                RR = (Rx + 1, Ry)
                edges = [
                    (L, R),
                    (L, UL),
                    (UL, UR),
                    (R, UR),
                    (L, DL),
                    (DL, DR),
                    (R, DR),
                    (L, LL),
                    (R, RR),
                ]
                local_graph, local_order = _local_graph_from_coords(edges, nb_coords)
                coord_to_local = {xy: k for k, xy in enumerate(nb_coords)}
                qmap = [idq(*nb_coords[k]) for k in local_order]

                # Schedule hint (example) for outer (kept commented):
                # schedule_hint_x = [[], [],[],[(1,0)]]
                # schedule_hint_z = [[], [],[],[(1,0)]]
                # if hasq(*DL):
                #     schedule_hint_x[0].append((coord_to_local[DL], coord_to_local[L]))
                #     schedule_hint_z[0].append((coord_to_local[L], coord_to_local[DL]))
                # if hasq(*DR):
                #     schedule_hint_x[0].append((coord_to_local[DR], coord_to_local[R]))
                #     schedule_hint_z[0].append((coord_to_local[R], coord_to_local[DR]))
                # if hasq(*LL):
                #     schedule_hint_x[1].append((coord_to_local[LL], coord_to_local[L]))
                #     schedule_hint_z[1].append((coord_to_local[L], coord_to_local[LL]))
                # if hasq(*RR):
                #     schedule_hint_x[1].append((coord_to_local[RR], coord_to_local[R]))
                #     schedule_hint_z[1].append((coord_to_local[R], coord_to_local[RR]))
                # if hasq(*UL):
                #     schedule_hint_x[2].append((coord_to_local[UL], coord_to_local[L]))
                #     schedule_hint_z[2].append((coord_to_local[L], coord_to_local[UL]))
                # if hasq(*UR):
                #     schedule_hint_x[2].append((coord_to_local[UR], coord_to_local[R]))
                #     schedule_hint_z[2].append((coord_to_local[R], coord_to_local[UR]))

                # Preferred edges with timing constraints (if endpoints exist):
                # DL-L: t=0; DR-R: t=0; LL-L: t=1; RR-R: t=1; UL-L: t=2; UR-R: t=2; L-R: t=3
                preferred_edges: dict[tuple[int, int], list[int] | None] = {}

                def add_pref(
                    a_xy: tuple[int, int], b_xy: tuple[int, int], t: int
                ) -> None:
                    if hasq(*a_xy) and hasq(*b_xy):
                        a = coord_to_local[a_xy]
                        b = coord_to_local[b_xy]
                        u, v = (a, b) if a <= b else (b, a)
                        preferred_edges[(u, v)] = [int(t)]

                add_pref(DL, L, 0)
                add_pref(DR, R, 0)
                add_pref(LL, L, 1)
                add_pref(RR, R, 1)
                add_pref(UL, L, 2)
                add_pref(UR, R, 2)
                # L-R always exists in the local graph
                a = coord_to_local[L]
                b = coord_to_local[R]
                u, v = (a, b) if a <= b else (b, a)
                preferred_edges[(u, v)] = [3]

                # shapes.append(StabiliserShape('X', local_graph, SEC_length, qmap, f"{color}outX({i},{j})", preferred_roots=[0,1]))
                # shapes.append(StabiliserShape('Z', local_graph, SEC_length, qmap, f"{color}outZ({i},{j})", preferred_roots=[0,1]))

                shapes.append(
                    StabiliserShape(
                        "X",
                        local_graph,
                        SEC_length,
                        qmap,
                        f"{color}outX({i},{j})",
                        preferred_roots=[0],
                        preferred_edges=preferred_edges,
                    )
                )
                shapes.append(
                    StabiliserShape(
                        "Z",
                        local_graph,
                        SEC_length,
                        qmap,
                        f"{color}outZ({i},{j})",
                        preferred_roots=[1],
                        preferred_edges=preferred_edges,
                    )
                )

    # Red stabs
    add_inner_outer(
        "R",
        L_of=lambda i, j: (4 * i + 2 * j, 3 * j),
        R_of=lambda i, j: (4 * i + 2 * j + 1, 3 * j),
    )
    # Blue stabs
    add_inner_outer(
        "B",
        L_of=lambda i, j: (4 * i + 2 * j + 2, 3 * j + 1),
        R_of=lambda i, j: (4 * i + 2 * j + 3, 3 * j + 1),
    )
    # Green stabs
    add_inner_outer(
        "G",
        L_of=lambda i, j: (4 * i + 2 * j, 3 * j + 2),
        R_of=lambda i, j: (4 * i + 2 * j + 1, 3 * j + 2),
    )

    base = BaseCode(num_qubits=qid, connectivity_graph=G, shapes=shapes)
    base.validate_local_connectivity()
    return base, coords


__all__ = ["build_colour_square_code", "in_bounds_square"]
