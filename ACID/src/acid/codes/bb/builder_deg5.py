from __future__ import annotations

import networkx as nx

from acid.base_code import BaseCode, StabiliserShape
from acid.codes.bb.algebra import GroupRing, Monomial, Polynomial
from acid.codes.bb.builder_hexconn import CodeSpec
from acid.codes.bb.midcycle import BBMidCycle
from acid.embedding import SquareGridEmbedding


def _h_template() -> nx.Graph:
    """Return the 6-node 'double-fan' H-shaped local connectivity graph.

    Nodes 0-1-2 form one path, 3-4-5 another, with a rung 1-4.
    """
    H = nx.Graph()
    H.add_nodes_from(range(6))
    H.add_edges_from([(0, 1), (1, 2), (3, 4), (4, 5), (1, 4)])
    return H


def build_code_from_spec(
    spec: CodeSpec,
) -> tuple[BaseCode, SquareGridEmbedding, list[tuple[int, int, str]]]:
    """
    Build a BaseCode with degree-5 connectivity derived directly from the BB polynomials.

    Each left-qubit l_q connects to exactly five right-qubits:
      - r_q (identity 'I')
      - r_{a2^{-1} q} ('A2')
      - r_{a3^{-1} q} ('A3')
      - r_{b2 q} ('B2')
      - r_{b3 q} ('B3')

    Local stabiliser connectivity uses a 6-node H-shaped template with SEC length 3,
    matching the arrangement in the classic example (double fan-out with a rung).
    """
    l, m = spec.l, spec.m

    ring = GroupRing(l, m)
    one = Monomial(0, 0, ring)
    a2 = Monomial(*spec.a2, ring)
    a3 = Monomial(*spec.a3, ring)
    b2 = Monomial(*spec.b2, ring)
    b3 = Monomial(*spec.b3, ring)

    A = Polynomial(frozenset({one, a2, a3}), ring)
    B = Polynomial(frozenset({one, b2, b3}), ring)

    fx = 1 if (l % 2 == 0) else 0
    fy = 1 if (m % 2 == 0) else 0
    if fx == 0 and fy == 0:
        raise ValueError(
            f"No valid homomorphism for code (l={l}, m={m}) — require at least one even"
        )
    use_fx = spec.fx
    use_fy = spec.fy
    if use_fx is None or use_fy is None:
        # If spec omitted homomorphisms, fall back to the canonical parity choice
        use_fx = fx if use_fx is None else use_fx
        use_fy = fy if use_fy is None else use_fy

    bb = BBMidCycle(
        ring, A, B, homomorphism_f_x=int(use_fx), homomorphism_f_y=int(use_fy)
    )
    embedding = SquareGridEmbedding(ring=bb.ring, pitch=1.0)

    # For sanity: object stabiliser supports for assert checks (as sets of qubit ids)
    obj_stabs = [
        set(embedding.qubit_id(*q) for q in q_s) for s, q_s in bb.stabilizers().items()
    ]

    H = _h_template()
    SEC_length = 3
    shapes: list[StabiliserShape] = []
    connections: list[tuple[int, int, str]] = []

    for ax in range(l):
        for ay in range(m):
            q = Monomial(ax, ay, bb.ring)
            even_odd = "O" if bb.even_odd_monomial(q) else "E"

            # Base L/R at q
            l_q = embedding.qubit_id(*q.as_LR_tuple("L"))
            r_q = embedding.qubit_id(*q.as_LR_tuple("R"))

            # Neighbours for X
            l_a2q = embedding.qubit_id(*(a2 * q).as_LR_tuple("L"))
            l_a3q = embedding.qubit_id(*(a3 * q).as_LR_tuple("L"))
            r_b2q = embedding.qubit_id(*(b2 * q).as_LR_tuple("R"))
            r_b3q = embedding.qubit_id(*(b3 * q).as_LR_tuple("R"))

            # Neighbours for Z
            r_a2invq = embedding.qubit_id(*(a2.inv() * q).as_LR_tuple("R"))
            r_a3invq = embedding.qubit_id(*(a3.inv() * q).as_LR_tuple("R"))
            l_b2invq = embedding.qubit_id(*(b2.inv() * q).as_LR_tuple("L"))
            l_b3invq = embedding.qubit_id(*(b3.inv() * q).as_LR_tuple("L"))

            # X stabiliser map (ordering matches H edges to actual device connections)
            x_map = [r_b2q, l_q, r_b3q, l_a2q, r_q, l_a3q]
            assert set(x_map) in obj_stabs
            label_x = f"X{even_odd}({ax},{ay})"
            shapes.append(StabiliserShape("X", H, SEC_length, x_map, label_x))

            # Z stabiliser map
            z_map = [r_a2invq, l_q, r_a3invq, l_b2invq, r_q, l_b3invq]
            assert set(z_map) in obj_stabs
            label_z = f"Z{even_odd}({ax},{ay})"
            shapes.append(StabiliserShape("Z", H, SEC_length, z_map, label_z))

            # Degree-5 global connectivity edges from l_q
            connections.extend(
                [
                    (l_q, r_q, "I"),
                    (l_q, r_a2invq, "A2"),
                    (l_q, r_a3invq, "A3"),
                    (l_q, r_b2q, "B2"),
                    (l_q, r_b3q, "B3"),
                ]
            )

    # Build global device graph
    G = nx.Graph()
    G.add_nodes_from(range(bb.num_qubits))
    for u, v, _ in connections:
        G.add_edge(u, v)

    base = BaseCode(
        num_qubits=bb.num_qubits,
        connectivity_graph=G,
        shapes=shapes,
        connection_classes=connections,
    )
    base.validate_local_connectivity()
    return base, embedding, connections
