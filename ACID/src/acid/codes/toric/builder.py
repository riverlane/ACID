from __future__ import annotations

import networkx as nx

from acid.base_code import BaseCode, StabiliserShape
from acid.codes.bb.algebra import GroupRing, Monomial
from acid.embedding import SquareGridEmbedding


def _path_template(n: int = 4) -> nx.Graph:
    """Return a simple path graph on n nodes labeled 0..n-1."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def build_toric_code(
    l: int, m: int, *, connectivity: str = "hex", sec_length: int = 2
) -> tuple[BaseCode, SquareGridEmbedding, list[tuple[int, int, str]]]:
    """
    Build a toric code (periodic l×m) using two-term balanced-product definitions:

      a2 = y^-1, b2 = x^-1.

    Stabiliser supports per anchor q:
      - X: (L q), (R b2 q), (L a2 q), (R q)
      - Z: (L q), (R a2 q), (L b2^-1 q), (R q)

    Connectivity edges (on device graph) from L(q) to R(...):
      - hex connectivity: ID: R(q), B2: R(b2 q), A2i: R(a2^-1 q)=R(y q)
      - grid connectivity: hex plus A2iB2: R(a2^-1 b2 q)=R(y x^-1 q)

    Returns (Code, embedding, connections) where `connections` are (u,v,class).
    """
    ring = GroupRing(l, m)
    emb = SquareGridEmbedding(ring=ring, pitch=1.0)

    # Monomials for shifts
    a2 = Monomial(0, -1, ring)  # y^-1
    a2_inv = a2.inv()  # y
    b2 = Monomial(-1, 0, ring)  # x^-1
    b2_inv = b2.inv()  # x

    # Local shape for X and Z stabilisers: 4-node path
    path4 = _path_template(4)
    shapes: list[StabiliserShape] = []
    connections: list[tuple[int, int, str]] = []

    for ax in range(l):
        for ay in range(m):
            q = Monomial(ax, ay, ring)
            # X-stabiliser mapping order (template nodes 0-1-2-3 edges):
            #   0: L(q)
            #   1: R(b2 q)
            #   2: L(a2 q)
            #   3: R(q)
            L_q = emb.qubit_id(*q.as_LR_tuple("L"))
            R_b2q = emb.qubit_id(*((b2 * q).as_LR_tuple("R")))
            L_a2q = emb.qubit_id(*((a2 * q).as_LR_tuple("L")))
            R_q = emb.qubit_id(*q.as_LR_tuple("R"))
            # Order to ensure template path edges are present in device graph
            #   0-1: R(b2 q) — L(q)
            #   1-2: L(q) — R(q)
            #   2-3: R(q) — L(a2 q)
            x_map = [R_b2q, L_q, R_q, L_a2q]
            shapes.append(
                StabiliserShape("X", path4, sec_length, x_map, f"X({ax},{ay})")
            )

            # Z-stabiliser mapping order (use a2^-1, b2^-1 for connectivity):
            #   0: R(a2^-1 q) = R(y q)
            #   1: L(q)
            #   2: R(q)
            #   3: L(b2^-1 q) = L(x q)
            R_a2invq = emb.qubit_id(*((a2_inv * q).as_LR_tuple("R")))
            L_b2invq = emb.qubit_id(*((b2_inv * q).as_LR_tuple("L")))
            z_map = [R_a2invq, L_q, R_q, L_b2invq]
            shapes.append(
                StabiliserShape("Z", path4, sec_length, z_map, f"Z({ax},{ay})")
            )

            # Connectivity edges per anchor q: from L(q) to R( ... )
            lq = L_q
            connections.append((lq, R_q, "ID"))
            connections.append((lq, R_b2q, "B2"))
            connections.append((lq, R_a2invq, "A2i"))
            if connectivity.lower() in ("grid", "grid4"):
                # Diagonal: R(a2^-1 b2 q) = R(y x^-1 q)
                R_a2inv_b2_q = emb.qubit_id(*((a2_inv * b2 * q).as_LR_tuple("R")))
                connections.append((lq, R_a2inv_b2_q, "A2iB2"))

    # Connectivity graph (undirected) ignores classes; edges are added between all pairs in connections
    G = nx.Graph()
    G.add_nodes_from(range(emb.num_qubits))
    for u, v, _cls in connections:
        G.add_edge(u, v)

    base = BaseCode(num_qubits=emb.num_qubits, connectivity_graph=G, shapes=shapes)
    base.validate_local_connectivity()
    return base, emb, connections
