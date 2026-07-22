from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import networkx as nx

from acid.codes.bb.algebra import GroupRing, Monomial, Polynomial
from acid.codes.bb.midcycle import BBMidCycle
from acid.embedding import SquareGridEmbedding
from acid.base_code import BaseCode, StabiliserShape

# Internal lookup of known BB codes by key. The polynomials are given in
# human-readable form for documentation (A_str/B_str) and normalised monomial
# exponents (ax, ay) after dividing A by y^2 and B by x^2, which does not
# change the code but fixes a canonical representative for construction.
CODE_TABLE: dict[str, dict] = {
    "bb72": {
        "l": 6,
        "m": 6,
        "A_str": "x^3 + y + y^2",
        "B_str": "y^3 + x + x^2",
        "a2": (0, 5),
        "a3": (3, 4),
        "b2": (5, 0),
        "b3": (4, 3),
        "fx": 1,
        "fy": 1,
    },
    "bb108": {
        "l": 9,
        "m": 6,
        "A_str": "x^3 + y + y^2",
        "B_str": "y^3 + x + x^2",
        "a2": (0, 5),
        "a3": (3, 4),
        "b2": (8, 0),
        "b3": (7, 3),
        "fx": 0,
        "fy": 1,
    },
    "bb144": {
        "l": 12,
        "m": 6,
        "A_str": "x^3 + y + y^2",
        "B_str": "y^3 + x + x^2",
        "a2": (0, 5),
        "a3": (3, 4),
        "b2": (11, 0),
        "b3": (10, 3),
        "fx": 1,
        "fy": 1,
    },
    "bb288": {
        "l": 12,
        "m": 12,
        "A_str": "x^3 + y^7 + y^2",
        "B_str": "y^3 + x + x^2",
        "a2": (0, 5),
        "a3": (3, 10),
        "b2": (11, 0),
        "b3": (10, 3),
        "fx": 1,
        "fy": 1,
    },
}


def known_code_keys() -> list[str]:
    return sorted(CODE_TABLE.keys())


@dataclass
class CodeSpec:
    """Specification for a bivariate bicycle (BB) code with hex connectivity.

    Attributes:
        key: Identifier string for the code (e.g. 'bb144').
        l: Size of the cyclic group Z_l (x-direction).
        m: Size of the cyclic group Z_m (y-direction).
        a2: Exponents (ax, ay) of the second monomial in polynomial A.
        a3: Exponents (ax, ay) of the third monomial in polynomial A.
        b2: Exponents (bx, by) of the second monomial in polynomial B.
        b3: Exponents (bx, by) of the third monomial in polynomial B.
        fx: Homomorphism flag for x (0 or 1); required when l is odd.
        fy: Homomorphism flag for y (0 or 1); required when m is odd.
    """

    key: str
    l: int
    m: int
    a2: Tuple[int, int]
    a3: Tuple[int, int]
    b2: Tuple[int, int]
    b3: Tuple[int, int]
    fx: Optional[int] = None
    fy: Optional[int] = None


def _poly_from_mons(ring: GroupRing, mons: List[Tuple[int, int]]) -> Polynomial:
    S = {Monomial(ax, ay, ring) for (ax, ay) in mons}
    return Polynomial(frozenset(S), ring)


def build_code_from_spec(
    spec: CodeSpec, *, fx: Optional[int] = None, fy: Optional[int] = None
) -> Tuple[BaseCode, SquareGridEmbedding, List[Tuple[int, int, str]]]:
    """Build a bivariate bicycle (BB) code with hex connectivity from a specification.

    Args:
        spec: CodeSpec object containing the code parameters.
        fx: Optional override for the homomorphism flag for x (0 or 1).
        fy: Optional override for the homomorphism flag for y (0 or 1).

    Returns:
        A tuple containing:
            - BaseCode: The constructed base code with connectivity and stabiliser shapes.
            - SquareGridEmbedding: The embedding of the code in a square grid.
            - List[Tuple[int,int,str]]: A list of connections (edges) in the code's
            connectivity graph, each represented as (left qubit, right qubit, label).
    """
    l, m = spec.l, spec.m

    # ring which defines the group structure for the code
    ring = GroupRing(l, m)

    # buidling the monomials
    one = Monomial(0, 0, ring)
    a2 = Monomial(*spec.a2, ring)
    a3 = Monomial(*spec.a3, ring)
    b2 = Monomial(*spec.b2, ring)
    b3 = Monomial(*spec.b3, ring)

    # building the polynomials
    A = Polynomial(frozenset({one, a2, a3}), ring)
    B = Polynomial(frozenset({one, b2, b3}), ring)

    fx = 1 if (l % 2 == 0) else 0
    fy = 1 if (m % 2 == 0) else 0
    if fx == 0 and fy == 0:
        raise ValueError(
            f"No valid homomorphism for code (l={l}, m={m}) — require at least one even"
        )
    use_fx = fx if fx is not None else spec.fx
    use_fy = fy if fy is not None else spec.fy
    if use_fx is None or use_fy is None:
        raise ValueError(
            f"Must specify homomorphism fx/fy for code {spec.key}; pass to build_code_from_spec or add to spec"
        )

    # build the BB code and its embedding
    bb = BBMidCycle(
        ring, A, B, homomorphism_f_x=int(use_fx), homomorphism_f_y=int(use_fy)
    )
    embedding = SquareGridEmbedding(ring=bb.ring, pitch=1.0)

    # cyclical graph of length 6 - ie a hexagon shaped graph
    hex_graph = nx.cycle_graph(6)

    SEC_length = 3
    shapes: List[StabiliserShape] = []
    connections: List[Tuple[int, int, str]] = []

    obj_stabs = [
        set(embedding.qubit_id(*q) for q in q_s) for s, q_s in bb.stabilizers().items()
    ]

    for ax in range(l):
        for ay in range(m):
            q = Monomial(ax, ay, bb.ring)
            even_odd = "O" if bb.even_odd_monomial(q) else "E"

            # left/right qubit for this monomial (a, b) in the embedding
            l_q = embedding.qubit_id(*q.as_LR_tuple("L"))
            r_q = embedding.qubit_id(*q.as_LR_tuple("R"))

            # other qubits part of the x stabilizer support for this monomial (a, b)
            l_a2q = embedding.qubit_id(*(a2 * q).as_LR_tuple("L"))
            l_a3q = embedding.qubit_id(*(a3 * q).as_LR_tuple("L"))
            r_b2q = embedding.qubit_id(*(b2 * q).as_LR_tuple("R"))
            r_b3q = embedding.qubit_id(*(b3 * q).as_LR_tuple("R"))

            # x stabilizer support qubits (left/right) for this monomial (a, b)
            x_map = [r_q, l_a2q, r_b2q, l_q, r_b3q, l_a3q]
            assert set(x_map) in obj_stabs
            label = f"X{even_odd}({ax},{ay})"

            # x stabiliser shape for this monomial (a, b) in the embedding
            shapes.append(StabiliserShape("X", hex_graph, SEC_length, x_map, label))

            # other qubits part of the z stabilizer support for this monomial (a, b)
            r_a2invq = embedding.qubit_id(*(a2.inv() * q).as_LR_tuple("R"))
            r_a3invq = embedding.qubit_id(*(a3.inv() * q).as_LR_tuple("R"))
            l_b2invq = embedding.qubit_id(*(b2.inv() * q).as_LR_tuple("L"))
            l_b3invq = embedding.qubit_id(*(b3.inv() * q).as_LR_tuple("L"))

            # z stabilizer support qubits (left/right) for this monomial (a, b)
            z_map = [r_q, l_b2invq, r_a2invq, l_q, r_a3invq, l_b3invq]
            assert set(z_map) in obj_stabs
            label = f"Z{even_odd}({ax},{ay})"
            # z stabiliser shape for this monomial (a, b) in the embedding
            shapes.append(StabiliserShape("Z", hex_graph, SEC_length, z_map, label))

            r_a2invb2q = embedding.qubit_id(*(a2.inv() * b2 * q).as_LR_tuple("R"))
            r_a3invb3q = embedding.qubit_id(*(a3.inv() * b3 * q).as_LR_tuple("R"))

            connections.extend(
                [
                    (l_q, r_a2invq, "A2"),
                    (l_q, r_a3invq, "A3"),
                    (l_q, r_b2q, "B2"),
                    (l_q, r_b3q, "B3"),
                    (l_q, r_a2invb2q, "A2B2^-1"),
                    (l_q, r_a3invb3q, "A3B3^-1"),
                ]
            )

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


def get_spec(key: str) -> CodeSpec:
    key_l = key.lower()
    cfg = CODE_TABLE.get(key_l)
    if cfg is None:
        raise ValueError(
            f"Unknown code key: {key}. Known keys: {', '.join(known_code_keys())}"
        )
    return CodeSpec(
        key=key_l,
        l=int(cfg["l"]),
        m=int(cfg["m"]),
        a2=tuple(cfg["a2"]),
        a3=tuple(cfg["a3"]),
        b2=tuple(cfg["b2"]),
        b3=tuple(cfg["b3"]),
        fx=int(cfg["fx"]) if "fx" in cfg else None,
        fy=int(cfg["fy"]) if "fy" in cfg else None,
    )
