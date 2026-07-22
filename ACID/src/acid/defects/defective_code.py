from __future__ import annotations

from typing import List, Tuple, Dict, Set, Iterable, cast
from pathlib import Path
import numpy as np


import networkx as nx

from acid.defects.quasi import QuasiProduct, QuasiStabiliser
from acid.base_code import BaseCode, StabiliserShape
from acid.defects.syndrome_extraction_circuit import SyndromeExtractionCircuit
from acid.scheduling.types import StabiliserTemplate
from acid.solver.schedule_solver import ScheduleSolver
from acid.scheduling.template_factory import TemplateFactory

from acid.device import DeviceVisualisation
from acid.embedding import Embedding
from acid.gf2_utils import (
    gf2_rank,
    gf2_nullspace,
    gf2_bidiagonalize,
    gf2_rref_colwise,
    gf2_rank_normal_numpy,
    gf2_is_in_span,
)
from acid.pauli import PauliString, CommutingPauliBasis, AntiCommutingPauliBasis


def matmul_mod2_transpose(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    # A: r x n, B: s x n => A * B^T : r x s
    r = len(A)
    s = len(B)
    n = len(A[0]) if r else 0
    C = [[0] * s for _ in range(r)]
    for i in range(r):
        Ai = A[i]
        for j in range(s):
            acc = 0
            Bj = B[j]
            for t in range(n):
                acc ^= Ai[t] & Bj[t]
            C[i][j] = acc & 1
    return C


def build_quasi_stabilisers(
    base: BaseCode,
    dropped_qubits: Set[int],
    defective_edges: Set[Tuple[int, int]] | None = None,
) -> List[QuasiStabiliser]:
    out: List[QuasiStabiliser] = []
    # Normalize defective edges as undirected (min,max)
    def_edges: Set[Tuple[int, int]] = set()
    if defective_edges:
        def_edges = {(u, v) if u <= v else (v, u) for (u, v) in defective_edges}
    for shape in base.shapes:
        G = shape.connectivity_subgraph
        keep_nodes = [i for i in G.nodes if shape.qubit_map[i] not in dropped_qubits]
        if not keep_nodes:
            # no qubits left in this stabiliser after dropout
            continue
        H = G.subgraph(keep_nodes).copy()
        # Remove edges whose mapped code-qubit pair is defective
        if def_edges:
            to_remove: List[Tuple[int, int]] = []
            for u0, v0 in H.edges():
                cu, cv = shape.qubit_map[u0], shape.qubit_map[v0]
                key = (cu, cv) if cu <= cv else (cv, cu)
                if key in def_edges:
                    to_remove.append((u0, v0))
            H.remove_edges_from(to_remove)
        for comp_idx, comp_nodes in enumerate(nx.connected_components(H)):
            support = frozenset(int(shape.qubit_map[i]) for i in comp_nodes)
            if not support:
                continue
            out.append(
                QuasiStabiliser(
                    pauli_type=shape.pauli_type,
                    support=support,
                    parent=shape,
                    component_index=comp_idx,
                    label=f"{shape.label}_c{comp_idx}",
                )
            )
    return out


def build_anticommutation_graph(quasi_stabs: List[QuasiStabiliser]) -> nx.Graph:
    """Return anti-commutation graph with label-nodes and 'quasi' attributes.

    Nodes are quasi labels (strings). Each node stores the QuasiStabiliser as
    attribute 'quasi'. Edges connect opposite types with odd-overlap support.
    Isolated nodes are pruned.
    """
    G = nx.Graph()
    for q in quasi_stabs:
        G.add_node(q.label, quasi=q)
    for i in range(len(quasi_stabs)):
        qi = quasi_stabs[i]
        for j in range(i + 1, len(quasi_stabs)):
            qj = quasi_stabs[j]
            if qi.pauli_type == qj.pauli_type:
                continue
            if (len(qi.support & qj.support) % 2) == 1:
                G.add_edge(qi.label, qj.label)
    isolates = [n for n, d in G.degree if d == 0]
    G.remove_nodes_from(isolates)
    return G


class DefectiveCode:
    """
    Centralized workflow for handling dropouts: builds quasi-stabilisers,
    anti-commutation graph, SVD-based products, and synthesizes a schedule.
    """

    def __init__(
        self,
        code: BaseCode,
        dropped_nodes: List[int] | Set[int] | Tuple[int, ...] = (),
        dropped_edges: List[Tuple[int, int]] | Tuple[Tuple[int, int], ...] = (),
        *,
        verify: bool = True,
    ) -> None:
        self.base_code = code
        self.dropped_nodes: Set[int] = set(int(q) for q in dropped_nodes)
        self.dropped_edges: List[Tuple[int, int]] = [
            (int(u), int(v)) for (u, v) in dropped_edges
        ]
        self.num_qubits = self.base_code.num_qubits - len(self.dropped_nodes)
        self.solver = None  # type: ScheduleSolver | None

        # Note: we keep all row vectors at base code width n.
        # The effective active-qubit count n_eff is only used in verification.

        # Build a connectivity graph that excludes dropped edges entirely
        self.connectivity_graph = nx.Graph()
        self.connectivity_graph.add_nodes_from(self.base_code.connectivity_graph.nodes)
        dropped_norm = {
            (min(int(u), int(v)), max(int(u), int(v))) for (u, v) in self.dropped_edges
        }
        for u, v, data in self.base_code.connectivity_graph.edges(data=True):
            a, b = (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))
            if (a, b) in dropped_norm:
                continue
            self.connectivity_graph.add_edge(
                u, v, **{k: v for k, v in data.items() if k != "defective"}
            )

        # Step 1: build quasis
        # Normalize dropped edges as undirected pairs
        def_edges_norm: Set[Tuple[int, int]] = set()
        for u, v in self.dropped_edges:
            a, b = (int(u), int(v))
            def_edges_norm.add((a, b) if a <= b else (b, a))
        self.all_quasis: List[QuasiStabiliser] = build_quasi_stabilisers(
            self.base_code, self.dropped_nodes, defective_edges=def_edges_norm
        )
        self.x_quasis = [quasi for quasi in self.all_quasis if quasi.pauli_type == "X"]
        self.z_quasis = [quasi for quasi in self.all_quasis if quasi.pauli_type == "Z"]
        # Labels for consistency across modules
        self.quasi_labels: List[str] = [q.label for q in self.all_quasis]

        # Step 2: anticomm graph (pruned, label-noded)
        self.anticomm_graph = build_anticommutation_graph(self.all_quasis)
        self.nontrivial_idx: Set[str] = set(self.anticomm_graph.nodes())
        # Convenience maps
        self.label_to_quasi: Dict[str, QuasiStabiliser] = {
            q.label: q for q in self.all_quasis
        }
        # Also create one-hot quasis for dropped qubits (kept separate from anticomm graph)
        self.dropped_qubit_quasis: Dict[str, QuasiStabiliser] = (
            self._build_dropped_quasis()
        )
        self.label_to_quasi.update(self.dropped_qubit_quasis)

        # Step 3: Derive product stabilisers and gauge operators from the
        # anticommutation graph (bipartite Z-X) via GF(2) diagonalization.
        # Build a labeled anticommutation graph using quasi labels
        self._compute_diag_from_anticomm()
        self.products: List[QuasiProduct] = self._build_products()
        # Store base stabiliser matrices HX/HZ at base width n for reuse
        _hx_supp, _hz_supp = self._stabiliser_supports()
        n_base = self.base_code.num_qubits
        # stabiliser matrices
        self.HX: List[List[int]] = self._rows_to_matrix(_hx_supp, n_base)
        self.HZ: List[List[int]] = self._rows_to_matrix(_hz_supp, n_base)
        self.gauges: List[Tuple[QuasiProduct | None, QuasiProduct | None]] = (
            self._build_gauges()
        )

        # Also store gauge matrices GX/GZ directly from gauges (includes one-hot drop gauges)
        def _xor_supports_labels(members: List[str]) -> List[int]:
            acc: Set[int] = set()
            for lab in members:
                q = self.label_to_quasi.get(lab)
                if q is None:
                    continue
                s = set(q.support)
                acc = (acc - s) | (s - acc)
            return sorted(acc)

        GX_mat: List[List[int]] = []
        GZ_mat: List[List[int]] = []
        for zg, xg in self.gauges:
            if xg is not None:
                if xg.members:
                    xs = _xor_supports_labels(xg.members)
                    if xs:
                        row = [0] * n_base
                        for q in xs:
                            if 0 <= q < n_base:
                                row[q] ^= 1
                        GX_mat.append(row)
                elif xg.label.startswith("QgX_drop_"):
                    q = int(xg.label.split("QgX_drop_")[-1])
                    if 0 <= q < n_base:
                        row = [0] * n_base
                        row[q] = 1
                        GX_mat.append(row)
            if zg is not None:
                if zg.members:
                    zs = _xor_supports_labels(zg.members)
                    if zs:
                        row = [0] * n_base
                        for q in zs:
                            if 0 <= q < n_base:
                                row[q] ^= 1
                        GZ_mat.append(row)
                elif zg.label.startswith("QgZ_drop_"):
                    q = int(zg.label.split("QgZ_drop_")[-1])
                    if 0 <= q < n_base:
                        row = [0] * n_base
                        row[q] = 1
                        GZ_mat.append(row)
        # gauge matrices (may be empty if no gauges)
        self.GX: List[List[int]] = GX_mat
        self.GZ: List[List[int]] = GZ_mat

        # Stabiliser triplets (templates/qubit maps) for all quasis
        self.triplets = self._reify_quasi_templates()

        # Step 4: compute a canonical set of logical operators (defect-aware)
        self.logical_X_rows, self.logical_Z_rows = self._compute_logical_rows()
        # Verify logicals and gauges structure and pairwise (anti)commutation (optional)
        if verify:
            self._verify_logicals_and_gauges()

    def logical_rows_2n(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Return paired logical rows in 2n format [X|Z] over GF(2).

        Rows are ordered so that i-th X row anticommutes with i-th Z row.
        """
        n = self.base_code.num_qubits

        def vec2n_from_supports(Xs: List[int], Zs: List[int]) -> List[int]:
            X = [0] * n
            Z = [0] * n
            for q in Xs:
                if 0 <= q < n:
                    X[q] ^= 1
            for q in Zs:
                if 0 <= q < n:
                    Z[q] ^= 1
            return X + Z

        Lx_2n: List[List[int]] = []
        Lz_2n: List[List[int]] = []
        for xs, zs in zip(self.logical_X_rows, self.logical_Z_rows):
            Lx_2n.append(vec2n_from_supports(xs, []))
            Lz_2n.append(vec2n_from_supports([], zs))
        return Lx_2n, Lz_2n

    def midcycle_untouched_stabilisers(self) -> CommutingPauliBasis:
        # Base-code stabilisers (un-products)
        x_supps: List[List[int]] = []
        z_supps: List[List[int]] = []
        for label in self.quasi_labels:
            if label in self.nontrivial_idx:
                continue
            stab = self.label_to_quasi[label]
            supp = list(stab.support)
            if stab.pauli_type == "X":
                x_supps.append(supp)
            else:
                z_supps.append(supp)
        return CommutingPauliBasis.from_supports(
            name="MidCycle/Base",
            priority=10,
            x_supports=x_supps,
            z_supports=z_supps,
            n=self.base_code.num_qubits,
        )

    def midcycle_product_stabilisers(self) -> CommutingPauliBasis | None:
        # Products derived from quasi SVD
        if not self.products:
            return None
        label_to_quasi: Dict[str, QuasiStabiliser] = self.label_to_quasi

        def xor_supports(members: List[str]) -> List[int]:
            acc: Set[int] = set()
            for lab in members:
                q = label_to_quasi.get(lab)
                if q is None:
                    continue
                s = set(q.support)
                acc = (acc - s) | (s - acc)
            return sorted(acc)

        x_supps: List[List[int]] = []
        z_supps: List[List[int]] = []
        for ps in self.products:
            supp = xor_supports(ps.members)
            if not supp:
                continue
            if ps.pauli_type == "X":
                x_supps.append(supp)
            else:
                z_supps.append(supp)
        if not x_supps and not z_supps:
            return None
        return CommutingPauliBasis.from_supports(
            name="Products",
            priority=0,
            x_supports=x_supps,
            z_supports=z_supps,
            n=self.base_code.num_qubits,
        )

    def _logical_pairs_mid_paulis(self) -> Tuple[List[PauliString], List[PauliString]]:
        n = self.base_code.num_qubits
        Lx = [PauliString.from_supports(xs, [], n) for xs in self.logical_X_rows]
        Lz = [PauliString.from_supports([], zs, n) for zs in self.logical_Z_rows]
        return Lx, Lz

    def _gauge_pairs_mid_paulis(self) -> Tuple[List[PauliString], List[PauliString]]:
        n = self.base_code.num_qubits
        # Build directly from stored gauge matrices (includes drop one-hots)
        Gx_ps = [
            PauliString.from_supports([i for i, b in enumerate(row) if b & 1], [], n)
            for row in self.GX
        ]
        Gz_ps = [
            PauliString.from_supports([], [i for i, b in enumerate(row) if b & 1], n)
            for row in self.GZ
        ]
        return Gx_ps, Gz_ps

    def logical_pairs_mid(self) -> AntiCommutingPauliBasis:
        Lx, Lz = self._logical_pairs_mid_paulis()
        return AntiCommutingPauliBasis(name="Lmid", X_rows=Lx, Z_rows=Lz)

    def _reify_quasi_templates(self) -> List[Tuple[StabiliserTemplate, List[int], str]]:
        tf = TemplateFactory()
        triplets: List[Tuple[StabiliserTemplate, List[int], str]] = []
        for i, q in enumerate(self.all_quasis):
            shape: StabiliserShape = q.parent  # type: ignore[assignment]
            G = shape.connectivity_subgraph
            kept_old = [j for j in G.nodes if shape.qubit_map[j] in q.support]
            # Subgraph restricted to kept nodes, excluding any edges that correspond
            # to dropped connections in the defective connectivity graph.
            H0 = G.subgraph(kept_old).copy()
            H = nx.Graph()
            H.add_nodes_from(H0.nodes())
            for u0, v0 in H0.edges():
                cu, cv = shape.qubit_map[u0], shape.qubit_map[v0]
                if self.connectivity_graph.has_edge(cu, cv):
                    H.add_edge(u0, v0)
            old_to_new = {old: new for new, old in enumerate(sorted(H.nodes()))}
            relabelled = nx.Graph()
            relabelled.add_nodes_from(range(len(old_to_new)))
            for u0, v0 in H.edges():
                relabelled.add_edge(old_to_new[u0], old_to_new[v0])
            SEC_len = int(shape.sec_cycle_length)
            # Always pass through preferences/hints (even if damaged); they will only mark schedules preferred when applicable.
            new_tmpl = tf.get_or_create(
                q.pauli_type,
                len(old_to_new),
                relabelled,
                SEC_len,
                name=f"Q_{i}",
                preferred_roots=shape.preferred_roots,
                preferred_edges=shape.preferred_edges,
                schedule_hint=shape.schedule_hint,
                layer_hint=shape.layer_hint,
            )
            qubit_map = [
                shape.qubit_map[old]
                for old in sorted(old_to_new.keys(), key=lambda x: old_to_new[x])
            ]
            triplets.append((new_tmpl, qubit_map, q.label))
        return triplets

    def _build_products(self) -> List[QuasiProduct]:
        products: List[QuasiProduct] = []
        # Z products are rows i >= r (over Z-side basis self.z_anti_qs)
        for i in range(self.r, len(self.U)):
            coeffs = self.U[i]
            members = [
                self.z_anti_qs[k].label for k, bit in enumerate(coeffs) if bit & 1
            ]
            if members:
                products.append(
                    QuasiProduct(label=f"QpZ_{i}", pauli_type="Z", members=members)
                )
        # X products are rows i >= r in VT (over X-side basis self.x_anti_qs)
        for i in range(self.r, len(self.VT)):
            coeffs = self.VT[i]
            members = [
                self.x_anti_qs[k].label for k, bit in enumerate(coeffs) if bit & 1
            ]
            if members:
                products.append(
                    QuasiProduct(label=f"QpX_{i}", pauli_type="X", members=members)
                )
        return products

    def _build_dropped_quasis(self) -> Dict[str, QuasiStabiliser]:
        """Create one-hot quasi stabilisers for each dropped qubit (X and Z type).

        These are not included in the anticomm graph or scheduling, but allow
        treating dropped-qubit gauges uniformly as products over quasi labels.
        """
        out: Dict[str, QuasiStabiliser] = {}
        import networkx as nx

        n = self.base_code.num_qubits
        if not self.dropped_nodes:
            return out
        G1 = nx.Graph()
        G1.add_nodes_from([0])
        for qq in sorted(int(q) for q in self.dropped_nodes):
            if not (0 <= qq < n):
                continue
            # X one-hot
            shx = StabiliserShape("X", G1, 1, [qq], f"DropX({qq})")
            qx = QuasiStabiliser(
                pauli_type="X",
                support=frozenset({qq}),
                parent=shx,
                component_index=0,
                label=shx.label,
            )
            out[qx.label] = qx
            # Z one-hot
            shz = StabiliserShape("Z", G1, 1, [qq], f"DropZ({qq})")
            qz = QuasiStabiliser(
                pauli_type="Z",
                support=frozenset({qq}),
                parent=shz,
                component_index=0,
                label=shz.label,
            )
            out[qz.label] = qz
        return out

    def _build_gauges(self) -> List[Tuple[QuasiProduct | None, QuasiProduct | None]]:
        gauges: List[Tuple[QuasiProduct | None, QuasiProduct | None]] = []
        # Z gauges are rows i < r from U (Z space)
        for i in range(self.r):
            coeffs = self.U[i]
            members = [
                self.z_anti_qs[k].label for k, bit in enumerate(coeffs) if bit & 1
            ]
            if members:
                gauges.append(
                    (
                        QuasiProduct(label=f"QgZ_{i}", pauli_type="Z", members=members),
                        None,
                    )
                )
        # X gauges are rows i < r from VT (X space)
        for i in range(self.r):
            coeffs = self.VT[i]
            members = [
                self.x_anti_qs[k].label for k, bit in enumerate(coeffs) if bit & 1
            ]
            if members:
                if i < len(gauges) and gauges[i][1] is None:
                    gauges[i] = (
                        gauges[i][0],
                        QuasiProduct(label=f"QgX_{i}", pauli_type="X", members=members),
                    )
                else:
                    gauges.append(
                        (
                            None,
                            QuasiProduct(
                                label=f"QgX_{i}", pauli_type="X", members=members
                            ),
                        )
                    )
        # Add one-hot gauge pairs for dropped qubits if not in span of HX/HZ
        if self.dropped_nodes:
            n = self.base_code.num_qubits
            for q in sorted(int(q) for q in self.dropped_nodes):
                if 0 <= q < n:
                    ex = [0] * n
                    ex[q] = 1
                    ez = [0] * n
                    ez[q] = 1
                    if not gf2_is_in_span(ex, self.HX) and not gf2_is_in_span(
                        ez, self.HZ
                    ):
                        # Add as proper member-labelled gauge qubit pair
                        zg = QuasiProduct(
                            label=f"QgZ_drop_{q}",
                            pauli_type="Z",
                            members=[f"DropZ({q})"],
                        )
                        xg = QuasiProduct(
                            label=f"QgX_drop_{q}",
                            pauli_type="X",
                            members=[f"DropX({q})"],
                        )
                        gauges.append((zg, xg))
        return gauges

    def _compute_diag_from_anticomm(self) -> None:
        # Build bipartite sets and adjacency matrix A (Z rows, X cols)
        nodes = list(nx.get_node_attributes(self.anticomm_graph, "quasi").values())
        self.x_anti_qs = [q for q in nodes if q.pauli_type == "X"]
        self.z_anti_qs = [q for q in nodes if q.pauli_type == "Z"]
        ZN = len(self.z_anti_qs)
        XN = len(self.x_anti_qs)
        x_lookup = {q_x.label: i for i, q_x in enumerate(self.x_anti_qs)}
        self.A: List[List[int]] = [[0] * XN for _ in range(ZN)]
        for i, zl in enumerate(self.z_anti_qs):
            for nbr_label in self.anticomm_graph.neighbors(zl.label):
                j = x_lookup[nbr_label]
                self.A[i][j] ^= 1
        self.U, V, self.r = gf2_bidiagonalize(self.A) if (ZN and XN) else ([], [], 0)
        # X coefficients as rows of V^T
        self.VT = [list(row) for row in zip(*V)] if V else []

    def _stabiliser_supports(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Return (X_rows, Z_rows) stabiliser supports (as lists of qubit ids).

        Uses isolate quasi-stabilisers (post-dropout components that do not
        participate in the anti-commutation graph) and product stabilisers
        inferred from the anticommutation graph diagonalisation. This ensures
        supports reflect post-dropout connectivity (e.g., boundary 4->3).
        """
        x_rows: List[List[int]] = []
        z_rows: List[List[int]] = []

        # Add isolate quasis (labels not present in the anticomm graph after pruning)
        label_to_quasi: Dict[str, QuasiStabiliser] = {
            q.label: q for q in self.all_quasis
        }
        for label in self.quasi_labels:
            if label in self.nontrivial_idx:
                continue
            q = label_to_quasi.get(label)
            if q is None:
                continue
            supp = sorted(list(q.support))
            if not supp:
                continue
            if q.pauli_type == "X":
                x_rows.append(supp)
            else:
                z_rows.append(supp)

        # Add product stabilisers derived from quasis
        def xor_supports_labels(members: List[str]) -> List[int]:
            acc: Set[int] = set()
            for lab in members:
                q = label_to_quasi.get(lab)
                if q is None:
                    continue
                s = set(q.support)
                acc = (acc - s) | (s - acc)
            return sorted(acc)

        for ps in self.products:
            supp = xor_supports_labels(ps.members)
            if not supp:
                continue
            if ps.pauli_type == "X":
                x_rows.append(supp)
            else:
                z_rows.append(supp)

        # Reduce to independent sets to avoid dependent SX/SZ
        # SX is [HX; GX] and SZ is [HZ; GZ]; we want to avoid dependent rows in either.
        def reduce_independent(rows: List[List[int]], n: int) -> List[List[int]]:
            M: List[List[int]] = []
            keep: List[List[int]] = []
            r = 0
            for supp in rows:
                vec = [0] * n
                for q in supp:
                    if 0 <= int(q) < n:
                        vec[int(q)] ^= 1
                if gf2_rank(M + [vec]) > r:
                    M.append(vec)
                    keep.append(supp)
                    r += 1
            return keep

        n = self.base_code.num_qubits
        x_rows = reduce_independent(x_rows, n)
        z_rows = reduce_independent(z_rows, n)
        return x_rows, z_rows

    def _rows_to_matrix(self, rows: List[List[int]], n: int) -> List[List[int]]:
        M: List[List[int]] = []
        for supp in rows:
            vec = [0] * n
            for q in supp:
                iq = int(q)
                if 0 <= iq < n:
                    vec[iq] ^= 1
            M.append(vec)
        return M

    def _compute_logical_rows(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Explicit construction of logical operators per the stated recipe.

        1) Build SZ = [HZ; GZ] and SX = [HX; GX] at base width n, assert full row rank.
        2) RREF SX, SZ; form nullspaces NX = Null(SX), NZ = Null(SZ).
        3) CX = [HX; NZ], CZ = [HZ; NX]; RREF both; the first m rows are independent; the
           remaining k = n_eff - 2*rank(HX) - rank(GX) non-zero rows are logicals (Z from CX, X from CZ).
        4) Pair the k X/Z logicals via bidiagonalization.
        """
        n = self.base_code.num_qubits

        # Base stabilisers (including products inferred earlier)
        HX = [row[:] for row in self.HX]
        HZ = [row[:] for row in self.HZ]

        # Gauges: use stored GX/GZ matrices
        GX = [row[:] for row in self.GX]
        GZ = [row[:] for row in self.GZ]

        # Step 1: SX, SZ and rank checks
        SX = HX + GX
        SZ = HZ + GZ
        rank_SX = gf2_rank(SX)
        rank_SZ = gf2_rank(SZ)
        if rank_SX != len(SX):
            raise AssertionError(
                f"SX has dependent rows (rank={rank_SX}, rows={len(SX)})"
            )
        if rank_SZ != len(SZ):
            raise AssertionError(
                f"SZ has dependent rows (rank={rank_SZ}, rows={len(SZ)})"
            )

        # Step 2: RREF (for structure) and nullspaces (no column permutations)
        NZ = gf2_nullspace(SZ)  # X-candidates that commute with SZ
        NX = gf2_nullspace(SX)  # Z-candidates that commute with SX

        # Step 3: Stack, RREF, and extract logical rows beyond first m
        CX = HX + NZ
        CZ = HZ + NX
        CX_rref, _ = cast(
            Tuple[List[List[int]], List[int]],
            gf2_rref_colwise(CX, clear_upper_triangle=False),
        )
        CZ_rref, _ = cast(
            Tuple[List[List[int]], List[int]],
            gf2_rref_colwise(CZ, clear_upper_triangle=False),
        )

        # Compute expected k and verify number of non-zero tail rows
        rank_HX = gf2_rank(HX)
        rank_HZ = gf2_rank(HZ)
        rank_GX = gf2_rank(GX)
        # With dropped qubits added as one-hot gauges in _gauge_pairs_mid_paulis, we use n
        k_expected = max(0, n - rank_HX - rank_HZ - rank_GX)

        # Assert: first m rows are non-zero (independent) in both
        for i in range(rank_HX):
            if not any(CX_rref[i]):
                raise AssertionError("First m rows of CX_rref not independent")
        for i in range(rank_HZ):
            if not any(CZ_rref[i]):
                raise AssertionError("First m rows of CZ_rref not independent")

        def tail_non_zero_rows(R: List[List[int]], start: int) -> List[List[int]]:
            return [row for row in R[start:] if any(row)]

        tail_CX = tail_non_zero_rows(CX_rref, rank_HX)  # these define Z logicals
        tail_CZ = tail_non_zero_rows(CZ_rref, rank_HZ)  # these define X logicals
        if len(tail_CX) != k_expected or len(tail_CZ) != k_expected:
            raise AssertionError(
                f"Unexpected logical count: CX_tail={len(tail_CX)} CZ_tail={len(tail_CZ)} expected={k_expected}"
            )

        Z_rows = [row[:] for row in tail_CZ[:k_expected]]
        X_rows = [row[:] for row in tail_CX[:k_expected]]

        # Step 4: Pair X/Z logicals (make them anti-commute in matched pairs)
        C = matmul_mod2_transpose(X_rows, Z_rows)
        _, U, _, V_T, _, r = gf2_rank_normal_numpy(np.array(C))

        # U, V, r = gf2_bidiagonalize(C)
        # V = np.array(V, dtype=np.int8).tolist()

        def apply_transform(
            T: List[List[int]], Rows: List[List[int]]
        ) -> List[List[int]]:
            """Apply a GF(2) transformation T to a list of row vectors Rows,
            returning the transformed rows.
            """
            if not Rows:
                return []
            out: List[List[int]] = []
            for coeffs in T:
                vec = [0] * len(Rows[0])
                for idx, bit in enumerate(coeffs):
                    if bit & 1:
                        vec = [a ^ b for a, b in zip(vec, Rows[idx])]
                out.append(vec)
            return out

        U_list: List[List[int]] = np.asarray(U, dtype=np.int8).tolist()
        VtT_list: List[List[int]] = np.asarray(V_T.T, dtype=np.int8).tolist()
        Xp = apply_transform(U_list, X_rows)
        Zp = apply_transform(VtT_list, Z_rows)

        def vec_to_support(v: List[int]) -> List[int]:
            return [i for i, b in enumerate(v) if b & 1]

        X_supports = [vec_to_support(Xp[i]) for i in range(min(k_expected, len(Xp)))]
        Z_supports = [vec_to_support(Zp[i]) for i in range(min(k_expected, len(Zp)))]
        return X_supports, Z_supports

    def _verify_logicals_and_gauges(self) -> None:
        """Verify Pauli type and pairwise (anti)commutation of logicals and gauges.

        - Pauli type: X-rows have only X support; Z-rows have only Z support.
        - Pairing: X_i anticommutes only with Z_i and commutes with all others.
        - Uses a single stacked product C = [Lx; Gx] * [Lz; Gz]^T over GF(2).
        """
        n = self.base_code.num_qubits
        n - len(self.dropped_nodes)

        # Build Pauli representations
        Lx_ps, Lz_ps = self._logical_pairs_mid_paulis()
        Gx_ps, Gz_ps = self._gauge_pairs_mid_paulis()

        # Pauli-type checks: X rows must have no Z bits; Z rows must have no X bits
        for i, p in enumerate(Lx_ps):
            if any(b & 1 for b in p.Z):
                raise AssertionError(f"Logical X[{i}] has non-zero Z part")
        for i, p in enumerate(Lz_ps):
            if any(b & 1 for b in p.X):
                raise AssertionError(f"Logical Z[{i}] has non-zero X part")
        for i, p in enumerate(Gx_ps):
            if any(b & 1 for b in p.Z):
                raise AssertionError(f"Gauge X[{i}] has non-zero Z part")
        for i, p in enumerate(Gz_ps):
            if any(b & 1 for b in p.X):
                raise AssertionError(f"Gauge Z[{i}] has non-zero X part")

        # Convert PauliStrings to row matrices over base n
        def rows_from_X(ps: List[PauliString]) -> List[List[int]]:
            return [row.X[:] for row in ps]

        def rows_from_Z(ps: List[PauliString]) -> List[List[int]]:
            return [row.Z[:] for row in ps]

        Lx = rows_from_X(Lx_ps)
        Lz = rows_from_Z(Lz_ps)
        Gx = rows_from_X(Gx_ps)
        Gz = rows_from_Z(Gz_ps)

        # Stabiliser supports (X and Z) from the code (including products)
        stab_x_supp, stab_z_supp = self._stabiliser_supports()
        Sx = self._rows_to_matrix(stab_x_supp, n)
        Sz = self._rows_to_matrix(stab_z_supp, n)

        # Pair counts (must match within each family)
        kx, kz = len(Lx), len(Lz)
        gx, gz = len(Gx), len(Gz)
        sx, sz = len(Sx), len(Sz)
        # Do not assert direct-sum here; we check counts against ranks below
        if kx != kz:
            raise AssertionError(f"Mismatched logical pairs: X={kx}, Z={kz}")
        if gx != gz:
            raise AssertionError(f"Mismatched gauge pairs: X={gx}, Z={gz}")

        # Stacked matrices and single anticomm matrix
        AX = Lx + Gx + Sx
        BZ = Lz + Gz + Sz
        if AX and (len(AX[0]) != n):
            raise AssertionError("X-matrix width mismatch with code size")
        if BZ and (len(BZ[0]) != n):
            raise AssertionError("Z-matrix width mismatch with code size")
        C = matmul_mod2_transpose(AX, BZ)

        # Expected block-diagonal identity: diag(I_k, I_g)
        k = kx
        g = gx

        # Helper to check identity and zero blocks
        def check_block_is_identity(
            mat: List[List[int]], r0: int, c0: int, sz: int, label: str
        ) -> None:
            for i in range(sz):
                for j in range(sz):
                    exp = 1 if i == j else 0
                    if mat[r0 + i][c0 + j] != exp:
                        raise AssertionError(f"{label} block not identity at ({i},{j})")

        def check_block_is_zero(
            mat: List[List[int]], r0: int, c0: int, rsz: int, csz: int, label: str
        ) -> None:
            for i in range(rsz):
                for j in range(csz):
                    if mat[r0 + i][c0 + j] != 0:
                        raise AssertionError(f"{label} block not zero at ({i},{j})")

        # Expect exactly k = n - rank(Sx_full) - rank(Sz_full) logical pairs (using base n)
        k_expected = max(0, n - gf2_rank(Sx) - gf2_rank(Sz) - gf2_rank(Gx))
        if kx != k_expected or kz != k_expected:
            raise AssertionError(
                f"Unexpected logical count: kx={kx}, kz={kz}, expected={k_expected}"
            )

        # Top-left: logicals vs logicals
        check_block_is_identity(C, 0, 0, k, "Logical")
        # Top-right: logical X vs gauge Z must commute
        check_block_is_zero(C, 0, k, k, g, "Logical/Gauge cross")
        # Bottom-left: gauge X vs logical Z must commute
        check_block_is_zero(C, k, 0, g, k, "Gauge/Logical cross")
        # Bottom-right: gauges vs gauges
        check_block_is_identity(C, k, k, g, "Gauge")
        # Cross with stabilisers: all must commute
        # Logical X vs Z-stabilisers
        check_block_is_zero(C, 0, k + g, k, sz, "Logical/Stabiliser-Z cross")
        # Gauge X vs Z-stabilisers
        check_block_is_zero(C, k, k + g, g, sz, "Gauge/Stabiliser-Z cross")
        # X-stabilisers vs Logical Z
        check_block_is_zero(C, k + g, 0, sx, k, "Stabiliser-X/Logical cross")
        # X-stabilisers vs Gauge Z
        check_block_is_zero(C, k + g, k, sx, g, "Stabiliser-X/Gauge cross")

    # Public API
    def stats(self) -> Dict:
        rank = self.r
        num_qpz = sum(1 for p in self.products if p.pauli_type == "Z")
        num_qpx = sum(1 for p in self.products if p.pauli_type == "X")
        # Anticommutation graph stats
        try:
            num_anticomm_nodes = len(list(self.anticomm_graph.nodes()))
            num_anticomm_edges = int(self.anticomm_graph.number_of_edges())
        except Exception:
            num_anticomm_nodes = len(self.nontrivial_idx)
            num_anticomm_edges = 0
        # Gauge counts (including drop gauges)
        gauges = getattr(self, "gauges", []) or []
        num_gauge_pairs_total = len(gauges)

        def _is_drop_g(lab: str | None) -> bool:
            return bool(lab) and (
                "_drop_" in lab
                or lab.startswith("QgX_drop_")
                or lab.startswith("QgZ_drop_")
            )

        num_drop_gauge_pairs = 0
        for pair in gauges:
            try:
                zg, xg = pair
            except Exception:
                zg, xg = None, None
            if (zg is not None and _is_drop_g(getattr(zg, "label", None))) or (
                xg is not None and _is_drop_g(getattr(xg, "label", None))
            ):
                num_drop_gauge_pairs += 1
        # Quasis that changed support vs parent stabiliser (ignore one-hot DropX/DropZ)
        num_quasi_changed_supports = 0
        for q in self.all_quasis:
            lab = getattr(q, "label", "") or ""
            if lab.startswith("DropX(") or lab.startswith("DropZ("):
                continue
            try:
                parent_supp = set(int(x) for x in q.parent.qubit_map)  # type: ignore[attr-defined]
            except Exception:
                parent_supp = set()
            if set(int(x) for x in q.support) != parent_supp:
                num_quasi_changed_supports += 1
        return {
            "num_quasi": len(self.all_quasis),
            "num_nontrivial": len(self.nontrivial_idx),
            "rank": rank,
            "num_QpZ": num_qpz,
            "num_QpX": num_qpx,
            # Extended reporting
            "num_anticomm_nodes": num_anticomm_nodes,
            "num_anticomm_edges": num_anticomm_edges,
            "num_gauge_pairs_total": num_gauge_pairs_total,
            "num_drop_gauge_pairs": num_drop_gauge_pairs,
            "num_quasi_changed_supports": num_quasi_changed_supports,
        }

    # Convenience helpers for downstream tools (read-only)
    def quasi_support(self, label: str) -> Tuple[str, List[int]]:
        """
        Return (pauli_type, sorted list of code-qubit ids) for a quasi or stabiliser label.

        Labels refer to entries from self.all_quasis (post-dropout components) and
        are the same labels that appear in schedules/layers.
        """
        q = self.label_to_quasi.get(label)
        if q is None:
            # allow falling back to original stabiliser labels (untouched)
            for stab in getattr(self.base_code, "stabilisers", {}).values():
                if stab.label == label:
                    return (stab.pauli_type, sorted(stab.qubit_map))
            raise KeyError(f"Unknown quasi label: {label}")
        return (q.pauli_type, sorted(list(q.support)))

    def anticommutation_graph(self) -> nx.Graph:
        return self.anticomm_graph.copy()

    def products_list(self) -> List[QuasiProduct]:
        return list(self.products)

    def prepare_solver(self, *, prune_params: dict | None = None) -> None:
        """Prepare an internal ScheduleSolver instance for repeated scheduling calls.

        If prune_params is provided, apply schedule pruning immediately after building the scheduling graph.
        """
        self.solver = ScheduleSolver(
            self.connectivity_graph,
            self.triplets,
            anticommutation_graph=self.anticomm_graph,
            product_stabilisers=self.products,
        )
        if prune_params is not None:
            if not isinstance(prune_params, dict):
                raise ValueError("prune_params must be a dict or None")
            try:
                M = int(prune_params["M"])  # required
                verbose = bool(prune_params.get("verbose", False))
            except Exception as e:
                raise ValueError(f"Invalid prune parameters: {e}")
            self.solver.apply_pruning(M=M, verbose=verbose)

    def schedule(self, L: int, solve_time: float = 60.0) -> SyndromeExtractionCircuit:
        """
        Build a layered schedule with product constraints using exactly L layers.

        Fails with a clear error if infeasible at the requested L.
        """
        if self.solver is None:
            self.prepare_solver()
        assert self.solver is not None
        try:
            layers = self.solver.create_layers_with_products(
                num_layers=L, solve_time=solve_time
            )
            return SyndromeExtractionCircuit(
                layers=layers, solve_time=solve_time, L=L, dcode=self
            )
        except Exception as e:
            from acid.solver.schedule_solver import (
                SchedulingInfeasibleError,
                SchedulingTimeLimitError,
                SchedulingModelInvalidError,
            )

            if isinstance(e, SchedulingInfeasibleError):
                raise ValueError(
                    f"Scheduling infeasible at L={L} with product constraints. "
                    f"Try increasing --layers (e.g., L>=4 for BB) or adjusting constraints.\n{e}"
                )
            if isinstance(e, SchedulingTimeLimitError):
                raise ValueError(
                    f"Scheduling hit time limit at L={L} without a feasible solution. "
                    f"Try increasing --solve_time.\n{e}"
                )
            if isinstance(e, SchedulingModelInvalidError):
                raise ValueError(f"Scheduling model invalid.\n{e}")
            raise

    def visualisation_stim(
        self,
        embedding: Embedding,
        *,
        colour_map: Dict[str, str] | None = None,
        include_reset: bool = False,
        debug: bool = False,
    ) -> str:
        """Return a .stim overlay for the device with polygons for quasis/products/gauges.

        include_reset: if True, include an initial TICK and a full-qubit R line.
        """
        # Build connections list, prefer labelled classes if available from base_code
        conns: List[Tuple[int, int, str]] = []
        bad_conns: Set[Tuple[int, int, str]] = set()
        if getattr(self.base_code, "connection_classes", None):
            # Use provided classes
            for u, v, cls in self.base_code.connection_classes or []:
                conns.append((int(u), int(v), str(cls)))
            # Mark defective edges across all classes sharing the same undirected pair
            dropped_norm = {
                (min(int(u), int(v)), max(int(u), int(v)))
                for (u, v) in self.dropped_edges
            }
            for u, v, cls in self.base_code.connection_classes or []:
                a, b = (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))
                if (a, b) in dropped_norm:
                    bad_conns.add((a, b, str(cls)))
            # Default colour map per class if not provided
            if colour_map is None:
                # Assign a few distinct colours; fall back to a default palette
                palette = [
                    "#4361ee",
                    "#2a9d8f",
                    "#e76f51",
                    "#f4a261",
                    "#e9c46a",
                    "#8a5cff",
                ]
                classes = sorted({cls for _, _, cls in conns})
                colours = {
                    cls: palette[i % len(palette)] for i, cls in enumerate(classes)
                }
            else:
                colours = colour_map
        else:
            # Fallback: single class 'E'
            for u, v in self.connectivity_graph.edges():
                conns.append((int(u), int(v), "E"))
            dropped_norm = {
                (min(int(u), int(v)), max(int(u), int(v)))
                for (u, v) in self.dropped_edges
            }
            for u, v in dropped_norm:
                bad_conns.add((u, v, "E"))
            colours = colour_map if colour_map is not None else {"E": "#4361ee"}

        if debug:
            print(
                "[viz] qubits=",
                self.base_code.num_qubits,
                "dropped_nodes=",
                len(self.dropped_nodes),
                "dropped_edges=",
                len(self.dropped_edges),
            )
        # Untouched stabilisers correspond to isolated quasis; use their post-dropout supports.
        if debug:
            print("[viz] base_code stabilisers:", len(list(self.base_code.shapes)))
        label_to_quasi: Dict[str, QuasiStabiliser] = {
            q.label: q for q in self.all_quasis
        }
        untouched: List[Tuple[str, Iterable[int]]] = []
        for label in self.quasi_labels:
            if label in self.nontrivial_idx:
                continue
            q = label_to_quasi.get(label)
            if q is None:
                continue
            untouched.append((q.pauli_type, sorted(list(q.support))))
        if debug:
            print("[viz] untouched stabilisers:", len(untouched))

        # Anticommuting quasi-stabilisers (nodes in anticomm graph)
        # Map label -> quasi info
        anticomm_items: List[Tuple[str, Iterable[int]]] = []
        for q in nx.get_node_attributes(self.anticomm_graph, "quasi").values():
            anticomm_items.append((q.pauli_type, sorted(q.support)))
        if debug:
            print(
                "[viz] anticomm quasis (nodes):",
                len(list(self.anticomm_graph.nodes())),
                "rendered:",
                len(anticomm_items),
            )

        # Product stabilisers: XOR supports of member quasis
        product_items: List[Tuple[str, Iterable[int]]] = []
        if debug:
            print("[viz] product specs:", len(self.products))
        for ps in self.products:
            acc: Set[int] = set()
            for m in ps.members:
                q = label_to_quasi.get(m)
                if q is None:
                    continue
                acc = acc ^ set(q.support)
            if acc:
                product_items.append((ps.pauli_type, sorted(acc)))
        if debug:
            print("[viz] product polygons:", len(product_items))

        # Gauge items: combinations for the first r rows/cols in the diagonalization
        gauge_items: List[Tuple[str, Iterable[int]]] = []

        if debug:
            print("[viz] gauge rank:", self.r)

        # helper to xor supports of member labels
        def xor_supports(members: List[QuasiStabiliser]) -> List[int]:
            acc: Set[int] = set()
            for q in members:
                s = set(q.support)
                acc = (acc - s) | (s - acc)  # symmetric difference
            return sorted(acc)

        # Z gauge
        for i in range(min(self.r, len(self.U))):
            members = [self.z_anti_qs[k] for k, bit in enumerate(self.U[i]) if bit & 1]
            supp = xor_supports(members)
            if supp:
                gauge_items.append(("Z", supp))
        # X gauge
        for i in range(min(self.r, len(self.VT))):
            members = [self.x_anti_qs[k] for k, bit in enumerate(self.VT[i]) if bit & 1]
            supp = xor_supports(members)
            if supp:
                gauge_items.append(("X", supp))
        if debug:
            print("[viz] gauge polygons:", len(gauge_items))

        dev = DeviceVisualisation(
            n_qubits=self.base_code.num_qubits,
            embedding=embedding,
            qubit_colouring={},
            connections=conns,
            defective_qubits=set(self.dropped_nodes),
            defective_connections=bad_conns,
            connection_class_colours=colours,
            polygons_untouched=untouched,
            polygons_anticomm=anticomm_items,
            polygons_products=product_items,
            polygons_gauge=gauge_items,
        )
        stim = dev.stim_with_overlays(include_reset=include_reset)
        # Adjust embedding header: keep TORUS for periodic embeddings, use PLANE for others
        lines = stim.rstrip().splitlines()
        try:
            from acid.embedding import SquareGridEmbedding

            is_torus = isinstance(embedding, SquareGridEmbedding)
        except Exception:
            is_torus = False
        if not is_torus:
            for i, ln in enumerate(lines):
                if ln.startswith("##! EMBEDDING TYPE=TORUS "):
                    lines[i] = (
                        f"##! EMBEDDING TYPE=PLANE LX={embedding.width} LY={embedding.height}"
                    )
                    break
        return "\n".join(lines) + "\n"

    def write_visualisation(
        self,
        out_path,
        embedding: Embedding,
        *,
        colour_map: Dict[str, str] | None = None,
        include_reset: bool = True,
    ) -> None:
        """
        Write a .stim visualisation of the current code highlighting dropped
        qubits and edges (wrapper over visualisation_stim).
        """
        text = self.visualisation_stim(
            embedding, colour_map=colour_map, include_reset=include_reset
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
