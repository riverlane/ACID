from __future__ import annotations

"""Overlay emitter for device geometry and schedule‑context polygons.

Produces a Stim text prefix with qubit coords, connection sheets per class,
and optional polygons for untouched/anticommuting/product/gauge regions.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .embedding import Embedding


@dataclass(frozen=True)
class Connection:
    a: tuple[int, int, int]
    b: tuple[int, int, int]
    z: int = 0
    defective: bool = False

    def key(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return tuple(sorted((self.a, self.b)))  # undirected


@dataclass
class DeviceVisualisation:
    n_qubits: int
    embedding: Embedding  # provides qubit_id, coords, id_to_tuple
    qubit_colouring: dict[int, str]  # qubit_id -> colour
    connections: list[tuple[int, int, str]]  # qubit_1, qubit_2, connection_class
    defective_qubits: set[int]  # qubit_ids. mark these qubits as defective in the .stim file
    defective_connections: set[
        tuple[int, int, str]
    ]  # (qubit_1, qubit_2, connection_class). mark as defective in the .stim file
    connection_class_colours: dict[str, str]  # connection_class -> colour
    # Optional polygon overlays per category: list of ('X'|'Z', [qubit_ids...])
    polygons_untouched: list[tuple[str, Iterable[int]]] | None = None
    polygons_anticomm: list[tuple[str, Iterable[int]]] | None = None
    polygons_products: list[tuple[str, Iterable[int]]] | None = None
    polygons_gauge: list[tuple[str, Iterable[int]]] | None = None

    def stim_with_overlays(self, include_reset: bool = True, *, debug: bool = False) -> str:
        lines: list[str] = []
        # Dynamic legend/comment header
        lines.append("# Legend")
        lines.append("# Qubits: L (c=0) = gold, R (c=1) = mediumseagreen")
        lines.append("# Connections by class (name: colour):")
        for name, colour in self.connection_class_colours.items():
            lines.append(f"#  - {name}: {colour}")

        # Prepare polygon categories first so all SHEET statements can appear at top
        cats: list[tuple[str, list[tuple[str, Iterable[int]]]]] = [
            ("UNTX", []),
            ("UNTZ", []),
            ("ANTIX", []),
            ("ANTIZ", []),
            ("PRODX", []),
            ("PRODZ", []),
            ("GAUGEX", []),
            ("GAUGEZ", []),
        ]

        def assign_items(items: list[tuple[str, Iterable[int]]], x_name: str, z_name: str):
            if not items:
                return
            for ptype, verts in items:
                if ptype and ptype.upper() == "X":
                    for idx, (nm, arr) in enumerate(cats):
                        if nm == x_name:
                            arr.append(tuple(int(q) for q in verts))
                            cats[idx] = (nm, arr)
                            break
                elif ptype:
                    for idx, (nm, arr) in enumerate(cats):
                        if nm == z_name:
                            arr.append(tuple(int(q) for q in verts))
                            cats[idx] = (nm, arr)
                            break

        assign_items(self.polygons_untouched or [], "UNTX", "UNTZ")
        assign_items(self.polygons_anticomm or [], "ANTIX", "ANTIZ")
        assign_items(self.polygons_products or [], "PRODX", "PRODZ")
        assign_items(self.polygons_gauge or [], "GAUGEX", "GAUGEZ")

        # Embedding declaration (torus)
        lines.append(
            f"##! EMBEDDING TYPE=TORUS LX={self.embedding.width} LY={self.embedding.height}"
        )
        # Build sheet layout: QUBITS (0), connection class sheets (1..C), then polygon sheets
        sheet_defs: list[tuple[str, int]] = [("QUBITS", 0)]
        base = 1
        for i, name in enumerate(self.connection_class_colours.keys()):
            sheet_defs.append((name, base + i))
        z_next = base + len(self.connection_class_colours)
        for nm, polys in cats:
            if polys:
                sheet_defs.append((nm, z_next))
                z_next += 1
        # Emit sheet headers at the top
        if debug:
            print("[viz] sheets ->", sheet_defs)
        for nm, z in sheet_defs:
            lines.append(f"##! SHEET NAME={nm} Z={z}")

        # Qubits + coords
        for qid in range(self.n_qubits):
            q_tuple = self.embedding.id_to_tuple(qid)
            coords = self.embedding.coords(*q_tuple)
            c = q_tuple[2] if len(q_tuple) > 2 else 0
            q_colour = self.qubit_colouring.get(qid, "gold" if c == 0 else "mediumseagreen")
            attrs = [
                f"Q={qid}",
                "SHEET=QUBITS",
                f"X={coords[0]:.6g}",
                f"Y={coords[1]:.6g}",
                f"COLOUR={q_colour}",
            ]
            if qid in self.defective_qubits:
                lines.append(f"##! HIGHLIGHT TARGET=QUBIT QUBITS={qid} COLOR=red")
                attrs.append("DEFECTIVE=true")
            lines.append(f"##! QUBIT {' '.join(attrs)}")
            lines.append(f"QUBIT_COORDS({coords[0]:.6g}, {coords[1]:.6g}) {qid}")

        # Prepare optional initial tick/reset; appended after overlays if enabled
        tick_reset: list[str] = []
        if include_reset:
            tick_reset.append("TICK")
            all_ids = [str(qid) for qid in range(self.n_qubits)]
            if all_ids:
                tick_reset.append("R " + " ".join(all_ids))

        def emit_conn_set(
            sheet: str,
            conns: list[tuple[int, int, str]],
            colour: str | None = None,
            defective: bool = False,
        ):
            edges = [f"{a}-{b}" for a, b, cls in conns if cls == sheet]
            if edges:
                # For defective connections, simply force colour red without extra flags.
                local_colour = "red" if defective else colour
                # Add explicit thickness for edges (Shatter directive)
                base = f"##! CONN SET SHEET={sheet} EDGES=(" + ",".join(edges) + ") THICKNESS=2"
                if local_colour:
                    base += f" COLOUR={local_colour}"
                lines.append(base)

        # Emit connections per class
        for name, colour in self.connection_class_colours.items():
            # Good connections
            good_conns = [
                conn
                for conn in self.connections
                if conn[2] == name and conn not in self.defective_connections
            ]
            emit_conn_set(name, good_conns, colour, defective=False)
            # Defective connections
            bad_conns = [conn for conn in self.defective_connections if conn[2] == name]
            emit_conn_set(name, bad_conns, colour, defective=True)

        # Append polygons (after QUBIT and CONN lines, before TICK)
        for nm, polys in cats:
            if not polys:
                continue
            if debug:
                print(f"[viz] polygons sheet {nm}: count={len(polys)}")
            for verts in polys:
                ordered = self._order_polygon_clockwise(verts)
                if "X" in nm:
                    lines.append(f"##! POLY SHEET={nm}")
                    lines.append(
                        "#!pragma POLYGON(1,0,0,0.15)  " + " ".join(str(q) for q in ordered)
                    )
                else:
                    lines.append(f"##! POLY SHEET={nm}")
                    lines.append(
                        "#!pragma POLYGON(0,0,1,0.15)  " + " ".join(str(q) for q in ordered)
                    )

        # Append the tick/reset after overlays, if requested
        lines.extend(tick_reset)

        # Highlight defective qubits, if any
        if self.defective_qubits:
            q_list = ",".join(str(q) for q in sorted(self.defective_qubits))
            lines.append(f"##! HIGHLIGHT TARGET=QUBIT QUBITS={q_list} COLOR=red")
        return "\n".join(lines) + "\n"

    def _order_polygon_clockwise(self, qids: Iterable[int]) -> list[int]:
        """Return qubit ids ordered clockwise around their centroid.

        Uses the embedding's coordinates for geometry. If there are fewer than
        3 unique vertices, returns the input order (deduplicated).
        """
        pts: list[tuple[int, float, float]] = []
        seen: set[int] = set()
        for q in qids:
            qi = int(q)
            if qi in seen:
                continue
            seen.add(qi)
            a, b, c = self.embedding.id_to_tuple(qi)
            x, y = self.embedding.coords(a, b, c)
            pts.append((qi, float(x), float(y)))
        if len(pts) < 3:
            return [p[0] for p in pts]
        cx = sum(p[1] for p in pts) / len(pts)
        cy = sum(p[2] for p in pts) / len(pts)

        def angle(p: tuple[int, float, float]) -> float:
            return math.atan2(p[2] - cy, p[1] - cx)

        # Sort by angle descending for clockwise order; tie-break by radius
        ordered = sorted(pts, key=lambda p: (-angle(p), (p[1] - cx) ** 2 + (p[2] - cy) ** 2))
        return [p[0] for p in ordered]
