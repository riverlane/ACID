from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from acid.defects.defective_code import DefectiveCode
from acid.pauli import AntiCommutingPauliBasis, CommutingPauliBasis, PauliString
from acid.scheduling.types import SyndromeExtractionLayer


from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import json

from acid.analysis.schedule import analyze_layers


@dataclass
class SyndromeExtractionCircuit:
    layers: List[SyndromeExtractionLayer]
    solve_time: float
    L: int
    dcode: "DefectiveCode"

    def to_memory_stim(self, cycles: int) -> str:
        # Generalized schedule emitter for arbitrary number of layers
        if not self.layers:
            return ""
        lines: List[str] = []
        n_qubits = self.dcode.base_code.num_qubits

        # Reset: RX on X-roots of layer 0, RZ everywhere else
          # reuse helpers
        init_layer = self.layers[0]
        A_x_roots, _ = init_layer.roots_by_basis()
        rx_set = set(A_x_roots)
        rz_ids = [str(q) for q in range(n_qubits) if q not in rx_set]
        rx_ids = [str(q) for q in sorted(rx_set)]
        if rz_ids:
            lines.append('RZ ' + ' '.join(rz_ids))
        if rx_ids:
            lines.append('RX ' + ' '.join(rx_ids))
        lines.append('TICK')

        # Initial expand of layer 0
        lines.extend(init_layer.emit_layer_expand())

        # Cycles: contract/measure/expand each layer in order
        for cycle_i in range(cycles):
            for k, Lk in enumerate(self.layers):
                if cycle_i == 0 and k == 0:
                    continue  # already did initial expand of layer 0
                lines.extend(Lk.emit_layer_contract())
                Lk_x_roots, Lk_z_roots = Lk.roots_by_basis()
                if Lk_x_roots:
                    lines.append('MX ' + ' '.join(str(q) for q in Lk_x_roots))
                if Lk_z_roots:
                    lines.append('MZ ' + ' '.join(str(q) for q in Lk_z_roots))
                lines.append('TICK')
                lines.extend(Lk.emit_layer_expand())

        # Final contract of layer 0 and measure all Z
        lines.extend(init_layer.emit_layer_contract())
        lines.append('MZ ' + ' '.join(str(q) for q in range(n_qubits)))
        return '\n'.join(lines) + '\n'

    # --- Serialization / Deserialization ---
    def to_layers_dict(self) -> Dict[str, Any]:
        """Serialize the circuit to a layers snapshot compatible with paper_data.

        Structure matches paper_data/bin/run_unit.py's ad hoc writer, so it can
        be used as a drop-in replacement for reproducible schedule round-tripping.
        """
        # Product membership summary for analysis
        try:
            products = self.dcode.products_list()  # type: ignore[attr-defined]
            prod_members = {p.label: set(p.members) for p in products}
        except Exception:
            prod_members = {}

        analysis = analyze_layers(prod_members, self.layers, interesting_labels=None)
        all_labels = list(getattr(self.dcode, "quasi_labels", []))  # type: ignore[attr-defined]

        out_layers: List[Dict[str, Any]] = []
        for t, Lk in enumerate(self.layers):
            chosen_ids = {stab.label: int(shed.id) for stab, shed in Lk.chosen.items()}
            full_map = {lab: (chosen_ids.get(lab) if lab in chosen_ids else None) for lab in all_labels}
            measured_labels = sorted(chosen_ids.keys())
            per_t = analysis['per_layer'][t] if t < len(analysis.get('per_layer', [])) else {"in_process": [], "completed": []}
            out_layers.append({
                "t": int(t),
                "schedule_id_by_label": full_map,
                "measured_labels": measured_labels,
                "products": {
                    "in_process": list(per_t.get("in_process", [])),
                    "completed": list(per_t.get("completed", [])),
                },
            })

        return {
            "L": int(self.L),
            "solve_time_ms": int(round(max(0.0, float(self.solve_time)) * 1000)),
            "labels": all_labels,
            "layers": out_layers,
            "product_completions": analysis.get('product_completions', {}),
        }

    def to_layers_json(self, path: str | None = None) -> str:
        """Return JSON string; optionally writes to path if provided.

        Use this in place of the ad hoc writer in the worker.
        """
        payload = self.to_layers_dict()
        s = json.dumps(payload, indent=2)
        if path is not None:
            with open(path, 'w') as f:
                f.write(s)
        return s

    @classmethod
    def from_layers_dict(
        cls,
        snapshot: Dict[str, Any],
        dcode: "DefectiveCode",
        *,
        verify: bool = True,
        strict: bool = True,
        backend: str = "solver",
    ) -> "SyndromeExtractionCircuit":
        """Rebuild a circuit from a layers snapshot.

        - verify=True: checks layer compatibility against a freshly computed
          scheduling graph for the given `dcode`.
        - strict=True: require the label set in the snapshot to match the
          dcode's quasi labels exactly.
        - backend: 'solver' (default) uses ScheduleSolver for validation and as
          the layer.code; 'light' builds a minimal stub object with the required
          attributes for downstream end-cycle routines.
        """
        from acid.scheduling.types import StabiliserTemplate, Stabiliser
        from acid.solver.schedule_solver import ScheduleSolver

        L_val = int(snapshot.get("L", 0))
        labels_in = list(snapshot.get("labels", []))
        layers_in = list(snapshot.get("layers", []))
        solve_time_ms = int(snapshot.get("solve_time_ms", 0))

        # Validate/align labels
        d_labels = list(getattr(dcode, "quasi_labels", []))  # type: ignore[attr-defined]
        set_in = set(labels_in)
        set_dc = set(d_labels)
        if strict and set_in != set_dc:
            missing = sorted(list(set_dc - set_in))
            extra = sorted(list(set_in - set_dc))
            raise ValueError(
                f"Label set mismatch between snapshot and defective code.\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}"
            )
        # Use intersection if not strict
        valid_labels = sorted(list(set_in & set_dc)) if not strict else labels_in

        # Build stabilisers map by label without solving
        triplets: List[Tuple[StabiliserTemplate, List[int], str]] = getattr(dcode, "triplets", [])  # type: ignore[attr-defined]
        stab_by_label: Dict[str, Stabiliser] = {}
        for tmpl, qmap, lab in triplets:
            stab_by_label[lab] = tmpl.make_stabiliser(qmap, lab)

        if verify or backend == "solver":
            # Build a solver to compute scheduling graph & reuse for layer.code
            solver = ScheduleSolver(
                dcode.connectivity_graph,  # type: ignore[attr-defined]
                triplets,
                anticommutation_graph=getattr(dcode, "anticomm_graph", None),  # type: ignore[attr-defined]
                product_stabilisers=getattr(dcode, "products", getattr(dcode, "products_list", lambda: [])()),  # type: ignore[attr-defined]
            )
            code_for_layer = solver
        else:
            # Minimal stub providing .stabilisers and .num_qubits
            class _StubCode:
                def __init__(self, stab_map: Dict[str, Stabiliser], n: int):
                    self.stabilisers = stab_map
                    self.num_qubits = int(n)

            code_for_layer = _StubCode(stab_by_label, getattr(dcode.base_code, "num_qubits", 0))  # type: ignore[attr-defined]

        layers_out: List[SyndromeExtractionLayer] = []
        for ent in layers_in:
            sched_map = ent.get("schedule_id_by_label", {})
            chosen: Dict[Stabiliser, Any] = {}
            for lab in valid_labels:
                sid = sched_map.get(lab, None)
                if sid is None:
                    continue
                stab = stab_by_label.get(lab)
                if stab is None:
                    continue  # filtered by non-strict path
                sid_int = int(sid)
                sheds = stab.stabiliser_template.schedules
                if not (0 <= sid_int < len(sheds)):
                    raise ValueError(f"Invalid schedule id {sid_int} for label {lab}; K={len(sheds)}")
                chosen[stab] = sheds[sid_int]
            Lk = SyndromeExtractionLayer(chosen=chosen, code=code_for_layer)
            if verify and hasattr(code_for_layer, "scheduling_graph"):
                if not Lk.is_compatible_with_scheduling_graph(code_for_layer.scheduling_graph):  # type: ignore[attr-defined]
                    raise ValueError("Layer not compatible with scheduling graph (verification failed)")
            layers_out.append(Lk)

        if len(layers_out) != L_val:
            # Not fatal, but warn via exception to enforce consistency
            raise ValueError(f"Snapshot L={L_val} but constructed {len(layers_out)} layers")

        return SyndromeExtractionCircuit(layers=layers_out, solve_time=float(solve_time_ms) / 1000.0, L=L_val, dcode=dcode)

    @classmethod
    def from_layers_json(
        cls,
        path: str,
        dcode: "DefectiveCode",
        *,
        verify: bool = True,
        strict: bool = True,
        backend: str = "solver",
    ) -> "SyndromeExtractionCircuit":
        with open(path, 'r') as f:
            snapshot = json.load(f)
        return cls.from_layers_dict(snapshot, dcode, verify=verify, strict=strict, backend=backend)

    def commuting_bases(self) -> List[CommutingPauliBasis]:
        bases: List[CommutingPauliBasis] = []
        bases.append(self.dcode.midcycle_untouched_stabilisers())
        prod = self.dcode.midcycle_product_stabilisers()
        if prod is not None:
            bases.append(prod)
        # EndCycle[i]: propagate base-code stabilisers not selected by layer i,
        # but we can use existing endcycle_expanded_stabilisers to construct supports.
        for idx, Lk in enumerate(self.layers):
            x_rows, z_rows, _ = Lk.endcycle_expanded_stabilisers()
            basis = CommutingPauliBasis.from_supports(name=f"EndCycle[{idx}]", priority=2+idx, x_supports=x_rows, z_supports=z_rows, n=self.dcode.base_code.num_qubits)
            bases.append(basis)
        # Sort by priority (higher first)
        bases.sort(key=lambda b: b.priority, reverse=True)
        return bases

    def anticommuting_bases(self) -> Dict[str, AntiCommutingPauliBasis]:
        bases: Dict[str, AntiCommutingPauliBasis] = {}
        Lx_mid, Lz_mid = self.dcode._logical_pairs_mid_paulis()
        Gx_mid, Gz_mid = self.dcode._gauge_pairs_mid_paulis()

        bases["Lmid"] = AntiCommutingPauliBasis(name="Lmid", X_rows=Lx_mid, Z_rows=Lz_mid)
        bases["Gmid"] = AntiCommutingPauliBasis(name="Gmid", X_rows=Gx_mid, Z_rows=Gz_mid)
        # Per-layer propagated logicals
        for idx, Lk in enumerate(self.layers):
            Xp: List[PauliString] = []
            Zp: List[PauliString] = []
            for p in Lx_mid:
                Xp.append(Lk.propagate(p))
            for p in Lz_mid:
                Zp.append(Lk.propagate(p))
            bases[f"L{idx}"] = AntiCommutingPauliBasis(name=f"L{idx}", X_rows=Xp, Z_rows=Zp)

        # Per-layer gauges (if any)
        # for idx, Lk in enumerate(self.layers):
        #     Xg, Zg = self.dcode._gauge_pairs_mid_paulis()
        #     if Xg or Zg:
        #         Xp: List[PauliString] = []
        #         Zp: List[PauliString] = []
        #         for p in Xg:
        #             Xp.append(Lk.propagate(p))
        #         for p in Zg:
        #             Zp.append(Lk.propagate(p))
        #         bases[f"G{idx}"] = AntiCommutingPauliBasis(name=f"G{idx}", X_rows=Xp, Z_rows=Zp)


        return bases
