from __future__ import annotations

from dataclasses import dataclass

from acid.defects.defective_code import DefectiveCode
from acid.defects.syndrome_extraction_circuit import SyndromeExtractionCircuit
from acid.embedding import Embedding
from acid.memory_experiment.builder import StimBuilder
from acid.memory_experiment.embedding_utils import place_ancillas_right_of_bbox
from acid.memory_experiment.noise import NoiseModel, NoNoiseModel

from .observables import plan_observables
from .product_detectors import plan_product_detectors
from .rec_log import MeasurementLog
from .registry import MemoryDetectorRegistry
from .schedule_index import ScheduleIndex
from .single_detectors import plan_quasi_detectors


@dataclass
class MemoryExperimentConfig:
    R: int
    ancilla_dx: float = 1.0
    ancilla_dy: float = 1.0


class MemoryExperiment:
    def __init__(
        self,
        *,
        dcode: DefectiveCode,
        circuit: SyndromeExtractionCircuit,
        embedding: Embedding,
        noise: NoiseModel | None = None,
        cfg: MemoryExperimentConfig | None = None,
    ) -> None:
        self.dcode = dcode
        self.circuit = circuit
        self.embedding = embedding
        self.noise = noise if noise is not None else NoNoiseModel()
        self.cfg = cfg if cfg is not None else MemoryExperimentConfig(R=1)

        # Logical pairs (for ancilla entanglement)
        Lx, Lz = self.dcode._logical_pairs_mid_paulis()  # type: ignore[attr-defined]
        self._Lx = Lx
        self._Lz = Lz
        self.k = min(len(Lx), len(Lz))

        self.L = len(self.circuit.layers)
        self.registry = MemoryDetectorRegistry()

    def _root_qubits_by_basis(self, layer) -> tuple[list[int], list[int]]:
        return layer.roots_by_basis()

    def _root_qubit_map(self, layer) -> dict[int, str]:
        m: dict[int, str] = {}
        for stab, shed in layer.chosen.items():
            root_q = stab.qubit_map[shed.root]
            m[root_q] = stab.label
        return m

    def _entangling_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for i in range(self.k):
            for q in self._Lx[i].x_support():
                pairs.append((-(i + 1), int(q)))
            for q in self._Lz[i].z_support():
                pairs.append((int(q), -(self.k + i + 1)))
        return pairs

    def build(
        self,
        include_x_detectors: bool = True,
        include_z_detectors: bool = True,
        *,
        include_state_prep: bool = True,
        debug: bool = False,
    ) -> str:
        # Prepend base visualisation
        overlay = self.dcode.visualisation_stim(self.embedding)
        overlay_lines = overlay.rstrip().splitlines()
        overlay_tick_count = sum(1 for ln in overlay_lines if ln.strip() == "TICK")
        # helper object to build the stim circuit
        stim_builder = StimBuilder(lines=list(overlay_lines))
        stim_builder._tick_count = overlay_tick_count  # type: ignore[attr-defined]

        # Ancillas (only if doing state prep)
        n = self.dcode.base_code.num_qubits
        zeros: list[int] = []
        plus: list[int] = []
        anc_coords: list[tuple[int, float, float]] = []
        if include_state_prep:
            zeros, plus, anc_coords = place_ancillas_right_of_bbox(
                self.embedding,
                list(range(n)),
                self.k,
                dx=self.cfg.ancilla_dx,
                dy=self.cfg.ancilla_dy,
            )
            for q, x, y in anc_coords:
                stim_builder.QUBIT_COORDS(q, x, y)
            if zeros:
                stim_builder.R(zeros)
            if plus:
                stim_builder.RX(plus)

        log = MeasurementLog()

        # Initial MPP: X then Z (no noise), only if doing state prep
        if include_state_prep:
            # X pass
            for lab in sorted(self.dcode.quasi_labels):  # type: ignore[attr-defined]
                typ, supp = self.dcode.quasi_support(lab)
                if typ != "X" or not include_x_detectors:
                    continue
                term = [[(typ, q) for q in supp]]
                rec = stim_builder.MPP_terms(term)[0]
                log.record_init_mpp("X", lab, rec)
            # Z pass
            for lab in sorted(self.dcode.quasi_labels):  # type: ignore[attr-defined]
                typ, supp = self.dcode.quasi_support(lab)
                if typ != "Z" or not include_z_detectors:
                    continue
                term = [[(typ, q) for q in supp]]
                rec = stim_builder.MPP_terms(term)[0]
                log.record_init_mpp("Z", lab, rec)

        # Entangle (noiseless), only if doing state prep
        ent_pairs: list[tuple[int, int]] = []
        if include_state_prep:
            ent_pairs_placeholder = self._entangling_pairs()
            anc_zero = zeros
            anc_plus = plus
            for c, t in ent_pairs_placeholder:
                if c < 0:
                    c = anc_plus[-c - 1]
                if t < 0:
                    t = anc_zero[-t - 1 - self.k]
                ent_pairs.append((int(c), int(t)))
                stim_builder.CX([(int(c), int(t))])
            stim_builder.tick()

        # Noisy rounds: R cycles over L layers
        stim_builder.memory_rounds(self.cfg.R, circuit=self.circuit, noise=self.noise, log=log)

        # Unentangle and final MPP only if doing state prep
        if include_state_prep:
            # Unentangle (reverse, noiseless)
            for c, t in reversed(ent_pairs):
                stim_builder.CX([(int(c), int(t))])
            stim_builder.tick()
            # Final MPP: X then Z
            for lab in sorted(self.dcode.quasi_labels):  # type: ignore[attr-defined]
                typ, supp = self.dcode.quasi_support(lab)
                if typ != "X" or not include_x_detectors:
                    continue
                term = [[(typ, q) for q in supp]]
                rec = stim_builder.MPP_terms(term)[0]
                log.record_final_mpp("X", lab, rec)
            stim_builder.tick()
            for lab in sorted(self.dcode.quasi_labels):  # type: ignore[attr-defined]
                typ, supp = self.dcode.quasi_support(lab)
                if typ != "Z" or not include_z_detectors:
                    continue
                term = [[(typ, q) for q in supp]]
                rec = stim_builder.MPP_terms(term)[0]
                log.record_final_mpp("Z", lab, rec)
            stim_builder.tick()

        # Ancilla observables (noiseless) — record final ancilla recs, but don't emit yet
        z_anc_recs: list[int] = []
        x_anc_recs: list[int] = []
        if include_state_prep:
            if zeros and include_z_detectors:
                z_anc_recs = stim_builder.MZ(zeros)
            if plus and include_x_detectors:
                x_anc_recs = stim_builder.MX(plus)

        # Plan and emit detectors (offline)
        # Detectors/observables only if requested (and typically require state prep)
        if include_x_detectors or include_z_detectors:
            sched = ScheduleIndex(self.dcode, self.circuit.layers)
            debug_filter = None
            quasi_plan = plan_quasi_detectors(
                dcode=self.dcode,
                layers=self.circuit.layers,
                sched=sched,
                log=log,
                R=self.cfg.R,
                include_x=include_x_detectors,
                include_z=include_z_detectors,
                filter_labels=debug_filter,
            )
            prod_plan = plan_product_detectors(
                dcode=self.dcode,
                layers=self.circuit.layers,
                sched=sched,
                log=log,
                R=self.cfg.R,
                include_x=include_x_detectors,
                include_z=include_z_detectors,
                debug=debug,
            )
            if debug:
                try:
                    print(f"[detectors] planning: single_quasi={len(quasi_plan.rec_sets)}")
                    print(f"[detectors] planning: product={len(prod_plan.rec_sets)}")
                except Exception:
                    pass
            next_id = 0
            for recs, info in zip(quasi_plan.rec_sets, quasi_plan.infos):
                stim_builder.DETECTOR(recs)
                info.id = next_id
                next_id += 1
                self.registry.add(info)
            for recs, info in zip(prod_plan.rec_sets, prod_plan.infos):
                stim_builder.DETECTOR(recs)
                info.id = next_id
                next_id += 1
                self.registry.add(info)

            # Plan and emit observables using propagated logicals across rounds
            obs_z_sets, obs_x_sets = plan_observables(
                dcode=self.dcode,
                layers=self.circuit.layers,
                sched=sched,
                log=log,
                R=self.cfg.R,
                Lx_mid=self._Lx,
                Lz_mid=self._Lz,
                ancilla_recs_x=x_anc_recs,
                ancilla_recs_z=z_anc_recs,
                include_x=include_x_detectors,
                include_z=include_z_detectors,
            )
            obs_index = 0
            for recs in obs_z_sets:
                stim_builder.OBSERVABLE_INCLUDE(obs_index, recs)
                obs_index += 1
            for recs in obs_x_sets:
                stim_builder.OBSERVABLE_INCLUDE(obs_index, recs)
                obs_index += 1

        return "\n".join(stim_builder.lines) + "\n"
