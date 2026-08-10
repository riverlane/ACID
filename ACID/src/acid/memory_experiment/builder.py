from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from acid.memory_experiment.noise import StimBuilderProtocol

if TYPE_CHECKING:
    from acid.defects.syndrome_extraction_circuit import SyndromeExtractionCircuit
    from acid.memory_experiment.noise import NoiseModel
    from acid.memory_experiment.rec_log import MeasurementLog
    from acid.scheduling.types import SyndromeExtractionLayer


@dataclass
class StimBuilder(StimBuilderProtocol):
    """
    Minimal Stim text builder that tracks measurement record indexes.

    - Maintains a list of lines to be joined into the final circuit.
    - Tracks current measurement count to resolve rec[-k] offsets at DETECTOR/OBS lines.
    - Provides helpers to append operations and get rec indices.
    """

    lines: list[str]
    _rec_count: int = 0
    _tick_count: int = 0

    def append_line(self, line: str) -> None:
        self.lines.append(line)

    def current_rec_index(self) -> int:
        return self._rec_count

    def tick(self) -> None:
        self.append_line("TICK")
        self._tick_count += 1

    def ticks(self) -> int:
        return self._tick_count

    # --- Gate appenders ---
    def R(self, qubits: Iterable[int]) -> None:
        qs = list(map(int, qubits))
        if qs:
            self.append_line("R " + " ".join(str(q) for q in qs))

    def RX(self, qubits: Iterable[int]) -> None:
        qs = list(map(int, qubits))
        if qs:
            self.append_line("RX " + " ".join(str(q) for q in qs))

    def CX(self, pairs: list[tuple[int, int]]) -> None:
        if not pairs:
            return
        flat: list[str] = []
        for c, t in pairs:
            flat.append(str(int(c)))
            flat.append(str(int(t)))
        self.append_line("CX " + " ".join(flat))

    def SWAP(self, pairs: list[tuple[int, int]]) -> None:
        if not pairs:
            return
        flat: list[str] = []
        for a, b in pairs:
            flat.append(str(int(a)))
            flat.append(str(int(b)))
        self.append_line("SWAP " + " ".join(flat))

    def MX(self, qubits: list[int]) -> list[int]:
        qs = list(map(int, qubits))
        if not qs:
            return []
        self.append_line("MX " + " ".join(str(q) for q in qs))
        recs = list(range(self._rec_count, self._rec_count + len(qs)))
        self._rec_count += len(qs)
        return recs

    def MZ(self, qubits: list[int]) -> list[int]:
        qs = list(map(int, qubits))
        if not qs:
            return []
        self.append_line("MZ " + " ".join(str(q) for q in qs))
        recs = list(range(self._rec_count, self._rec_count + len(qs)))
        self._rec_count += len(qs)
        return recs

    def MPP_terms(self, terms: list[list[tuple[str, int]]]) -> list[int]:
        """
        Emit an MPP instruction where each term is [[('X',q1),('X',q2)], [('Z',q3),...], ...].
        Returns the list of rec indices produced.
        """
        if not terms:
            return []
        parts: list[str] = []
        for term in terms:
            if not term:
                continue
            parts.append("*".join(f"{p}{int(q)}" for p, q in term))
        if not parts:
            return []
        self.append_line("MPP " + " ".join(parts))
        recs = list(range(self._rec_count, self._rec_count + len(parts)))
        self._rec_count += len(parts)
        return recs

    def QUBIT_COORDS(self, q: int, x: float, y: float) -> None:
        self.append_line(f"QUBIT_COORDS({x:.6g}, {y:.6g}) {int(q)}")

    def DETECTOR(self, rec_indices: list[int]) -> None:
        """Emit a DETECTOR referencing given absolute rec indices (0-based)."""
        if not rec_indices:
            return
        # Convert to rec[-k] relative to current _rec_count
        rels = [-(self._rec_count - ri) for ri in rec_indices]
        parts = [f"rec[{r}]" for r in rels]
        self.append_line("DETECTOR " + " ".join(parts))

    def OBSERVABLE_INCLUDE(self, obs_index: int, rec_indices: list[int]) -> None:
        if not rec_indices:
            return
        rels = [-(self._rec_count - ri) for ri in rec_indices]
        parts = [f"rec[{r}]" for r in rels]
        self.append_line(f"OBSERVABLE_INCLUDE({int(obs_index)}) " + " ".join(parts))

    def layer_contract(
        self,
        layer: SyndromeExtractionLayer,
        *,
        noise: NoiseModel,
    ) -> None:
        """Emit the contraction (first) half of a syndrome extraction layer.

        Applies forward CX steps with noise and a TICK after each step.
        """
        for step in layer.collect_cx_stim():
            if step:
                pairs = [(int(c), int(t)) for c, t in step]
                self.CX(pairs)
                noise.apply_after_gate(self, "CX", pairs)
            self.tick()

    def layer_expand(
        self,
        layer: SyndromeExtractionLayer,
        *,
        noise: NoiseModel,
        qubit_map: list[int] | None = None,
        dead_qubits: set[int] | None = None,
        dead_connections: set[tuple[int, int]] | None = None,
    ) -> None:
        """Emit the expansion (second) half of a syndrome extraction layer.

        Applies reversed CX steps with noise and a TICK after each step.

        Args:
            layer: The syndrome extraction layer.
            noise: Noise model applied after each CX step.
            qubit_map: Optional remapping list where qubit_map[q] gives the
                new physical position of qubit q. If None, qubit IDs are
                used unchanged.
            dead_qubits: Optional set of qubit IDs that are considered dropped.
            dead_connections: Optional set of (u, v) tuples that are considered dropped.
        """
        if dead_qubits is None:
            dead_qubits = set()
        if dead_connections is None:
            dead_connections = set()
        for step in reversed(layer.collect_cx_stim()):
            if step:
                if qubit_map is not None:
                    pairs = [(qubit_map[int(c)], qubit_map[int(t)]) for c, t in step]
                else:
                    pairs = [(int(c), int(t)) for c, t in step]
                for c, t in pairs:
                    if c in dead_qubits:
                        msg = f"The layer expansion step contains a CX with a dead qubit {c}. This is not allowed."
                        raise ValueError(msg)
                    if t in dead_qubits:
                        msg = f"The layer expansion step contains a CX with a dead qubit {t}. This is not allowed."
                        raise ValueError(msg)
                    e = (min(c, t), max(c, t))
                    if e in dead_connections:
                        msg = f"The layer expansion step contains a CX with a dead connection {e}. This is not allowed."
                        raise ValueError(msg)
                self.CX(pairs)
                noise.apply_after_gate(self, "CX", pairs)
            self.tick()

    def layer_measure_reset(
        self,
        layer: SyndromeExtractionLayer,
        *,
        noise: NoiseModel,
        log: MeasurementLog | None = None,
        round_idx: int | None = None,
        layer_idx: int | None = None,
    ) -> None:
        """Measure and reset root qubits of a syndrome extraction layer."""
        self.layer_measure(layer, noise=noise, log=log, round_idx=round_idx, layer_idx=layer_idx)
        self.layer_reset(layer, noise=noise)

    def layer_measure(
        self,
        layer: SyndromeExtractionLayer,
        *,
        noise: NoiseModel,
        log: MeasurementLog | None = None,
        round_idx: int | None = None,
        layer_idx: int | None = None,
    ) -> None:
        """Measure root qubits of a syndrome extraction layer.

        Emits MX/MZ for roots then a TICK. Measurements are recorded into log
        if log, round_idx, and layer_idx are all provided.
        """
        x_roots, z_roots = layer.roots_by_basis()
        if x_roots:
            noise.apply_before_measure(self, "X", x_roots)
            x_recs = self.MX(x_roots)
            if log is not None and round_idx is not None and layer_idx is not None:
                for q, rec in zip(x_roots, x_recs):
                    log.record_layer_meas(round_idx, layer_idx, "X", int(q), int(rec))
        if z_roots:
            noise.apply_before_measure(self, "Z", z_roots)
            z_recs = self.MZ(z_roots)
            if log is not None and round_idx is not None and layer_idx is not None:
                for q, rec in zip(z_roots, z_recs):
                    log.record_layer_meas(round_idx, layer_idx, "Z", int(q), int(rec))
        self.tick()

    def layer_reset(
        self,
        layer: SyndromeExtractionLayer,
        *,
        noise: NoiseModel,
        qubit_map: list[int] | None = None,
    ) -> None:
        """Reset root qubits of a syndrome extraction layer.

        Emits RX/R for roots then a TICK.

        Args:
            layer: The syndrome extraction layer.
            noise: Noise model applied after reset.
            qubit_map: Optional remapping list where qubit_map[q] gives the
                new physical position of qubit q. If None, qubit IDs are
                used unchanged.
        """
        x_roots, z_roots = layer.roots_by_basis()
        if qubit_map is not None:
            x_roots = [qubit_map[q] for q in x_roots]
            z_roots = [qubit_map[q] for q in z_roots]
        if x_roots:
            self.RX(x_roots)
            noise.apply_after_reset(self, x_roots, basis="X")
        if z_roots:
            self.R(z_roots)
            noise.apply_after_reset(self, z_roots, basis="Z")
        self.tick()

    def layer_cycle(
        self,
        layer: SyndromeExtractionLayer,
        *,
        noise: NoiseModel,
        log: MeasurementLog,
        round_idx: int,
        layer_idx: int,
    ) -> None:
        """Emit a full syndrome extraction cycle: contract → measure/reset → expand."""
        self.layer_contract(layer, noise=noise)
        self.layer_measure_reset(
            layer, noise=noise, log=log, round_idx=round_idx, layer_idx=layer_idx
        )
        self.layer_expand(layer, noise=noise)

    def memory_rounds(
        self,
        n: int,
        *,
        circuit: SyndromeExtractionCircuit,
        noise: NoiseModel,
        log: MeasurementLog,
    ) -> None:
        """Append n memory cycles (contract, measure, reset, expand) to the circuit."""
        for r in range(1, n + 1):
            for t, Lk in enumerate(circuit.layers):
                self.layer_cycle(Lk, noise=noise, log=log, round_idx=r, layer_idx=t)

    def swap_routing_layers(
        self,
        layers: list[list[tuple[int, int]]],
        *,
        noise: NoiseModel | None = None,
    ) -> None:
        """Append swap routing layers to the circuit.

        Emits one SWAP instruction per layer with a TICK after each, applying
        optional noise after each SWAP layer via noise.apply_after_gate with
        gate="SWAP".

        Args:
            layers: Output of solve_swap_routing — a list of swap layers, each
                layer being a list of (i, j) qubit pairs to swap simultaneously.
            noise: Optional noise model. If provided, apply_after_gate is called
                with gate="SWAP" after each layer.
        """
        for layer in layers:
            pairs = [(int(a), int(b)) for a, b in layer]
            self.SWAP(pairs)
            if noise is not None:
                noise.apply_after_gate(self, "SWAP", pairs)
            self.tick()
