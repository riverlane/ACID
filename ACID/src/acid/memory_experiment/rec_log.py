from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MeasurementLog:
    """
    Stores only rec indices for measurements used by detector planning.

    - init_mpp: per-basis dict of quasi label -> rec index
    - final_mpp: per-basis dict of quasi label -> rec index
    - per_layer: mapping (round_idx, layer_idx, basis) -> {root_qid -> rec}
    """

    init_mpp: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"X": {}, "Z": {}}
    )
    final_mpp: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"X": {}, "Z": {}}
    )
    per_layer: dict[tuple[int, int, str], dict[int, int]] = field(default_factory=dict)

    def record_init_mpp(self, basis: str, label: str, rec: int) -> None:
        b = basis.upper()
        self.init_mpp[b][label] = int(rec)

    def record_final_mpp(self, basis: str, label: str, rec: int) -> None:
        b = basis.upper()
        self.final_mpp[b][label] = int(rec)

    def record_layer_meas(
        self, round_idx: int, layer_idx: int, basis: str, root_q: int, rec: int
    ) -> None:
        key = (int(round_idx), int(layer_idx), basis.upper())
        self.per_layer.setdefault(key, {})[int(root_q)] = int(rec)
