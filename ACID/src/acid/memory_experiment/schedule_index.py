from __future__ import annotations

from dataclasses import dataclass

from acid.defects.defective_code import DefectiveCode
from acid.scheduling.types import SyndromeExtractionLayer


@dataclass
class ScheduleIndex:
    dcode: DefectiveCode
    layers: list[SyndromeExtractionLayer]

    def __post_init__(self) -> None:
        self.L = len(self.layers)
        # Per-layer measured labels and label->root maps
        self.per_layer_labels: list[set[str]] = []
        self.label_to_root: list[dict[str, int]] = []
        self.root_to_label: list[dict[int, str]] = []
        for Lk in self.layers:
            labs = set(stab.label for stab in Lk.chosen.keys())
            self.per_layer_labels.append(labs)
            lab_to_root: dict[str, int] = {}
            root_to_lab: dict[int, str] = {}
            for stab, shed in Lk.chosen.items():
                rq = stab.qubit_map[shed.root]
                lab_to_root[stab.label] = rq
                root_to_lab[rq] = stab.label
            self.label_to_root.append(lab_to_root)
            self.root_to_label.append(root_to_lab)

        # Quasi basis map
        self.basis_of: dict[str, str] = {}
        for lab in self.dcode.quasi_labels:  # type: ignore[attr-defined]
            typ, _ = self.dcode.quasi_support(lab)
            self.basis_of[lab] = typ
        # Basis label sets for MPP events
        self.labels_x: set[str] = {lab for lab, b in self.basis_of.items() if b == "X"}
        self.labels_z: set[str] = {lab for lab, b in self.basis_of.items() if b == "Z"}

        # Contracting layer indices per label (within one schedule period)
        self.contracting_ts: dict[str, list[int]] = {}
        for t, labs in enumerate(self.per_layer_labels):
            for lab in labs:
                self.contracting_ts.setdefault(lab, []).append(t)

        # Anticommutation neighbors
        G = self.dcode.anticommutation_graph()
        self.neighbors: dict[str, set[str]] = {}
        for u, v in G.edges():
            self.neighbors.setdefault(u, set()).add(v)
            self.neighbors.setdefault(v, set()).add(u)

    def rounds_for_label(self, lab: str, R: int) -> list[tuple[int, int]]:
        """Return (r,t) pairs for rounds 1..R where label lab contracts (measured)."""
        ts = self.contracting_ts.get(lab, [])
        return [(r, t) for r in range(1, R + 1) for t in ts]

    def any_anticomm_measured_between(
        self, lab: str, A: tuple[int, int], B: tuple[int, int]
    ) -> bool:
        """
        Return True if any layer strictly between (A,B) measures a quasi that anticommutes with 'lab'.
        A,B are (round, layer) indices with 1-based round and 0-based layer.
        """
        if A >= B:
            return False
        L = self.L
        a_idx = (A[0] - 1) * L + A[1]
        b_idx = (B[0] - 1) * L + B[1]
        nbrs = self.neighbors.get(lab, set())
        if not nbrs:
            return False
        for idx in range(a_idx + 1, b_idx):
            t = idx % L
            labs = self.per_layer_labels[t]
            if labs & nbrs:
                return True
        return False

    # --- Unified anticomm guard across init/final and schedule layers ---
    def _event_index(
        self, kind: str, basis: str | None, rt: tuple[int, int] | None, R: int
    ) -> int:
        """Map an anchor (kind,basis,rt) to a linear event index.

        Event order:
          0: initX
          1: initZ
          2..(2+R*L-1): per-round layers in order
          2+R*L: finalX
          2+R*L+1: finalZ
        """
        if kind == "init":
            assert basis in ("X", "Z")
            return 0 if basis == "X" else 1
        if kind == "contract":
            assert rt is not None
            r, t = rt
            return 2 + (int(r) - 1) * self.L + int(t)
        if kind == "final":
            assert basis in ("X", "Z")
            return 2 + R * self.L + (0 if basis == "X" else 1)
        raise ValueError(f"Unknown event kind: {kind}")

    def _measured_labels_at_event(self, eidx: int, R: int) -> set[str]:
        if eidx == 0:
            return self.labels_x
        if eidx == 1:
            return self.labels_z
        last_x = 2 + R * self.L
        last_z = last_x + 1
        if eidx == last_x:
            return self.labels_x
        if eidx == last_z:
            return self.labels_z
        # schedule layer
        if 2 <= eidx < last_x:
            t = (eidx - 2) % self.L
            return self.per_layer_labels[t]
        raise ValueError("Event index out of range")

    def any_anticomm_measured_between_events(
        self,
        *,
        lab: str,
        basis: str,
        A: tuple[str, tuple[int, int] | None],
        B: tuple[str, tuple[int, int] | None],
        R: int,
    ) -> bool:
        """
        Return True if any anticommuting quasi of 'lab' is measured at any event strictly
        between anchors A and B (which may be init/final or a contract layer).
        """
        eA = self._event_index(A[0], basis if A[0] != "contract" else None, A[1], R)
        eB = self._event_index(B[0], basis if B[0] != "contract" else None, B[1], R)
        if eA >= eB:
            return False
        nbrs = self.neighbors.get(lab, set())
        if not nbrs:
            return False
        for e in range(eA + 1, eB):
            labs = self._measured_labels_at_event(e, R)
            if labs & nbrs:
                return True
        return False
