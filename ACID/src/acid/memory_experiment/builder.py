from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class StimBuilder:
    """
    Minimal Stim text builder that tracks measurement record indexes.

    - Maintains a list of lines to be joined into the final circuit.
    - Tracks current measurement count to resolve rec[-k] offsets at DETECTOR/OBS lines.
    - Provides helpers to append operations and get rec indices.
    """

    lines: List[str]
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

    def CX(self, pairs: List[Tuple[int, int]]) -> None:
        if not pairs:
            return
        flat: List[str] = []
        for c, t in pairs:
            flat.append(str(int(c)))
            flat.append(str(int(t)))
        self.append_line("CX " + " ".join(flat))

    def MX(self, qubits: List[int]) -> List[int]:
        qs = list(map(int, qubits))
        if not qs:
            return []
        self.append_line("MX " + " ".join(str(q) for q in qs))
        recs = list(range(self._rec_count, self._rec_count + len(qs)))
        self._rec_count += len(qs)
        return recs

    def MZ(self, qubits: List[int]) -> List[int]:
        qs = list(map(int, qubits))
        if not qs:
            return []
        self.append_line("MZ " + " ".join(str(q) for q in qs))
        recs = list(range(self._rec_count, self._rec_count + len(qs)))
        self._rec_count += len(qs)
        return recs

    def MPP_terms(self, terms: List[List[Tuple[str, int]]]) -> List[int]:
        """
        Emit an MPP instruction where each term is [[('X',q1),('X',q2)], [('Z',q3),...], ...].
        Returns the list of rec indices produced.
        """
        if not terms:
            return []
        parts: List[str] = []
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

    def DETECTOR(self, rec_indices: List[int]) -> None:
        """Emit a DETECTOR referencing given absolute rec indices (0-based)."""
        if not rec_indices:
            return
        # Convert to rec[-k] relative to current _rec_count
        rels = [-(self._rec_count - ri) for ri in rec_indices]
        parts = [f"rec[{r}]" for r in rels]
        self.append_line("DETECTOR " + " ".join(parts))

    def OBSERVABLE_INCLUDE(self, obs_index: int, rec_indices: List[int]) -> None:
        if not rec_indices:
            return
        rels = [-(self._rec_count - ri) for ri in rec_indices]
        parts = [f"rec[{r}]" for r in rels]
        self.append_line(f"OBSERVABLE_INCLUDE({int(obs_index)}) " + " ".join(parts))
