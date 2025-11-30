from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


class StimBuilderProtocol:
    """Minimal protocol the builder exposes for noise hooks."""

    def append_line(self, line: str) -> None: ...
    def current_rec_index(self) -> int: ...


class NoiseModel:
    """
    Declarative noise model with hooks called by the experiment builder.

    Hooks are invoked only during noisy rounds (steps 4–5).
    Implementations should emit stim lines onto the builder that introduce noise
    on the specified targets.
    """

    def apply_after_gate(self, builder: StimBuilderProtocol, gate: str, targets: List[Tuple[int, ...]]) -> None:
        """Called after a gate is emitted.

        - gate: gate name (e.g., "CX").
        - targets: list of tuples (e.g., [(c,t), ...] for a 2q gate).
        """
        return None

    def apply_after_reset(self, builder: StimBuilderProtocol, qubits: Iterable[int], *, basis: str = 'Z') -> None:
        """Called after a reset line on the given qubits.

        basis: 'Z' for R (|0>), 'X' for RX (|+>)
        """
        return None

    def apply_before_measure(self, builder: StimBuilderProtocol, basis: str, qubits_or_terms: Iterable) -> None:
        """Called immediately before measurement.

        - basis: 'X' or 'Z' for MX/MZ; 'PP' for MPP.
        - qubits_or_terms: for MX/MZ it is a list of qubit ids; for MPP a list of terms, each term is a list of (pauli, qid).
        """
        return None


class NoNoiseModel(NoiseModel):
    """No-op noise model."""
    pass


@dataclass
class DepolarizingNoiseModel(NoiseModel):
    """
    Simple depolarizing and flip model aligned with stim's reference:
      - After CX: DEPOLARIZE2(p2) on its targets (matches --after_clifford_depolarization for 2q gates).
      - Before measurement: apply a flip that anti-commutes with the basis
            * before M / MZ: X_ERROR(p1)
            * before MX: Z_ERROR(p1)
      - After reset: apply a flip that anti-commutes with the prepared basis
            * after R (|0>, Z reset): X_ERROR(p1)
            * after RX (|+>, X reset): Z_ERROR(p1)

    MPP currently left noiseless, but can be extended.
    """
    p1: float = 0.0  # single-qubit flip prob (pre-measure and post-reset; anti-commuting)
    p2: float = 0.0  # two-qubit depolarizing after CX

    def apply_after_gate(self, builder: StimBuilderProtocol, gate: str, targets: List[Tuple[int, ...]]) -> None:
        if self.p2 <= 0:
            return
        if gate.upper() in ("CX", "CNOT"):
            for c, t in targets:
                builder.append_line(f"DEPOLARIZE2({self.p2}) {c} {t}")

    def apply_before_measure(self, builder: StimBuilderProtocol, basis: str, qubits_or_terms: Iterable) -> None:
        if self.p1 <= 0:
            return
        # Apply flips that anti-commute with the measured basis
        if basis.upper() == 'X':  # MX
            for q in list(qubits_or_terms):
                builder.append_line(f"Z_ERROR({self.p1}) {int(q)}")
        elif basis.upper() == 'Z':  # M / MZ
            for q in list(qubits_or_terms):
                builder.append_line(f"X_ERROR({self.p1}) {int(q)}")
        # For 'PP' (MPP), extend if needed.

    def apply_after_reset(self, builder: StimBuilderProtocol, qubits: Iterable[int], *, basis: str = 'Z') -> None:
        if self.p1 <= 0:
            return
        # Apply flips that anti-commute with the prepared basis
        if basis.upper() == 'Z':  # R
            for q in list(qubits):
                builder.append_line(f"X_ERROR({self.p1}) {int(q)}")
        elif basis.upper() == 'X':  # RX
            for q in list(qubits):
                builder.append_line(f"Z_ERROR({self.p1}) {int(q)}")
