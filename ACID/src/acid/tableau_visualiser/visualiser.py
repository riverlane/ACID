from __future__ import annotations
import stim
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union

from ..gf2_utils import (
    gf2_left_nullspace,
    gf2_rank,
    gf2_is_in_span,
)
from acid.pauli import CommutingPauliBasis, AntiCommutingPauliBasis


def pauli_string_to_row(p: stim.PauliString) -> List[int]:
    s = str(p)
    if s and s[0] in "+-":
        s = s[1:]
    qubits = [c for c in s]
    n = len(qubits)
    X = [0] * n
    Z = [0] * n
    for i, c in enumerate(qubits):
        if c == "X":
            X[i] = 1
        elif c == "Z":
            Z[i] = 1
        elif c == "Y":
            X[i] = 1
            Z[i] = 1
        else:
            pass
    return X + Z


@dataclass
class TableauSnapshot:
    tick_index: int
    n: int
    stabiliser_sections: List[
        Tuple[Union[CommutingPauliBasis, AntiCommutingPauliBasis], List[int]]
    ]
    logical_descriptions: List[str]

    def to_ansi(self) -> str:
        out: List[str] = []
        out.append(f"Snapshot @ TICK #{self.tick_index}")
        out.append("Stabilisers:")
        if not self.stabiliser_sections:
            out.append("  (none)")
        count_sum = 0
        for basis, row_indices in self.stabiliser_sections:
            count = len(row_indices)
            count_sum += count
            preview = ",".join(str(i) for i in row_indices[:8])
            if count == 0:
                out.append(f"  [ {basis.name} ] count=0 of {len(basis.rows)}")
            else:
                out.append(
                    f"  [ {basis.name} ] count={count} of {len(basis.rows)} indices=[{preview}{',' if count > 8 else ''}{'...' if count > 8 else ''}]"
                )

        out.append("Logical/Gauges:")
        if not self.logical_descriptions:
            out.append("  (none)")
        else:
            for s in self.logical_descriptions:
                count_sum += 1
                out.append(f"  Stabilised by: {s}")
        out.append(f"Unknown: {self.n - count_sum}")
        return "\n".join(out)


class TableauVisualiser:
    def __init__(
        self,
        circuit: stim.Circuit,
        commuting_bases: List[CommutingPauliBasis],
        anticommuting_bases: Dict[str, AntiCommutingPauliBasis],
    ) -> None:
        self.circuit = circuit
        self.instructions = list(circuit)
        # Determine n from circuit target range (assume qubits 0..max)
        max_q = 0
        for inst in self.instructions:
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    if t.value > max_q:
                        max_q = t.value
        self.n = max_q + 1
        # Sort commuting bases by priority descending
        self.commuting_bases = list(commuting_bases)
        self.commuting_bases.sort(key=lambda b: b.priority, reverse=True)
        self.anticommuting_bases = dict(anticommuting_bases)

        # Simulation state
        self.sim = stim.TableauSimulator()
        self.ip = 0
        self.tick_count = 0

    def step_instruction(self) -> bool:
        if self.ip >= len(self.instructions):
            return False
        inst = self.instructions[self.ip]
        name = inst.name
        if name == "TICK":
            self.tick_count += 1
            self.ip += 1
            return True
        if name == "QUBIT_COORDS":
            self.ip += 1
            return True
        c = stim.Circuit()
        c.append(name, inst.targets_copy(), inst.gate_args_copy())
        self.sim.do(c)
        self.ip += 1
        return True

    def step_to_next_tick(self) -> bool:
        while self.ip < len(self.instructions):
            inst = self.instructions[self.ip]
            self.step_instruction()
            if inst.name == "TICK":
                return True
        return False

    def _current_stabilizer_rows(self) -> List[List[int]]:
        stabs = self.sim.canonical_stabilizers()
        return [pauli_string_to_row(p) for p in stabs]

    def snapshot(self) -> TableauSnapshot:
        S_rows = self._current_stabilizer_rows()

        chosen_basis_rows: List[List[int]] = []

        # Commuting bases membership
        sections: List[
            Tuple[CommutingPauliBasis, AntiCommutingPauliBasis], List[int]
        ] = []
        for b in self.commuting_bases[::-1]:
            idxs: List[int] = []
            for j, p in enumerate(b.rows):
                row = p.to_2n()
                if not gf2_is_in_span(row, S_rows):
                    continue
                if gf2_is_in_span(row, chosen_basis_rows):
                    continue
                idxs.append(j)
                chosen_basis_rows.append(row)

            sections.append((b, idxs))

        # Anti-commuting bases intersections
        logical_desc: List[str] = []
        for label, anti in self.anticommuting_bases.items():
            # M = [B; S], left-nullspace yields combinations
            B = anti.stacked_2n()
            M = B + S_rows
            L = gf2_left_nullspace(M)
            if not L:
                continue
            p = len(B)
            kx = len(anti.X_rows)
            Y_acc: List[List[int]] = []
            for w in L:
                if len(w) != len(M):
                    continue
                x = w[:p]
                yi = [0] * (2 * self.n)
                for idx, bit in enumerate(x):
                    if bit & 1:
                        row = B[idx]
                        yi = [a ^ b for a, b in zip(yi, row)]
                if gf2_rank(Y_acc + [yi]) == gf2_rank(Y_acc):
                    continue
                Y_acc.append(yi)
                parts: List[str] = []
                for j in range(kx):
                    if x[j] & 1:
                        parts.append(f"X{j + 1}")
                for j in range(len(anti.Z_rows)):
                    if x[kx + j] & 1:
                        parts.append(f"Z{j + 1}")
                logical_desc.append(f"[{label}] " + ("".join(parts) if parts else "1"))

        return TableauSnapshot(
            tick_index=self.tick_count,
            n=self.n,
            stabiliser_sections=sections,
            logical_descriptions=logical_desc,
        )
