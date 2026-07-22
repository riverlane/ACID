from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from ..gf2_utils import gf2_rank

PauliKind = Literal["stabiliser", "gauge", "logical"]


@dataclass
class PauliBasis:
    name: str
    kind: PauliKind
    priority: int
    rows: List[List[int]]  # each row length 2n (X|Z) representation over GF(2)

    def validate(self, n: int) -> None:
        for r in self.rows:
            if len(r) != 2 * n:
                raise ValueError(
                    f"Basis {self.name} has row with wrong length (expected {2 * n})"
                )
        # Check linear independence (not strictly required, but recommended)
        if gf2_rank(self.rows) != len(self.rows):
            raise ValueError(f"Basis {self.name} rows are not linearly independent")


def pauli_to_bin_row(pauli: str) -> List[int]:
    """Convert a Pauli string like '+X_Z' into a 2n binary row [X...|Z...].

    Ignores the sign and underscores. Y -> X=1, Z=1.
    """
    s = pauli.strip()
    if s and s[0] in "+-":
        s = s[1:]
    qubits = [c for c in s if c in "XYZI_"]
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
            # I or _
            pass
    return X + Z


def default_single_qubit_basis(n: int) -> PauliBasis:
    """Create a default single-qubit basis containing all Z_i and X_i.

    Ordering: all Z_i for i in [0..n-1], then all X_i.
    Kind: 'stabiliser', priority 0.
    """
    rows: List[List[int]] = []
    # Z_i
    for i in range(n):
        X = [0] * n
        Z = [0] * n
        Z[i] = 1
        rows.append(X + Z)
    # X_i
    for i in range(n):
        X = [0] * n
        Z = [0] * n
        X[i] = 1
        rows.append(X + Z)
    return PauliBasis(
        name="Single Qubit Paulis", kind="stabiliser", priority=0, rows=rows
    )
