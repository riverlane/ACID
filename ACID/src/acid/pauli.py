from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

from acid.gap_distance import compute_nkd_with_gap

from acid.gf2_utils import gf2_rank, gf2_is_in_span, gf2_left_nullspace


@dataclass
class PauliString:
    n: int
    X: List[int]
    Z: List[int]

    @staticmethod
    def from_supports(Xs: Iterable[int], Zs: Iterable[int], n: int) -> "PauliString":
        X = [0] * n
        Z = [0] * n
        for q in Xs:
            if 0 <= int(q) < n:
                X[int(q)] ^= 1
        for q in Zs:
            if 0 <= int(q) < n:
                Z[int(q)] ^= 1
        return PauliString(n=n, X=X, Z=Z)

    @staticmethod
    def from_2n(row: Iterable[int]) -> "PauliString":
        r = [int(v) & 1 for v in row]
        assert len(r) % 2 == 0
        n = len(r) // 2
        X = r[:n]
        Z = r[n:]
        return PauliString(n=n, X=X, Z=Z)

    def support(self) -> List[int]:
        return [i for i, b in enumerate(self.X + self.Z) if b & 1]
    
    def z_support(self) -> List[int]:
        return [i for i, b in enumerate(self.Z) if b & 1]
    
    def x_support(self) -> List[int]:
        return [i for i, b in enumerate(self.X) if b & 1]

    def to_2n(self) -> List[int]:
        return [int(v) & 1 for v in (self.X + self.Z)]

    def to_supports(self) -> Tuple[set[int], set[int]]:
        Xs = {i for i, b in enumerate(self.X) if b & 1}
        Zs = {i for i, b in enumerate(self.Z) if b & 1}
        return Xs, Zs

    def copy(self) -> "PauliString":
        return PauliString(n=self.n, X=self.X[:], Z=self.Z[:])

    def symplectic_dot(self, other: "PauliString") -> int:
        assert self.n == other.n
        acc = 0
        for i in range(self.n):
            acc ^= (self.X[i] & other.Z[i])
            acc ^= (self.Z[i] & other.X[i])
        return acc & 1

    def commutes_with(self, other: "PauliString") -> bool:
        return self.symplectic_dot(other) == 0

    def multiply(self, other: "PauliString") -> "PauliString":
        assert self.n == other.n
        X = [(a ^ b) & 1 for a, b in zip(self.X, other.X)]
        Z = [(a ^ b) & 1 for a, b in zip(self.Z, other.Z)]
        return PauliString(n=self.n, X=X, Z=Z)

    def conj_cnot(self, control: int, target: int) -> None:
        # X on control toggles X on target
        if 0 <= control < self.n and 0 <= target < self.n:
            if self.X[control] & 1:
                self.X[target] ^= 1
            # Z on target toggles Z on control
            if self.Z[target] & 1:
                self.Z[control] ^= 1

    def conj_steps(self, steps: List[List[Tuple[int, int]]]) -> None:
        for ops in steps:
            for c, t in ops:
                self.conj_cnot(c, t)

    def weight(self) -> int:
        return sum(self.X) + sum(self.Z)

    def __repr__(self) -> str:
        return f"PauliString(n={self.n}, X={self.X}, Z={self.Z})"


@dataclass
class CommutingPauliBasis:
    name: str
    priority: int
    rows: List[PauliString]

    def as_2n_matrix(self) -> List[List[int]]:
        return [p.to_2n() for p in self.rows]

    def count_membership_in(self, S_rows_2n: List[List[int]]) -> Tuple[int, List[int]]:
        # Return (count, indices) where basis.rows[idx] ∈ span(S)
        idxs: List[int] = []
        for j, p in enumerate(self.rows):
            if gf2_is_in_span(p.to_2n(), S_rows_2n):
                idxs.append(j)
        return len(idxs), idxs

    @staticmethod
    def from_supports(name: str, priority: int, x_supports: List[List[int]], z_supports: List[List[int]], n: int) -> "CommutingPauliBasis":
        # Build PauliStrings then greedily reduce to an independent set
        rows: List[PauliString] = []
        for supp in x_supports:
            rows.append(PauliString.from_supports(supp, [], n))
        for supp in z_supports:
            rows.append(PauliString.from_supports([], supp, n))
        # Independent reduction
        acc: List[List[int]] = []
        keep: List[PauliString] = []
        r = 0
        for p in rows:
            row = p.to_2n()
            if gf2_rank(acc + [row]) > r:
                acc.append(row)
                keep.append(p)
                r += 1
        return CommutingPauliBasis(name=name, priority=int(priority), rows=keep)


@dataclass
class AntiCommutingPauliBasis:
    name: str
    X_rows: List[PauliString]
    Z_rows: List[PauliString]

    def stacked_2n(self) -> List[List[int]]:
        return [p.to_2n() for p in (self.X_rows + self.Z_rows)]

    def describe_intersection_with(self, S_rows_2n: List[List[int]]) -> List[str]:
        # Left nullspace trick on M = [B; S], where B = [X_rows; Z_rows]
        B = self.stacked_2n()
        if not B:
            return []
        M = B + S_rows_2n
        L = gf2_left_nullspace(M)
        if not L:
            return []
        desc: List[str] = []
        p = len(B)
        kx = len(self.X_rows)
        Y_acc: List[List[int]] = []
        for w in L:
            if len(w) != len(M):
                continue
            x = w[:p]
            # yi = x * B
            yi = [0] * (2 * self.X_rows[0].n)
            for idx, bit in enumerate(x):
                if bit & 1:
                    row = B[idx]
                    yi = [a ^ b for a, b in zip(yi, row)]
            # Keep independent yi
            if gf2_rank(Y_acc + [yi]) == gf2_rank(Y_acc):
                continue
            Y_acc.append(yi)
            parts: List[str] = []
            for j in range(kx):
                if x[j] & 1:
                    parts.append(f"X{j+1}")
            for j in range(len(self.Z_rows)):
                if x[kx + j] & 1:
                    parts.append(f"Z{j+1}")
            desc.append("".join(parts) if parts else "1")
        return desc


# CSS/Stabiliser code containers and helpers

@dataclass
class StabiliserCode:
    num_qubits: int
    row_labels: List[str]
    Hx: List[List[int]]
    Hz: List[List[int]]

    def __len__(self) -> int:
        return len(self.row_labels)

    @property
    def symplectic(self) -> List[List[int]]:
        return [hx_row + hz_row for hx_row, hz_row in zip(self.Hx, self.Hz)]

    def nkd_via_gap(self, *, gap_exe: str = "gap", trials: int = 1000, mindist: int = 0, debug: int = 1, timeout: int = 300) -> tuple[int, int, int]:
        n, k, d = compute_nkd_with_gap(self.Hx, self.Hz, gap_exe=gap_exe, trials=trials, mindist=mindist, debug=debug, timeout=timeout)
        return n, k, d



    # Future: add gauges if desired as another AntiCommutingPauliBasis

