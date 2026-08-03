from __future__ import annotations

"""Embedding abstractions for mapping code coordinates to planar layouts.

- SquareGridEmbedding: periodic (torus) with L/R qubits per cell.
- CoordMapEmbedding: explicit integer grid coordinates (planar).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .codes.bb.algebra import GroupRing, Monomial


class Embedding(ABC):
    @abstractmethod
    def qubit_coords_to_index(self, a: int, b: int, c: int) -> int:
        pass

    @abstractmethod
    def id_to_tuple(self, qid: int) -> tuple[int, int, int]:
        pass

    @abstractmethod
    def coords(self, a: int, b: int, c: int) -> tuple[float, float]:
        pass

    @abstractmethod
    def id_and_coords_for(self, g: Monomial, c: int) -> tuple[int, tuple[float, float]]:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass


@dataclass
class SquareGridEmbedding(Embedding):
    """Periodic (torus) square-grid embedding for bivariate bicycle codes.

    Each cell (a, b) in Z_l x Z_m holds two qubits: a left qubit (c=0) and a
    right qubit (c=1). Left qubits are offset half a pitch in x; right qubits
    are offset half a pitch in y, producing a checkerboard-style layout.

    Attributes:
        ring: The group ring Z_l x Z_m defining the code lattice.
        pitch: Spacing between adjacent cells in the planar layout.
        num_qubits: Total number of qubits (set automatically to l * m * 2).sq
    """

    ring: GroupRing
    pitch: float = 1.0
    num_qubits: int = 0  # will be set in __post_init__

    def __post_init__(self):
        self.num_qubits = self.ring.l * self.ring.m * 2

    def qubit_coords_to_index(self, a: int, b: int, c: int) -> int:
        a0, b0 = self.ring.canonical(a, b)
        return ((a0 * self.ring.m) + b0) * 2 + (c & 1)

    def id_to_tuple(self, qid: int) -> tuple[int, int, int]:
        assert 0 <= qid < self.num_qubits, "Qubit ID out of range"
        a = (qid // 2) // self.ring.m
        b = (qid // 2) % self.ring.m
        c = qid % 2
        return (a, b, c)

    def coords(self, a: int, b: int, c: int) -> tuple[float, float]:
        a0, b0 = self.ring.canonical(a, b)
        p = self.pitch
        x = a0 * p + (c & 1) * (p / 2)
        y = b0 * p + (1 - (c & 1)) * (p / 2)
        return x, y

    def id_and_coords_for(self, g: Monomial, c: int) -> tuple[int, tuple[float, float]]:
        i = self.qubit_coords_to_index(g.a, g.b, c)
        return i, self.coords(g.a, g.b, c)

    def shifted_positions(self, qubits: list[int], da: int, db: int) -> list[int]:
        """Return qubit indices shifted by (da, db) on the periodic lattice.

        Each qubit's (a, b, c) coordinate is offset by (da, db) modulo the ring
        dimensions, preserving the left/right label c.

        Args:
            qubits: List of qubit indices to shift.
            da: Offset in the first lattice direction (Z_l).
            db: Offset in the second lattice direction (Z_m).

        Returns:
            List of shifted qubit indices, in the same order as the input.
        """
        result = []
        for qid in qubits:
            a, b, c = self.id_to_tuple(qid)
            result.append(self.qubit_coords_to_index(a + da, b + db, c))
        return result

    @property
    def height(self) -> int:
        return self.ring.m

    @property
    def width(self) -> int:
        return self.ring.l


class CoordMapEmbedding(Embedding):
    def __init__(self, xy_to_id: dict[tuple[int, int], int]):
        self._xy_to_id = dict(xy_to_id)
        self._id_to_xy = {qid: xy for xy, qid in self._xy_to_id.items()}
        self.num_qubits = len(self._id_to_xy)
        self._width = max(x for x, _ in self._xy_to_id) + 1 if self._xy_to_id else 0
        self._height = max(y for _, y in self._xy_to_id) + 1 if self._xy_to_id else 0

    def qubit_coords_to_index(self, a: int, b: int, c: int) -> int:
        return self._xy_to_id[(a, b)]

    def id_to_tuple(self, qid: int) -> tuple[int, int, int]:
        x, y = self._id_to_xy[qid]
        return (x, y, 0)

    def coords(self, a: int, b: int, c: int):
        return float(a), float(b)

    def id_and_coords_for(self, g, c: int):
        raise NotImplementedError

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width
