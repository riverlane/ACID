from __future__ import annotations
"""Embedding abstractions for mapping code coordinates to planar layouts.

- SquareGridEmbedding: periodic (torus) with L/R qubits per cell.
- CoordMapEmbedding: explicit integer grid coordinates (planar).
"""

from dataclasses import dataclass
from typing import Tuple, Dict
from abc import ABC, abstractmethod

from .codes.bb.algebra import GroupRing, Monomial


class Embedding(ABC):

    @abstractmethod
    def qubit_id(self, a: int, b: int, c: int) -> int:
        pass

    @abstractmethod
    def id_to_tuple(self, qid: int) -> Tuple[int, int, int]:
        pass
    
    @abstractmethod
    def coords(self, a: int, b: int, c: int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def id_and_coords_for(self, g: Monomial, c: int) -> Tuple[int, Tuple[float, float]]:
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
    ring: GroupRing
    pitch: float = 1.0
    num_qubits: int = 0  # will be set in __post_init__

    def __post_init__(self):
        self.num_qubits = self.ring.l * self.ring.m * 2

    def qubit_id(self, a: int, b: int, c: int) -> int:
        a0, b0 = self.ring.canonical(a, b)
        return ((a0 * self.ring.m) + b0) * 2 + (c & 1)
    
    def id_to_tuple(self, qid: int) -> Tuple[int, int, int]:
        assert 0 <= qid < self.num_qubits, "Qubit ID out of range"
        a = (qid // 2) // self.ring.m
        b = (qid // 2) % self.ring.m
        c = qid % 2
        return (a, b, c)

    def coords(self, a: int, b: int, c: int) -> Tuple[float, float]:
        a0, b0 = self.ring.canonical(a, b)
        p = self.pitch
        x = a0 * p + (c & 1) * (p / 2)
        y = b0 * p + (1 - (c & 1)) * (p / 2)
        return x, y

    def id_and_coords_for(self, g: Monomial, c: int) -> Tuple[int, Tuple[float, float]]:
        i = self.qubit_id(g.a, g.b, c)
        return i, self.coords(g.a, g.b, c)
    
    @property
    def height(self) -> int:
        return self.ring.m  
    @property
    def width(self) -> int:
        return self.ring.l

class CoordMapEmbedding(Embedding):
    def __init__(self, xy_to_id: Dict[Tuple[int, int], int]):
        self._xy_to_id = dict(xy_to_id)
        self._id_to_xy = {qid: xy for xy, qid in self._xy_to_id.items()}
        self.num_qubits = len(self._id_to_xy)
        self._width = max(x for x, _ in self._xy_to_id.keys()) + 1 if self._xy_to_id else 0
        self._height = max(y for _, y in self._xy_to_id.keys()) + 1 if self._xy_to_id else 0

    def qubit_id(self, a: int, b: int, c: int) -> int:
        return self._xy_to_id[(a, b)]

    def id_to_tuple(self, qid: int) -> Tuple[int, int, int]:
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
