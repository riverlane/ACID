from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from acid.codes.bb.algebra import GroupRing, Monomial, Polynomial
from acid.pauli import StabiliserCode

import numpy as np


@dataclass
class BBMidCycle:
    ring: GroupRing
    A: Polynomial
    B: Polynomial
    homomorphism_f_x: int
    homomorphism_f_y: int

    def __post_init__(self):
        self.l, self.m = self.ring.l, self.ring.m
        if not {self.homomorphism_f_x, self.homomorphism_f_y}.issubset({0, 1}):
            raise ValueError("homomorphism_f_x and homomorphism_f_y must be 0 or 1")
        match (self.homomorphism_f_x, self.homomorphism_f_y):
            case (1, 1):
                assert self.l % 2 == self.m % 2 == 0
            case (1, 0):
                assert self.l % 2 == 0
            case (0, 1):
                assert self.m % 2 == 0
            case (0, 0):
                assert False
        self.num_qubits = self.l * self.m * 2

    def even_odd_coords(self,a: int, b:int, _c: int) -> int:      
        return (a*self.homomorphism_f_x+b*self.homomorphism_f_y) % 2
    
    def even_odd_monomial(self,g: Monomial) -> int:
        return self.even_odd_coords(g.a,g.b,0)

    def stabilizers(self) -> Dict[Tuple[int, int, str], Set[Tuple[int, int, int]]]:
        out: Dict[Tuple[int, int, str], Set[Tuple[int, int, int]]] = {}
        for a in range(self.ring.l):
            for b in range(self.ring.m):
                g = Monomial(a, b, self.ring)
                X_support: Set[Tuple[int, int, int]] = set()
                for t in self.A:
                    h = t * g
                    X_support.add((h.a, h.b,0))
                for t in self.B:
                    h = t * g
                    X_support.add((h.a, h.b,1))
                out[(g.a, g.b, "X")] = X_support
                Z_support: Set[Tuple[int, int, int]] = set()
                for t in self.B:
                    h = t.inv() * g
                    Z_support.add((h.a, h.b,0))
                for t in self.A:
                    h = t.inv() * g
                    Z_support.add((h.a, h.b,1))
                out[(g.a, g.b, "Z")] = Z_support
        return out

    def midcycle_parity_check_matrix(self) -> StabiliserCode:
        n = self.num_qubits
        def qid(a: int, b: int, c: int) -> int:
            a0, b0 = self.ring.canonical(a, b)
            return ((a0 * self.ring.m) + b0) * 2 + (c & 1)
        stabs = self.stabilizers()
        x_keys = [(a, b, basis) for (a, b, basis) in stabs.keys() if basis == 'X']
        z_keys = [(a, b, basis) for (a, b, basis) in stabs.keys() if basis == 'Z']
        Hx: list[list[int]] = []
        Hz: list[list[int]] = []
        labels: list[str] = []
        for a, b, _ in sorted(x_keys):
            row = [0] * n
            for (aa, bb, cc) in stabs[(a, b, 'X')]:
                row[qid(aa, bb, cc)] ^= 1
            if any(row):
                Hx.append(row)
                labels.append(f"X({a},{b})")
        for a, b, _ in sorted(z_keys):
            row = [0] * n
            for (aa, bb, cc) in stabs[(a, b, 'Z')]:
                row[qid(aa, bb, cc)] ^= 1
            if any(row):
                Hz.append(row)
                labels.append(f"Z({a},{b})")
        return StabiliserCode(num_qubits=n, row_labels=labels, Hx=Hx, Hz=Hz)

