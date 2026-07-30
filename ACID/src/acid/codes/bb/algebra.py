from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GroupRing:
    """Represents the group ring Z_l x Z_m, where elements are pairs (a, b) with
    a in Z_l and b in Z_m."""

    l: int
    m: int

    def canonical(self, a: int, b: int) -> tuple[int, int]:
        al = a % self.l
        bm = b % self.m
        return al, bm


@dataclass(frozen=True)
class Monomial:
    a: int
    b: int
    ring: GroupRing

    def __post_init__(self) -> None:
        ca, cb = self.ring.canonical(self.a, self.b)
        object.__setattr__(self, "a", ca)
        object.__setattr__(self, "b", cb)

    def __mul__(self, other: Monomial) -> Monomial:
        if self.ring != other.ring:
            raise ValueError("Mismatched group rings")
        return Monomial(self.a + other.a, self.b + other.b, self.ring)

    def inv(self) -> Monomial:
        return Monomial(-self.a, -self.b, self.ring)

    @classmethod
    def from_str(cls, s: str, ring: GroupRing) -> Monomial:
        t = s.strip()
        m = re.fullmatch(r"x\^(\d+)y\^(\d+)", t)
        if not m:
            raise ValueError(f"Invalid monomial string: {s!r}")
        a = int(m.group(1))
        b = int(m.group(2))
        return Monomial(a, b, ring)

    def as_string(self) -> str:
        return f"x^{self.a}y^{self.b}"

    def __str__(self) -> str:
        return self.as_string()

    def __repr__(self) -> str:
        return (
            f'Monomial.from_str("{self.as_string()}", '
            f"GroupRing({self.ring.l}, {self.ring.m}))"
        )

    def to_coordinate_LR_tuple(self, left_right: Literal["L", "R"]) -> tuple[int, int, int]:
        """Returns a tuple (a, b, side) where side is 0 for 'L' and 1 for 'R'."""
        if left_right not in ("L", "R"):
            raise ValueError("left_right must be 'L' or 'R'")
        return (self.a, self.b, 0 if left_right == "L" else 1)


@dataclass(frozen=True)
class Polynomial:
    terms: frozenset[Monomial]
    ring: GroupRing

    def __post_init__(self) -> None:
        # Ensure all terms are in canonical form and same ring; cancel duplicates (mod 2)
        seen = {}
        for t in self.terms:
            if t.ring != self.ring:
                raise ValueError("Term ring mismatch")
            key = (t.a, t.b)
            seen[key] = 1 ^ seen.get(key, 0)
        canonical = frozenset(
            Monomial(a, b, self.ring) for (a, b), v in seen.items() if v
        )
        object.__setattr__(self, "terms", canonical)

    @staticmethod
    def from_exponents(exps: Iterable[tuple[int, int]], ring: GroupRing) -> Polynomial:
        return Polynomial(frozenset(Monomial(a, b, ring) for a, b in exps), ring)

    @staticmethod
    def from_string(s: str, ring: GroupRing) -> Polynomial:
        t = s.strip().replace(" ", "")
        if t == "" or t == "0":
            return Polynomial(frozenset(), ring)
        parts = t.split("+")
        counts = {}
        for p in parts:
            m = Monomial.from_str(p, ring)
            key = (m.a, m.b)
            counts[key] = 1 ^ counts.get(key, 0)
        kept = frozenset(Monomial(a, b, ring) for (a, b), v in counts.items() if v)
        return Polynomial(kept, ring)

    def __iter__(self) -> Iterator[Monomial]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def add(self, other: Polynomial) -> Polynomial:
        if self.ring != other.ring:
            raise ValueError("Mismatched group rings")
        # Symmetric difference of term sets (mod 2)
        s = {(t.a, t.b) for t in self.terms}
        for t in other.terms:
            key = (t.a, t.b)
            if key in s:
                s.remove(key)
            else:
                s.add(key)
        return Polynomial(frozenset(Monomial(a, b, self.ring) for a, b in s), self.ring)

    def mul(self, other: Polynomial) -> Polynomial:
        if self.ring != other.ring:
            raise ValueError("Mismatched group rings")
        # Distribute and cancel even multiplicities (mod 2)
        counts = {}
        for t1 in self.terms:
            for t2 in other.terms:
                a = (t1.a + t2.a) % self.ring.l
                b = (t1.b + t2.b) % self.ring.m
                key = (a, b)
                counts[key] = 1 ^ counts.get(key, 0)
        return Polynomial(
            frozenset(Monomial(a, b, self.ring) for (a, b), v in counts.items() if v),
            self.ring,
        )

    def left_multiply(self, mono: Monomial) -> Polynomial:
        if mono.ring != self.ring:
            raise ValueError("Mismatched group rings")
        return Polynomial(
            frozenset(
                Monomial(mono.a + t.a, mono.b + t.b, self.ring) for t in self.terms
            ),
            self.ring,
        )

    def inverse(self) -> Polynomial:
        return Polynomial(
            frozenset(
                Monomial((-t.a) % self.ring.l, (-t.b) % self.ring.m, self.ring)
                for t in self.terms
            ),
            self.ring,
        )

    def as_string(self) -> str:
        if not self.terms:
            return "0"
        # Deterministic order: by a then b
        parts = [
            f"x^{t.a}y^{t.b}" for t in sorted(self.terms, key=lambda t: (t.a, t.b))
        ]
        return "+".join(parts)

    def __str__(self) -> str:
        return self.as_string()

    def __repr__(self) -> str:
        return (
            f'Polynomial.from_string("{self.as_string()}", '
            f"GroupRing({self.ring.l}, {self.ring.m}))"
        )
