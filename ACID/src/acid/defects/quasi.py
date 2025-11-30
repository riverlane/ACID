from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class QuasiProduct:
    label: str
    pauli_type: str
    # Members are labels of quasi-stabilisers (strings)
    members: List[str]

@dataclass(frozen=True)
class QuasiStabiliser:
    pauli_type: str           # 'X' or 'Z'
    support: frozenset[int]   # code-qubit ids
    parent: object            # original stabiliser (or shape) carrying label/graph/map
    component_index: int      # 0,1,2,... within parent stabiliser
    label: str
