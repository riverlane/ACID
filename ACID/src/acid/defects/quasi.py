from dataclasses import dataclass


@dataclass(frozen=True)
class QuasiProduct:
    label: str
    pauli_type: str
    # Members are labels of quasi-stabilisers (strings)
    members: list[str]


@dataclass(frozen=True)
class QuasiStabiliser:
    """
    Represents a quasi-stabiliser, which is a stabiliser that may have been modified.

    Args:
        pauli_type (str): The type of Pauli operator ('X' or 'Z').
        support (frozenset[int]): The set of code-qubit ids that this quasi-stabiliser acts on.
        parent (object): The original stabiliser (or shape) that this quasi-stabiliser is
            derived from, carrying label/graph/map information.
        component_index (int): The index of this quasi-stabiliser within its
            parent stabiliser (0, 1, 2, ...).
        label (str): A unique label for this quasi-stabiliser.
    """

    pauli_type: str  # 'X' or 'Z'
    support: frozenset[int]  # code-qubit ids
    parent: object  # original stabiliser (or shape) carrying label/graph/map
    component_index: int  # 0,1,2,... within parent stabiliser
    label: str
