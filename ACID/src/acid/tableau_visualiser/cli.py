from __future__ import annotations

import argparse
import json
from pathlib import Path

import stim

from acid.pauli import CommutingPauliBasis, PauliString

from .basis import default_single_qubit_basis
from .visualiser import TableauVisualiser


def load_commuting_bases(path: Path, n: int) -> list[CommutingPauliBasis]:
    """Load commuting bases from a JSON file.

    Supported format:
    [
      {"name":"...","kind":"stabiliser","priority":int,"rows":[[0/1,... length 2n], ...]},
      ...
    ]
    Rows are 2n binary (X|Z). Non-'stabiliser' kinds are ignored.
    """
    data = json.loads(path.read_text())
    out: list[CommutingPauliBasis] = []
    for obj in data:
        if str(obj.get("kind", "stabiliser")).lower() != "stabiliser":
            continue
        name = str(obj.get("name", "User Basis"))
        prio = int(obj.get("priority", 0))
        rows = [PauliString.from_2n(list(map(int, r))) for r in obj.get("rows", [])]
        out.append(CommutingPauliBasis(name=name, priority=prio, rows=rows))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tableau visualiser for stim circuits (commuting bases)"
    )
    ap.add_argument("stim_file", type=str, help="Path to .stim circuit")
    ap.add_argument(
        "--bases",
        type=str,
        help="Optional JSON file defining additional commuting bases",
    )
    ap.add_argument(
        "--ticks", type=int, default=0, help="How many TICKs to step (0=all)"
    )
    args = ap.parse_args()

    circ = stim.Circuit(Path(args.stim_file).read_text())

    # Instantiate once to detect n
    vis = TableauVisualiser(circ, commuting_bases=[], anticommuting_bases={})
    n = vis.n

    commuting: list[CommutingPauliBasis] = []
    # Default single-qubit commuting basis
    default_basis_2n = default_single_qubit_basis(n)
    default_rows = [PauliString.from_2n(r) for r in default_basis_2n.rows]
    commuting.append(
        CommutingPauliBasis(
            name=default_basis_2n.name,
            priority=default_basis_2n.priority,
            rows=default_rows,
        )
    )

    if args.bases:
        commuting.extend(load_commuting_bases(Path(args.bases), n))

    # Recreate with bases
    vis = TableauVisualiser(circ, commuting_bases=commuting, anticommuting_bases={})

    # Initial snapshot (tick 0)
    snap = vis.snapshot()
    print(snap.to_ansi())

    steps = args.ticks
    count = 0
    if steps <= 0:
        # step through all ticks
        while vis.step_to_next_tick():
            snap = vis.snapshot()
            print()
            print(snap.to_ansi())
            count += 1
    else:
        for _ in range(steps):
            if not vis.step_to_next_tick():
                break
            snap = vis.snapshot()
            print()
            print(snap.to_ansi())
            count += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
