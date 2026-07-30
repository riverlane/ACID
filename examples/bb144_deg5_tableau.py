#!/usr/bin/env python3
from __future__ import annotations

import argparse

import stim
from acid.codes.bb import builder_deg5 as deg5
from acid.codes.bb.builder_hexconn import get_spec
from acid.defects.defective_code import DefectiveCode
from acid.tableau_visualiser.visualiser import TableauVisualiser


def run(layers: int = 5, solve_time: float = 60.0, ticks: int = 30) -> None:
    # 1) Build bb 144 code with degree-5 connectivity (no defects)
    spec = get_spec("bb144")
    base, _embedding, _ = deg5.build_code_from_spec(spec)
    print(f"Built code 'bb144' (deg5): n={base.num_qubits}, shapes={len(base.shapes)}")

    # 2) No dropouts
    dcode = DefectiveCode(base, dropped_nodes=[], dropped_edges=[])

    # 3) Compile a schedule
    print(f"Solving schedule: L={layers}, solve_time={solve_time}s ...")
    circuit = dcode.schedule(L=layers, solve_time=solve_time)

    # 4) Build a memory circuit (1 cycle) and prepare tableau bases
    circ_text = circuit.to_memory_stim(cycles=1)
    commuting = circuit.commuting_bases()
    anticommuting = circuit.anticommuting_bases()

    # 5) Visualise first `ticks` TICKs in the tableau
    vis = TableauVisualiser(
        stim.Circuit(circ_text),
        commuting_bases=commuting,
        anticommuting_bases=anticommuting,
    )
    snap = vis.snapshot()
    print(snap.to_ansi())
    stepped = 0
    while stepped < ticks and vis.step_to_next_tick():
        print()
        print(vis.snapshot().to_ansi())
        stepped += 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="bb 144 (deg5) tableau: print first ticks of compiled schedule"
    )
    ap.add_argument(
        "--layers", type=int, default=5, help="Schedule layers L (default: 5)"
    )
    ap.add_argument(
        "--solve-time",
        type=float,
        default=60.0,
        help="Solver time limit in seconds (default: 60)",
    )
    ap.add_argument(
        "--ticks", type=int, default=30, help="Number of TICKs to print (default: 30)"
    )
    args = ap.parse_args()
    run(layers=args.layers, solve_time=args.solve_time, ticks=args.ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
