#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from acid.codes.bb.builder_hexconn import build_code_from_spec, get_spec
from acid.defects.defective_code import DefectiveCode
from acid.memory_experiment.experiment import MemoryExperiment, MemoryExperimentConfig
from acid.stim_to_shatter_url import prompt_open_shatter


def run(
    solve_time: float = 30.0,
    n_dropped_qubits: int = 0,
    n_dropped_couplers: int = 0,
    rounds: int = 1,
    out: str | None = None,
    output_state_prep: bool = False,
    output_detectors_and_observables: bool = False,
    no_web_prompt: bool = False,
    drop_qubits: str | None = None,
    drop_couplers: str | None = None,
) -> None:
    # 1) Build BB bb288 with hex connectivity
    key = "bb288"
    spec = get_spec(key)
    base, embedding, connections = build_code_from_spec(spec)
    print(f"Built base code '{key}' (hex): n={base.num_qubits}, shapes={len(base.shapes)}")

    # 2) Select dropped qubits and couplers (explicit lists override random counts)
    uniq_edges = sorted({(min(u, v), max(u, v)) for (u, v, _cls) in connections})
    all_qubits = list(range(base.num_qubits))

    def parse_qubits(s: str | None) -> list[int]:
        if not s:
            return []
        parts = [p for p in re.split(r"[\s,]+", s.strip()) if p]
        out: list[int] = []
        for p in parts:
            out.append(int(p))
        return sorted({q for q in out})

    def parse_couplers(s: str | None) -> list[tuple[int, int]]:
        if not s:
            return []
        pairs = [p for p in re.split(r"[\s,]+", s.strip()) if p]
        out: list[tuple[int, int]] = []
        for token in pairs:
            if "-" not in token:
                raise ValueError(f"Invalid coupler token '{token}'. Use 'u-v'.")
            a_s, b_s = token.split("-", 1)
            a = int(a_s)
            b = int(b_s)
            u, v = (a, b) if a <= b else (b, a)
            out.append((u, v))
        return sorted({t for t in out})

    # Note: the CLI parsing for explicit lists is added below in main(); here we expect
    # to be called with either explicit lists parsed in or counts provided.
    # For this script, we derive explicit from environment via main's args.

    # Placeholders; main() will pass via closure by rebinding variables
    explicit_qubits = parse_qubits(drop_qubits)
    explicit_couplers = parse_couplers(drop_couplers)

    if explicit_qubits and n_dropped_qubits:
        raise SystemExit("Specify either --n-dropped-qubits or --drop-qubits, not both.")
    if explicit_couplers and n_dropped_couplers:
        raise SystemExit("Specify either --n-dropped-couplers or --drop-couplers, not both.")

    if explicit_qubits:
        for q in explicit_qubits:
            if q not in all_qubits:
                raise SystemExit(f"Dropped qubit {q} out of range [0..{base.num_qubits - 1}]")
        dropped_nodes: list[int] = explicit_qubits
    else:
        nQ = max(0, int(n_dropped_qubits))
        dropped_nodes = random.sample(all_qubits, min(nQ, len(all_qubits))) if nQ > 0 else []

    if explicit_couplers:
        valid = set(uniq_edges)
        for e in explicit_couplers:
            if e not in valid:
                raise SystemExit(f"Dropped coupler {e[0]}-{e[1]} not in device connectivity")
        dropped_edges: list[tuple[int, int]] = explicit_couplers
    else:
        nE = max(0, int(n_dropped_couplers))
        dropped_edges = random.sample(uniq_edges, min(nE, len(uniq_edges))) if nE > 0 else []
    print(f"Dropouts: nodes={dropped_nodes} edges={dropped_edges}")

    # 3) Defective code and stats
    dcode = DefectiveCode(base, dropped_nodes=dropped_nodes, dropped_edges=dropped_edges)
    stats = dcode.stats()
    print("Stats:")
    for k in sorted(stats.keys()):
        print(f"  {k}: {stats[k]}")

    # 4) Find smallest feasible L starting at 2
    print(f"Searching for feasible schedule starting at L=2 (solve_time={solve_time}s)")
    circuit = None
    L_found = None
    for L_try in range(2, 7):  # try L=2..6
        try:
            print(f"  Trying L={L_try} ...", flush=True)
            circuit = dcode.schedule(L=L_try, solve_time=solve_time)
            print(f"  Found feasible schedule at L={L_try}")
            L_found = L_try
            break
        except Exception as e:
            print(f"  L={L_try} failed: {e}", flush=True)
            continue
    if circuit is None:
        raise SystemExit("No feasible schedule found for L in [2..6]. Try increasing solve time.")
    counts = [len(layer.chosen) for layer in circuit.layers]
    print(f"Layered schedule chosen counts: {counts}")

    # 5) Emit a full memory experiment with R rounds (detectors, observables optional)
    exp = MemoryExperiment(
        dcode=dcode,
        circuit=circuit,
        embedding=embedding,
        cfg=MemoryExperimentConfig(R=int(rounds)),
    )
    stim_text = exp.build(
        include_x_detectors=bool(output_detectors_and_observables),
        include_z_detectors=bool(output_detectors_and_observables),
        include_state_prep=bool(output_state_prep),
        debug=False,
    )
    # BB circuits: set gate style for Shatter visualisation
    stim_text = "##! GATESTYLE DROOP=0 THICKNESS=1.5\n" + stim_text

    # 6) Write a single combined output with descriptive name
    if out is None:
        out_name = f"bb_288_hex_dropQ{n_dropped_qubits}_dropE{n_dropped_couplers}_L{L_found}_R{rounds}.stim"
        out_path = Path("visualisations") / out_name
    else:
        out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(stim_text)
    print(f"Wrote combined overlay+experiment to: {out_path}")
    if not no_web_prompt:
        prompt_open_shatter(str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description="BB bb288 (hex): compile and emit memory experiment")
    ap.add_argument("--solve-time", type=float, default=30.0)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", type=str)
    ap.add_argument(
        "--n-dropped-qubits",
        type=int,
        default=1,
        help="Number of randomly dropped qubits (mutually exclusive with --drop-qubits)",
    )
    ap.add_argument(
        "--n-dropped-couplers",
        type=int,
        default=0,
        help="Number of randomly dropped couplers (mutually exclusive with --drop-couplers)",
    )
    ap.add_argument(
        "--drop-qubits",
        type=str,
        help="Comma/space-separated list of qubit ids to drop (e.g. '1,2,5')",
    )
    ap.add_argument(
        "--drop-couplers",
        type=str,
        help="Comma/space-separated list of couplers 'u-v' to drop (e.g. '1-7,3-8')",
    )
    ap.add_argument("--output-state-prep", action="store_true")
    ap.add_argument("--output-detectors-and-observables", action="store_true")
    ap.add_argument(
        "--no-web-prompt",
        action="store_true",
        help="Do not prompt to open Shatter in a browser",
    )
    args = ap.parse_args()
    if args.output_detectors_and_observables and not args.output_state_prep:
        ap.error("--output-detectors-and-observables requires --output-state-prep")
    run(
        solve_time=args.solve_time,
        n_dropped_qubits=args.n_dropped_qubits,
        n_dropped_couplers=args.n_dropped_couplers,
        rounds=args.rounds,
        out=args.out,
        output_state_prep=args.output_state_prep,
        output_detectors_and_observables=args.output_detectors_and_observables,
        no_web_prompt=args.no_web_prompt,
        drop_qubits=args.drop_qubits,
        drop_couplers=args.drop_couplers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
