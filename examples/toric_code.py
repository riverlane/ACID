#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from acid.codes.toric.builder import build_toric_code
from acid.defects.defective_code import DefectiveCode
from acid.memory_experiment.builder import StimBuilder
from acid.memory_experiment.noise import NoNoiseModel
from acid.memory_experiment.rec_log import MeasurementLog
from acid.solver.swap import solve_swap_routing, verify_swap_routing
from acid.stim_to_shatter_url import prompt_open_shatter


def run(
    distance: int = 3,
    solve_time: float = 60.0,
    n_dropped_qubits: int = 0,
    n_dropped_couplers: int = 0,
    out: str | None = None,
    no_web_prompt: bool = False,
    drop_qubits: str | None = None,
    drop_couplers: str | None = None,
    swap_offset_str: str | None = None,
    swap_time_limit_s: float = 60.0,
) -> None:
    # Surface code on a square grid with square edges to the boundary (includes boundary singles)
    base, embedding, connections = build_toric_code(
        int(distance), int(distance), connectivity="grid"
    )
    print(f"Built toric code, d={distance}: n={base.num_qubits}, shapes={len(base.shapes)}")

    # Dropouts (explicit lists override random counts)
    uniq_edges = sorted({(min(u, v), max(u, v)) for (u, v) in base.connectivity_graph.edges()})
    all_qubits = list(range(base.num_qubits))

    def parse_qubits(s: str | None) -> list[int]:
        if not s:
            return []
        return sorted({int(p) for p in re.split(r"[\s,]+", s.strip()) if p})

    def parse_couplers(s: str | None) -> list[tuple[int, int]]:
        if not s:
            return []
        out: list[tuple[int, int]] = []
        for token in [p for p in re.split(r"[\s,]+", s.strip()) if p]:
            if "-" not in token:
                raise ValueError(f"Invalid coupler token '{token}'. Use 'u-v'.")
            a_s, b_s = token.split("-", 1)
            a = int(a_s)
            b = int(b_s)
            u, v = (a, b) if a <= b else (b, a)
            out.append((u, v))
        return sorted({t for t in out})

    explicit_qubits = parse_qubits(drop_qubits)
    explicit_couplers = parse_couplers(drop_couplers)
    if swap_offset_str is None:
        swap_offset = (0, 0)
    else:
        swap_offset_parts = swap_offset_str.split(",")
        if len(swap_offset_parts) != 2 or not all(
            re.fullmatch(r"-?\d+", p) for p in swap_offset_parts
        ):
            raise SystemExit(
                f"Invalid --swap-offset '{swap_offset_str}'. Must be two integers separated by a comma."
            )
        swap_offset = tuple(int(x) for x in swap_offset_parts)
        assert len(swap_offset) == 2, "swap_offset must be a tuple of two integers"
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

    original_dcode = DefectiveCode(base, dropped_nodes=dropped_nodes, dropped_edges=dropped_edges)
    print("Stats:")
    for k in original_dcode.stats().keys():
        print(f"  {k}: {original_dcode.stats()[k]}")

    # Find smallest feasible L
    original_circuit = None
    L_found = None
    print(f"Searching feasible L in [2..6] (solve_time={solve_time}s)")
    for L_try in range(2, 7):
        try:
            print(f"  Trying L={L_try} ...", flush=True)
            original_circuit = original_dcode.schedule(L=L_try, solve_time=solve_time)
            L_found = L_try
            print(f"  Found L={L_try}")
            break
        except Exception as e:
            print(f"  L={L_try} failed: {e}")
    if original_circuit is None:
        raise SystemExit("No feasible schedule found for L in [2..6].")

    for i, layer in enumerate(original_circuit.layers):
        print(len(layer.x_roots) + len(layer.z_roots), f"root qubits in layer {i}")

    da, db = swap_offset

    virtual_dead_qubits = embedding.get_shifted_positions(list(dropped_nodes), -da, -db)
    virtual_dead_couplers = {
        (min(u, v), max(u, v)) for u, v in embedding.get_shifted_connections(dropped_edges, -da, -db)
    }

    total_dead_qubits = set(dropped_nodes) | set(virtual_dead_qubits)
    total_dead_couplers = {(min(u, v), max(u, v)) for u, v in dropped_edges} | virtual_dead_couplers
    virtual_dcode = DefectiveCode(
        base,
        dropped_nodes=total_dead_qubits,
        dropped_edges=total_dead_couplers,
    )

    if virtual_dcode.k != original_dcode.k:
        raise SystemExit(
            f"Virtual defective code has different k ({virtual_dcode.k}) than "
            f"original ({original_dcode.k}). This is a failure of the protocol. "
            "This means there are too many defects in the original code for the "
            "given swap offset."
        )

    # Find smallest feasible L
    virtual_circuit = None
    L_found = None
    print(f"Searching feasible L in [2..6] (solve_time={solve_time}s)")
    for L_try in range(2, 7):
        try:
            print(f"  Trying L={L_try} ...", flush=True)
            virtual_circuit = virtual_dcode.schedule(L=L_try, solve_time=solve_time)
            L_found = L_try
            print(f"  Found L={L_try}")
            break
        except Exception as e:
            print(f"  L={L_try} failed: {e}")
    if virtual_circuit is None:
        raise SystemExit("No feasible schedule found for L in [2..6].")

    for i, layer in enumerate(virtual_circuit.layers):
        print(len(layer.x_roots) + len(layer.z_roots), f"root qubits in layer {i}")

    # Swap routing: pick the layer with the most root qubits as dont_care,
    # then route the remaining live qubits to their shifted positions.
    best_layer = max(virtual_circuit.layers, key=lambda lay: len(lay.x_roots) + len(lay.z_roots))
    best_idx = virtual_circuit.layers.index(best_layer)
    dont_care = best_layer.x_roots | best_layer.z_roots
    print(
        f"Swap routing: using layer {best_idx} ({len(dont_care)} root qubits as dont_care), "
        f"offset={swap_offset}"
    )
    conns_2d = [(u, v) for u, v, _ in connections]
    dead_pos = total_dead_qubits
    print(dont_care)
    swap_layers = solve_swap_routing(
        embedding=embedding,
        connections=conns_2d,
        dead_positions=set(dropped_nodes),
        dead_connections=set(dropped_edges),
        offset=swap_offset,
        root_qubits=dont_care | set(virtual_dead_qubits),
        # the virutal dead qubits are treated
        # as dont_care as this is performed on the virtual code, where they are treated as dead but
        # are physically alive in the original code, so we need to treat them as dont_care in the
        # swap routing
        time_limit_s=swap_time_limit_s,
    )
    if swap_layers is None:
        print("  No swap routing solution found within limits.")
    else:
        print(f"  Found {len(swap_layers)} swap layer(s):")
        for t, sl in enumerate(swap_layers):
            print(f"    Layer {t}: {len(sl)} swap(s)")
        # Build the same target the solver used, then verify
        live = set(range(embedding.num_qubits)) - dead_pos - dont_care
        shifted = embedding.get_shifted_positions(sorted(live), *swap_offset)
        swap_target = {
            dest: q
            for q, dest in zip(sorted(live), shifted)
            if dest not in dead_pos and dest not in dont_care
        }
        ok = verify_swap_routing(swap_layers, embedding.num_qubits, swap_target)
        print(f"  Verification: {'PASS' if ok else 'FAIL'}")

    if swap_layers is None:
        print("Cannot build swap circuit: no swap routing solution found.")
        return

    # Map each qubit to its shifted position directly from the embedding.
    da, db = swap_offset
    N = embedding.num_qubits
    new_pos = embedding.get_shifted_positions(list(range(N)), da, db)

    # Build stim circuit: first half → measure/reset roots → swaps → shifted second half
    overlay = original_dcode.visualisation_stim(embedding)
    overlay_lines = overlay.rstrip().splitlines()
    overlay_tick_count = sum(1 for ln in overlay_lines if ln.strip() == "TICK")
    sb = StimBuilder(lines=list(overlay_lines))
    sb._tick_count = overlay_tick_count  # type: ignore[attr-defined]
    noise = NoNoiseModel()
    log = MeasurementLog()

    sb.memory_rounds(1, circuit=original_circuit, noise=noise, log=log)

    # First half: contract
    sb.layer_contract(best_layer, noise=noise)

    # Measure roots before the swap
    sb.layer_measure(best_layer, noise=noise)

    # Swap layers
    sb.swap_routing_layers(swap_layers, noise=noise)

    sb.set_offset(new_pos)  # remap qubit IDs to their shifted positions

    # Reset roots at their shifted positions, then expand at shifted positions
    sb.layer_reset(best_layer, noise=noise)

    # Second half: expand with qubit IDs remapped to shifted positions
    sb.layer_expand(best_layer, noise=noise)

    stim_text = "\n".join(sb.lines) + "\n"

    # Output
    if out is None:
        code_type = "toric"
        da, db = swap_offset
        out_name = (
            f"{code_type}_d{distance}_dropQ{n_dropped_qubits}_dropE"
            f"{n_dropped_couplers}_L{L_found}_swap{da}_{db}.stim"
        )
        out_path = Path("visualisations") / out_name
    else:
        out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(stim_text)
    print(f"Wrote combined overlay+experiment to: {out_path}")
    if not no_web_prompt:
        prompt_open_shatter(str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unrotated surface (grid): compile and emit memory experiment"
    )
    ap.add_argument("--distance", type=int, default=7)
    ap.add_argument("--solve-time", type=float, default=60.0)
    ap.add_argument("--out", type=str)
    ap.add_argument(
        "--n-dropped-qubits",
        type=int,
        default=0,
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
    ap.add_argument(
        "--swap-offset",
        type=str,
        default="1,0",
        help="Lattice offset for swap routing as 'da,db' (default: '1,0')",
    )
    ap.add_argument(
        "--swap-time-limit",
        type=float,
        default=60.0,
        help="CP-SAT time limit in seconds for swap routing (default: 60)",
    )
    ap.add_argument(
        "--no-web-prompt",
        action="store_true",
        help="Do not prompt to open Shatter in a browser",
    )
    args = ap.parse_args()
    run(
        distance=args.distance,
        solve_time=args.solve_time,
        n_dropped_qubits=args.n_dropped_qubits,
        n_dropped_couplers=args.n_dropped_couplers,
        out=args.out,
        no_web_prompt=args.no_web_prompt,
        drop_qubits=args.drop_qubits,
        drop_couplers=args.drop_couplers,
        swap_offset_str=args.swap_offset,
        swap_time_limit_s=args.swap_time_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
