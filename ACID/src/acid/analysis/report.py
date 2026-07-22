from __future__ import annotations

from pathlib import Path
from typing import List

from acid.defects.defective_code import DefectiveCode
from acid.analysis.schedule import analyze_layers
from acid.defects.syndrome_extraction_circuit import SyndromeExtractionCircuit


def write_schedule_report(
    out_path: Path, dcode: DefectiveCode, Ls: List[int], *, solve_time: float = 60.0
) -> None:
    """
    Generic report writer for any DefectiveCode + schedule lengths.
    Includes stats, per-layer measured/in-process/completed, and product completions.
    """
    lines: List[str] = []
    lines.append("# Schedule Report\n")
    stats = dcode.stats()
    lines.append(
        f"- Quasis: {stats.get('num_quasi')} nontrivial={stats.get('num_nontrivial')} rank={stats.get('rank')}\n"
    )
    products = dcode.products_list()
    prod_members = {p.label: set(p.members) for p in products}
    interesting = set(dcode.anticommutation_graph().nodes())
    for L in Ls:
        lines.append(f"\n## L={L}\n")
        try:
            circuit = dcode.schedule(L, solve_time=solve_time)
            layers = circuit.layers
            result = analyze_layers(
                prod_members, layers, interesting_labels=interesting
            )
            # Per-layer table
            lines.append(
                "\n| Layer | Measured | In-process | Completed |\n|------:|----------|------------|-----------|\n"
            )
            for t, rec in enumerate(result["per_layer"]):
                m = ",".join(rec["measured"]) if rec["measured"] else "-"
                ip = ",".join(rec["in_process"]) if rec["in_process"] else "-"
                cp = ",".join(rec["completed"]) if rec["completed"] else "-"
                lines.append(f"| {t} | {m} | {ip} | {cp} |\n")
            # Product completions
            lines.append("\nCompletions:\n")
            for p, compl in sorted(result["product_completions"].items()):
                lines.append(f"- {p}: {sorted(compl)}\n")
        except Exception as e:
            lines.append(f"- Solve failed: {e}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def write_schedule_report_from_circuit(
    out_path: Path, dcode: DefectiveCode, circuit: SyndromeExtractionCircuit
) -> None:
    products = dcode.products_list()
    prod_members = {p.label: set(p.members) for p in products}
    interesting = set(dcode.anticommutation_graph().nodes())
    lines = []
    layers = circuit.layers
    result = analyze_layers(prod_members, layers, interesting_labels=interesting)
    # Per-layer table
    lines.append(
        "\n| Layer | Measured | In-process | Completed |\n|------:|----------|------------|-----------|\n"
    )
    for t, rec in enumerate(result["per_layer"]):
        m = ",".join(rec["measured"]) if rec["measured"] else "-"
        ip = ",".join(rec["in_process"]) if rec["in_process"] else "-"
        cp = ",".join(rec["completed"]) if rec["completed"] else "-"
        lines.append(f"| {t} | {m} | {ip} | {cp} |\n")
    # Product completions
    lines.append("\nCompletions:\n")
    for p, compl in sorted(result["product_completions"].items()):
        lines.append(f"- {p}: {sorted(compl)}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
