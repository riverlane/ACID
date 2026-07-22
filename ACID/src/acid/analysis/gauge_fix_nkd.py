from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from typing import List, Tuple, Optional

from acid.defects.defective_code import DefectiveCode
from acid.pauli import PauliString, StabiliserCode


@dataclass
class GaugeFixNKDResult:
    g: int
    combos_total: int
    eval_count: int
    timed_out: bool
    best_choice: Optional[List[int]]
    best_n: Optional[int]
    best_k: Optional[int]
    best_d: Optional[int]


def _basis_rows_to_Hx_Hz(
    rows: List[PauliString], n: int
) -> Tuple[List[List[int]], List[List[int]]]:
    Hx: List[List[int]] = []
    Hz: List[List[int]] = []
    for p in rows:
        if any(b & 1 for b in p.Z):
            # Treat as Z row
            Hz.append(p.Z[:])
        elif any(b & 1 for b in p.X):
            Hx.append(p.X[:])
        # Otherwise it's empty; skip
    return Hx, Hz


def _build_code_for_choice(
    dcode: DefectiveCode,
    choice_bits: List[int],
) -> StabiliserCode:
    """
    Build a CSS StabiliserCode by gauge-fixing:
      - include all untouched stabilisers
      - include all product stabilisers
      - for each gauge pair i, include either X or Z per choice_bits[i] (1->X, 0->Z)
    """
    n = dcode.base_code.num_qubits
    # Untouched + products
    base = dcode.midcycle_untouched_stabilisers()
    prod = dcode.midcycle_product_stabilisers()
    rows: List[PauliString] = []
    rows.extend(base.rows)
    if prod is not None:
        rows.extend(prod.rows)
    # Gauge picks
    Gx, Gz = dcode._gauge_pairs_mid_paulis()
    g = min(len(Gx), len(Gz))
    for i in range(g):
        if choice_bits[i] & 1:
            rows.append(Gx[i])
        else:
            rows.append(Gz[i])
    # Convert to Hx/Hz
    Hx_rows, Hz_rows = _basis_rows_to_Hx_Hz(rows, n)
    labels: List[str] = [f"r{i}" for i in range(len(Hx_rows) + len(Hz_rows))]
    return StabiliserCode(num_qubits=n, row_labels=labels, Hx=Hx_rows, Hz=Hz_rows)


def gauge_fixed_nkd(
    dcode: DefectiveCode,
    *,
    gap_exe: str = "gap",
    trials: int = 1000,
    mindist: int = 0,
    debug: int = 1,
    per_eval_timeout_s: int = 300,
    global_timeout_s: int = 600,
) -> GaugeFixNKDResult:
    """
    Enumerate gauge choices and compute (n,k,d) via GAP for the gauge-fixed code, tracking
    the lowest d found. Stops when all 2^g choices are explored or when global_timeout_s
    elapses. Each GAP evaluation is given per_eval_timeout_s. Returns the best-so-far,
    along with a timeout flag if enumeration was cut short.
    """
    Gx, Gz = dcode._gauge_pairs_mid_paulis()
    g = min(len(Gx), len(Gz))
    combos_total = 1 << g
    start = time.monotonic()
    best_d: Optional[int] = None
    best_n: Optional[int] = None
    best_k: Optional[int] = None
    best_choice: Optional[List[int]] = None
    eval_count = 0
    for bits in product([0, 1], repeat=g):
        # Check global timeout
        if (time.monotonic() - start) > float(global_timeout_s):
            return GaugeFixNKDResult(
                g=g,
                combos_total=combos_total,
                eval_count=eval_count,
                timed_out=True,
                best_choice=best_choice,
                best_n=best_n,
                best_k=best_k,
                best_d=best_d,
            )
        choice = list(bits)
        code = _build_code_for_choice(dcode, choice)
        try:
            n, k, d = code.nkd_via_gap(
                gap_exe=gap_exe,
                trials=int(trials),
                mindist=int(mindist),
                debug=int(debug),
                timeout=int(per_eval_timeout_s),
            )
        except Exception:
            # Treat evaluation failure as infinite distance; continue
            n, k, d = code.num_qubits, -1, -1
        eval_count += 1
        if best_d is None or (0 <= d < best_d) or (best_d < 0 and d >= 0):
            best_d = int(d)
            best_n = int(n)
            best_k = int(k)
            best_choice = choice
    # Completed all choices within time
    return GaugeFixNKDResult(
        g=g,
        combos_total=combos_total,
        eval_count=eval_count,
        timed_out=False,
        best_choice=best_choice,
        best_n=best_n,
        best_k=best_k,
        best_d=best_d,
    )
