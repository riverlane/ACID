from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence


def _validate_binary_matrix(M: Sequence[Sequence[int]]) -> None:
    rows = len(M)
    if rows == 0:
        return
    cols = len(M[0])
    for r, row in enumerate(M):
        if len(row) != cols:
            raise ValueError("All rows must have the same number of columns")
        for v in row:
            if v not in (0, 1):
                raise ValueError("Matrix entries must be 0/1 for GF(2)")


def _gap_matrix_literal(M: Sequence[Sequence[int]]) -> str:
    # One(F)*[[...], [...]] ensures elements are in GF(2)
    if not M:
        return "One(F)*[]"
    rows = ["[" + ",".join(str(int(v)) for v in row) + "]" for row in M]
    return "One(F)*[" + ",".join(rows) + "]"


def compute_nkd_with_gap(
    Hx: Sequence[Sequence[int]],
    Hz: Sequence[Sequence[int]],
    *,
    gap_exe: str = "gap",
    trials: int = 1000,
    mindist: int = 0,
    debug: int = 1,
    timeout: int = 300,
) -> tuple[int, int, int]:
    """
    Compute (n, k, d) using GAP + QDistRnd package.

    - Hx, Hz: binary matrices over GF(2) as lists of lists.
    - gap_exe: path to GAP executable in your PATH (e.g., 'gap' or '/usr/local/bin/gap').
      GAP must be able to LoadPackage("QDistRnd") via its configured package paths.
    - trials, mindist, debug: parameters passed to DistRandCSS.
    - timeout: seconds to allow the GAP process to run.
    """
    if shutil.which(gap_exe) is None:
        raise FileNotFoundError(f"GAP executable not found: {gap_exe}")

    _validate_binary_matrix(Hx)
    _validate_binary_matrix(Hz)

    gap_lines: list[str] = []
    gap_lines.append("F := GF(2);")
    gap_lines.append(
        'if not LoadPackage("QDistRnd") then Error("QDistRnd package not found"); fi;'
    )
    gap_lines.append(f"Hx := {_gap_matrix_literal(Hx)};")
    gap_lines.append(f"Hz := {_gap_matrix_literal(Hz)};")
    # Robust column count even when one side is empty
    gap_lines.append("n := Maximum(NrCols(Hx), NrCols(Hz));")
    gap_lines.append("k := n - RankMat(Hx) - RankMat(Hz);")
    gap_lines.append(
        f"d := DistRandCSS(Hz, Hx, {int(trials)}, {int(mindist)}, {int(debug)} : field := F);"
    )
    # Print with explicit sentinels to simplify parsing
    # Emit sentinels without embedding literal newlines inside a GAP string
    gap_lines.append('Print("N=");')
    gap_lines.append("Print(n);")
    gap_lines.append('Print("\\n");')
    gap_lines.append('Print("K=");')
    gap_lines.append("Print(k);")
    gap_lines.append('Print("\\n");')
    gap_lines.append('Print("D=");')
    gap_lines.append("Print(d);")
    gap_lines.append('Print("\\n");')
    gap_lines.append("QUIT;")

    script = "\n".join(gap_lines)
    # Run GAP without autoloaded packages or init files to avoid Browse/ncurses issues.
    # Flags:
    #   -q : quiet
    #   -A : do not autoload packages
    #   -r : do not read init files (e.g., .gaprc)
    proc = subprocess.run(
        [gap_exe, "-q", "-A", "-r"],
        input=script.encode(),
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GAP failed: {proc.stderr.decode(errors='ignore')}")
    out_lines = proc.stdout.decode(errors="ignore").splitlines()

    # First try to parse explicit sentinel lines
    n = k = d = None
    for ln in out_lines:
        s = ln.strip()
        if s.startswith("N="):
            try:
                n = int(s[2:])
            except ValueError:
                pass
        elif s.startswith("K="):
            try:
                k = int(s[2:])
            except ValueError:
                pass
        elif s.startswith("D="):
            try:
                d = int(s[2:])
            except ValueError:
                pass
    if n is not None and k is not None and d is not None:
        return n, k, d

    # Fallback: look for a summary like '[[n,k,d]]'
    summary = None
    for ln in reversed(out_lines):
        m = re.search(r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(-?\d+)\s*\]\]", ln)
        if m:
            summary = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            break
    if summary is not None:
        return summary

    # As a last resort, try to extract the last integer as d, with n,k from ranks
    ints = [int(x) for x in re.findall(r"-?\d+", "\n".join(out_lines))]
    if ints:
        d = ints[-1]
        # Compute n,k directly from inputs
        nn = len(Hx[0]) if Hx else (len(Hz[0]) if Hz else 0)
        # We don't have rank over GF(2) here; let caller ignore k if needed
        return nn, -1, d

    raise RuntimeError("GAP output did not contain parsable NKD results")
    return n, k, d
