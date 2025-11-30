#!/usr/bin/env bash
set -euo pipefail

# Minimal batch: R=1, no state prep, no detectors/observables, nq=0, nc=0.
# Assumes: `pip install ./ACID` has been run and `python` resolves the same environment.

SOLVE_TIME=${SOLVE_TIME:-60}
DIST=${DIST:-7}

echo "[run-all] Using SOLVE_TIME=${SOLVE_TIME} DIST=${DIST}"

python examples/bb_288_deg5.py --solve-time "${SOLVE_TIME}" --rounds 1 --n-dropped-qubits 0 --n-dropped-couplers 0 --no-web-prompt
python examples/bb_144_deg5.py --solve-time "${SOLVE_TIME}" --rounds 1 --n-dropped-qubits 0 --n-dropped-couplers 0 --no-web-prompt
python examples/bb_144_hex.py  --solve-time "${SOLVE_TIME}" --rounds 1 --n-dropped-qubits 0 --n-dropped-couplers 0 --no-web-prompt
python examples/bb_288_hex.py  --solve-time "${SOLVE_TIME}" --rounds 1 --n-dropped-qubits 0 --n-dropped-couplers 0 --no-web-prompt

python examples/surface_unrotated_grid.py --distance "${DIST}" --solve-time "${SOLVE_TIME}" --rounds 1 --n-dropped-qubits 0 --n-dropped-couplers 0 --no-web-prompt
python examples/surface_unrotated_hex.py  --distance "${DIST}" --solve-time "${SOLVE_TIME}" --rounds 1 --n-dropped-qubits 0 --n-dropped-couplers 0 --no-web-prompt

python examples/colour_hex_d3.py           --distance "${DIST}" --solve-time "${SOLVE_TIME}" --rounds 1 --n-dropped-qubits 0 --n-dropped-couplers 0 --no-web-prompt
python examples/colour_square_deg4.py      --distance "${DIST}" --solve-time "${SOLVE_TIME}" --rounds 1 --n-dropped-qubits 0 --n-dropped-couplers 0 --no-web-prompt
python examples/colour_square_superdense.py --distance "${DIST}" --solve-time "${SOLVE_TIME}" --rounds 1 --n-dropped-qubits 0 --n-dropped-couplers 0 --no-web-prompt

echo "[run-all] Done. Outputs under visualisations/."
