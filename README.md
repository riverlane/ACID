## ACID Paper Source Code

This repo contains the core source code that implements ACID as described in my paper [link here].

The core source code is provided as a python package. To install it into your active python environment, run:

`pip install ./ACID`

You can then run the examples in /examples, for example by running:

`python examples/surface_unrotated_grid.py --distance 7 --solve-time 30 --rounds 1 --n-dropped-qubits 1 --n-dropped-couplers 1`

which will randomly select 1 qubit and 1 coupler to drop, and create a syndrome extraction circuit of minimal length as a .stim file with some extra markup, and open the result in my visualiser [Shatter](https://stasiu51.github.io/Shatter/), which is based on [Crumble](https://algassert.com/crumble). By default, the circuit produced will not contain state preparation, or detector or observable definitions: this is to reduce load when rendered in the visualiser. These can be re-enabled by passing `--output-state-prep` and `--output-detectors-and-observables` to the scripts.

All codes can be run in the defect-free case with run_all_minimal.sh. The outputs of this shell file are already available in /visualisations.