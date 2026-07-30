## ACID Paper Source Code

This repo contains the core source code that implements ACID as described in my paper [link here].

### Installation

To simply use ACID, install via pip:

```bash
pip install ./ACID
```

### Running Examples

You can run the examples in /examples, for example by running:

```bash
python examples/surface_unrotated_grid.py --distance 7 --solve-time 30 --rounds 1 --n-dropped-qubits 1 --n-dropped-couplers 1
```

This will randomly select 1 qubit and 1 coupler to drop, and create a syndrome extraction circuit of minimal length as a .stim file with some extra markup, and open the result in my visualiser [Shatter](https://stasiu51.github.io/Shatter/), which is based on [Crumble](https://algassert.com/crumble). By default, the circuit produced will not contain a state preparation step, or detector or observable definitions: this is to reduce load when rendered in the visualiser. These can be re-enabled by passing `--output-state-prep` and `--output-detectors-and-observables` to the scripts.

All codes can be run in the defect-free case with `run_all_minimal.sh`. The outputs of this shell file are already available in /visualisations.

### Development

#### Install `uv` (Python Package Manager)

For development, this project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or use a package manager:
- **macOS**: `brew install uv`
- **Linux**: `pip install uv` or check [docs](https://docs.astral.sh/uv/getting-started/installation/)
- **Windows**: See [installation guide](https://docs.astral.sh/uv/getting-started/installation/)

#### Setup Development Environment

To install with dev tools (ruff, ty):

```bash
uv sync --extra dev
```

Or just install dependencies:

```bash
uv sync
```

To run examples during development:

```bash
uv run examples/surface_unrotated_grid.py --distance 7 --solve-time 30 --rounds 1 --n-dropped-qubits 1 --n-dropped-couplers 1
```
