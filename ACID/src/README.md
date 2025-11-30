ACID Library — Module Overview
==============================

Core modules live under `acid/` and are installable as a Python package.
Example scripts are provided at the repository top-level under `examples/`.

Key Packages
------------

- `acid.base_code`: Base containers for device connectivity and stabiliser shapes.
- `acid.codes.*`: Builders for surface, colour, and BB/bb codes.
- `acid.defects.*`: Subsystem construction around dropouts (quasi‑stabilisers, products, gauges) and schedule orchestration.
- `acid.scheduling.*`: Local schedule enumeration, compatibility checks, and layer representation.
- `acid.solver.schedule_solver`: CP‑SAT model for global scheduling with product‑ordering constraints.
- `acid.analysis.*`: Schedule reporting and optional distance analysis (GAP/QDistRnd required for the latter).
- `acid.memory_experiment.*`: Stim circuit builder with detectors and observables for memory experiments.

Optional Tools
--------------

- Visualisation: `acid-tableau` (installed with the package; `stim` is a dependency)
- Distance via GAP: only needed for `acid.analysis.gauge_fix_nkd` and `StabiliserCode.nkd_via_gap`.
