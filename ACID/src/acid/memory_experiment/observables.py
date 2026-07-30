from __future__ import annotations

from acid.defects.defective_code import DefectiveCode
from acid.pauli import PauliString
from acid.scheduling.types import SyndromeExtractionLayer

from .rec_log import MeasurementLog
from .schedule_index import ScheduleIndex


def plan_observables(
    *,
    dcode: DefectiveCode,
    layers: list[SyndromeExtractionLayer],
    sched: ScheduleIndex,
    log: MeasurementLog,
    R: int,
    Lx_mid: list[PauliString],
    Lz_mid: list[PauliString],
    ancilla_recs_x: list[int],
    ancilla_recs_z: list[int],
    include_x: bool = True,
    include_z: bool = True,
) -> tuple[list[list[int]], list[list[int]]]:
    L = len(layers)

    obs_z: list[list[int]] = []
    if include_z:
        for i, anc_rec in enumerate(ancilla_recs_z):
            if i >= len(Lz_mid):
                break
            Pmid = Lz_mid[i]
            recs: list[int] = []
            for r in range(1, R + 1):
                for t in range(L):
                    Pc = layers[t].propagate(Pmid)
                    S = set(Pc.z_support())
                    root_map = log.per_layer.get((r, t, "Z"), {})
                    root_map_other = log.per_layer.get((r, t, "X"), {})
                    for q in S:
                        if q in root_map_other:
                            raise RuntimeError(
                                f"Observable-Z support overlaps X measurement at round={r}, layer={t}, qubit={q}"
                            )
                        if q in root_map:
                            recs.append(int(root_map[q]))
            recs.append(int(anc_rec))
            obs_z.append(recs)

    obs_x: list[list[int]] = []
    if include_x:
        for i, anc_rec in enumerate(ancilla_recs_x):
            if i >= len(Lx_mid):
                break
            Pmid = Lx_mid[i]
            recs: list[int] = []
            for r in range(1, R + 1):
                for t in range(L):
                    Pc = layers[t].propagate(Pmid)
                    S = set(Pc.x_support())
                    root_map = log.per_layer.get((r, t, "X"), {})
                    root_map_other = log.per_layer.get((r, t, "Z"), {})
                    for q in S:
                        if q in root_map_other:
                            raise RuntimeError(
                                f"Observable-X support overlaps Z measurement at round={r}, layer={t}, qubit={q}"
                            )
                        if q in root_map:
                            recs.append(int(root_map[q]))
            recs.append(int(anc_rec))
            obs_x.append(recs)

    return obs_z, obs_x
