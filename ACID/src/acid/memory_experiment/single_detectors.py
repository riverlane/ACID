from __future__ import annotations

from dataclasses import dataclass, field

from acid.defects.defective_code import DefectiveCode
from acid.pauli import PauliString
from acid.scheduling.types import SyndromeExtractionLayer

from .rec_log import MeasurementLog
from .schedule_index import ScheduleIndex


@dataclass
class DetectorInfo:
    id: int
    kind: str  # 'quasi' | 'product'
    label: str
    basis: str | None  # 'X' | 'Z' for quasi; None for product
    start: dict[str, object]
    end: dict[str, object]
    intervening: list[tuple[int, int]] = field(default_factory=list)
    recs: list[int] = field(default_factory=list)


@dataclass
class DetectorPlan:
    rec_sets: list[list[int]] = field(default_factory=list)
    infos: list[DetectorInfo] = field(default_factory=list)


def plan_quasi_detectors(
    *,
    dcode: DefectiveCode,
    layers: list[SyndromeExtractionLayer],
    sched: ScheduleIndex,
    log: MeasurementLog,
    R: int,
    include_x: bool = True,
    include_z: bool = True,
    filter_labels: set[str] | None = None,
) -> DetectorPlan:
    plan = DetectorPlan()
    det_id = 0
    L = len(layers)
    n = dcode.base_code.num_qubits

    for lab in dcode.quasi_labels:  # type: ignore[attr-defined]
        # dont include if filtered out or basis not included
        if filter_labels is not None and lab not in filter_labels:
            continue
        basis = sched.basis_of[lab]
        if (basis == "X" and not include_x) or (basis == "Z" and not include_z):
            continue
        # Mid-cycle PauliString for this quasi
        typ, supp = dcode.quasi_support(lab)
        if typ != basis:
            typ = basis
        Pmid = PauliString.from_supports(
            supp if basis == "X" else [], [] if basis == "X" else supp, n
        )

        # Contracting events across rounds
        events: list[tuple[str, object]] = []
        events.append(("init", None))
        rt_list = sched.rounds_for_label(lab, R)
        for r, t in rt_list:
            events.append(("contract", (r, t)))
        events.append(("final", None))

        for i in range(len(events) - 1):
            A = events[i]
            B = events[i + 1]
            # Anchors for event guard
            if sched.any_anticomm_measured_between_events(
                lab=lab, basis=basis, A=(A[0], A[1]), B=(B[0], B[1]), R=R
            ):
                continue

            # Sentinel (r,t) to enumerate intervening schedule layers
            a_rt: tuple[int, int] | None = (1, -1) if A[0] == "init" else A[1]  # type: ignore[assignment]
            b_rt: tuple[int, int] | None = (R, L) if B[0] == "final" else B[1]  # type: ignore[assignment]

            # Collect recs
            recs: list[int] = []
            intervening: list[tuple[int, int]] = []
            if A[0] == "init":
                rec_init = log.init_mpp[basis][lab]
                recs.append(int(rec_init))

            a_idx = (a_rt[0] - 1) * L + a_rt[1]
            b_idx = (b_rt[0] - 1) * L + b_rt[1]
            for idx in range(a_idx + 1, b_idx):
                r_i = idx // L + 1
                t_i = idx % L
                Pc = layers[t_i].propagate(Pmid)
                if basis == "X":
                    S = set(Pc.x_support())
                    key = (r_i, t_i, "X")
                else:
                    S = set(Pc.z_support())
                    key = (r_i, t_i, "Z")
                root_map = log.per_layer.get(key, {})
                for q in S:
                    if q in root_map:
                        recs.append(int(root_map[q]))
                if S:
                    intervening.append((r_i, t_i))

            if B[0] == "contract":
                r_b, t_b = b_rt  # type: ignore[misc]
                root_q = sched.label_to_root[t_b][lab]
                rec = log.per_layer[(r_b, t_b, basis)][root_q]
                recs.append(int(rec))
            elif B[0] == "final":
                rec = log.final_mpp[basis][lab]
                recs.append(int(rec))

            if not recs:
                continue

            plan.rec_sets.append(recs)
            info = DetectorInfo(
                id=det_id,
                kind="quasi",
                label=lab,
                basis=basis,
                start={
                    "type": A[0],
                    "round": a_rt[0] if a_rt else None,
                    "layer": a_rt[1] if a_rt else None,
                },
                end={
                    "type": B[0],
                    "round": b_rt[0] if b_rt else None,
                    "layer": b_rt[1] if b_rt else None,
                },
                intervening=intervening,
                recs=list(recs),
            )
            plan.infos.append(info)
            det_id += 1

    return plan
