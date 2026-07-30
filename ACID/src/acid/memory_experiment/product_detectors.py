from __future__ import annotations

from dataclasses import dataclass

from acid.analysis.schedule import analyze_layers
from acid.defects.defective_code import DefectiveCode
from acid.pauli import PauliString
from acid.scheduling.types import SyndromeExtractionLayer

from .rec_log import MeasurementLog
from .schedule_index import ScheduleIndex
from .single_detectors import DetectorInfo, DetectorPlan


def _lin_idx(L: int, r: int, t: int) -> int:
    # Flatten (round, layer) onto a single timeline index.
    return (int(r) - 1) * int(L) + int(t)


def plan_product_detectors(
    *,
    dcode: DefectiveCode,
    layers: list[SyndromeExtractionLayer],
    sched: ScheduleIndex,
    log: MeasurementLog,
    R: int,
    include_x: bool = True,
    include_z: bool = True,
    debug: bool = False,
) -> DetectorPlan:

    # L: layers per round, used for linearizing schedule coordinates.
    L = len(layers)
    plan = DetectorPlan()

    products = [
        p
        for p in dcode.products_list()
        if len(p.members) > 1
        and ((p.pauli_type == "X" and include_x) or (p.pauli_type == "Z" and include_z))
    ]
    if not products:
        return plan

    # Precompute per-product completion layers within a round.
    prod_plan = build_product_plan(dcode, layers)

    # Precompute mid-cycle PauliStrings for all members by basis
    # We'll build per-product cache on the fly

    next_id = 0
    for current_product in products:
        basis = current_product.pauli_type
        members = list(current_product.members)
        # Build completion sequence B_i:
        # B0 = init_mpp, intermediate B_i = per-round completion events,
        # BN = final_mpp. Each contract event carries member latest times c_q
        # and A_i (earliest among those latest times).
        sorted_completion_times = sorted(
            prod_plan.completions.get(current_product.label, [])
        )
        completions: list[
            tuple[
                str,  # kind
                tuple[int, int] | None,  # (round, layer) product completion time
                dict[
                    str, tuple[int, int]
                ],  # member -> most recent (round, layer) contraction time
                tuple[int, int] | None,  # earliest among those most recent contraction times
            ]
        ] = []
        completions.append(("init_mpp", None, {}, None))
        for round_number in range(1, R + 1):
            for completion_time in sorted_completion_times:
                # we want the last time each member was measured in this round, up to the
                # completion time.
                member_to_lastest_time_measured: dict[str, tuple[int, int]] = {}
                for product_member in members:
                    ts_all = sched.contracting_ts.get(product_member, [])
                    ts_le = [t for t in ts_all if t <= completion_time]
                    if not ts_le:
                        raise RuntimeError(
                            f"Product completion invariant violated: member {product_member} not "
                            f"measured by t*={completion_time} in round {round_number} for product "
                            f"{current_product.label}"
                        )
                    latest_measurement_time = max(ts_le)
                    member_to_lastest_time_measured[product_member] = (
                        round_number,
                        latest_measurement_time,
                    )
                earliest_member_time = min(
                    member_to_lastest_time_measured.values(),
                    key=lambda rt: _lin_idx(L, rt[0], rt[1]),
                )
                completions.append(
                    (
                        "contract",
                        (round_number, completion_time),
                        member_to_lastest_time_measured,
                        earliest_member_time,
                    )
                )
        completions.append(("final_mpp", None, {}, None))

        if debug:
            try:
                print(
                    f"[product-debug] Product {current_product.label} basis={basis} members={len(members)}"
                )
                for idx, (
                    kind,
                    B_rt,
                    member_to_lastest_time_measured,
                    earliest_member_time,
                ) in enumerate(completions):
                    if kind == "contract":
                        print(
                            f"  - B[{idx}] kind=contract B_rt={B_rt} A_rt={earliest_member_time} c_q={{"
                            + ", ".join(
                                f"{m}:{rt}"
                                for m, rt in member_to_lastest_time_measured.items()
                            )
                            + "}}"
                        )
                    else:
                        print(f"  - B[{idx}] kind={kind}")
            except Exception:
                pass

        # Helper: active support at a layer by XOR of supports of active members
        def active_support_at(
            r: int,
            t: int,
            cqi: dict[str, tuple[int, int]],
            cqi1: dict[str, tuple[int, int]],
        ) -> set[int]:
            idx = _lin_idx(L, r, t)
            acc: set[int] = set()
            for m in members:
                # previous/next contraction indices
                prev_rt = cqi.get(m)
                next_rt = cqi1.get(m)
                prev_idx = (
                    -1 if prev_rt is None else _lin_idx(L, prev_rt[0], prev_rt[1])
                )
                next_idx = (
                    R * L if next_rt is None else _lin_idx(L, next_rt[0], next_rt[1])
                )
                if prev_idx < idx <= next_idx:
                    # propagate member mid-cycle Pauli
                    typ, supp = dcode.quasi_support(m)
                    n = dcode.base_code.num_qubits
                    Pmid = PauliString.from_supports(
                        supp if basis == "X" else [], [] if basis == "X" else supp, n
                    )
                    Pc = layers[t].propagate(Pmid)
                    S = set(Pc.x_support() if basis == "X" else Pc.z_support())
                    # XOR with accumulator
                    if not acc:
                        acc = set(S)
                    else:
                        # symmetric difference
                        if len(S) > len(acc):
                            S, acc = acc, S
                        for q in S:
                            if q in acc:
                                acc.remove(q)
                            else:
                                acc.add(q)
            return acc

        # Synthesize detectors per adjacent completion pair
        for i in range(len(completions) - 1):
            kind_a, B_rt_a, latest_a, A_rt_a = completions[i]
            kind_b, B_rt_b, latest_b, A_rt_b = completions[i + 1]

            # Detector window is (A_i, B_{i+1}] in linearized schedule time.
            A_lin = -1 if A_rt_a is None else _lin_idx(L, A_rt_a[0], A_rt_a[1])
            B_next_lin = R * L if B_rt_b is None else _lin_idx(L, B_rt_b[0], B_rt_b[1])

            # c_q maps member -> latest contraction time at B_i/B_{i+1}.
            cqi = latest_a  # may be empty for init
            cqi1 = latest_b  # may be empty for final

            recs: list[int] = []
            intervening_layers: list[tuple[int, int]] = []

            if debug:
                try:
                    print(
                        f"    [window {i}] start={{kind:{kind_a},A_rt:{A_rt_a},B_rt:{B_rt_a}}} -> end={{kind:{kind_b},B_rt:{B_rt_b}}}"
                    )
                    if latest_a:
                        print(
                            "      c_q_i   = {"
                            + ", ".join(f"{m}:{rt}" for m, rt in latest_a.items())
                            + "}"
                        )
                    if latest_b:
                        print(
                            "      c_q_ip1 = {"
                            + ", ".join(f"{m}:{rt}" for m, rt in latest_b.items())
                            + "}"
                        )
                except Exception:
                    pass

            # (1) First window starts from an init MPP parity snapshot.
            if i == 0:
                for product_member in members:
                    rec0 = log.init_mpp[basis][product_member]
                    recs.append(int(rec0))

            # (2) For each layer in the window, add same-basis root recs on the
            # active propagated product support.
            layer_summaries: list[
                tuple[int, int, int, int]
            ] = []  # (r,t, |S|, added_recs)
            for idx in range(A_lin + 1, min(B_next_lin, R * L) + 1):
                if idx >= R * L:
                    break
                r_i = idx // L + 1
                t_i = idx % L
                S = active_support_at(r_i, t_i, cqi, cqi1)
                root_same = log.per_layer.get((r_i, t_i, basis), {})
                root_other = log.per_layer.get(
                    (r_i, t_i, "Z" if basis == "X" else "X"), {}
                )
                # Opposite-basis overlap indicates an invalid detector composition.
                if any((q in root_other) for q in S):
                    bad = [q for q in S if q in root_other]
                    raise RuntimeError(
                        f"Product {current_product.label} support overlaps opposite-basis roots at r={r_i}, t={t_i}, qubits={bad}"
                    )
                added = 0
                for q in S:
                    if q in root_same:
                        recs.append(int(root_same[q]))
                        added += 1
                layer_summaries.append((r_i, t_i, len(S), added))
                if S:
                    intervening_layers.append((r_i, t_i))

            # (3) Last window closes with final MPP parity snapshot.
            if kind_b == "final_mpp":
                for product_member in members:
                    recf = log.final_mpp[basis][product_member]
                    recs.append(int(recf))

            if not recs:
                continue

            info = DetectorInfo(
                id=next_id,
                kind="product",
                label=current_product.label,
                basis=None,
                start={
                    "type": kind_a,
                    "round": A_rt_a[0] if A_rt_a else None,
                    "layer": A_rt_a[1] if A_rt_a else None,
                },
                end={
                    "type": kind_b,
                    "round": B_rt_b[0] if B_rt_b else None,
                    "layer": B_rt_b[1] if B_rt_b else None,
                },
                intervening=intervening_layers,
                recs=list(recs),
            )
            plan.rec_sets.append(recs)
            plan.infos.append(info)
            next_id += 1

            if debug:
                try:
                    print("      layers (r,t): active_qubits -> added_recs")
                    for r_i, t_i, ssz, added in layer_summaries:
                        print(f"        (r={r_i}, t={t_i}): {ssz} -> {added}")
                except Exception:
                    pass

    return plan


@dataclass
class ProductPlan:
    members: dict[str, set[str]]  # prod_label -> set(member labels)
    completions: dict[
        str, list[int]
    ]  # prod_label -> list of layer indices where completed


def build_product_plan(
    dcode: DefectiveCode, layers: list[SyndromeExtractionLayer]
) -> ProductPlan:
    """
    Use analyze_layers to determine per-layer product completion points.

    product_completions[label] is a list of layer indices t where all members
    needed for that product have been measured by layer t in a round.
    """
    prods = dcode.products_list()
    prod_members: dict[str, set[str]] = {
        p.label: set(p.members) for p in prods if len(p.members) > 1
    }
    result = analyze_layers(prod_members, layers, interesting_labels=None)
    completions: dict[str, list[int]] = result["product_completions"]
    return ProductPlan(members=prod_members, completions=completions)
