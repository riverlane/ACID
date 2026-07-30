from __future__ import annotations

"""Core scheduling types: templates, schedules, and layer selection."""

import itertools
from dataclasses import dataclass

from networkx import DiGraph, Graph

from .enumeration import enumerate_all_schedules


class Stabiliser:
    """A placed stabiliser instance built from a template and a qubit map."""

    def __init__(
        self,
        stabiliser_template: StabiliserTemplate,
        qubit_map: list[int],
        label: str,
    ):
        self.stabiliser_template = stabiliser_template
        self.qubit_map: list[int] = list(qubit_map)
        self.qubit_map_reverse = {q: i for i, q in enumerate(qubit_map)}
        self.label = label

    @property
    def pauli_type(self) -> str:
        return self.stabiliser_template.pauli_type

    @property
    def connectivity_subgraph(self) -> Graph:
        return self.stabiliser_template.connectivity_subgraph

    @property
    def qubit_set(self) -> set[int]:
        return set(self.qubit_map)


class StabiliserSchedule:
    """A concrete contraction schedule for a stabiliser template.

    Stores per-timestep directed ops and derived Pauli frames for compatibility.
    """

    def __init__(
        self,
        shed_id: int,
        schedule_length: int,
        root: int,
        raw_ops: list[list[tuple[int, int]]],
        stabiliser_template: StabiliserTemplate,
    ):
        assert len(raw_ops) == schedule_length
        self.id = shed_id
        self.length = schedule_length
        self.n_qubits = stabiliser_template.n_qubits
        self.root = root
        self.stabiliser_template = stabiliser_template
        self.pauli_type = stabiliser_template.pauli_type
        self.pauli_frames = self.calculate_pauli_prop(raw_ops)
        self.preferred: bool = False

        if self.pauli_type == "X":
            self.ops = raw_ops
            self.reversed_ops = [[(b, a) for (a, b) in step] for step in raw_ops]
        else:
            self.ops = [[(b, a) for (a, b) in step] for step in raw_ops]
            self.reversed_ops = raw_ops

    def calculate_pauli_prop(self, raw_ops):
        pauli_frames = [[1] * self.n_qubits]
        for i, ops in enumerate(raw_ops):
            pauli_frames.append(pauli_frames[i].copy())
            for a, b in ops:
                pauli_frames[i + 1][a] = pauli_frames[i][a] ^ pauli_frames[i][b]
        assert pauli_frames[self.length] == [
            (1 if q_i == self.root else 0) for q_i in range(self.n_qubits)
        ]
        return pauli_frames[:-1]

    def compatible(
        self, other: StabiliserSchedule, self_to_other_qubits: dict
    ) -> bool:
        # Map local-overlap indices in both directions between the two schedules.
        other_to_self_qubits = {v: k for k, v in self_to_other_qubits.items()}
        shared_qubits_in_self = set(self_to_other_qubits.keys())
        shared_qubits_in_other = set(self_to_other_qubits.values())

        # If both roots land on the same physical qubit, they cannot coexist.
        if self.root in shared_qubits_in_self and other.root in shared_qubits_in_other:
            if self_to_other_qubits[self.root] == other.root:
                return False

        # Put both schedules into a shared orientation so edge comparisons are meaningful
        # across X/Z templates.
        ops_1_standard = self.ops if other.pauli_type == "X" else self.reversed_ops
        ops_2_standard = other.ops if self.pauli_type == "X" else other.reversed_ops

        # Keep only operations touching overlap qubits, timestep by timestep.
        filtered_ops_1 = []
        filtered_ops_2 = []
        for ops_1, ops_2 in zip(ops_1_standard, ops_2_standard):
            filtered_ops_1.append(
                [
                    (a, b)
                    for (a, b) in ops_1
                    if a in shared_qubits_in_self or b in shared_qubits_in_self
                ]
            )
            filtered_ops_2.append(
                [
                    (a, b)
                    for (a, b) in ops_2
                    if a in shared_qubits_in_other or b in shared_qubits_in_other
                ]
            )

            # Overlap qubits used by schedule-1 at this timestep, represented
            # in schedule-2 indexing for direct clash detection.
            filtered_op_qubits_1 = set(
                self_to_other_qubits[q]
                for q in itertools.chain(*filtered_ops_1[-1])
                if q in shared_qubits_in_self
            )

            # Reject simultaneous use of an overlap qubit unless the pair is the
            # same mapped edge (accounting for opposite Pauli orientation).
            for a2, b2 in filtered_ops_2[-1]:
                if a2 in shared_qubits_in_other and b2 in shared_qubits_in_other:
                    a_flip_2, b_flip_2 = (
                        (a2, b2) if self.pauli_type == other.pauli_type else (b2, a2)
                    )
                    if (
                        other_to_self_qubits[a_flip_2],
                        other_to_self_qubits[b_flip_2],
                    ) in filtered_ops_1[-1]:
                        continue
                if a2 in filtered_op_qubits_1 or b2 in filtered_op_qubits_1:
                    return False

        # Pauli-frame compatibility checks across each timestep.
        pf_1 = self.pauli_frames
        pf_2 = other.pauli_frames
        for ops_1, ops_2, pf_1_t, pf_2_t in zip(
            filtered_ops_1, filtered_ops_2, pf_1, pf_2
        ):
            # Validate schedule-1 ops against schedule-2 frame state.
            for a1, b1 in ops_1:
                if a1 in shared_qubits_in_self and b1 in shared_qubits_in_self:
                    a_flip_1, b_flip_1 = (
                        (a1, b1) if self.pauli_type == other.pauli_type else (b1, a1)
                    )
                    if (
                        self_to_other_qubits[a_flip_1],
                        self_to_other_qubits[b_flip_1],
                    ) in ops_2:
                        continue
                if (
                    b1 in shared_qubits_in_self
                    and pf_2_t[self_to_other_qubits[b1]] == 1
                ):
                    return False

            # Validate schedule-2 ops against schedule-1 frame state.
            for a2, b2 in ops_2:
                if a2 in shared_qubits_in_other and b2 in shared_qubits_in_other:
                    a_flip_2, b_flip_2 = (
                        (a2, b2) if self.pauli_type == other.pauli_type else (b2, a2)
                    )
                    if (
                        other_to_self_qubits[a_flip_2],
                        other_to_self_qubits[b_flip_2],
                    ) in ops_1:
                        continue
                if (
                    b2 in shared_qubits_in_other
                    and pf_1_t[other_to_self_qubits[b2]] == 1
                ):
                    return False

                # No shared-qubit or frame conflicts found.
        return True


class StabiliserTemplate:
    next_id = 0

    def __init__(
        self,
        pauli_type: str,
        n_qubits: int,
        connectivity_subgraph: Graph,
        SEC_cycle_length: int,
        name: str = "",
        preferred_roots: list[int] = None,
        preferred_edges: dict[tuple[int, int], list[int] | None] | None = None,
        schedule_hint: list[list[tuple[int, int]]] | None = None,
        layer_hint: int | None = None,
    ):
        self.template_id = StabiliserTemplate.next_id
        StabiliserTemplate.next_id += 1
        self.n_qubits = n_qubits
        self.pauli_type = pauli_type
        assert connectivity_subgraph.number_of_nodes() == n_qubits
        assert set(connectivity_subgraph.nodes) == set(range(n_qubits))
        assert pauli_type in ("X", "Z")
        self.connectivity_subgraph = connectivity_subgraph
        self.name = name
        self.preferred_roots = preferred_roots
        self.preferred_edges = preferred_edges if preferred_edges is not None else {}
        self.schedules = self.make_schedules(SEC_cycle_length)
        self.schedule_hint_index = self.find_schedule_hint_index(schedule_hint)
        self.layer_hint = layer_hint
        if (self.schedule_hint_index is None) != (self.layer_hint is None):
            raise ValueError(
                "Both schedule_hint and layer_hint must be provided together."
            )

    def make_schedules(self, SEC_cycle_length: int):
        schedules = []
        import os

        verbose_enum = os.environ.get("FAB_SCHEDULE_ENUM_VERBOSE", "0") == "1"
        if verbose_enum:
            try:
                from tqdm import tqdm  # type: ignore

                iter_src = list(
                    enumerate_all_schedules(
                        self.connectivity_subgraph, SEC_cycle_length
                    )
                )
                iterator = enumerate(iter_src)
                pbar = tqdm(
                    total=len(iter_src),
                    desc=f"[enum] {self.name or 'tmpl'}",
                    leave=False,
                )
                use_pbar = True
            except Exception:
                iterator = enumerate(
                    enumerate_all_schedules(
                        self.connectivity_subgraph, SEC_cycle_length
                    )
                )
                pbar = None
                use_pbar = False
        else:
            iterator = enumerate(
                enumerate_all_schedules(self.connectivity_subgraph, SEC_cycle_length)
            )
            pbar = None
            use_pbar = False

        for i, schedule in iterator:
            sched = StabiliserSchedule(
                i, SEC_cycle_length, schedule.root, schedule.steps, self
            )
            # Mark preferred according to preferred_roots/edges rules
            # If no preferences are provided (both preferred_roots is None and preferred_edges empty/None),
            # then no schedules are preferred.
            has_any_pref = (self.preferred_roots is not None) or bool(
                self.preferred_edges
            )
            root_ok = (
                True
                if self.preferred_roots is None
                else (sched.root in self.preferred_roots)
            )
            edges_ok = True
            time_ok = True
            if self.preferred_edges:
                # Build used edge -> timestep map (undirected local edge indices)
                used: dict[tuple[int, int], int] = {}
                for t, ops in enumerate(sched.ops):
                    for a, b in ops:
                        e = (a, b) if a <= b else (b, a)
                        used[e] = t
                pref_keys = set(
                    (u, v) if u <= v else (v, u)
                    for (u, v) in self.preferred_edges.keys()
                )
                used_keys = set(used.keys())
                # Only allowed edges may be used
                if not used_keys.issubset(pref_keys):
                    edges_ok = False
                # For edges with time constraints, require usage at allowed times
                if edges_ok:
                    for e_raw, times in self.preferred_edges.items():
                        e = (
                            (e_raw[0], e_raw[1])
                            if e_raw[0] <= e_raw[1]
                            else (e_raw[1], e_raw[0])
                        )
                        if times is not None:
                            # Must be used and at an allowed timestep
                            t_used = used.get(e, None)
                            if t_used is None or t_used not in set(
                                int(x) for x in times
                            ):
                                time_ok = False
                                break
            sched.preferred = bool(has_any_pref and root_ok and edges_ok and time_ok)
            schedules.append(sched)
            if use_pbar and pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        if len(schedules) == 0:
            raise ValueError(
                f"No valid schedules found for stabiliser template {self.name}"
            )
        return schedules

    def make_stabiliser(self, qubits: list[int], label: str) -> Stabiliser:
        return Stabiliser(self, qubits, label)

    def find_schedule_hint_index(
        self, schedule_hint: list[list[tuple[int, int]]] | None
    ) -> int | None:
        if schedule_hint is None:
            return None
        default_schedule_sorted = [sorted(step) for step in schedule_hint]
        for i, schedule in enumerate(self.schedules):
            if default_schedule_sorted == [sorted(step) for step in schedule.ops]:
                return i
        raise ValueError(
            "Provided schedule_hint does not match any enumerated schedule."
        )

    def __hash__(self):
        return self.template_id


@dataclass
class SyndromeExtractionLayer:
    chosen: dict[Stabiliser, StabiliserSchedule]
    code: object  # expects .stabilisers and .num_qubits

    def __post_init__(self):
        self.cycle_length = next(iter(self.chosen.items()))[1].length
        for _, schedule in self.chosen.items():
            assert schedule.length == self.cycle_length, (
                "All schedules in a syndrome extraction layer must have the same length"
            )

        self.x_roots = set(
            s.qubit_map[shed.root]
            for s, shed in self.chosen.items()
            if s.stabiliser_template.pauli_type == "X"
        )
        self.z_roots = set(
            s.qubit_map[shed.root]
            for s, shed in self.chosen.items()
            if s.stabiliser_template.pauli_type == "Z"
        )

    def collect_CNOTS(self) -> list[list[tuple[int, int]]]:
        all_CNOTS: list[set[tuple[int, int]]] = [
            set() for _ in range(self.cycle_length)
        ]
        for t in range(self.cycle_length):
            used_qubits = set()
            for stab, schedule in self.chosen.items():
                for a, b in schedule.ops[t]:
                    control, target = stab.qubit_map[b], stab.qubit_map[a]
                    if (target in used_qubits or control in used_qubits) and (
                        control,
                        target,
                    ) not in all_CNOTS[t]:
                        raise ValueError("Invalid schedule. Clashing qubits.")
                    all_CNOTS[t].add((control, target))
                    used_qubits.add(target)
                    used_qubits.add(control)
        return all_CNOTS

    def is_compatible_with_scheduling_graph(self, scheduling_graph: DiGraph) -> bool:
        for stab, schedule in self.chosen.items():
            if stab.label not in [s.label for s in scheduling_graph]:
                return False
            if schedule.id in stab.disallowed_schedules:
                return False
        for u, v in scheduling_graph.edges():
            if u in self.chosen and v in self.chosen:
                data = scheduling_graph.get_edge_data(u, v)
                pair = (self.chosen[u].id, self.chosen[v].id)
                if pair not in data["allowed_pairs"]:
                    return False
        return True

    def endcycle_expanded_stabilisers(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        if self.code is None:
            raise ValueError("SyndromeExtractionLayer has no associated code")
        steps = self.collect_CNOTS()
        chosen_set = set(self.chosen.keys())
        excluded_roots: set[int] = set()
        for stab, schedule in self.chosen.items():
            root_idx = schedule.root
            excluded_roots.add(stab.qubit_map[root_idx])
        x_rows: list[list[int]] = []
        z_rows: list[list[int]] = []
        for stab in self.code.stabilisers.values():
            if stab in chosen_set:
                continue
            support = set(stab.qubit_map)
            if stab.pauli_type == "X":
                for ops in steps:
                    for control, target in ops:
                        if control in support:
                            if target in support:
                                support.remove(target)
                            else:
                                support.add(target)
                x_rows.append(sorted(support))
            else:
                for ops in steps:
                    for control, target in ops:
                        if target in support:
                            if control in support:
                                support.remove(control)
                            else:
                                support.add(control)
                z_rows.append(sorted(support))
        kept_qubits = [
            q for q in range(self.code.num_qubits) if q not in excluded_roots
        ]
        return x_rows, z_rows, kept_qubits

    def endcycle_parity_check_matrix(self) -> dict:
        x_rows, z_rows, kept_qubits = self.endcycle_expanded_stabilisers()
        col_index = {q: i for i, q in enumerate(kept_qubits)}
        n_eff = len(kept_qubits)
        Hx = [[0] * n_eff for _ in range(len(x_rows))]
        Hz = [[0] * n_eff for _ in range(len(z_rows))]
        labels: list[str] = []
        for i, supp in enumerate(x_rows):
            labels.append(f"X_row_{i}")
            for q in supp:
                j = col_index.get(q)
                if j is not None:
                    Hx[i][j] ^= 1
        for k, supp in enumerate(z_rows):
            labels.append(f"Z_row_{k}")
            for q in supp:
                j = col_index.get(q)
                if j is not None:
                    Hz[k][j] ^= 1
        from acid.pauli import StabiliserCode

        return StabiliserCode(num_qubits=n_eff, row_labels=labels, Hx=Hx, Hz=Hz)

    def propagate(self, P: PauliString) -> PauliString:

        if P.n != self.code.num_qubits:
            raise ValueError("PauliString has wrong length for this code")
        Q = P.copy()
        steps = self.collect_CNOTS()
        Q.conj_steps(steps)
        return Q

    def collect_cx_stim(self) -> list[list[tuple[int, int]]]:
        return self.collect_CNOTS()

    @staticmethod
    def cx_line_stim(step: list[tuple[int, int]]) -> str:
        if not step:
            return ""
        flat: list[str] = []
        for c, t in step:
            flat.append(str(c))
            flat.append(str(t))
        return "CX " + " ".join(flat)

    def emit_layer_contract(self) -> list[str]:
        lines: list[str] = []
        for step in self.collect_cx_stim():
            s = self.cx_line_stim(step)
            if s:
                lines.append(s)
            lines.append("TICK")
        return lines

    def emit_layer_expand(self) -> list[str]:
        lines: list[str] = []
        steps = self.collect_cx_stim()
        for step in reversed(steps):
            s = self.cx_line_stim(step)
            if s:
                lines.append(s)
            lines.append("TICK")
        return lines

    def roots_by_basis(self) -> tuple[list[int], list[int]]:
        x_roots: list[int] = []
        z_roots: list[int] = []
        for stab, shed in self.chosen.items():
            root_q = stab.qubit_map[shed.root]
            if stab.pauli_type == "X":
                x_roots.append(root_q)
            else:
                z_roots.append(root_q)
        return sorted(x_roots), sorted(z_roots)
