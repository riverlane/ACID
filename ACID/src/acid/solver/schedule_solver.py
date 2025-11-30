from __future__ import annotations

from acid.defects.quasi import QuasiStabiliser, QuasiProduct


"""
ScheduleSolver — CP-SAT scheduler with product-stabiliser iteration constraints.

This module extends the baseline scheduling in solver.py by adding support for
measuring product stabilisers via sequential measurement of quasi-stabilisers,
subject to ordering constraints derived from an anti-commutation graph.

Design notes and mapping to the user’s proposal:

- We reuse the stabiliser types and schedule machinery from solver.solver
  (StabiliserTemplate, Stabiliser, StabiliserSchedule, SyndromeExtractionLayer).
  ScheduleSolver subclasses Code to inherit compatibility graph construction
  and schedule enumeration.

- Inputs:
  - connectivity_graph, stabilisers: same semantics as Code; typically these
    stabilisers are the quasi-stabilisers you intend to measure (with their
    own connectivity subgraphs and qubit maps), not the original full ones.
  - anticommutation_graph: an undirected graph whose nodes are stabiliser
    labels (matching those passed in `stabilisers`). An edge between labels u
    and v indicates they are expected to anti-commute (opposite types and odd
    overlap). We verify that expectation against the actual overlaps.
  - product_stabilisers: list of product specs describing how quasi-stabilisers
    are grouped into products (all members in a product must have the same
    Pauli type).

- Iteration counters and flags per product:
  For each product P and each layer t, we add:
    - c_P[t] ∈ [0..num_layers]: non-decreasing integer counter of completed
      iterations of P (how many times all members of P have been measured since
      the start).
    - f_{P,i}[t] ∈ {0,1}: for each quasi-stabiliser i ∈ P, indicates whether i
      has been measured at least once in the current (partial) iteration by
      layer t.

  Recurrence (t >= 1):
    - Reset_P[t] ≡ AND_i f_{P,i}[t-1] (i.e., last layer completed an iteration)
    - c_P[t] = c_P[t-1] + Reset_P[t]
    - If Reset_P[t] then f_{P,i}[t] = 0 for all i (start of new iteration)
      else f_{P,i}[t] = OR(f_{P,i}[t-1], measured(i, t))

  Base layer t=0:
    - c_P[0] = 0; f_{P,i}[0] = measured(i, 0)

  This matches the intended "count up an iteration" semantics. Note we reset
  flags on increment; this is a minor correction to the step (2) phrasing that
  would otherwise prevent accumulation.

- Cross-product gating constraints (per quasi q and layer t):
  Let S_X be the set of X-type products containing q (if q is X), and S_Z the
  Z-type products containing q (if q is Z). Let Opp(q) be the set of products
  of the opposite type that contain any quasi that anti-commutes with q.

  When measured(q, t) = 1, enforce:
    - If q is X-type: for each Pz in Opp(q), for each Px in S_X,
         c_{Pz}[t] == c_{Px}[t].
      (Stay in lockstep w.r.t. opposing bundles.)
    - If q is Z-type: for each Px in Opp(q), for each Pz in S_Z,
         c_{Px}[t] >= c_{Pz}[t] + 1.
      (Only measure Z’s after opposing X bundles have advanced.)

  These are enforced via reified equalities/inequalities.

Outputs:
  - create_layers_with_products(num_layers, solve_time) returns a list of
    SyndromeExtractionLayer objects (same structure as in solver.Code), with the
    chosen schedule per stabiliser per layer. The layer choices respect both the
    usual per-layer compatibility and the product iteration constraints.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, Optional
import itertools

import networkx as nx
from ortools.sat.python import cp_model

# Reuse core types from the base solver
from acid.scheduling.types import (
    Stabiliser,
    StabiliserTemplate,
    StabiliserSchedule,
    SyndromeExtractionLayer,
)


class SchedulingError(Exception):
    pass

class SchedulingInfeasibleError(SchedulingError):
    pass

class SchedulingTimeLimitError(SchedulingError):
    pass

class SchedulingModelInvalidError(SchedulingError):
    pass





class ScheduleSolver:
    """Scheduling solver with product-stabiliser constraints.

    Parameters:
      - connectivity_graph: device connectivity
      - stabilisers: list of (StabiliserTemplate, qubit_map, label) for the
        quasi-stabilisers that can be measured.
      - anticommutation_graph: nx.Graph with node set equal to the stabiliser
        labels; edges indicate expected anti-commutation pairs.
      - product_stabilisers: list of ProductSpec grouping stabilisers of the
        same Pauli type into products measured iteratively.

    Inherits Code initialisation to build stabiliser objects, precompute
    schedule dependencies, and construct the scheduling graph of allowed pairs.
    """

    def __init__(
        self,
        connectivity_graph: nx.Graph,
        stabilisers: List[Tuple[StabiliserTemplate, List[int], str]],
        *,
        anticommutation_graph: nx.Graph,
        product_stabilisers: List[QuasiProduct],
    ) -> None:
        # Standalone initialisation (no inheritance from Code)
        self.num_qubits = connectivity_graph.number_of_nodes()
        self.connectivity_graph = connectivity_graph
        # Build stabiliser objects
        self.stabilisers = self.make_stabilisers(stabilisers)
        # Compute schedule interoperability structures
        self.shape_neighbours = self.get_shape_neighbours()
        self.schedule_dependencies = self.calculate_schedule_dependencies()
        self.scheduling_graph = self.create_shedule_graph()
        self.anticomm_graph: nx.Graph = anticommutation_graph.copy()
        self.product_specs: List[QuasiProduct] = list(product_stabilisers)

        # Validate that anticomm graph nodes match our stabiliser labels
        stab_labels = set(self.stabilisers.keys())
        if not set(self.anticomm_graph.nodes).issubset(stab_labels):
            missing = set(self.anticomm_graph.nodes) - stab_labels
            raise ValueError(
                f"Anticommutation graph has unknown nodes not in provided stabilisers: {sorted(missing)}"
            )

        # Validate products: labels exist and types match
        for ps in self.product_specs:
            if ps.pauli_type not in ('X', 'Z'):
                raise ValueError(f"Product {ps.label} has invalid pauli_type: {ps.pauli_type}")
            if not ps.members:
                raise ValueError(f"Product {ps.label} has no members")
            for m in ps.members:
                if m not in self.stabilisers:
                    raise ValueError(f"Product {ps.label} references unknown stabiliser label: {m}")
                if self.stabilisers[m].pauli_type != ps.pauli_type:
                    raise ValueError(
                        f"Product {ps.label} pauli_type {ps.pauli_type} does not match member {m} type {self.stabilisers[m].pauli_type}"
                    )

        # Verify commutation according to anticomm_graph: for each pair of
        # opposite-type stabilisers, their overlap parity must match whether an edge exists.
        # If there is an edge => odd overlap; else => even overlap.
        self._verify_expected_commutations()

        # Build quick lookup structures for product membership and anticomm products
        self._build_product_indices()
        # Optional pruning state (populated via apply_pruning)
        self._prune_allowed_ids_by_label: Dict[str, List[int]] | None = None
        self._pruned_graph: nx.DiGraph | None = None

    def apply_pruning(self, *, M: int, verbose: bool = False) -> None:
        from .prune import prune_schedule_graph
        res = prune_schedule_graph(self.scheduling_graph, M, verbose=verbose)
        self._prune_allowed_ids_by_label = {lab: sorted(list(xs)) for lab, xs in res.allowed_per_label.items()}
        self._pruned_graph = res.filtered_graph

    # --- Compatibility helpers (copied from legacy Code class) ---
    def make_stabilisers(self, stabilisers) -> dict[str, Stabiliser]:
        stabiliser_objs: Dict[str, Stabiliser] = {}
        for template, qubits, label in stabilisers:
            stabiliser = template.make_stabiliser(qubits, label)
            for u, v in stabiliser.connectivity_subgraph.edges:
                assert self.connectivity_graph.has_edge(stabiliser.qubit_map[u], stabiliser.qubit_map[v]), \
                    "Stabiliser connectivity not compatible with code connectivity"
            assert label not in stabiliser_objs, f"Duplicate stabiliser label {label}"
            stabiliser_objs[label] = stabiliser
        return stabiliser_objs

    def get_shape_neighbours(self):
        template_neighbours = set()
        for stab1_unsorted, stab2_unsorted in itertools.combinations(self.stabilisers.values(), 2):
            stab1, stab2 = sorted([stab1_unsorted, stab2_unsorted], 
                                  key=lambda s: (s.stabiliser_template.template_id, tuple(s.qubit_map_reverse)))
            overlap_12 = {stab1.qubit_map_reverse[q]: stab2.qubit_map_reverse[q] for q in sorted(stab1.qubit_set.intersection(stab2.qubit_set))}
            if not overlap_12:
                continue
            template_neighbours.add(
                (stab1.stabiliser_template, stab2.stabiliser_template, tuple(overlap_12.keys()), tuple(overlap_12.values())))
        return list(template_neighbours)

    def calculate_schedule_dependencies(self):
        import os
        verbose_build = os.environ.get("FAB_SCHEDULE_BUILD_VERBOSE", "0") == "1"
        dependencies = {}
        it_pairs = self.shape_neighbours
        if verbose_build:
            try:
                from tqdm import tqdm  # type: ignore
                it_pairs = tqdm(list(it_pairs), desc="[build] dep pairs", leave=False)
            except Exception:
                pass
        for template1, template2, indices1, indices2 in it_pairs:
            key = (template1.template_id, template2.template_id, tuple(sorted(zip(indices1, indices2))))
            dependencies[key] = []
            common_qubits = dict(zip(indices1, indices2))
            iter_s1 = enumerate(template1.schedules)
            if verbose_build:
                try:
                    from tqdm import tqdm  # type: ignore
                    iter_s1 = enumerate(template1.schedules)
                    inner_total = len(template1.schedules) * len(template2.schedules)
                    pbar = tqdm(total=inner_total, desc=f"[build] deps {template1.name or template1.template_id}-{template2.name or template2.template_id}", leave=False)
                except Exception:
                    pbar = None
            else:
                pbar = None
            for s1_i, schedule1 in iter_s1:
                for s2_i, schedule2 in enumerate(template2.schedules):
                    if schedule1.compatible(schedule2, common_qubits):
                        dependencies[key].append((s1_i, s2_i))
                    if pbar is not None:
                        pbar.update(1)
            if pbar is not None:
                pbar.close()
        return dependencies

    def create_shedule_graph(self):
        import os
        verbose_build = os.environ.get("FAB_SCHEDULE_BUILD_VERBOSE", "0") == "1"
        scheduling_graph = nx.DiGraph()
        scheduling_graph.add_nodes_from(self.stabilisers.values())
        it_stabs = list(self.stabilisers.values())
        if verbose_build:
            try:
                from tqdm import tqdm  # type: ignore
                it_prog = tqdm(range(len(it_stabs)), desc="[build] sched graph", leave=False)
            except Exception:
                it_prog = None
        else:
            it_prog = None
        # Running stats for number of schedules per node and allowed_pairs per edge
        node_sched_counts: list[int] = []
        edge_allowed_counts: list[int] = []

        def _median(vals: list[int]) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            n = len(s)
            return float(s[n//2]) if (n % 2 == 1) else 0.5 * float(s[n//2-1] + s[n//2])
        for stab_1_i_unsorted, stab1_unsorted in enumerate(it_stabs):
            qubit_set1 = stab1_unsorted.qubit_set
            template1 = stab1_unsorted.stabiliser_template
            # Track node schedule count once per node
            try:
                node_sched_counts.append(len(template1.schedules))
            except Exception:
                pass
            for stab2_unsorted in list(self.stabilisers.values())[stab_1_i_unsorted+1:]:
                overlap = qubit_set1.intersection(stab2_unsorted.qubit_set)
                if not overlap:
                    continue
                stab1, stab2 = sorted([stab1_unsorted, stab2_unsorted], 
                                  key=lambda s: (s.stabiliser_template.template_id, tuple(s.qubit_map_reverse)))
                mapping = {stab1.qubit_map_reverse[q]: stab2.qubit_map_reverse[q] for q in stab1.qubit_set.intersection(stab2.qubit_set)}
                key = (stab1.stabiliser_template.template_id, stab2.stabiliser_template.template_id, tuple(sorted(zip(mapping.keys(), mapping.values()))))
                if len(self.schedule_dependencies[key]) == len(stab1.stabiliser_template.schedules) * len(stab2.stabiliser_template.schedules):
                    continue
                ap = self.schedule_dependencies[key]
                try:
                    edge_allowed_counts.append(len(ap))
                except Exception:
                    pass
                scheduling_graph.add_edge(stab1, stab2, allowed_pairs=ap)
            if it_prog is not None:
                # Update running stats in progress bar postfix
                try:
                    ns_min = min(node_sched_counts) if node_sched_counts else 0
                    ns_med = _median(node_sched_counts)
                    ns_max = max(node_sched_counts) if node_sched_counts else 0
                    ea_min = min(edge_allowed_counts) if edge_allowed_counts else 0
                    ea_med = _median(edge_allowed_counts) if edge_allowed_counts else 0.0
                    ea_max = max(edge_allowed_counts) if edge_allowed_counts else 0
                    it_prog.set_postfix({
                        "nodes": f"{len(node_sched_counts)}",
                        "K[min/med/max]": f"{ns_min}/{int(ns_med) if ns_med.is_integer() else ns_med}/{ns_max}",
                        "edges": f"{len(edge_allowed_counts)}",
                        "AP[min/med/max]": f"{ea_min}/{int(ea_med) if isinstance(ea_med, float) and ea_med.is_integer() else ea_med}/{ea_max}",
                    })
                except Exception:
                    pass
                it_prog.update(1)
        if it_prog is not None:
            it_prog.close()
        return scheduling_graph

    # --- Verification helpers ---
    def _verify_expected_commutations(self) -> None:
        labels = list(self.stabilisers.keys())
        label_to_stab = self.stabilisers
        G = self.anticomm_graph
        # Only consider opposite-type pairs; equal types always commute here
        for i, a in enumerate(labels):
            sa = label_to_stab[a]
            for b in labels[i + 1 :]:
                sb = label_to_stab[b]
                if sa.pauli_type == sb.pauli_type:
                    continue
                odd = (len(sa.qubit_set.intersection(sb.qubit_set)) % 2) == 1
                has_edge = G.has_edge(a, b)
                if odd != has_edge:
                    raise ValueError(
                        f"Commutation mismatch for pair ({a},{b}): expected {'anticommute' if has_edge else 'commute'}, got {'anticommute' if odd else 'commute'}."
                    )

    def _build_product_indices(self) -> None:
        # Map product label -> ProductSpec and member Stabiliser objects
        self.products_by_label: Dict[str, QuasiProduct] = {p.label: p for p in self.product_specs}
        # Stabiliser label -> list of product labels of same type containing it
        self.products_by_member: Dict[str, List[str]] = {lab: [] for lab in self.stabilisers}
        for p in self.product_specs:
            for m in p.members:
                self.products_by_member[m].append(p.label)

        # For each stabiliser label q, collect opposite-type products that contain
        # any anticomm neighbor of q
        self.opp_products_by_label: Dict[str, Set[str]] = {lab: set() for lab in self.stabilisers}
        for u, v in self.anticomm_graph.edges():
            # u-v anticommute; add products of type(v) to u, and type(u) to v
            su, sv = self.stabilisers[u], self.stabilisers[v]
            # For node u, collect products that contain v and match sv.type
            for p in self.product_specs:
                if p.pauli_type == sv.pauli_type and (v in p.members):
                    self.opp_products_by_label[u].add(p.label)
            # For node v, collect products that contain u and match su.type
            for p in self.product_specs:
                if p.pauli_type == su.pauli_type and (u in p.members):
                    self.opp_products_by_label[v].add(p.label)

    # --- Scheduling with products ---
    def create_layers_with_products(
        self,
        num_layers: int,
        solve_time: float = 10.0,
    ) -> List[SyndromeExtractionLayer]:
        """Build and solve CP-SAT with product/iteration constraints.

        Returns a list of SyndromeExtractionLayer objects, one per layer.
        """
        model = cp_model.CpModel()

        # Optional pruning already applied in prepare_solver
        allowed_ids_by_label: Dict[str, List[int]] = {}
        if self._prune_allowed_ids_by_label is not None and self._pruned_graph is not None:
            allowed_ids_by_label = {lab: list(ids) for lab, ids in self._prune_allowed_ids_by_label.items()}
            scheduling_graph = self._pruned_graph
        else:
            scheduling_graph = self.scheduling_graph
            for stab in self.stabilisers.values():
                K = len(stab.stabiliser_template.schedules)
                allowed_ids_by_label[stab.label] = list(range(K))

        # Decision variables: assignment[label][t][k]
        # where k indexes schedules for that stabiliser template.
        assignment: Dict[str, List[Dict[int, cp_model.IntVar]]] = {}
        measured: Dict[str, List[cp_model.IntVar]] = {}
        for stab in self.stabilisers.values():
            kept_ids = allowed_ids_by_label.get(stab.label)
            if not kept_ids:
                raise RuntimeError(f"No allowed schedules for stabiliser {stab.label}")
            per_layer: List[Dict[int, cp_model.IntVar]] = []
            measured_layer: List[cp_model.IntVar] = []
            for t in range(num_layers):
                var_map: Dict[int, cp_model.IntVar] = {}
                for k_id in kept_ids:
                    var_map[k_id] = model.NewBoolVar(f"assign__{stab.label}__t{t}__k{k_id}")
                # At most one schedule for this stabiliser in this layer
                model.Add(sum(var_map.values()) <= 1)
                per_layer.append(var_map)
                m = model.NewBoolVar(f"measured__{stab.label}__t{t}")
                model.Add(sum(var_map.values()) == m)
                measured_layer.append(m)
                # Schedule hint only if it exists and is kept
                ds = getattr(stab.stabiliser_template, "schedule_hint_index", None)
                dl = getattr(stab.stabiliser_template, "layer_hint", None)
                if ds is not None and dl is not None and dl == t and ds in var_map:
                    # Hint: prefer this schedule at this layer
                    model.AddHint(var_map[ds], 1)
            assignment[stab.label] = per_layer
            measured[stab.label] = measured_layer

        # Coverage: every stabiliser must be measured at least once across layers
        for lab, mlist in measured.items():
            model.Add(sum(mlist) >= 1)

        # Compatibility constraints per layer, using precomputed scheduling_graph
        for u, v in scheduling_graph.edges():
            data = scheduling_graph.get_edge_data(u, v)
            allowed = set(data['allowed_pairs'])
            u_allowed_ids = allowed_ids_by_label[u.label]
            v_allowed_ids = allowed_ids_by_label[v.label]
            for t in range(num_layers):
                u_map = assignment[u.label][t]
                v_map = assignment[v.label][t]
                for ku in u_allowed_ids:
                    for kv in v_allowed_ids:
                        if (ku, kv) not in allowed:
                            model.Add(u_map[ku] + v_map[kv] <= 1)

        # --- Product iteration variables and constraints ---
        # For each product P and layer t:
        #   c_P[t] ∈ [0..num_layers]
        #   f_{P,i}[t] for each member i
        #   Reset_P[t] for t>=1: AND_i f_{P,i}[t-1]
        #   c_P[t] = c_P[t-1] + Reset_P[t] (t>=1); c_P[0] = 0
        #   If Reset_P[t]: f_{P,i}[t] = 0; else f_{P,i}[t] = OR(f_{P,i}[t-1], measured(i, t))  ###### SHOULD BE if reset f{P,i}[t] = measured(i, t)

        product_counters: Dict[str, List[cp_model.IntVar]] = {}
        product_flags: Dict[str, Dict[str, List[cp_model.IntVar]]] = {}
        product_inprogress: Dict[str, List[cp_model.IntVar]] = {}
        product_reset: Dict[str, List[cp_model.IntVar]] = {}

        for p in self.product_specs:
            # counters
            product_counters[p.label] = [model.NewIntVar(0, num_layers, f"ctr__{p.label}__t{t}") for t in range(num_layers)]
            product_inprogress[p.label] = [model.NewBoolVar(f"inprog__{p.label}__t{t}") for t in range(num_layers)]
            # member flags per layer
            flags_for_members: Dict[str, List[cp_model.IntVar]] = {}
            for m in p.members:
                flags_for_members[m] = [model.NewBoolVar(f"flag__{p.label}__{m}__t{t}") for t in range(num_layers)]
            product_flags[p.label] = flags_for_members
            # resets (t>=1); for t=0 use constant 0 via a fixed false bool
            reset_vars: List[cp_model.IntVar] = []
            for t in range(num_layers):
                if t == 0:
                    r = model.NewBoolVar(f"reset__{p.label}__t0")
                    model.Add(r == 0)
                else:
                    r = model.NewBoolVar(f"reset__{p.label}__t{t}")
                    # r == AND_i f_{i}[t-1]
                    # decompose AND via AddBoolAnd and AndBoolOr
                    prev_flags = [flags_for_members[m][t - 1] for m in p.members]
                    model.AddBoolAnd(prev_flags).OnlyEnforceIf(r)
                    model.AddBoolOr([f.Not() for f in prev_flags]).OnlyEnforceIf(r.Not())
                reset_vars.append(r)
                # In-progress flag is true iff any of the member flags is true
                model.AddMaxEquality(product_inprogress[p.label][t],
                                     [flags_for_members[m][t] for m in p.members])

            product_reset[p.label] = reset_vars

            # counter recurrence and flag recurrence
            # c[0] = 0
            c = product_counters[p.label]
            model.Add(c[0] == 0)
            # t >= 1: c[t] = c[t-1] + reset[t]
            for t in range(1, num_layers):
                model.Add(c[t] == c[t - 1] + reset_vars[t])
            # flag update
            for m in p.members:
                f = flags_for_members[m]
                # t = 0: f[0] = measured(m, 0)
                model.Add(f[0] == measured[m][0])
                for t in range(1, num_layers):
                    # If reset[t] then f[t] = measured(m, t)
                    model.Add(f[t] == measured[m][t]).OnlyEnforceIf(reset_vars[t])
                    # Else f[t] = OR(f[t-1], measured(m,t))
                    not_reset = reset_vars[t].Not()
                    model.Add(f[t] >= f[t - 1]).OnlyEnforceIf(not_reset)
                    model.Add(f[t] >= measured[m][t]).OnlyEnforceIf(not_reset)
                    model.Add(f[t] <= f[t - 1] + measured[m][t]).OnlyEnforceIf(not_reset)

        # --- Cross-product gating constraints ---
        # If measured(q, t) then:
        #   - q is X: for each Pz in Opp(q) and each Px in Products(q): c_Pz[t] == c_Px[t]
        #   - q is Z: for each Px in Opp(q) and each Pz in Products(q): c_Px[t] >= c_Pz[t] + 1
        # NOTE: The orphan-case below was left disabled previously due to invalid reification on IntVar.
        #       Uncomment the main gating when opp_products/same_products are defined and valid.
        for q_label, stab in self.stabilisers.items():
            q_type = stab.pauli_type
            same_products = self.products_by_member.get(q_label, [])
            opp_products = sorted(self.opp_products_by_label.get(q_label, set()))
            if same_products:
                if not opp_products:
                    continue  # nothing to gate
                for t in range(num_layers):
                    mqt = measured[q_label][t]
                    if q_type == 'X':
                        for pz_label in opp_products:
                            cz = product_counters[pz_label][t]
                            for px_label in same_products:
                                cx = product_counters[px_label][t]
                                model.Add(cx == cz).OnlyEnforceIf(mqt)
                    else:  # 'Z'
                        for px_label in opp_products:
                            cx = product_counters[px_label][t]
                            for pz_label in same_products:
                                cz = product_counters[pz_label][t]
                                # strict > as cx >= cz + 1
                                model.Add(cx >= cz + 1).OnlyEnforceIf(mqt)
            else:
                # Orphan quasi stabiliser: forbid measuring q in any layer where an
                # anticomm-opposite product is in-progress (any member flagged in the
                # current iteration). Implement as m(q,t) + inprogress(P_opp,t) <= 1.
                if not opp_products:
                    continue
                for t in range(num_layers):
                    mqt = measured[q_label][t]
                    for p_opp_label in opp_products:
                        inprog = product_inprogress[p_opp_label][t]
                        model.Add(mqt + inprog <= 1)

            

        # Objective: big-M lexicographic maximize (min_group_coverage, total_quasi_measured)
        # Build group totals: products use their final counter; non-product stabs use sum of measured flags.
        group_totals: List[cp_model.IntVar] = []
        for p in self.product_specs:
            group_totals.append(product_counters[p.label][-1])
        for lab, stab in self.stabilisers.items():
            if not self.products_by_member.get(lab, []):
                cnt = model.NewIntVar(0, num_layers, f"count__{lab}")
                model.Add(cnt == sum(measured[lab]))
                group_totals.append(cnt)
        # z <= each group; maximization pushes z to the minimum of the groups
        z = model.NewIntVar(0, num_layers, "min_group_coverage")
        for g in group_totals:
            model.Add(z <= g)
        # Tie breaker: total measured across all stabs
        total_quasi = model.NewIntVar(0, len(self.stabilisers) * num_layers, "total_quasi_measured")
        model.Add(total_quasi == sum(sum(measured[lab]) for lab in self.stabilisers))
        # Big-M weighting
        M = len(self.stabilisers) * num_layers + 1
        model.Maximize(z * M + total_quasi)


        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(solve_time)
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Distinguish common failure modes
            status_name = None
            try:
                status_name = solver.StatusName(status)
            except Exception:
                status_name = str(status)
            if status == cp_model.INFEASIBLE:
                raise SchedulingInfeasibleError(f"Proven infeasible (L={num_layers}, time_limit={solve_time}s)")
            if status == cp_model.MODEL_INVALID:
                raise SchedulingModelInvalidError("Model invalid")
            # Otherwise, treat as time limit / unknown
            raise SchedulingTimeLimitError(f"Time limit or unknown status ({status_name}) with no feasible solution found (L={num_layers}, time_limit={solve_time}s)")

        layers: List[SyndromeExtractionLayer] = []
        for t in range(num_layers):
            chosen: Dict[Stabiliser, StabiliserSchedule] = {}
            for stab in self.stabilisers.values():
                var_map = assignment[stab.label][t]  # Dict[int(schedule_id) -> IntVar]
                # iterate actual ids and vars
                picked = False
                for k_id, var in var_map.items():
                    if solver.Value(var) == 1:
                        chosen[stab] = stab.stabiliser_template.schedules[int(k_id)]
                        picked = True
                        break
                # It is valid that a stabiliser is not measured in a given layer (<=1 per layer and sum across layers >=1)
            layers.append(SyndromeExtractionLayer(chosen=chosen, code=self))
        return layers
