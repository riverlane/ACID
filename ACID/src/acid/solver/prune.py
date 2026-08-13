from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass
class PruningResult:
    allowed_per_label: dict[str, set[int]]
    filtered_graph: nx.DiGraph
    stats: dict[str, object]


def _summary(nums: list[int]) -> dict[str, float]:
    if not nums:
        return {"min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0}
    a = sorted(nums)
    n = len(a)
    mid = a[n // 2] if (n % 2) == 1 else (0.5 * (a[n // 2 - 1] + a[n // 2]))
    s = sum(a)
    return {
        "min": float(a[0]),
        "max": float(a[-1]),
        "median": float(mid),
        "mean": float(s) / float(n),
    }


def prune_schedule_graph(
    scheduling_graph: nx.DiGraph,
    M: int,
    *,
    verbose: bool = False,
) -> PruningResult:
    if M < 1:
        raise ValueError("M must be >= 1")
    if scheduling_graph is None or not isinstance(scheduling_graph, nx.DiGraph):
        raise TypeError("scheduling_graph must be a networkx.DiGraph")

    # Stabiliser labels and schedule counts
    labels: list[str] = []
    K_by_label: dict[str, int] = {}
    for stab in scheduling_graph.nodes():
        lab = getattr(stab, "label", None)
        if not isinstance(lab, str) or not lab:
            raise ValueError("All scheduling_graph nodes must have a non-empty .label")
        if lab in K_by_label:
            raise ValueError(f"Duplicate stabiliser label in scheduling_graph: {lab}")
        tmpl = getattr(stab, "stabiliser_template", None)
        if tmpl is None or not hasattr(tmpl, "schedules"):
            raise ValueError("Node missing stabiliser_template.schedules")
        K = len(tmpl.schedules)
        if K <= 0:
            raise ValueError(f"Stabiliser {lab} has zero schedules")
        labels.append(lab)
        K_by_label[lab] = K

    # Preferred ids per label
    preferred_ids: dict[str, set[int]] = {}
    for stab in scheduling_graph.nodes():
        lab = stab.label
        tmpl = stab.stabiliser_template
        pref = {i for i, sch in enumerate(tmpl.schedules) if getattr(sch, "preferred", False)}
        preferred_ids[lab] = pref

    # Keep sets per label
    kept: dict[str, set[int]] = {lab: set() for lab in labels}
    preferred_total = 0
    for lab in labels:
        K = K_by_label[lab]
        target = min(M, K)
        pref = preferred_ids.get(lab, set())
        if len(pref) > target:
            raise RuntimeError(
                f"Preferred schedules ({len(pref)}) exceed M={target} for stabiliser {lab}"
            )
        if K <= target:
            kept[lab] = set(range(K))
        else:
            kept[lab] = set(pref)
        preferred_total += len(pref)

    if preferred_total == 0 and verbose:
        print("[prune] warning: no preferred schedules marked across the schedule graph")

    # Precompute neighbor compatibility maps for scoring
    from collections import defaultdict

    map_out: dict[object, dict[str, dict[int, set[int]]]] = {}
    map_in: dict[object, dict[str, dict[int, set[int]]]] = {}

    node_by_label: dict[str, object] = {stab.label: stab for stab in scheduling_graph.nodes()}
    it_edges = scheduling_graph.edges(data=True)
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore

            it_edges = tqdm(list(it_edges), desc="[prune] build compat maps", leave=False)
        except Exception:
            pass
    for u, v, data in it_edges:
        allowed = set(data.get("allowed_pairs") or [])
        u_map = map_out.setdefault(u, {})
        mout = u_map.setdefault(v.label, defaultdict(set))
        for ku, kv in allowed:
            mout[int(ku)].add(int(kv))
        v_map_in = map_in.setdefault(v, {})
        minv = v_map_in.setdefault(u.label, defaultdict(set))
        for ku, kv in allowed:
            minv[int(kv)].add(int(ku))

    # Scoring and fill for labels where K > M
    it_labels = labels
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore

            it_labels = list(it_labels)
            it_labels = tqdm(it_labels, desc="[prune] score+fill labels", leave=False)
        except Exception:
            pass
    for lab in it_labels:
        K = K_by_label[lab]
        target = min(M, K)
        if len(kept[lab]) >= target:
            continue
        pref = preferred_ids.get(lab, set())
        candidates = [i for i in range(K) if i not in pref]
        u_node = node_by_label[lab]
        out_by_label = map_out.get(u_node, {})
        in_by_label = map_in.get(u_node, {})

        # Neighbor preferred sets
        neigh_pref: dict[str, set[int]] = {}
        for _, v, _ in scheduling_graph.out_edges(u_node, data=True):
            v_lab = v.label
            neigh_pref[v_lab] = preferred_ids.get(v_lab, set())
        for w, _, _ in scheduling_graph.in_edges(u_node, data=True):
            w_lab = w.label
            neigh_pref[w_lab] = preferred_ids.get(w_lab, set())

        scored: list[tuple[int, int, int]] = []
        for i in candidates:
            primary = 0
            tieb = 0
            for v_lab, by_ku in out_by_label.items():
                kvs = by_ku.get(int(i), set())
                if kvs:
                    tieb += len(kvs)
                    primary += len(kvs & neigh_pref.get(v_lab, set()))
            for w_lab, by_kv in in_by_label.items():
                kws = by_kv.get(int(i), set())
                if kws:
                    tieb += len(kws)
                    primary += len(kws & neigh_pref.get(w_lab, set()))
            scored.append((primary, tieb, int(i)))

        slots = target - len(kept[lab])
        for _, _, idx in sorted(scored, key=lambda x: (-x[0], -x[1], x[2]))[:slots]:
            kept[lab].add(int(idx))

    # Filter graph
    G2 = nx.DiGraph()
    G2.add_nodes_from(scheduling_graph.nodes())
    for u, v, data in scheduling_graph.edges(data=True):
        allowed = set(data.get("allowed_pairs") or [])
        u_lab = u.label
        v_lab = v.label
        filt = {(ku, kv) for (ku, kv) in allowed if ku in kept[u_lab] and kv in kept[v_lab]}
        G2.add_edge(u, v, allowed_pairs=filt)

    counts_list = [K_by_label[lab] for lab in labels]
    preferred_counts = [len(preferred_ids.get(lab, set())) for lab in labels]
    stats: dict[str, object] = {
        "schedule_counts_by_label": {lab: K_by_label[lab] for lab in labels},
        "schedule_counts_summary": _summary(counts_list),
        "preferred_count_by_label": {lab: len(preferred_ids.get(lab, set())) for lab in labels},
        "preferred_counts_summary": _summary(preferred_counts),
        "selected_count_by_label": {lab: len(kept[lab]) for lab in labels},
        "params": {"M": int(M)},
    }

    return PruningResult(allowed_per_label=kept, filtered_graph=G2, stats=stats)
