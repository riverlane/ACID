"""Enumerate local contraction schedules over a connected graph and root.

Schedules are lists of directed moves per timestep that gather onto the root.
"""

from typing import Hashable, Iterable, List, Tuple, Dict
import itertools as it
import networkx as nx
from networkx.algorithms.tree.mst import SpanningTreeIterator
from collections import namedtuple

# A Schedule is: {"root": node, "steps": List[List[Tuple[u,v]]]}
# where steps[t] is the list of directed edges executed at timestep t.

Schedule = namedtuple("Schedule", ["root", "steps"])


def enumerate_all_schedules(G: nx.Graph, max_steps: int) -> Iterable[Dict[str, object]]:
    """
    Enumerate every valid gather schedule for each spanning tree of G and each root,
    subject to a horizon of `max_steps`. Each yielded item is:
        {"root": r, "steps": steps}
    where `steps` is a list of length max_steps; steps[t] is a list of directed (u,v).
    """
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected.")

    # Iterate all spanning trees (arbitrary order if unweighted).
    for T in SpanningTreeIterator(G, minimum=True):
        T = nx.Graph(T)
        adj = T.adj

        for root in T.nodes:
            # Compute subtree heights for this root (post-order).
            height = _compute_subtree_heights(adj, root)
            # Root-level feasibility: need at least its subtree height worth of steps.
            if max_steps < height[root]:
                continue

            # Enumerate schedules that gather to `root` within max_steps.
            for steps in _gather_subtree_schedules(
                adj, height, root, parent=None, max_steps_left=max_steps
            ):
                yield Schedule(root, steps)


def _compute_subtree_heights(
    adj: Dict[Hashable, Dict[Hashable, dict]], root: Hashable
) -> Dict[Hashable, int]:
    """
    Return subtree heights for a fixed root: height[u] = max distance from u to
    any descendant (leaf has height 0). Uses post-order DFS rooted at `root`.
    """
    height: Dict[Hashable, int] = {}

    def dfs(u: Hashable, parent: Hashable) -> int:
        h = 0
        for v in adj[u]:
            if v == parent:
                continue
            h = max(h, 1 + dfs(v, u))
        height[u] = h
        return h

    dfs(root, None)
    return height


def _gather_subtree_schedules(
    adj: Dict[Hashable, Dict[Hashable, dict]],
    height: Dict[Hashable, int],
    u: Hashable,
    parent: Hashable,
    max_steps_left: int,
) -> Iterable[List[List[Tuple[Hashable, Hashable]]]]:
    """
    Enumerate schedules that gather all tokens in the subtree rooted at `u`
    (w.r.t. the chosen global root) *to u* within `max_steps_left` steps.

    Returns a list of length max_steps_left; at time t we output a set of edges
    directed towards u. This function does NOT schedule a move from u to `parent`.
    """
    # Height-based feasibility: you need at least the subtree height worth of steps.
    if max_steps_left < height[u]:
        return  # infeasible

    # Children (neighbors except parent)
    children = [v for v in adj[u] if v != parent]
    k = len(children)

    # Base case: leaf-subtree needs no internal moves.
    if k == 0:
        yield [[] for _ in range(max_steps_left)]
        return

    # Choose distinct transfer times a_i in [0, max_steps_left)
    time_slots = range(max_steps_left)
    for assign in it.permutations(time_slots, k):
        # Quick feasibility: each child must finish its own subtree by time a_i.
        if any(a_i < height[child] for child, a_i in zip(children, assign)):
            continue

        # Build child iterators; each yields schedules of length a_i.
        per_child_iters = [
            _gather_subtree_schedules(adj, height, child, u, a_i)
            for child, a_i in zip(children, assign)
        ]

        # Combine (Cartesian product) across children and overlay their steps.
        for combo in it.product(*per_child_iters):
            steps = [[] for _ in range(max_steps_left)]

            # Merge each child's internal schedule, then add final transfer at a_i.
            for (child, a_i), child_sched in zip(zip(children, assign), combo):
                # Internal moves: t in [0, a_i)
                for t in range(len(child_sched)):
                    steps[t].extend(child_sched[t])
                # Final transfer at time a_i: move everything child -> u.
                steps[a_i].append((child, u))

            yield steps
