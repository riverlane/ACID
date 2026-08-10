"""CP-SAT solver for qubit routing via swap layers on a grid.

Given an embedding with dead qubits/connections, finds a minimal sequence of
parallel swap layers that moves a subset of qubits to target positions.
"""

from __future__ import annotations

from time import time
from typing import cast

from ortools.sat.python import cp_model

from acid.embedding import SquareGridEmbedding

import networkx as nx


def solve_swap_routing(
    *,
    embedding: SquareGridEmbedding,
    connections: list[tuple[int, int]],
    dead_positions: set[int],
    dead_connections: set[tuple[int, int]],
    offset: tuple[int, int],
    root_qubits: set[int] | None = None,
    max_layers: int = 10,
    time_limit_s: float = 60.0,
) -> list[list[tuple[int, int]]] | None:
    """Performs the swap routing to shift non-contracted qubits by the given
    lattice offset.

    The swap is performed in one of the 'end-cycle' state of the code,
    i.e. when some gauges have been contracted and others have been
    expanded. The qubits that the gauges have been contracted onto are
    referred to here as root qubits and the qubits that the other gauges have
    expanded onto are referred to as data qubits.

    This function computes a sequence of swap layers, i.e., layers of swaps on
    that moves the data qubits by the given offset on the periodic lattice,
    without care for where the root qubits end up.

    Args:
        embedding: The square grid embedding defining the lattice.
        connections: List of (i, j) undirected edges in the grid.
        dead_positions: Positions that cannot participate in swaps.
        dead_connections: Edges that cannot be used for swaps.
        offset: (da, db) shift on the periodic lattice.
        root_qubits: The qubits that the gauges have been contracted onto.
            These qubits are allowed to be moved around and not required
            to end up in any particular position, providing they don't
            conflict with data qubits. If None, no qubits are
            considered root qubits.
        max_layers: Maximum number of swap layers to try.
        time_limit_s: CP-SAT solver time limit in seconds.

    Returns:
        A list of swap layers (each layer is a list of (i,j) swaps performed),
        or None if no solution found within limits.

    Raises:
        ValueError: If a qubit is shifted to a dead position.
    """
    N = embedding.num_qubits
    da, db = offset

    if root_qubits is None:
        root_qubits = set()

    # Dead positions stay in place (enforced structurally: no edges touch them).
    # Compute target only for live positions that we care about.
    live = set(range(N)) - dead_positions - root_qubits
    all_qubits = sorted(live)
    shifted = embedding.shifted_positions(all_qubits, da, db)
    # target[dest] = q means qubit q must end up at position dest.
    # Destination cannot be a dead qubit because dead qubits cannot move so it would be invalid.
    target: dict[int, int] = {}
    for q, dest in zip(all_qubits, shifted):
        if dest in dead_positions:
            print(dead_positions)
            msg = f"Qubit {q} is shifted to position {dest} which is dead"
            raise ValueError(msg)
        target[dest] = q
    # Normalize connections and build adjacency
    edges: list[tuple[int, int]] = []
    dead_norm = {(min(a, b), max(a, b)) for (a, b) in dead_connections}
    for a, b in connections:
        e = (min(a, b), max(a, b))
        if e in dead_norm:
            continue
        if a in dead_positions or b in dead_positions:
            continue
        edges.append(e)

    # Precompute full all-pairs shortest paths (one BFS per position).  This is
    # O(N*(N+E)) but done once; the result is reused across all layer counts to
    # tightly restrict perm variable domains inside the CP-SAT model.
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    graph.add_edges_from(edges)
    all_dist_dict = nx.floyd_warshall(graph)
    # Convert dict of dicts to list of dicts, filtering out infinity distances
    all_dist: list[dict[int, int]] = []
    for i in range(N):
        d: dict[int, int] = {}
        if i in all_dist_dict:
            for j, dist_val in all_dist_dict[i].items():
                if dist_val != float("inf"):
                    d[j] = int(dist_val)
        all_dist.append(d)

    # Lower bound: max BFS distance any constrained qubit must travel
    lo = max(
        (all_dist[q].get(dest, max_layers + 1) for dest, q in target.items() if q != dest),
        default=1,
    )

    print(
        f"Swap routing: {len(all_qubits)} qubits, {len(edges)} edges, {len(target)} "
        f"constraints, lower bound {lo} layers, max {max_layers} layers"
    )
    # Linear search over number of layers, starting from lower bound.  Stop at first feasible solution.
    for num_layers in range(lo, max_layers + 1):
        result, status = _solve_for_layers(
            N=N,
            edges=edges,
            target=target,
            all_dist=all_dist,
            num_layers=num_layers,
            time_limit_s=time_limit_s,
        )
        if result is not None:
            return result
        elif status == cp_model.UNKNOWN:
            print(
                f"CP-SAT solver returned UNKNOWN for {num_layers} layers. This means "
                f"the solver hit the time limit without proving infeasibility."
                "No further layers will be attempted as higher number of layers will "
                "almost certainly be unknown. Please increase the time limit or reduce the "
                "number of layers."
            )
            break
        print(f"No solution found with {num_layers} layers, trying {num_layers + 1} layers...")
    return None


def verify_swap_routing(
    layers: list[list[tuple[int, int]]],
    num_positions: int,
    target: dict[int, int],
) -> bool:
    """Verify that applying the swap layers produces the expected target permutation.

    Args:
        layers: List of swap layers, each layer is a list of (i, j) swaps.
        num_positions: Total number of positions.
        target: Mapping from position -> qubit that must end up there.

    Returns:
        True if the final permutation matches target at all constrained positions.

    Raises:
        ValueError: If a swap layer is invalid (position swapped more than once).
    """
    perm = list(range(num_positions))
    for t, layer in enumerate(layers):
        touched: set[int] = set()
        for a, b in layer:
            if a in touched or b in touched:
                raise ValueError(f"Layer {t}: position swapped more than once ({a}, {b})")
            touched.add(a)
            touched.add(b)
            perm[a], perm[b] = perm[b], perm[a]

    for pos, qubit in target.items():
        if perm[pos] != qubit:
            return False
    return True


def _create_position_variables(
    model: cp_model.CpModel,
    N: int,
    all_dist: list[dict[int, int]],
    qubit_to_dest: dict[int, int],
    num_layers: int,
) -> list[list[cp_model.IntVar]] | None:
    """Create the curr_pos[t][i] variables for the CP-SAT model.

    Args:
        model: The CP-SAT model to add variables to.
        N: Total number of positions.
        all_dist: Precomputed shortest-path distances between all pairs of positions.
        qubit_to_dest: Mapping from qubit -> its required destination (only for non-trivial constraints).
        num_layers: Number of swap layers to use.
    Returns:
        A list of lists of IntVar, where curr_pos[t][i] is the variable representing which qubit
        is at position i after t layers.
    """
    curr_pos: list[list[cp_model.IntVar]] = []
    for layer_index in range(num_layers + 1):
        layer_vars = []
        for pos_index in range(N):
            if layer_index == 0:
                current_pos_var = model.new_int_var(pos_index, pos_index, f"perm_0_{pos_index}")
            else:
                valid_qubit_candidates = []
                for qubit_candidate, distance_to_qubit in all_dist[pos_index].items():
                    if distance_to_qubit > layer_index:
                        continue
                    dest = qubit_to_dest.get(qubit_candidate)
                    if (
                        dest is not None
                        and all_dist[pos_index].get(dest, N) > num_layers - layer_index
                    ):
                        continue
                    valid_qubit_candidates.append(qubit_candidate)
                if not valid_qubit_candidates:
                    return None
                current_pos_var = model.new_int_var_from_domain(
                    cp_model.Domain.from_values(valid_qubit_candidates),
                    f"perm_{layer_index}_{pos_index}",
                )
            layer_vars.append(current_pos_var)
        curr_pos.append(layer_vars)
    return curr_pos


def _create_swap_and_source_variables(
    model: cp_model.CpModel,
    N: int,
    edges: list[tuple[int, int]],
    num_layers: int,
) -> tuple[list[list[cp_model.IntVar]], list[list[cp_model.IntVar]]]:
    """Create swap and source variables and link them with constraints.

    Swap variables: swap[t][e] is a BoolVar indicating whether edge e is swapped at layer t.
    Source variables: src[t][i] is an IntVar indicating the source position of the
        qubit that ends up at position i after layer t.

    Args:
        model: The CP-SAT model to add variables and constraints to.
        N: Total number of positions.
        edges: List of (i, j) undirected edges.
        num_layers: Number of swap layers.

    Returns:
        A tuple (swap, src) where swap[t][e] is a BoolVar for each edge swap
        and src[t][i] is an IntVar for the source position of each qubit.
    """
    E = len(edges)

    swap: list[list[cp_model.IntVar]] = []
    for layer_index in range(num_layers):
        layer_swaps = [model.new_bool_var(f"swap_{layer_index}_{e_idx}") for e_idx in range(E)]
        swap.append(layer_swaps)

    # adjacency: position -> list of (edge_index, neighbor)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(N)]
    for idx, (a, b) in enumerate(edges):
        adj[a].append((idx, b))
        adj[b].append((idx, a))

    src: list[list[cp_model.IntVar]] = []
    for layer_index in range(num_layers):
        layer_src = []
        for pos_index in range(N):
            possible = [pos_index] + [nb for (_, nb) in adj[pos_index]]
            layer_src.append(
                model.new_int_var_from_domain(
                    cp_model.Domain.from_values(possible), f"src_{layer_index}_{pos_index}"
                )
            )
        src.append(layer_src)

    # Link swap vars to src vars
    for layer_index in range(num_layers):
        for e_idx, (a, b) in enumerate(edges):
            # If swap[t][e_idx] is active: src[t][a] = b, src[t][b] = a
            model.add(src[layer_index][a] == b).only_enforce_if(swap[layer_index][e_idx])
            model.add(src[layer_index][b] == a).only_enforce_if(swap[layer_index][e_idx])

        for pos_index in range(N):
            incident = [swap[layer_index][e_idx] for (e_idx, _) in adj[pos_index]]
            if not incident:
                # No edges incident - cannot move
                model.add(src[layer_index][pos_index] == pos_index)
            else:
                # For each incident edge, if a swap on that edge is not active,
                # then src[t][i] != neighbor
                for e_idx, nb in adj[pos_index]:
                    model.add(src[layer_index][pos_index] != nb).only_enforce_if(
                        swap[layer_index][e_idx].negated()
                    )

        # At most one swap per position per layer
        for pos_index in range(N):
            incident = [swap[layer_index][e_idx] for (e_idx, _) in adj[pos_index]]
            if incident:
                model.add_at_most_one(incident)

    return swap, src


def _solve_for_layers(
    N: int,
    edges: list[tuple[int, int]],
    target: dict[int, int],
    all_dist: list[dict[int, int]],
    num_layers: int,
    time_limit_s: float,
    *,
    optimize: bool = True,
) -> tuple[list[list[tuple[int, int]]] | None, cp_model.CpSolverStatus]:
    """Solve the swap routing problem for a fixed number of layers.

    Args:
        N: Total number of positions.
        edges: List of (i, j) undirected edges in the grid.
        target: Mapping from position -> qubit that must end up there.
        all_dist: Precomputed shortest-path distances between all pairs of positions.
        num_layers: Number of swap layers to use.
        time_limit_s: CP-SAT solver time limit in seconds.
        optimize: If True, minimize total swaps. If False, find any feasible solution.
    Returns:
        A list of lists of tuples, where the outer list is indexed by layer and the inner
        list contains the swaps (i, j) for that layer. Returns None if no solution is found.
    """
    # qubit -> its required destination (only for non-trivial constraints)
    qubit_to_dest: dict[int, int] = {src: dest for dest, src in target.items() if src != dest}

    model = cp_model.CpModel()

    # curr_pos[t][i] = which qubit is at position i after t layers.
    # Domain of curr_pos[t][i] is restricted to qubits that:
    #   (a) can reach position i from their starting point in t swaps, AND
    #   (b) if constrained, can still reach their destination from i in
    #       num_layers-t swaps (the corridor/hourglass intersection).
    # For t=0 this collapses to the identity {i}.

    # Position variables
    curr_pos = _create_position_variables(model, N, all_dist, qubit_to_dest, num_layers)
    if curr_pos is None:
        # No valid candidates for some position at some layer, so no solution is possible.
        return None, cp_model.INFEASIBLE

    # Swap and source variables
    swap, src = _create_swap_and_source_variables(model, N, edges, num_layers)

    # Transition: perm[t+1][i] = perm[t][src[t][i]]
    for layer_index in range(num_layers):
        for pos_index in range(N):
            model.add_element(
                src[layer_index][pos_index],
                curr_pos[layer_index],
                curr_pos[layer_index + 1][pos_index],
            )

    # AllDifferent per layer for faster propagation
    for layer_index in range(num_layers + 1):
        model.add_all_different(curr_pos[layer_index])

    # AllDifferent on src per layer (each source used exactly once)
    for layer_index in range(num_layers):
        model.add_all_different(src[layer_index])

    # Target constraints
    for pos, qubit in target.items():
        model.add(curr_pos[num_layers][pos] == qubit)

    # Decision strategy: branch on swap variables first
    all_swaps = [swap[t][e] for t in range(num_layers) for e in range(len(edges))]

    if optimize:
        total_swaps = sum(all_swaps)
        model.minimize(total_swaps)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s

    # TODO: add parameters to control number of threads, linearization level, etc.
    solver.parameters.num_workers = 8
    solver.parameters.linearization_level = 0
    model.add_decision_strategy(all_swaps, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

    start_time = time()
    status = solver.solve(model)
    print(f"CP-SAT solver finished in {time() - start_time:.2f}s with status {status}")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result: list[list[tuple[int, int]]] = []
        for layer_index in range(num_layers):
            layer_result: list[tuple[int, int]] = []
            for e_idx, (a, b) in enumerate(edges):
                if solver.value(swap[layer_index][e_idx]):
                    layer_result.append((a, b))
            result.append(layer_result)
        return result, status
    if status == cp_model.INFEASIBLE:
        print(f"CP-SAT solver reports INFEASIBLE for {num_layers} layers")
    else:
        print(f"CP-SAT solver reports status {status} for {num_layers} layers")
    return None, status
