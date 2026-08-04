"""CP-SAT solver for qubit routing via swap layers on a grid.

Given an embedding with dead qubits/connections, finds a minimal sequence of
parallel swap layers that moves a subset of qubits to target positions.
"""

from __future__ import annotations

from time import time

from ortools.sat.python import cp_model

from acid.embedding import SquareGridEmbedding


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

    The swap is performed in one of the 'end'-cycle' state of the code,
    ie when some gauges have been contracted and others have been
    expanded. The qubits that the gauges have been contracted onto are
    refered to here as root qubits and the qubits that the other gauges have
    been expanded onto are refered to as data qubits.

    This function computes a sequence of swap layers, ie layers of swaps on
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
            to end up in any particular position, providng they don't
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
    # Skip if the destination is dead or dont_care (can't place there).
    target: dict[int, int] = {}
    for q, dest in zip(all_qubits, shifted):
        if dest in dead_positions:
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

    # Adjacency: position -> list of (edge_index, neighbor)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(N)]
    for idx, (a, b) in enumerate(edges):
        adj[a].append((idx, b))
        adj[b].append((idx, a))

    # Precompute full all-pairs shortest paths (one BFS per position).  This is
    # O(N*(N+E)) but done once; the result is reused across all layer counts to
    # tightly restrict perm variable domains inside the CP-SAT model.
    all_dist: list[dict[int, int]] = [_bfs_from(adj, i) for i in range(N)]

    # Binary search on number of layers needed
    for num_layers in range(1, max_layers + 1):
        result = _solve_for_layers(N, edges, adj, target, all_dist, num_layers, time_limit_s)
        if result is not None:
            return result
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


def _bfs_from(
    adj: list[list[tuple[int, int]]],
    source: int,
) -> dict[int, int]:
    """Return shortest-path distances from source to all reachable nodes."""
    dist: dict[int, int] = {source: 0}
    frontier = [source]
    while frontier:
        next_frontier = []
        for node in frontier:
            d = dist[node]
            for _, nb in adj[node]:
                if nb not in dist:
                    dist[nb] = d + 1
                    next_frontier.append(nb)
        frontier = next_frontier
    return dist


def _solve_for_layers(
    N: int,
    edges: list[tuple[int, int]],
    adj: list[list[tuple[int, int]]],
    target: dict[int, int],
    all_dist: list[dict[int, int]],
    num_layers: int,
    time_limit_s: float,
) -> list[list[tuple[int, int]]] | None:
    """Solve the swap routing problem for a fixed number of layers.
    
    Args:
        N: Total number of positions.
        edges: List of (i, j) undirected edges in the grid.
        adj: Adjacency list of edges for each position.
        target: Mapping from position -> qubit that must end up there.
        all_dist: Precomputed shortest-path distances between all pairs of positions.
        num_layers: Number of swap layers to use.
        time_limit_s: CP-SAT solver time limit in seconds.
    """
    # qubit -> its required destination (only for non-trivial constraints)
    qubit_to_dest: dict[int, int] = {src: dest for dest, src in target.items() if src != dest}

    model = cp_model.CpModel()
    E = len(edges)

    # curr_pos[t][i] = which qubit is at position i after t layers.
    # Domain of curr_pos[t][i] is restricted to qubits that:
    #   (a) can reach position i from their starting point in t swaps, AND
    #   (b) if constrained, can still reach their destination from i in
    #       num_layers-t swaps (the corridor/hourglass intersection).
    # For t=0 this collapses to the identity {i}.
    curr_pos: list[list[cp_model.IntVar]] = []
    for t in range(num_layers + 1):
        layer_vars = []
        for i in range(N):
            if t == 0:
                v = model.new_int_var(i, i, f"perm_0_{i}")
            else:
                valid = []
                for q, d_iq in all_dist[i].items():
                    if d_iq > t:
                        continue
                    dest = qubit_to_dest.get(q)
                    if dest is not None and all_dist[i].get(dest, N) > num_layers - t:
                        continue
                    valid.append(q)
                if not valid:
                    return None
                v = model.new_int_var_from_domain(
                    cp_model.Domain.from_values(valid), f"perm_{t}_{i}"
                )
            layer_vars.append(v)
        curr_pos.append(layer_vars)
    # Swap decision variables
    swap: list[list[cp_model.IntVar]] = []
    for t in range(num_layers):
        layer_swaps = []
        for e_idx in range(E):
            s = model.new_bool_var(f"swap_{t}_{e_idx}")
            layer_swaps.append(s)
        swap.append(layer_swaps)

    # Source variable: src[t][i] = which position i pulls its value from
    src: list[list[cp_model.IntVar]] = []
    for t in range(num_layers):
        layer_src = []
        for i in range(N):
            # Domain: self or any live neighbor
            possible = [i] + [nb for (_, nb) in adj[i]]
            s = model.new_int_var_from_domain(cp_model.Domain.from_values(possible), f"src_{t}_{i}")
            layer_src.append(s)
        src.append(layer_src)

    # Link swap vars to src vars
    for t in range(num_layers):
        for e_idx, (a, b) in enumerate(edges):
            # If swap[t][e_idx] is active: src[t][a] = b, src[t][b] = a
            model.add(src[t][a] == b).only_enforce_if(swap[t][e_idx])
            model.add(src[t][b] == a).only_enforce_if(swap[t][e_idx])

        # If no swap touches position i: src[t][i] = i
        for i in range(N):
            # Edges incident to position i
            incident = [swap[t][e_idx] for (e_idx, _) in adj[i]]
            # No edges incident - cannot move
            if not incident:
                model.add(src[t][i] == i)
            # For each incident edge, if a swap on that edge is not active, then
            # src[t][i] != neighbor
            else:
                for e_idx, nb in adj[i]:
                    model.add(src[t][i] != nb).only_enforce_if(swap[t][e_idx].negated())

        # At most one swap per position per layer
        for i in range(N):
            incident = [swap[t][e_idx] for (e_idx, _) in adj[i]]
            if incident:
                # At most one of the incident swaps can be active for qubit i
                model.add_at_most_one(incident)

    # Transition: perm[t+1][i] = perm[t][src[t][i]]
    for t in range(num_layers):
        for i in range(N):
            model.add_element(src[t][i], curr_pos[t], curr_pos[t + 1][i])

    # AllDifferent per layer for faster propagation
    for t in range(num_layers + 1):
        model.add_all_different(curr_pos[t])

    # AllDifferent on src per layer (each source used exactly once)
    for t in range(num_layers):
        model.add_all_different(src[t])

    # Target constraints
    for pos, qubit in target.items():
        model.add(curr_pos[num_layers][pos] == qubit)

    # Minimize total swaps (optional but helps find clean solutions)
    total_swaps = sum(swap[t][e] for t in range(num_layers) for e in range(E))
    model.minimize(total_swaps)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result: list[list[tuple[int, int]]] = []
        for t in range(num_layers):
            layer_result: list[tuple[int, int]] = []
            for e_idx, (a, b) in enumerate(edges):
                if solver.value(swap[t][e_idx]):
                    layer_result.append((a, b))
            result.append(layer_result)
        return result
    return None
