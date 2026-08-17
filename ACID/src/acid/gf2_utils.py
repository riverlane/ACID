from __future__ import annotations

import numpy as np


def gf2_rref_colwise(M: list[list[int]], clear_upper_triangle=True) -> list[list[int]]:
    """Return column-wise row-reduced echelon form over GF(2). Return pivot cols.

    M is a list of rows of equal length containing 0/1.
    """
    if not M:
        return []
    # Transpose M to get columns as rows
    m, n = len(M), len(M[0])
    A = [row[:] for row in M]
    pivots: list[int] = []
    for r in range(m):
        # Find pivot in row r
        pivot_c = None
        for c in range(n):
            if A[r][c] & 1:
                pivot_c = c
                pivots.append(pivot_c)
                break
        if pivot_c is None:
            # R is linearly dependent on previous rows
            continue

        # Eliminate other rows at column r
        for rr in range(m):
            if (not clear_upper_triangle) and rr < r:
                continue
            if rr != r and (A[rr][pivot_c] & 1):
                for j in range(n):
                    A[rr][j] ^= A[r][j]
        # assert gf2_rank(A[:r+1]) == gf2_rank(M[:r + 1]) == gf2_rank(A[:r+1] + M[:r + 1])
        if r == m:
            break

    return A, pivots


def gf2_rref_rowwise(
    M: list[list[int]], clear_upper_triangle=True
) -> tuple[list[list[int]], list[int]]:
    """Return row-reduced echelon form over GF(2) and list of pivot columns.

    M is a list of rows of equal length containing 0/1.
    """
    if not M:
        return [], []
    A = [row[:] for row in M]
    m = len(A)
    n = len(A[0])
    pivots: list[int] = []
    r = 0
    for c in range(n):
        # find pivot
        pivot = None
        for i in range(r, m):
            if A[i][c] & 1:
                pivot = i
                break
        if pivot is None:
            # C is linearly dependent on previous columns
            continue
        # swap
        A[r], A[pivot] = A[pivot], A[r]
        # eliminate other rows at column c
        for i in range(m):
            if (not clear_upper_triangle) and i < r:
                continue
            if i != r and (A[i][c] & 1):
                for j in range(c, n):
                    A[i][j] ^= A[r][j]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A, pivots


def gf2_rank(M: list[list[int]]) -> int:
    if not M:
        return 0
    _, piv = gf2_rref_rowwise(M)
    return len(piv)


def gf2_nullspace(M: list[list[int]]) -> list[list[int]]:
    """Return a basis for the right nullspace of M over GF(2).

    Each vector x satisfies M x = 0 (treating rows of M and column vector x).
    """
    if not M:
        return []
    # RREF of M
    R, piv = gf2_rref_rowwise(M)
    m = len(R)
    n = len(R[0]) if m else 0
    pivot_pos = set(piv)
    free_cols = [j for j in range(n) if j not in pivot_pos]
    basis: list[list[int]] = []
    # For each free variable, set it to 1 and solve for pivot vars
    for f in free_cols:
        x = [0] * n
        x[f] = 1
        for i, p in enumerate(piv):
            # Row i expresses: x[p] + sum_j R[i][j] x[j] = 0
            x[p] = 0
            # Only need contributions from free columns
            for j in free_cols:
                if R[i][j] & 1 and x[j] & 1:
                    x[p] ^= 1
        basis.append(x)
    return basis


def gf2_left_nullspace(M: list[list[int]]) -> list[list[int]]:
    """Return a basis for the left nullspace of M, i.e., vectors w with w M = 0.

    Compute nullspace of M^T.
    """
    if not M:
        return []
    # Transpose M to get M^T (n x m)
    m = len(M)
    n = len(M[0]) if m else 0
    MT: list[list[int]] = [[0] * m for _ in range(n)]
    for i in range(m):
        row = M[i]
        for j in range(n):
            if row[j] & 1:
                MT[j][i] ^= 1
    # Right nullspace of M^T gives left nullspace of M
    return gf2_nullspace(MT)


def gf2_bidiagonalize(
    A: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], int]:
    """
    Perform GF(2) row/column elimination to diagonalize A via
    U * A * V^T = diag(I_r, 0), returning (U, V, r).

    - U is a square matrix (rows x rows) of row operations.
    - V is a square matrix (cols x cols) of column operations.
    - r is the rank (number of ones placed on the diagonal).
    """
    b = len(A)
    a = len(A[0]) if b else 0
    M = [row[:] for row in A]
    U = [[1 if i == j else 0 for j in range(b)] for i in range(b)]
    V = [[1 if i == j else 0 for j in range(a)] for i in range(a)]
    i = j = 0
    r = 0
    while i < b and j < a:
        # find first nonzero entry in submatrix M[i:b, j:a], this is the pivot
        pi = pj = None
        found = False
        for ii in range(i, b):
            for jj in range(j, a):
                if M[ii][jj] & 1:
                    pi, pj = ii, jj
                    found = True
                    break
            if found:
                break
        if not found:
            # no more pivots; we're done
            break
        # swap found pibot into position (i, j)
        if pi != i:
            # swap rows i and pi in M and U
            M[i], M[pi] = M[pi], M[i]
            U[i], U[pi] = U[pi], U[i]
        if pj != j:
            # swap columns j and pj in M and V
            for rr in range(b):
                M[rr][j], M[rr][pj] = M[rr][pj], M[rr][j]
            for rr in range(a):
                V[rr][j], V[rr][pj] = V[rr][pj], V[rr][j]
        # Clear column j except at row i
        for ii in range(b):
            if ii != i and (M[ii][j] & 1):
                for jj in range(a):
                    M[ii][jj] ^= M[i][jj]
                for jj in range(b):
                    U[ii][jj] ^= U[i][jj]
        # Clear row i except at col j
        for jj in range(a):
            if jj != j and (M[i][jj] & 1):
                for rr in range(b):
                    M[rr][jj] ^= M[rr][j]
                for rr in range(a):
                    V[rr][jj] ^= V[rr][j]
        i += 1
        j += 1
        r += 1
    return U, V, r


def gf2_is_in_span(v: list[int], basis: list[list[int]]) -> bool:
    """Check if vector v is in the span of basis rows over GF(2)."""
    if not basis:
        return all(x == 0 for x in v)
    # Augment basis temporarily and check rank doesn't increase
    n = len(basis[0])
    if len(v) != n:
        raise ValueError("Dimension mismatch in gf2_is_in_span")
    rank_before = gf2_rank(basis)
    rank_after = gf2_rank(basis + [v])
    return rank_after == rank_before


def gf2_are_not_in_span(rows: list[list[int]], basis: list[list[int]]) -> bool:
    """Check if matrix has null intersection with basis rows over GF(2)."""
    if not basis:
        return all(x == 0 for x in rows)
    # Augment basis temporarily and check rank doesn't increase
    n = len(basis[0])
    if len(rows) != n:
        raise ValueError("Dimension mismatch in gf2_is_in_span")
    rank_basis = gf2_rank(basis)
    rank_rows = gf2_rank(rows)
    rank_total = gf2_rank(basis + rows)
    return rank_total == rank_basis + rank_rows


def gf2_rank_normal_numpy(
    A_in,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute the rank normal form of matrix A modulo 2 and track transformations.
    Returns R_inv, C_inv and R, C such that A = R @ rank_normal @ C and R_inv @ A @ C_inv = rank_normal.
    Very similar to gf2_bidiagonalize, but returns the rank normal form instead of just the rank.

    Args:
        A_in (np.ndarray): Input matrix with integer or boolean dtype

    Returns:
        tuple: (rank_normal, R_inv, R, C_inv, C, rank)
            rank_normal (np.ndarray): Rank normal form of A
            R_inv (np.ndarray): Inverse row transformation matrix in GL(m,2)
            R (np.ndarray): Row transformation matrix in GL(m,2)
            C_inv (np.ndarray): Inverse column permutation matrix
            C (np.ndarray): Column permutation matrix
            rank (int): Rank of the matrix
    """
    # Make a copy, and convert to int8 if not already

    if A_in.dtype.kind not in "bi":
        raise ValueError("Input array must have integer dtype")
    A = np.array(A_in, dtype=np.int8)
    m, n = A.shape

    # Initialize identity matrices for R and C
    R_inv = np.eye(m, dtype=np.int8)
    R = np.eye(m, dtype=np.int8)
    C_inv = np.eye(n, dtype=np.int8)
    C = np.eye(n, dtype=np.int8)

    rank_so_far = 0

    while True:
        # Find the next pivot
        possible_pivots = np.nonzero(np.sum(A[rank_so_far:, rank_so_far:], axis=0))[0]

        # No possible pivots? We're done.
        if len(possible_pivots) == 0:
            return A, R_inv, R, C_inv, C, rank_so_far

        pivot_column = possible_pivots[0] + rank_so_far
        pivot_row = np.nonzero(A[rank_so_far:, pivot_column])[0][0] + rank_so_far

        # Permute pivot row into place
        A[[pivot_row, rank_so_far]] = A[[rank_so_far, pivot_row]]
        R_inv[[pivot_row, rank_so_far]] = R_inv[[rank_so_far, pivot_row]]
        R[:, [pivot_row, rank_so_far]] = R[:, [rank_so_far, pivot_row]]

        # Permute pivot column into place
        A[:, [pivot_column, rank_so_far]] = A[:, [rank_so_far, pivot_column]]
        C_inv[:, [pivot_column, rank_so_far]] = C_inv[:, [rank_so_far, pivot_column]]
        C[[pivot_column, rank_so_far]] = C[[rank_so_far, pivot_column]]

        # Find which rows need to be flipped (in-place pivoting workaround)
        A[rank_so_far, rank_so_far] = 0
        targets = np.nonzero(A[:, rank_so_far])[0]
        # (Undo the terrible hack)
        A[rank_so_far, rank_so_far] = 1

        # Flip the rows
        A[targets] ^= A[rank_so_far]
        R_inv[targets] ^= R_inv[rank_so_far]
        R[:, rank_so_far] ^= np.sum(R[:, targets], axis=1) % 2
        rank_so_far += 1


def gf2_get_generator_coefficients(
    G: list[list[int]], v: list[list[int]]
) -> list[list[int]] | None:
    if not G:
        return None if any(any(row) for row in v) else []
    m = len(G)
    n = len(G[0])
    if any(len(row) != n for row in v):
        raise ValueError("Dimension mismatch in gf2_get_generator_coefficients")
    # RREF G alone so v rows never get swapped into G positions.
    aug = [G[i][:] + [1 if j == i else 0 for j in range(m)] for i in range(m)]
    rref_G, pivots = gf2_rref_rowwise(aug)
    pivot_map: dict[int, int] = {c: r for r, c in enumerate(pivots)}
    # Reduce each v row against the G basis and collect coefficients.
    coefficients = []
    for v_row in v:
        row = v_row[:] + [0] * m
        for c in range(n):
            if (row[c] & 1) and c in pivot_map:
                r_idx = pivot_map[c]
                for k in range(n + m):
                    row[k] ^= rref_G[r_idx][k]
        if any(row[j] for j in range(n)):  # non-zero residual → not in span
            return None
        coefficients.append(row[n:])
    return coefficients
