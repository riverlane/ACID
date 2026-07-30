from __future__ import annotations

from collections.abc import Iterable

from acid.embedding import Embedding


def data_bbox_xy(
    embedding: Embedding, data_ids: Iterable[int]
) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) bbox of given data qubits using embedding coords."""
    xs: list[float] = []
    ys: list[float] = []
    for q in data_ids:
        a, b, c = embedding.id_to_tuple(int(q))
        x, y = embedding.coords(a, b, c)
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), min(ys), max(xs), max(ys)


def place_ancillas_right_of_bbox(
    embedding: Embedding,
    data_ids: Iterable[int],
    k: int,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[list[int], list[int], list[tuple[int, float, float]]]:
    """
    Return (zero_ancilla_ids, plus_ancilla_ids, qubit_coord_triplets) placing 2k ancillas in a vertical column
    to the right of the data bbox with a blank row between each pair.

    - Ancilla ids are contiguous starting after max(data_ids).
    - Coordinates use (x_right + dx, y_min + i*dy) scheme.
    - The caller is responsible for emitting QUBIT_COORDS lines.
    """
    data_ids = list(map(int, data_ids))
    if data_ids:
        max_id = max(data_ids)
    else:
        max_id = -1
    start = max_id + 1
    zeros = [start + 2 * i for i in range(k)]
    plus = [start + 2 * i + 1 for i in range(k)]
    anc_ids = []
    anc_ids.extend(zeros)
    anc_ids.extend(plus)

    x0, y0, x1, y1 = data_bbox_xy(embedding, data_ids)
    x_right = x1 + abs(dx)
    y_min = y0
    coords: list[tuple[int, float, float]] = []
    for i in range(k):
        # place zero-ancilla then plus-ancilla with a blank row (dy) between pairs
        y_z = y_min + float(3 * i) * abs(dy)
        y_x = y_min + float(3 * i + 1) * abs(dy)
        coords.append((zeros[i], x_right, y_z))
        coords.append((plus[i], x_right, y_x))
    return zeros, plus, coords
