from __future__ import annotations

from acid.scheduling.types import SyndromeExtractionLayer


def analyze_layers(
    product_members: dict[str, set[str]],
    layers: list[SyndromeExtractionLayer],
    *,
    interesting_labels: set[str] | None = None,
) -> dict:
    """
    Generic schedule analysis usable for any code and any schedule:
      - per-layer measured labels (optionally filtered to 'interesting_labels')
      - per-layer products in-process (any member measured this layer)
      - per-layer products completed (all members covered since last completion)
      - per-product completion layers

    Args:
      product_members: mapping product_label -> set(member_labels)
      layers: list of SyndromeExtractionLayer
      interesting_labels: optional subset of labels to display in 'measured'

    Returns a dict with keys: 'per_layer', 'product_completions'.
    """
    if interesting_labels is None:
        interesting_labels = set()

    # Progress since last completion per product
    progress: dict[str, set[str]] = {p: set() for p in product_members}
    completions: dict[str, list[int]] = {p: [] for p in product_members}

    per_layer = []
    for t, layer in enumerate(layers):
        # actual measured set
        all_measured = sorted(stab.label for stab in layer.chosen)
        mset = set(all_measured)
        # display-only measured (optional filter)
        if interesting_labels:
            measured = sorted(
                [lab for lab in all_measured if lab in interesting_labels]
            )
        else:
            measured = []

        in_process = []
        completed = []
        # Update progress
        for p, members in product_members.items():
            if mset & members:
                in_process.append(p)
            progress[p] |= mset & members
        # Check completions at end of layer
        for p, members in product_members.items():
            if progress[p] and progress[p] >= members:
                completions[p].append(t)
                completed.append(p)
                progress[p].clear()

        per_layer.append(
            {
                "measured": measured,
                "in_process": sorted(in_process),
                "completed": sorted(completed),
            }
        )

    return {
        "per_layer": per_layer,
        "product_completions": completions,
    }
