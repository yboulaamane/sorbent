"""Butina clustering.

    from rdkit.ML.Cluster import Butina
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

**Feed ClusterData a square matrix, not the flat lower triangle.** This is the
single most consequential thing in this file, and it is invisible unless you
read RDKit's source. ``Butina.ClusterData`` is pure Python, and given a 1D
triangle it begins:

    dist_matrix = np.zeros((nPts, nPts))    # full n x n, float64
    idx = np.tril_indices(nPts, -1)         # two int64 arrays of n(n-1)/2
    dist_matrix[idx] = data
    dist_matrix += dist_matrix.T

So a compact float32 triangle is immediately expanded into a float64 square
*plus* a pair of index arrays twice its size. Given a correctly shaped n x n
array instead, it skips all of that and uses the array as-is - keeping your
dtype. Measured peak allocation, identical clusterings from both:

    n = 2,000     1D 104 MB     2D  17 MB
    n = 4,000     1D 416 MB     2D  65 MB
    n = 6,000     1D 936 MB     2D 147 MB     (6.4x)

Projected at the 20,000 cap: the 1D path wants ~7.2 GB (3.2 square + 3.2
indices + 0.8 triangle); the 2D float32 path wants 1.6 GB and nothing else.

float32 rather than float64 halves it again and changes nothing: Tanimoto
lives in [0, 1], and clusterings computed at both widths were identical. Only
a distance within ~1e-7 of the cutoff could flip, which is neither systematic
nor meaningful.

Filling the matrix row by row matters too - accumulating a Python list of
floats first would cost ~25 bytes an entry (8 for the pointer, plus a 24-byte
float object), which is 6.4 GB at the cap before RDKit sees any of it.

**Butina is O(n^2) in both time and memory**, and no dtype fixes that:

    n =  6,000    12 s    0.15 GB
    n = 20,000  ~140 s    1.60 GB   <- MAX_EXACT_CLUSTER_SIZE
    n = 250,000    ---     100 GB   <- the library cap. Not happening.

So exact Butina cannot reach the library cap and never will. Above the cap the
options are, in increasing order of effort:

  - Cluster the Murcko scaffolds instead of the molecules. Usually 10-50x
    fewer, and ``scaffolds.py`` already computes them.
  - Sphere exclusion / leader-follower, which is O(n*k) and streams:
    ``rdSimDivPickers.LeaderPicker``. The right answer for a real 200k library.
  - Cluster a random or diverse subset and assign the rest by nearest centroid.

``build_distance_matrix`` returns the flat lower triangle because that is the
documented interchange form and what most code expects; ``butina_cluster``
does not use it, building the square form directly for the reason above. The
flat layout, for reference, is: for molecule i, distances to 0..i-1,
concatenated. Row 0 is empty, which is why ``fingerprints.bulk_tanimoto``
returns [] for an empty target list.

Sorbent's contract: cluster 0 is the largest cluster, and within a cluster the
first element is the Butina centroid. RDKit happens to return clusters
largest-first already, but that is not documented as a guarantee, so the sort
here is explicit. It is a stable sort, so equal-sized clusters keep Butina's
own ordering and the result is deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit.ML.Cluster import Butina

from sorbent.chem.fingerprints import bulk_tanimoto

#: Above this, exact Butina clustering is refused rather than attempted. The
#: pipeline turns this into a clear job error naming the alternative. Chosen
#: from the measurements above: ~140 s and 1.6 GB, both tolerable inside a
#: worker process; 50,000 would be ~15 min and 10 GB, which is not.
MAX_EXACT_CLUSTER_SIZE = 20_000


def build_distance_matrix(fingerprints: list[Any]) -> NDArray[np.float32]:
    """Flattened lower-triangle Tanimoto *distance* matrix (1 - similarity).

    Returns numpy rather than a list - see the module docstring; at the cap the
    list form is 6.4 GB against 0.80 GB. ``len()`` and iteration behave the
    same.

    Provided as the conventional interchange form. ``butina_cluster`` does NOT
    use it: handing this shape to ClusterData makes it allocate a float64
    square plus index arrays, four times the peak of building the square
    directly.
    """
    n = len(fingerprints)
    matrix = np.empty(n * (n - 1) // 2, dtype=np.float32)
    position = 0
    for i in range(1, n):
        # Row i is this molecule against every earlier one. bulk_tanimoto is a
        # C++ loop and 17x faster than a comprehension over tanimoto().
        row = np.asarray(bulk_tanimoto(fingerprints[i], fingerprints[:i]), dtype=np.float32)
        matrix[position : position + i] = 1.0 - row
        position += i
    return matrix


def _square_distance_matrix(fingerprints: list[Any]) -> NDArray[np.float32]:
    """Full symmetric n x n distance matrix, float32.

    Twice the entries of the lower triangle, but it is what ClusterData wants:
    given this shape it uses the array directly instead of rebuilding a float64
    square and a pair of tril_indices arrays from the flat form. Net effect at
    the cap is 1.6 GB against ~7.2 GB.

    Each row is written to both triangles as it is computed, so the transient
    is a single row rather than a second full matrix.
    """
    n = len(fingerprints)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(1, n):
        row = np.asarray(bulk_tanimoto(fingerprints[i], fingerprints[:i]), dtype=np.float32)
        row = 1.0 - row
        matrix[i, :i] = row
        matrix[:i, i] = row
    return matrix


def butina_cluster(fingerprints: list[Any], cutoff: float = 0.35) -> list[list[int]]:
    """Cluster and return lists of indices, largest cluster first.

    ``cutoff`` is a *distance* (1 - Tanimoto), so 0.35 groups molecules more
    similar than 0.65.

    Refuses above MAX_EXACT_CLUSTER_SIZE rather than running out of memory an
    hour in. The check is first, before the fingerprints are touched at all,
    so an oversized call fails immediately and cheaply.
    """
    n = len(fingerprints)
    if n > MAX_EXACT_CLUSTER_SIZE:
        raise ValueError(
            f"exact Butina clustering refused for {n:,} molecules: the limit is "
            f"{MAX_EXACT_CLUSTER_SIZE:,}, above which the distance matrix exceeds "
            f"a gigabyte and O(n^2) time becomes minutes. Cluster the Murcko "
            f"scaffolds instead, or use sphere exclusion "
            f"(rdSimDivPickers.LeaderPicker), which is O(n*k) and streams."
        )
    # Butina on an empty or single-molecule input is a degenerate case not
    # worth handing to RDKit.
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Square, not the flat triangle from build_distance_matrix - see the
    # module docstring. This is a 6.4x difference in peak memory.
    distances = _square_distance_matrix(fingerprints)
    clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
    # Stable, so equal-sized clusters keep Butina's ordering - deterministic.
    return sorted((list(cluster) for cluster in clusters), key=len, reverse=True)


def assign_clusters(clusters: list[list[int]], n_molecules: int) -> list[int]:
    """Invert the cluster lists into a per-molecule cluster id array.

    Butina partitions completely - every molecule lands in exactly one cluster
    - so anything else means the caller built the lists wrong, and this raises
    rather than returning an array with holes in it that would surface much
    later as a null cluster_id on an arbitrary molecule.
    """
    assigned: list[int | None] = [None] * n_molecules
    for cluster_id, members in enumerate(clusters):
        for index in members:
            if not 0 <= index < n_molecules:
                raise ValueError(
                    f"cluster {cluster_id} contains index {index}, outside 0..{n_molecules - 1}"
                )
            if assigned[index] is not None:
                raise ValueError(
                    f"molecule {index} is in both cluster {assigned[index]} "
                    f"and cluster {cluster_id}"
                )
            assigned[index] = cluster_id

    missing = [i for i, value in enumerate(assigned) if value is None]
    if missing:
        raise ValueError(
            f"{len(missing)} molecule(s) were in no cluster, first at index {missing[0]}"
        )
    return [value for value in assigned if value is not None]
