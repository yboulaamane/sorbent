"""Butina clustering.

    from rdkit.ML.Cluster import Butina
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

``dists`` is the LOWER TRIANGLE of the distance matrix, flattened, excluding the
diagonal - i.e. for molecule i, distances to 0..i-1, concatenated. Building it
with BulkTanimotoSimilarity one row at a time is the standard idiom.

The thing to understand before you run this on a real library: Butina is
O(n^2) in both time and memory. At 200k molecules the distance matrix alone is
~2 x 10^10 entries. It will not fit. Your options, in increasing order of
effort:

  - Cap clustering at some n (say 20k) and return a clear error above it. This
    is the honest scaffold default; see MAX_EXACT_CLUSTER_SIZE.
  - Sphere-exclusion / leader-follower clustering, which is O(n*k) and
    streams. ``rdSimDivPickers.LeaderPicker`` does this and is the right answer
    for large libraries.
  - Cluster scaffolds instead of molecules - usually 10-50x fewer.

Winnow's contract: cluster 0 is the largest cluster, and within a cluster the
first element is the Butina centroid.
"""

from __future__ import annotations

from typing import Any

#: Above this, exact Butina clustering is refused rather than attempted. The
#: pipeline turns this into a clear job error naming the alternative.
MAX_EXACT_CLUSTER_SIZE = 20_000


def build_distance_matrix(fingerprints: list[Any]) -> list[float]:
    """Flattened lower-triangle Tanimoto *distance* matrix (1 - similarity)."""
    raise NotImplementedError


def butina_cluster(fingerprints: list[Any], cutoff: float = 0.35) -> list[list[int]]:
    """Cluster and return lists of indices, largest cluster first.

    Raise ValueError (not MemoryError, after the fact) when the input exceeds
    MAX_EXACT_CLUSTER_SIZE.
    """
    raise NotImplementedError


def assign_clusters(clusters: list[list[int]], n_molecules: int) -> list[int]:
    """Invert the cluster lists into a per-molecule cluster id array."""
    raise NotImplementedError
