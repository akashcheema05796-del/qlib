# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
HMM state label aligner for walk-forward regime backtesting.

When an HMM is re-fitted on a new window, state indices are arbitrary —
"state 0" in window N+1 may represent a completely different market regime
than "state 0" in window N (label switching). This module resolves that by
aligning new emission means to a stored reference via the Hungarian algorithm.

Usage
-----
::

    aligner = HMMLabelAligner()

    # First window: set the reference
    aligner.fit(model_window1)

    # Subsequent windows: get permutation
    perm = aligner.align(model_window2)
    # perm[j] = i means new state j should be relabelled as i
    # Apply to a states array:
    aligned_states = perm[raw_states]
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...log import get_module_logger

logger = get_module_logger("HMMLabelAligner")


class HMMLabelAligner:
    """Align HMM state labels across sequential re-fits.

    Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
    to find the permutation of new state indices that minimises the sum of
    Euclidean distances between reference and new emission means.

    Parameters
    ----------
    distance : {"euclidean", "cosine"}
        Distance metric used to build the cost matrix.

    Attributes
    ----------
    reference_means_ : np.ndarray or None
        Emission means of the reference model, shape (K, D).
    n_alignments_ : int
        Number of times ``align()`` has been called.
    """

    def __init__(self, distance: str = "euclidean"):
        if distance not in ("euclidean", "cosine"):
            raise ValueError(f"distance must be 'euclidean' or 'cosine', got '{distance}'")
        self.distance = distance
        self.reference_means_: Optional[np.ndarray] = None
        self.n_alignments_: int = 0

    def fit(self, model) -> "HMMLabelAligner":
        """Store emission means of *model* as the reference.

        Call this once on the first fitted HMM.  All subsequent calls to
        ``align()`` will map new models onto this reference.

        Parameters
        ----------
        model : fitted GaussianHMM
            Must have a ``.means_`` attribute, shape (K, D).
        """
        self.reference_means_ = model.means_.copy()
        self.n_alignments_ = 0
        logger.info(
            "HMMLabelAligner: reference set with K=%d states, D=%d features.",
            model.n_components, model.means_.shape[1],
        )
        return self

    def align(self, model) -> np.ndarray:
        """Return a permutation array mapping new state indices to reference labels.

        Parameters
        ----------
        model : fitted GaussianHMM
            Must have the same number of states K as the reference.

        Returns
        -------
        perm : np.ndarray, shape (K,)
            ``perm[j] = i`` means new state ``j`` should be relabelled as ``i``.
            Apply to a state sequence via ``perm[states_array]``.

        Notes
        -----
        The reference means are updated to the aligned new means after each
        call so the aligner tracks gradual regime drift over time.
        """
        if self.reference_means_ is None:
            logger.warning("HMMLabelAligner.fit() not called; using model as reference.")
            self.fit(model)
            return np.arange(model.n_components)

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:
            raise ImportError(
                "scipy is required for HMMLabelAligner. "
                "Install it with: pip install scipy"
            ) from e

        ref = self.reference_means_
        new = model.means_
        K = len(ref)

        if len(new) != K:
            raise ValueError(
                f"Model has {len(new)} states but reference has {K}. "
                "Re-initialise the aligner for a different state count."
            )

        cost = self._cost_matrix(ref, new)
        row_ind, col_ind = linear_sum_assignment(cost)

        # perm[j] = i means new-state j maps to reference-state i
        perm = np.empty(K, dtype=int)
        for r, c in zip(row_ind, col_ind):
            perm[c] = r

        total_cost = cost[row_ind, col_ind].sum()
        logger.info(
            "HMMLabelAligner: alignment %d complete.  "
            "Permutation=%s  total_distance=%.4f",
            self.n_alignments_ + 1, perm.tolist(), total_cost,
        )

        # Update reference to aligned new means (tracks gradual drift)
        aligned_means = new[perm]
        self.reference_means_ = aligned_means
        self.n_alignments_ += 1

        return perm

    def _cost_matrix(self, ref: np.ndarray, new: np.ndarray) -> np.ndarray:
        """Build K×K cost matrix between reference and new emission means."""
        K = len(ref)
        cost = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if self.distance == "euclidean":
                    cost[i, j] = np.linalg.norm(ref[i] - new[j])
                else:  # cosine
                    a, b = ref[i], new[j]
                    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
                    cost[i, j] = 1.0 - np.dot(a, b) / denom
        return cost

    def apply_permutation(
        self,
        states: np.ndarray,
        perm: np.ndarray,
    ) -> np.ndarray:
        """Apply a permutation to a states array.

        Parameters
        ----------
        states : np.ndarray of int, shape (T,)
        perm : np.ndarray of int, shape (K,)
            As returned by ``align()``.

        Returns
        -------
        aligned_states : np.ndarray of int, shape (T,)
        """
        return perm[states]
