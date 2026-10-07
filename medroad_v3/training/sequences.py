"""
Sequence construction for the recurrent and attention models.

The original ``build_sequences`` tiled a single window vector ``seq_len``
times, so the LSTM and the Transformer each received a constant sequence. A
recurrent model over a constant sequence has nothing to recur over and
collapses to a feed-forward network on the same input the gradient-boosted
trees already see, which makes the three learners far less complementary than
the ensemble rationale assumes.

This module builds genuine sequences: for each inference window, the preceding
``seq_len`` windows belonging to the same patient, in time order. Windows near
the start of a stay are left-padded by repeating the earliest available vector,
with a companion mask marking which timesteps are real.
"""
from __future__ import annotations

import logging

import numpy as np

from medroad_v3 import config

logger = logging.getLogger(__name__)


def build_patient_sequences(
    X: np.ndarray,
    patient_ids: np.ndarray,
    window_starts: np.ndarray,
    seq_len: int = config.SEQ_LEN,
    return_mask: bool = False,
):
    """
    Build (N, seq_len, D) sequences respecting patient boundaries.

    Parameters
    ----------
    X
        (N, D) window feature vectors.
    patient_ids
        (N,) patient identifier per row. Sequences never cross patients.
    window_starts
        (N,) sortable window timestamps, used to order windows within a patient.
    seq_len
        Number of timesteps per sequence, the current window plus the
        ``seq_len - 1`` preceding ones.
    return_mask
        If True also return a (N, seq_len) boolean mask that is False on
        left-padded timesteps.

    The output row order matches the input row order, so labels and any
    train/test split computed on X remain valid without reindexing.
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    pid = np.asarray(patient_ids)
    ts = np.asarray(window_starts)

    seqs = np.empty((n, seq_len, d), dtype=np.float32)
    mask = np.zeros((n, seq_len), dtype=bool)

    order = np.lexsort((ts, pid))
    pid_sorted = pid[order]
    boundaries = np.flatnonzero(pid_sorted[1:] != pid_sorted[:-1]) + 1
    groups = np.split(order, boundaries)

    short = 0
    for g in groups:
        for pos, row in enumerate(g):
            lo = max(0, pos - seq_len + 1)
            hist = g[lo:pos + 1]
            k = len(hist)
            if k < seq_len:
                short += 1
                pad = np.repeat(X[hist[0]][None, :], seq_len - k, axis=0)
                seqs[row] = np.vstack([pad, X[hist]])
                mask[row, seq_len - k:] = True
            else:
                seqs[row] = X[hist]
                mask[row] = True

    logger.info(
        "built %d sequences of length %d over %d patients (%d left-padded)",
        n, seq_len, len(groups), short,
    )
    return (seqs, mask) if return_mask else seqs


def sequence_variation(seqs: np.ndarray) -> float:
    """
    Mean per-feature standard deviation across the time axis.

    A value at or near zero means the sequences are constant and the recurrent
    and attention models cannot extract anything the static vector does not
    already contain. Worth asserting in training: it is the check that would
    have caught the tiling.
    """
    return float(np.mean(np.std(seqs, axis=1)))


def build_sequences_legacy(
    X: np.ndarray, seq_len: int = config.SEQ_LEN
) -> np.ndarray:
    """
    The original tiling behaviour, retained only so that results produced
    before the fix can be reproduced. Do not use for new training.
    """
    logger.warning(
        "build_sequences_legacy produces constant sequences; the LSTM and "
        "Transformer will have no temporal signal to learn from"
    )
    return np.tile(X[:, np.newaxis, :], (1, seq_len, 1))
