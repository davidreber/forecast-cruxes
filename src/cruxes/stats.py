"""The one summary statistic the published files report.

Kept in its own module so that both the set-inclusion aggregation and the
baseline aggregation can use it without importing each other.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mean_and_error"]


def mean_and_error(values) -> tuple:
    """Mean, sample standard deviation, standard error and count.

    Values that are ``None`` are dropped, which is how a question that has no
    record at a given contrast count stays out of the average. With no values
    at all the three statistics are ``None`` and the count is zero. With one
    value the deviation and the error are zero rather than undefined, matching
    the recorded files.
    """
    array = np.asarray([v for v in values if v is not None], dtype=float)
    n = len(array)
    if n == 0:
        return None, None, None, 0
    if n == 1:
        return float(array.mean()), 0.0, 0.0, 1
    std = float(array.std(ddof=1))
    return float(array.mean()), std, std / float(np.sqrt(n)), n
