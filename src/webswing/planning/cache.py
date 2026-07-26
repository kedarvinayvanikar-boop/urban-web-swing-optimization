r"""Transfer-result caching and edge derivation for the global A* search.

A* must not re-run `optimization.local_transfer.solve_local_transfer` for
every occurrence of the same (discretized incoming state, candidate anchor)
pair -- that pair is exactly what `planning.state.PlanningState` was built
to make hashable, so `TransferCache` memoizes on it directly. This is the
"transfer-result caching" requirement from CLAUDE.md's Global Trajectory
Planning section.

This module also supplies the two conversions `astar.py` needs to turn a
cached `LocalTransferResult` into a graph edge:

- `edge_cost`: elapsed travel time (CLAUDE.md: "edge cost equal to elapsed
  travel time"), or infinite for an infeasible transfer.
- `resulting_planning_state`: applies the attachment-capture transition
  (`dynamics.attachment.attach_to_anchor`) to the transfer's ballistic
  final state and discretizes the resulting attached state
  (`planning.state.discretize_state`) into the neighbour node. Returns
  `None` if the transfer itself failed, or if the capture transition is
  rejected (attachment range exceeded, or degenerate geometry) -- a
  successful ballistic capture-region entry does not by itself guarantee a
  valid attachment.
"""

from __future__ import annotations

import math
from typing import Callable

from webswing.config import PhysicalParameters
from webswing.dynamics.attachment import attach_to_anchor
from webswing.exceptions import AttachmentRangeExceededError, NonPositiveWebLengthError
from webswing.optimization.local_transfer import LocalTransferResult
from webswing.planning.state import PlanningState, PlanningStateResolution, discretize_state


class TransferCache:
    """Memoizes `LocalTransferResult` by (from_state, to_anchor_id).

    Attributes
    ----------
    hits : int
        Number of `get_or_compute` calls served from the cache.
    misses : int
        Number of `get_or_compute` calls that invoked `compute`.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[PlanningState, str], LocalTransferResult] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self,
        from_state: PlanningState,
        to_anchor_id: str,
        compute: Callable[[], LocalTransferResult],
    ) -> LocalTransferResult:
        """Return the cached result for (from_state, to_anchor_id), computing it once if absent.

        Parameters
        ----------
        from_state : PlanningState
            Discretized incoming planning state.
        to_anchor_id : str
            Candidate target anchor identifier.
        compute : Callable[[], LocalTransferResult]
            Called only on a cache miss to produce the result.

        Returns
        -------
        LocalTransferResult
            The cached or newly computed result.
        """
        key = (from_state, to_anchor_id)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        result = compute()
        self._store[key] = result
        return result

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: tuple[PlanningState, str]) -> bool:
        return key in self._store


def edge_cost(result: LocalTransferResult) -> float:
    """Return the graph edge cost (elapsed travel time) for a transfer result.

    Parameters
    ----------
    result : LocalTransferResult
        Outcome of a local swing-to-anchor transfer optimization.

    Returns
    -------
    float
        `result.release_time + result.ballistic_time` if `result.success`
        is True, else `math.inf`.
    """
    if not result.success or result.ballistic_time is None:
        return math.inf
    return result.release_time + result.ballistic_time


def resulting_planning_state(
    result: LocalTransferResult,
    to_anchor_id: str,
    to_anchor_position: tuple[float, float],
    params: PhysicalParameters,
    resolution: PlanningStateResolution,
    max_attachment_range: float | None = None,
) -> PlanningState | None:
    """Return the discretized planning state resulting from a successful transfer's capture.

    Applies the attachment-capture transition to `result.final_state` at
    the target anchor, then discretizes the resulting attached state.

    Parameters
    ----------
    result : LocalTransferResult
        Outcome of a local swing-to-anchor transfer optimization.
    to_anchor_id : str
        Identifier of the anchor `result` captured at.
    to_anchor_position : tuple[float, float]
        Position of that anchor, in meters.
    params : PhysicalParameters
        Mass and gravitational acceleration.
    resolution : PlanningStateResolution
        Bin widths to discretize the resulting attached state under.
    max_attachment_range : float or None, optional
        Maximum permitted web length at attachment, in meters. See
        `dynamics.attachment.attach_to_anchor`.

    Returns
    -------
    PlanningState or None
        The discretized neighbour node, or None if `result.success` is
        False, or the attachment transition itself is rejected (exceeds
        `max_attachment_range`, or degenerate capture geometry).
    """
    if not result.success or result.final_state is None:
        return None
    x, y, vx, vy = result.final_state
    try:
        attachment = attach_to_anchor(
            x, y, vx, vy, to_anchor_position[0], to_anchor_position[1], params, max_attachment_range
        )
    except (NonPositiveWebLengthError, AttachmentRangeExceededError):
        return None
    return discretize_state(to_anchor_id, attachment.z, resolution)
