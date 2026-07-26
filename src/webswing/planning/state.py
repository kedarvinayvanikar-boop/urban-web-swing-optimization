r"""Discretized planning-graph state for global trajectory search.

CLAUDE.md is explicit that a planning node must not be modelled as only an
anchor identifier: transfer feasibility and cost depend on the incoming
attachment state (angle, angular rate, web length, radial rate), so the
anchor alone is not a Markov state for the search. `PlanningState` pairs an
anchor identifier with a discretized bin index along each of those four
continuous state dimensions, giving a node that is both a sufficient
statistic for evaluating future transfers and, being a frozen dataclass of
hashable fields, directly usable as a dict/set key for A*'s closed-set and
duplicate-state detection (`heuristic.py` and `astar.py`, not yet built,
consume this module).

Binning convention
--------------------
Each continuous axis is partitioned into half-open bins of a fixed width:
bin index i covers `[i * width, (i + 1) * width)`. This applies uniformly
to positive and negative values (e.g. width 0.1: theta in [-0.1, 0.0) is
bin -1, [0.0, 0.1) is bin 0), via floor division. Web length ell is always
strictly positive (enforced upstream by the dynamics layer), so its bin
index is always >= 0 in practice, but no such restriction is imposed here.

Angle wraparound is not handled: theta is treated as an unwrapped real
number for binning purposes. This matches how theta is used throughout the
dynamics layer (no modular reduction is applied there either) and is a
reasonable simplification for a swinging web, which does not wrap past
+-pi in normal operation; it is not a correctness guarantee for a
trajectory that somehow did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from webswing.exceptions import InvalidPhysicalParameterError
from webswing.dynamics.events import assert_state_finite


@dataclass(frozen=True)
class PlanningStateResolution:
    """Bin widths for discretizing a continuous attached state into planning nodes.

    Parameters
    ----------
    theta_bin_width : float
        Bin width along the swing-angle axis, in radians. Must be finite
        and strictly positive.
    omega_bin_width : float
        Bin width along the angular-rate axis, in rad/s. Must be finite and
        strictly positive.
    ell_bin_width : float
        Bin width along the web-length axis, in meters. Must be finite and
        strictly positive.
    nu_bin_width : float
        Bin width along the radial-rate axis, in m/s. Must be finite and
        strictly positive.

    Raises
    ------
    InvalidPhysicalParameterError
        If any bin width is not finite and strictly positive.
    """

    theta_bin_width: float
    omega_bin_width: float
    ell_bin_width: float
    nu_bin_width: float

    def __post_init__(self) -> None:
        for name in ("theta_bin_width", "omega_bin_width", "ell_bin_width", "nu_bin_width"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise InvalidPhysicalParameterError(
                    f"{name} must be finite and strictly positive, got {value!r}"
                )


@dataclass(frozen=True)
class PlanningState:
    """A discretized planning-graph node: an anchor plus a binned attachment state.

    Parameters
    ----------
    anchor_id : str
        Identifier of the anchor this state is attached to.
    theta_bin, omega_bin, ell_bin, nu_bin : int
        Bin indices along each state axis (see module docstring, Binning
        convention).
    """

    anchor_id: str
    theta_bin: int
    omega_bin: int
    ell_bin: int
    nu_bin: int


def discretize_state(
    anchor_id: str, state: np.ndarray, resolution: PlanningStateResolution
) -> PlanningState:
    """Discretize a continuous attached state into a `PlanningState` node.

    Parameters
    ----------
    anchor_id : str
        Identifier of the anchor `state` is attached to.
    state : np.ndarray, shape (4,)
        Continuous attached state [theta, omega, ell, nu].
    resolution : PlanningStateResolution
        Bin widths for each axis.

    Returns
    -------
    PlanningState
        The discretized node.

    Raises
    ------
    ValueError
        If `state` is not a finite array of shape (4,).
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (4,):
        raise ValueError(f"state must have shape (4,), got {state.shape}")
    assert_state_finite(state, label="planning state")

    theta, omega, ell, nu = state
    return PlanningState(
        anchor_id=anchor_id,
        theta_bin=math.floor(theta / resolution.theta_bin_width),
        omega_bin=math.floor(omega / resolution.omega_bin_width),
        ell_bin=math.floor(ell / resolution.ell_bin_width),
        nu_bin=math.floor(nu / resolution.nu_bin_width),
    )


def representative_state(state: PlanningState, resolution: PlanningStateResolution) -> np.ndarray:
    """Return the bin-center continuous state for a `PlanningState` node.

    Used to seed a continuous initial state (e.g. for
    `optimization.local_transfer.LocalTransferProblem`) from a discrete
    planning-graph node, such as one popped off the A* open set.

    Parameters
    ----------
    state : PlanningState
        Discrete planning node.
    resolution : PlanningStateResolution
        Bin widths for each axis; must be the same resolution `state` was
        discretized under.

    Returns
    -------
    np.ndarray, shape (4,)
        Continuous state [theta, omega, ell, nu] at the center of the bin
        `state` occupies along each axis.
    """
    theta = (state.theta_bin + 0.5) * resolution.theta_bin_width
    omega = (state.omega_bin + 0.5) * resolution.omega_bin_width
    ell = (state.ell_bin + 0.5) * resolution.ell_bin_width
    nu = (state.nu_bin + 0.5) * resolution.nu_bin_width
    return np.array([theta, omega, ell, nu], dtype=float)
