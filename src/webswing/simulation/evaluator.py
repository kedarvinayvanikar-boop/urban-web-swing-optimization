r"""Force-derived quantities (tension, load factor) evaluated over a planned route.

`simulation.trajectory.Trajectory` deliberately retains only the reduced
kinematic view of a planned route (Cartesian position/velocity, web length,
angular/radial rate) -- it drops `theta` and the control profile once
converted, since visualization needs positions, not forces. Web tension
requires both of those (`dynamics.swing.web_tension` needs theta and
l_ddot = u(t)), so it cannot be recovered from a `Trajectory` alone. This
module re-walks the same planned-route edges `simulation.trajectory` does,
re-simulating each already-solved decision vector (again, an inexpensive
replay, not a re-optimization) to recover what a `Trajectory` discards.

This duplicates `trajectory.py`'s per-edge walk (re-simulate, then handle
the destination-interception truncation) rather than consuming a
`Trajectory` object. That is a deliberate trade-off: CLAUDE.md separates
trajectory assembly and evaluation into distinct files, and the two
produce genuinely different outputs (kinematic replay vs. force-derived
margins) from the same underlying edges, so some shared "walk the edges"
structure is repeated across the two modules rather than merging them into
one file.

Only tension and load factor (and their margins against `SwingConstraints`)
are covered here. Web-length and radial-speed margins do not need this
module -- they are directly computable from `Trajectory.web_lengths` and
`Trajectory.radial_rates` against a `SwingConstraints` without any further
force derivation.

The trivial (zero-edge) case
--------------------------------
Mirroring `trajectory.py`'s fallback for an already-at-destination start:
before any transfer has been solved, no control has been commanded, so
l_ddot = 0 is used for that single point's tension -- a resting/idle
convention, not a measured or optimized value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from webswing.config import PhysicalParameters, SwingConstraints
from webswing.dynamics.swing import web_tension
from webswing.optimization.controls import equal_interval_control
from webswing.optimization.local_transfer import simulate_transfer
from webswing.planning.astar import GOAL_NODE, PlannedTransfer, SearchEdge


@dataclass(frozen=True)
class TrajectoryEvaluation:
    """Per-sample force-derived quantities, index-aligned with a `Trajectory`.

    Produced from the same `path` and in the same order `assemble_trajectory`
    walks it, so `TrajectoryEvaluation.tensions[i]` corresponds to
    `Trajectory.times[i]` for a `Trajectory` built from the same `path`.

    Parameters
    ----------
    tensions : np.ndarray, shape (N,)
        Web tension at each sample, in newtons. NaN during ballistic
        samples (no web tension while in free flight).
    load_factors : np.ndarray, shape (N,)
        T / (m * g) at each sample, dimensionless. NaN during ballistic
        samples.
    tension_margins : np.ndarray, shape (N,)
        `constraints.tension_max - tension`. NaN during ballistic samples.
    load_factor_margins : np.ndarray, shape (N,)
        `constraints.load_factor_max - load_factor`. NaN during ballistic
        samples.
    """

    tensions: np.ndarray
    load_factors: np.ndarray
    tension_margins: np.ndarray
    load_factor_margins: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.tensions)
        for name, value in (
            ("load_factors", self.load_factors),
            ("tension_margins", self.tension_margins),
            ("load_factor_margins", self.load_factor_margins),
        ):
            if len(value) != n:
                raise ValueError(f"{name} has length {len(value)}, expected {n} (matching tensions)")


def _single_point_evaluation(
    start_state: np.ndarray, params: PhysicalParameters, constraints: SwingConstraints
) -> TrajectoryEvaluation:
    theta, omega, ell, _nu = start_state
    tension = web_tension(theta, omega, ell, 0.0, params)
    load_factor = tension / (params.mass * params.gravity)
    return TrajectoryEvaluation(
        tensions=np.array([tension]),
        load_factors=np.array([load_factor]),
        tension_margins=np.array([constraints.tension_max - tension]),
        load_factor_margins=np.array([constraints.load_factor_max - load_factor]),
    )


def evaluate_trajectory(
    path: tuple[SearchEdge, ...],
    start_state: np.ndarray,
    params: PhysicalParameters,
    constraints: SwingConstraints,
) -> TrajectoryEvaluation:
    """Evaluate tension and load factor over a planned route's samples.

    Parameters
    ----------
    path : tuple[SearchEdge, ...]
        `SearchResult.path` from `planning.astar.plan_route`. Every edge's
        `label` must be a `PlannedTransfer`.
    start_state : np.ndarray, shape (4,)
        Continuous attached state [theta, omega, ell, nu] at the start
        (used only when `path` is empty; see module docstring).
    params : PhysicalParameters
        Mass and gravitational acceleration.
    constraints : SwingConstraints
        Feasibility thresholds to compute margins against.

    Returns
    -------
    TrajectoryEvaluation
        Index-aligned with a `Trajectory` built from the same `path` (see
        `simulation.trajectory.assemble_trajectory`).

    Raises
    ------
    TypeError
        If any edge's `label` is not a `PlannedTransfer`.
    """
    if not path:
        return _single_point_evaluation(start_state, params, constraints)

    tensions: list[float] = []
    load_factors: list[float] = []
    tension_margins: list[float] = []
    load_factor_margins: list[float] = []

    for edge in path:
        if not isinstance(edge.label, PlannedTransfer):
            raise TypeError(
                "evaluate_trajectory requires a path whose edge labels are "
                f"PlannedTransfer (as produced by planning.astar.plan_route); "
                f"got {type(edge.label)!r}"
            )
        problem = edge.label.problem
        result = edge.label.result
        control = equal_interval_control(
            result.solution[1:], result.release_time, problem.u_min, problem.u_max
        )
        sim = simulate_transfer(result.solution, problem)

        for t_local, state in zip(sim.swing_times, sim.swing_states):
            theta, omega, ell, _nu = state
            ell_ddot = control(float(t_local))
            tension = web_tension(theta, omega, ell, ell_ddot, params)
            load_factor = tension / (params.mass * params.gravity)
            tensions.append(tension)
            load_factors.append(load_factor)
            tension_margins.append(constraints.tension_max - tension)
            load_factor_margins.append(constraints.load_factor_max - load_factor)

        ballistic_times_local = sim.ballistic_times
        if ballistic_times_local is not None and edge.to_node == GOAL_NODE:
            cutoff_local_time = edge.cost - result.release_time
            ballistic_times_local = ballistic_times_local[ballistic_times_local <= cutoff_local_time + 1.0e-9]
        n_ballistic = 0 if ballistic_times_local is None else len(ballistic_times_local)
        tensions.extend([math.nan] * n_ballistic)
        load_factors.extend([math.nan] * n_ballistic)
        tension_margins.extend([math.nan] * n_ballistic)
        load_factor_margins.extend([math.nan] * n_ballistic)

    return TrajectoryEvaluation(
        tensions=np.array(tensions),
        load_factors=np.array(load_factors),
        tension_margins=np.array(tension_margins),
        load_factor_margins=np.array(load_factor_margins),
    )
