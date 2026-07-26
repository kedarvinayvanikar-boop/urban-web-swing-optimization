r"""Feasibility margin functions for a simulated swing trajectory.

Evaluates a sampled attached-swing trajectory (times and states, together
with the radial control that produced it) against the thresholds in
`config.SwingConstraints`: web tension (upper bound `tension_max` and the
`T > 0` lower bound), load factor `T/(m*g)`, web-length bounds, and radial-
speed bounds -- the constraints listed in CLAUDE.md's Local Swing
Optimization section:

    ell_min <= ell(t) <= ell_max,   |l_dot(t)| <= nu_max,   0 < T(t) <= T_max.

Every margin below follows the convention margin(t) >= 0 means feasible at
that sample, matching the sign convention `scipy.optimize.minimize`'s SLSQP
solver expects for inequality constraints (`fun(x) >= 0`), so
`optimization.local_transfer` can pass these margins directly into a
constraint callable. `SwingConstraintMargins.max_violation` is exactly the
"maximum constraint violation" value CLAUDE.md requires every optimization
result to record.

The `0 < T(t)` lower bound is represented here as the margin `T(t) >= 0`
(non-strict): a continuous solver needs a well-defined value at the
boundary, and `dynamics.events.make_swing_tension_nonpositive_event`
already terminates integration before `T` reaches zero during simulation,
so this margin is a feasibility measure for the optimizer, not a substitute
for that termination event.

This module deliberately does not evaluate collision/clearance margins
against city geometry or capture-feasibility constraints; those require the
specific target anchor and capture-time decisions that belong to
`optimization.local_transfer`, not to a standalone, target-agnostic
constraint evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from webswing.config import PhysicalParameters, SwingConstraints
from webswing.dynamics.swing import RadialControl, web_tension


@dataclass(frozen=True)
class SwingConstraintMargins:
    """Per-sample feasibility margins for a simulated swing trajectory.

    Every array has shape (N,), one entry per sampled time in `times`. A
    margin >= 0 at a given sample means that constraint is satisfied there.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Sample times, in seconds.
    tension_upper : np.ndarray, shape (N,)
        `tension_max - T(t)`.
    tension_lower : np.ndarray, shape (N,)
        `T(t)` (the `0 <= T(t)` margin; see module docstring).
    length_lower : np.ndarray, shape (N,)
        `ell(t) - ell_min`.
    length_upper : np.ndarray, shape (N,)
        `ell_max - ell(t)`.
    radial_speed : np.ndarray, shape (N,)
        `nu_max - |l_dot(t)|`.
    load_factor : np.ndarray, shape (N,)
        `load_factor_max - T(t) / (m * g)`.
    """

    times: np.ndarray
    tension_upper: np.ndarray
    tension_lower: np.ndarray
    length_lower: np.ndarray
    length_upper: np.ndarray
    radial_speed: np.ndarray
    load_factor: np.ndarray

    @property
    def min_margin(self) -> float:
        """Return the worst (most negative, or smallest positive) margin over all samples and categories."""
        all_margins = np.concatenate(
            [
                self.tension_upper,
                self.tension_lower,
                self.length_lower,
                self.length_upper,
                self.radial_speed,
                self.load_factor,
            ]
        )
        return float(np.min(all_margins))

    @property
    def max_violation(self) -> float:
        """Return the maximum constraint violation, in the units of the violating margin.

        Zero if every margin is non-negative (fully feasible); otherwise
        the magnitude of the most negative margin.
        """
        return float(max(0.0, -self.min_margin))

    @property
    def is_feasible(self) -> bool:
        """Return whether every margin at every sample is non-negative."""
        return self.min_margin >= 0.0


def swing_constraint_margins(
    times: np.ndarray,
    states: np.ndarray,
    params: PhysicalParameters,
    control: RadialControl,
    constraints: SwingConstraints,
) -> SwingConstraintMargins:
    """Evaluate all `SwingConstraints` margins over a sampled swing trajectory.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Sample times, in seconds, corresponding to `states` row-for-row.
    states : np.ndarray, shape (N, 4)
        Sampled attached states [theta, omega, ell, nu] at each time in
        `times`.
    params : PhysicalParameters
        Mass and gravitational acceleration.
    control : RadialControl
        The radial control u(t) = l_ddot that produced `states`; evaluated
        at each sample time to compute tension and load factor.
    constraints : SwingConstraints
        Feasibility thresholds to evaluate against.

    Returns
    -------
    SwingConstraintMargins
        Per-sample margins for every constraint category.

    Raises
    ------
    ValueError
        If `times` is not 1-D, `states` is not shape (N, 4), or their
        leading dimensions disagree.
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)

    if times.ndim != 1:
        raise ValueError(f"times must be 1-D, got shape {times.shape}")
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"states must have shape (N, 4), got shape {states.shape}")
    if states.shape[0] != times.shape[0]:
        raise ValueError(
            f"times and states must have the same leading dimension, "
            f"got {times.shape[0]} and {states.shape[0]}"
        )

    thetas = states[:, 0]
    omegas = states[:, 1]
    ells = states[:, 2]
    nus = states[:, 3]

    ell_ddots = np.array([control(t) for t in times])
    tensions = np.array(
        [
            web_tension(thetas[i], omegas[i], ells[i], ell_ddots[i], params)
            for i in range(times.shape[0])
        ]
    )

    return SwingConstraintMargins(
        times=times,
        tension_upper=constraints.tension_max - tensions,
        tension_lower=tensions,
        length_lower=ells - constraints.ell_min,
        length_upper=constraints.ell_max - ells,
        radial_speed=constraints.nu_max - np.abs(nus),
        load_factor=constraints.load_factor_max - tensions / (params.mass * params.gravity),
    )
