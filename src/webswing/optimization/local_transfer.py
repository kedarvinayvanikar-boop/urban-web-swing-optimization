r"""Constrained local optimization of a single swing-to-anchor transfer.

For a fixed current attached state and a fixed candidate target anchor,
solves the constrained nonlinear optimization problem from CLAUDE.md's
Local Swing Optimization section: choose a release time and the parameters
of a bounded radial control profile u(t) to minimize transfer time subject
to the swing feasibility constraints (`optimization.constraints`) and
reaching the target anchor's capture region during the subsequent ballistic
flight.

Decision vector and modelling choices
----------------------------------------
The decision vector is `x = [t_release, u_1, ..., u_N]`: the release time,
followed by N equal-interval radial-control segment values (see
`optimization.controls.equal_interval_control`). Two choices CLAUDE.md
leaves open are resolved as follows, and documented rather than silently
assumed:

- The target anchor is fixed per call, not searched over. Candidate-anchor
  search across a fixed set is `planning`'s responsibility (CLAUDE.md:
  "optional target anchor within a fixed candidate set"); this module
  solves one transfer at a time.
- Capture time is not a free decision variable. The ballistic phase
  terminates at the first entry to the target anchor's capture region (an
  event), and attachment itself is modelled as instantaneous
  (`dynamics.attachment`), so t_capture = 0 and J = t_release + t_ballistic.

Fixed-shape constraint evaluation across early swing termination
--------------------------------------------------------------------
If a swing-phase event (tension, length, radial speed, load factor, ground
contact) fires before the requested release time, `solve_ivp` returns a
solution defined only up to that earlier time. `scipy.optimize.minimize`
requires its constraint function to return a fixed-length vector for every
candidate x, so this module samples the swing dense output at a fixed
count of times spanning `[0, t_release]`, clipping any sample time beyond
the actually-reached final time down to that final time. Every sample past
the truncation point then repeats the state at the truncation boundary --
which is itself at or past a violated margin (that is why the event fired)
-- so the fixed-length constraint vector still signals infeasibility
rather than fabricating feasibility for times never actually reached.

Collision checking is post-hoc, not embedded per-iteration
----------------------------------------------------------------
If a `City` is supplied, the full swing + ballistic trajectory at the
optimizer's final candidate is checked against it once, after optimization
converges, via `geometry.collision.trajectory_collides_with_city`. A
collision overrides the result to infeasible with an explicit
`failure_reason`. Collision is not embedded as a live per-iteration
constraint during the search itself; a solution that numerically converges
but collides is caught and rejected at reporting time, not avoided during
the search. This is a scope simplification -- embedding a differentiable
collision/clearance margin into the live constraint set is deferred.

Surrogate objective for infeasible candidates
--------------------------------------------------
`scipy.optimize.minimize` requires a finite real objective value at every
candidate x, including ones that fail to reach the release time or fail to
reach the capture region. For those candidates this module falls back to
`(time actually reached) + PENALTY_WEIGHT * (constraint violation
magnitude)`, a documented surrogate distinct from the true objective
`t_release + t_ballistic`, which applies only once a candidate is fully
feasible and captured. Per CLAUDE.md, this surrogate must not be confused
with the global minimum-time objective; it exists only to give the solver
a continuous, roughly-monotonic gradient signal on infeasible candidates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, minimize

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.dynamics.ballistic import ballistic_state_derivative
from webswing.dynamics.events import (
    make_ballistic_capture_region_event,
    make_ballistic_ground_impact_event,
    make_ballistic_max_duration_event,
    make_ballistic_x_max_event,
    make_ballistic_x_min_event,
    make_ballistic_y_max_event,
    make_swing_ground_contact_event,
    make_swing_length_max_event,
    make_swing_length_min_event,
    make_swing_load_factor_max_event,
    make_swing_radial_speed_max_event,
    make_swing_tension_max_event,
    make_swing_tension_nonpositive_event,
)
from webswing.dynamics.release import release_to_ballistic_state
from webswing.dynamics.swing import swing_state_derivative
from webswing.geometry.buildings import City
from webswing.geometry.collision import trajectory_collides_with_city
from webswing.optimization.constraints import SwingConstraintMargins, swing_constraint_margins
from webswing.optimization.controls import equal_interval_control

_N_SWING_SAMPLES = 20
_N_BALLISTIC_SAMPLES = 50
_PENALTY_WEIGHT = 1.0e3
_TIME_TOLERANCE_S = 1.0e-9


@dataclass(frozen=True)
class LocalTransferProblem:
    """Fixed inputs defining one local swing-to-anchor transfer optimization.

    Parameters
    ----------
    initial_state : np.ndarray, shape (4,)
        Attached state [theta, omega, ell, nu] at the start of the transfer.
    current_anchor : tuple[float, float]
        Anchor the initial state is attached to, in meters.
    target_anchor : tuple[float, float]
        Candidate anchor to capture at the end of the ballistic phase, in
        meters.
    params : PhysicalParameters
        Mass and gravitational acceleration.
    constraints : SwingConstraints
        Swing feasibility thresholds.
    u_min, u_max : float
        Radial control bounds, in m/s^2. Must be finite with `u_max > u_min`.
    n_control_segments : int
        Number of equal-interval control segments (N >= 1) spanning
        `[0, t_release]`.
    t_release_min, t_release_max : float
        Bounds on the release-time decision variable, in seconds. Must be
        finite with `0 < t_release_min < t_release_max`.
    capture_radius : float
        Radius of the target anchor's feasible capture region, in meters.
        Must be finite and strictly positive.
    ballistic_domain : BallisticDomain
        Domain bounds and maximum duration for the ballistic phase.
    city : City or None, optional
        Environment to check the final candidate trajectory against for
        collision (post-hoc; see module docstring). None disables the check.

    Raises
    ------
    ValueError
        If any field is structurally invalid per the descriptions above.
    """

    initial_state: np.ndarray
    current_anchor: tuple[float, float]
    target_anchor: tuple[float, float]
    params: PhysicalParameters
    constraints: SwingConstraints
    u_min: float
    u_max: float
    n_control_segments: int
    t_release_min: float
    t_release_max: float
    capture_radius: float
    ballistic_domain: BallisticDomain
    city: City | None = None

    def __post_init__(self) -> None:
        state = np.asarray(self.initial_state, dtype=float)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError(f"initial_state must be a finite array of shape (4,), got {state!r}")
        object.__setattr__(self, "initial_state", state)

        for name in ("current_anchor", "target_anchor"):
            point = getattr(self, name)
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError(f"{name} must be a finite (x, y) pair, got {point!r}")

        if not (math.isfinite(self.u_min) and math.isfinite(self.u_max) and self.u_max > self.u_min):
            raise ValueError(
                f"u_min ({self.u_min!r}) and u_max ({self.u_max!r}) must be finite with u_max > u_min"
            )
        if not (isinstance(self.n_control_segments, int) and self.n_control_segments >= 1):
            raise ValueError(
                f"n_control_segments must be an integer >= 1, got {self.n_control_segments!r}"
            )
        if not (
            math.isfinite(self.t_release_min)
            and math.isfinite(self.t_release_max)
            and 0.0 < self.t_release_min < self.t_release_max
        ):
            raise ValueError(
                f"t_release_min ({self.t_release_min!r}) and t_release_max "
                f"({self.t_release_max!r}) must satisfy 0 < t_release_min < t_release_max"
            )
        if not (math.isfinite(self.capture_radius) and self.capture_radius > 0.0):
            raise ValueError(
                f"capture_radius must be finite and strictly positive, got {self.capture_radius!r}"
            )

    @property
    def decision_dimension(self) -> int:
        """Return the length of the decision vector x = [t_release, u_1, ..., u_N]."""
        return 1 + self.n_control_segments


@dataclass(frozen=True)
class TransferSimulation:
    """Full outcome of simulating one candidate decision vector.

    Parameters
    ----------
    t_release : float
        Requested release time, in seconds (decoded from x[0]).
    swing_times, swing_states : np.ndarray
        Fixed-count sampled swing trajectory (see module docstring on
        fixed-shape constraint evaluation).
    reached_release : bool
        Whether the swing integration reached `t_release` without an
        earlier terminal event firing.
    swing_margins : SwingConstraintMargins
        Feasibility margins over the sampled swing trajectory.
    ballistic_times, ballistic_states : np.ndarray or None
        Sampled ballistic trajectory, or None if the swing phase never
        reached release.
    captured : bool
        Whether the ballistic phase entered the target anchor's capture
        region.
    capture_margin : float or None
        `capture_radius - (closest approach to the target anchor)`; >= 0
        means the trajectory came within the capture radius at some
        sampled point. None if the swing phase never reached release.
    ballistic_time : float or None
        Elapsed ballistic flight time at capture (if captured) or at
        whatever event stopped the ballistic phase (if not).
    final_state : np.ndarray or None
        Ballistic state [x, y, vx, vy] at capture. None if not captured.
    n_integrations : int
        Number of `solve_ivp` calls performed to produce this outcome (1 if
        the swing phase never reached release, otherwise 2).
    """

    t_release: float
    swing_times: np.ndarray
    swing_states: np.ndarray
    reached_release: bool
    swing_margins: SwingConstraintMargins
    ballistic_times: np.ndarray | None
    ballistic_states: np.ndarray | None
    captured: bool
    capture_margin: float | None
    ballistic_time: float | None
    final_state: np.ndarray | None
    n_integrations: int


def _decode(x: np.ndarray) -> tuple[float, np.ndarray]:
    return float(x[0]), np.asarray(x[1:], dtype=float)


def _swing_events(problem: LocalTransferProblem) -> list:
    return [
        make_swing_ground_contact_event(*problem.current_anchor),
        make_swing_tension_max_event(problem.constraints),
        make_swing_tension_nonpositive_event(),
        make_swing_load_factor_max_event(problem.constraints),
        make_swing_length_min_event(problem.constraints),
        make_swing_length_max_event(problem.constraints),
        make_swing_radial_speed_max_event(problem.constraints),
    ]


def _ballistic_events(problem: LocalTransferProblem) -> list:
    return [
        make_ballistic_ground_impact_event(),
        make_ballistic_capture_region_event(*problem.target_anchor, problem.capture_radius),
        make_ballistic_max_duration_event(problem.ballistic_domain.max_duration),
        make_ballistic_x_min_event(problem.ballistic_domain),
        make_ballistic_x_max_event(problem.ballistic_domain),
        make_ballistic_y_max_event(problem.ballistic_domain),
    ]


def simulate_transfer(x: np.ndarray, problem: LocalTransferProblem) -> TransferSimulation:
    """Simulate one candidate decision vector against a `LocalTransferProblem`.

    Parameters
    ----------
    x : np.ndarray, shape (problem.decision_dimension,)
        Decision vector [t_release, u_1, ..., u_N].
    problem : LocalTransferProblem
        Fixed transfer problem inputs.

    Returns
    -------
    TransferSimulation
        Full sampled trajectory, feasibility margins, and capture outcome.
    """
    t_release, control_params = _decode(x)
    control = equal_interval_control(control_params, t_release, problem.u_min, problem.u_max)

    swing_sol = solve_ivp(
        swing_state_derivative,
        (0.0, t_release),
        problem.initial_state,
        args=(problem.params, control),
        events=_swing_events(problem),
        dense_output=True,
        rtol=1e-9,
        atol=1e-11,
        method="RK45",
    )
    t_final = float(swing_sol.t[-1])
    reached_release = math.isclose(t_final, t_release, rel_tol=0.0, abs_tol=_TIME_TOLERANCE_S)

    sample_times = np.linspace(0.0, t_release, _N_SWING_SAMPLES)
    clipped_times = np.minimum(sample_times, t_final)
    swing_states = swing_sol.sol(clipped_times).T
    swing_margins = swing_constraint_margins(
        clipped_times, swing_states, problem.params, control, problem.constraints
    )

    n_integrations = 1
    if not reached_release:
        return TransferSimulation(
            t_release=t_release,
            swing_times=clipped_times,
            swing_states=swing_states,
            reached_release=False,
            swing_margins=swing_margins,
            ballistic_times=None,
            ballistic_states=None,
            captured=False,
            capture_margin=None,
            ballistic_time=None,
            final_state=None,
            n_integrations=n_integrations,
        )

    z_release = swing_sol.y[:, -1]
    q0 = release_to_ballistic_state(z_release, *problem.current_anchor)

    # solve_ivp's adaptive stepping only checks a terminal event's sign at
    # each accepted step's endpoints; a smooth ballistic arc can take very
    # large steps, and a brief capture-region entry/exit can occur entirely
    # inside one such step, invisible to the endpoint-based sign check even
    # though the true continuous path passes through the capture disk. This
    # max_step bound -- the time to cross a capture-radius-scale distance at
    # the release speed, capped so it is never a large fraction of the
    # flight's own duration budget -- keeps steps fine enough that the dip
    # cannot hide inside a single one.
    release_speed = float(np.hypot(q0[2], q0[3]))
    max_step = min(
        problem.capture_radius / max(release_speed, 1.0e-6),
        problem.ballistic_domain.max_duration / 100.0,
    )

    ballistic_sol = solve_ivp(
        ballistic_state_derivative,
        (0.0, problem.ballistic_domain.max_duration),
        q0,
        args=(problem.params,),
        events=_ballistic_events(problem),
        dense_output=True,
        max_step=max_step,
        rtol=1e-9,
        atol=1e-11,
        method="RK45",
    )
    n_integrations += 1

    ballistic_t_final = float(ballistic_sol.t[-1])
    ballistic_sample_times = np.linspace(0.0, ballistic_t_final, _N_BALLISTIC_SAMPLES)
    ballistic_states = ballistic_sol.sol(ballistic_sample_times).T

    xa, ya = problem.target_anchor
    distances = np.hypot(ballistic_states[:, 0] - xa, ballistic_states[:, 1] - ya)
    capture_margin = float(problem.capture_radius - np.min(distances))

    capture_event_index = 1  # order fixed by _ballistic_events
    captured = len(ballistic_sol.t_events[capture_event_index]) > 0
    if captured:
        ballistic_time = float(ballistic_sol.t_events[capture_event_index][0])
        final_state = np.asarray(ballistic_sol.y_events[capture_event_index][0], dtype=float)
    else:
        ballistic_time = ballistic_t_final
        final_state = None

    return TransferSimulation(
        t_release=t_release,
        swing_times=clipped_times,
        swing_states=swing_states,
        reached_release=True,
        swing_margins=swing_margins,
        ballistic_times=ballistic_sample_times,
        ballistic_states=ballistic_states,
        captured=captured,
        capture_margin=capture_margin,
        ballistic_time=ballistic_time,
        final_state=final_state,
        n_integrations=n_integrations,
    )


def _objective(sim: TransferSimulation) -> float:
    if not sim.reached_release:
        return float(sim.swing_times[-1]) + _PENALTY_WEIGHT * max(0.0, -sim.swing_margins.min_margin)
    if sim.captured:
        return sim.t_release + sim.ballistic_time
    return (
        sim.t_release
        + sim.ballistic_time
        + _PENALTY_WEIGHT * max(0.0, -(sim.capture_margin if sim.capture_margin is not None else 0.0))
    )


def _constraint_vector(sim: TransferSimulation) -> np.ndarray:
    margins = sim.swing_margins
    parts = [
        margins.tension_upper,
        margins.tension_lower,
        margins.length_lower,
        margins.length_upper,
        margins.radial_speed,
        margins.load_factor,
    ]
    capture_margin = sim.capture_margin if sim.capture_margin is not None else -1.0
    return np.concatenate(parts + [np.array([capture_margin])])


class _MemoizedSimulator:
    """Caches the last simulated x to avoid re-integrating for objective/constraint pairs."""

    def __init__(self, problem: LocalTransferProblem) -> None:
        self._problem = problem
        self._last_key: tuple[float, ...] | None = None
        self._last_sim: TransferSimulation | None = None
        self.n_integrations = 0

    def __call__(self, x: np.ndarray) -> TransferSimulation:
        key = tuple(np.round(np.asarray(x, dtype=float), 12))
        if key == self._last_key and self._last_sim is not None:
            return self._last_sim
        sim = simulate_transfer(x, self._problem)
        self.n_integrations += sim.n_integrations
        self._last_key = key
        self._last_sim = sim
        return sim


@dataclass(frozen=True)
class LocalTransferResult:
    """Outcome of a local swing-to-anchor transfer optimization.

    Every field required by CLAUDE.md's Local Swing Optimization reporting
    requirement is present: solver name, initial guess, termination status,
    objective value, maximum constraint violation, evaluation/integration
    counts, final transfer state, and failure reason if infeasible.

    Parameters
    ----------
    solver : str
        Name of the `scipy.optimize.minimize` method used.
    initial_guess : np.ndarray
        Decision vector the solver was started from.
    solution : np.ndarray
        Final decision vector [t_release, u_1, ..., u_N] found by the
        solver (regardless of `success`; a caller inspecting a failed
        result can still see what was tried).
    success : bool
        True only if the solver converged AND the swing phase reached
        release AND the ballistic phase captured the target anchor AND (if
        a city was supplied) the resulting trajectory does not collide with
        it. A numerically "successful" but physically infeasible result is
        never reported as `success=True`.
    termination_status : int
        Raw `scipy.optimize.OptimizeResult.status` code.
    message : str
        Raw solver termination message.
    objective_value : float
        Final objective value (see module docstring on the surrogate
        objective for infeasible candidates).
    max_constraint_violation : float
        Maximum constraint violation at the final candidate; 0.0 if fully
        feasible.
    n_objective_evaluations : int
        `scipy.optimize.OptimizeResult.nfev`.
    n_dynamics_integrations : int
        Total number of `solve_ivp` calls performed across the whole
        optimization (cache-deduplicated).
    final_state : np.ndarray or None
        Ballistic state [x, y, vx, vy] at capture, or None if not captured.
    release_time : float
        Optimized release time, in seconds.
    ballistic_time : float or None
        Elapsed ballistic flight time at capture, or None if not captured.
    failure_reason : str or None
        Human-readable reason `success` is False; None if `success` is True.
    """

    solver: str
    initial_guess: np.ndarray
    solution: np.ndarray
    success: bool
    termination_status: int
    message: str
    objective_value: float
    max_constraint_violation: float
    n_objective_evaluations: int
    n_dynamics_integrations: int
    final_state: np.ndarray | None
    release_time: float
    ballistic_time: float | None
    failure_reason: str | None


def solve_local_transfer(
    problem: LocalTransferProblem,
    initial_guess: np.ndarray | None = None,
    solver: str = "SLSQP",
    options: dict | None = None,
) -> LocalTransferResult:
    """Solve the constrained local swing-to-anchor transfer optimization.

    Parameters
    ----------
    problem : LocalTransferProblem
        Fixed transfer problem inputs.
    initial_guess : np.ndarray or None, optional
        Initial decision vector. Defaults to the midpoint release time with
        zero control on every segment (a passive swing), a deterministic,
        reproducible choice requiring no random seed.
    solver : str, optional
        `scipy.optimize.minimize` method name. Defaults to "SLSQP", per
        CLAUDE.md's suggested initial options.
    options : dict or None, optional
        Extra options forwarded to `scipy.optimize.minimize`.

    Returns
    -------
    LocalTransferResult
        See `LocalTransferResult` for field descriptions.
    """
    if initial_guess is None:
        t0 = 0.5 * (problem.t_release_min + problem.t_release_max)
        initial_guess = np.concatenate([[t0], np.zeros(problem.n_control_segments)])
    initial_guess = np.asarray(initial_guess, dtype=float)

    lower = np.concatenate([[problem.t_release_min], np.full(problem.n_control_segments, problem.u_min)])
    upper = np.concatenate([[problem.t_release_max], np.full(problem.n_control_segments, problem.u_max)])
    bounds = Bounds(lb=lower, ub=upper)

    simulator = _MemoizedSimulator(problem)

    def objective(x: np.ndarray) -> float:
        return _objective(simulator(x))

    def constraint_fn(x: np.ndarray) -> np.ndarray:
        return _constraint_vector(simulator(x))

    result = minimize(
        objective,
        initial_guess,
        method=solver,
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraint_fn}],
        options=options or {"maxiter": 200, "ftol": 1e-8},
    )

    final_sim = simulator(result.x)
    max_violation = float(max(0.0, -np.min(_constraint_vector(final_sim))))

    collides = False
    if problem.city is not None and final_sim.reached_release:
        swing_points = [
            (
                problem.current_anchor[0] + ell * math.sin(theta),
                problem.current_anchor[1] - ell * math.cos(theta),
            )
            for theta, _omega, ell, _nu in final_sim.swing_states
        ]
        ballistic_points = (
            [(float(px), float(py)) for px, py in final_sim.ballistic_states[:, :2]]
            if final_sim.ballistic_states is not None
            else []
        )
        full_path = swing_points + ballistic_points
        collides = trajectory_collides_with_city(full_path, problem.city)

    failure_reason: str | None = None
    if not result.success:
        failure_reason = str(result.message)
    elif not final_sim.reached_release:
        failure_reason = "swing phase violated a feasibility constraint before reaching release time"
    elif not final_sim.captured:
        failure_reason = "ballistic trajectory did not reach the target anchor's capture region"
    elif collides:
        failure_reason = "trajectory collides with city geometry"

    success = bool(result.success and final_sim.reached_release and final_sim.captured and not collides)

    return LocalTransferResult(
        solver=solver,
        initial_guess=initial_guess,
        solution=np.asarray(result.x, dtype=float),
        success=success,
        termination_status=int(result.status),
        message=str(result.message),
        objective_value=float(result.fun),
        max_constraint_violation=max_violation,
        n_objective_evaluations=int(result.nfev),
        n_dynamics_integrations=simulator.n_integrations,
        final_state=final_sim.final_state,
        release_time=float(result.x[0]),
        ballistic_time=final_sim.ballistic_time,
        failure_reason=failure_reason,
    )
