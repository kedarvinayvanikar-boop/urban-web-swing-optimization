r"""Terminal event functions for the attached-swing and ballistic integrators.

Each factory in this module returns a `solve_ivp`-compatible event callable:
a function of the same `(t, y, *args)` signature as the corresponding state
derivative (`swing_state_derivative` or `ballistic_state_derivative`), whose
return value crosses zero at the event of interest, stamped with `.terminal`
(stop integration when found) and `.direction` (the sign of the crossing
`solve_ivp` should look for) attributes as scipy expects.

Swing events operate on z = [theta, omega, ell, nu] with args
(params, control), matching `swing_state_derivative`. Ballistic events
operate on q = [x, y, vx, vy] with args (params,), matching
`ballistic_state_derivative`.

Of the termination conditions enumerated in CLAUDE.md, this module
implements every one that requires no interface beyond `dynamics` and
`config`: ground contact, web tension exceeding its maximum, tension
becoming non-positive, load-factor violation, web length leaving its
permitted interval, radial speed exceeding its permitted magnitude, a
requested release time, maximum ballistic duration, anchor capture-region
entry, and exit from an axis-aligned bounding box standing in for the city
domain. Building collision, web/building line-of-sight, and destination-
region entry require polygon geometry (`geometry`, not yet implemented) and
are deferred; a caller assembling full termination coverage must add those
events separately once that module exists.

Load factor is defined as T / (m*g): since web tension is the only
non-gravitational force acting on the point mass, this is exactly the
proper acceleration the body experiences, in units of standard gravity
(the same construction as an aircraft load factor n = L/W). It is
numerically derived from the same tension value as the tension-max event
but is a physically distinct, independently configurable limit (human
tolerance vs. web strength), per CLAUDE.md.

NaN/infinite state detection is not expressible as a zero-crossing event
(comparisons against NaN are always False), so `assert_state_finite` is a
plain guard function intended for direct use in an integration driver
rather than as a `solve_ivp` event.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.dynamics.swing import RadialControl, web_tension
from webswing.exceptions import NonFiniteStateError

SwingEvent = Callable[[float, np.ndarray, PhysicalParameters, RadialControl], float]
BallisticEvent = Callable[[float, np.ndarray, PhysicalParameters], float]


def _mark(event: Callable, *, terminal: bool, direction: float) -> Callable:
    event.terminal = terminal
    event.direction = direction
    return event


# --- Swing (web-attached) events --------------------------------------------


def make_swing_ground_contact_event(x_anchor: float, y_anchor: float) -> SwingEvent:
    """Build an event firing when the swinging body reaches the ground, y = 0.

    Parameters
    ----------
    x_anchor, y_anchor : float
        Position of the anchor the body is currently attached to, in meters.

    Returns
    -------
    SwingEvent
        Returns y = y_anchor - ell*cos(theta); terminal, direction=-1
        (fires on a downward crossing through zero).
    """

    def event(t: float, z: np.ndarray, params: PhysicalParameters, control: RadialControl) -> float:
        theta, _omega, ell, _nu = z
        return y_anchor - ell * math.cos(theta)

    return _mark(event, terminal=True, direction=-1.0)


def make_swing_tension_max_event(constraints: SwingConstraints) -> SwingEvent:
    """Build an event firing when web tension exceeds `constraints.tension_max`.

    Returns
    -------
    SwingEvent
        Returns T(t) - tension_max; terminal, direction=+1 (fires when
        tension rises through the limit).
    """

    def event(t: float, z: np.ndarray, params: PhysicalParameters, control: RadialControl) -> float:
        theta, omega, ell, _nu = z
        tension = web_tension(theta, omega, ell, control(t), params)
        return tension - constraints.tension_max

    return _mark(event, terminal=True, direction=1.0)


def make_swing_tension_nonpositive_event() -> SwingEvent:
    """Build an event firing when web tension reaches zero (loss of tension).

    A web cannot provide compressive force; T <= 0 means loss of tension and
    termination of attached motion, per CLAUDE.md.

    Returns
    -------
    SwingEvent
        Returns T(t); terminal, direction=-1 (fires when tension falls
        through zero).
    """

    def event(t: float, z: np.ndarray, params: PhysicalParameters, control: RadialControl) -> float:
        theta, omega, ell, _nu = z
        return web_tension(theta, omega, ell, control(t), params)

    return _mark(event, terminal=True, direction=-1.0)


def make_swing_load_factor_max_event(constraints: SwingConstraints) -> SwingEvent:
    """Build an event firing when the load factor T/(m*g) exceeds its maximum.

    Returns
    -------
    SwingEvent
        Returns T(t)/(m*g) - load_factor_max; terminal, direction=+1.
    """

    def event(t: float, z: np.ndarray, params: PhysicalParameters, control: RadialControl) -> float:
        theta, omega, ell, _nu = z
        tension = web_tension(theta, omega, ell, control(t), params)
        load_factor = tension / (params.mass * params.gravity)
        return load_factor - constraints.load_factor_max

    return _mark(event, terminal=True, direction=1.0)


def make_swing_length_min_event(constraints: SwingConstraints) -> SwingEvent:
    """Build an event firing when web length falls to `constraints.ell_min`.

    Returns
    -------
    SwingEvent
        Returns ell(t) - ell_min; terminal, direction=-1.
    """

    def event(t: float, z: np.ndarray, params: PhysicalParameters, control: RadialControl) -> float:
        _theta, _omega, ell, _nu = z
        return ell - constraints.ell_min

    return _mark(event, terminal=True, direction=-1.0)


def make_swing_length_max_event(constraints: SwingConstraints) -> SwingEvent:
    """Build an event firing when web length rises to `constraints.ell_max`.

    Returns
    -------
    SwingEvent
        Returns ell(t) - ell_max; terminal, direction=+1.
    """

    def event(t: float, z: np.ndarray, params: PhysicalParameters, control: RadialControl) -> float:
        _theta, _omega, ell, _nu = z
        return ell - constraints.ell_max

    return _mark(event, terminal=True, direction=1.0)


def make_swing_radial_speed_max_event(constraints: SwingConstraints) -> SwingEvent:
    """Build an event firing when |l_dot| exceeds `constraints.nu_max`.

    Returns
    -------
    SwingEvent
        Returns |nu(t)| - nu_max; terminal, direction=+1.
    """

    def event(t: float, z: np.ndarray, params: PhysicalParameters, control: RadialControl) -> float:
        _theta, _omega, _ell, nu = z
        return abs(nu) - constraints.nu_max

    return _mark(event, terminal=True, direction=1.0)


def make_swing_release_time_event(release_time: float) -> SwingEvent:
    """Build an event firing at a caller-requested release time.

    Models the optimizer's choice of release time (CLAUDE.md, Local Swing
    Optimization) as an integration termination condition.

    Parameters
    ----------
    release_time : float
        Simulation time at which release is requested, in seconds.

    Returns
    -------
    SwingEvent
        Returns t - release_time; terminal, direction=+1.
    """

    def event(t: float, z: np.ndarray, params: PhysicalParameters, control: RadialControl) -> float:
        return t - release_time

    return _mark(event, terminal=True, direction=1.0)


# --- Ballistic events --------------------------------------------------------


def make_ballistic_ground_impact_event() -> BallisticEvent:
    """Build an event firing when the ballistic trajectory reaches y = 0.

    Returns
    -------
    BallisticEvent
        Returns y(t); terminal, direction=-1.
    """

    def event(t: float, q: np.ndarray, params: PhysicalParameters) -> float:
        _x, y, _vx, _vy = q
        return y

    return _mark(event, terminal=True, direction=-1.0)


def make_ballistic_max_duration_event(max_duration: float) -> BallisticEvent:
    """Build an event firing after `max_duration` seconds of ballistic flight.

    Parameters
    ----------
    max_duration : float
        Maximum permitted ballistic flight duration, in seconds.

    Returns
    -------
    BallisticEvent
        Returns t - max_duration; terminal, direction=+1.
    """

    def event(t: float, q: np.ndarray, params: PhysicalParameters) -> float:
        return t - max_duration

    return _mark(event, terminal=True, direction=1.0)


def make_ballistic_capture_region_event(
    x_anchor: float, y_anchor: float, capture_radius: float
) -> BallisticEvent:
    """Build an event firing when the body enters an anchor's capture region.

    Parameters
    ----------
    x_anchor, y_anchor : float
        Candidate anchor position, in meters.
    capture_radius : float
        Radius of the feasible capture region around the anchor, in meters.
        Must be finite and strictly positive.

    Returns
    -------
    BallisticEvent
        Returns |q_xy - anchor| - capture_radius; terminal, direction=-1
        (fires when the body moves inside the capture radius).
    """

    def event(t: float, q: np.ndarray, params: PhysicalParameters) -> float:
        x, y, _vx, _vy = q
        return math.hypot(x - x_anchor, y - y_anchor) - capture_radius

    return _mark(event, terminal=True, direction=-1.0)


def make_ballistic_x_min_event(domain: BallisticDomain) -> BallisticEvent:
    """Build an event firing when x falls to `domain.x_min` (city domain exit).

    Returns
    -------
    BallisticEvent
        Returns x(t) - x_min; terminal, direction=-1.
    """

    def event(t: float, q: np.ndarray, params: PhysicalParameters) -> float:
        x, _y, _vx, _vy = q
        return x - domain.x_min

    return _mark(event, terminal=True, direction=-1.0)


def make_ballistic_x_max_event(domain: BallisticDomain) -> BallisticEvent:
    """Build an event firing when x rises to `domain.x_max` (city domain exit).

    Returns
    -------
    BallisticEvent
        Returns x(t) - x_max; terminal, direction=+1.
    """

    def event(t: float, q: np.ndarray, params: PhysicalParameters) -> float:
        x, _y, _vx, _vy = q
        return x - domain.x_max

    return _mark(event, terminal=True, direction=1.0)


def make_ballistic_y_max_event(domain: BallisticDomain) -> BallisticEvent:
    """Build an event firing when y rises to `domain.y_max` (city domain exit).

    Returns
    -------
    BallisticEvent
        Returns y(t) - y_max; terminal, direction=+1.
    """

    def event(t: float, q: np.ndarray, params: PhysicalParameters) -> float:
        _x, y, _vx, _vy = q
        return y - domain.y_max

    return _mark(event, terminal=True, direction=1.0)


# --- Numerical validity (not a solve_ivp event) -----------------------------


def assert_state_finite(state: np.ndarray, label: str = "state") -> None:
    """Raise if any component of `state` is NaN or infinite.

    Not usable as a `solve_ivp` event (NaN comparisons are always False and
    cannot register a zero crossing); intended for direct use by an
    integration driver on each accepted step or on the final solution array.

    Parameters
    ----------
    state : np.ndarray
        State vector to check (swing z or ballistic q).
    label : str, optional
        Human-readable label included in the raised exception.

    Raises
    ------
    NonFiniteStateError
        If any component of `state` is not finite.
    """
    if not np.all(np.isfinite(state)):
        raise NonFiniteStateError(state, label)
