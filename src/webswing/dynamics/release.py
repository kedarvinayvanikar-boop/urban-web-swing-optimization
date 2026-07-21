r"""Release transition: attached pendulum state to Cartesian ballistic state.

At web release the attached state z = [theta, omega, l, nu] is converted to
the Cartesian ballistic state q = [x, y, vx, vy] used by the projectile
dynamics in `ballistic.py`. Under the coordinate convention documented in
CLAUDE.md (theta from the downward vertical, anchor at (x_a, y_a)):

    x = x_a + l*sin(theta),          y = y_a - l*cos(theta)

Differentiating with the product rule (l and theta both time-varying):

    x_dot = l_dot*sin(theta) + l*theta_dot*cos(theta)
          = nu*sin(theta) + l*omega*cos(theta)
    y_dot = -l_dot*cos(theta) + l*theta_dot*sin(theta)
          = -nu*cos(theta) + l*omega*sin(theta)

No artificial impulse is applied: the transition is a pure change of
coordinates, not a physical event. It carries no time argument, so
simulation time continuity is preserved by construction as long as the
caller does not reset its integration clock when switching from the
attached to the ballistic integrator.
"""

from __future__ import annotations

import math

import numpy as np

from webswing.exceptions import NonPositiveWebLengthError


def attached_position(theta: float, ell: float, x_anchor: float, y_anchor: float) -> tuple[float, float]:
    """Map swing angle and web length to Cartesian position.

    Parameters
    ----------
    theta : float
        Swing angle from the downward vertical, in radians.
    ell : float
        Web length, in meters. Must be finite and strictly positive.
    x_anchor, y_anchor : float
        Anchor position, in meters.

    Returns
    -------
    tuple[float, float]
        (x, y) position, in meters.

    Raises
    ------
    NonPositiveWebLengthError
        If `ell` is non-positive or non-finite.
    """
    if not (math.isfinite(ell) and ell > 0.0):
        raise NonPositiveWebLengthError(ell)
    x = x_anchor + ell * math.sin(theta)
    y = y_anchor - ell * math.cos(theta)
    return x, y


def attached_velocity(theta: float, omega: float, ell: float, nu: float) -> tuple[float, float]:
    """Map swing state rates to Cartesian velocity.

    Parameters
    ----------
    theta : float
        Swing angle from the downward vertical, in radians.
    omega : float
        Angular rate theta_dot, in rad/s.
    ell : float
        Web length, in meters. Must be finite and strictly positive.
    nu : float
        Radial rate l_dot, in m/s.

    Returns
    -------
    tuple[float, float]
        (vx, vy) velocity, in m/s.

    Raises
    ------
    NonPositiveWebLengthError
        If `ell` is non-positive or non-finite.
    """
    if not (math.isfinite(ell) and ell > 0.0):
        raise NonPositiveWebLengthError(ell)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    vx = nu * sin_t + ell * omega * cos_t
    vy = -nu * cos_t + ell * omega * sin_t
    return vx, vy


def release_to_ballistic_state(z: np.ndarray, x_anchor: float, y_anchor: float) -> np.ndarray:
    """Convert an attached pendulum state to a Cartesian ballistic state.

    Parameters
    ----------
    z : np.ndarray, shape (4,)
        Attached state [theta, omega, l, nu].
    x_anchor, y_anchor : float
        Position of the anchor the state is currently attached to, in
        meters.

    Returns
    -------
    np.ndarray, shape (4,)
        Ballistic state [x, y, vx, vy], suitable as the initial condition
        for `ballistic_state_derivative`. Position and velocity are
        continuous with the attached state by construction; no impulse is
        added.

    Raises
    ------
    NonPositiveWebLengthError
        If the web length component of `z` is non-positive or non-finite.
    """
    theta, omega, ell, nu = z
    x, y = attached_position(theta, ell, x_anchor, y_anchor)
    vx, vy = attached_velocity(theta, omega, ell, nu)
    return np.array([x, y, vx, vy], dtype=float)
