r"""Controlled variable-length pendulum dynamics for web-attached motion.

Implements the first-order controlled ODE system derived from the Lagrangian

    T = (1/2) m (l_dot^2 + l^2 theta_dot^2),   U = -m g l cos(theta),
    L = T - U,

under the coordinate convention x = x_a + l*sin(theta), y = y_a - l*cos(theta)
(theta measured from the downward vertical; documented in CLAUDE.md). The
Euler-Lagrange equation for theta gives

    theta_ddot = -(2*l_dot/l)*theta_dot - (g/l)*sin(theta),

and with the radial control u(t) = l_ddot, the first-order state
z = [theta, omega, l, nu] (omega = theta_dot, nu = l_dot) evolves as

    theta_dot = omega,
    omega_dot = -(2*nu/l)*omega - (g/l)*sin(theta),
    l_dot     = nu,
    nu_dot    = u.

The control is supplied as an open-loop function of time only, matching the
optimal-control formulation in which u(t) is parameterized directly.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from webswing.config import PhysicalParameters
from webswing.exceptions import NonPositiveWebLengthError

RadialControl = Callable[[float], float]
"""Open-loop radial control profile u(t) = l_ddot(t), in meters per second squared."""


def swing_state_derivative(
    t: float,
    z: np.ndarray,
    params: PhysicalParameters,
    control: RadialControl,
) -> np.ndarray:
    r"""Evaluate the state derivative of the controlled variable-length pendulum.

    Parameters
    ----------
    t : float
        Simulation time, in seconds. Passed through to `control`.
    z : np.ndarray, shape (4,)
        State vector [theta, omega, l, nu]: swing angle (rad), angular rate
        (rad/s), web length (m), radial rate (m/s).
    params : PhysicalParameters
        Mass and gravitational acceleration.
    control : RadialControl
        Commanded radial acceleration u(t) = l_ddot, in m/s^2.

    Returns
    -------
    np.ndarray, shape (4,)
        Derivative [theta_dot, omega_dot, l_dot, nu_dot].

    Raises
    ------
    NonPositiveWebLengthError
        If the web length component of `z` is non-positive or non-finite;
        the 1/l terms and the coordinate mapping are singular there.
    """
    theta, omega, ell, nu = z
    if not (math.isfinite(ell) and ell > 0.0):
        raise NonPositiveWebLengthError(ell)

    u = control(t)
    theta_dot = omega
    omega_dot = -(2.0 * nu / ell) * omega - (params.gravity / ell) * math.sin(theta)
    ell_dot = nu
    nu_dot = u
    return np.array([theta_dot, omega_dot, ell_dot, nu_dot], dtype=float)


def web_tension(
    theta: float,
    omega: float,
    ell: float,
    ell_ddot: float,
    params: PhysicalParameters,
) -> float:
    r"""Evaluate web tension from the radial force balance.

    T = m*(l*omega^2 + g*cos(theta) - l_ddot), under the sign convention in
    which positive l_ddot is web extension. This is the radial component of
    Newton's second law for the coordinate mapping used throughout this
    package; the sign on l_ddot must not be flipped without re-deriving the
    force balance under a different radial convention.

    Parameters
    ----------
    theta : float
        Swing angle from the downward vertical, in radians.
    omega : float
        Angular rate theta_dot, in rad/s.
    ell : float
        Web length, in meters. Must be finite and strictly positive.
    ell_ddot : float
        Radial acceleration l_ddot at the instant of evaluation, in m/s^2.
    params : PhysicalParameters
        Mass and gravitational acceleration.

    Returns
    -------
    float
        Web tension, in newtons. This function evaluates the force balance
        only; classifying T against [0, T_max] as a feasibility constraint
        is the responsibility of the event/constraint layer.

    Raises
    ------
    NonPositiveWebLengthError
        If `ell` is non-positive or non-finite.
    """
    if not (math.isfinite(ell) and ell > 0.0):
        raise NonPositiveWebLengthError(ell)
    return params.mass * (ell * omega**2 + params.gravity * math.cos(theta) - ell_ddot)
