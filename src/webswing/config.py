"""Typed physical parameter containers for the dynamics equations.

Physical constants are represented as immutable, validated dataclasses
rather than bare floats or dictionaries, so every consumer of the swing and
ballistic equations of motion shares a single validated source of mass and
gravitational acceleration, in SI units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from webswing.exceptions import InvalidPhysicalParameterError

_STANDARD_GRAVITY_M_PER_S2 = 9.80665


@dataclass(frozen=True)
class PhysicalParameters:
    """Physical constants entering the equations of motion.

    Parameters
    ----------
    mass : float
        Point mass of the swinging body, in kilograms. Must be finite and
        strictly positive.
    gravity : float, optional
        Magnitude of gravitational acceleration, in meters per second
        squared. Defaults to standard gravity. Must be finite and strictly
        positive; the sign convention g = (0, -g) is applied by the dynamics
        equations, not encoded in this value.

    Raises
    ------
    InvalidPhysicalParameterError
        If `mass` or `gravity` is not finite and strictly positive.
    """

    mass: float
    gravity: float = _STANDARD_GRAVITY_M_PER_S2

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mass) and self.mass > 0.0):
            raise InvalidPhysicalParameterError(
                f"mass must be finite and strictly positive, got {self.mass!r}"
            )
        if not (math.isfinite(self.gravity) and self.gravity > 0.0):
            raise InvalidPhysicalParameterError(
                f"gravity must be finite and strictly positive, got {self.gravity!r}"
            )


@dataclass(frozen=True)
class SwingConstraints:
    """Feasibility thresholds for web-attached swinging motion.

    Web-strength and human-tolerance limits are kept as separate fields per
    CLAUDE.md: `tension_max` bounds the absolute force the web must
    withstand, while `load_factor_max` bounds T/(m*g), the proper
    acceleration (in units of g) the body experiences from that same
    tension. They are numerically related but are independent configurable
    limits.

    Parameters
    ----------
    tension_max : float
        Maximum web tension the web can sustain, in newtons. Must be finite
        and strictly positive.
    load_factor_max : float
        Maximum permitted load factor T/(m*g), dimensionless. Must be finite
        and strictly positive.
    ell_min : float
        Minimum permitted web length, in meters. Must be finite and strictly
        positive.
    ell_max : float
        Maximum permitted web length, in meters. Must be finite and strictly
        greater than `ell_min`.
    nu_max : float
        Maximum permitted magnitude of radial speed |l_dot|, in m/s. Must be
        finite and strictly positive.

    Raises
    ------
    InvalidPhysicalParameterError
        If any field is outside its valid domain, or if `ell_max <= ell_min`.
    """

    tension_max: float
    load_factor_max: float
    ell_min: float
    ell_max: float
    nu_max: float

    def __post_init__(self) -> None:
        for name in ("tension_max", "load_factor_max", "ell_min", "ell_max", "nu_max"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise InvalidPhysicalParameterError(
                    f"{name} must be finite and strictly positive, got {value!r}"
                )
        if not (self.ell_max > self.ell_min):
            raise InvalidPhysicalParameterError(
                f"ell_max ({self.ell_max!r}) must be strictly greater than "
                f"ell_min ({self.ell_min!r})"
            )


@dataclass(frozen=True)
class BallisticDomain:
    """Bounding box and duration limit for ballistic (post-release) motion.

    This is a minimal axis-aligned bounding-box placeholder for "the bounded
    city domain" pending the full city geometry module (`geometry/`); it is
    not a substitute for building-collision or destination-region checks.

    Parameters
    ----------
    x_min, x_max : float
        Horizontal domain bounds, in meters. Must be finite with
        `x_max > x_min`.
    y_max : float
        Upper vertical domain bound, in meters. Must be finite and strictly
        positive; the lower bound is the ground, y = 0, handled by the
        ground-impact event rather than this field.
    max_duration : float
        Maximum permitted ballistic flight duration, in seconds. Must be
        finite and strictly positive.

    Raises
    ------
    InvalidPhysicalParameterError
        If any field is outside its valid domain, or if `x_max <= x_min`.
    """

    x_min: float
    x_max: float
    y_max: float
    max_duration: float

    def __post_init__(self) -> None:
        for name in ("x_min", "x_max", "y_max", "max_duration"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise InvalidPhysicalParameterError(f"{name} must be finite, got {value!r}")
        if not (self.y_max > 0.0):
            raise InvalidPhysicalParameterError(
                f"y_max must be strictly positive, got {self.y_max!r}"
            )
        if not (self.max_duration > 0.0):
            raise InvalidPhysicalParameterError(
                f"max_duration must be strictly positive, got {self.max_duration!r}"
            )
        if not (self.x_max > self.x_min):
            raise InvalidPhysicalParameterError(
                f"x_max ({self.x_max!r}) must be strictly greater than x_min ({self.x_min!r})"
            )
