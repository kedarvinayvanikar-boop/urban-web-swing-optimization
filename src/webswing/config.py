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
