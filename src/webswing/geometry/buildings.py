r"""Deterministic urban geometry: building and city environment definitions.

Under the coordinate convention documented in CLAUDE.md (x horizontal,
y vertical, ground at y = 0), each building is a simple polygon resting on
the ground: its lowest vertex lies at y = 0, its highest vertex defines the
roof elevation, and its horizontal/vertical extents define width and
height. This module validates that those descriptive scalar fields are
consistent with the polygon actually supplied, rather than trusting two
independent (and possibly drifting) representations of the same shape.

This module implements only the data model: `Building`, `DestinationRegion`,
and `City` (a validated collection of buildings plus a destination region).
It does not implement point-in-polygon, segment-polygon intersection,
continuous-trajectory collision, or anchor line-of-sight checks; those
algorithms belong to `geometry.collision` and `geometry.anchors`
respectively, and operate on the data structures defined here. Candidate
anchor points are stored on `Building` but their derivation from roof
geometry is also `geometry.anchors`'s responsibility, not this module's;
`Building.candidate_anchors` here is populated from whatever the caller
supplies (fixed roof points or the output of that derivation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from webswing.exceptions import InvalidGeometryError

_GROUND_ELEVATION_M = 0.0
_GEOMETRY_TOLERANCE_M = 1.0e-6


def _polygon_bounding_box(vertices: np.ndarray) -> tuple[float, float, float, float]:
    """Return (min_x, max_x, min_y, max_y) for a polygon's vertex array."""
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def _polygon_signed_area(vertices: np.ndarray) -> float:
    """Return the shoelace-formula signed area of a polygon.

    Positive for counter-clockwise vertex ordering, negative for clockwise.
    Magnitude near zero indicates a degenerate (collinear or self-crossing
    to zero net area) polygon.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@dataclass(frozen=True)
class Building:
    """A single urban obstacle: a ground-resting polygon with descriptive metadata.

    Parameters
    ----------
    building_id : str
        Unique, non-empty identifier for this building.
    vertices : np.ndarray, shape (N, 2), N >= 3
        Polygon boundary vertices, in meters. Must be finite and describe a
        non-degenerate polygon whose lowest point rests on the ground
        (y = 0, within `_GEOMETRY_TOLERANCE_M`).
    width : float
        Horizontal extent of the building, in meters. Must be finite,
        strictly positive, and consistent with the polygon's bounding-box
        width (max_x - min_x).
    height : float
        Vertical extent of the building, in meters. Must be finite, strictly
        positive, and consistent with the polygon's bounding-box height
        (max_y - min_y).
    roof_elevation : float
        Elevation of the roof, in meters. Must be finite and consistent with
        the polygon's maximum y-coordinate.
    candidate_anchors : tuple[tuple[float, float], ...], optional
        Candidate web-attachment points associated with this building, in
        meters. Each must be a finite (x, y) pair. Defaults to empty; this
        module does not derive anchors from geometry (see
        `geometry.anchors`).
    safety_margin : float, optional
        Additional obstacle clearance to apply around this building, in
        meters. Must be finite and non-negative. Defaults to 0.0.

    Raises
    ------
    InvalidGeometryError
        If any field is structurally invalid or inconsistent with the
        supplied polygon, per the parameter descriptions above.
    """

    building_id: str
    vertices: np.ndarray
    width: float
    height: float
    roof_elevation: float
    candidate_anchors: tuple[tuple[float, float], ...] = ()
    safety_margin: float = 0.0

    def __post_init__(self) -> None:
        if not self.building_id:
            raise InvalidGeometryError("building_id must be a non-empty string")

        verts = np.asarray(self.vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            raise InvalidGeometryError(
                f"vertices must have shape (N, 2) with N >= 3, got shape {verts.shape}"
            )
        if not np.all(np.isfinite(verts)):
            raise InvalidGeometryError("vertices must be finite")

        min_x, max_x, min_y, max_y = _polygon_bounding_box(verts)
        signed_area = _polygon_signed_area(verts)
        if abs(signed_area) <= _GEOMETRY_TOLERANCE_M**2:
            raise InvalidGeometryError(
                f"polygon is degenerate (near-zero signed area {signed_area!r})"
            )

        if not (math.isfinite(self.width) and self.width > 0.0):
            raise InvalidGeometryError(
                f"width must be finite and strictly positive, got {self.width!r}"
            )
        if not (math.isfinite(self.height) and self.height > 0.0):
            raise InvalidGeometryError(
                f"height must be finite and strictly positive, got {self.height!r}"
            )
        if not math.isfinite(self.roof_elevation):
            raise InvalidGeometryError(
                f"roof_elevation must be finite, got {self.roof_elevation!r}"
            )
        if not (math.isfinite(self.safety_margin) and self.safety_margin >= 0.0):
            raise InvalidGeometryError(
                f"safety_margin must be finite and non-negative, got {self.safety_margin!r}"
            )

        if abs(min_y - _GROUND_ELEVATION_M) > _GEOMETRY_TOLERANCE_M:
            raise InvalidGeometryError(
                f"building must rest on the ground (y=0); got min_y={min_y!r}"
            )
        if abs(self.width - (max_x - min_x)) > _GEOMETRY_TOLERANCE_M:
            raise InvalidGeometryError(
                f"width {self.width!r} is inconsistent with the polygon bounding-box "
                f"extent {max_x - min_x!r}"
            )
        if abs(self.height - (max_y - min_y)) > _GEOMETRY_TOLERANCE_M:
            raise InvalidGeometryError(
                f"height {self.height!r} is inconsistent with the polygon bounding-box "
                f"extent {max_y - min_y!r}"
            )
        if abs(self.roof_elevation - max_y) > _GEOMETRY_TOLERANCE_M:
            raise InvalidGeometryError(
                f"roof_elevation {self.roof_elevation!r} is inconsistent with the "
                f"polygon's maximum y-coordinate {max_y!r}"
            )

        anchors: list[tuple[float, float]] = []
        for anchor in self.candidate_anchors:
            ax, ay = anchor
            if not (math.isfinite(ax) and math.isfinite(ay)):
                raise InvalidGeometryError(f"invalid candidate anchor {anchor!r}")
            anchors.append((float(ax), float(ay)))

        verts.setflags(write=False)
        object.__setattr__(self, "vertices", verts)
        object.__setattr__(self, "candidate_anchors", tuple(anchors))

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return (min_x, max_x, min_y, max_y) of the polygon, in meters."""
        return _polygon_bounding_box(self.vertices)

    @property
    def area(self) -> float:
        """Return the polygon's area, in square meters. Always non-negative."""
        return abs(_polygon_signed_area(self.vertices))


@dataclass(frozen=True)
class DestinationRegion:
    """An axis-aligned rectangular destination (goal) region.

    Parameters
    ----------
    x_min, x_max : float
        Horizontal bounds of the region, in meters. Must be finite with
        `x_max > x_min`.
    y_min, y_max : float
        Vertical bounds of the region, in meters. Must be finite with
        `y_max > y_min`; `y_min` must be at or above the ground (y = 0).

    Raises
    ------
    InvalidGeometryError
        If any bound is non-finite, mis-ordered, or `y_min` is below ground.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "x_max", "y_min", "y_max"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise InvalidGeometryError(f"{name} must be finite, got {value!r}")
        if not (self.x_max > self.x_min):
            raise InvalidGeometryError(
                f"x_max ({self.x_max!r}) must be strictly greater than x_min ({self.x_min!r})"
            )
        if not (self.y_max > self.y_min):
            raise InvalidGeometryError(
                f"y_max ({self.y_max!r}) must be strictly greater than y_min ({self.y_min!r})"
            )
        if self.y_min < _GROUND_ELEVATION_M - _GEOMETRY_TOLERANCE_M:
            raise InvalidGeometryError(
                f"y_min must be at or above the ground (y=0), got {self.y_min!r}"
            )


@dataclass(frozen=True)
class City:
    """A validated collection of buildings plus a destination region.

    Parameters
    ----------
    buildings : tuple[Building, ...]
        The buildings in the environment. May be empty. Every
        `building_id` must be unique.
    destination : DestinationRegion
        The goal region for the trajectory-planning problem.

    Raises
    ------
    InvalidGeometryError
        If any two buildings share a `building_id`.
    """

    buildings: tuple[Building, ...]
    destination: DestinationRegion

    def __post_init__(self) -> None:
        buildings = tuple(self.buildings)
        ids = [building.building_id for building in buildings]
        if len(ids) != len(set(ids)):
            duplicates = sorted({building_id for building_id in ids if ids.count(building_id) > 1})
            raise InvalidGeometryError(f"duplicate building_id values: {duplicates!r}")
        object.__setattr__(self, "buildings", buildings)

    def get_building(self, building_id: str) -> Building:
        """Return the building with the given identifier.

        Raises
        ------
        KeyError
            If no building with `building_id` exists in this city.
        """
        for building in self.buildings:
            if building.building_id == building_id:
                return building
        raise KeyError(f"no building with building_id={building_id!r}")

    def all_candidate_anchors(self) -> tuple[tuple[str, tuple[float, float]], ...]:
        """Return every (building_id, anchor) pair across all buildings, in building order."""
        return tuple(
            (building.building_id, anchor)
            for building in self.buildings
            for anchor in building.candidate_anchors
        )
