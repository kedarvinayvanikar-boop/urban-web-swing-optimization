"""Minimal end-to-end example: build a small city, plan a route, and render it.

Runs the full pipeline (geometry -> planning -> simulation -> visualization)
on a small, fast scenario and writes a static overview figure and an
animated GIF to outputs/figures/ and outputs/animations/. Intended as a
runnable reference for the README and for anyone exploring the package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.planning.state import PlanningStateResolution
from webswing.simulation.runner import run_simulation
from webswing.visualization.animation import render_animation, save_animation
from webswing.visualization.static import render_static_overview

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def build_city() -> City:
    vertices = np.array([[20.0, 0.0], [24.0, 0.0], [24.0, 27.0], [20.0, 27.0]])
    building = Building(
        building_id="B1",
        vertices=vertices,
        width=4.0,
        height=27.0,
        roof_elevation=27.0,
        candidate_anchors=((20.0, 27.0), (24.0, 27.0)),
    )
    destination = DestinationRegion(x_min=18.0, x_max=26.0, y_min=24.0, y_max=30.0)
    return City(buildings=(building,), destination=destination)


def main() -> None:
    params = PhysicalParameters(mass=1.0, gravity=9.80665)
    constraints = SwingConstraints(
        tension_max=200.0, load_factor_max=20.0, ell_min=1.0, ell_max=30.0, nu_max=5.0
    )
    domain = BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=10.0)
    resolution = PlanningStateResolution(
        theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
    )

    city = build_city()
    start_anchor_id = "start"
    start_anchor_position = (0.0, 50.0)
    start_state = np.array([-0.3, 1.0, 10.0, 0.0])

    run = run_simulation(
        start_anchor_id=start_anchor_id,
        start_anchor_position=start_anchor_position,
        start_state=start_state,
        city=city,
        params=params,
        constraints=constraints,
        resolution=resolution,
        u_min=-0.5,
        u_max=0.5,
        n_control_segments=3,
        t_release_min=0.1,
        t_release_max=3.0,
        capture_radius=4.0,
        ballistic_domain=domain,
        max_expansions=50,
    )

    print(f"success: {run.success}")
    if not run.success:
        print(f"failure reason: {run.failure_reason}")
        return

    print(f"total travel time: {run.search_result.total_cost:.2f} s")
    print(f"nodes expanded: {run.search_result.nodes_expanded}")

    figures_dir = OUTPUT_ROOT / "figures"
    animations_dir = OUTPUT_ROOT / "animations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    animations_dir.mkdir(parents=True, exist_ok=True)

    overview_fig = render_static_overview(
        city, run.search_result.path, run.trajectory, start_anchor_position
    )
    overview_path = figures_dir / "static_overview.png"
    overview_fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    print(f"wrote {overview_path}")

    anim_fig, anim = render_animation(
        city,
        run.search_result.path,
        run.trajectory,
        run.evaluation,
        constraints,
        start_anchor_id,
        start_anchor_position,
    )
    gif_path = animations_dir / "demo.gif"
    if save_animation(anim, str(gif_path), fps=20):
        print(f"wrote {gif_path}")
    else:
        print("animation export skipped: no local GIF/MP4 encoder available")


if __name__ == "__main__":
    main()
