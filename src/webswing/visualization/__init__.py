"""Static plotting, animation, and HUD rendering of stored trajectory output.

Per CLAUDE.md, this package must not rerun the optimizer or numerical
integrator while rendering; every function here consumes already-assembled
`simulation.trajectory.Trajectory` / `simulation.evaluator.TrajectoryEvaluation`
/ `planning.astar.SearchResult` data.
"""
