# Kinematic Optimization of Urban Brachistochrone Trajectories: Anchor Selection and Optimal Control

## Methodology

This document is the research-methodology reference for the `webswing` package: a
two-dimensional, physics-based simulation and trajectory optimizer for
Spider-Man-inspired web-swinging through a deterministic urban environment. It is
written to be directly adaptable into an Overleaf/LaTeX manuscript; every mathematical
expression below is valid LaTeX, and every derivation, convention, tolerance, and
algorithmic claim is sourced from the actual implementation in `src/webswing/`, not
reconstructed independently of it. Where a claim about the software's behavior is made,
the responsible module is named so it can be checked directly against the code.

Per this project's scientific-integrity requirements, this document distinguishes
throughout between statements that follow from classical mechanics (physical laws),
statements that are modelling choices this project made among several defensible
alternatives (assumptions), and statements that are true only of this specific
software's chosen discretization, numerical tolerances, and search parameters (not
general facts about the underlying continuous problem). No numerical result, benchmark
value, or optimality claim below is asserted beyond what the implementation actually
establishes.

---

### 1. Problem Definition

Given a start state (a position, and either an attachment state or a free-flight
velocity) and a destination region within a deterministic two-dimensional urban
environment, the problem is to find a minimum-time feasible trajectory from the start
to the destination, where motion alternates between two regimes:

1. **Web-attached motion** — a controlled variable-length pendulum swinging from a
   fixed anchor point, with the web's length actively controlled (extended or
   retracted) subject to actuation, tension, and human-tolerance limits.
2. **Ballistic motion** — unconstrained projectile motion under gravity alone,
   following release of the web.

A trajectory is *feasible* if it never violates web-tension, load-factor, web-length,
web-retraction/extension-speed, or obstacle-clearance constraints at any sampled point,
and *minimum-time* with respect to a specific candidate-anchor graph, control
parameterization, and numerical tolerance set — not, in general, the true infimum over
every physically conceivable control law, which this project does not claim to have
found (see §14, §15, and §19).

### 2. Coordinate and Sign Conventions

All quantities are expressed in a single, fixed two-dimensional Cartesian frame:

- $x$: horizontal position, increasing to the right.
- $y$: vertical position, increasing upward.
- Ground: $y = 0$.
- Gravity: $\mathbf{g} = (0, -g)$, $g > 0$ (implemented as `PhysicalParameters.gravity`,
  `src/webswing/config.py`, defaulting to standard gravity
  $g = 9.80665\ \mathrm{m/s^2}$).

For a web anchored at $\mathbf{a} = (x_a, y_a)$, the swing angle $\theta$ is measured
from the **downward vertical**, and web length $\ell > 0$ is the anchor-to-body
distance. The forward coordinate mapping, used identically throughout the dynamics,
transition, tension, planning, and visualization code, is

$$
x = x_a + \ell \sin\theta, \qquad y = y_a - \ell \cos\theta. \tag{2.1}
$$

This sign convention is a modelling choice, not a physical necessity — an equally valid
mechanics results from, e.g., measuring $\theta$ from the upward vertical or reversing
the direction of increasing $x$ — but every equation in this project assumes exactly
(2.1), and no module flips the sign of $\ell$, $\theta$, or the radial-acceleration
control $u = \ddot\ell$ relative to it (`src/webswing/dynamics/swing.py`,
`src/webswing/dynamics/release.py`, `src/webswing/dynamics/attachment.py`).

### 3. Modelling Assumptions

The following are assumptions this project makes, not laws of physics, listed
separately from the derivations that follow so they can be revisited independently:

- The swinging body is a **point mass** $m$ (`PhysicalParameters.mass`); no rotational
  inertia, body extent, or air resistance is modelled for either attached or ballistic
  motion.
- The web is treated as a **massless, inextensible-except-by-control** rigid link of
  instantaneous length $\ell(t)$: it can only pull along its own axis (tension
  $T \geq 0$; see §8) and cannot push, bend, or store elastic energy beyond the
  work done by the radial control itself.
- Web attachment (§11) is **instantaneous** and **non-conservative**: a new anchor's
  radial velocity component is removed in zero time, in an idealized inelastic
  capture, not a spring-damper or gradual-tensioning model.
- Ballistic motion is **drag-free**: no aerodynamic force acts on the body once
  released (`src/webswing/dynamics/ballistic.py`).
- The urban environment is **deterministic and fully known** in advance: buildings are
  fixed, ground-resting simple polygons (`src/webswing/geometry/buildings.py`); there
  is no sensing uncertainty, dynamic obstacle, or partial observability.
- Web-attachment anchors are a **fixed, finite candidate set** derived from building
  roof geometry (`src/webswing/geometry/anchors.py`), not an arbitrary continuous
  choice of attachment point on every building surface.
- The story's "Spider-Man-inspired" framing is a narrative motivation for the
  scenario, not a source of any physical assumption; every equation of motion below is
  ordinary Newtonian/Lagrangian mechanics under the assumptions listed here (see §25).

### 4. Derivation of the Variable-Length Pendulum Lagrangian

For a point mass $m$ swinging on a web of instantaneous length $\ell(t)$ at angle
$\theta(t)$ from the downward vertical (mapping (2.1)), differentiate to obtain the
Cartesian velocity components in terms of the generalized coordinates $(\ell, \theta)$
and their rates $(\dot\ell, \dot\theta)$:

$$
\dot x = \dot\ell \sin\theta + \ell\dot\theta\cos\theta, \qquad
\dot y = -\dot\ell \cos\theta + \ell\dot\theta\sin\theta. \tag{4.1}
$$

The kinetic energy $T = \tfrac{1}{2}m(\dot x^2 + \dot y^2)$ expands, using
$\sin^2 + \cos^2 = 1$ and the cross terms
$(\dot\ell\sin\theta)(\ell\dot\theta\cos\theta) + (-\dot\ell\cos\theta)(\ell\dot\theta\sin\theta) = 0$
(they cancel exactly), to

$$
T = \tfrac{1}{2} m\left(\dot\ell^2 + \ell^2\dot\theta^2\right). \tag{4.2}
$$

Taking gravitational potential energy zero at the anchor's height and using
$y = y_a - \ell\cos\theta$, the height *below* the anchor is $\ell\cos\theta$, so

$$
U = -mg\,\ell\cos\theta. \tag{4.3}
$$

The Lagrangian is

$$
L(\ell, \theta, \dot\ell, \dot\theta) = T - U
= \tfrac{1}{2}m\left(\dot\ell^2 + \ell^2\dot\theta^2\right) + mg\,\ell\cos\theta. \tag{4.4}
$$

This is the Lagrangian of a standard planar pendulum with a *time-varying, actively
controlled* radial coordinate, rather than a fixed length — the "variable-length"
generalization CLAUDE.md's model calls for.

### 5. Euler-Lagrange Derivation

The system has two generalized coordinates, $\theta$ and $\ell$, but only $\theta$ is
a *free* coordinate in the sense of an unforced Euler-Lagrange equation; $\ell$ is
directly actuated by the radial control $u(t) = \ddot\ell(t)$ (a kinematic, not
dynamic, degree of freedom — see §6). The Euler-Lagrange equation for $\theta$,

$$
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot\theta}\right) - \frac{\partial L}{\partial \theta} = 0,
$$

is evaluated term by term. From (4.4):

$$
\frac{\partial L}{\partial \dot\theta} = m\ell^2\dot\theta, \qquad
\frac{d}{dt}\left(m\ell^2\dot\theta\right) = m\left(2\ell\dot\ell\dot\theta + \ell^2\ddot\theta\right), \qquad
\frac{\partial L}{\partial \theta} = -mg\,\ell\sin\theta.
$$

Substituting,

$$
m\left(2\ell\dot\ell\dot\theta + \ell^2\ddot\theta\right) + mg\,\ell\sin\theta = 0,
$$

and dividing through by $m\ell^2$ (valid for $\ell > 0$, the physically required web
length; see §12 and the singularity this creates) gives

$$
\ddot\theta = -\frac{2\dot\ell}{\ell}\dot\theta - \frac{g}{\ell}\sin\theta. \tag{5.1}
$$

The first term is a Coriolis-like coupling between radial and angular motion — it
vanishes identically when $\dot\ell = 0$ (a fixed-length pendulum), recovering the
ordinary simple-pendulum equation $\ddot\theta = -(g/\ell)\sin\theta$ exactly, which is
the analytic oracle used to validate this equation for $\dot\ell \equiv 0$ (§18).

### 6. Controlled First-Order ODE System

Introducing $\omega = \dot\theta$ and $\nu = \dot\ell$ and treating $u(t) = \ddot\ell(t)$
as an externally supplied open-loop control (not a free Euler-Lagrange coordinate — the
web's extension/retraction is actuated, not force-free), the state
$\mathbf{z} = (\theta, \omega, \ell, \nu)$ evolves under the first-order system

$$
\dot\theta = \omega, \qquad
\dot\omega = -\frac{2\nu}{\ell}\omega - \frac{g}{\ell}\sin\theta, \qquad
\dot\ell = \nu, \qquad
\dot\nu = u(t). \tag{6.1}
$$

This is implemented exactly by `swing_state_derivative`
(`src/webswing/dynamics/swing.py`), integrated with `scipy.integrate.solve_ivp` (§17).
The function raises a domain-specific `NonPositiveWebLengthError`
(`src/webswing/exceptions.py`) rather than silently propagating a division-by-zero or
NaN if $\ell \leq 0$ is ever reached, since (6.1) is singular there. The control
interface (`src/webswing/optimization/controls.py`) supports three forms required by
CLAUDE.md: a constant radial acceleration (`ConstantControl`), a piecewise-constant
profile over caller-specified breakpoints (`PiecewiseConstantControl`), and an
equal-interval piecewise-constant profile parameterized directly by an optimizer's flat
decision vector (`equal_interval_control`) — the bridge used by the local optimal
control formulation in §13.

No energy is added to the system except through the work done by the control $u(t)$ via
the $\dot\ell$/$\ell$ coupling terms in (6.1); there is no artificial gain term anywhere
in the right-hand side.

### 7. Cartesian Coordinate and Velocity Mapping

Equation (2.1) is the position mapping; its time derivative, (4.1), is the velocity
mapping, restated here in terms of the state variables $(\theta, \omega, \ell, \nu)$:

$$
\dot x = \nu\sin\theta + \ell\omega\cos\theta, \qquad
\dot y = -\nu\cos\theta + \ell\omega\sin\theta. \tag{7.1}
$$

Both mappings are implemented in `src/webswing/dynamics/release.py`
(`attached_position`, `attached_velocity`) and are the *only* place Cartesian
position/velocity is derived from attached state — release transitions (§10),
trajectory assembly, and the animation/HUD layer all call through these two functions
rather than re-deriving the mapping, so a convention change would only need to happen
in one place.

### 8. Radial Force Balance and Web-Tension Derivation

Web tension is derived independently of the Lagrangian, from Newton's second law
resolved along the radial direction. Define the outward radial unit vector
$\mathbf{e}_r = (\sin\theta, -\cos\theta)$ (anchor toward body, per (2.1)) and the
tangential unit vector $\mathbf{e}_\theta = (\cos\theta, \sin\theta)$, orthogonal to
$\mathbf{e}_r$. Differentiating with respect to $\theta$,
$d\mathbf{e}_r/d\theta = \mathbf{e}_\theta$ and $d\mathbf{e}_\theta/d\theta = -\mathbf{e}_r$.
Writing the position as $\mathbf{r} = \ell\,\mathbf{e}_r$ and differentiating twice
(both $\ell$ and $\theta$ time-varying) reproduces the velocity mapping (7.1),
$\mathbf{v} = \dot\ell\,\mathbf{e}_r + \ell\omega\,\mathbf{e}_\theta$, and gives the
acceleration

$$
\mathbf{a} = \left(\ddot\ell - \ell\omega^2\right)\mathbf{e}_r + \left(2\dot\ell\,\omega + \ell\dot\omega\right)\mathbf{e}_\theta,
$$

so the radial acceleration component is $a_r = \ddot\ell - \ell\omega^2$: the
translational term $\ddot\ell$ (radial acceleration from the web's own length change)
combined with the centripetal term $-\ell\omega^2$ (toward the anchor, since the body
moves on a circle of instantaneous radius $\ell$). Gravity's radial component is
$\mathbf{g}\cdot\mathbf{e}_r = (0,-g)\cdot(\sin\theta,-\cos\theta) = g\cos\theta$ —
positive (outward) when $\theta$ is small, matching the intuition that a body hanging
nearly straight below the anchor has its weight pulling it further from the anchor,
resisted by tension. Newton's second law along $\mathbf{e}_r$, with tension $T \geq 0$
acting toward the anchor (i.e. $-T$ along $\mathbf{e}_r$, since a web can only pull),
is

$$
m\left(\ddot\ell - \ell\omega^2\right) = m g \cos\theta - T.
$$

Solving for $T$:

$$
T = m\left(\ell\omega^2 + g\cos\theta - \ddot\ell\right). \tag{8.1}
$$

As a sanity check, at $\theta = 0$, $\omega = 0$, $\ddot\ell = 0$ (hanging at rest
directly below the anchor), (8.1) gives $T = mg$, tension exactly supporting the
body's weight, as expected. This matches `web_tension`
(`src/webswing/dynamics/swing.py`) exactly, and was independently verified by symbolic
differentiation of (2.1) rather than taken on faith. Per CLAUDE.md
and the module's own docstring, the sign on $\ddot\ell$ in (8.1) is tied to the
specific radial convention (2.1) and must not be flipped independently of it. A
swinging state is infeasible if $T > T_{\max}$ or $T \leq 0$ (a web cannot push); both
are implemented as terminal integration events
(`make_swing_tension_max_event`, `make_swing_tension_nonpositive_event`,
`src/webswing/dynamics/events.py`), not merely post-hoc checks, so an integration never
silently continues past a physically invalid tension state.

Load factor, a physically distinct constraint from web strength, is defined as
$n = T/(mg)$ — the proper acceleration the body experiences, in units of standard
gravity, exactly analogous to an aircraft load factor $n = L/W$
(`src/webswing/dynamics/events.py` module docstring). `SwingConstraints`
(`src/webswing/config.py`) keeps `tension_max` (web-strength limit) and
`load_factor_max` (human-tolerance limit) as independent configurable fields, per
CLAUDE.md's explicit requirement that the two not be conflated even though both are
functions of the same $T(t)$.

### 9. Ballistic Equations

Once released, the body is in unconstrained free fall under (2's) gravity convention,
with state $\mathbf{q} = (x, y, v_x, v_y)$:

$$
\dot x = v_x, \qquad \dot y = v_y, \qquad \dot v_x = 0, \qquad \dot v_y = -g. \tag{9.1}
$$

Implemented as `ballistic_state_derivative`
(`src/webswing/dynamics/ballistic.py`) and integrated with `solve_ivp` even though a
closed-form solution exists, per CLAUDE.md's requirement that the closed form serve as
a validation oracle rather than the runtime code path (§18):

$$
x(t) = x_0 + v_{x,0}t, \qquad
y(t) = y_0 + v_{y,0}t - \tfrac{1}{2}gt^2, \qquad
v_x(t) = v_{x,0}, \qquad
v_y(t) = v_{y,0} - gt. \tag{9.2}
$$

### 10. Release Transition

At web release, the attached state $\mathbf{z} = (\theta, \omega, \ell, \nu)$ is
converted to the ballistic state $\mathbf{q} = (x, y, v_x, v_y)$ via the position
mapping (2.1) and velocity mapping (7.1) directly — **no impulse is applied**. Position,
velocity, and simulation time are all continuous across the transition by construction:
position and velocity because (2.1)/(7.1) are the same functions used throughout
attached motion, evaluated at the instant of release, and time because the release
transition (`release_to_ballistic_state`,
`src/webswing/dynamics/release.py`) carries no time argument at all — the caller's
integration clock is simply not reset. This is verified directly against coordinate
differentiation, per CLAUDE.md's requirement, in `tests/dynamics/test_release.py`.

### 11. Attachment Transition (Impulse Model)

A *new* web attachment is not automatically energy-preserving (unlike release, which is
a pure re-parameterization of the same physical state). Given a pre-attachment
Cartesian state $(x, y, v_x, v_y)$ and a candidate anchor $\mathbf{a} = (x_a, y_a)$,
define $\mathbf{r} = (x - x_a,\, y - y_a)$, $\ell = |\mathbf{r}|$, and the outward
radial unit vector $\mathbf{e}_r = \mathbf{r}/\ell$. Under convention (2.1),
$\mathbf{r} = \ell(\sin\theta, -\cos\theta)$, so the swing angle is recovered directly
from position without a separate division:

$$
\theta = \operatorname{atan2}(x - x_a,\ y_a - y). \tag{11.1}
$$

The velocity decomposes orthogonally as $\mathbf{v} = \mathbf{v}_r + \mathbf{v}_t$,
where $\mathbf{v}_r = (\mathbf{v}\cdot\mathbf{e}_r)\mathbf{e}_r$ is the radial component
and $\mathbf{v}_t = \mathbf{v} - \mathbf{v}_r$ the tangential remainder. The
**documented default capture model** (an explicit modelling choice, not derived from a
conservation law) is an idealized inelastic attachment: the radial component is removed
instantaneously and the tangential component retained,

$$
\mathbf{v}^+ = \mathbf{v}_t. \tag{11.2}
$$

Because $\mathbf{v}_r \perp \mathbf{v}_t$ by construction, the Pythagorean identity
$|\mathbf{v}|^2 = |\mathbf{v}_r|^2 + |\mathbf{v}_t|^2$ gives the kinetic-energy loss

$$
\Delta E = \tfrac{1}{2}m\left(|\mathbf{v}|^2 - |\mathbf{v}^+|^2\right) = \tfrac{1}{2}m|\mathbf{v}_r|^2 \geq 0 \tag{11.3}
$$

for *every* input — the loss is nonnegative by construction, not merely in the typical
case, and this project's implementation surfaces $\Delta E$ on every successful
attachment (`AttachmentResult.energy_loss`,
`src/webswing/dynamics/attachment.py`) rather than discarding it, per CLAUDE.md's
requirement to not conceal attachment energy loss.

The retained tangential velocity is converted to an angular rate using (7.1) with
$\nu = 0$ (the radial rate is exactly zero immediately after capture, since $v_r$ was
just removed): since $\mathbf{v}_t \perp \mathbf{e}_r$, it is necessarily parallel to
$\mathbf{e}_\theta = (\cos\theta, \sin\theta)$ (the only other direction in the plane),
giving

$$
\omega = \frac{v_{x,t}\cos\theta + v_{y,t}\sin\theta}{\ell}, \tag{11.4}
$$

exact regardless of how $\mathbf{v}_t$ arose. Attachment is rejected — no state
transition occurs — if the computed $\ell$ exceeds a configured maximum attachment
range (`AttachmentRangeExceededError`) or the pre-attachment position coincides with
the anchor within a numerical tolerance (a degenerate geometry in which $\theta$ is
undefined, `NonPositiveWebLengthError`). As documented in
`src/webswing/dynamics/attachment.py`, this module implements only the kinematic and
energetic core plus these two rejections; line-of-sight/building-intersection
rejection is applied separately by the planning layer (§12, §14), which composes this
transition with `geometry.anchors.anchor_has_line_of_sight` before ever accepting an
edge into the search graph.

### 12. Collision and Geometric Constraints

The urban environment (`src/webswing/geometry/buildings.py`) is a deterministic
collection of `Building` polygons (each a simple, ground-resting polygon validated
against its own descriptive width/height/roof-elevation fields) plus one axis-aligned
`DestinationRegion`. Geometric algorithms (`src/webswing/geometry/collision.py`)
operate on this data model:

- **Point-in-polygon**: even-odd (crossing-number) ray casting, preceded by an
  explicit boundary check so a point exactly on an edge is reliably classified inside.
- **Segment-segment / segment-polygon intersection**: an orientation-sign test with an
  explicit collinear/on-segment fallback, so a segment endpoint landing exactly on
  another segment (a tangent "touching corner") is correctly reported as an
  intersection rather than missed to floating-point noise.
- **Boundary policy**: polygon and region boundaries are treated as *closed*
  (inclusive) throughout — a trajectory that merely grazes a wall is conservatively
  treated as colliding with it, not as clear of it.
- **Bounding-box acceleration**: a cheap necessary-but-not-sufficient bounding-box
  overlap pre-filter (`segment_bounding_boxes_overlap`) is applied before the exact
  polygon test, expanded by the applicable clearance margin.
- **Continuous vs. sampled checking**: `segment_collides_with_city` tests the *exact*
  line segment between two given states, so it cannot tunnel between the two endpoints
  themselves. It *can* still miss a collision if the endpoints supplied to it are
  spaced farther apart than the true continuous path's curvature justifies between
  samples — a documented limitation of segment-sampling as a proxy for a continuous
  path, not a defect specific to any one caller (§20).
- **Anchor line-of-sight and self-occlusion** (`src/webswing/geometry/anchors.py`): a
  roof-corner anchor sits exactly on its own building's (closed) boundary, so a naive
  visibility check from any point to that anchor would always report a spurious
  self-collision. Every line-of-sight check therefore explicitly excludes the anchor's
  own building from the obstruction test, while still fully checking every other
  building.
- **Obstacle clearance**: each `Building` carries an optional `safety_margin`, combined
  additively with any caller-supplied `extra_clearance`, and enforced via
  `segment_polygon_min_distance` wherever exact intersection alone would allow a
  trajectory to pass arbitrarily close to (but not touching) a wall.

### 13. Local Optimal-Control Formulation

For a fixed current attached state and a fixed candidate target anchor,
`optimization.local_transfer.LocalTransferProblem` defines a single constrained
nonlinear optimization: choose a release time and a bounded, equal-interval
piecewise-constant radial control profile to reach the target anchor's capture region
in minimum time.

**Decision vector.**
$$
\mathbf{x} = (t_{\text{release}},\, u_1, \ldots, u_N), \qquad u_{\min} \leq u_i \leq u_{\max}, \qquad t_{\text{release,min}} \leq t_{\text{release}} \leq t_{\text{release,max}}, \tag{13.1}
$$
where $u_i$ is the constant radial acceleration on the $i$-th of $N$ equal-width
intervals of $[0, t_{\text{release}}]$ (`equal_interval_control`, §6).

**Objective.** With capture modelled as instantaneous (§11, so $t_{\text{capture}}=0$)
and the ballistic phase terminating at first entry to the target anchor's capture
region (an event, not a free decision), the true objective reduces to

$$
J = t_{\text{release}} + t_{\text{ballistic}}. \tag{13.2}
$$

For candidates that fail to reach release or fail to capture, `minimize` still requires
a finite, informative objective value at every evaluated point; the implementation uses
two distinct documented **surrogates**, $w = 10^3$, neither ever confused with the true
objective (13.2) (`src/webswing/optimization/local_transfer.py`, `_objective`):

$$
J_{\text{surrogate}} =
\begin{cases}
t_{\text{swing reached}} + w \cdot \max(0,\, -\text{swing margin}) & \text{swing phase never reached } t_{\text{release}}, \\[4pt]
t_{\text{release}} + t_{\text{ballistic}} + w \cdot \max(0,\, -\text{capture margin}) & \text{reached release but did not capture}.
\end{cases} \tag{13.2'}
$$

Both surrogates are strictly non-decreasing in how badly the candidate misses
feasibility, giving SLSQP a continuous gradient signal even where the true objective
(13.2) is undefined, without ever reporting a surrogate value as if it were (13.2).

**Constraints.** The swing-phase feasibility margins (all required to be $\geq 0$;
`src/webswing/optimization/constraints.py`) are

$$
T_{\max} - T(t) \geq 0, \qquad T(t) \geq 0, \qquad
\ell(t) - \ell_{\min} \geq 0, \qquad \ell_{\max} - \ell(t) \geq 0, \tag{13.3}
$$
$$
\nu_{\max} - |\dot\ell(t)| \geq 0, \qquad n_{\max} - T(t)/(mg) \geq 0, \tag{13.4}
$$
evaluated at a **fixed count** of sample times spanning $[0, t_{\text{release}}]$
(`_N_SWING_SAMPLES = 20`); if a swing event fires before $t_{\text{release}}$ is
reached, later samples repeat the state at the truncation boundary — itself at or past
a violated margin, which is exactly why the event fired — so the fixed-length
constraint vector `scipy.optimize.minimize` requires still signals infeasibility rather
than fabricating feasibility for times never actually simulated. A capture-margin
constraint, $\text{capture\_radius} - (\text{closest ballistic approach to the target
anchor}) \geq 0$, is appended as the final constraint component.

**Solver.** `scipy.optimize.minimize` with `method="SLSQP"` by default (CLAUDE.md's
suggested initial option), a deterministic midpoint-release/zero-control initial guess
(no random seed needed), and `{"maxiter": 200, "ftol": 1e-8}` unless overridden. Every
`LocalTransferResult` records solver name, initial guess, raw termination status and
message, objective value, maximum constraint violation, objective-evaluation count,
and the total number of `solve_ivp` integrations performed (cache-deduplicated across
repeated decision vectors via `_MemoizedSimulator`) — the full CLAUDE.md reporting
requirement.

**Collision is post-hoc, not a live constraint.** If a `City` is supplied, the
converged candidate's full swing+ballistic path is checked once, after optimization,
against `geometry.collision.trajectory_collides_with_city`; a collision overrides
`success` to `False` with an explicit `failure_reason`, but is never embedded as a
differentiable per-iteration constraint during the search itself. This is a documented
scope simplification (§19), not an oversight.

**Numerically successful $\neq$ feasible.** `LocalTransferResult.success` is `True`
only if the SLSQP solver itself converged **and** the swing phase reached release
**and** the ballistic phase captured the target anchor **and** (if checked) the result
does not collide — a solver return code of "converged" alone is never reported as a
usable result.

### 14. Global State-Space Construction

A planning node cannot be modelled as an anchor identifier alone: transfer feasibility
and cost depend on the *incoming attachment state*
$(\theta, \omega, \ell, \nu)$, so the anchor alone is not a Markov state for the search
(CLAUDE.md, and `src/webswing/planning/state.py`'s module docstring). Instead,
`PlanningState` pairs an anchor identifier with a **discretized bin index** along each
of the four continuous attachment-state dimensions:

$$
\text{bin}(v, w) = \left\lfloor \frac{v}{w} \right\rfloor, \tag{14.1}
$$

applied independently to $\theta$ (bin width $w_\theta$), $\omega$ ($w_\omega$),
$\ell$ ($w_\ell$), and $\nu$ ($w_\nu$) — a half-open, floor-division binning uniform
across positive and negative values, with angle wraparound *not* handled (a documented
simplification reasonable for a swinging web that does not wrap past $\pm\pi$ in normal
operation, not a correctness guarantee if it somehow did). Because `PlanningState` is a
frozen, hashable dataclass, it serves directly as a dict/set key for A*'s closed-set and
duplicate-state detection. `representative_state` maps a discrete node back to its
**bin-center** continuous state, $(i + \tfrac12)w$ per axis, used to seed the next local
transfer optimization from a node popped off the open set.

**Edges.** Each edge in the graph is one locally optimized feasible transfer (§13)
between two `PlanningState` nodes, with edge cost equal to elapsed travel time
(`release_time + ballistic_time` if successful, $+\infty$ otherwise;
`planning.cache.edge_cost`). Repeated `(incoming state, candidate anchor)` pairs are
memoized (`TransferCache`) so A* never re-solves the same local optimization twice.
A successful ballistic capture does not by itself guarantee a valid attachment — the
resulting continuous state is passed through the attachment transition (§11), and only
if that transition is *also* accepted does an edge to the resulting discretized node
exist. Per CLAUDE.md, the goal is the destination *region*, not a specific anchor: for
every candidate transfer, the already-solved decision vector is re-simulated
(inexpensively, not re-optimized) and its sampled ballistic path scanned for direct
entry into the destination region; if found, an additional edge to a distinguished goal
node is added at that (possibly earlier, cheaper) interception time, alongside the edge
to the transfer's intended target anchor.

### 15. A\* Heuristic Admissibility

`astar_search` (`src/webswing/planning/astar.py`) is a generic, physics-free A*
implementation: closed-set handling, duplicate-state detection (a node already closed
is never re-expanded, correct under a *consistent* heuristic), parent reconstruction,
and a search-size termination limit, over any hashable node type and caller-supplied
neighbor/goal functions.

**The default heuristic is $h \equiv 0$.** CLAUDE.md permits
$h(n) = d_{\text{straight}}(n, \text{goal})/v_{\max}$ as an admissible lower bound *only
if* $v_{\max}$ is a genuine, established hard upper bound on achievable speed under this
model. **No such bound has been derived or certified in this project.** Establishing one
is nontrivial: ballistic speed can grow with the domain's height range via gravity, and
swing speed can additionally grow from work done by the radial control over however
much of the remaining path lies ahead, which is not obviously bounded without further
assumptions (e.g. a maximum total control-energy budget) that this project has not
introduced. Consequently, `zero_heuristic` is the implemented default and recommended
choice, which reduces A\* to Dijkstra's algorithm while preserving correctness — a
direct, explicit application of CLAUDE.md's fallback rule rather than an
unsubstantiated speed-bound claim. $h \equiv 0$ is trivially admissible (it never
overestimates a non-negative remaining cost) and trivially **consistent**:
$0 \leq \text{cost}(u, v) + 0$ holds for every non-negative edge cost, so the
non-reopening closed-set behavior above is provably correct under it.

A `speed_bound_heuristic(v_{\max})` is provided for future use once a $v_{\max}$ is
independently derived and justified, but supplying an unjustified or too-large
$v_{\max}$ **silently breaks A\*'s optimality guarantee without raising any runtime
error** — this is documented explicitly in `src/webswing/planning/heuristic.py` as a
caller responsibility, not a validated feature of this release. Its consistency (not
merely admissibility) on this specific graph has not been separately verified either.

Because the default heuristic reduces search to Dijkstra, any minimum-time claim this
project makes is properly stated as: *minimum-time within the defined candidate-anchor
graph, discretization resolution, and control parameterization, found by exhaustive
(non-heuristic-pruned) shortest-path search over that graph* — not a claim about the
true continuous-control minimum-time trajectory, which is a strictly different
(generally smaller) quantity this project does not compute (§19).

### 16. Algorithmic Complexity

Let:

- $B$ = number of buildings, $A$ = total number of candidate anchors across all
  buildings (`City.all_candidate_anchors`), so the candidate-anchor graph has $A + 1$
  possible attachment points (including the start anchor).
- $K$ = number of discretized bins actually populated per anchor along the four state
  axes (bounded above by, but generally far smaller than, the product of each axis's
  bin count over the state space actually explored — planning nodes are generated
  lazily, not pre-enumerated).
- $F$ = number of feasible outgoing line-of-sight-visible anchors considered from a
  given node (bounded by $A$).
- $C_{\text{opt}}$ = cost of one local transfer optimization (`solve_local_transfer`):
  a bounded number of SLSQP iterations, each requiring one swing-phase and (if release
  is reached) one ballistic-phase `solve_ivp` integration, i.e.
  $C_{\text{opt}} = O(\text{iterations} \times C_{\text{integrate}})$, memoized per
  `(from_state, to_anchor)` pair so repeated graph edges cost one integration instead
  of one per occurrence.
- $C_{\text{integrate}}$ = cost of one `solve_ivp` call, itself a function of the
  requested tolerances, event count, and trajectory duration/curvature (§17) — not a
  fixed constant, and not further bounded here.

Each node expansion considers up to $F$ outgoing edges, each requiring one
(cache-deduplicated) local optimization; with $N = (A+1) \cdot K$ total reachable
planning nodes, worst-case A\*/Dijkstra graph search is
$O(N \log N + N \cdot F \cdot C_{\text{opt}})$ using a binary-heap open set (as
implemented, via `heapq`), the $N \log N$ term from heap operations and the second term
dominated by local-optimization cost, which is by far the more expensive component in
practice (each $C_{\text{opt}}$ call performs on the order of tens of full ODE
integrations). Transfer-result caching turns what would otherwise be
$O(N \cdot F)$ *independent* optimizations (one per edge traversal, including
duplicates) into $O(N \cdot F)$ **memoized lookups** after at most $O(N \cdot F)$
unique `(from_state, anchor)` pairs are ever actually solved — the caching does not
change the worst-case bound but removes redundant re-solving of the same pair reached
by different search paths, which is the actual source of savings in practice. This
project has not measured empirical wall-clock scaling beyond the small example scenario
in `examples/basic_run.py`; the complexity statement above is structural, not a
benchmarked result (§19 and the Scientific Integrity requirement against fabricated
benchmark values).

### 17. Numerical Integration Configuration

Both integrators use `scipy.integrate.solve_ivp` with method `"RK45"` (explicit
Runge-Kutta 4(5)), relative tolerance $\texttt{rtol}=10^{-9}$, and absolute tolerance
$\texttt{atol}=10^{-11}$, with `dense_output=True` so terminal events and
post-hoc resampling can use continuous interpolation rather than only the discrete
accepted-step grid (`src/webswing/optimization/local_transfer.py`,
`simulate_transfer`). These tolerances are currently **inline literals at the single
call site**, not exposed through a dedicated, separately configurable typed integration
settings object — a gap relative to CLAUDE.md's stated requirement that integration
configuration ("relative tolerance, absolute tolerance, maximum integration step,
integration method, event-location tolerance, and maximum simulation duration") be
exposed as explicit, adjustable configuration; this is recorded here as a known
limitation (§20/§24), not silently presented as fully configurable.

**Maximum step bound (ballistic phase only).** `solve_ivp`'s adaptive stepping checks a
terminal event's sign only at each accepted step's endpoints; a smooth ballistic arc
can take a large step, and a brief capture-region entry/exit can occur entirely inside
one such step — invisible to the endpoint-based sign check even though the continuous
path truly passes through the capture disk. The implementation bounds the ballistic
integrator's `max_step` to

$$
\Delta t_{\max} = \min\left(\frac{r_{\text{capture}}}{\max(v_{\text{release}}, 10^{-6})},\ \frac{T_{\text{ballistic,max}}}{100}\right), \tag{17.1}
$$

the time to cross a capture-radius-scale distance at the release speed, capped so it is
never a large fraction of the flight's own duration budget — chosen specifically to
close a real event-detection gap found during development, not a generic numerical
safety margin. Event-location (root-finding) tolerance is left at `solve_ivp`'s own
internal default rather than separately configured. Maximum ballistic duration is
enforced as an explicit terminal event
(`make_ballistic_max_duration_event`, §9) rather than only as an integrator time-span
bound, so it is recorded as a distinct, inspectable termination reason.

**Finite-state checking.** `assert_state_finite`
(`src/webswing/dynamics/events.py`) raises `NonFiniteStateError` if any state component
is NaN or infinite; this is *not* expressible as a `solve_ivp` zero-crossing event
(any comparison against NaN is `False`, so a NaN state can never trigger a sign-change
event) and is applied directly to sampled/output states by the integration driver
instead.

### 18. Validation Against Analytic Solutions

Per CLAUDE.md, closed-form solutions are used as validation oracles, not as the runtime
code path. This project's test suite (not reproduced numerically in this document,
since specific pass/fail outcomes belong to CI output, not a methodology description)
validates, among other properties:

- **Ballistic motion** (`tests/dynamics/test_ballistic.py`) against the closed-form
  solution (9.2): position/velocity agreement, horizontal-velocity conservation,
  constant $-g$ vertical acceleration, and ground-impact time.
- **Swing dynamics** (`tests/dynamics/test_swing.py`): the fixed-length reduction
  ($\dot\ell = \ddot\ell = 0$ recovering the ordinary pendulum, §5), small-angle
  behavior against the linearized pendulum, energy conservation for a fixed-length,
  uncontrolled pendulum, and energy change under radial actuation matching the work
  done by the control.
- **Coordinate transformations** (`tests/dynamics/test_release.py`,
  `tests/dynamics/test_attachment.py`): release-transition continuity checked directly
  against coordinate differentiation (§10), and attachment-transition tangential-
  velocity preservation / radial-velocity removal / nonnegative energy loss checked
  against the closed-form expressions in §11.
- **Geometry** (`tests/geometry/test_collision.py`,
  `tests/geometry/test_anchors.py`): point-in/on/outside-polygon classification,
  segment-boundary touching cases, and anchor self-occlusion exclusion, each against
  hand-constructed cases with a known correct answer.
- **Planning** (`tests/planning/test_astar.py`,
  `tests/planning/test_heuristic.py`): A\* correctness on a small, manually verifiable
  graph, and its exact equivalence to Dijkstra when $h \equiv 0$.

Every such test specifies a physically justified absolute/relative tolerance
appropriate to the quantity under test (per this project's testing standard of never
weakening a tolerance solely to make a failing test pass); the specific numeric
tolerance values live with each test, not duplicated here, so this document cannot
drift out of sync with them.

### 19. Optimization Limitations

The following are explicit, acknowledged limitations of the optimization and planning
layers, not omissions discovered after the fact:

- **Local, not global, minimization.** `solve_local_transfer` uses SLSQP, a local
  nonlinear solver, from a single deterministic initial guess (no multi-start), so a
  reported "optimal" local transfer is a local optimum of that particular
  parameterization, not a certified global optimum of the underlying non-convex
  problem.
- **Fixed control parameterization.** The radial control is restricted to $N$
  equal-interval piecewise-constant segments; the true infimum-time control law is not
  restricted to this family, so even a perfectly solved instance of (13.1)-(13.4) is a
  minimum-time result *within this parameterization*, not an unconstrained
  optimal-control result.
- **Discretized global search.** The planning layer searches a graph built from binned
  attachment states (§14) and a finite anchor set (§12); it does not search the
  continuous space of all possible attachment states, release times, and controls
  jointly. Two different discretization resolutions can produce two different
  "optimal" routes.
- **Non-admissible heuristic option exists but is not the default and is not proven
  consistent** (§15) — using it without independently justifying $v_{\max}$ silently
  forfeits any optimality claim.
- **Collision is a post-hoc filter, not a live constraint** (§13) — a converged local
  transfer can be numerically optimal by (13.2)-(13.4) and still be discarded entirely
  for colliding, rather than the collision margin shaping the search toward a
  compliant optimum directly.
- **No proven certificate of infeasibility.** A failed local transfer or a
  no-route-found global search result means no *feasible solution was found* by this
  solver configuration within its iteration/expansion limits — it is not a proof that
  no feasible trajory exists.

Consequently, per CLAUDE.md's scientific-integrity requirement, this project's outputs
must be described as "a minimum-time trajectory found within the defined
candidate-anchor graph and control parameterization," never as "the globally optimal
trajectory," in any report, figure caption, or downstream document.

### 20. Sources of Discretization Error

- **Planning-state binning** (§14): two continuous attachment states in the same bin
  are treated as identical for graph-search purposes (both edge feasibility and cost
  are computed from the bin-center representative state, not the true incoming state),
  introducing an error bounded by the bin widths but not otherwise quantified here.
- **Fixed-count constraint sampling** (§13): swing-phase feasibility is evaluated at a
  fixed 20 sample times, not continuously; a constraint violation strictly between two
  samples could in principle be missed by the optimizer's constraint function, though
  `dynamics.events` terminal events (§6, §8) independently catch tension/length/
  radial-speed/load-factor violations during the underlying integration regardless of
  the optimizer's own sampling.
- **Segment-based collision sampling** (§12): `trajectory_collides_with_city` is exact
  *between* the supplied sample points but can tunnel through a thin obstacle if
  consecutive samples are spaced farther apart than the obstacle scale and path
  curvature justify — a documented, inherent limitation of segment-sampling as a proxy
  for the continuous path, not resolved by this project via an enforced maximum
  sampling interval at every call site.
- **Numerical integration tolerance** (§17): `rtol`/`atol` bound local per-step error,
  not global trajectory error over a long integration; no global error bound is derived
  or asserted here.
- **Destination-interception truncation**: when a ballistic edge is truncated at first
  entry into the destination region, the kept sample times are clipped to a fixed
  sampling grid, not the exact continuous entry instant; the assembled trajectory's
  *total duration* uses the edge's own analytically tracked cost (not the last kept
  grid sample's time) specifically to avoid accumulating this grid-rounding error
  across multiple edges (`src/webswing/simulation/trajectory.py`).

### 21. Sensitivity Analysis

This project does not currently include a systematic sensitivity study (e.g. a swept
parameter grid over bin widths, control-segment count, or tolerance values with
recorded route-cost/feasibility outcomes); no such results are fabricated here to fill
that gap. Based on the structural role each parameter plays in the modules above, the
quantities most likely to materially affect a route's reported minimum time are: the
planning-state bin widths (coarser bins merge more distinct states, generally
*shortening* apparent search effort at the cost of losing distinctions that could yield
a cheaper true route), the number of radial control segments $N$ (more segments enlarge
the reachable control family and can only improve or match the best transfer found, at
increased per-optimization cost), and `capture_radius` (a larger radius eases capture
feasibility but coarsens the geometric precision of "reaching" an anchor). A rigorous
sensitivity study is future work, not a claim this document makes.

### 22. Reproducibility Requirements

- **Language/runtime**: Python $\geq 3.12$ (`pyproject.toml`); developed and last
  verified under Python 3.13.9.
- **Core numerical dependencies** (as installed in the environment this document was
  written against; `pyproject.toml` does not currently pin exact versions, which is
  itself a reproducibility gap recorded here rather than concealed):
  NumPy 2.3.5, SciPy 1.16.3, Matplotlib 3.10.6.
- **Determinism**: `solve_local_transfer`'s default initial guess is a fixed
  midpoint-release/zero-control vector (no random seed required); where this project's
  test suite does use randomness (e.g. reproducible star-field placement in
  visualization, unrelated to physics), it is seeded via `numpy.random.Generator`
  with a fixed seed, per this project's testing standard.
- **Integration tolerances**: `rtol`$=10^{-9}$, `atol`$=10^{-11}$, method `"RK45"`,
  §17 — required to reproduce a reported trajectory's numerical values to the stated
  precision.
- **Full parameter set**: reproducing a specific reported route requires the exact
  `PhysicalParameters`, `SwingConstraints`, `BallisticDomain`,
  `PlanningStateResolution`, control bounds ($u_{\min}, u_{\max}$, $N$), release-time
  bounds, `capture_radius`, and `max_expansions` used to produce it — every one of
  these is a required, explicit argument to `simulation.runner.run_simulation` (no
  hidden global default silently substitutes for an omitted physical parameter).

### 23. Interpretation of Results

A `SearchResult`/`Trajectory` produced by this pipeline should be read as: *the
minimum-elapsed-time route among all feasible transfers this software's local
optimizer could find, between anchors in the supplied city's candidate set, using the
supplied control parameterization and planning-state discretization, subject to the
supplied physical constraints* — with every qualifier in that sentence load-bearing.
`LocalTransferResult.max_constraint_violation`, `TrajectoryEvaluation`'s tension/
load-factor margins, and `SearchResult.failure_reason` are the mechanisms by which a
consumer of a result can check *how* feasible (or why infeasible) a given output
actually is, rather than trusting a bare success flag.

### 24. Physical Limitations of the Model

- No air resistance on either attached or ballistic motion (§3), which would in
  reality reduce peak speeds and alter both swing and ballistic trajectories,
  especially at the speeds a "minimum time" objective tends to favor.
- No body extent, rotational inertia, or orientation — a point mass cannot model
  posture changes, tucking, or aerodynamic body shaping a real swinging body would use.
- The web is massless and applies force only along its own axis; no web elasticity,
  damping, or slack-then-taut dynamics are modelled (a slack web is treated as loss of
  tension and termination of attached motion, per §8, not as a separate compliant
  regime).
- Attachment is instantaneous and non-conservative by construction (§11); a real
  grappling/adhesion process would have finite duration and a different, apparatus-
  specific energy-loss profile.
- The city is fully known and static; no sensing, uncertainty, or dynamic obstacles.
- Integration tolerances and discretization resolutions (§17, §20) bound numerical
  error but do not eliminate it; no claim of exact physical realism is made beyond the
  stated assumptions.

### 25. Fictional Assumptions vs. Real Mechanics

To make the distinction required by CLAUDE.md explicit and in one place: the *only*
fictional element of this project is its narrative framing — a human-scale point mass
swinging through a city on a controllable web is not something a real unaided human can
do. Every equation in §4-§11 (Lagrangian derivation, Euler-Lagrange equation, the
first-order ODE system, the Cartesian mappings, the tension/load-factor formulas, the
ballistic equations, and the release/attachment transitions) is ordinary Newtonian and
Lagrangian mechanics, with no fictional physics substituted anywhere: gravity is
standard gravity, tension is derived from an unmodified radial force balance, and no
energy is created or destroyed except through the explicitly modelled work of the
radial control and the explicitly modelled, always-nonnegative attachment energy loss.
The optimization and planning layers (§13-§16) are likewise standard nonlinear
constrained optimization (SLSQP) and graph search (A\*/Dijkstra); nothing about the
"Spider-Man-inspired" framing relaxes, replaces, or shortcuts any part of the
mathematics. Where this project's *model* is a simplification of real mechanics (§3,
§24), that simplification is stated as a modelling assumption, not attributed to the
fictional premise.

---

*This document should be treated as a living reference: any change to a sign
convention, tolerance, default parameter, or algorithmic choice in the modules cited
above must be reflected here in the same change, per this project's requirement that
scientific documentation not silently drift out of sync with the implementation it
describes.*
