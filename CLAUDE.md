# CLAUDE.md — Operating Contract

## Project

**Kinematic Optimization of Urban Brachistochrone Trajectories: Anchor Selection and Optimal Control**

A mathematically rigorous simulation of Spider-Man-inspired web-swinging through a two-dimensional
urban environment. The system computes a minimum-time feasible trajectory from a specified start
state to a destination region. Motion alternates between:

1. Web-attached motion — a controlled variable-length pendulum attached to a fixed anchor.
2. Ballistic motion — projectile motion after web release.
3. Web attachment events — a ballistic trajectory connects to a new anchor and transitions back
   into constrained swinging motion.

The final system combines continuous-time dynamics, nonlinear constrained optimization, geometric
collision detection, graph search, automated testing, animation, and research-grade mathematical
documentation.

---

## Approval and Scope Control

- Never create the entire project in one response.
- Never advance automatically from one major component to another.
- Work only on the exact file, test, derivation, or component requested.
- Stop after completing the requested scope.
- Wait for explicit approval before beginning additional work.
- Do not interpret silence as approval.
- Do not create unrelated files for convenience.
- Before modifying an existing file, inspect its current contents and all directly related tests.
- Preserve existing interfaces unless a change is mathematically or architecturally necessary.
- When an interface change is necessary, explain the reason before applying it.

## Git Rules

- Never run `git commit`.
- Never run `git push`.
- Never amend, rebase, squash, reset, or rewrite Git history.
- Never invent commit dates.
- Never execute a commit on the user's behalf.
- After completing and validating an approved change, stage only the files associated with that
  change.
- Use `git diff --staged` and `git status` to show exactly what is staged.
- Propose one conventional commit message, but do not execute it.
- The user provides the final commit command and the required `--date` value.

Conventional commit examples:

```text
feat(dynamics): implement controlled variable-length pendulum equations
test(dynamics): verify free-fall trajectory against closed-form solution
feat(planning): add admissible A-star trajectory heuristic
docs(methodology): derive swinging equations from the Lagrangian
fix(collision): correct segment-polygon boundary handling
```

## Testing Requirements

- Every mathematical component must have tests before dependent components are started.
- A mathematical implementation is not complete until its tests pass.
- Use `pytest`.
- Use deterministic random seeds for stochastic tests.
- Prefer analytic comparisons, invariants, boundary cases, and property-based checks over tests
  that merely confirm code execution.
- Numerical tests must specify physically justified absolute and relative tolerances.
- Never weaken a tolerance only to make a failing test pass.
- Investigate whether a failure comes from the implementation, numerical integration, event
  handling, or an incorrect test assumption.
- Never delete a valid failing test to continue development.
- Run the narrowest relevant tests while editing.
- Run the full test suite before staging a completed component.
- Report the exact commands executed and the final test result.

## Scientific Integrity

- Do not describe a trajectory as the absolute global optimum unless it has been mathematically
  established under the stated discretization, control parameterization, anchor set, and numerical
  tolerances.
- Use language such as "minimum-time trajectory found within the defined candidate-anchor graph
  and control parameterization" when global optimality has not been proven.
- Clearly distinguish modelling assumptions from physical laws.
- Do not fabricate numerical results, benchmark values, plots, citations, or conclusions.
- Do not conceal solver failures, integration failures, infeasible controls, or numerical
  instability.
- Record solver termination status and constraint violations.
- Preserve units consistently.
- Use SI units unless a documented alternative is explicitly required.
- Document all sign conventions and coordinate conventions.
- Do not silently change the physics model to obtain a visually attractive result.

## Documentation Style

Every source file must contain:

- A module-level docstring describing its scientific responsibility.
- Complete type annotations.
- NumPy-style docstrings for public classes and functions.
- Definitions of parameters, units, return values, assumptions, and failure conditions.
- Technical comments explaining non-obvious mathematical transformations, numerical methods,
  event functions, coordinate conversions, and algorithmic decisions.

Comments must be research-grade and implementation-specific. Do not use tutorial-style narration
such as "Now we calculate...", "In this step...", "Here we simply...", "As you can see...",
"First, we will...", "This teaches us...". Do not use conversational commentary, motivational
language, AI-style checklists, or comments that merely restate the code. Mathematical comments
must explain why the implementation corresponds to the stated model.

## Code Quality

- Target Python 3.12 or the repository's configured version if one already exists.
- Use `numpy`, `scipy`, `matplotlib`, and `pytest` where appropriate.
- Use `shapely` only if selected deliberately for robust polygon operations.
- Use immutable data structures where practical.
- Prefer explicit scientific data classes over unstructured dictionaries.
- Separate physics, geometry, optimization, planning, visualization, and persistence concerns.
- Avoid hidden mutable global state.
- Avoid unexplained constants.
- Store physical and numerical parameters in typed configuration objects.
- Validate public inputs.
- Raise domain-specific exceptions for invalid physical states.
- Use structured logging rather than scattered print statements in library code.
- Use reproducible random generators through `numpy.random.Generator`.
- Do not introduce machine learning unless explicitly requested.
- Do not introduce a web framework, cloud service, database, notebook, or user interface unless
  explicitly requested.

## Required Completion Report

After every approved change, report only:

1. Files created or modified.
2. Mathematical or algorithmic responsibility implemented.
3. Tests added.
4. Test commands and results.
5. Files staged.
6. Proposed conventional commit message.
7. Known limitations or unresolved numerical issues.

Then stop.

---

## Scientific Model

### Coordinate Convention

Two-dimensional Cartesian coordinate system:

- $x$: horizontal position, increasing to the right.
- $y$: vertical position, increasing upward.
- Ground: $y = 0$.
- Gravity: $\mathbf{g} = (0, -g)$, where $g > 0$.

For a fixed web anchor at $\mathbf{a} = (x_a, y_a)$, define the swing angle $\theta$ relative to
the downward vertical direction. The Spider-Man position during attached motion is:

$$x = x_a + \ell \sin\theta, \qquad y = y_a - \ell \cos\theta.$$

This convention must be used consistently throughout the dynamics, geometry, transition, tension,
and visualization code.

### Attached Swinging Dynamics

**State vector.** The variable-length pendulum cannot be represented only by $\theta$ and $\ell$.
Use the state:

$$\mathbf{z} = \begin{bmatrix} \theta \\ \omega \\ \ell \\ \nu \end{bmatrix}, \qquad
\omega = \dot\theta, \qquad \nu = \dot\ell.$$

The radial control is $u(t) = \ddot\ell$. Positive $u$ increases web length under the chosen
convention; negative $u$ retracts the web.

**Lagrangian.** For point mass $m$:

$$T = \tfrac{1}{2} m \left( \dot\ell^2 + \ell^2 \dot\theta^2 \right), \qquad
U = -mg\ell\cos\theta, \qquad L = T - U.$$

The Euler-Lagrange equation for $\theta$ gives:

$$\ddot\theta = -\frac{2\dot\ell}{\ell}\dot\theta - \frac{g}{\ell}\sin\theta.$$

With $u = \ddot\ell$, the first-order controlled system is:

$$\dot\theta = \omega, \qquad
\dot\omega = -\frac{2\nu}{\ell}\omega - \frac{g}{\ell}\sin\theta, \qquad
\dot\ell = \nu, \qquad \dot\nu = u.$$

Implement this system using `scipy.integrate.solve_ivp`. The control interface must support at
least: constant radial acceleration, piecewise-constant radial acceleration, and a bounded
parameterized control profile suitable for optimization. Do not introduce an arbitrary energy
gain — mechanical energy may change only through work performed by web retraction or extension.

**Cartesian velocity.** Derived from the coordinate mapping:

$$\dot x = \dot\ell \sin\theta + \ell \dot\theta \cos\theta, \qquad
\dot y = -\dot\ell \cos\theta + \ell \dot\theta \sin\theta.$$

These equations must be used for release transitions and all reported velocity values.

**Web tension.** Under the stated sign convention, radial force balance gives:

$$T = m \left( \ell \omega^2 + g\cos\theta - \ddot\ell \right).$$

Do not use a tension expression with the opposite sign on $\ddot\ell$ unless the radial coordinate
convention is changed and the full derivation is updated consistently. A swinging state is
infeasible if $T > T_{\max}$. A web cannot provide compressive force; unless a slack-web model is
explicitly introduced, also treat $T \le 0$ as loss of tension and termination of attached motion.

Keep web-strength limits and human acceleration limits separate — they are different constraints.
Support configurable constraints for: maximum web tension, maximum permitted load factor or
acceleration magnitude, minimum web length, maximum web length, maximum retraction/extension
speed, maximum radial acceleration, and minimum obstacle clearance.

**Swing termination events.** The attached integrator must support terminal event detection for:
ground contact, building collision, web tension exceeding its maximum, web tension becoming
non-positive, web length leaving its permitted interval, radial speed exceeding its permitted
magnitude, load-factor violation, requested release condition, numerical state invalidity, and web
segment intersecting a building when line-of-sight constraints are active. All event functions
must be individually testable.

### Ballistic Free-Fall Dynamics

**State vector.**

$$\mathbf{q} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}.$$

Without aerodynamic drag: $\dot x = v_x$, $\dot y = v_y$, $\dot v_x = 0$, $\dot v_y = -g$.

Implement the ballistic trajectory using `solve_ivp`, even though a closed-form solution exists.
The closed-form solution must be used as the primary test oracle:

$$x(t) = x_0 + v_{x,0} t, \qquad y(t) = y_0 + v_{y,0} t - \tfrac{1}{2} g t^2, \qquad
v_x(t) = v_{x,0}, \qquad v_y(t) = v_{y,0} - g t.$$

**Ballistic termination events.** Support terminal events for: ground impact, building collision,
entry into the destination region, entry into the feasible capture region of an anchor, exceeding
a maximum ballistic duration, exiting the bounded city domain, and numerical invalidity.

### Release Transition

At web release, convert the attached state to Cartesian ballistic state using the exact position
and velocity mappings above. The release transition must preserve position continuity, velocity
continuity, and simulation time continuity. No artificial impulse may be applied during release.
Tests must verify the transition against direct coordinate differentiation.

### Attachment Transition

A new web attachment is not automatically energy-preserving. Given a candidate anchor
$\mathbf{a}$, define $\mathbf{r} = \mathbf{x} - \mathbf{a}$, $\ell = |\mathbf{r}|$, and the
outward radial unit vector $\mathbf{e}_r = \mathbf{r}/\ell$. Decompose the pre-attachment velocity
$\mathbf{v} = \mathbf{v}_r + \mathbf{v}_t$, where $\mathbf{v}_r = (\mathbf{v}\cdot\mathbf{e}_r)\mathbf{e}_r$
and $\mathbf{v}_t = \mathbf{v} - \mathbf{v}_r$.

Use a documented default capture model in which the radial velocity is removed instantaneously and
the tangential velocity is retained: $\mathbf{v}^+ = \mathbf{v}_t$. This is an idealized inelastic
attachment. Record the kinetic-energy loss:

$$\Delta E = \tfrac{1}{2} m \left( |\mathbf{v}|^2 - |\mathbf{v}^+|^2 \right).$$

Reject the attachment if: the anchor is outside the maximum attachment range; the web segment
intersects a building; the required attachment impulse exceeds a configured limit; the resulting
state violates tension or acceleration constraints; the computed web length is outside its
permitted interval; or the attachment geometry is numerically degenerate.

Convert the retained tangential velocity into the corresponding angular velocity using the
established coordinate convention. Do not conceal the energy loss introduced by attachment.

### Urban Environment

The city is deterministic two-dimensional geometry. Each building must contain at least: unique
identifier, polygon boundary, width, height, roof elevation, candidate anchor points, and optional
safety margin. Candidate anchors should initially be derived from geometrically valid roof corners
or explicitly configured roof points.

The environment must support: point-in-polygon checks, segment-polygon intersection checks,
continuous trajectory collision checks, bounding-box acceleration, ground collision,
destination-region checks, anchor line-of-sight validation, and configurable obstacle clearance.

Do not rely only on coarse sampled points for collision detection when a continuous event or
segment intersection method is available. If trajectory sampling is used as a fallback, document
the possibility of tunnelling and enforce a maximum sampling interval derived from velocity and
obstacle scale.

### Local Swing Optimization

For a fixed current state and candidate anchor transfer, solve a constrained nonlinear
optimization problem. The optimizer may choose: release time; release angle when uniquely defined
by time; parameters of the radial control profile $u(t)$; optional capture time; optional target
anchor within a fixed candidate set.

The control must remain bounded: $u_{\min} \le u(t) \le u_{\max}$. Additional constraints include:

$$\ell_{\min} \le \ell(t) \le \ell_{\max}, \qquad |\dot\ell(t)| \le \nu_{\max}, \qquad
0 < T(t) \le T_{\max},$$

together with collision, clearance, capture, and load constraints.

Use `scipy.optimize` with a solver appropriate for nonlinear constrained optimization. Suitable
initial options include `SLSQP` and `trust-constr`. Use deterministic multi-start initialization
when needed. Do not report a failed optimizer result as feasible.

Every optimization result must record: solver name, initial guess, termination status, objective
value, maximum constraint violation, number of objective evaluations, number of dynamics
integrations, final transfer state, and failure reason if infeasible.

A local transfer objective may minimize $J = t_{\text{release}} + t_{\text{ballistic}} +
t_{\text{capture}}$, subject to reaching a valid capture state. When a capture target is not
fixed, a surrogate objective may combine forward progress and travel time, but the surrogate must
be explicitly documented and must not be confused with the global minimum-time objective.

### Global Trajectory Planning

The global planner must use A* or a rigorously justified equivalent shortest-path method. Do not
model a planning node using only an anchor identifier — transfer feasibility and cost depend on
incoming velocity and attachment state, so the anchor alone is not a Markov state. A planning node
must include a discretized physical state sufficient to evaluate future transfers, such as: anchor
identifier, attachment angle bin, angular velocity bin, web-length bin, radial-velocity bin,
optional arrival-time or control-state information.

Each edge represents a locally optimized feasible transfer between two planning states, with edge
cost equal to elapsed travel time. The planner must support: candidate anchor generation, feasible
neighbour generation, local transfer optimization, transfer-result caching, duplicate-state
detection, closed-set handling, parent reconstruction, destination interception during ballistic
motion, search termination limits, and explicit reporting when no feasible route is found.

**A* heuristic.** The heuristic must be admissible if global optimality within the discretized
state graph is claimed. A possible lower bound is $h(n) = d_{\text{straight}}(n, \text{goal}) /
v_{\max}$, only when $v_{\max}$ is a valid hard upper bound under the model. If no defensible
finite speed bound has been established, use $h(n) = 0$, which reduces A* to Dijkstra's algorithm
while preserving correctness. Do not use an aggressive non-admissible heuristic while claiming
optimality.

**Complexity analysis.** Documentation must distinguish: number of candidate anchors, number of
discretized states per anchor, number of feasible outgoing transfers, cost of one local
optimization, cost of one dynamics integration, worst-case graph-search complexity, and effects of
transfer caching and pruning.

### Numerical Integration Requirements

Use explicit integration tolerances. Configuration must expose: relative tolerance, absolute
tolerance, maximum integration step, integration method, event-location tolerance, and maximum
simulation duration.

The implementation must detect: NaN or infinite states, invalid web lengths, integrator failure,
event-order ambiguity, simultaneous terminal events, excessive solver steps, and constraint
violations between sampled output points.

Dense output may be used for event refinement and animation. Do not use animation sampling as the
numerical integration grid.

---

## Testing Expectations

The final repository must eventually include tests for all of the following.

**Swing dynamics:** state derivative dimensions and units; fixed-length reduction when
$\dot\ell = 0$ and $\ddot\ell = 0$; small-angle behaviour against the linearized pendulum; energy
conservation for a fixed-length pendulum without control; energy change under radial actuation;
tension calculations for analytically simple states; tension loss; maximum-tension termination;
web-length and radial-speed constraints.

**Ballistic dynamics:** agreement with the closed-form trajectory; horizontal velocity
conservation; vertical acceleration equal to $-g$; ground-impact time; building collision;
destination interception.

**Coordinate transformations:** position conversion; velocity conversion; release continuity;
angle reconstruction; attachment-state reconstruction; degenerate-anchor rejection.

**Geometry:** point inside and outside polygons; boundary handling; segment crossing a building;
segment touching a corner; safe tangent contact according to the chosen policy; clearance margins;
ground intersection; anchor visibility.

**Attachment:** tangential velocity preservation; radial velocity removal; energy-loss
calculation; impulse-limit rejection; invalid line-of-sight rejection; resulting-state constraint
checks.

**Optimization:** feasible analytic toy cases; infeasible transfer detection; bound enforcement;
reproducibility from fixed seeds; solver-failure propagation; constraint-violation reporting.

**Global planning:** A* correctness on a small manually verifiable graph; equivalence to Dijkstra
when $h = 0$; admissible-heuristic behaviour; parent reconstruction; state-discretization
consistency; transfer caching; no-route behaviour; goal interception during a ballistic edge.

---

## Target Repository Structure

This is the intended final architecture. Do not create all of it immediately.

```text
.
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── src/
│   └── webswing/
│       ├── __init__.py
│       ├── config.py
│       ├── exceptions.py
│       ├── models.py
│       ├── geometry/
│       │   ├── __init__.py
│       │   ├── buildings.py
│       │   ├── collision.py
│       │   └── anchors.py
│       ├── dynamics/
│       │   ├── __init__.py
│       │   ├── swing.py
│       │   ├── ballistic.py
│       │   ├── events.py
│       │   ├── release.py
│       │   └── attachment.py
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── controls.py
│       │   ├── local_transfer.py
│       │   └── constraints.py
│       ├── planning/
│       │   ├── __init__.py
│       │   ├── state.py
│       │   ├── heuristic.py
│       │   ├── astar.py
│       │   └── cache.py
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── trajectory.py
│       │   ├── evaluator.py
│       │   └── runner.py
│       └── visualization/
│           ├── __init__.py
│           ├── static.py
│           ├── animation.py
│           └── hud.py
├── tests/
│   ├── geometry/
│   ├── dynamics/
│   ├── optimization/
│   ├── planning/
│   └── simulation/
├── configs/
├── examples/
├── docs/
│   ├── methodology.md
│   ├── assumptions.md
│   └── numerical_validation.md
└── outputs/
    ├── figures/
    ├── animations/
    └── trajectories/
```

Changes to this structure require an architectural reason and explicit user approval.

---

## Visualization Requirements

The completed visualization system must use `matplotlib.animation`. It must render: building
polygons; ground; start location; destination region; candidate anchors; selected anchors; active
web line; full trajectory trace; attached and ballistic portions with visually distinguishable
line styles or markers; current Spider-Man position; current velocity vector; collision or
constraint-failure location when applicable.

The dynamic HUD must display: simulation time; current motion mode; position; speed; active
anchor; web length; retraction or extension speed; angular velocity; current tension; maximum
permitted tension; load factor; current planner edge; cumulative travel time.

Animation code must consume stored trajectory output. It must not rerun the optimizer or numerical
integrator while rendering. Support deterministic export to a standard animation format when the
necessary local encoder is available.

---

## Academic Documentation

Create a research-grade Markdown document later in the repository at `docs/methodology.md`. It
must be suitable for direct adaptation into an Overleaf research paper. All mathematical
expressions must use valid LaTeX.

The document must eventually include: (1) problem definition; (2) coordinate and sign conventions;
(3) modelling assumptions; (4) derivation of the variable-length pendulum Lagrangian; (5)
Euler-Lagrange derivation; (6) controlled first-order ODE system; (7) Cartesian coordinate and
velocity mapping; (8) radial force balance and web-tension derivation; (9) ballistic equations;
(10) release transition; (11) attachment impulse model; (12) collision and geometric constraints;
(13) local optimal-control formulation; (14) global state-space construction; (15) A* heuristic
admissibility; (16) algorithmic complexity; (17) numerical integration configuration; (18)
validation against analytic solutions; (19) optimization limitations; (20) sources of
discretization error; (21) sensitivity analysis; (22) reproducibility requirements; (23)
interpretation of results; (24) physical limitations of the model; (25) clear distinction between
fictional assumptions and real mechanics.

Do not describe numerical evidence as a mathematical proof. Do not claim physical realism beyond
the assumptions actually implemented.

---

## Expected Development Behaviour

When a component is requested, inspect the repository and identify the smallest coherent set of
files required. Before writing code, state concisely:

- The exact file or files to be modified.
- The mathematical responsibility of each file.
- The tests that will validate the implementation.
- Any interface assumptions.

Then perform only the approved work. Do not continue into the next component. Do not generate
placeholder implementations for future modules. Do not leave `TODO` implementations that silently
return dummy values. If a requested component depends on an interface that does not yet exist,
define only the minimum typed interface required for the current work.
