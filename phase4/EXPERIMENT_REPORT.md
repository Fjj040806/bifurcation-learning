# Phase 4 — Robustness and non-radial dynamics

## Fixed research question
Can finite observations support recovery of bifurcation locations, stability and attractors? Which observation designs and declared structural priors improve recovery?

## Completed experiments
- 150 unconstrained polynomial robustness fits: passive, random, three probe amplitudes; two windows, three noise levels, five acquisition seeds.
- 150 rotationally equivariant fits on the same acquisitions.
- 20 polynomial one-sided extrapolation fits.
- 24 NODE sensitivity runs: six controls, two acquisition seeds, two model seeds.
- Non-radial FHN: one polynomial fit and five NODE runs.
- Local Hopf coefficients and 21 periodic shooting points rechecked at tighter tolerance.
- True-trajectory glow renders in PNG, GIF and MP4.

## Protocol and interpretation limits
Radial training acquisitions each cost 4096 scalar observations: 32 trajectories × 64 full-state samples. Parameters are [-.4,-.2,.2,.4].
Acquisition uses oracle attractor anchors for passive/probe designs; anchor search and physical perturbation costs remain excluded.
Probe offsets are absolute radii .005,.02,.1. Noise is iid Gaussian observation noise with absolute coordinate standard deviation 0,.01,.05; no process noise.
Windows T=5,20 retain 64 samples. Increasing the window also changes sampling interval, physical training segment length and RK4 discretization. It is not an isolated effect of observation duration.
NODE controls share a clean transient validation set (1024 scalar observations). This is idealized checkpoint selection, not noisy real-world validation. No true derivative, Jacobian or critical-point targets are used for training.
Validation differs from Phase 3; compare scores within this phase. The same NODE architecture, optimizer and 3600-step cap are retained. Histories are saved, but global optimizer convergence is not claimed.
Four NODE runs per control combine two data seeds and two initialization seeds; they are not four independent acquisition replicates.

## Polynomial noise sensitivity
T=5 median candidate-location error; each cell below has five unique candidates:

| Design | sigma=0 | sigma=.01 | sigma=.05 |
|---|---:|---:|---:|
| Random | 1.33e-5 | .00192 | .04583 |
| Probe .005 | 4.75e-9 | .01148 | .09412 |
| Probe .02 | 1.07e-7 | .00305 | .08271 |
| Probe .10 | 1.75e-6 | .00231 | .02366 |

Tiny probes are accurate without noise but vulnerable when noise is comparable to the perturbation. Filtering and errors-in-variables bias also contribute; this is not a universal optimal-amplitude law.
All raw seeds, candidates, failures and orientations are retained in robustness_summary.json and CSV/JSON tables. Passive failures are retained even though the sensitivity plot focuses on informative designs.

## Explicit structural prior
Learn every coefficient in:
zdot=[b0+b1*mu+b2*|z|²+i(c0+c1*mu+c2*|z|²)]z.
The critical point -b0/b1 is estimated, not supplied. Frequency and crossing orientation are checked.
This six-coefficient rotationally equivariant class is a subspace of the unconstrained cubic dictionary. It is used only for radial systems, never imposed on FHN.
At T=5,sigma=.01, median errors are random .00134, probe .005 .00726, probe .02 .00193, probe .10 .00252.
The declared prior improves some cells but not all: no universal benefit is claimed.

## NODE sensitivity

| Design | T | sigma | Correctly oriented unique crossings |
|---|---:|---:|---:|
| Random | 5 | 0 | 4/4 |
| Probe .02 | 5 | 0 | 0/4 |
| Probe .10 | 5 | 0 | 4/4 |
| Probe .02 | 20 | 0 | 0/4 |
| Random | 5 | .01 | 4/4 |
| Probe .02 | 5 | .01 | 2/4 |

The noiseless small-probe group has one crossing with reversed direction; it is not correct recovery.
Larger-probe median error is .08315 versus .03739 for random transients. Larger excitation improves this NODE's consistency in these controls.
The T=20 failures cannot be attributed solely to longer observation because other temporal factors change.

## One-sided extrapolation
Polynomial training uses only mu=[-.4,-.3,-.2,-.1].
Random: 5/5 candidates, median error 4.03e-5 without noise and .00617 at sigma=.01.
Probe: 5/5 candidates noiseless, median 6.55e-6; 4/5 noisy, conditional median .12258.
This is extrapolation with the true cubic dictionary, not prior-free or NODE extrapolation.

## Non-radial FHN learning
xdot=x-x³/3-y+I; ydot=.08(x+.7-.8y).
Train I=[.12,.22,.44,.54], validate [.18,.48], test [.30,.36].
Training: 32 trajectories × 128 full-state observations =8192 scalar samples, T=20.
Validation/test each cost 2048 scalar samples. Initial x∈[-2,2],y∈[-.5,1.5] are uniformly sampled; no equilibrium anchors initialize this dataset.
State and parameter normalization use training statistics only. Predictions/Jacobians are transformed back to physical coordinates consistently.
Five NODE seeds train for 3600 steps; the polynomial uses Savitzky–Golay estimated derivatives.

True lower-current Hopf point: .33128133745.
Polynomial candidate: .32959536268; error .00168597.
NODE errors: .04913,.00371,.06864,.08549,.06819.
All five FHN NODEs have positive crossing slopes and positive first Lyapunov coefficients: local subcritical type is numerically preserved while location shifts.
This validates non-radial learning and diagnostics, not the three-way sampling ranking on FHN.

## Hopf and periodic-orbit diagnostics
First Lyapunov coefficients use the standard complex-eigenvector expression with numerical second/third derivatives. Steps h=.002,.001 give consistent signs.
Known coefficient targets -2 (true Hopf) and .97197108199 (true FHN) pass assertions.
Five selected radial NODEs have negative coefficients and positive crossing slopes; five FHN NODEs have positive coefficients and slopes.
These checks numerically support local nondegeneracy; they are not interval-arithmetic certificates.

Shooting fixes y0=y_eq and x0=x_eq+radius, then solves for parameter and period satisfying flow(T,z0)=z0.
For a closed planar autonomous orbit, the nontrivial Floquet multiplier is exp(integral trace(J) dt). The tangent multiplier is one.
Selected stable radial cycles and unstable FHN cycles are recovered.
21 points pass independent tolerance refinement; maximum tighter closing error is 2.49e-10.
One FHN NODE shooting point at section radius .16 fails and remains recorded. It does not prove a fold or cycle disappearance.
These are local fixed-section continuation points, not a full pseudo-arclength/global bifurcation diagram.

## Scientific glow rendering
rendered_phase_flow.png/gif/mp4 use actual particle trajectories stored in render_particle_trajectories.npz.
Left: Hopf mu=.25. Right: FHN I=.36. Both parameters stay fixed; clocks differ and are labeled.
Dark background, orange/purple trail colors and Gaussian bloom are presentation effects. Brightness/density do not encode probability, uncertainty or energy.
The showcase omits axes and scales FHN x/y differently. Quantitative comparisons use conventional figures and arrays.
The reference image informs style only and is not edited.

## Reproduce
From repository root, with requirements-phase1.txt:
    OPENBLAS_NUM_THREADS=1 python phase4_robustness.py
    OPENBLAS_NUM_THREADS=1 python phase4_equivariant.py
    OPENBLAS_NUM_THREADS=1 python phase4_fhn.py
    OPENBLAS_NUM_THREADS=1 python phase4_cycles.py
    OPENBLAS_NUM_THREADS=1 python phase4_finalize.py
    OPENBLAS_NUM_THREADS=1 python render_dynamics.py
FHN training must finish before cycle diagnostics. Rendering needs ffmpeg with libx264.
Source data, weights, histories and numerical audits are included.

## Remaining work
Phase 4 empirical scope is complete. The bounded narrative is: orbit observations can miss transverse information; coverage and model prior interact; noise undermines tiny probes; local bifurcation type may be right while location remains biased.
Do not claim universal optimal sampling, complete NODE convergence, a noise threshold or complete global attractor recovery.
Phase 5: literature re-verification, paper/source/PDF, final reproducibility review, repository presentation and GitHub delivery.
