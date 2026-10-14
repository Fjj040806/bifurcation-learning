# Phase 0 — Research protocol v0.1

Date: 2026-10-14. Status: scoped literature review and provisional protocol complete; no learned-model experiments yet.

## Research question — preserve throughout the project

When observations cover limited initial conditions, time horizons and parameter ranges, can learned differential equations recover bifurcation locations, stability and attractors? Which observation designs and declared structural priors improve recovery at a matched measurement budget?

Provisional title: Beyond Orbit Fit: Observability and Data Design for Learning Hopf Bifurcations.

This refines the original question without replacing it. The central outcome is dynamical correctness, not a leaderboard of trajectory errors.

## Literature positioning

This is a targeted review, not an exhaustive systematic review. Selected full-text sections of the 2025 closest comparator were inspected; other entries below are based on primary-source abstracts/searchable descriptions. Detailed implementation review remains required before claiming methodological novelty.

| Work | Established contribution | Consequence for this project |
|---|---|---|
| Brunton, Proctor & Kutz, SINDy (2016), https://arxiv.org/abs/1509.03580 | Sparse equation discovery includes parameterized normal forms. | Include a parameter-aware polynomial/SINDy baseline; do not claim parameterized discovery is new. |
| Nieters et al., A trajectory-based loss function to learn missing terms in bifurcating dynamical systems (2021), https://www.nature.com/articles/s41598-021-99609-x | Studies loss design for missing-term learning across several bifurcation types. | Dynamics-aware losses are established; compare relevant losses if feasible. |
| Beregi et al., Using scientific machine learning for experimental bifurcation analysis of dynamic systems (2023), https://arxiv.org/abs/2110.11854 | Connects hybrid modeling and experimental bifurcation analysis. | Learning a bifurcation diagram alone is not new. |
| Lee, Barton & Renson, Modelling of physical systems with a Hopf bifurcation using mechanistic models and machine learning (2023), https://arxiv.org/abs/2209.06910 | Combines a normal-form mechanistic model with learned components for limit-cycle prediction. | A Hopf-constrained hybrid is a strong-prior comparator, not automatically a novel method. |
| van Tegelen et al., Neural Ordinary Differential Equations for Learning and Extrapolating System Dynamics Across Bifurcations (2025), https://arxiv.org/abs/2507.19036 | Parameter-dependent NODEs recover bifurcations; studies extrapolation, limited/noisy observations and information content. Code: https://github.com/evantegelen/neural-bifurcations | Closest overlap. Reconstructing bifurcations and saying coverage matters are insufficient novelty claims. |
| An et al., Residual-Based Multi-peak Sampling Algorithm in Inverse Problems of Dynamical Systems (2023 preprint), https://arxiv.org/abs/2304.10842 | Adaptive residual sampling for stochastic inverse dynamics with bifurcation evaluation. | Adaptive sampling broadly is established; compare uniform/random allocation, and distinguish deterministic transverse probes. |
| Matzakos & Sfyrakis, Comparing Physics-Informed and Neural ODE Approaches…Morris–Lecar (2026 preprint), https://arxiv.org/abs/2603.26921 | Compares learning frameworks across neuronal bifurcation regimes. | Generic physics-informed versus NODE comparison is not enough. |

## Provisional contribution and novelty gate

Candidate contribution: quantify how transverse excitation and parameter coverage jointly affect dynamical recovery, using analytically controlled ambiguity examples, matched measurement budgets, independent continuation diagnostics and a non-radial validation system.

No priority claim is made. Before paper submission, check adjacent literature on persistent excitation, structural/practical identifiability, experimental design, limit-cycle system identification and derivative-aware learning. If this exact contribution already exists, reproduce it explicitly and seek a defensible extension.

## Analytic motivation: identical observations, opposite stability

Let s=x²+y² and define a smooth planar vector field

    dx/dt = -y + a x (1-s)
    dy/dt =  x + a y (1-s).

In polar coordinates, dr/dt=a r(1-r²), dtheta/dt=1.
On r=1, the observed motion is x=cos(t+theta0), y=sin(t+theta0), independently of a.
Yet the transverse exponent is d[a r(1-r²)]/dr at r=1 = -2a.
Thus a=+1 has a stable observed cycle and a=-1 an unstable observed cycle despite identical noiseless orbit data for arbitrarily long observation.

This is our elementary illustrative construction, not a claim of a new theorem. It shows stability non-identifiability from this restricted observation set; it does not itself demonstrate a shifted Hopf point.
Small off-orbit perturbations distinguish the two systems.

## Systems

S1: diagnostic Hopf normal form: zdot=(mu-mu_c+i omega)z-|z|²z. Ground truth mu_c=0, omega=1; learner is not supplied these values unless explicitly marked.
S2: quintic amplitude/frequency family: zdot=[mu-mu_c-a|z|²-b|z|⁴+i(omega+c|z|²)]z. Choose a>0,b>=0 initially; later vary criticality only as a declared extension.
S3: non-radial planar oscillator with a numerically validated Hopf point (candidate FitzHugh–Nagumo). Freeze equations and parameters in Phase 1 after verifying bifurcation. This is required for claims beyond radial normal-form families.
Duffing/chaos/PDEs are deferred to avoid diluting the first paper.

## Information available to each learner

Primary track: full-state sampled trajectories and known constant parameter per trajectory; no true derivatives, Jacobians, bifurcation locations or test labels.
Simple models estimate derivatives or use integral/trajectory fitting; report preprocessing.
Hybrid track may know equilibrium or linear terms: evaluate separately because such priors can reveal the critical point.
Symmetry-constrained track knows rotational equivariance for S1/S2 only; never impose this on S3.
Truth derivatives, Floquet exponents and continuation branches are evaluation-only.
Do not use true Hopf labels/Jacobians in training then claim data-only discovery.

## Pilot protocol to freeze after numerical verification

State scaling fixed using training statistics only. Parameters for S1: domain [-0.4,0.4].
Parameter split A: interpolation with a withheld band around zero.
Parameter split B: one-sided training on negative mu, testing across zero (extrapolation).
Split C: training on both sides with no withheld band, unseen intermediate parameters.
Report performance separately for each split; successful extrapolation is not guaranteed by smoothness alone.
Train/validation/test split by entire trajectory and parameter, never adjacent time points.

Measurement budgets: 4096 and 16384 scalar measurements as initial pilot choices (each full state costs two).
Main matched-budget designs: passive near-attractor observations, randomly initialized transients, small perturbation probes and mixed observations.
Count probe reference observations, repeated trials and pilot acquisition used to locate equilibria/attractors. Match unique measurements; separately report integrated simulation time, number of trajectories and model compute.
An oracle near-true-critical sampling design is allowed only as an upper-bound comparator. A deployable adaptive design uses training-derived uncertainty and spends from the same budget.

Initial observation windows: T=5,20, with fixed sampling interval where budgets permit; acquisition details may change in pilot and must be logged.
Noise track: observational Gaussian noise only, sigma=0,0.01,0.05 relative to a declared training-derived scale; process noise is a separate future extension.
Five training seeds in pilot; ten in final main comparisons if compute permits.
Matched validation access and tuning budgets; do not tune on the held-out bifurcation.

## Models and diagnostics

Models: parameterized polynomial/SINDy; smooth tanh NODE; partial-knowledge hybrid; rotationally equivariant model for radial systems only.
Smooth activations support local derivative analysis. Detailed architectures selected in pilot.

Primary metric: absolute critical-parameter error, plus explicit no-crossing/multiple-crossing/root-failure rate. Never silently discard diagnostic failures.
Secondary: stability classification outside a declared critical tolerance band; equilibrium error; period/amplitude error; cycle existence and transverse/Floquet stability.
Trajectory MSE is reported separately against these dynamical metrics.
Spurious equilibria count is a bounded-domain multi-start search result, not proof of global absence.
Diagnostics: follow equilibria, evaluate eigenvalue crossings; verify a Hopf candidate has nonzero frequency and transversality. Classification as super/subcritical requires nonlinear analysis or validated periodic branches.
Use continuation and long-time trajectories as independent checks. Near criticality, fixed-time simulations can confuse slow transients with attractors: increase horizon and record unresolved cases.
Repeat truth calculations with tighter tolerances/alternative step sizes to separate solver error from learning error.
Report paired seed-wise effects and uncertainty; no claim that different seeds represent different real physical systems.

## Mathematical analysis goals

Completed here: restricted-orbit non-identifiability example.
Next: sufficient conditions for local stability recovery away from criticality, accounting for equilibrium displacement and eigenvalue conditioning.
Conditional root bound: if scalar crossing function g has |g'|>=m>0 in a neighborhood and learned crossing function differs uniformly by <=epsilon, a learned root in that neighborhood obeys |mu_hat-mu_c|<=epsilon/m. Root existence and uniqueness need additional conditions; this does not certify a full Hopf bifurcation.
Hopf-type robustness additionally needs smoothness, transversality and nondegenerate nonlinear coefficients; C0 trajectory fit is inadequate.
These are planned analyses, not established results of our trained models.

## Visualization register

P0-F1 (created): identical orbit / radial vector field / perturbed radii. Python + SciPy + Matplotlib; analytic illustration only.
P1-F1: parameter-colored phase portraits and equilibrium/periodic branches, stable/unstable distinguished.
P1-A1: parameter sweep animation at fixed mu per clip. Do not ramp mu within a trajectory and label it an autonomous bifurcation diagram.
P2-F1: training coverage over state–parameter space.
P3-F1: trajectory error versus critical-point error, all seeds and failures shown.
P3-F2: budget × observation design effects; confidence intervals.
P4-F1: truth/learned continuation branches and transverse stability.
P4-A1: matched perturbation experiments, truth and learned models side by side.
P5: README GIFs + interactive parameter selection; scientific figures export SVG/PDF/PNG.
Every figure states its data source, settings and whether schematic, truth simulation or learned result. No generated art for quantitative evidence.

## Deliverable and stage ledger

Final: original English paper with source/PDF, complete runnable GitHub project, dependency lock/configs, diagnostics/tests, figures/animations and evidence tables.
Phase 0: completed scoped review, refined question, provisional contribution, analytic motivation and protocol.
Phase 1: pending equations, analytical truth, validated solvers and continuation.
Phase 2: pending minimal training/evaluation loop.
Phase 3: pending controlled data-design comparison.
Phase 4: pending robustness and non-radial extension.
Phase 5: pending paper and showcase.
Progress is milestone-based: 1 of 6 phases complete, not an estimate that one sixth of total labor is complete.

## Phase 1 acceptance criteria

1. Verified S1/S2 equilibrium and cycle formulas with unit checks.
2. S3 equation/parameter set and reference Hopf point independently verified.
3. Data generator enforces whole-trajectory splits and budget accounting.
4. At least one phase portrait, reference bifurcation plot and parameter sweep animation.
5. No neural-model claim until actual training and independent diagnostics are complete.
