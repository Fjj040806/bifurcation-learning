# Phase 3 — Observation design under matched budgets

## Research question
Can learned differential equations recover bifurcation locations, stability and attractors from finite observations, and which observation designs and declared priors improve this recovery?

This stage answers a controlled subquestion: conditional on known attractor anchors, how do passive orbit observations, random transients and small transverse probes compare?

## Protocol
System: cubic Hopf normal form, true critical parameter zero.
Training parameters: -0.4,-0.2,0.2,0.4; common validation parameters -0.3,0.3; test parameters -0.1,0.1.
This is a parameter-interpolation experiment with no observations in the critical training gap. It is not one-sided extrapolation.

Two training budgets: 4096 and 16384 scalar measurements. Every trajectory has 64 observed (x,y) states over T=5, so one costs 128 scalar measurements. There are 32 or 128 training trajectories, equally allocated over the four parameters. The larger budget adds trajectories, not new parameter values or longer windows.
Common validation and test sets each cost 1024 scalar measurements. Validation observations are used to select checkpoints, so they are part of the model-selection information and must not be hidden in a data-only interpretation.
All repeated zero observations in the passive negative-mu cases are counted as measurements; unique state-parameter counts are separately recorded.

Acquisition designs:
- Passive: radius zero for negative mu; exact limit-cycle radius sqrt(mu) for positive mu, with random phase.
- Random: initial radius uniform on [0.05,0.8], random phase.
- Probe: positive-mu anchor radius plus/minus 0.02 (absolute units, not 2%); negative-mu origin perturbed to radius 0.02; random phase.

These are idealized oracle-anchored designs. Acquisition uses true attractor anchors; learners receive only sampled states and parameter labels. Costs of settling, finding attractors and delivering physical perturbations are excluded. Thus this is a matched observation count conditional on anchor access, not an end-to-end acquisition-cost comparison.
Only noiseless full-state data is tested here.

## Models and optimization controls
Unconstrained NODE: same width-24 two-layer tanh architecture, 746 parameters and trajectory loss as Phase 2.
Paired seeds 0–4 use the same initial weights and optimizer random seed across designs at each budget.
Batch size 48, six-step RK4 segments, Adam learning rate 0.003, norm clipping 5.
Every run trains 3600 steps, with evaluation at 1200 and 3600; at each cap use the best validation checkpoint available within that cap.
Total: 30 NODE training runs, 60 saved/evaluated checkpoints. Six polynomial fits supplement them.
No test crossing is used for tuning or model selection.
Fixed-grid balanced training losses and gradient norms are audited for saved checkpoints; full validation histories are retained.
Matched step budgets and learning curves control optimization effort, but do not establish global convergence. Some validation curves plateau; others remain variable or improve. Do not attribute all differences solely to information geometry.

Polynomial comparator: the 13-term dictionary contains the true cubic model; ordinary least squares on Savitzky–Golay estimated derivatives, not true derivatives and not sparse SINDy. It supplies a strong structural prior absent from the NODE.

## Main results at 3600 steps

| Scalar budget | Design | NODE runs with a unique candidate crossing | Median absolute crossing error among those runs |
|---|---|---:|---:|
| 4096 | Passive | 1/5 | 0.263624 |
| 4096 | Random | 5/5 | 0.027710 |
| 4096 | Probe | 3/5 | 0.159920 |
| 16384 | Passive | 1/5 | 0.264990 |
| 16384 | Random | 5/5 | 0.045078 |
| 16384 | Probe | 2/5 | 0.269918 |

Failure rates must be read alongside conditional medians; a median based on one successful run is not a robust performance estimate.
Failure means no unique candidate found by the implemented branch/crossing diagnostic within [-0.4,0.4], not proof that a learned system has no bifurcation anywhere.
Candidate Hopf crossings are audited using equilibrium residuals, determinant and crossing slopes. Full learned periodic branches and first Lyapunov coefficients are not yet certified.
The reported stability accuracy in results.json is independently recomputed from the largest real eigenvalue; failed roots are recorded separately.

Polynomial crossing errors:
- Random: 2.39e-5 (4096), 1.38e-5 (16384).
- Probe: 1.29e-7 (4096), 1.23e-7 (16384).
- Passive: no reliable crossing recovered; the equilibrium solver flagged 138 and 126 grid roots respectively, and the candidate dictionary is rank deficient.
The passive polynomial numerical diagnostic failure is reported explicitly; it is not itself a proof of non-identifiability. The separate exact argument in MATHEMATICAL_ANALYSIS.md supplies that restricted-observation conclusion.

## What the experiment supports
1. Purely on-attractor observations leave relevant transverse information absent in this chosen system. Increasing the observation count fourfold does not remove the two polynomial-column dependencies: rank stays 11/13.
2. Both random and probe designs produce rank 13/13, but probes are less well conditioned: smallest singular value at budget 4096 is 0.0607 versus 0.5892 for random data, with the same unscaled dictionary.
3. Small probes work extremely well with the correct polynomial dictionary; they do not automatically produce accurate recovery with an unconstrained NODE.
4. In these settings, random transients give the most consistent NODE candidate-crossing recovery. More observations at unchanged compute do not monotonically improve candidate location.
5. Observation design interacts with model prior and optimization. Full numerical rank and low training loss are insufficient performance guarantees.

Balanced saved-model training losses at 3600 steps (median over five seeds, budget 4096):
Passive 4.44e-6; random 3.79e-6; probe 2.26e-6.
Thus the poor probe test behavior cannot be diagnosed simply from the relative size of training segment loss.
It still may involve optimizer bias, architecture and weak sensitivity; the experiment does not isolate these mechanisms causally.

## Uncertainty and limitations
Five paired initialization seeds, one fixed generated dataset per design/budget: the spread is optimization variability conditional on those datasets, not population uncertainty over data acquisition.
Raw seed points and diagnostic failures are displayed. Paired error differences are supplied only where both crossings exist; no failed run is silently imputed as zero.
No broad significance or superiority claim is made from five seeds.
Dictionary singular values depend on scaling; their comparison here uses the same unscaled features.
The oracle anchor assumption, noiseless cubic model, short windows and fixed parameter gap constrain generalization.

## Figures and animation
- training_coverage.png/svg: state-space coverage for equal counts.
- design_comparison.png/svg: every candidate error at both compute caps; failures labeled separately.
- optimization_curves.png/svg: each validation history and pointwise median.
- sampling_geometry.gif: actual training trajectories at fixed mu=0.2. It visualizes acquisition, not learned dynamics.
- dictionary_identifiability.json: ranks, singular values and unique observations.
- derivative_audit.json, optimization_audit.json: independent diagnostics.

## Reproduce
From repository root with requirements-phase1.txt:
    OPENBLAS_NUM_THREADS=1 python phase3_compare.py
    OPENBLAS_NUM_THREADS=1 python phase3_analyze.py
Runtime is several minutes on the current CPU. Source, data, 60 NODE checkpoints, polynomial coefficients and histories are included.

## Stage outcome and Phase 4
Phase 3 controlled observation comparison is complete.
The refined finding is a model–observation interaction, not a universal claim that small probes win.
Phase 4 priorities: multiple acquisition seeds, noise, perturbation amplitudes and observation horizons; optimization sensitivity; symmetry-informed comparators; non-radial FHN learning; learned periodic-branch/criticality validation and one-sided extrapolation.
Phase 5 remains paper writing, final repository packaging and showcase publication.
