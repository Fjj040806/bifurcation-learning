# Phase 2 — Minimal learned-dynamics experiment

## Research question
Can limited trajectory observations support recovery of bifurcation locations, stability and attractors? Which observation designs and priors help at matched measurement budgets?

This phase establishes a functioning learning/diagnostic loop. It does not test the final observation-design hypothesis.

## Data and training
Use the Phase 1 cubic Hopf dataset: 16 training trajectories at mu=-0.4,-0.2,0.2,0.4, each 64 full-state samples over T=5. Total training budget 2048 scalar observations. Separate validation parameters -0.3,0.3 and test parameters -0.1,0.1. The withheld critical band is an interpolation experiment, not one-sided extrapolation.

Polynomial baseline: 13 candidate terms including parameter-state interactions and cubic monomials; ordinary least squares, not sparsity-promoting SINDy. Derivatives estimated by a 9-point degree-3 Savitzky–Golay filter, using interior samples only. No true derivative labels.

Neural ODE: input (x,y,mu), two width-24 tanh hidden layers, two output components, 746 learned parameters. Train on six-step trajectory segments using RK4 integration and an explicit reverse-mode gradient implemented in NumPy. The neural network defines a continuous vector field, evaluated afterward with independent adaptive DOP853 integration.

Five random seeds, 1200 Adam steps each, batch size 48, learning rate 0.003, gradient clipping norm 5. Choose the checkpoint by 12-step validation trajectory MSE every 100 steps. No test labels are used for checkpoint selection.
NumPy implementation avoids a dependency on unavailable PyTorch. Gradient checks cover one entry in each parameter tensor; learned analytic Jacobians are independently checked with centered differences.

## Evaluation
At eight held-out trajectories, evaluate mean state-coordinate squared error on t=0..5 and t=0..30, using a 0.1 evaluation interval and truth generated at tighter solver tolerance.
Follow learned equilibria over mu in [-0.4,0.4]; find trace-zero crossings with positive determinant. Audit crossing transversality, frequency and equilibrium residual.
Reported critical-parameter error means the error of a candidate Hopf crossing relative to true mu=0. A nonzero first Lyapunov coefficient and periodic-branch continuation of each learned model have not yet been certified. Do not call these complete Hopf reconstructions.
Stability accuracy uses 151 grid points outside |mu|<=0.025. There are no equilibrium-root or rollout failures in this pilot. The determinant is checked positive on an independent audit grid; this is not a global proof.
The continued branch follows one root; this does not exclude additional learned equilibria elsewhere.

## Results
See results.csv/json for all seeds and audit.json for numerical checks.
Polynomial short MSE: 5.06e-10; candidate crossing error: 1.96e-5.
NODE short MSE ranges 2.20e-4..1.19e-3; crossing errors range 0.02598..0.17355.
These absolute state-coordinate MSEs are not normalized accuracy percentages.
The baseline dictionary contains the exact cubic structure, explaining its favorable result on this clean synthetic example. It is not evidence that polynomial models universally outperform NODEs.
The NODE is trained on short segments with a fixed small training budget; incomplete optimization, model capacity and coverage may all contribute to error. This pilot does not isolate their causal effects.
For NODE seed 3 versus seed 4, the latter has lower trajectory MSE but larger crossing error: the two criteria need not rank models identically in this pilot.
Five seeds are pilot variability measurements, not enough for broad statistical conclusions.

## Outputs
learning_diagnostics.png/svg: learned Jacobian crossings and trajectory-MSE versus crossing-error comparison.
learned_dynamics.gif: truth, polynomial and median-validation NODE seed 2 at held-out mu=0.1.
Saved NODE weights, polynomial coefficients, training histories, results tables and derivative audit.
The animation uses a predeclared initial state (0.65,0); it is an illustration at a held-out parameter, separate from the eight-trajectory aggregate metric.

## Reproduce
From repository root:
    python phase1_build.py
    OPENBLAS_NUM_THREADS=1 python phase2_train.py
    python phase2_audit.py
Use requirements-phase1.txt. The model files are NumPy archives.

## Next stage
Phase 3 compares passive orbit sampling, randomized transients and small transverse probes at matched budgets, with parameter splits, training-compute controls and optimization-convergence checks. Structural-prior comparators remain separate.
S2/S3 learning and periodic-branch validation are pending. No paper-level novelty or publication readiness is claimed.
