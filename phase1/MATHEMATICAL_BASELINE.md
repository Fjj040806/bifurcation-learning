# Phase 1 — Mathematical and numerical truth

## Fixed research question
Can learned differential equations recover bifurcation points, stability and attractors under limited initial conditions, time horizons and parameter coverage? Which matched-budget observation designs and declared priors help?

This phase constructs reference truth only. No learned model is trained.

## 1. Cubic Hopf system
For z=x+iy, zdot=(mu+i)z-|z|²z.
Writing z=r exp(i theta) yields rdot=mu r-r³ and thetadot=1.
The origin has eigenvalues mu±i. Its stability changes at mu=0.
For mu>0, the periodic solution has r*=sqrt(mu), period 2pi and transverse exponent -2mu.
For mu<0 the origin is asymptotically stable; at mu=0, rdot=-r³ still attracts but non-exponentially.

Set u=r². Then udot=2mu u-2u².
For mu!=0:
u(t)=mu/[1+(mu/u0-1)exp(-2mu t)].
At mu=0: u(t)=u0/(1+2u0 t).
These formulas are independent numerical verification targets.

## 2. Quintic amplitude / nonlinear frequency family
zdot=[mu-a|z|²-b|z|⁴+i(1+c|z|²)]z, a>0,b>=0.
rdot=r(mu-a r²-b r⁴).
Positive cycle squared radius:
s*=2mu/[a+sqrt(a²+4bmu)] for mu>0.
This numerically stable expression also applies at b=0.
The cycle transverse exponent is -2a s*-4b(s*)² < 0.
Its period is 2pi/(1+c s*), provided the angular frequency is positive.
Reference example a=1,b=0.5,c=0.2.
The Hopf point is again zero, so this is a validation of richer nonlinear behavior, not independent evidence of arbitrary-system generalization.

## 3. Non-radial FitzHugh–Nagumo reference
xdot=x-x³/3-y+I
ydot=epsilon(x+alpha-beta y)
with epsilon=0.08, alpha=0.7, beta=0.8.

At equilibrium:
y*=(x*+alpha)/beta
I=(x*+alpha)/beta-x*+(x*)³/3.
Derivative of this equilibrium parameterization is 1/beta-1+(x*)²>0.
Hence the equilibrium is unique and its branch can be followed without folds.

Jacobian:
J=[[1-(x*)²,-1],[epsilon,-epsilon beta]].
Trace=1-(x*)²-epsilon beta.
Determinant=epsilon[1-beta+beta(x*)²]>0.
Trace vanishes at x_H=±sqrt(1-epsilon beta).
We focus on the lower-current crossing x_H<0, with
I_H=(x_H+alpha)/beta-x_H+x_H³/3.
At this crossing frequency is sqrt(det J)>0.
The real-eigenvalue crossing derivative is
d Re(lambda)/dI=-x_H/[1/beta-1+x_H²]>0.
Thus the origin of the local coordinates loses stability as I increases.

We calculate the first Lyapunov coefficient using normalized complex left/right eigenvectors, the bilinear derivative B and trilinear derivative C:
l1=Re <p,C(q,q,qbar)-2B(q,J^-1 B(q,qbar))+B(qbar,(2iw Identity-J)^-1 B(q,q))>/(2w).
Here B(u,v)=(-2x_H u_x v_x,0), C(u,v,w)=(-2u_x v_x w_x,0), <p,q>=1.
Its positive sign in verification.json indicates a subcritical local Hopf for this parameter set. Its numerical magnitude depends on eigenvector normalization.
This matters: transient oscillation or the appearance of a large stable cycle cannot by itself locate this Hopf. Unstable periodic-branch continuation is not yet implemented; global cycle folds and full attractor structure remain future diagnostics.

## Numerical verification and reproducibility
Run: python phase1_build.py
Dependencies: Python, NumPy, SciPy, Matplotlib, Pillow.
Environment versions are recorded in environment.json.
DOP853 simulations are checked against exact squared-radius and phase formulas.
Tightened tolerances independently check state trajectories for cubic Hopf and FHN.
FHN trace root is compared to its analytic critical current.
Verification includes crossing frequency, transversality and first Lyapunov sign.
Results are written to verification.json. Assertions stop generation if critical checks fail.

pilot_trajectories.csv is a small demonstrator, not a frozen final benchmark.
Parameter groups: train [-0.4,-0.2,0.2,0.4], validation [-0.3,0.3], test [-0.1,0.1].
Each parameter has four random-initial-condition trajectories and 64 full-state observations.
Training cost is 2048 scalar measurements; validation/test each cost 1024 and are separately accounted.
All observations from a trajectory share its split; no parameter is shared across splits.
Only full-state trajectories and parameter labels are supplied; no derivative labels.
Matching passive/probe budgets and noisy variants belongs to Phase 3.

## Figures and animations
- reference_branches.png/svg: analytic radial cycle branches and FHN equilibrium/eigenvalue crossing. FHN periodic branches are not shown.
- hopf_dynamics.gif: three independently simulated fixed-parameter systems; no parameter ramp inside a trajectory.
- transverse_probes.gif: 2% radial perturbations around an identical observed orbit, comparing stable and unstable transverse dynamics.
The outward unstable trajectory freezes at the displayed radius boundary; this is display clipping, not a physical stopping force.
These are scientific truth simulations, not machine-learning results.

## Stage ledger
Phase 0: completed scoped positioning and provisional protocol.
Phase 1: completed analytic truth, verified reference solver, equilibrium crossing, initial figures/animations and dataset generator.
Phase 2 next: train polynomial and NODE baselines; implement learned equilibrium/eigenvalue diagnostics; compare trajectory error with critical-point recovery.
Periodic-branch continuation and final publication-grade validation are outstanding tasks within Phases 3–4.
