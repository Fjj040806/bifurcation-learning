# Observation geometry and identifiability

## Fixed research question
Which finite trajectory observations support recovery of bifurcations, equilibrium stability and attractors, under explicit priors and matched measurement budgets?

## Exact ambiguity on the observed invariant sets
For z=x+iy, consider the family
    zdot = i z + c(mu)(mu-|z|²)z.
The true cubic Hopf model corresponds to c(mu)=1.
At mu>0, on the circle |z|²=mu, every member generates the identical orbit zdot=i z.
At mu<0, observing only z=0 supplies zdot=0 for every member.
Thus these exact passive observations do not identify c(mu), even with infinitely many samples on those same sets.
The origin's real eigenvalue part is c(mu)mu, and the cycle transverse exponent at positive mu is -2c(mu)mu.
Changing the sign of c reverses these stability properties without changing the observed data.
This statement concerns the restricted training observation set. It is not a claim that the full experiment, which also has transient validation observations, is observationally indistinguishable.

If c is smooth and nonzero around zero, this family retains a crossing at zero while potentially reversing direction/criticality. Therefore it is an exact stability ambiguity, not a demonstration of arbitrarily shifted Hopf location.
In the larger space of unrestricted smooth vector fields, perturbations vanishing on the finite observed invariant sets can alter unobserved local derivatives. Additional priors are needed for unique extrapolation.

## Dictionary rank
On the positive-mu circles:
    mu*x = x³ + x*y²,
    mu*y = x²*y + y³.
These relations also hold on the observed negative-mu origin.
Therefore the 13-column polynomial design matrix has at least two exact dependencies for ideal passive data, regardless of sample count.
The finite-precision rank and singular spectrum are recorded in dictionary_identifiability.json.
Random transients and transverse probes depart from these sets and can break these dependencies. Full numerical rank alone does not guarantee a well-conditioned identification problem.

## Local probe information
Around a positive-mu circle r*=sqrt(mu), write r=r*+delta.
For the true cubic system:
    deltadot = -2mu*delta + O(delta²).
Off-circle observations reveal this derivative; on-circle observations do not.
Near zero mu, the recovery time scale 1/(2mu) increases, so a fixed T=5 probe may yield little observable relaxation.
Small amplitude alone is not universally better: signals can be weak and poorly conditioned, especially with noise. Phase 4 will test this.

## Scope
These are elementary analytic consequences of the chosen equations. They motivate the experiments but are not claimed as previously unknown theorems.
The evidence sought is how this information geometry interacts with finite budgets, model training and bifurcation diagnostics.
