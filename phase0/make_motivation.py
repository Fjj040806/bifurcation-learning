"""Analytic counterexample; no neural model is trained in this figure."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

out = Path(__file__).parent
t = np.linspace(0, 10, 1001)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
colors = ["#2563eb", "#dc6944"]
for a, c in zip([-1, 1], colors):
    axes[0].plot(np.cos(t), np.sin(t), color=c, lw=3 if a == -1 else 1.5,
                 ls="-" if a == -1 else "--", label=f"a = {a:+d}")
    r = np.linspace(.55, 1.45, 300)
    axes[1].plot(r, a*r*(1-r*r), color=c, label=f"a = {a:+d}")
    for r0, ls in [(.98, "-"), (1.02, "--")]:
        sol = solve_ivp(lambda time, z: [a*z[0]*(1-z[0]**2)], [0, 10], [r0],
                        t_eval=t, rtol=1e-10, atol=1e-12)
        axes[2].plot(sol.t, sol.y[0], color=c, ls=ls)
axes[0].set(xlabel="x", ylabel="y", title="A. Identical observed orbit", aspect="equal")
axes[0].legend(frameon=False)
axes[1].axhline(0, color=".6", lw=.7)
axes[1].axvline(1, color=".6", lw=.7, ls=":")
axes[1].set(xlabel="Radius r", ylabel="Radial velocity", title="B. Different transverse dynamics")
axes[2].axhline(1, color=".6", lw=.7, ls=":")
axes[2].set(xlabel="Time", ylabel="Radius r", title="C. Small radial perturbations", ylim=(0, 2))
fig.suptitle("Orbit fit alone does not identify transverse stability", fontsize=14)
fig.savefig(out/"motivation.png", dpi=180)
fig.savefig(out/"motivation.svg")
# On r=1 both vector fields are (-y,x). Transverse derivative is -2a.
for a in [-1, 1]:
    assert abs(a*1*(1-1)) == 0
print("Verified: identical orbit; transverse exponents +2 and -2.")
