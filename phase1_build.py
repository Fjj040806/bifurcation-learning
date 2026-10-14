"""Phase 1 reference dynamics, diagnostics, datasets and scientific animations."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT=Path(__file__).parent
OUT=ROOT/"phase1"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})

def radial(t,z,mu,a=1,b=0,c=0):
    x,y=z; s=x*x+y*y; g=mu-a*s-b*s*s; w=1+c*s
    return [g*x-w*y,w*x+g*y]

def fhn(t,z,I):
    x,y=z
    return [x-x**3/3-y+I,.08*(x+.7-.8*y)]

def equilibrium(I):
    x=brentq(lambda x:(x+.7)/.8-x+x**3/3-I,-4,4,xtol=1e-13)
    return np.array([x,(x+.7)/.8])

def jac(z):
    x,y=z
    return np.array([[1-x*x,-1],[.08,-.064]])

def integrate(fun,z,t,rtol=1e-9,atol=1e-11):
    sol=solve_ivp(fun,[t[0],t[-1]],z,t_eval=t,rtol=rtol,atol=atol,method="DOP853")
    if not sol.success or len(sol.t)!=len(t): raise RuntimeError(sol.message)
    return sol.y.T

xc=-np.sqrt(1-.08*.8)
Ic=(xc+.7)/.8-xc+xc**3/3
omega=np.sqrt(np.linalg.det(jac([xc,0])))
numeric=brentq(lambda I:np.trace(jac(equilibrium(I))),.2,.5,xtol=1e-13)
# First Lyapunov coefficient in the convention of the standard eigenvector formula.
A=jac([xc,0]).astype(complex)
ev,Q=np.linalg.eig(A); q=Q[:,np.argmax(ev.imag)]
ep,P=np.linalg.eig(A.T); p=P[:,np.argmin(abs(ep+1j*omega))]
p=p/np.conj(np.vdot(p,q))
def B(u,v): return np.array([-2*xc*u[0]*v[0],0],complex)
def C(u,v,w): return np.array([-2*u[0]*v[0]*w[0],0],complex)
l1=float(np.real(np.vdot(p,C(q,q,np.conj(q))-2*B(q,np.linalg.solve(A,B(q,np.conj(q))))+
                        B(np.conj(q),np.linalg.solve(2j*omega*np.eye(2)-A,B(q,q)))))/(2*omega))

checks={}
assert abs(numeric-Ic)<1e-10
assert abs(np.trace(A))<1e-12 and omega>0
checks["fhn_hopf_analytic"]=float(Ic)
checks["fhn_hopf_root"]=float(numeric)
checks["fhn_hopf_frequency"]=float(omega)
checks["fhn_first_lyapunov_coefficient"]=l1
checks["fhn_transversality_real_eigenvalue_slope"]=float(-xc/(1/.8-1+xc*xc))
assert l1>0 # Subcritical local Hopf; global cycle structure is not certified here.
t=np.linspace(0,40,801)
mu=.2; r0=.15
z=integrate(lambda t,z:radial(t,z,mu),[r0,0],t)
u=mu/(1+(mu/r0**2-1)*np.exp(-2*mu*t))
err=float(np.max(abs(np.sum(z*z,axis=1)-u)))
assert err<1e-7
checks["hopf_squared_radius_max_error"]=err
z2=integrate(lambda t,z:radial(t,z,mu),[r0,0],t,1e-11,1e-13)
checks["hopf_tolerance_refinement_max_state_difference"]=float(np.max(abs(z-z2)))
assert checks["hopf_tolerance_refinement_max_state_difference"]<1e-6
for a,b,c in [(1,0,0),(1,.5,.2)]:
    r=np.sqrt(2*mu/(a+np.sqrt(a*a+4*b*mu)))
    sol=integrate(lambda t,z:radial(t,z,mu,a,b,c),[r,0],t)
    assert np.max(abs(np.linalg.norm(sol,axis=1)-r))<1e-7
    phase=np.unwrap(np.arctan2(sol[:,1],sol[:,0]))
    assert np.max(abs(phase-(1+c*r*r)*t))<1e-6

# Validate the non-radial solver against tighter tolerances.
tf=np.linspace(0,120,1201)
zf=integrate(lambda t,z:fhn(t,z,.34),equilibrium(.34)+[.04,0],tf)
zf2=integrate(lambda t,z:fhn(t,z,.34),equilibrium(.34)+[.04,0],tf,1e-11,1e-13)
checks["fhn_tolerance_refinement_max_state_difference"]=float(np.max(abs(zf-zf2)))
assert checks["fhn_tolerance_refinement_max_state_difference"]<1e-5

# Reference branch plots. Only equilibria are continued for FHN at this stage.
fig,axs=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
m=np.linspace(-.4,.4,401)
for a,b,c,label in [(1,0,0,"Cubic Hopf"),(1,.5,.2,"Quintic family")]:
    pos=m[m>0]; rr=np.sqrt(2*pos/(a+np.sqrt(a*a+4*b*pos)))
    axs[0].plot(pos,rr,label=label)
axs[0].plot(m[m<0],np.zeros(sum(m<0)),color="black",label="Stable equilibrium")
axs[0].plot(m[m>0],np.zeros(sum(m>0)),"k--",label="Unstable equilibrium")
axs[0].set(xlabel="Parameter mu",ylabel="Cycle radius / equilibrium",title="Radial reference branches")
axs[0].legend(fontsize=8)
Is=np.linspace(.2,.5,401); eq=np.array([equilibrium(I) for I in Is])
growth=(1-eq[:,0]**2-.064)/2
stable=Is<Ic
axs[1].plot(Is[stable],eq[stable,0],color="#2563eb",label="Stable equilibrium")
axs[1].plot(Is[~stable],eq[~stable,0],"--",color="#dc6944",label="Unstable equilibrium")
axs[1].axvline(Ic,color=".5",ls=":")
axs[1].set(xlabel="Applied current I",ylabel="Equilibrium x",title="FitzHugh–Nagumo equilibrium branch")
axs[1].legend(fontsize=8)
axs[2].plot(Is,growth,color="#2563eb")
axs[2].axhline(0,color=".5",lw=1); axs[2].axvline(Ic,color=".5",ls=":")
axs[2].set(xlabel="Applied current I",ylabel="Real part of eigenvalues",title=f"Hopf crossing: I = {Ic:.6f}")
fig.savefig(OUT/"reference_branches.png",dpi=180)
fig.savefig(OUT/"reference_branches.svg")
plt.close(fig)

# Independent clips evolve at fixed mu; parameter is never ramped during a trajectory.
ta=np.linspace(0,30,301)
mus=[-.2,0,.2]
traces=[integrate(lambda t,z,mu=mu:radial(t,z,mu),[.8,0],ta) for mu in mus]
fig,axs=plt.subplots(1,3,figsize=(10,3.4),constrained_layout=True)
lines=[];dots=[]
for ax,mu in zip(axs,mus):
    ax.set(xlim=(-.9,.9),ylim=(-.9,.9),aspect="equal",xlabel="x",ylabel="y",title=f"Fixed mu = {mu:+.1f}")
    ax.plot(0,0,"ko",ms=3)
    if mu>0:
        th=np.linspace(0,2*np.pi,200); ax.plot(np.sqrt(mu)*np.cos(th),np.sqrt(mu)*np.sin(th),":",color=".5")
    lines.append(ax.plot([],[],color="#2563eb",lw=1.4)[0]);dots.append(ax.plot([],[],"o",color="#dc6944",ms=5)[0])
fig.suptitle("True Hopf dynamics: stable origin / critical decay / stable cycle")
def update(k):
    j=k*3
    for line,dot,z in zip(lines,dots,traces):
        line.set_data(z[:j+1,0],z[:j+1,1]);dot.set_data([z[j,0]],[z[j,1]])
    return lines+dots
FuncAnimation(fig,update,frames=101,interval=70).save(OUT/"hopf_dynamics.gif",writer=PillowWriter(fps=14))
plt.close(fig)

# Perturbation stability animation; analytically identical motion on observed circle.
ta=np.linspace(0,3,121)
fig,axs=plt.subplots(1,2,figsize=(7,3.6),constrained_layout=True)
pert=[]
for ax,a in zip(axs,[1,-1]):
    th=np.linspace(0,2*np.pi,250);ax.plot(np.cos(th),np.sin(th),color=".65",ls=":")
    ax.set(xlim=(-1.6,1.6),ylim=(-1.6,1.6),aspect="equal",xlabel="x",ylabel="y",title=f"a={a:+d}: "+("stable" if a==1 else "unstable"))
    for r0,col in [(.98,"#2563eb"),(1.02,"#dc6944")]:
        # Stop outward unstable trajectory before finite-time blow-up.
        sol=solve_ivp(lambda t,z:[-z[1]+a*z[0]*(1-np.dot(z,z)),z[0]+a*z[1]*(1-np.dot(z,z))],
                      [0,3],[r0,0],dense_output=True,rtol=1e-10,atol=1e-12,
                      events=lambda t,z:1.7-np.linalg.norm(z))
        last=sol.t_events[0][0] if len(sol.t_events[0]) else 3
        zz=sol.sol(np.minimum(ta,last)).T
        line=ax.plot([],[],color=col)[0];dot=ax.plot([],[],"o",color=col)[0]
        pert.append((zz,line,dot))
fig.suptitle("Same circular orbit; different response to 2% radial probes")
def perturb(k):
    for zz,line,dot in pert:
        line.set_data(zz[:k+1,0],zz[:k+1,1]);dot.set_data([zz[k,0]],[zz[k,1]])
FuncAnimation(fig,perturb,frames=121,interval=60).save(OUT/"transverse_probes.gif",writer=PillowWriter(fps=16))
plt.close(fig)

# Small demonstrator dataset; not the final learning benchmark.
rng=np.random.default_rng(42)
parameters={"train":[-.4,-.2,.2,.4],"validation":[-.3,.3],"test":[-.1,.1]}
rows=[];trajectory_id=0
for split,params in parameters.items():
    for mu in params:
        for trial in range(4):
            rad=rng.uniform(.05,.8); theta=rng.uniform(0,2*np.pi)
            times=np.linspace(0,5,64); zz=integrate(lambda t,z:radial(t,z,mu),[rad*np.cos(theta),rad*np.sin(theta)],times)
            rows.extend([[split,trajectory_id,mu,float(t),float(x),float(y)] for t,(x,y) in zip(times,zz)])
            trajectory_id+=1
import csv
with (OUT/"pilot_trajectories.csv").open("w") as f:
    w=csv.writer(f);w.writerow(["split","trajectory_id","mu","time","x","y"]);w.writerows(rows)
checks["pilot_scalar_measurements_by_split"]={s:2*sum(r[0]==s for r in rows) for s in parameters}
assert checks["pilot_scalar_measurements_by_split"]["train"]==2048
assert not (set(parameters["train"]) & set(parameters["test"]))
(OUT/"verification.json").write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
