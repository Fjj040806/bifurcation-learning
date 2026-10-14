"""Local Hopf coefficient and fixed-section shooting diagnostics."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import learning_core as core
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R=Path(__file__).parent;O=R/"phase4"
class Polynomial:
    def __init__(self,c):self.c=c
    def f(self,z,m):return core.features(z,m)@self.c,None
    def jac(self,z,m):
        h=1e-5
        return np.column_stack([(self.f((z+h*np.eye(2)[k])[None],m)[0][0]-self.f((z-h*np.eye(2)[k])[None],m)[0][0])/(2*h) for k in range(2)])
class TruthHopf:
    def f(self,z,m):
        x,y=z.T;s=x*x+y*y;return np.column_stack([(m-s)*x-y,x+(m-s)*y]),None
    def jac(self,z,m):x,y=z;return np.array([[m-3*x*x-y*y,-1-2*x*y],[1-2*x*y,m-x*x-3*y*y]])
class TruthFHN:
    def f(self,z,m):
        x,y=z.T;return np.column_stack([x-x**3/3-y+m,.08*(x+.7-.8*y)]),None
    def jac(self,z,m):x,y=z;return np.array([[1-x*x,-1],[.08,-.064]])
class PhysicalNet:
    def __init__(self,n,meta):
        self.n=n;self.center=meta["center"];self.scale=meta["scale"];self.mean=float(meta["parameter_center"]);self.sd=float(meta["parameter_scale"])
    def f(self,z,m):return self.scale*self.n.f((z-self.center)/self.scale,(m-self.mean)/self.sd)[0],None
    def jac(self,z,m):return np.diag(self.scale)@self.n.jac((z-self.center)/self.scale,(m-self.mean)/self.sd)@np.diag(1/self.scale)
def loadnet(path,seed):
    n=core.Net(seed);a=np.load(path);n.p=[a[f"p{i}"] for i in range(6)];return n

def coefficient(model,m,guess,h):
    e=root(lambda z:model.f(z[None],m)[0][0],guess,tol=1e-11)
    if np.linalg.norm(e.fun)>1e-7:raise ValueError("equilibrium")
    z=e.x;A=model.jac(z,m).astype(complex);vals,Q=np.linalg.eig(A);idx=np.argmax(vals.imag);omega=vals[idx].imag;q=Q[:,idx]
    if omega<=1e-5 or abs(vals[idx].real)>1e-5:raise ValueError("Hopf candidate")
    ev,P=np.linalg.eig(A.T);p=P[:,np.argmin(abs(ev+1j*omega))];p=p/np.conj(np.vdot(p,q))
    HH=np.zeros((2,2,2));TT=np.zeros((2,2,2,2));basis=np.eye(2)
    for k in range(2):
        HH[:,:,k]=(model.jac(z+h*basis[k],m)-model.jac(z-h*basis[k],m))/(2*h)
        for j in range(2):
            if j==k:
                TT[:,:,j,k]=(model.jac(z+h*basis[k],m)-2*model.jac(z,m)+model.jac(z-h*basis[k],m))/h**2
            else:
                TT[:,:,j,k]=(model.jac(z+h*basis[j]+h*basis[k],m)-model.jac(z+h*basis[j]-h*basis[k],m)-
                               model.jac(z-h*basis[j]+h*basis[k],m)+model.jac(z-h*basis[j]-h*basis[k],m))/(4*h*h)
    def B(u,v):return np.einsum("ijk,j,k->i",HH,u,v)
    def C(u,v,w):return np.einsum("ijkl,j,k,l->i",TT,u,v,w)
    expr=C(q,q,np.conj(q))-2*B(q,np.linalg.solve(A,B(q,np.conj(q))))+B(np.conj(q),np.linalg.solve(2j*omega*np.eye(2)-A,B(q,q)))
    return float(np.real(np.vdot(p,expr))/(2*omega)),z,float(omega)

def branch(model,mh,guess,radii,bounds,frequency):
    previous=np.array([mh,2*np.pi/frequency]);points=[]
    for radius in radii:
        def evaluate(v,store=False):
            m,T=v
            if not bounds[0]<m<bounds[1] or not 1<T<100:return None
            e=root(lambda z:model.f(z[None],m)[0][0],guess,tol=1e-10)
            if np.linalg.norm(e.fun)>1e-7:return None
            z0=e.x+np.array([radius,0])
            def f(t,z):
                return [*model.f(z[:2][None],m)[0][0],np.trace(model.jac(z[:2],m))]
            def boundary(t,z):return 10-np.linalg.norm(z[:2])
            boundary.terminal=True
            sol=solve_ivp(f,(0,T),[*z0,0],method="DOP853",rtol=1e-9,atol=1e-11,events=boundary)
            if not sol.success or abs(sol.t[-1]-T)>1e-8:return None
            return sol.y[:2,-1]-z0,float(sol.y[2,-1]),z0
        def residual(v):
            a=evaluate(v);return a[0] if a is not None else np.array([100+abs(v[0]),100+abs(v[1])])
        solution=root(residual,previous,tol=1e-9)
        a=evaluate(solution.x)
        if a is None or np.linalg.norm(a[0])>1e-7:
            points.append({"section_radius":radius,"status":"shooting_failed"});continue
        m,T=solution.x;previous=solution.x
        points.append({"section_radius":radius,"status":"verified_periodic_point","parameter":float(m),"period":float(T),
                       "closing_residual":float(np.linalg.norm(a[0])),"transverse_multiplier":float(np.exp(a[1])),
                       "log_transverse_multiplier":a[1],"initial_state":a[2].tolist()})
    return points

xc=-np.sqrt(.936);IH=(xc+.7)/.8-xc+xc**3/3
models=[("truth_hopf",TruthHopf(),0,np.zeros(2),(-.4,.4),"hopf"),
        ("truth_fhn",TruthFHN(),IH,np.array([xc,(xc+.7)/.8]),(.1,.6),"fhn"),
        ("polynomial_fhn",Polynomial(np.load(O/"fhn_polynomial.npy")),json.loads((O/"fhn_results.json").read_text())[0]["crossings"][0],np.array([-1.,0.]),(.1,.6),"fhn")]
phase3=json.loads((R/"phase3/results.json").read_text())
for seed in range(5):
    rec=next(r for r in phase3 if r["model"]=="node" and r["design"]=="random" and r["budget"]==4096 and r["seed"]==seed and r["training_steps"]==3600)
    if len(rec["crossings"])==1:
        models.append((f"node_hopf_seed{seed}",loadnet(R/"phase3"/f"node_random_4096_seed{seed}_step3600.npz",seed),rec["crossings"][0],np.zeros(2),(-.4,.4),"hopf"))
meta=np.load(O/"fhn_dataset.npz")
for rec in json.loads((O/"fhn_results.json").read_text())[1:]:
    if len(rec["crossings"])==1:
        seed=rec["seed"];models.append((f"node_fhn_seed{seed}",PhysicalNet(loadnet(O/f"fhn_node_seed{seed}.npz",seed),meta),rec["crossings"][0],np.array([-1.,0.]),(.1,.6),"fhn"))
records=[]
for name,model,mh,guess,bounds,system in models:
    try:
        l1,e,w=coefficient(model,mh,guess,.002);l2,_,_=coefficient(model,mh,guess,.001)
        if name=="truth_hopf":assert abs(l1+2)<1e-5
        if name=="truth_fhn":assert abs(l1-.97197108199)<1e-5
        signconsistent=np.sign(l1)==np.sign(l2) and abs(l1)>1e-5
        rec={"model":name,"system":system,"candidate":mh,"lyapunov_h002":l1,"lyapunov_h001":l2,
             "relative_step_difference":abs(l1-l2)/max(abs(l1),1e-10),"sign_consistent":bool(signconsistent),"frequency":w}
        # Continue a selected set; all seeds receive the local coefficient audit.
        if name in ["truth_hopf","truth_fhn","polynomial_fhn","node_hopf_seed0","node_fhn_seed0"]:
            radii=[.06,.12,.2,.3,.4] if system=="hopf" else [.03,.06,.1,.16]
            rec["periodic_section_points"]=branch(model,mh,e,radii,bounds,w)
        records.append(rec);print(json.dumps(rec),flush=True)
    except (ValueError,np.linalg.LinAlgError) as ex:records.append({"model":name,"status":"coefficient_failed","reason":str(ex)})
(O/"hopf_and_cycle_audit.json").write_text(json.dumps(records,indent=2))
fig,axs=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
for ax,system in zip(axs,["hopf","fhn"]):
    for r in records:
        if r.get("system")!=system or "periodic_section_points" not in r:continue
        pts=[p for p in r["periodic_section_points"] if p["status"]=="verified_periodic_point"]
        for stable,marker in [(True,"o"),(False,"x")]:
            a=[p for p in pts if (p["transverse_multiplier"]<1)==stable]
            if a:ax.plot([p["parameter"] for p in a],[p["section_radius"] for p in a],marker+"-",label=r["model"]+(" stable" if stable else " unstable"),ms=5)
    ax.set(xlabel="mu" if system=="hopf" else "Applied current I",ylabel="x-section displacement from equilibrium",title=system+" periodic shooting points")
    ax.legend(fontsize=7)
fig.suptitle("Periodic orbits checked by closure and transverse Floquet multiplier")
fig.savefig(O/"periodic_branches.png",dpi=180);fig.savefig(O/"periodic_branches.svg")
print("Cycle diagnostics complete.",flush=True)
