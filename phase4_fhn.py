"""Non-radial validation: polynomial and normalized trajectory-trained NODE."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root,brentq
from scipy.signal import savgol_filter
import learning_core as core
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R=Path(__file__).parent;O=R/"phase4";O.mkdir(exist_ok=True)
epsilon=.08;beta=.8;alpha=.7
xc=-np.sqrt(1-epsilon*beta);IH=(xc+alpha)/beta-xc+xc**3/3
def fhn(t,z,I):x,y=z;return [x-x**3/3-y+I,epsilon*(x+alpha-beta*y)]
def eq(I):
    x=brentq(lambda x:(x+alpha)/beta-x+x**3/3-I,-4,4)
    return np.array([x,(x+alpha)/beta])
def integrate(z0,I,t):
    s=solve_ivp(lambda t,z:fhn(t,z,I),(t[0],t[-1]),z0,t_eval=t,method="DOP853",rtol=1e-10,atol=1e-12)
    assert s.success;return s.y.T
params={"train":[.12,.22,.44,.54],"validation":[.18,.48],"test":[.30,.36]}
time=np.linspace(0,20,128);rng=np.random.default_rng(440)
data=[]
for split,Is in params.items():
    for I in Is:
        for j in range(8 if split=="train" else 4):
            # No oracle equilibrium initialization: random physical states.
            z=integrate([rng.uniform(-2,2),rng.uniform(-.5,1.5)],I,time);data.append((split,I,z))
train=[r for r in data if r[0]=="train"];val=[r for r in data if r[0]=="validation"]
states=np.concatenate([z for _,m,z in train]);center=states.mean(0);scale=states.std(0)
meanI=np.mean(params["train"]);scaleI=np.std(params["train"])
np.savez_compressed(O/"fhn_dataset.npz",states=np.array([z for s,I,z in data]),parameters=np.array([I for s,I,z in data]),splits=np.array([s for s,I,z in data]),time=time,center=center,scale=scale,parameter_center=meanI,parameter_scale=scaleI)
class PhysicalNet:
    def __init__(self,net):self.net=net
    def f(self,z,I):return scale*self.net.f((z-center)/scale,(I-meanI)/scaleI)[0],None
    def jac(self,z,I):return np.diag(scale)@self.net.jac((z-center)/scale,(I-meanI)/scaleI)@np.diag(1/scale)
class Poly:
    def __init__(self,c):self.c=c
    def f(self,z,I):return core.features(z,I)@self.c,None
    def jac(self,z,I):
        h=1e-5
        return np.column_stack([(self.f((z+h*np.eye(2)[j])[None],I)[0][0]-self.f((z-h*np.eye(2)[j])[None],I)[0][0])/(2*h) for j in range(2)])
def diagnostics(model):
    grid=np.linspace(.1,.6,151);last=np.array([-1.,0.]);traces=[];eqs=[];fail=0
    for I in grid:
        r=root(lambda z:model.f(z[None],I)[0][0],last,jac=lambda z:model.jac(z,I),tol=1e-9)
        ok=np.isfinite(r.fun).all() and np.linalg.norm(r.fun)<1e-7
        if ok:last=r.x
        else:fail+=1
        eqs.append(last.copy());traces.append(np.trace(model.jac(last,I)) if ok else np.nan)
    roots=[]
    for j in range(len(grid)-1):
        if np.isfinite(traces[j:j+2]).all() and traces[j]*traces[j+1]<0:
            def g(I):
                r=root(lambda z:model.f(z[None],I)[0][0],eqs[j])
                if np.linalg.norm(r.fun)>1e-7:raise ValueError("residual")
                return np.trace(model.jac(r.x,I))
            try:
                h=brentq(g,grid[j],grid[j+1]);e=root(lambda z:model.f(z[None],h)[0][0],eqs[j])
                if np.linalg.det(model.jac(e.x,h))>0:roots.append(float(h))
            except (ValueError,RuntimeError):pass
    errors=[];rollfail=0
    for split,I,z in data:
        if split!="test":continue
        s=solve_ivp(lambda t,z:model.f(z[None],I)[0][0],(0,20),z[0],t_eval=time,method="DOP853",rtol=1e-8,atol=1e-10)
        if not s.success or len(s.t)!=len(time):rollfail+=1
        else:errors.append(float(np.mean((s.y.T-z)**2)))
    return {"crossings":roots,"critical_error":abs(roots[0]-IH) if len(roots)==1 else None,
            "root_failures":fail,"test_rollout_failures":rollfail,"test_mse":float(np.mean(errors)) if errors else None},(grid,np.array(traces)/2)

X=[];Y=[]
for _,I,z in train:
    X.append(core.features(z[4:-4],I));Y.append(savgol_filter(z,9,3,deriv=1,delta=20/127,axis=0)[4:-4])
c=np.linalg.lstsq(np.concatenate(X),np.concatenate(Y),rcond=1e-10)[0];np.save(O/"fhn_polynomial.npy",c)
poly=Poly(c);metrics,curve=diagnostics(poly);records=[{"model":"polynomial",**metrics}];curves=[("Polynomial",curve)]
print(json.dumps(records[-1]),flush=True)
normal=[(I,(z-center)/scale) for _,I,z in train]
v0=np.array([(z[0]-center)/scale for _,I,z in val]);vm=np.array([(I-meanI)/scaleI for _,I,z in val]);vt=np.array([(z[1:13]-center)/scale for _,I,z in val]).transpose(1,0,2)
core.dt=20/127
for seed in range(5):
    n=core.Net(seed);rng=np.random.default_rng(100+seed);mom=[np.zeros_like(p) for p in n.p];var=[np.zeros_like(p) for p in n.p]
    best=np.inf;bestp=None;history=[]
    for step in range(1,3601):
        ids=rng.integers(0,len(normal),48);starts=rng.integers(0,121,48)
        z0=np.array([normal[i][1][s] for i,s in zip(ids,starts)]);mus=np.array([(normal[i][0]-meanI)/scaleI for i in ids])
        target=np.array([normal[i][1][s+1:s+7] for i,s in zip(ids,starts)]).transpose(1,0,2)
        loss,grad=core.objective(n,z0,mus,target);norm=np.sqrt(sum(np.sum(g*g) for g in grad));grad=[g*min(1,5/max(norm,1e-12)) for g in grad]
        for j,g in enumerate(grad):
            mom[j]=.9*mom[j]+.1*g;var[j]=.999*var[j]+.001*g*g
            n.p[j]-=.003*(mom[j]/(1-.9**step))/(np.sqrt(var[j]/(1-.999**step))+1e-8)
        if step%100==0:
            vl=core.objective(n,v0,vm,vt)[0];history.append([step,float(loss),float(vl)])
            if vl<best:best=vl;bestp=[p.copy() for p in n.p];bs=step
    n.p=bestp;np.savez(O/f"fhn_node_seed{seed}.npz",**{f"p{i}":p for i,p in enumerate(n.p)})
    (O/f"fhn_history_seed{seed}.json").write_text(json.dumps(history))
    met,curve=diagnostics(PhysicalNet(n));rec={"model":"node","seed":seed,"validation_normalized_mse":float(best),"best_step":bs,**met}
    records.append(rec);curves.append((f"NODE {seed}",curve));(O/"fhn_results.json").write_text(json.dumps(records,indent=2))
    print(json.dumps(rec),flush=True)
fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
truthx=np.array([eq(I)[0] for I in curves[0][1][0]])
ax.plot(curves[0][1][0],(1-truthx**2-.064)/2,"k--",label="Truth")
for name,(grid,g) in curves:ax.plot(grid,g,label=name,alpha=.7)
ax.axhline(0,color=".6");ax.axvline(IH,color=".6",ls=":")
ax.set(xlabel="Applied current I",ylabel="Half trace of equilibrium Jacobian",title="Non-radial FHN: learned candidate Hopf crossing")
ax.legend(fontsize=8);fig.savefig(O/"fhn_crossings.png",dpi=180);fig.savefig(O/"fhn_crossings.svg")
print("FHN learning complete.",flush=True)
