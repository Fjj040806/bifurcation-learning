"""Polynomial robustness grid and selected NODE controls, with acquisition replication."""
from pathlib import Path
import json,time
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import root,brentq
import learning_core as core
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R=Path(__file__).parent;O=R/"phase4";O.mkdir(exist_ok=True)
class Poly:
    def __init__(self,c):self.c=c
    def f(self,z,mu):return core.features(z,mu)@self.c,None
    def jac(self,z,mu):
        h=1e-5
        return np.column_stack([(self.f((z+h*np.eye(2)[k])[None],mu)[0][0]-self.f((z-h*np.eye(2)[k])[None],mu)[0][0])/(2*h) for k in range(2)])

def acquire(design,T,amplitude,noise,seed,one_sided=False):
    rng=np.random.default_rng(4000+seed);ts=np.linspace(0,T,64);data=[]
    params=[-.4,-.3,-.2,-.1] if one_sided else [-.4,-.2,.2,.4]
    for m in params:
        for j in range(8):
            theta=rng.uniform(0,2*np.pi);anchor=np.sqrt(max(m,0))
            r=anchor if design=="passive" else (rng.uniform(.05,.8) if design=="random" else
                (max(0,anchor+(1 if j%2==0 else -1)*amplitude) if m>0 else amplitude))
            z=core.truth([r*np.cos(theta),r*np.sin(theta)],m,ts)
            # Noise standard deviation is absolute state units, not a percentage.
            z=z+rng.normal(0,noise,z.shape)
            data.append(("train",m,z))
    return data

def crossing(model):
    grid=np.linspace(-.4,.4,81);last=np.zeros(2);vals=[];eqs=[];fails=0
    for m in grid:
        rr=root(lambda z:model.f(z[None],m)[0][0],last,jac=lambda z:model.jac(z,m),tol=1e-9)
        ok=np.isfinite(rr.fun).all() and np.linalg.norm(rr.fun)<1e-7
        if ok:last=rr.x
        else:fails+=1
        eqs.append(last.copy());vals.append(np.trace(model.jac(last,m)) if ok else np.nan)
    roots=[];slopes=[]
    for i in range(len(grid)-1):
        if np.isfinite(vals[i:i+2]).all() and vals[i]*vals[i+1]<0:
            def g(m):
                rr=root(lambda z:model.f(z[None],m)[0][0],eqs[i],tol=1e-9)
                if np.linalg.norm(rr.fun)>1e-7:raise ValueError("residual")
                return np.trace(model.jac(rr.x,m))
            try:
                mh=brentq(g,grid[i],grid[i+1]);rr=root(lambda z:model.f(z[None],mh)[0][0],eqs[i])
                if np.linalg.det(model.jac(rr.x,mh))>0:
                    roots.append(float(mh));slopes.append(float((g(mh+1e-5)-g(mh-1e-5))/4e-5))
            except (ValueError,RuntimeError):pass
    return {"crossings":roots,"crossing_error":abs(roots[0]) if len(roots)==1 else None,"root_failures":fails,"crossing_slopes":slopes}

polyrows=[]
for design in ["passive","random","probe"]:
    amps=[.005,.02,.1] if design=="probe" else [.02]
    for amp in amps:
        for T in [5,20]:
            for noise in [0,.01,.05]:
                for seed in range(5):
                    data=acquire(design,T,amp,noise,seed)
                    X=[];Y=[]
                    for _,m,z in data:
                        X.append(core.features(z[4:-4],m));Y.append(savgol_filter(z,9,3,deriv=1,delta=T/63,axis=0)[4:-4])
                    X=np.concatenate(X);Y=np.concatenate(Y)
                    c=np.linalg.lstsq(X,Y,rcond=1e-10)[0];met=crossing(Poly(c))
                    polyrows.append({"design":design,"amplitude":amp,"T":T,"noise_sigma":noise,"acquisition_seed":seed,"budget":4096,
                                     "dictionary_rank":int(np.linalg.matrix_rank(X,tol=1e-8)),**met})
                print("polynomial",design,amp,T,noise,flush=True)
(O/"polynomial_robustness.json").write_text(json.dumps(polyrows,indent=2))

# One-sided parameter coverage: separate polynomial extrapolation track.
extra=[]
for design in ["random","probe"]:
    for noise in [0,.01]:
        for seed in range(5):
            data=acquire(design,5,.02,noise,seed,True);X=[];Y=[]
            for _,m,z in data:
                X.append(core.features(z[4:-4],m));Y.append(savgol_filter(z,9,3,deriv=1,delta=5/63,axis=0)[4:-4])
            c=np.linalg.lstsq(np.concatenate(X),np.concatenate(Y),rcond=1e-10)[0]
            extra.append({"design":design,"noise_sigma":noise,"acquisition_seed":seed,"training_parameters":[-.4,-.3,-.2,-.1],**crossing(Poly(c))})
(O/"polynomial_extrapolation.json").write_text(json.dumps(extra,indent=2))

# Replicated NODE acquisition: two acquisition seeds × two initialization seeds.
# Only predeclared informative designs are used; Phase 3 remains the passive control.
controls=[("random",.02,5,0),("probe",.02,5,0),("probe",.1,5,0),("probe",.02,20,0),
          ("random",.02,5,.01),("probe",.02,5,.01)]
config={"polynomial_fits":len(polyrows),"extrapolation_fits":len(extra),"node_controls":controls,
        "acquisition_seeds":[0,1],"model_seeds":[0,1],"steps":3600,"scalar_training_budget":4096,
        "noise_definition":"absolute iid Gaussian observation noise; no process noise",
        "anchor_cost":"oracle anchor setup excluded, conditional design comparison",
        "longer_window":"same 64 points, hence coarser time sampling"}
(O/"robustness_config.json").write_text(json.dumps(config,indent=2))
node=[];ts=np.linspace(0,5,64)
val=[]
for m in [-.3,.3]:
    for j in range(4):
        theta=j*1.7;z=core.truth([.4*np.cos(theta),.4*np.sin(theta)],m,ts);val.append((m,z))
v0=np.array([z[0] for m,z in val]);vm=np.array([m for m,z in val]);vt=np.array([z[1:13] for m,z in val]).transpose(1,0,2)
for control_id,(design,amp,T,noise) in enumerate(controls):
    for acquisition_seed in [0,1]:
        data=acquire(design,T,amp,noise,acquisition_seed)
        for seed in [0,1]:
            net=core.Net(seed);rng=np.random.default_rng(100+seed)
            moment=[np.zeros_like(p) for p in net.p];var=[np.zeros_like(p) for p in net.p]
            best=np.inf;bestp=None;history=[]
            for step in range(1,3601):
                ids=rng.integers(0,32,48);starts=rng.integers(0,57,48)
                z0=np.array([data[i][2][s] for i,s in zip(ids,starts)]);mus=np.array([data[i][1] for i in ids])
                target=np.array([data[i][2][s+1:s+7] for i,s in zip(ids,starts)]).transpose(1,0,2)
                core.dt=T/63;loss,grad=core.objective(net,z0,mus,target)
                norm=np.sqrt(sum(np.sum(g*g) for g in grad));grad=[g*min(1,5/max(norm,1e-12)) for g in grad]
                for j,g in enumerate(grad):
                    moment[j]=.9*moment[j]+.1*g;var[j]=.999*var[j]+.001*g*g
                    net.p[j]-=.003*(moment[j]/(1-.9**step))/(np.sqrt(var[j]/(1-.999**step))+1e-8)
                if step%100==0:
                    core.dt=5/63;vl=core.objective(net,v0,vm,vt)[0]
                    history.append({"step":step,"train_batch_mse":float(loss),"validation_mse":float(vl)})
                    if vl<best:best=vl;bestp=[p.copy() for p in net.p];beststep=step
            net.p=bestp
            tag=f"control{control_id}_data{acquisition_seed}_seed{seed}"
            np.savez(O/(tag+".npz"),**{f"p{i}":p for i,p in enumerate(net.p)})
            (O/(tag+"_history.json")).write_text(json.dumps(history,indent=2))
            record={"control":control_id,"design":design,"amplitude":amp,"T":T,"noise_sigma":noise,"acquisition_seed":acquisition_seed,
                    "model_seed":seed,"validation_mse":float(best),"best_step":beststep,**crossing(net)}
            node.append(record);(O/"node_robustness.json").write_text(json.dumps(node,indent=2))
            print(json.dumps(record),flush=True)

fig,axs=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
for ax,T in zip(axs,[5,20]):
    for d,amp,label,col in [("random",.02,"Random","#2563eb"),("probe",.005,"Probe .005","#cc5566"),("probe",.02,"Probe .02","#20a078"),("probe",.1,"Probe .1","#a66a20")]:
        for j,noise in enumerate([0,.01,.05]):
            runs=[r for r in polyrows if r["design"]==d and r["amplitude"]==amp and r["T"]==T and r["noise_sigma"]==noise]
            errors=[r["crossing_error"] for r in runs if r["crossing_error"] is not None]
            if errors:ax.scatter(np.full(len(errors),j),errors,color=col,alpha=.6,s=16)
            ax.plot(j,np.median(errors) if errors else np.nan,"s",color=col,label=label if j==0 else None)
    ax.set(yscale="log",xticks=range(3),xticklabels=["0",".01",".05"],xlabel="Observation noise sigma",ylabel="Candidate crossing error",title=f"Polynomial; window T={T}")
    ax.legend(fontsize=7)
fig.suptitle("Robustness: all acquisition seeds; failure counts retained in source tables")
fig.savefig(O/"polynomial_sensitivity.png",dpi=180);fig.savefig(O/"polynomial_sensitivity.svg")
print("Robustness track complete.",flush=True)
