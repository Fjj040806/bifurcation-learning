"""Audits, optimization diagnostics and animations for completed Phase 3 runs."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.optimize import root
import learning_core as core
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

R=Path(__file__).parent;O=R/"phase3"
records=json.loads((O/"results.json").read_text())
audit=[];convergence=[]
for rec in records:
    if rec["model"]!="node":continue
    d=rec["design"];b=rec["budget"];s=rec["seed"];step=rec["training_steps"]
    net=core.Net(s);ar=np.load(O/f"node_{d}_{b}_seed{s}_step{step}.npz")
    net.p=[ar[f"p{i}"] for i in range(6)]
    data=np.load(O/f"data_{d}_{b}.npz");zz=data["states"];mus=data["parameters"]
    starts=[0,16,32,48]
    z0=np.concatenate([zz[:,j] for j in starts]);mm=np.tile(mus,4)
    target=np.concatenate([zz[:,j+1:j+7] for j in starts],axis=0).transpose(1,0,2)
    loss,grad=core.objective(net,z0,mm,target)
    convergence.append({"design":d,"budget":b,"seed":s,"steps":step,"balanced_training_segment_mse":float(loss),
                        "balanced_gradient_norm":float(np.sqrt(sum(np.sum(g*g) for g in grad)))})
    grid=np.linspace(-.4,.4,161);last=np.zeros(2);acc=[];minimum_det=np.inf;fail=0
    for m in grid:
        rr=root(lambda z:net.f(z[None],m)[0][0],last,jac=lambda z:net.jac(z,m),tol=1e-10)
        if not rr.success or np.linalg.norm(rr.fun)>1e-7:fail+=1;continue
        last=rr.x;j=net.jac(last,m);minimum_det=min(minimum_det,np.linalg.det(j))
        if abs(m)>.025:acc.append((max(np.linalg.eigvals(j).real)<0)==(m<0))
    rec["stability_accuracy"]=float(np.mean(acc)) if acc else None
    candidates=[]
    for mh in rec["crossings"]:
        rr=root(lambda z:net.f(z[None],mh)[0][0],np.zeros(2),tol=1e-10)
        j=net.jac(rr.x,mh);h=1e-5
        def g(m):
            e=root(lambda z:net.f(z[None],m)[0][0],rr.x,tol=1e-10)
            return np.trace(net.jac(e.x,m))/2
        candidates.append({"mu":mh,"determinant":float(np.linalg.det(j)),"crossing_slope":float((g(mh+h)-g(mh-h))/(2*h)),"residual":float(np.linalg.norm(rr.fun))})
    audit.append({"design":d,"budget":b,"seed":s,"steps":step,"minimum_branch_determinant":float(minimum_det) if np.isfinite(minimum_det) else None,"root_failures":fail,"candidates":candidates})
(O/"derivative_audit.json").write_text(json.dumps(audit,indent=2))
(O/"optimization_audit.json").write_text(json.dumps(convergence,indent=2))
(O/"results.json").write_text(json.dumps(records,indent=2))
keys=sorted(set().union(*(r.keys() for r in records)))
with (O/"results.csv").open("w") as f:
    writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
# Training convergence means optimization diagnostics, not proof of global convergence.
fig,axs=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
for row,b in enumerate([4096,16384]):
    for col,d in enumerate(["passive","random","probe"]):
        ax=axs[row,col];hist=[]
        for s in range(5):
            h=json.loads((O/f"history_{d}_{b}_seed{s}.json").read_text())
            x=[v["step"] for v in h];y=[v["validation_segment_mse"] for v in h];hist.append(y)
            ax.plot(x,y,color="#2563eb",alpha=.2)
        ax.plot(x,np.median(hist,axis=0),color="#2563eb",lw=2)
        ax.axvline(1200,color=".6",ls=":");ax.set(yscale="log",xlabel="Adam steps",ylabel="Validation segment MSE",title=f"{d}; budget {b}")
fig.suptitle("Optimization control: all seeds and median; no claim of full convergence")
fig.savefig(O/"optimization_curves.png",dpi=180);fig.savefig(O/"optimization_curves.svg");plt.close(fig)

# Training geometry animation, fixed positive mu=0.2.
fig,axs=plt.subplots(1,3,figsize=(10,3.6),constrained_layout=True);artists=[]
for ax,d in zip(axs,["passive","random","probe"]):
    data=np.load(O/f"data_{d}_4096.npz");zs=data["states"][data["parameters"]==.2]
    th=np.linspace(0,2*np.pi,200);r=np.sqrt(.2)
    ax.plot(r*np.cos(th),r*np.sin(th),":",color=".65")
    ax.set(xlim=(-.8,.8),ylim=(-.8,.8),aspect="equal",xlabel="x",ylabel="y",title=d)
    for z in zs:
        line=ax.plot([],[],lw=1,alpha=.65,color="#2563eb")[0];dot=ax.plot([],[],"o",ms=3,color="#dc6944")[0]
        artists.append((z,line,dot))
fig.suptitle("Acquisition geometry: fixed mu=0.2, same 64 observations per trajectory")
def update(k):
    for z,line,dot in artists:
        line.set_data(z[:k+1,0],z[:k+1,1]);dot.set_data([z[k,0]],[z[k,1]])
FuncAnimation(fig,update,frames=64,interval=90).save(O/"sampling_geometry.gif",writer=PillowWriter(fps=12));plt.close(fig)

# Paired effects: crossing failures remain separate, never imputed as zero.
pairs=[]
for b in [4096,16384]:
    for d in ["passive","probe"]:
        delta=[]
        for seed in range(5):
            a=next(r for r in records if r["model"]=="node" and r["budget"]==b and r["design"]==d and r["seed"]==seed and r["training_steps"]==3600)
            c=next(r for r in records if r["model"]=="node" and r["budget"]==b and r["design"]=="random" and r["seed"]==seed and r["training_steps"]==3600)
            if a["critical_parameter_error"] is not None and c["critical_parameter_error"] is not None:
                delta.append(a["critical_parameter_error"]-c["critical_parameter_error"])
        pairs.append({"budget":b,"contrast":d+" minus random","valid_pairs":len(delta),"paired_error_differences":delta,
                      "median_paired_difference":float(np.median(delta)) if delta else None})
(O/"paired_effects.json").write_text(json.dumps(pairs,indent=2))
print(json.dumps({"summaries":json.loads((O/"summary.json").read_text()),"ranks":json.loads((O/"dictionary_identifiability.json").read_text()),"paired_effects":pairs},indent=2))
