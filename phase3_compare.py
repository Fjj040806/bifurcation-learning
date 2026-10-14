"""Matched-observation design comparison, with paired seeds and compute checkpoints."""
from pathlib import Path
import json,time,csv
import numpy as np
from scipy.signal import savgol_filter
import learning_core as core
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R=Path(__file__).parent;O=R/"phase3";O.mkdir(exist_ok=True)
CONFIG={"budgets":[4096,16384],"designs":["passive","random","probe"],"seeds":list(range(5)),
        "train_parameters":[-.4,-.2,.2,.4],"checkpoints":[1200,3600],
        "batch_size":48,"steps_per_segment":6,"dt":core.dt,"duration":5,"points_per_trajectory":64,
        "learning_rate":.003,"validation_parameters":[-.3,.3],"test_parameters":[-.1,.1],
        "anchor_information":"Oracle attractor anchors used by the acquisition procedure; never provided to learners. Setup cost excluded; this is a conditional design experiment."}
(O/"config.json").write_text(json.dumps(CONFIG,indent=2))
ts=np.linspace(0,5,64)
def dataset(design,budget):
    rng=np.random.default_rng(530);data=[]
    n=budget//128//4
    for mu in CONFIG["train_parameters"]:
        for i in range(n):
            theta=rng.uniform(0,2*np.pi)
            anchor=np.sqrt(max(mu,0))
            if design=="passive":r=anchor
            elif design=="random":r=rng.uniform(.05,.8)
            else:r=max(0,anchor+(1 if i%2==0 else -1)*.02) if mu>0 else .02
            z=core.truth([r*np.cos(theta),r*np.sin(theta)],mu,ts)
            data.append(("train",mu,z))
    assert len(data)*64*2==budget
    np.savez_compressed(O/f"data_{design}_{budget}.npz",states=np.array([z for _,m,z in data]),parameters=np.array([m for _,m,z in data]),time=ts)
    return data

rr=list(csv.DictReader((R/"phase1/pilot_trajectories.csv").open()))
common=[]
for tid in sorted({int(r["trajectory_id"]) for r in rr}):
    rows=[r for r in rr if int(r["trajectory_id"])==tid]
    if rows[0]["split"]!="train":common.append((rows[0]["split"],float(rows[0]["mu"]),np.array([[float(r["x"]),float(r["y"])] for r in rows])))
core.test=[r for r in common if r[0]=="test"]
val=[r for r in common if r[0]=="validation"]
v0=np.array([z[0] for _,m,z in val]);vm=np.array([m for _,m,z in val])
vt=np.array([z[1:13] for _,m,z in val]).transpose(1,0,2)

class Poly:
    def __init__(self,c):self.c=c
    def f(self,z,mu):return core.features(z,mu)@self.c,None
    def jac(self,z,mu):
        eps=1e-6
        return np.column_stack([(self.f((z+eps*np.eye(2)[k])[None],mu)[0][0]-self.f((z-eps*np.eye(2)[k])[None],mu)[0][0])/(2*eps) for k in range(2)])

records=[];ranks=[];curves={}
for budget in CONFIG["budgets"]:
    for design in CONFIG["designs"]:
        data=dataset(design,budget)
        X=[];Y=[]
        for _,m,z in data:
            X.append(core.features(z[4:-4],m));Y.append(savgol_filter(z,9,3,deriv=1,delta=core.dt,axis=0)[4:-4])
        X=np.concatenate(X);Y=np.concatenate(Y);c=np.linalg.lstsq(X,Y,rcond=1e-10)[0]
        ranks.append({"budget":budget,"design":design,"dictionary_rank":int(np.linalg.matrix_rank(X,tol=1e-8)),"dictionary_columns":13,
                      "singular_values":np.linalg.svd(X,compute_uv=False).tolist(),
                      "unique_state_parameter_rows_rounded_10dp":len(np.unique(np.round(np.column_stack([np.concatenate([z for _,m,z in data]),np.repeat([m for _,m,z in data],64)]),10),axis=0))})
        np.save(O/f"poly_{design}_{budget}.npy",c)
        met,curve=core.metrics(Poly(c));records.append({"model":"polynomial","design":design,"budget":budget,"seed":None,"training_steps":0,**met})
        for seed in CONFIG["seeds"]:
            start=time.perf_counter();net=core.Net(seed);rng=np.random.default_rng(100+seed)
            mom=[np.zeros_like(p) for p in net.p];var=[np.zeros_like(p) for p in net.p]
            best=np.inf;bestp=None;beststep=None;history=[];lastloss=None
            for step in range(1,3601):
                ids=rng.integers(0,len(data),48);starts=rng.integers(0,57,48)
                z0=np.array([data[i][2][s] for i,s in zip(ids,starts)]);mus=np.array([data[i][1] for i in ids])
                tar=np.array([data[i][2][s+1:s+7] for i,s in zip(ids,starts)]).transpose(1,0,2)
                loss,grad=core.objective(net,z0,mus,tar)
                norm=np.sqrt(sum(np.sum(g*g) for g in grad));grad=[g*min(1,5/max(norm,1e-12)) for g in grad]
                for j,g in enumerate(grad):
                    mom[j]=.9*mom[j]+.1*g;var[j]=.999*var[j]+.001*g*g
                    net.p[j]-=.003*(mom[j]/(1-.9**step))/(np.sqrt(var[j]/(1-.999**step))+1e-8)
                if step%100==0:
                    vl=core.objective(net,v0,vm,vt)[0]
                    history.append({"step":step,"train_batch_loss":float(loss),"validation_segment_mse":float(vl)})
                    if vl<best:best=vl;bestp=[p.copy() for p in net.p];beststep=step
                if step in CONFIG["checkpoints"]:
                    live=net.p;net.p=[p.copy() for p in bestp]
                    met,curve=core.metrics(net);curves[(design,budget,seed,step)]=curve
                    filename=f"node_{design}_{budget}_seed{seed}_step{step}.npz"
                    np.savez(O/filename,**{f"p{i}":p for i,p in enumerate(net.p)})
                    net.p=live
                    rec={"model":"node","design":design,"budget":budget,"seed":seed,"training_steps":step,"best_checkpoint_step":beststep,
                         "validation_segment_mse":float(best),"elapsed_seconds":time.perf_counter()-start,**met}
                    records.append(rec)
                    print(json.dumps({k:rec[k] for k in ["design","budget","seed","training_steps","critical_parameter_error","validation_segment_mse"]}),flush=True)
            (O/f"history_{design}_{budget}_seed{seed}.json").write_text(json.dumps(history,indent=2))
            (O/"results.json").write_text(json.dumps(records,indent=2))
(O/"dictionary_identifiability.json").write_text(json.dumps(ranks,indent=2))
keys=sorted(set().union(*(r.keys() for r in records)))
with (O/"results.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)

# Summaries retain diagnostic failures rather than dropping them invisibly.
summ=[]
for budget in CONFIG["budgets"]:
    for design in CONFIG["designs"]:
        for steps in CONFIG["checkpoints"]:
            runs=[r for r in records if r["model"]=="node" and r["budget"]==budget and r["design"]==design and r["training_steps"]==steps]
            vals=[r["critical_parameter_error"] for r in runs if r["critical_parameter_error"] is not None]
            summ.append({"budget":budget,"design":design,"steps":steps,"crossing_failure_count":5-len(vals),
                         "crossing_error_median":float(np.median(vals)) if vals else None,
                         "short_mse_median":float(np.median([r["short_test_mse"] for r in runs])),
                         "errors":vals})
(O/"summary.json").write_text(json.dumps(summ,indent=2))
fig,axs=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
colors={"passive":"#dc6944","random":"#2563eb","probe":"#20a078"}
for ax,budget in zip(axs,CONFIG["budgets"]):
    for j,design in enumerate(CONFIG["designs"]):
        for off,steps in [(-.13,1200),(.13,3600)]:
            ss=next(s for s in summ if s["budget"]==budget and s["design"]==design and s["steps"]==steps)
            ax.scatter(np.full(len(ss["errors"]),j+off),ss["errors"],color=colors[design],alpha=.65,marker="o" if steps==1200 else "s")
            if ss["errors"]:ax.plot([j+off-.07,j+off+.07],[ss["crossing_error_median"]]*2,color="black",lw=2)
            ax.text(j+off,.415,f'{ss["crossing_failure_count"]}/5',ha="center",fontsize=8)
    ax.set(xticks=range(3),xticklabels=CONFIG["designs"],ylabel="Candidate crossing error",ylim=(-.01,.45),title=f"{budget} scalar observations")
fig.suptitle("Observation design × compute: circles 1200 steps; squares 3600\nTop labels: missing / ambiguous crossings; black bars: medians")
fig.savefig(O/"design_comparison.png",dpi=180);fig.savefig(O/"design_comparison.svg");plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(10,3.2),constrained_layout=True)
for ax,design in zip(axs,CONFIG["designs"]):
    d=np.load(O/f"data_{design}_4096.npz")
    for m in CONFIG["train_parameters"]:
        zz=d["states"][d["parameters"]==m].reshape(-1,2);ax.scatter(zz[:,0],zz[:,1],s=2,alpha=.4,label=f"mu={m}")
    ax.set(xlabel="x",ylabel="y",title=design,aspect="equal",xlim=(-.85,.85),ylim=(-.85,.85))
axs[-1].legend(fontsize=7)
fig.suptitle("Training state coverage: same observation count, different excitation")
fig.savefig(O/"training_coverage.png",dpi=180);fig.savefig(O/"training_coverage.svg");plt.close(fig)
print("Phase 3 comparison complete.",flush=True)
