"""Summaries, candidate orientation, periodic tolerance refinement and report assets."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R=Path(__file__).parent;O=R/"phase4"
scope={"__file__":str(R/"phase4_cycles.py"),"__name__":"phase4_definitions"}
exec((R/"phase4_cycles.py").read_text().split("xc=-np.sqrt(.936)")[0],scope)
core=scope["core"];loadnet=scope["loadnet"];Polynomial=scope["Polynomial"];PhysicalNet=scope["PhysicalNet"]
summary={}
for name in ["polynomial_robustness","polynomial_extrapolation","node_robustness"]:
    rows=json.loads((O/(name+".json")).read_text())
    for r in rows:
        r["orientation_matches_truth"]=bool(len(r["crossings"])==1 and len(r.get("crossing_slopes",[]))==1 and r["crossing_slopes"][0]>0)
    (O/(name+".json")).write_text(json.dumps(rows,indent=2))
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (O/(name+".csv")).open("w") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    groups=[]
    by={}
    for r in rows:
        fields=["control"] if name=="node_robustness" else (["design","amplitude","T","noise_sigma"] if name=="polynomial_robustness" else ["design","noise_sigma"])
        key=tuple(r[k] for k in fields);by.setdefault(key,[]).append(r)
    for key,rr in by.items():
        e=[r["crossing_error"] for r in rr if r["crossing_error"] is not None]
        groups.append({**dict(zip(fields,key)),"n":len(rr),"unique_candidates":len(e),"correct_orientation":sum(r["orientation_matches_truth"] for r in rr),
                       "median_error_conditional":float(np.median(e)) if e else None})
    summary[name]=groups
(O/"robustness_summary.json").write_text(json.dumps(summary,indent=2))
cycles=json.loads((O/"hopf_and_cycle_audit.json").read_text())
meta=np.load(O/"fhn_dataset.npz")
def getmodel(name):
    if name=="truth_hopf":return scope["TruthHopf"]()
    if name=="truth_fhn":return scope["TruthFHN"]()
    if name=="polynomial_fhn":return Polynomial(np.load(O/"fhn_polynomial.npy"))
    seed=int(name.split("seed")[-1])
    if "node_hopf" in name:return loadnet(R/"phase3"/f"node_random_4096_seed{seed}_step3600.npz",seed)
    return PhysicalNet(loadnet(O/f"fhn_node_seed{seed}.npz",seed),meta)
refinement=[]
for r in cycles:
    if "candidate" not in r:continue
    m=getmodel(r["model"]);mh=r["candidate"];guess=np.zeros(2) if r["system"]=="hopf" else np.array([-1.,0.])
    e=root(lambda z:m.f(z[None],mh)[0][0],guess)
    def g(mu):
        eq=root(lambda z:m.f(z[None],mu)[0][0],e.x)
        return np.trace(m.jac(eq.x,mu))/2
    h=1e-5;r["crossing_slope"]=(g(mh+h)-g(mh-h))/(2*h)
    r["local_nondegeneracy_numerically_supported"]=bool(r["sign_consistent"] and abs(r["crossing_slope"])>1e-5)
    for point in r.get("periodic_section_points",[]):
        if point["status"]!="verified_periodic_point":continue
        mu,T,z0=point["parameter"],point["period"],np.array(point["initial_state"])
        def f(t,z):return [*m.f(z[:2][None],mu)[0][0],np.trace(m.jac(z[:2],mu))]
        sol=solve_ivp(f,(0,T),[*z0,0],method="DOP853",rtol=1e-11,atol=1e-13)
        err=float(np.linalg.norm(sol.y[:2,-1]-z0));change=abs(float(np.exp(sol.y[2,-1]))-point["transverse_multiplier"])
        assert sol.success and err<1e-6 and change<1e-5
        refinement.append({"model":r["model"],"section_radius":point["section_radius"],"tight_closing_error":err,"multiplier_change":float(change)})
(O/"hopf_and_cycle_audit.json").write_text(json.dumps(cycles,indent=2))
(O/"periodic_tolerance_refinement.json").write_text(json.dumps(refinement,indent=2))

node=json.loads((O/"node_robustness.json").read_text())
fig,ax=plt.subplots(figsize=(10,4.3),constrained_layout=True)
labels=["Random","Probe .02","Probe .10","Probe .02 T20","Random + noise","Probe + noise"]
for c,label in enumerate(labels):
    rr=[r for r in node if r["control"]==c]
    for j,r in enumerate(rr):
        if r["crossing_error"] is not None:
            ax.scatter(c+(j-1.5)*.055,r["crossing_error"],marker="o" if r["orientation_matches_truth"] else "x",color="#2563eb" if r["orientation_matches_truth"] else "#dc6944",s=45)
    correct=sum(r["orientation_matches_truth"] for r in rr)
    ax.text(c,.405,f"{correct}/4",ha="center",fontsize=9)
ax.set(xticks=range(6),xticklabels=labels,ylim=(-.01,.44),ylabel="Candidate crossing error",title="NODE sensitivity: labels count correctly oriented unique crossings")
ax.tick_params(axis="x",labelrotation=15)
fig.savefig(O/"node_sensitivity.png",dpi=180);fig.savefig(O/"node_sensitivity.svg")
print(json.dumps({"counts":{k:len(json.loads((O/(k+".json")).read_text())) for k in ["polynomial_robustness","polynomial_extrapolation","node_robustness","fhn_results"]},
                 "periodic_refinement_points":len(refinement),"largest_tight_closing_error":max(r["tight_closing_error"] for r in refinement),
                 "node_summary":summary["node_robustness"]},indent=2))
