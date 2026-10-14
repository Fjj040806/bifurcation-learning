"""Declared rotational prior: fit all radial and angular coefficients from data."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import savgol_filter

R=Path(__file__).parent;O=R/"phase4"
scope={"__file__":str(R/"phase4_robustness.py"),"__name__":"acquisition_definitions"}
exec((R/"phase4_robustness.py").read_text().split("polyrows=[]")[0],scope)
acquire=scope["acquire"]
def dictionary(z,mu):
    x,y=z.T;s=x*x+y*y
    g=np.column_stack([np.ones(len(x)),np.full(len(x),mu),s])
    w=np.column_stack([np.ones(len(x)),np.full(len(x),mu),s])
    return np.vstack([np.column_stack([x[:,None]*g,-y[:,None]*w]),
                      np.column_stack([y[:,None]*g,x[:,None]*w])])
records=[]
for previous in json.loads((O/"polynomial_robustness.json").read_text()):
    d,T,amp,noise,seed=[previous[k] for k in ["design","T","amplitude","noise_sigma","acquisition_seed"]]
    data=acquire(d,T,amp,noise,seed);X=[];Y=[]
    for _,mu,z in data:
        derivative=savgol_filter(z,9,3,deriv=1,delta=T/63,axis=0)[4:-4]
        X.append(dictionary(z[4:-4],mu));Y.append(derivative.T.ravel())
    X=np.concatenate(X);Y=np.concatenate(Y);c=np.linalg.lstsq(X,Y,rcond=1e-10)[0]
    mh=-c[0]/c[1] if abs(c[1])>1e-8 else None
    frequency=abs(c[3]+c[4]*mh) if mh is not None else None
    valid=mh is not None and -.4<mh<.4 and frequency>1e-5
    records.append({"design":d,"T":T,"amplitude":amp,"noise_sigma":noise,"acquisition_seed":seed,"coefficients":c.tolist(),
                    "rank":int(np.linalg.matrix_rank(X,tol=1e-8)),"columns":6,
                    "candidate":float(mh) if valid else None,"crossing_error":abs(float(mh)) if valid else None,
                    "orientation_matches_truth":bool(valid and c[1]>0),"frequency":float(frequency) if valid else None})
(O/"equivariant_robustness.json").write_text(json.dumps(records,indent=2))
# Persist the exact replicated NODE observations without retraining.
controls=json.loads((O/"robustness_config.json").read_text())["node_controls"]
for control,(design,amp,T,noise) in enumerate(controls):
    for seed in [0,1]:
        data=acquire(design,T,amp,noise,seed)
        np.savez_compressed(O/f"robust_data_control{control}_seed{seed}.npz",states=np.array([z for s,m,z in data]),parameters=np.array([m for s,m,z in data]),time=np.linspace(0,T,64))
print("Completed 150 equivariant-prior fits; saved 12 replicated NODE acquisition datasets.",flush=True)
