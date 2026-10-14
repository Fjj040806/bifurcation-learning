"""Independent local-derivative and root audit using saved models; no retraining."""
from pathlib import Path
import json,numpy as np
from scipy.optimize import root
R=Path(__file__).parent
scope={"__file__":str(R/"phase2_train.py"),"__name__":"audit_definitions"}
exec((R/"phase2_train.py").read_text().split("results=[];curves={}")[0],scope)
Net=scope["Net"];objective=scope["objective"];tra=scope["train"]
net=Net(12);z0=tra[0][2][:1];mu=np.array([tra[0][1]]);target=tra[0][2][1:4,None,:]
loss,grad=objective(net,z0,mu,target);errors=[]
for j,p in enumerate(net.p):
    index=(0,)*p.ndim;old=p[index];eps=1e-6
    p[index]=old+eps;lp=objective(net,z0,mu,target)[0]
    p[index]=old-eps;lm=objective(net,z0,mu,target)[0]
    p[index]=old;errors.append(abs((lp-lm)/(2*eps)-grad[j][index]))
assert max(errors)<1e-7
out={"gradient_error_by_parameter_tensor":errors,"candidates":[]}
results=json.loads((R/"phase2/results.json").read_text())["results"]
for rr in results:
    if rr["model"]=="polynomial":model=scope["Polynomial"]()
    else:
        model=Net(rr["seed"]);saved=np.load(R/"phase2"/(rr["model"]+".npz"));model.p=[saved[f"p{i}"] for i in range(6)]
    mh=rr["crossings"][0];eq=root(lambda z:model.f(z[None],mh)[0][0],np.zeros(2),tol=1e-10)
    j=model.jac(eq.x,mh);delta=1e-5
    def growth(m):
        e=root(lambda z:model.f(z[None],m)[0][0],eq.x,tol=1e-10)
        return np.trace(model.jac(e.x,m))/2
    slope=(growth(mh+delta)-growth(mh-delta))/(2*delta)
    numerical=np.column_stack([(model.f((eq.x+delta*np.eye(2)[k])[None],mh)[0][0]-model.f((eq.x-delta*np.eye(2)[k])[None],mh)[0][0])/(2*delta) for k in range(2)])
    discrepancy=float(np.max(abs(j-numerical)));assert discrepancy<1e-7
    minimum_det=min(float(np.linalg.det(model.jac(root(lambda z:model.f(z[None],m)[0][0],eq.x).x,m))) for m in np.linspace(-.4,.4,81))
    assert minimum_det>0 and slope>0
    out["candidates"].append({"model":rr["model"],"frequency":float(np.sqrt(np.linalg.det(j))),"crossing_slope":float(slope),"equilibrium_residual":float(np.linalg.norm(eq.fun)),"jacobian_finite_difference_error":discrepancy,"min_determinant_on_audit_grid":minimum_det})
(R/"phase2/audit.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
