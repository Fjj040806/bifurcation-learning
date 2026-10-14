"""Trajectory-trained neural ODE with explicit NumPy RK4 backpropagation."""
from pathlib import Path
import csv,json,time
import numpy as np
from scipy.optimize import root,brentq
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

R=Path(__file__).parent; O=R/"phase2";O.mkdir(exist_ok=True)
rows=list(csv.DictReader((R/"phase1/pilot_trajectories.csv").open()))
tra=[]
for tid in sorted({int(r["trajectory_id"]) for r in rows}):
    rr=[r for r in rows if int(r["trajectory_id"])==tid]
    tra.append((rr[0]["split"],float(rr[0]["mu"]),np.array([[float(r["x"]),float(r["y"])] for r in rr])))
dt=5/63

class Net:
    def __init__(self,seed):
        g=np.random.default_rng(seed);h=24
        self.p=[g.normal(0,.6,(3,h)),np.zeros(h),g.normal(0,.25,(h,h)),np.zeros(h),g.normal(0,.03,(h,2)),np.zeros(2)]
    def f(self,z,mu):
        inp=np.column_stack([z,np.broadcast_to(mu,len(z))])
        w,b,v,c,u,d=self.p;a=np.tanh(inp@w+b);h=np.tanh(a@v+c)
        return h@u+d,(inp,a,h)
    def back(self,g,cache):
        inp,a,h=cache;w,b,v,c,u,d=self.p
        gh=(g@u.T)*(1-h*h);ga=(gh@v.T)*(1-a*a)
        dp=[inp.T@ga,ga.sum(0),a.T@gh,gh.sum(0),h.T@g,g.sum(0)]
        return (ga@w.T)[:,:2],dp
    def step(self,z,mu):
        k1,c1=self.f(z,mu);k2,c2=self.f(z+dt*k1/2,mu);k3,c3=self.f(z+dt*k2/2,mu);k4,c4=self.f(z+dt*k3,mu)
        return z+dt*(k1+2*k2+2*k3+k4)/6,(c1,c2,c3,c4)
    def stepback(self,g,caches):
        gz=g.copy();gs=[dt*g/6,dt*g/3,dt*g/3,dt*g/6]
        dp=[np.zeros_like(p) for p in self.p]
        for k in [3,2,1,0]:
            dz,dpar=self.back(gs[k],caches[k])
            dp=[a+b for a,b in zip(dp,dpar)];gz+=dz
            if k==3:gs[2]+=dt*dz
            elif k==2:gs[1]+=dt*dz/2
            elif k==1:gs[0]+=dt*dz/2
        return gz,dp
    def jac(self,z,mu):
        _,cache=self.f(np.array(z)[None],mu)
        return np.vstack([self.back(np.eye(2)[i:i+1],cache)[0][0] for i in range(2)])

def objective(net,z0,mus,target):
    z=z0.copy();caches=[];pred=[]
    for j in range(len(target)):
        z,cache=net.step(z,mus);caches.append(cache);pred.append(z)
    pred=np.array(pred);diff=pred-target
    loss=np.mean(diff**2);direct=2*diff/diff.size
    g=np.zeros_like(z);dp=[np.zeros_like(p) for p in net.p]
    for j in reversed(range(len(target))):
        g,grad=net.stepback(g+direct[j],caches[j]);dp=[a+b for a,b in zip(dp,grad)]
    return loss,dp

# Meaningful gradient check of the implemented solver differentiation.
train=[x for x in tra if x[0]=="train"]; val=[x for x in tra if x[0]=="validation"];test=[x for x in tra if x[0]=="test"]
n=Net(0);target=train[0][2][1:4,None,:]
loss,gr=objective(n,train[0][2][:1],np.array([train[0][1]]),target)
old=n.p[0][0,0];h=1e-6
n.p[0][0,0]=old+h;lp=objective(n,train[0][2][:1],np.array([train[0][1]]),target)[0]
n.p[0][0,0]=old-h;lm=objective(n,train[0][2][:1],np.array([train[0][1]]),target)[0]
n.p[0][0,0]=old
gradient_error=abs((lp-lm)/(2*h)-gr[0][0,0]);assert gradient_error<1e-7

def features(z,mu):
    x,y=np.asarray(z).T;m=np.broadcast_to(mu,len(x))
    return np.column_stack([np.ones(len(x)),x,y,m,m*x,m*y,x*x,x*y,y*y,x**3,x*x*y,x*y*y,y**3])

# Polynomial baseline: derivatives estimated only from sampled training trajectories.
X=[];Y=[]
for _,mu,z in train:
    deriv=savgol_filter(z,9,3,deriv=1,delta=dt,axis=0)
    X.append(features(z[4:-4],mu));Y.append(deriv[4:-4])
coef=np.linalg.lstsq(np.concatenate(X),np.concatenate(Y),rcond=1e-10)[0]
np.save(O/"polynomial_coefficients.npy",coef)

class Polynomial:
    def f(self,z,mu):return features(z,mu)@coef,None
    def jac(self,z,mu):
        eps=1e-6
        return np.column_stack([(self.f((z+eps*np.eye(2)[j])[None],mu)[0][0]-self.f((z-eps*np.eye(2)[j])[None],mu)[0][0])/(2*eps) for j in range(2)])

def rollout(model,z0,mu,times):
    sol=solve_ivp(lambda t,z:model.f(z[None],mu)[0][0],(times[0],times[-1]),z0,t_eval=times,method="DOP853",rtol=1e-8,atol=1e-10)
    if not sol.success or len(sol.t)!=len(times):return None
    return sol.y.T

def truth(z0,mu,times):
    def f(t,z):
        x,y=z;s=x*x+y*y;return [(mu-s)*x-y,x+(mu-s)*y]
    return solve_ivp(f,(times[0],times[-1]),z0,t_eval=times,method="DOP853",rtol=1e-10,atol=1e-12).y.T

def metrics(model):
    times=np.linspace(0,30,301);short=[];long=[]
    for _,mu,z in test:
        pred=rollout(model,z[0],mu,times);true=truth(z[0],mu,times)
        if pred is None:short.append(None);long.append(None)
        else:short.append(float(np.mean((pred[times<=5]-true[times<=5])**2)));long.append(float(np.mean((pred-true)**2)))
    grid=np.linspace(-.4,.4,161);last=np.zeros(2);eqs=[];traces=[];dets=[];valid=[]
    for mu in grid:
        rr=root(lambda z:model.f(z[None],mu)[0][0],last,jac=lambda z:model.jac(z,mu),tol=1e-10)
        ok=rr.success and np.linalg.norm(rr.fun)<1e-7
        if ok:last=rr.x
        jj=model.jac(last,mu);eqs.append(last.copy());traces.append(np.trace(jj));dets.append(np.linalg.det(jj));valid.append(ok)
    crossings=[]
    for i in range(len(grid)-1):
        if valid[i] and valid[i+1] and traces[i]*traces[i+1]<0:
            def crossing(mu):
                rr=root(lambda z:model.f(z[None],mu)[0][0],eqs[i],tol=1e-10)
                if np.linalg.norm(rr.fun)>1e-7:raise RuntimeError("equilibrium residual")
                return np.trace(model.jac(rr.x,mu))
            try:
                mh=brentq(crossing,grid[i],grid[i+1],xtol=1e-10)
                rr=root(lambda z:model.f(z[None],mh)[0][0],eqs[i]);jj=model.jac(rr.x,mh)
                if np.linalg.det(jj)>0:crossings.append(float(mh))
            except Exception:pass
    band=np.abs(grid)>.025
    okmask=band&np.array(valid)
    correct=(np.array(traces)<0)==(grid<0)
    return {"short_test_mse":float(np.mean([v for v in short if v is not None])) if any(v is not None for v in short) else None,
            "long_test_mse":float(np.mean([v for v in long if v is not None])) if any(v is not None for v in long) else None,
            "rollout_failures":sum(v is None for v in long),"equilibrium_root_failures":sum(not v for v in valid),
            "crossings":crossings,"critical_parameter_error":abs(crossings[0]) if len(crossings)==1 else None,
            "stability_accuracy":float(np.mean(correct[okmask])),"equilibrium_displacement_max":float(np.max(np.linalg.norm(eqs,axis=1)))},(grid,np.array(traces)/2,np.array(eqs))

results=[];curves={};models={}
poly=Polynomial();met,curve=metrics(poly);results.append({"model":"polynomial",**met});curves["polynomial"]=curve;models["polynomial"]=poly
for seed in range(5):
    net=Net(seed);rng=np.random.default_rng(100+seed);moment=[np.zeros_like(p) for p in net.p];variance=[np.zeros_like(p) for p in net.p]
    best=np.inf;bestp=None;history=[]
    v0=np.array([z[0] for _,mu,z in val]);vm=np.array([mu for _,mu,z in val]);vt=np.array([z[1:13] for _,mu,z in val]).transpose(1,0,2)
    for epoch in range(1,1201):
        ids=rng.integers(0,len(train),48);starts=rng.integers(0,57,48)
        z0=np.array([train[i][2][s] for i,s in zip(ids,starts)]);mus=np.array([train[i][1] for i in ids])
        tar=np.array([train[i][2][s+1:s+7] for i,s in zip(ids,starts)]).transpose(1,0,2)
        loss,grad=objective(net,z0,mus,tar)
        norm=np.sqrt(sum(np.sum(g*g) for g in grad));grad=[g*min(1,5/max(norm,1e-12)) for g in grad]
        for j,g in enumerate(grad):
            moment[j]=.9*moment[j]+.1*g;variance[j]=.999*variance[j]+.001*g*g
            net.p[j]-=.003*(moment[j]/(1-.9**epoch))/(np.sqrt(variance[j]/(1-.999**epoch))+1e-8)
        if epoch%100==0:
            vl=objective(net,v0,vm,vt)[0];history.append([epoch,loss,vl])
            if vl<best:best=vl;bestp=[p.copy() for p in net.p]
    net.p=bestp;np.savez(O/f"node_seed{seed}.npz",**{f"p{i}":p for i,p in enumerate(net.p)})
    met,curve=metrics(net);name=f"node_seed{seed}";results.append({"model":name,"seed":seed,"validation_segment_mse":float(best),**met});curves[name]=curve;models[name]=net
    (O/f"training_seed{seed}.json").write_text(json.dumps(history))
    print(json.dumps(results[-1]),flush=True)

(O/"results.json").write_text(json.dumps({"gradient_check_absolute_error":gradient_error,"results":results},indent=2))
with (O/"results.csv").open("w") as f:
    keys=["model","short_test_mse","long_test_mse","critical_parameter_error","stability_accuracy","rollout_failures","equilibrium_root_failures"]
    w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");w.writeheader();w.writerows(results)
fig,axs=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
axs[0].plot([-.4,.4],[-.4,.4],"k--",label="Truth")
for name,(grid,g,eq) in curves.items():axs[0].plot(grid,g,alpha=.7,label=name.replace("node_seed","NODE "))
axs[0].axhline(0,color=".6",lw=.7);axs[0].set(xlabel="mu",ylabel="Half trace of learned equilibrium Jacobian",title="Independent stability diagnostic");axs[0].legend(fontsize=7)
for r in results:
    if r["critical_parameter_error"] is not None:
        axs[1].scatter(r["short_test_mse"],r["critical_parameter_error"],marker="s" if r["model"]=="polynomial" else "o",label=r["model"])
axs[1].set(xscale="log",xlabel="Short trajectory MSE (held-out parameters)",ylabel="Absolute critical-parameter error",title="Prediction versus bifurcation recovery")
fig.savefig(O/"learning_diagnostics.png",dpi=180);fig.savefig(O/"learning_diagnostics.svg");plt.close(fig)

# Median seed by VALIDATION score: never select using test bifurcation error.
ordered=sorted(results[1:],key=lambda r:r["validation_segment_mse"]);representative=ordered[len(ordered)//2]["model"]
tm=np.linspace(0,25,251);mu=.1;true=truth([.65,0],mu,tm)
preds=[true,rollout(poly,np.array([.65,0]),mu,tm),rollout(models[representative],np.array([.65,0]),mu,tm)]
fig,axs=plt.subplots(1,3,figsize=(10,3.5),constrained_layout=True);artists=[]
for ax,label,zz in zip(axs,["Truth","Polynomial",representative],preds):
    ax.set(xlim=(-.75,.75),ylim=(-.75,.75),aspect="equal",xlabel="x",ylabel="y",title=label)
    ax.plot(true[:,0],true[:,1],":",color=".75")
    line=ax.plot([],[],color="#2563eb")[0];dot=ax.plot([],[],"o",color="#dc6944")[0];artists.append((zz,line,dot))
fig.suptitle("Held-out mu=0.1: true and learned continuous-time dynamics")
def update(k):
    j=min(k*3,250)
    for zz,line,dot in artists:
        if zz is not None:line.set_data(zz[:j+1,0],zz[:j+1,1]);dot.set_data([zz[j,0]],[zz[j,1]])
FuncAnimation(fig,update,frames=85,interval=65).save(O/"learned_dynamics.gif",writer=PillowWriter(fps=15))
plt.close(fig)
print("Phase 2 complete; representative:",representative,flush=True)
