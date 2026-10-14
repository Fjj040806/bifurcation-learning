import numpy as np
from scipy.optimize import root,brentq
from scipy.integrate import solve_ivp
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

def features(z,mu):
    x,y=np.asarray(z).T;m=np.broadcast_to(mu,len(x))
    return np.column_stack([np.ones(len(x)),x,y,m,m*x,m*y,x*x,x*y,y*y,x**3,x*x*y,x*y*y,y**3])

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

