"""Procedural glow rendering of actual ODE trajectories (not generated imagery)."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from PIL import Image,ImageDraw,ImageFilter,ImageChops,ImageFont
import subprocess

R=Path(__file__).parent;O=R/"phase4";O.mkdir(exist_ok=True)
W,H=1600,900
fontpath="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    font=ImageFont.truetype(fontpath if Path(fontpath).exists() else "DejaVuSans.ttf",25)
    small=ImageFont.truetype(fontpath if Path(fontpath).exists() else "DejaVuSans.ttf",18)
except OSError:
    font=ImageFont.load_default(size=25);small=ImageFont.load_default(size=18)
rng=np.random.default_rng(712)
t=np.linspace(0,50,241)
N=150
theta=rng.uniform(0,2*np.pi,N);r0=rng.uniform(.08,1.45,N)
mu=.25
rad=np.sqrt(mu/(1+(mu/r0[:,None]**2-1)*np.exp(-2*mu*t[None])))
hopf=np.stack([rad*np.cos(theta[:,None]+t),rad*np.sin(theta[:,None]+t)],axis=-1)
initial=np.column_stack([rng.uniform(-2,2,90),rng.uniform(-.35,1.4,90)])
def vector(t,z):
    xy=z.reshape(-1,2);x,y=xy.T
    return np.column_stack([x-x**3/3-y+.36,.08*(x+.7-.8*y)]).ravel()
sol=solve_ivp(vector,(0,90),initial.ravel(),t_eval=np.linspace(0,90,241),method="DOP853",rtol=1e-9,atol=1e-11)
assert sol.success
fhn=sol.y.T.reshape(241,90,2).transpose(1,0,2)
np.savez_compressed(O/"render_particle_trajectories.npz",hopf=hopf,fhn=fhn,hopf_time=t,fhn_time=sol.t)

def mapxy(z,bounds,xoffset):
    xmin,xmax,ymin,ymax=bounds
    return np.column_stack([xoffset+45+(z[:,0]-xmin)/(xmax-xmin)*710,
                            780-(z[:,1]-ymin)/(ymax-ymin)*650])

def render(k,static=False,width=W,height=H):
    layer=Image.new("RGB",(W,H),(0,0,0));dr=ImageDraw.Draw(layer)
    for paths,bounds,offset in [(hopf,(-1.6,1.6,-1.6,1.6),0),(fhn,(-2.5,2.5,-.55,1.6),800)]:
        for n,path in enumerate(paths):
            start=0 if static else max(0,k-45)
            xy=mapxy(path[start:k+1],bounds,offset)
            # Color encodes trail age; brightness has no quantitative meaning.
            for j in range(1,len(xy)):
                age=j/max(1,len(xy)-1)
                if n%7==0:color=(int(180*age),int(45*age),int(240*age))
                else:color=(int(255*(.2+.8*age)),int(175*age**2),int(30*age))
                dr.line([tuple(xy[j-1]),tuple(xy[j])],fill=color,width=2)
            if len(xy):
                x,y=xy[-1];dr.ellipse((x-1.7,y-1.7,x+1.7,y+1.7),fill=(255,215,100))
    bloom=ImageChops.screen(layer.filter(ImageFilter.GaussianBlur(14)),layer.filter(ImageFilter.GaussianBlur(4)))
    pic=ImageChops.screen(layer,bloom)
    # Retain a dark warm background, annotation outside the scientific panels.
    pic=ImageChops.screen(pic,Image.new("RGB",(W,H),(9,4,7)))
    draw=ImageDraw.Draw(pic)
    draw.text((42,26),"DYNAMICAL SYSTEMS  /  SIMULATED PHASE-SPACE FLOW",font=font,fill=(236,228,212))
    draw.text((46,86),"Hopf oscillator  |  mu = 0.25",font=small,fill=(211,191,173))
    draw.text((846,86),"FitzHugh–Nagumo  |  I = 0.36",font=small,fill=(211,191,173))
    draw.line((800,122,800,790),fill=(42,25,28))
    draw.text((44,826),"True ODE trajectories. Glow and color are presentation effects; particle density is not probability.",font=small,fill=(163,146,139))
    draw.text((44,854),f"Independent clocks: Hopf t={t[k]:.1f}; FHN t={sol.t[k]:.1f}. No parameter ramp.",font=small,fill=(163,146,139))
    return pic if (width,height)==(W,H) else pic.resize((width,height),Image.Resampling.LANCZOS)

render(240,True).save(O/"rendered_phase_flow.png")
# Direct raw frames to ffmpeg; no temporary image sequence.
cmd=["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgb24","-s","1000x562","-r","15","-i","-",
     "-an","-vcodec","libx264","-pix_fmt","yuv420p","-crf","21",str(O/"rendered_phase_flow.mp4")]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
for k in range(0,241,2):p.stdin.write(render(k,width=1000,height=562).tobytes())
p.stdin.close();error=p.stderr.read();rc=p.wait()
if rc:raise RuntimeError(error.decode()[-1000:])
# GIF is a smaller looping preview; MP4 is the primary high-quality animation.
subprocess.run(["ffmpeg","-y","-i",str(O/"rendered_phase_flow.mp4"),"-vf","fps=10,scale=800:-1:flags=lanczos","-loop","0",str(O/"rendered_phase_flow.gif")],
               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
(O/"render_metadata.json").write_text(json.dumps({"seed":712,"systems":{"hopf":{"mu":.25,"particles":150,"time":[0,50]},"fhn":{"I":.36,"particles":90,"time":[0,90]}},
    "presentation":"screen compositing with Gaussian bloom; color is trail age/style, not scalar diagnostic",
    "data":"render_particle_trajectories.npz; scientifically exact axes omitted in showcase only",
    "coordinates":"left: x,y [-1.6,1.6]; right: x [-2.5,2.5], y [-.55,1.6]; unequal coordinate scales in FHN showcase"},indent=2))
print("Rendered PNG, MP4 and GIF complete.",flush=True)
