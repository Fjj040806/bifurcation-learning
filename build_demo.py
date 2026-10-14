"""Export saved scientific weights into a self-contained offline HTML demo."""
from pathlib import Path
import json,re
import numpy as np
from learning_core import Net
R=Path(__file__).resolve().parent
records=json.loads((R/"phase3/results.json").read_text())
record=next(r for r in records if r["model"]=="node" and r["design"]=="random" and r["budget"]==4096 and r["seed"]==0 and r["training_steps"]==3600)
weights=np.load(R/"phase3/node_random_4096_seed0_step3600.npz")
models={"hopf":{"parameters":[weights[f"p{i}"].tolist() for i in range(6)],"true_critical":0,"learned_critical":record["crossings"][0]}}
weights=np.load(R/"phase4/fhn_node_seed0.npz");meta=np.load(R/"phase4/fhn_dataset.npz")
record=json.loads((R/"phase4/fhn_results.json").read_text())[1]
models["fhn"]={"parameters":[weights[f"p{i}"].tolist() for i in range(6)],"center":meta["center"].tolist(),"scale":meta["scale"].tolist(),
               "parameter_center":float(meta["parameter_center"]),"parameter_scale":float(meta["parameter_scale"]),
               "true_critical":.3312813374547458,"learned_critical":record["crossings"][0]}
payload="const MODELS = "+json.dumps(models)+";\n// END MODEL DATA\n"
(R/"demo/models.js").write_text(payload)
html=(R/"demo/index.html").read_text()
if '<script src="models.js"></script><script>' in html:
    html=html.replace('<script src="models.js"></script><script>',"<script>\n"+payload)
else:
    html=re.sub(r"const MODELS = .*?// END MODEL DATA\n",lambda m:payload,html,count=1,flags=re.S)
(R/"demo/index.html").write_text(html)
fixtures=[]
for system in ["hopf","fhn"]:
    net=Net(0);net.p=[np.array(p) for p in models[system]["parameters"]]
    for parameter in ([-.1,.1,.3] if system=="hopf" else [.2,.36,.5]):
        for state in [[.2,-.3],[-.8,.15]]:
            z=np.array(state)
            if system=="hopf":f=net.f(z[None],parameter)[0][0]
            else:f=meta["scale"]*net.f(((z-meta["center"])/meta["scale"])[None],(parameter-float(meta["parameter_center"]))/float(meta["parameter_scale"]))[0][0]
            fixtures.append({"system":system,"parameter":parameter,"state":state,"expected":f.tolist()})
(R/"tests/demo_expected.json").write_text(json.dumps(fixtures,indent=2))
print("Built standalone demo with saved model weights.")
