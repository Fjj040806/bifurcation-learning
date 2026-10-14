"""Generate manuscript tables directly from saved experimental evidence."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1]
main=json.loads((R/"phase3/summary.json").read_text())
noise=json.loads((R/"phase4/polynomial_robustness.json").read_text())
fhn=json.loads((R/"phase4/fhn_results.json").read_text())
lines=[]
lines.extend([r"\begin{table}[tb]",r"\centering\small",r"\caption{Main NODE comparison at 3600 steps. Candidate counts and conditional median errors are both reported.}\label{tab:main}",
              r"\begin{tabular}{rlrr}\toprule","Budget & Design & Unique candidates & Median error \\\\",r"\midrule"])
for r in main:
    if r["steps"]!=3600:continue
    e="--" if r["crossing_error_median"] is None else f'{r["crossing_error_median"]:.6f}'
    lines.append(f'{r["budget"]} & {r["design"].capitalize()} & {5-r["crossing_failure_count"]}/5 & {e} '+r"\\")
lines.extend([r"\bottomrule\end{tabular}",r"\end{table}"])
lines.extend([r"\begin{table}[tb]",r"\centering\small",r"\caption{Unconstrained polynomial noise sensitivity at $T=5$: candidate error medians over five acquisition seeds. Every displayed cell has five unique candidates.}\label{tab:noise}",
              r"\begin{tabular}{lrrr}\toprule",r"Design & $\sigma=0$ & $\sigma=0.01$ & $\sigma=0.05$ \\",r"\midrule"])
for d,a,label in [("random",.02,"Random"),("probe",.005,"Probe 0.005"),("probe",.02,"Probe 0.02"),("probe",.1,"Probe 0.10")]:
    vals=[]
    for s in [0,.01,.05]:
        e=[r["crossing_error"] for r in noise if r["design"]==d and r["amplitude"]==a and r["T"]==5 and r["noise_sigma"]==s]
        assert len(e)==5 and all(x is not None for x in e)
        v=float(np.median(e));vals.append(f'{v:.2e}' if s==0 else f'{v:.5f}')
    lines.append(label+" & "+" & ".join(vals)+r"\\")
lines.extend([r"\bottomrule\end{tabular}",r"\end{table}"])
lines.extend([r"\begin{table}[tb]",r"\centering\small",r"\caption{FHN learned critical currents and physical-coordinate test MSE over $T=20$. True current $I_H=0.33128133745$. All entries recover one candidate crossing.}\label{tab:fhn}",
              r"\begin{tabular}{lrrr}\toprule","Model & Candidate current & Location error & Test MSE \\\\",r"\midrule"])
for r in fhn:
    label="Polynomial" if r["model"]=="polynomial" else f'NODE seed {r["seed"]}'
    lines.append(f'{label} & {r["crossings"][0]:.6f} & {r["critical_error"]:.6f} & {r["test_mse"]:.3g} '+r"\\")
lines.extend([r"\bottomrule\end{tabular}",r"\end{table}"])
(R/"paper/generated_tables.tex").write_text("\n".join(lines)+"\n")
import re
blocks=re.findall(r"\\begin\{table\}.*?\\end\{table\}", "\n".join(lines), flags=re.S)
assert len(blocks)==3
for name,block in zip(["main","noise","fhn"],blocks):
    (R/f"paper/generated_{name}_table.tex").write_text(block+"\n")
print("Built three evidence-derived manuscript tables.")
