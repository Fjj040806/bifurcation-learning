"""Single entry point for verification, experiments and paper build."""
from pathlib import Path
import argparse,subprocess,sys,shutil,os
R=Path(__file__).resolve().parent
def run(script):
    env=os.environ.copy();env.setdefault("OPENBLAS_NUM_THREADS","1")
    subprocess.run([sys.executable,str(R/script)],cwd=R,env=env,check=True)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["verify","reference","full","paper"],nargs="?",default="verify")
    args=parser.parse_args()
    if args.mode=="verify":
        subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=R,check=True)
        node=shutil.which("node")
        if node:subprocess.run([node,str(R/"tests/demo_smoke.cjs")],cwd=R,check=True)
        print("Verification completed. No model retraining.")
    elif args.mode=="reference":run("phase1_build.py")
    elif args.mode=="full":
        for script in ["phase0/make_motivation.py","phase1_build.py","phase2_train.py","phase2_audit.py","phase3_compare.py","phase3_analyze.py",
                       "phase4_robustness.py","phase4_equivariant.py","phase4_fhn.py","phase4_cycles.py","phase4_finalize.py",
                       "render_dynamics.py","build_demo.py"]:run(script)
        print("Experiments regenerated. Rebuild manuscript tables with paper mode.")
    elif args.mode=="paper":
        run("paper/build_tables.py")
        for _ in range(2):subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error","manuscript.tex"],cwd=R/"paper",check=True,stdout=subprocess.DEVNULL)
        print("Built paper/manuscript.pdf")
if __name__=="__main__":main()
