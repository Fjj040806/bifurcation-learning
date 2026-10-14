"""Numerical and evidence-integrity checks; no full retraining."""
import json,sys,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
import learning_core as core

class NumericalChecks(unittest.TestCase):
    def test_rk4_reverse_gradient(self):
        net=core.Net(7);z0=np.array([[.3,.2],[.1,-.4]]);mu=np.array([.2,-.2])
        target=np.zeros((3,2,2));_,grad=core.objective(net,z0,mu,target)
        for j,p in enumerate(net.p):
            idx=(0,)*p.ndim;old=p[idx];h=1e-6
            p[idx]=old+h;plus=core.objective(net,z0,mu,target)[0]
            p[idx]=old-h;minus=core.objective(net,z0,mu,target)[0]
            p[idx]=old
            self.assertAlmostEqual((plus-minus)/(2*h),grad[j][idx],places=7)
    def test_analytic_network_jacobian(self):
        net=core.Net(4);z=np.array([.25,-.17]);h=1e-6
        fd=np.column_stack([(net.f((z+h*np.eye(2)[j])[None],.1)[0][0]-
                             net.f((z-h*np.eye(2)[j])[None],.1)[0][0])/(2*h) for j in range(2)])
        np.testing.assert_allclose(fd,net.jac(z,.1),rtol=1e-6,atol=1e-8)
    def test_saved_observation_budgets(self):
        for b in [4096,16384]:
            for d in ["passive","random","probe"]:
                a=np.load(R/f"phase3/data_{d}_{b}.npz")
                self.assertEqual(a["states"].size,b)
                self.assertTrue(np.isfinite(a["states"]).all())
        for p in (R/"phase4").glob("robust_data_control*_seed*.npz"):
            self.assertEqual(np.load(p)["states"].size,4096)
    def test_main_summary_preserves_failures(self):
        rows=json.loads((R/"phase3/results.json").read_text())
        summaries=json.loads((R/"phase3/summary.json").read_text())
        for s in summaries:
            rr=[r for r in rows if r["model"]=="node" and r["design"]==s["design"] and
                r["budget"]==s["budget"] and r["training_steps"]==s["steps"]]
            self.assertEqual(len(rr),5)
            e=[r["critical_parameter_error"] for r in rr if r["critical_parameter_error"] is not None]
            self.assertEqual(5-len(e),s["crossing_failure_count"])
            if e:self.assertAlmostEqual(float(np.median(e)),s["crossing_error_median"])
    def test_exact_rank_ambiguity_and_periodic_refinement(self):
        ranks=json.loads((R/"phase3/dictionary_identifiability.json").read_text())
        self.assertTrue(all(r["dictionary_rank"]==(11 if r["design"]=="passive" else 13) for r in ranks))
        checks=json.loads((R/"phase4/periodic_tolerance_refinement.json").read_text())
        self.assertEqual(len(checks),21)
        self.assertTrue(all(r["tight_closing_error"]<1e-6 and r["multiplier_change"]<1e-5 for r in checks))
    def test_experiment_counts_and_finite_json(self):
        expected={"polynomial_robustness":150,"equivariant_robustness":150,"polynomial_extrapolation":20,
                  "node_robustness":24,"fhn_results":6}
        for name,n in expected.items():
            self.assertEqual(len(json.loads((R/f"phase4/{name}.json").read_text())),n)
        def reject(value):raise ValueError("Non-finite JSON constant: "+value)
        for p in R.rglob("*.json"):json.loads(p.read_text(),parse_constant=reject)
if __name__=="__main__":unittest.main()
