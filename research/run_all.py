"""Run the whole stress battery in order and tee each report to results/."""
import subprocess, sys, os, pathlib

HERE = pathlib.Path(__file__).parent
STRESS = HERE / "stress"
ORDER = ["s00_verify", "s01_conditioning", "s01c_underflow",
         "s02_relevance_blowup", "s03_weight_validity",
         "s04_principal_hypothesis", "s05_stability", "s06_reserve",
         "s07_oracle_regret", "s08_irreversibility", "s09_chain",
         "s10_backward_recursion"]

if __name__ == "__main__":
    py = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    only = sys.argv[2:] or ORDER
    (HERE / "results").mkdir(exist_ok=True)
    for name in only:
        print(f"\n{'='*78}\n{name}\n{'='*78}", flush=True)
        r = subprocess.run([py, str(STRESS / f"{name}.py")],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stdout.write(r.stderr)
        (HERE / "results" / f"{name}.txt").write_text(r.stdout + r.stderr)
