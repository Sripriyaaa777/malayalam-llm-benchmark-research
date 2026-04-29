"""
run_all.py — Run all experiments in order and generate final metrics.

Usage:
    cd scripts/
    python run_all.py [--skip-exp1-2] [--skip-exp3] [--skip-exp4-5]
                      [--skip-exp6] [--skip-stats] [--skip-errors]
"""
import os, sys, argparse, subprocess
from datetime import datetime

SCRIPTS = os.path.dirname(os.path.abspath(__file__))

def run(script, label):
    print(f"\n{'='*70}")
    print(f"  RUNNING: {label}")
    print(f"{'='*70}\n")
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, script)],
        cwd=SCRIPTS
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed (exit {result.returncode}).")
        print("Fix the error and re-run with --skip-* flags to resume.")
        sys.exit(result.returncode)
    print(f"\n[OK] {label} completed.\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-exp1-2",  action="store_true")
    parser.add_argument("--skip-exp3",    action="store_true")
    parser.add_argument("--skip-exp4-5",  action="store_true")
    parser.add_argument("--skip-exp6",    action="store_true")
    parser.add_argument("--skip-stats",   action="store_true")
    parser.add_argument("--skip-errors",  action="store_true")
    args = parser.parse_args()

    start = datetime.now()
    print(f"\nMALAYALAM LLM BENCHMARK — FULL PIPELINE  (6 models)")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated total time: ~90–120 minutes")

    if not args.skip_exp1_2:
        run("exp1_2_zeroshot_3shot.py", "Exp 1 & 2 — 0-shot / 3-shot (100 samples, Llama 3.3 + Mistral)")
    if not args.skip_exp3:
        run("exp3_5shot_100.py",        "Exp 3 — 5-shot improved (100 samples, Llama 3.3 + Mistral)")
    if not args.skip_exp4_5:
        run("exp4_5_500sample.py",      "Exp 4 & 5 — 500 samples, ALL 6 models")
    if not args.skip_exp6:
        run("exp6_romanization.py",     "Exp 6 — Romanization control (Llama 3.3)")
    if not args.skip_stats:
        run("statistical_tests.py",     "Statistical significance tests")
    if not args.skip_errors:
        run("error_analysis.py",        "Error analysis (Mistral Large)")
    run("generate_metrics_matrix.py",   "Master metrics matrix (final tables)")

    duration = (datetime.now()-start).total_seconds()/60
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {duration:.1f} minutes")
    print(f"  Key output: results/metrics_matrix_*.txt")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
