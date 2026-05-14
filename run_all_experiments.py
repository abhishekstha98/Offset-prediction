"""
run_all_experiments.py — Cross-platform orchestrator for the full experiment suite.

Experiment groups:
  baseline     — Original OffsetMPT across all 4 CV modes
  multi_channel — MultiChannelOffsetModel experiments, including SLOBO and withholding
  ablation     — Channel ablation experiments on SLOBO
  (all)        — Run all groups sequentially

Single experiment:
    python run_all_experiments.py --experiment baseline_random
    python run_all_experiments.py --experiment mc_slobo
    python run_all_experiments.py --experiment ablate_terrain

Experiment group:
    python run_all_experiments.py --group baseline
    python run_all_experiments.py --group multi_channel
    python run_all_experiments.py --group ablation

All experiments:
    python run_all_experiments.py

List available keys:
    python run_all_experiments.py --list
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

from tqdm.auto import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROGRESS_ENABLED = sys.stdout.isatty()


def progress_message(message):
    """Write messages without corrupting active tqdm bars."""
    if PROGRESS_ENABLED:
        tqdm.write(message)
    else:
        print(message, flush=True)


def run_command(command, log_file, cwd):
    """Run a command and stream clean line-based output to console and log file."""
    progress_message(f"\n{'='*70}")
    progress_message(f"  EXECUTING: {' '.join(str(c) for c in command)}")
    progress_message(f"  LOGGING TO: {log_file}")
    progress_message(f"  WORKING DIR: {cwd}")
    progress_message(f"{'='*70}\n")

    start_time = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            [str(c) for c in command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            env=env,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

        process.stdout.close()
        process.wait()

    elapsed = time.time() - start_time
    if process.returncode == 0:
        progress_message(f"\n[SUCCESS] Completed in {elapsed/60:.2f} minutes.")
    else:
        progress_message(
            f"\n[FAILED] Exited with error code {process.returncode} after {elapsed/60:.2f} minutes."
        )

    return process.returncode


def build_experiments(root_dir, python_exe, out_dir):
    """Return the full ordered list of experiment definitions."""
    train_script      = root_dir / "src" / "train.py"
    withholding_script = root_dir / "src" / "station_withholding_test.py"

    return [
        # ── Baseline experiments ──────────────────────────────────────────────
        {
            "key":   "baseline_random",
            "group": "baseline",
            "name":  "Baseline / Random CV",
            "command": [python_exe, "-u", str(train_script), "--cv_mode", "random",
                        "--model_type", "baseline"],
            "log":   out_dir / "baseline_random.log",
        },
        {
            "key":   "baseline_slobo",
            "group": "baseline",
            "name":  "Baseline / SLOBO",
            "command": [python_exe, "-u", str(train_script), "--cv_mode", "slobo",
                        "--model_type", "baseline"],
            "log":   out_dir / "baseline_slobo.log",
        },
        {
            "key":   "baseline_stlobo",
            "group": "baseline",
            "name":  "Baseline / ST-LOBO",
            "command": [python_exe, "-u", str(train_script), "--cv_mode", "st_lobo",
                        "--model_type", "baseline"],
            "log":   out_dir / "baseline_stlobo.log",
        },
        {
            "key":   "baseline_withholding",
            "group": "baseline",
            "name":  "Baseline / Station Withholding",
            "command": [python_exe, "-u", str(withholding_script)],
            "log":   out_dir / "baseline_withholding.log",
        },
        # ── Multi-channel full model ──────────────────────────────────────────
        {
            "key":   "mc_slobo",
            "group": "multi_channel",
            "name":  "MultiChannel / SLOBO (all channels)",
            "command": [python_exe, "-u", str(train_script), "--cv_mode", "slobo",
                        "--model_type", "multi_channel", "--active_channels", "all"],
            "log":   out_dir / "mc_slobo.log",
        },
        {
            "key":   "mc_stlobo",
            "group": "multi_channel",
            "name":  "MultiChannel / ST-LOBO (all channels)",
            "command": [python_exe, "-u", str(train_script), "--cv_mode", "st_lobo",
                        "--model_type", "multi_channel", "--active_channels", "all"],
            "log":   out_dir / "mc_stlobo.log",
        },
        {
            "key":   "mc_withholding",
            "group": "multi_channel",
            "name":  "MultiChannel / Station Withholding (all channels)",
            "command": [python_exe, "-u", str(withholding_script),
                        "--model_type", "multi_channel", "--active_channels", "all"],
            "log":   out_dir / "mc_withholding.log",
        },
        # ── Channel ablation experiments (SLOBO) ─────────────────────────────
        {
            "key":   "ablate_terrain",
            "group": "ablation",
            "name":  "Ablation: Remove Terrain Channel",
            "command": [python_exe, "-u", str(train_script), "--cv_mode", "slobo",
                        "--model_type", "multi_channel",
                        "--active_channels", "temperature,humidity_stability,wind"],
            "log":   out_dir / "mc_ablate_terrain.log",
        },
        {
            "key":   "ablate_pressure",
            "group": "ablation",
            "name":  "Ablation: Remove Humidity/Stability Channel",
            "command": [python_exe, "-u", str(train_script), "--cv_mode", "slobo",
                        "--model_type", "multi_channel",
                        "--active_channels", "temperature,wind,terrain"],
            "log":   out_dir / "mc_ablate_pressure.log",
        },
        {
            "key":   "ablate_temperature",
            "group": "ablation",
            "name":  "Ablation: Remove Temperature Channel",
            "command": [python_exe, "-u", str(train_script), "--cv_mode", "slobo",
                        "--model_type", "multi_channel",
                        "--active_channels", "humidity_stability,wind,terrain"],
            "log":   out_dir / "mc_ablate_temperature.log",
        },
    ]


def main():
    parser = argparse.ArgumentParser(
        description="ERA5 Offset — Full Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Run ALL experiments sequentially:\n"
            "    python run_all_experiments.py\n\n"
            "  Run a single experiment:\n"
            "    python run_all_experiments.py --experiment baseline_slobo\n"
            "    python run_all_experiments.py --experiment mc_slobo\n"
            "    python run_all_experiments.py --experiment ablate_terrain\n\n"
            "  Run an experiment group:\n"
            "    python run_all_experiments.py --group baseline\n"
            "    python run_all_experiments.py --group multi_channel\n"
            "    python run_all_experiments.py --group ablation\n\n"
            "  List all available experiment keys:\n"
            "    python run_all_experiments.py --list\n"
        ),
    )
    parser.add_argument("--experiment", "-e", type=str, default=None, metavar="KEY",
                        help="Run only this experiment key.")
    parser.add_argument("--group", "-g", type=str, default=None, metavar="GROUP",
                        help="Run all experiments in this group: baseline | multi_channel | ablation")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available experiment keys and exit.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    out_dir  = root_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    python_exe  = sys.executable
    experiments = build_experiments(root_dir, python_exe, out_dir)

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        groups = {}
        for exp in experiments:
            groups.setdefault(exp["group"], []).append(exp)
        for group_name, exps in groups.items():
            print(f"\n  Group: {group_name}")
            for exp in exps:
                print(f"    {exp['key']:<25}  →  {exp['name']}")
                print(f"    {'':25}     log: {exp['log'].name}")
        return

    # ── --group ───────────────────────────────────────────────────────────────
    if args.group is not None:
        group_key = args.group.strip().lower()
        selected  = [e for e in experiments if e["group"] == group_key]
        if not selected:
            valid = sorted({e["group"] for e in experiments})
            print(f"[ERROR] Unknown group '{group_key}'. Valid groups: {valid}")
            sys.exit(1)
        experiments = selected

    # ── --experiment ──────────────────────────────────────────────────────────
    elif args.experiment is not None:
        key   = args.experiment.strip().lower()
        match = [e for e in experiments if e["key"] == key]
        if not match:
            valid = [e["key"] for e in experiments]
            print(f"[ERROR] Unknown experiment key '{key}'. Valid keys: {valid}")
            sys.exit(1)
        experiments = match

    # ── Run selected experiments ───────────────────────────────────────────────
    progress_message("Starting ERA5 Offset Experiment Runner")
    progress_message(f"Project root: {root_dir}")
    progress_message(f"Running {len(experiments)} experiment(s)")

    experiment_bar = tqdm(
        enumerate(experiments, start=1),
        total=len(experiments),
        desc="Experiments",
        unit="exp",
        leave=True,
        dynamic_ncols=True,
        disable=not PROGRESS_ENABLED,
    )

    for i, exp in experiment_bar:
        progress_message(f"\n--- Experiment {i}/{len(experiments)}: {exp['name']} ---")
        rc = run_command(exp["command"], exp["log"], cwd=root_dir)
        if rc != 0:
            progress_message(f"\n[CRITICAL] Experiment '{exp['name']}' failed. Halting.")
            sys.exit(1)
        if PROGRESS_ENABLED:
            experiment_bar.set_postfix(current=exp["key"])

    progress_message("\n" + "="*70)
    progress_message(f"  {len(experiments)} EXPERIMENT(S) COMPLETED SUCCESSFULLY!")
    progress_message(f"  Logs are saved in: {out_dir}/")
    progress_message("="*70)


if __name__ == "__main__":
    main()
