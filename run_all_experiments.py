"""
run_all_experiments.py — Cross-platform orchestrator for the Ablation Study.

This script sequentially runs:
1. Random Split Ablation
2. SLOBO Ablation
3. ST-LOBO Ablation
4. Station Withholding Generalization Test

Outputs are captured into the `outputs/` directory. 
Warning: This script will take a long time to finish as it is running ~4 sets of models sequentially.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_command(command, log_file):
    """Run a command using subprocess and stream output to both console and log file."""
    print(f"\n{'='*70}")
    print(f"  EXECUTING: {' '.join(command)}")
    print(f"  LOGGING TO: {log_file}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Open the log file for writing
    with open(log_file, "w", encoding="utf-8") as f:
        # Popen executes the command, grabbing stdout/stderr together
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, # Line buffered
        )
        
        # Read line by line as it comes in
        for line in process.stdout:
            sys.stdout.write(line) # Print to console
            f.write(line)          # Write to file
            f.flush()
            
    # Wait for the command to finish completely
    process.wait()
    
    elapsed = time.time() - start_time
    if process.returncode == 0:
        print(f"\n[SUCCESS] Completed in {elapsed/60:.2f} minutes.")
    else:
        print(f"\n[FAILED] Exited with error code {process.returncode} after {elapsed/60:.2f} minutes.")
    
    return process.returncode

def main():
    # Setup paths
    root_dir = Path(__file__).parent.absolute()
    out_dir = root_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    
    # Use the python executable currently running this script to ensure environments match
    python_exe = sys.executable
    
    # Define the experiment queue
    experiments = [
        {
            "name": "Random Ablation",
            "command": [python_exe, "src/train.py", "--cv_mode", "random"],
            "log": out_dir / "random_ablation.log"
        },
        {
            "name": "SLOBO Ablation",
            "command": [python_exe, "src/train.py", "--cv_mode", "slobo"],
            "log": out_dir / "slobo_ablation.log"
        },
        {
            "name": "ST-LOBO Ablation",
            "command": [python_exe, "src/train.py", "--cv_mode", "st_lobo"],
            "log": out_dir / "st_lobo_ablation.log"
        },
        {
            "name": "Station Withholding Test",
            "command": [python_exe, "src/station_withholding_test.py"],
            "log": out_dir / "withholding_test.log"
        }
    ]
    
    print("Starting the ERA5 Offset Batch Experiment Runner...")
    print(f"Total experiments to run: {len(experiments)}")
    
    for i, exp in enumerate(experiments):
        print(f"\n--- Starting Experiment {i+1}/{len(experiments)}: {exp['name']} ---")
        return_code = run_command(exp["command"], exp["log"])
        
        if return_code != 0:
            print(f"\n[CRITICAL] Experiment {exp['name']} failed. Halting batch execution.")
            sys.exit(1)
            
    print("\n" + "="*70)
    print("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉")
    print(f"  Logs are saved in the {out_dir}/ directory.")
    print("="*70)

if __name__ == "__main__":
    main()
