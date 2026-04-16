"""
plot_results.py — Parse experiment logs and generate plots.

Individual plots:
  1. Random Ablation  — Training curve per fold (Val MAE Tmax + Train Loss)
  2. SLOBO Ablation   — Training curve per fold
  3. ST-LOBO Ablation — Training curve per fold (val MAE Tmax)
  4. Station Withholding — Training curves per m-value + degradation curve

Comparison plots:
  5. Median Val MAE Tmax across the three CV methods (bar + individual fold dots)
  6. Baseline vs Best Val MAE per CV method
  7. ST-LOBO: Test MAE per fold checkpoint (bar chart)
  8. Withholding: Seen vs Unseen MAE degradation curve

Run:
    python plot_results.py

Outputs go to outputs/plots/
"""

import re
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
OUTPUTS     = ROOT / "outputs"
PLOT_DIR    = OUTPUTS / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LOG_RANDOM      = OUTPUTS / "baseline_random.log"
LOG_SLOBO       = OUTPUTS / "mc_slobo.log"
LOG_ST_LOBO     = OUTPUTS / "mc_stlobo.log"
LOG_WITHHOLDING = OUTPUTS / "mc_withholding.log"
LOG_ABLATE_TERRAIN = OUTPUTS / "mc_ablate_terrain.log"
LOG_ABLATE_PRESSURE = OUTPUTS / "mc_ablate_pressure.log"
LOG_ABLATE_TEMPERATURE = OUTPUTS / "mc_ablate_temperature.log"

# ─────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.85,
})


# ─────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────

def parse_ablation_log(path: Path):
    """
    Parse random / slobo / st_lobo logs.
    Returns:
        folds: dict  fold_name → {epochs, train_loss, val_mae_tmax, val_mae_tmin, baseline_tmax}
        summary: dict  key → (median, std)
    """
    RE_FOLD    = re.compile(r"Fold (.+)")
    RE_EPOCH   = re.compile(
        r"Epoch\s+(\d+)/\d+\s*\|"
        r"\s*Train Loss:\s*([\d.]+)\s*\|"
        r"\s*Val MAE Tmax:\s*([\d.]+)\s*\|"
        r"\s*Val MAE Tmin:\s*([\d.]+)\s*\|"
        r"\s*Baseline Tmax:\s*([\d.]+)"
    )
    RE_SUMMARY = re.compile(r"(val_mae_tmax|val_mae_tmin|baseline_mae_tmax|baseline_mae_tmin)"
                             r"\s*:\s*([\d.]+)\s*\(std=([\d.]+)\)")

    folds   = {}
    summary = {}
    current = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = RE_FOLD.search(line)
            if m:
                current = m.group(1).strip()
                folds[current] = dict(epochs=[], train_loss=[],
                                      val_mae_tmax=[], val_mae_tmin=[],
                                      baseline_tmax=[])
                continue

            m = RE_EPOCH.search(line)
            if m and current:
                folds[current]["epochs"].append(int(m.group(1)))
                folds[current]["train_loss"].append(float(m.group(2)))
                folds[current]["val_mae_tmax"].append(float(m.group(3)))
                folds[current]["val_mae_tmin"].append(float(m.group(4)))
                folds[current]["baseline_tmax"].append(float(m.group(5)))
                continue

            m = RE_SUMMARY.search(line)
            if m:
                summary[m.group(1)] = (float(m.group(2)), float(m.group(3)))

    return folds, summary


def parse_stlobo_test_mae(path: Path):
    """
    Also parse the 'Fold s=X, t=Y -> Test MAE Tmax: Z' lines from the ST-LOBO log.
    Returns: list of (fold_name, test_mae_tmax)
    """
    RE_TEST = re.compile(r"Fold (.+?)\s*->\s*Test MAE Tmax:\s*([\d.]+)")
    final = []
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = RE_TEST.search(line)
            if m:
                final.append((m.group(1).strip(), float(m.group(2))))
    return final


def parse_withholding_log(path: Path):
    """
    Parse withholding log.
    Returns:
        curves: dict  m → {epochs, loss, monitor_mae, unseen_mae}
        final:  list of dicts  {m, seen_mae, unseen_mae, unseen_std}
    """
    RE_EXP    = re.compile(r"Experiment: Withholding m=(\d+)")
    RE_CURVE  = re.compile(
        r"Epoch\s+(\d+):\s*\[avg_train_loss=([\d.]+),\s*avg_seen_mae=([\d.]+)(?:,\s*avg_unseen_mae=([\d.]+))?\]"
    )
    RE_SEEN    = re.compile(r"^\s*SEEN\s+mean MAE Tmax:\s*([\d.]+)")
    RE_UNSEEN  = re.compile(r"^\s*UNSEEN\s+mean MAE Tmax:\s*([\d.]+)[^\d]+([\d.]+)")

    curves  = {}
    final   = []
    cur_m   = None
    in_curves_block = False

    with open(path, encoding="utf-8") as f:
        for line in f:
            if "[AGGREGATED_CURVES_START]" in line:
                in_curves_block = True
                continue
            if "[AGGREGATED_CURVES_END]" in line:
                in_curves_block = False
                continue

            m_match = RE_EXP.search(line)
            if m_match:
                cur_m = int(m_match.group(1))
                curves[cur_m] = dict(epochs=[], loss=[], monitor_mae=[], unseen_mae=[])
                continue

            if in_curves_block and cur_m is not None:
                c_match = RE_CURVE.search(line)
                if c_match:
                    curves[cur_m]["epochs"].append(int(c_match.group(1)))
                    curves[cur_m]["loss"].append(float(c_match.group(2)))
                    seen_mae = float(c_match.group(3))
                    curves[cur_m]["monitor_mae"].append(seen_mae)
                    
                    if c_match.group(4):
                        curves[cur_m]["unseen_mae"].append(float(c_match.group(4)))
                    else:
                        curves[cur_m]["unseen_mae"].append(float("nan"))
                continue

            # Parse Final Aggregated Results
            s_match = RE_SEEN.search(line)
            if s_match and cur_m is not None:
                # Store it temporarily, wait for UNSEEN to append to final if m > 0
                if cur_m == 0:
                     final.append({"m": 0, "seen_mae": float(s_match.group(1)), "unseen_mae": float("nan"), "unseen_std": 0.0})
                else:
                    seen_mae_tmp = float(s_match.group(1))
                continue

            u_match = RE_UNSEEN.search(line)
            if u_match and cur_m is not None and cur_m > 0:
                final.append({
                    "m": cur_m, 
                    "seen_mae": seen_mae_tmp, 
                    "unseen_mae": float(u_match.group(1)),
                    "unseen_std": float(u_match.group(2))
                })
                # Do NOT set cur_m = None here because curves come after this block
                continue

    final.sort(key=lambda x: x["m"])
    return curves, final


# ─────────────────────────────────────────────────
# Individual plots — Ablation training curves
# ─────────────────────────────────────────────────

def plot_ablation(folds, summary, title, out_path):
    """Three-panel: (left) Val MAE Tmax, (mid) Val MAE Tmin, (right) Train Loss."""
    fold_names = list(folds.keys())
    n = len(fold_names)
    cmap = matplotlib.colormaps["tab20" if n > 8 else "tab10"].resampled(max(n, 1))
    colors = [cmap(i) for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    ax_tmax, ax_tmin, ax_loss = axes

    for i, fname in enumerate(fold_names):
        fd   = folds[fname]
        eps  = fd["epochs"]
        c    = colors[i]
        lbl  = f"Fold {fname}"

        ax_tmax.plot(eps, fd["val_mae_tmax"], color=c, linewidth=1.6, label=lbl)
        if fd["baseline_tmax"]:
            ax_tmax.axhline(fd["baseline_tmax"][0], color=c, linestyle="--", alpha=0.4, linewidth=1.0)
            
        ax_tmin.plot(eps, fd["val_mae_tmin"], color=c, linewidth=1.6, label=lbl)

        ax_loss.plot(eps, fd["train_loss"], color=c, linewidth=1.6, label=lbl)

    # Annotate medians
    if "val_mae_tmax" in summary:
        med, _ = summary["val_mae_tmax"]
        ax_tmax.axhline(med, color="black", linestyle=":", linewidth=1.4, label=f"Median {med:.4f}")
    if "val_mae_tmin" in summary:
        med, _ = summary["val_mae_tmin"]
        ax_tmin.axhline(med, color="black", linestyle=":", linewidth=1.4, label=f"Median {med:.4f}")

    ax_tmax.set_xlabel("Epoch")
    ax_tmax.set_ylabel("Val MAE Tmax (°C)")
    ax_tmax.set_title("Validation MAE — Tmax")
    ax_tmax.legend(fontsize=8, ncol=2)
    
    ax_tmin.set_xlabel("Epoch")
    ax_tmin.set_ylabel("Val MAE Tmin (°C)")
    ax_tmin.set_title("Validation MAE — Tmin")
    ax_tmin.legend(fontsize=8, ncol=2)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_channel_ablation_comparison(full_summary, terrain_summary, pressure_summary, temperature_summary, out_path):
    """Compare full multi-channel SLOBO against the three channel-ablation variants."""
    labels = [
        "Full MC",
        "No Terrain",
        "No Pressure",
        "No Temperature",
    ]
    summaries = [
        full_summary,
        terrain_summary,
        pressure_summary,
        temperature_summary,
    ]

    tmax_vals = [s["val_mae_tmax"][0] for s in summaries]
    tmax_stds = [s["val_mae_tmax"][1] for s in summaries]
    tmin_vals = [s["val_mae_tmin"][0] for s in summaries]
    tmin_stds = [s["val_mae_tmin"][1] for s in summaries]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Channel Ablation Comparison (SLOBO)", fontsize=13, fontweight="bold", y=1.02)

    colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]

    for ax, vals, stds, metric_label in (
        (axes[0], tmax_vals, tmax_stds, "Validation MAE Tmax"),
        (axes[1], tmin_vals, tmin_stds, "Validation MAE Tmin"),
    ):
        bars = ax.bar(x, vals, yerr=stds, capsize=4, color=colors, edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("MAE (°C)")
        ax.set_title(metric_label)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────
# Individual plot — ST-LOBO test MAE bar
# ─────────────────────────────────────────────────

def plot_stlobo_test(test_results, summary, out_path):
    """Bar chart of per-fold Test MAE Tmax from ST-LOBO cross-evaluation."""
    fold_names = [r[0] for r in test_results]
    test_maes  = [r[1] for r in test_results]

    best_idx = int(np.argmin(test_maes))
    colors = [PALETTE[0]] * len(fold_names)
    colors[best_idx] = PALETTE[2]  # highlight best fold

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(fold_names, test_maes, color=colors, edgecolor="white", width=0.5)

    # Annotate values
    for bar, val in zip(bars, test_maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("ST-LOBO Fold", fontsize=11)
    ax.set_ylabel("Test MAE Tmax (°C)", fontsize=11)
    ax.set_title("ST-LOBO: Test-Set MAE Tmax per Fold Checkpoint\n"
                 "(green = best fold selected for production)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(min(test_maes) * 0.97, max(test_maes) * 1.02)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────
# Individual plot — Withholding training curves
# ─────────────────────────────────────────────────

def plot_withholding_curves(curves, out_path):
    """Training curves (monitor MAE) for each m-value."""
    m_values = sorted(curves.keys())
    colors   = PALETTE[:len(m_values)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Station Withholding — Training Curves per m", fontsize=13,
                 fontweight="bold", y=1.01)

    ax_mae, ax_loss = axes

    for i, m in enumerate(m_values):
        c  = colors[i]
        fd = curves[m]
        lbl = f"m={m}"

        ax_mae.plot(fd["epochs"], fd["monitor_mae"],
                    color=c, linewidth=1.6, label=lbl)
        ax_loss.plot(fd["epochs"], fd["loss"],
                     color=c, linewidth=1.6, label=lbl)

    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("Monitor MAE Tmax (°C)")
    ax_mae.set_title("Monitor MAE vs Epoch")
    ax_mae.legend(fontsize=9)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Training Loss vs Epoch")
    ax_loss.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────
# Individual plot — Withholding degradation curve
# ─────────────────────────────────────────────────

def plot_withholding_degradation(final, out_path):
    """Seen vs Unseen MAE vs m — the spatial generalisation degradation curve."""
    ms         = [r["m"] for r in final]
    seen_maes  = [r["seen_mae"]   for r in final]
    unseen_maes= [r.get("unseen_mae", float('nan')) for r in final]
    unseen_stds= [r.get("unseen_std", 0.0) for r in final]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ms, seen_maes, "o-",  color=PALETTE[0], linewidth=2,
            markersize=7, label="Seen (train) MAE Tmax")

    # Only plot unseen where we have real values
    valid_idxs = [i for i, u in enumerate(unseen_maes) if not np.isnan(u)]
    if valid_idxs:
        vms = [ms[i] for i in valid_idxs]
        vus = [unseen_maes[i] for i in valid_idxs]
        vstds = [unseen_stds[i] for i in valid_idxs]
        
        ax.errorbar(vms, vus, yerr=vstds, fmt="s--", color=PALETTE[1], linewidth=2,
                markersize=7, capsize=4, capthick=1.5, label="Unseen (val) MAE Tmax ± std")

    ax.set_xlabel("Stations withheld (m)", fontsize=11)
    ax.set_ylabel("MAE Tmax (°C)", fontsize=11)
    ax.set_title("Spatial Generalisation Degradation Curve\n(Averaged over repeated samples per m)", fontsize=12,
                 fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────
# Comparison plot 1 — Median Val MAE across CV modes
# ─────────────────────────────────────────────────

def plot_cv_comparison(random_folds, random_summary,
                       slobo_folds,  slobo_summary,
                       stlobo_folds, stlobo_summary,
                       out_path):
    """
    Bar chart comparing median Val MAE Tmax/Tmin + baseline across methods.
    Overlays individual fold dots.
    """
    methods = ["Random", "SLOBO", "ST-LOBO"]
    summaries = [random_summary, slobo_summary, stlobo_summary]
    all_folds = [random_folds,  slobo_folds,   stlobo_folds]

    metrics = ["val_mae_tmax", "val_mae_tmin", "baseline_mae_tmax"]
    labels  = ["Val MAE Tmax", "Val MAE Tmin", "Baseline Tmax"]
    cols    = [PALETTE[0], PALETTE[1], PALETTE[4]]

    x   = np.arange(len(methods))
    w   = 0.22
    off = [-w, 0, w]

    fig, ax = plt.subplots(figsize=(10, 6))

    for j, (metric, label, col) in enumerate(zip(metrics, labels, cols)):
        medians = [s[metric][0] if metric in s else float("nan") for s in summaries]
        stds    = [s[metric][1] if metric in s else float("nan") for s in summaries]
        bars = ax.bar(x + off[j], medians, w, label=label,
                      color=col, alpha=0.85, edgecolor="white",
                      yerr=stds, capsize=4, error_kw={"elinewidth": 1.2})

        # Overlay individual fold dots for val metrics only
        if metric.startswith("val"):
            fold_key = "val_mae_tmax" if "tmax" in metric else "val_mae_tmin"
            for i, folds in enumerate(all_folds):
                fold_bests = [min(fd[fold_key]) for fd in folds.values() if fd[fold_key]]
                jitter = np.random.uniform(-0.04, 0.04, len(fold_bests))
                ax.scatter(x[i] + off[j] + jitter, fold_bests,
                           color="white", edgecolors=col, s=30,
                           linewidths=1.2, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel("MAE (°C)", fontsize=11)
    ax.set_title("Cross-Validation Methods: Median MAE Comparison\n"
                 "(bars = median ± std, dots = individual fold bests)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(bottom=1.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────
# Comparison plot 2 — Best fold val MAE per CV mode
# ─────────────────────────────────────────────────

def plot_best_fold_per_method(random_folds, slobo_folds, stlobo_folds, wh_curves, wh_final, out_path):
    """
    For each method: training curve and val curve of the best fold/m-value.
    Three panels: Left = Val MAE Tmax, Mid = Val MAE Tmin, Right = Train Loss.
    """
    data = {"Random": random_folds, "SLOBO": slobo_folds, "ST-LOBO": stlobo_folds}
    keys = ["Random", "SLOBO", "ST-LOBO", "Withholding"]
    colors = {k: c for k, c in zip(keys, PALETTE)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Best Fold/Seed Curves Comparison Across Validation Strategies",
                 fontsize=14, fontweight="bold", y=1.05)
    
    # Subtitle proving identical folds
    fig.text(0.5, 0.96, "(SLOBO and ST-LOBO use Identical Spatial Folds via shared Seed=42)", 
             ha="center", fontsize=10, style='italic')

    ax_tmax, ax_tmin, ax_loss = axes

    for method, folds in data.items():
        valid_folds = {k: v for k, v in folds.items() if v["val_mae_tmax"]}
        if not valid_folds:
            continue
        best_fold = min(valid_folds, key=lambda k: min(valid_folds[k]["val_mae_tmax"]))
        fd = valid_folds[best_fold]
        
        lbl = f"{method} (fold {best_fold}, best={min(fd['val_mae_tmax']):.4f})"
        ax_tmax.plot(fd["epochs"], fd["val_mae_tmax"], color=colors[method], linewidth=2.0, label=lbl)
        
        lbl_tmin = f"{method} (fold {best_fold}, best={min(fd['val_mae_tmin']):.4f})"
        ax_tmin.plot(fd["epochs"], fd["val_mae_tmin"], color=colors[method], linewidth=2.0, label=lbl_tmin)
        
        ax_loss.plot(fd["epochs"], fd["train_loss"], color=colors[method], linewidth=2.0, label=lbl)

    # Add Withholding
    if wh_final and wh_curves:
        valid_wh = [r for r in wh_final if not np.isnan(r["unseen_mae"]) and r["m"] > 0]
        if valid_wh:
            best_wh_run = min(valid_wh, key=lambda r: r["unseen_mae"])
            best_m = best_wh_run["m"]
            best_val = best_wh_run["unseen_mae"]
            if best_m in wh_curves:
                fd = wh_curves[best_m]
                lbl = f"Withholding (m={best_m}, best={best_val:.4f})"
                ax_tmax.plot(fd["epochs"], fd["unseen_mae"], color=colors["Withholding"], linewidth=2.0, label=lbl)
                ax_loss.plot(fd["epochs"], fd["loss"], color=colors["Withholding"], linewidth=2.0, label=lbl)

    ax_tmax.set_xlabel("Epoch", fontsize=11)
    ax_tmax.set_ylabel("Val MAE Tmax (°C)", fontsize=11)
    ax_tmax.set_title("Validation MAE Tmax", fontsize=12)
    ax_tmax.legend(fontsize=9)
    
    ax_tmin.set_xlabel("Epoch", fontsize=11)
    ax_tmin.set_ylabel("Val MAE Tmin (°C)", fontsize=11)
    ax_tmin.set_title("Validation MAE Tmin (N/A for Withholding)", fontsize=12)
    ax_tmin.legend(fontsize=9)

    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Train Loss", fontsize=11)
    ax_loss.set_title("Training Loss", fontsize=12)
    ax_loss.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")





# ─────────────────────────────────────────────────
# Comparison plot 3 — All folds of all methods
# ─────────────────────────────────────────────────

def plot_all_folds_overlay(random_folds, slobo_folds, stlobo_folds, out_path):
    """
    Overlay all fold curves for all three methods in a 2×3 grid (Top: Tmax, Bottom: Tmin).
    """
    data_list = [
        ("Random Ablation",  random_folds),
        ("SLOBO Ablation",   slobo_folds),
        ("ST-LOBO Ablation", stlobo_folds),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
    fig.suptitle("Val MAE — All Folds by Method", fontsize=15,
                 fontweight="bold", y=1.02)
                 
    fig.text(0.5, 0.98, "(SLOBO and ST-LOBO use Identical Spatial Folds via shared Seed=42)", 
             ha="center", fontsize=11, style='italic')

    for col, (name, folds) in enumerate(data_list):
        ax_tmax = axes[0, col]
        ax_tmin = axes[1, col]
        
        valid_folds = {k: v for k, v in folds.items() if v["val_mae_tmax"]}
        n = len(valid_folds)
        cmap_name = "tab20" if n > 8 else "tab10"
        cmap = matplotlib.colormaps[cmap_name].resampled(max(n, 1))
        shades = [cmap(i) for i in range(n)]
        
        for i, (fname, fd) in enumerate(valid_folds.items()):
            ax_tmax.plot(fd["epochs"], fd["val_mae_tmax"], color=shades[i], linewidth=1.5, label=f"Fold {fname}")
            ax_tmin.plot(fd["epochs"], fd["val_mae_tmin"], color=shades[i], linewidth=1.5, label=f"Fold {fname}")
            
        ax_tmax.set_title(f"{name} (Tmax)", fontsize=12, fontweight="bold")
        ax_tmax.set_xlabel("Epoch")
        ax_tmax.set_ylabel("Val MAE Tmax (°C)")
        ax_tmax.legend(fontsize=7, ncol=2)
        
        ax_tmin.set_title(f"{name} (Tmin)", fontsize=12, fontweight="bold")
        ax_tmin.set_xlabel("Epoch")
        ax_tmin.set_ylabel("Val MAE Tmin (°C)")
        ax_tmin.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────
# Comparison plot 4 — Best Overall Val MAE Tmax Bar Chart
# ─────────────────────────────────────────────────

def plot_best_overall_bar_chart(random_folds, slobo_folds, stlobo_folds, wh_final, out_path):
    """
    Bar chart comparing the absolute BEST Val MAE Tmax across the four strategies.
    Only Tmax is shown because Withholding does not log Tmin.
    """
    methods = ["Random\n(Best)", "SLOBO\n(Best)", "ST-LOBO\n(Best)", "Withholding\n(Best m)"]
    
    def get_best(folds):
        valid = [min(f["val_mae_tmax"]) for f in folds.values() if f["val_mae_tmax"]]
        return min(valid) if valid else float("nan")
        
    best_random = get_best(random_folds)
    best_slobo = get_best(slobo_folds)
    best_stlobo = get_best(stlobo_folds)
    
    best_wh = float("nan")
    if wh_final:
        valid_wh = [r["unseen_mae"] for r in wh_final if not np.isnan(r["unseen_mae"]) and r["m"] > 0]
        if valid_wh:
            best_wh = min(valid_wh)
            
    bests = [best_random, best_slobo, best_stlobo, best_wh]
    cols = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(methods, bests, color=cols, alpha=0.85, edgecolor="white", width=0.6)
    
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        if not np.isnan(yval):
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
            
    ax.set_ylabel("Best Validation MAE Tmax (°C)", fontsize=11)
    ax.set_title("Absolute Best Validation MAE across Four Evaluation Strategies", 
                 fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=1.2)
    
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    print("Parsing logs...")
    random_folds, random_summary   = parse_ablation_log(LOG_RANDOM)
    slobo_folds,  slobo_summary    = parse_ablation_log(LOG_SLOBO)
    stlobo_folds, stlobo_summary   = parse_ablation_log(LOG_ST_LOBO)
    ablate_terrain_folds, ablate_terrain_summary = parse_ablation_log(LOG_ABLATE_TERRAIN)
    ablate_pressure_folds, ablate_pressure_summary = parse_ablation_log(LOG_ABLATE_PRESSURE)
    ablate_temperature_folds, ablate_temperature_summary = parse_ablation_log(LOG_ABLATE_TEMPERATURE)
    stlobo_test                    = parse_stlobo_test_mae(LOG_ST_LOBO)
    wh_curves,    wh_final         = parse_withholding_log(LOG_WITHHOLDING)

    print(f"\nFolds parsed  →  Random: {len(random_folds)}, "
          f"SLOBO: {len(slobo_folds)}, ST-LOBO: {len(stlobo_folds)}")
    print(f"Withholding m-values: {sorted(wh_curves.keys())}")
    print(f"\nGenerating plots → {PLOT_DIR}\n")

    # ── Individual plots ──────────────────────────
    plot_ablation(random_folds, random_summary,
                  "Random Ablation — Training Curves",
                  PLOT_DIR / "01_random_ablation.png")

    plot_ablation(slobo_folds, slobo_summary,
                  "SLOBO Ablation — Training Curves",
                  PLOT_DIR / "02_slobo_ablation.png")

    plot_ablation(stlobo_folds, stlobo_summary,
                  "ST-LOBO Ablation — Training Curves (Val)",
                  PLOT_DIR / "03_stlobo_ablation.png")

    plot_ablation(ablate_terrain_folds, ablate_terrain_summary,
                  "Channel Ablation — Remove Terrain",
                  PLOT_DIR / "11_ablate_terrain.png")

    plot_ablation(ablate_pressure_folds, ablate_pressure_summary,
                  "Channel Ablation — Remove Pressure",
                  PLOT_DIR / "12_ablate_pressure.png")

    plot_ablation(ablate_temperature_folds, ablate_temperature_summary,
                  "Channel Ablation — Remove Temperature",
                  PLOT_DIR / "13_ablate_temperature.png")

    plot_channel_ablation_comparison(
        slobo_summary,
        ablate_terrain_summary,
        ablate_pressure_summary,
        ablate_temperature_summary,
        PLOT_DIR / "14_channel_ablation_comparison.png",
    )

    if stlobo_test:
        plot_stlobo_test(stlobo_test, stlobo_summary,
                         PLOT_DIR / "04_stlobo_test_mae.png")

    plot_withholding_curves(wh_curves,
                            PLOT_DIR / "05_withholding_training_curves.png")

    if wh_final:
        plot_withholding_degradation(wh_final,
                                     PLOT_DIR / "06_withholding_degradation.png")

    # ── Comparison plots ──────────────────────────
    plot_cv_comparison(random_folds, random_summary,
                       slobo_folds,  slobo_summary,
                       stlobo_folds, stlobo_summary,
                       PLOT_DIR / "07_cv_method_comparison.png")

    plot_best_fold_per_method(random_folds, slobo_folds, stlobo_folds, wh_curves, wh_final,
                              PLOT_DIR / "08_best_fold_curves.png")

    plot_all_folds_overlay(random_folds, slobo_folds, stlobo_folds,
                           PLOT_DIR / "09_all_folds_overlay.png")

    plot_best_overall_bar_chart(random_folds, slobo_folds, stlobo_folds, wh_final,
                                PLOT_DIR / "10_best_overall_bar.png")

    print(f"\nDone. {len(list(PLOT_DIR.glob('*.png')))} plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
