from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT

DEFAULT_INPUT_DIR = DOWNSTREAM_OUTPUT_ROOT / "figure_g_gene_perturbation_screen"
FIGURE_DIR = DEFAULT_INPUT_DIR
VOLCANO_CSV = DEFAULT_INPUT_DIR / "figure_g_gene_perturbation_fate_volcano_summary.csv"
TRAJECTORY_CSV = DEFAULT_INPUT_DIR / "figure_g_top15_stacked_trajectory_fractions.csv"
SELECTED_GENES_CSV = DEFAULT_INPUT_DIR / "figure_g_top15_selected_genes.csv"

FATE_COLORS = {
    "M": "#64B6AC",
    "EN": "#E55761",
    "Others": "#ECECEC",
}
UP_COLOR = "#E64B5D"
DOWN_COLOR = "#2E3D76"
NEUTRAL_COLOR = "#B8B8B8"
FDR_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw Figure G from perturbation-screen CSV files.")
    parser.add_argument("--volcano-csv", type=Path, default=VOLCANO_CSV)
    parser.add_argument("--trajectory-csv", type=Path, default=TRAJECTORY_CSV)
    parser.add_argument("--selected-genes-csv", type=Path, default=SELECTED_GENES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT_DIR)
    return parser


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _label_extreme_genes(ax: plt.Axes, df: pd.DataFrame, n_per_side: int = 7) -> None:
    if df.empty:
        return
    up = df[df["direction"] == "up"].nlargest(n_per_side, "minus_log10_p")
    down = df[df["direction"] == "down"].nlargest(n_per_side, "minus_log10_p")
    for _, row in pd.concat([up, down]).iterrows():
        color = UP_COLOR if row["direction"] == "up" else DOWN_COLOR
        ax.annotate(
            str(row["gene"]),
            (float(row["log2_fold_change"]), float(row["minus_log10_p"])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            color=color,
        )


def plot_volcano(fates: list[str], output_name: str) -> None:
    df = pd.read_csv(VOLCANO_CSV)
    fig, axes = plt.subplots(1, len(fates), figsize=(3.0 * len(fates), 2.75), squeeze=False)

    for ax, fate in zip(axes.ravel(), fates):
        sub = df[df["fate"] == fate].copy()
        sig = (sub["q_value_bh"] < FDR_THRESHOLD) & (sub["log2_fold_change"].abs() >= LOG2FC_THRESHOLD)
        up = sig & (sub["log2_fold_change"] > 0)
        down = sig & (sub["log2_fold_change"] < 0)

        ax.scatter(
            sub.loc[~sig, "log2_fold_change"],
            sub.loc[~sig, "minus_log10_p"],
            s=10,
            color=NEUTRAL_COLOR,
            alpha=0.55,
            linewidths=0,
        )
        ax.scatter(
            sub.loc[down, "log2_fold_change"],
            sub.loc[down, "minus_log10_p"],
            s=13,
            color=DOWN_COLOR,
            alpha=0.88,
            linewidths=0,
        )
        ax.scatter(
            sub.loc[up, "log2_fold_change"],
            sub.loc[up, "minus_log10_p"],
            s=13,
            color=UP_COLOR,
            alpha=0.88,
            linewidths=0,
        )
        ax.axvline(0.0, color="#707070", linewidth=0.7, linestyle="--", alpha=0.7)
        ax.axhline(-np.log10(FDR_THRESHOLD), color="#707070", linewidth=0.7, linestyle=":", alpha=0.7)
        ax.set_title(f"{fate} fate", fontsize=10)
        ax.set_xlabel("log2(fold-change)")
        ax.set_ylabel("-log10(P)")
        ax.grid(True, color="#EEEEEE", linewidth=0.5)
        _label_extreme_genes(ax, sub[sig])

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURE_DIR / f"{output_name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _condition_order() -> list[str]:
    selected = pd.read_csv(SELECTED_GENES_CSV)
    genes = [str(gene) for gene in selected["gene"].tolist()]
    return ["Unperturbed", *[f"{gene} z+10" for gene in genes]]


def _plot_stacked_panel(ax: plt.Axes, traj: pd.DataFrame, condition: str) -> None:
    sub = traj[traj["condition"] == condition].sort_values("time")
    time = sub["time"].to_numpy(dtype=float)
    y_m = 100.0 * sub["M"].to_numpy(dtype=float)
    y_en = 100.0 * sub["EN"].to_numpy(dtype=float)
    y_other = 100.0 * sub["Others"].to_numpy(dtype=float)
    ax.stackplot(
        time,
        y_m,
        y_en,
        y_other,
        colors=[FATE_COLORS["M"], FATE_COLORS["EN"], FATE_COLORS["Others"]],
        linewidth=0,
        alpha=0.98,
    )
    ax.set_xlim(2.0, 5.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_yticks([0, 50, 100])
    ax.grid(True, color="#E7E7E7", linewidth=0.45)
    ax.set_title(condition, fontsize=8.5, pad=3)


def plot_top15_trajectories() -> None:
    traj = pd.read_csv(TRAJECTORY_CSV)
    order = [condition for condition in _condition_order() if condition in set(traj["condition"])]
    fig, axes = plt.subplots(4, 4, figsize=(8.0, 6.2), sharex=True, sharey=True)

    for ax, condition in zip(axes.ravel(), order):
        _plot_stacked_panel(ax, traj, condition)
    for ax in axes.ravel()[len(order) :]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (d)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fate (%)")

    handles = [
        plt.Line2D([0], [0], color=FATE_COLORS["M"], linewidth=6),
        plt.Line2D([0], [0], color=FATE_COLORS["EN"], linewidth=6),
        plt.Line2D([0], [0], color=FATE_COLORS["Others"], linewidth=6),
    ]
    fig.legend(handles, ["M", "EN", "Others"], loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURE_DIR / f"figure_g_top15_gene_z10_4x4_trajectories.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for condition in order:
        fig, ax = plt.subplots(figsize=(1.85, 1.45))
        _plot_stacked_panel(ax, traj, condition)
        ax.set_xlabel("Time (d)", fontsize=7)
        ax.set_ylabel("Fate (%)", fontsize=7)
        fig.tight_layout()
        safe_name = condition.lower().replace(" ", "_").replace("+", "plus")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIGURE_DIR / f"figure_g_{safe_name}_trajectory.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    global FIGURE_DIR, VOLCANO_CSV, TRAJECTORY_CSV, SELECTED_GENES_CSV
    args = build_parser().parse_args()
    VOLCANO_CSV = args.volcano_csv.expanduser().resolve()
    TRAJECTORY_CSV = args.trajectory_csv.expanduser().resolve()
    SELECTED_GENES_CSV = args.selected_genes_csv.expanduser().resolve()
    FIGURE_DIR = args.output_dir.expanduser().resolve()
    missing = [path for path in (VOLCANO_CSV, TRAJECTORY_CSV, SELECTED_GENES_CSV) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Figure G input files: " + ", ".join(str(path) for path in missing))
    _setup_style()
    plot_volcano(["M", "EN"], "figure_g_volcano_m_en")
    plot_volcano(["M", "EN", "Others"], "figure_g_volcano_m_en_others")
    plot_top15_trajectories()


if __name__ == "__main__":
    main()
