from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT


DEFAULT_CANDIDATES = "1:0.3,1:0.5,1:0.7,1:1.0"
ACTION_COLOR = "#2F6DB3"
POTENTIAL_COLOR = "#D98C1F"
SUM_COLOR = "#B23A48"
CANDIDATE_COLORS = ["#B23A48", "#7A4EAB", "#2A9D8F", "#C06C2B", "#4B6584", "#D35400"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build downstream criticality panels from action and potential components."
    )
    parser.add_argument("--curve-csv", type=Path, required=True)
    parser.add_argument("--action-column", default=None)
    parser.add_argument("--potential-column", default=None)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, help="Comma-separated alpha:beta pairs.")
    parser.add_argument("--selected-alpha", type=float, default=None)
    parser.add_argument("--selected-beta", type=float, default=None)
    parser.add_argument("--label", default="dataset")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--output-dir", type=Path, default=DOWNSTREAM_OUTPUT_ROOT / "figure4_action_potential_criticality")
    return parser


def parse_candidates(text: str) -> list[dict[str, float | str]]:
    candidates: list[dict[str, float | str]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Candidate '{item}' must use alpha:beta format.")
        alpha_text, beta_text = item.split(":", 1)
        alpha = float(alpha_text)
        beta = float(beta_text)
        candidates.append({"alpha": alpha, "beta": beta, "label": f"alpha={alpha:g}, beta={beta:g}"})
    if not candidates:
        raise ValueError("At least one alpha:beta candidate is required.")
    return candidates


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return slug or "dataset"


def resolve_column(df: pd.DataFrame, explicit: str | None, candidates: list[str], label: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"{label} column '{explicit}' not found in {list(df.columns)}")
        return explicit
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not infer {label} column. Tried: {candidates}")


def peak_info(time: np.ndarray, values: np.ndarray) -> tuple[float, float, int]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan, np.nan, -1
    idx_candidates = np.where(finite)[0]
    idx_local = int(np.nanargmax(values[finite]))
    idx = int(idx_candidates[idx_local])
    return float(time[idx]), float(values[idx]), idx


def apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=10)


def save_single_curve(df: pd.DataFrame, column: str, color: str, title: str, ylabel: str, out_path: Path) -> None:
    time = df["time"].to_numpy(dtype=np.float64)
    values = df[column].to_numpy(dtype=np.float64)
    peak_time, peak_value, _ = peak_info(time, values)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=220)
    ax.plot(time, values, color=color, linewidth=2.5)
    ax.scatter([peak_time], [peak_value], s=54, color=color, edgecolor="white", linewidth=0.9, zorder=3)
    ax.axvline(peak_time, color=color, linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    curve_csv = args.curve_csv.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(curve_csv).sort_values("time").reset_index(drop=True)
    action_col = resolve_column(source_df, args.action_column, ["action", "Action_norm", "action_norm"], "action")
    potential_col = resolve_column(
        source_df,
        args.potential_column,
        ["potential", "Q_div_abs_mass_norm", "Q_div_abs_mass", "potential_norm"],
        "potential",
    )
    candidates = parse_candidates(args.candidates)
    selected_alpha = float(args.selected_alpha if args.selected_alpha is not None else candidates[0]["alpha"])
    selected_beta = float(args.selected_beta if args.selected_beta is not None else candidates[0]["beta"])

    label = str(args.label)
    prefix = slugify(args.output_prefix or label)
    df = pd.DataFrame(
        {
            "time": source_df["time"].to_numpy(dtype=np.float64),
            "action": source_df[action_col].to_numpy(dtype=np.float64),
            "potential": source_df[potential_col].to_numpy(dtype=np.float64),
        }
    )

    summary_rows = []
    for candidate in candidates:
        alpha = float(candidate["alpha"])
        beta = float(candidate["beta"])
        col = f"criticality_alpha{alpha:g}_beta{beta:g}"
        df[col] = alpha * df["action"] + beta * df["potential"]
        peak_time, peak_value, peak_idx = peak_info(
            df["time"].to_numpy(dtype=np.float64),
            df[col].to_numpy(dtype=np.float64),
        )
        action_component = float(alpha * df.loc[peak_idx, "action"]) if peak_idx >= 0 else np.nan
        potential_component = float(beta * df.loc[peak_idx, "potential"]) if peak_idx >= 0 else np.nan
        summary_rows.append(
            {
                "alpha": alpha,
                "beta": beta,
                "label": str(candidate["label"]),
                "score_column": col,
                "peak_time": peak_time,
                "peak_value": peak_value,
                "peak_action_component": action_component,
                "peak_potential_component": potential_component,
                "action_to_potential_ratio_at_peak": action_component / potential_component
                if np.isfinite(potential_component) and abs(potential_component) > 1e-12
                else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    selected_col = f"criticality_alpha{selected_alpha:g}_beta{selected_beta:g}"
    if selected_col not in df.columns:
        raise ValueError(f"Selected alpha/beta pair was not included in candidates: {selected_alpha:g}:{selected_beta:g}")
    selected_row = summary_df.loc[summary_df["score_column"] == selected_col].iloc[0]

    action_png = out_dir / f"{prefix}_action.png"
    potential_png = out_dir / f"{prefix}_potential.png"
    candidates_png = out_dir / f"{prefix}_additive_candidates.png"
    overlay_png = out_dir / f"{prefix}_action_potential_overlay.png"
    curves_csv = out_dir / f"{prefix}_action_potential_criticality_curves.csv"
    summary_csv = out_dir / f"{prefix}_action_potential_criticality_summary.csv"
    manifest_json = out_dir / f"{prefix}_action_potential_criticality_manifest.json"

    save_single_curve(df, "action", ACTION_COLOR, f"{label}: action", "action", action_png)
    save_single_curve(df, "potential", POTENTIAL_COLOR, f"{label}: potential", "potential", potential_png)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=220)
    for color, candidate in zip(CANDIDATE_COLORS, candidates):
        alpha = float(candidate["alpha"])
        beta = float(candidate["beta"])
        col = f"criticality_alpha{alpha:g}_beta{beta:g}"
        row = summary_df.loc[summary_df["score_column"] == col].iloc[0]
        ax.plot(df["time"], df[col], color=color, linewidth=2.25, label=str(candidate["label"]))
        ax.scatter([row["peak_time"]], [row["peak_value"]], s=42, color=color, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_title(f"{label}: alpha * action + beta * potential", fontsize=15, pad=10)
    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel("criticality", fontsize=12)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(candidates_png, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=220)
    ax.plot(df["time"], df["action"], color=ACTION_COLOR, linewidth=2.2, label="action")
    ax.plot(df["time"], df["potential"], color=POTENTIAL_COLOR, linewidth=2.2, label="potential")
    ax.plot(
        df["time"],
        df[selected_col],
        color=SUM_COLOR,
        linewidth=2.7,
        label=f"criticality ({selected_alpha:g} * action + {selected_beta:g} * potential)",
    )
    ax.scatter(
        [selected_row["peak_time"]],
        [selected_row["peak_value"]],
        s=54,
        color=SUM_COLOR,
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )
    ax.axvline(selected_row["peak_time"], color=SUM_COLOR, linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_title(f"{label}: selected action/potential criticality", fontsize=15, pad=10)
    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel("component value / weighted sum", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(overlay_png, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(curves_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    manifest = {
        "source_curve_csv": str(curve_csv),
        "action_column": action_col,
        "potential_column": potential_col,
        "definition": "criticality = alpha * action + beta * potential",
        "selected_alpha": selected_alpha,
        "selected_beta": selected_beta,
        "candidates": candidates,
        "outputs": {
            "action_png": str(action_png),
            "potential_png": str(potential_png),
            "candidates_png": str(candidates_png),
            "overlay_png": str(overlay_png),
            "curves_csv": str(curves_csv),
            "summary_csv": str(summary_csv),
            "manifest_json": str(manifest_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
