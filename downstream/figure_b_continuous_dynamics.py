from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT, PIUOT_FIG_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw the continuous dynamics panel from piuot/export_trajectory_points.py output.",
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=PIUOT_FIG_ROOT / "dataset" / "trajectory_points" / "dataset_trajectory_points.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DOWNSTREAM_OUTPUT_ROOT / "figure_b_continuous_dynamics")
    parser.add_argument("--prefix", default="figure_b")
    parser.add_argument("--max-observed-per-time", type=int, default=1200)
    parser.add_argument("--max-predicted-per-time", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _time_color(time_value: float, t_min: float, t_max: float):
    denom = max(float(t_max) - float(t_min), 1e-8)
    return plt.cm.viridis((float(time_value) - float(t_min)) / denom)


def _subsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=int(seed)).sort_index()


def _draw_time_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    kind: str,
    marker: str,
    size: float,
    alpha: float,
    t_min: float,
    t_max: float,
    max_per_time: int,
    seed: int,
) -> None:
    sub = df[df["kind"] == kind]
    for idx, time_value in enumerate(sorted(sub["time"].unique())):
        part = _subsample(sub[np.isclose(sub["time"], float(time_value))], max_per_time, seed + idx)
        if part.empty:
            continue
        ax.scatter(
            part["plot_x"],
            part["plot_y"],
            s=size,
            marker=marker,
            c=[_time_color(float(time_value), t_min, t_max)],
            alpha=alpha,
            linewidths=0,
            label=f"{kind} {float(time_value):g}",
        )


def _draw_trajectories(ax: plt.Axes, df: pd.DataFrame, *, alpha: float = 0.16, linewidth: float = 0.65) -> None:
    traj = df[df["kind"] == "trajectory"].sort_values(["trajectory_id", "time"])
    segments = []
    colors = []
    t_min = float(df["time"].min())
    t_max = float(df["time"].max())
    for _, part in traj.groupby("trajectory_id", sort=False):
        xy = part[["plot_x", "plot_y"]].to_numpy(dtype=float)
        times = part["time"].to_numpy(dtype=float)
        if xy.shape[0] < 2:
            continue
        pair_segments = np.stack([xy[:-1], xy[1:]], axis=1)
        segments.extend(pair_segments)
        colors.extend([_time_color(float(t), t_min, t_max) for t in times[:-1]])
    if segments:
        lc = LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha, zorder=1)
        ax.add_collection(lc)


def _style_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("projection 1")
    ax.set_ylabel("projection 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E7E7E7", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    args = build_parser().parse_args()
    points_csv = args.points_csv.expanduser().resolve()
    if not points_csv.exists():
        raise FileNotFoundError(
            f"Missing trajectory points CSV: {points_csv}. Run piuot/export_trajectory_points.py first."
        )
    df = pd.read_csv(points_csv)
    required = {"kind", "time", "plot_x", "plot_y", "trajectory_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{points_csv} is missing required columns: {sorted(missing)}")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_min = float(df["time"].min())
    t_max = float(df["time"].max())

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=220)
    _draw_trajectories(ax, df)
    _draw_time_points(
        ax,
        df,
        kind="observed",
        marker="o",
        size=5.0,
        alpha=0.44,
        t_min=t_min,
        t_max=t_max,
        max_per_time=int(args.max_observed_per_time),
        seed=int(args.seed),
    )
    _draw_time_points(
        ax,
        df,
        kind="predicted",
        marker="^",
        size=8.0,
        alpha=0.62,
        t_min=t_min,
        t_max=t_max,
        max_per_time=int(args.max_predicted_per_time),
        seed=int(args.seed) + 1000,
    )
    _style_axes(ax)
    ax.set_title("Prediction of Continuous Cell Dynamics")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=t_min, vmax=t_max))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("time")
    fig.tight_layout()

    outputs = {}
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{args.prefix}_continuous_dynamics.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        outputs[ext] = str(path)
    plt.close(fig)

    manifest = {
        "points_csv": str(points_csv),
        "outputs": outputs,
        "columns": {
            "kind": "kind",
            "time": "time",
            "x": "plot_x",
            "y": "plot_y",
            "trajectory_id": "trajectory_id",
        },
    }
    manifest_path = out_dir / f"{args.prefix}_continuous_dynamics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
