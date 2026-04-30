from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT

TIMES = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
TIME_LABELS = {
    0.0: "0 d",
    1.0: "1 d",
    1.5: "1.5 d",
    2.0: "2 d",
    2.5: "2.5 d",
    3.0: "3 d",
    4.0: "4 d",
    5.0: "5 d",
}

# Color source: CSV time column only. Do not infer time classes from PNG colors.
TIME_COLORS = {
    0.0: "#D5AC2F",
    1.0: "#8BBF64",
    1.5: "#EBCDC3",
    2.0: "#EF6A4A",
    2.5: "#87CEE3",
    3.0: "#2F89C8",
    4.0: "#007C82",
    5.0: "#C7376F",
}

VIEW = {
    "name": "view06_reverse_high",
    "elev": 44,
    "azim": 128,
    "focal_length": 0.70,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw Figure C from a potential-landscape points CSV.")
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DOWNSTREAM_OUTPUT_ROOT / "figure_c_umap_potential_landscape")
    parser.add_argument("--prefix", default="figure_c")
    parser.add_argument("--x-column", default="plot_x")
    parser.add_argument("--y-column", default="plot_y")
    parser.add_argument("--time-column", default="time_obs")
    parser.add_argument("--z-column", default="surface_z")
    parser.add_argument("--marked", action="store_true", help="Also write the legacy M/En marker-check panel.")
    return parser


def _prepare_df(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.points_csv.expanduser().resolve())
    required = [args.x_column, args.y_column, args.time_column, args.z_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}; available columns: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "umap_1": df[args.x_column].astype(float),
            "umap_2": df[args.y_column].astype(float),
            "time": df[args.time_column].astype(float),
            "surface_z": df[args.z_column].astype(float),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _configure_time_palette(df: pd.DataFrame) -> None:
    global TIMES, TIME_LABELS, TIME_COLORS
    TIMES = [float(v) for v in sorted(df["time"].unique())]
    TIME_LABELS = {
        value: (f"{int(round(value))} d" if abs(value - round(value)) < 1e-8 else f"{value:g} d")
        for value in TIMES
    }
    cmap = plt.get_cmap("viridis")
    denom = max(len(TIMES) - 1, 1)
    TIME_COLORS = {value: cmap(idx / denom) for idx, value in enumerate(TIMES)}


def make_surface(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pad_x = (df["umap_1"].max() - df["umap_1"].min()) * 0.16
    pad_y = (df["umap_2"].max() - df["umap_2"].min()) * 0.18
    x = np.linspace(df["umap_1"].min() - pad_x, df["umap_1"].max() + pad_x, 112)
    y = np.linspace(df["umap_2"].min() - pad_y, df["umap_2"].max() + pad_y, 112)
    xx, yy = np.meshgrid(x, y)

    pts = df[["umap_1", "umap_2"]].to_numpy()
    vals = df["surface_z"].to_numpy()
    linear = LinearNDInterpolator(pts, vals)
    nearest = NearestNDInterpolator(pts, vals)
    zz = linear(xx, yy)
    zz[np.isnan(zz)] = nearest(xx[np.isnan(zz)], yy[np.isnan(zz)])
    zz = gaussian_filter(zz, sigma=2.2)

    border = max(4, int(0.06 * zz.shape[0]))
    zz[:border, :] = gaussian_filter(zz[:border, :], sigma=3.5)
    zz[-border:, :] = gaussian_filter(zz[-border:, :], sigma=3.5)
    zz[:, :border] = gaussian_filter(zz[:, :border], sigma=3.5)
    zz[:, -border:] = gaussian_filter(zz[:, -border:], sigma=3.5)
    return xx, yy, zz


def interp_points_z(df: pd.DataFrame, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    linear = LinearNDInterpolator(grid, zz.ravel())
    nearest = NearestNDInterpolator(grid, zz.ravel())
    z = linear(df["umap_1"], df["umap_2"])
    missing = np.isnan(z)
    z[missing] = nearest(df.loc[missing, "umap_1"], df.loc[missing, "umap_2"])
    return np.asarray(z, dtype=float)


def draw_side_walls(ax, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, zbase: float) -> None:
    wall_color = (0.88, 0.88, 0.86, 0.72)
    edge_color = (0.80, 0.80, 0.78, 0.34)
    stride = 2
    for row in (0, -1):
        x = xx[row, ::stride]
        y = yy[row, ::stride]
        z = zz[row, ::stride]
        ax.plot_surface(
            np.vstack([x, x]),
            np.vstack([y, y]),
            np.vstack([np.full_like(z, zbase), z]),
            color=wall_color,
            edgecolor=edge_color,
            linewidth=0.22,
            antialiased=True,
            shade=False,
        )
    for col in (0, -1):
        x = xx[::stride, col]
        y = yy[::stride, col]
        z = zz[::stride, col]
        ax.plot_surface(
            np.vstack([x, x]).T,
            np.vstack([y, y]).T,
            np.vstack([np.full_like(z, zbase), z]).T,
            color=wall_color,
            edgecolor=edge_color,
            linewidth=0.22,
            antialiased=True,
            shade=False,
        )


def plot_surface_base(ax, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, zbase: float) -> None:
    draw_side_walls(ax, xx, yy, zz, zbase)
    light_source = LightSource(azdeg=305, altdeg=30)
    face = light_source.shade(zz, cmap=plt.cm.Greys_r, vert_exag=0.75, blend_mode="soft")
    face[..., :3] = 0.68 * face[..., :3] + 0.32 * np.array([0.94, 0.935, 0.91])
    face[..., 3] = 0.46
    ax.plot_surface(
        xx,
        yy,
        zz,
        rstride=1,
        cstride=1,
        facecolors=face,
        edgecolor=(0.50, 0.50, 0.48, 0.50),
        linewidth=0.19,
        antialiased=True,
        shade=False,
    )


def setup_axes(ax, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, zbase: float) -> None:
    ax.set_xlabel("UMAP 1", labelpad=10, fontsize=10)
    ax.set_ylabel("UMAP 2", labelpad=10, fontsize=10)
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_box_aspect((1.45, 1.05, 0.48))
    ax.view_init(elev=VIEW["elev"], azim=VIEW["azim"])
    ax.set_proj_type("persp", focal_length=VIEW["focal_length"])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_zlim(zbase, zz.max() + 0.35)
    ax.set_axis_off()


def add_m_en_markers(ax, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> None:
    markers = [
        (2.55, 9.18, "#111111"),  # En branch
        (7.78, 10.05, "#6F6F6F"),  # M branch
    ]
    for x0, y0, color in markers:
        point = pd.DataFrame({"umap_1": [x0], "umap_2": [y0]})
        z0 = float(interp_points_z(point, xx, yy, zz)[0]) + 0.16
        ax.scatter(
            [x0],
            [y0],
            [z0],
            s=620,
            facecolors="none",
            edgecolors=color,
            linewidths=2.2,
            depthshade=False,
            zorder=40,
        )


def plot_main(df: pd.DataFrame, figure_dir: Path, prefix: str, marked: bool = False) -> Path:
    xx, yy, zz = make_surface(df)
    point_z = interp_points_z(df, xx, yy, zz) + 0.085

    fig = plt.figure(figsize=(7.2, 7.1), dpi=320)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    zbase = float(zz.min() - 1.55)
    plot_surface_base(ax, xx, yy, zz, zbase)

    for t in TIMES:
        sub = df[np.isclose(df["time"], t)]
        idx = sub.index.to_numpy()
        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            point_z[idx],
            s=11.0,
            color=TIME_COLORS[t],
            alpha=0.98,
            edgecolors=(1, 1, 1, 0.78),
            linewidths=0.08,
            depthshade=False,
            zorder=30,
        )

    if marked:
        add_m_en_markers(ax, xx, yy, zz)

    setup_axes(ax, xx, yy, zz, zbase)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    name = f"{prefix}_umap_potential_marked.png" if marked else f"{prefix}_umap_potential.png"
    out = figure_dir / name
    fig.savefig(out, transparent=False, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def plot_legend_only(figure_dir: Path, prefix: str) -> Path:
    fig, ax = plt.subplots(figsize=(1.28, 2.25), dpi=320)
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=TIME_COLORS[t],
            markeredgecolor="none",
            markersize=6.4,
            label=TIME_LABELS[t],
        )
        for t in TIMES
    ]
    legend = ax.legend(
        handles=handles,
        title="Predicted",
        loc="center left",
        frameon=False,
        borderaxespad=0,
        labelspacing=0.24,
        handletextpad=0.48,
    )
    legend.get_title().set_fontsize(10.5)
    for text in legend.get_texts():
        text.set_fontsize(8.8)
    out = figure_dir / f"{prefix}_time_legend.png"
    fig.savefig(out, transparent=False, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


def write_branch_time_check(df: pd.DataFrame, figure_dir: Path, prefix: str) -> Path:
    en_mask = (df["umap_1"] < 3.6) & (df["umap_2"] > 8.2)
    m_mask = (df["umap_1"] > 6.8) & (df["umap_2"] > 9.15)
    rows = []
    for branch, mask in [("En/left terminal", en_mask), ("M/right terminal", m_mask)]:
        sub = df.loc[mask]
        counts = sub["time"].value_counts().sort_index()
        for time_value, count in counts.items():
            rows.append(
                {
                    "branch": branch,
                    "time": float(time_value),
                    "n": int(count),
                    "branch_total": int(len(sub)),
                    "center_umap_1": float(sub["umap_1"].mean()),
                    "center_umap_2": float(sub["umap_2"].mean()),
                }
            )
    out = figure_dir / f"{prefix}_branch_time_check.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    args = build_parser().parse_args()
    figure_dir = args.output_dir.expanduser().resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    points_csv = args.points_csv.expanduser().resolve()
    df = _prepare_df(args)
    _configure_time_palette(df)
    main_png = plot_main(df, figure_dir, args.prefix, marked=False)
    marked_png = plot_main(df, figure_dir, args.prefix, marked=True) if args.marked else None
    legend_png = plot_legend_only(figure_dir, args.prefix)
    branch_csv = write_branch_time_check(df, figure_dir, args.prefix)
    outputs = {
        "main_png": str(main_png),
        "marked_png": str(marked_png) if marked_png is not None else None,
        "legend_png": str(legend_png),
        "branch_time_check_csv": str(branch_csv),
    }
    summary = {
        "figure_id": "figure_c",
        "title": "UMAP potential landscape",
        "points_csv": str(points_csv),
        "columns": {
            "x": args.x_column,
            "y": args.y_column,
            "time": args.time_column,
            "z": args.z_column,
        },
        "view": VIEW,
        "outputs": outputs,
    }
    (figure_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
