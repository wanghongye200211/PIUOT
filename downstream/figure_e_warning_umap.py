from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT

DEFAULT_INPUT = DOWNSTREAM_OUTPUT_ROOT / "figure_e_warning_umap" / "warning_landscape_umap_cell_scores.csv"
DEFAULT_OUTPUT_DIR = DOWNSTREAM_OUTPUT_ROOT / "figure_e_warning_umap"

WARNING_CMAP = LinearSegmentedColormap.from_list(
    "warning_low_light_high_red",
    ["#fffaf0", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"],
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Figure e: standalone UMAP warning landscape."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-column", default="umap_1")
    parser.add_argument("--y-column", default="umap_2")
    parser.add_argument("--warning-column", default="warning")
    parser.add_argument("--prefix", default="figure_e_warning_umap")
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.92)
    parser.add_argument("--pad", type=float, default=0.25)
    return parser


def _save(fig: plt.Figure, stem: Path) -> None:
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(stem.with_suffix(suffix), dpi=600 if suffix == ".png" else None, bbox_inches="tight")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _draw_panel(
    df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    warning_column: str,
    stem: Path,
    annotated: bool,
    point_size: float,
    alpha: float,
    pad: float,
) -> dict[str, str]:
    plot_df = df.sort_values(warning_column, ascending=True).copy()
    norm = Normalize(vmin=0.0, vmax=float(plot_df[warning_column].max()))

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=240)
    scatter = ax.scatter(
        plot_df[x_column],
        plot_df[y_column],
        c=plot_df[warning_column],
        cmap=WARNING_CMAP,
        norm=norm,
        s=point_size,
        linewidths=0,
        alpha=alpha,
    )
    ax.set_xlim(float(df[x_column].min()) - pad, float(df[x_column].max()) + pad)
    ax.set_ylim(float(df[y_column].min()) - pad, float(df[y_column].max()) + pad)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    if annotated:
        ax.set_title("Figure e | warning landscape", fontsize=12)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("warning")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelbottom=False, labelleft=False)

    fig.tight_layout()
    _save(fig, stem)
    plt.close(fig)
    return {
        "png": _rel(stem.with_suffix(".png")),
        "svg": _rel(stem.with_suffix(".svg")),
        "pdf": _rel(stem.with_suffix(".pdf")),
    }


def main() -> None:
    args = build_parser().parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    required = [args.x_column, args.y_column, args.warning_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}; available columns: {list(df.columns)}")

    clean_stem = output_dir / f"{args.prefix}_clean"
    annotated_stem = output_dir / f"{args.prefix}_annotated"
    outputs = {
        "clean": _draw_panel(
            df,
            x_column=args.x_column,
            y_column=args.y_column,
            warning_column=args.warning_column,
            stem=clean_stem,
            annotated=False,
            point_size=float(args.point_size),
            alpha=float(args.alpha),
            pad=float(args.pad),
        ),
        "annotated": _draw_panel(
            df,
            x_column=args.x_column,
            y_column=args.y_column,
            warning_column=args.warning_column,
            stem=annotated_stem,
            annotated=True,
            point_size=float(args.point_size),
            alpha=float(args.alpha),
            pad=float(args.pad),
        ),
    }
    manifest = {
        "figure": "Figure e",
        "description": "Standalone UMAP warning landscape.",
        "input_csv": _rel(input_csv),
        "columns": {"x": args.x_column, "y": args.y_column, "warning": args.warning_column},
        "color": {
            "colormap": "custom low-light to high-red",
            "vmin": 0.0,
            "vmax": float(df[args.warning_column].max()),
            "note": "Low warning values are intentionally light; cells are drawn in ascending warning order.",
        },
        "outputs": outputs,
    }
    manifest_path = output_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
