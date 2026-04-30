from __future__ import annotations

import argparse
import json
import math
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


DEFAULT_METRICS = ("w1", "w2_sq", "mmd_rbf")
DEFAULT_TICKS = {
    "w1": [1.0, 2.0, 4.0],
    "w2_sq": [1.0, 4.0, 16.0],
    "mmd_rbf": [0.02, 0.1, 0.5],
}
DEFAULT_METRIC_LABELS = {
    "w1": "W1 distance",
    "w2_sq": r"$W_2^2$ distance",
    "w2": "W2 distance",
    "mmd_rbf": "MMD",
    "mmd": "MMD",
}
DEFAULT_MODEL_LABELS = {
    "PIUOT-official-gae15": "PIUOT\nGAE15",
    "TrajectoryNet": "Trajectory\nNet",
}
DEFAULT_COLORS = [
    "#c83b2b",
    "#1f77b4",
    "#2ca02c",
    "#7f8c8d",
    "#e87d19",
    "#9467bd",
    "#8c564b",
    "#17becf",
]


def slugify(value: str) -> str:
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"_", "-", "."}:
            keep.append(char)
        else:
            keep.append("_")
    slug = "".join(keep).strip("._")
    return slug or "metric"


def parse_csv_list(value: str | None, default: tuple[str, ...] | None = None) -> list[str]:
    if value is None:
        return list(default or ())
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_key_value_list(value: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not value:
        return parsed
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE in '{item}'")
        key, raw_val = item.split("=", 1)
        parsed[key.strip()] = raw_val.replace("\\n", "\n").strip()
    return parsed


def parse_tick_spec(value: str | None) -> dict[str, list[float]]:
    ticks: dict[str, list[float]] = {}
    if not value:
        return ticks
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected METRIC=v1,v2,v3 in '{item}'")
        metric, raw_values = item.split("=", 1)
        tick_values = [float(part.strip()) for part in raw_values.split(",") if part.strip()]
        if not tick_values:
            raise ValueError(f"No ticks provided for metric '{metric}'")
        ticks[metric.strip()] = tick_values
    return ticks


def format_tick(value: float) -> str:
    if value >= 1 and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def auto_log_ticks(values: np.ndarray) -> list[float]:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return [1.0]

    vmin = float(positive.min())
    vmax = float(positive.max())
    lo = math.floor(math.log10(vmin))
    hi = math.ceil(math.log10(vmax))
    candidates: list[float] = []
    for power in range(lo - 1, hi + 2):
        scale = 10.0**power
        for base in (1.0, 2.0, 4.0, 5.0):
            tick = base * scale
            if vmin <= tick <= vmax:
                candidates.append(tick)

    if len(candidates) <= 3:
        return candidates or [vmin, vmax]

    positions = np.linspace(0, len(candidates) - 1, 3).round().astype(int)
    return [float(candidates[idx]) for idx in positions]


def prepare_dataframe(args: argparse.Namespace, metrics: list[str]) -> pd.DataFrame:
    df = pd.read_csv(args.per_time_csv)
    required = {"model", *metrics}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in per-time CSV: {missing}")

    if not args.include_initial_time and "time" in df.columns:
        numeric_time = pd.to_numeric(df["time"], errors="coerce")
        if numeric_time.notna().any():
            min_time = float(numeric_time.min())
            df = df[numeric_time > min_time].copy()

    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=metrics)
    return df


def resolve_model_order(df: pd.DataFrame, explicit_order: list[str]) -> list[str]:
    if explicit_order:
        missing = [model for model in explicit_order if model not in set(df["model"])]
        if missing:
            raise ValueError(f"Models in --model-order are absent from CSV: {missing}")
        return explicit_order
    return list(dict.fromkeys(df["model"].astype(str).tolist()))


def compute_ylim(values: np.ndarray, ticks: list[float] | None, log_y: bool) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if log_y:
        finite = finite[finite > 0]
    if finite.size == 0:
        return (0.1, 10.0) if log_y else (0.0, 1.0)

    data_min = float(finite.min())
    data_max = float(finite.max())
    if ticks:
        tick_arr = np.asarray(ticks, dtype=float)
        tick_arr = tick_arr[np.isfinite(tick_arr)]
        if log_y:
            tick_arr = tick_arr[tick_arr > 0]
        if tick_arr.size:
            data_min = min(data_min, float(tick_arr.min()))
            data_max = max(data_max, float(tick_arr.max()))

    if log_y:
        return data_min / 1.6, data_max * 1.7
    pad = (data_max - data_min) * 0.08 if data_max > data_min else 1.0
    return data_min - pad, data_max + pad


def plot_metric(
    *,
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    model_order: list[str],
    model_labels: dict[str, str],
    colors: dict[str, str],
    ticks: list[float] | None,
    out_path: Path,
    complete: bool,
    dpi: int,
    point_size: float,
    log_y: bool,
) -> None:
    fig = plt.figure(figsize=(6.0, 6.0), dpi=dpi)
    if complete:
        ax = fig.add_axes([0.22, 0.21, 0.72, 0.70])
    else:
        ax = fig.add_axes([0.16, 0.10, 0.78, 0.84])

    positions = np.arange(1, len(model_order) + 1)
    data = [df[df["model"] == model][metric].to_numpy(dtype=float) for model in model_order]
    if log_y and any(np.any(vals <= 0) for vals in data):
        raise ValueError(f"Metric '{metric}' contains non-positive values; log-scale plotting is not valid.")

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.7},
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )
    for patch, model in zip(box["boxes"], model_order):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.25)
        patch.set_edgecolor(colors[model])

    for idx, model in enumerate(model_order, start=1):
        vals = df[df["model"] == model][metric].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, vals.size)
        ax.scatter(
            np.full(vals.size, idx) + jitter,
            vals,
            s=point_size,
            color=colors[model],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

    if log_y:
        ax.set_yscale("log")

    flat_values = np.concatenate([vals for vals in data if vals.size])
    ax.set_ylim(*compute_ylim(flat_values, ticks, log_y))
    if ticks:
        ax.set_yticks(ticks)
        if complete:
            ax.set_yticklabels([format_tick(tick) for tick in ticks], fontsize=13)
        else:
            ax.set_yticklabels([])

    if complete:
        ax.set_xticks(positions)
        ax.set_xticklabels([model_labels.get(model, model) for model in model_order], fontsize=11)
        ax.set_ylabel(metric_label, fontsize=14)
        ax.set_title(metric_label, fontsize=16, pad=12)
        ax.tick_params(axis="both", width=1.8, length=5, labelsize=12)
    else:
        ax.set_xticks([])
        ax.tick_params(axis="y", width=1.6, length=5)
        ax.tick_params(axis="x", length=0)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2.0)

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Figure D benchmark metric panels from per-time model comparison metrics."
    )
    parser.add_argument("--per-time-csv", type=Path, required=True, help="CSV with model, time, and metric columns.")
    parser.add_argument("--output-dir", type=Path, default=DOWNSTREAM_OUTPUT_ROOT / "figure_d_benchmark_metrics")
    parser.add_argument("--output-prefix", default="figure_d")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--model-order", default=None, help="Comma-separated model order. Defaults to CSV order.")
    parser.add_argument(
        "--model-labels",
        default=None,
        help="Optional semicolon list: raw_model=display label; use \\n for line breaks.",
    )
    parser.add_argument(
        "--metric-labels",
        default=None,
        help="Optional semicolon list: metric=display label.",
    )
    parser.add_argument(
        "--tick-spec",
        default=None,
        help="Optional semicolon list: metric=v1,v2,v3. Defaults to manuscript ticks for w1/w2_sq/mmd_rbf.",
    )
    parser.add_argument("--auto-ticks", action="store_true", help="Use data-driven sparse log ticks instead.")
    parser.add_argument("--include-initial-time", action="store_true", help="Do not drop the earliest time point.")
    parser.add_argument("--linear-y", action="store_true", help="Use linear y scale instead of log scale.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=92.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = parse_csv_list(args.metrics, DEFAULT_METRICS)
    df = prepare_dataframe(args, metrics)
    model_order = resolve_model_order(df, parse_csv_list(args.model_order))

    model_labels = dict(DEFAULT_MODEL_LABELS)
    model_labels.update(parse_key_value_list(args.model_labels))
    metric_labels = dict(DEFAULT_METRIC_LABELS)
    metric_labels.update(parse_key_value_list(args.metric_labels))

    manual_ticks = parse_tick_spec(args.tick_spec)
    colors = {model: DEFAULT_COLORS[idx % len(DEFAULT_COLORS)] for idx, model in enumerate(model_order)}

    out_dir = args.output_dir.expanduser().resolve()
    complete_dir = out_dir / "complete"
    clean_dir = out_dir / "clean"
    complete_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, dict[str, str]] = {}
    tick_manifest: dict[str, list[float]] = {}
    for metric in metrics:
        values = df[metric].to_numpy(dtype=float)
        if args.auto_ticks:
            ticks = auto_log_ticks(values) if not args.linear_y else None
        else:
            ticks = manual_ticks.get(metric, DEFAULT_TICKS.get(metric))
            if ticks is None and not args.linear_y:
                ticks = auto_log_ticks(values)
        tick_manifest[metric] = [float(tick) for tick in ticks] if ticks else []

        metric_slug = slugify(metric)
        complete_path = complete_dir / f"{args.output_prefix}_{metric_slug}_complete.png"
        clean_path = clean_dir / f"{args.output_prefix}_{metric_slug}_clean.png"
        label = metric_labels.get(metric, metric)

        plot_metric(
            df=df,
            metric=metric,
            metric_label=label,
            model_order=model_order,
            model_labels=model_labels,
            colors=colors,
            ticks=ticks,
            out_path=complete_path,
            complete=True,
            dpi=args.dpi,
            point_size=args.point_size,
            log_y=not args.linear_y,
        )
        plot_metric(
            df=df,
            metric=metric,
            metric_label=label,
            model_order=model_order,
            model_labels=model_labels,
            colors=colors,
            ticks=ticks,
            out_path=clean_path,
            complete=False,
            dpi=args.dpi,
            point_size=args.point_size,
            log_y=not args.linear_y,
        )
        generated[metric] = {"complete": str(complete_path), "clean": str(clean_path)}

    summary = (
        df.groupby("model", as_index=False)[metrics]
        .mean()
        .set_index("model")
        .reindex(model_order)
        .reset_index()
    )
    manifest = {
        "name": "Figure D benchmark metric comparison",
        "source_per_time_csv": str(args.per_time_csv.expanduser().resolve()),
        "dropped_initial_time": not args.include_initial_time and "time" in pd.read_csv(args.per_time_csv, nrows=1).columns,
        "metrics": metrics,
        "model_order": model_order,
        "model_labels": {model: model_labels.get(model, model) for model in model_order},
        "ticks": tick_manifest,
        "scale": "linear" if args.linear_y else "log",
        "generated": generated,
        "summary_from_plotted_rows": summary.to_dict(orient="records"),
    }
    (out_dir / f"{args.output_prefix}_benchmark_metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
