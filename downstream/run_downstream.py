from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT, PIUOT_ROOT


MANUAL_RUN_NAME = "piuot_run"
MANUAL_SEED = 0
MANUAL_DATA_PATH = str(PIUOT_ROOT / "input" / "input.h5ad")
MANUAL_EMBEDDING_KEY = "X_gae15"
MANUAL_RAW_TIME_KEY = "t"
MANUAL_CHECKPOINT = "auto"
MANUAL_OUTPUT_LABEL = "dataset"
MANUAL_OUTPUT_SLUG = "dataset"
MANUAL_STATE_KEY = "consensus_cluster"
MANUAL_FATE_KEY = "phenotype_facs"
MANUAL_ANALYSIS_DEVICE = "cpu"
MANUAL_PERTURB_DEVICE = "cpu"
MANUAL_CRITICAL_INDICATOR = "product"
MANUAL_CRITICAL_WINDOW_START = None
MANUAL_CRITICAL_WINDOW_END = None
MANUAL_ANCHOR_MIN_TIME = None
MANUAL_NORMALIZE_START_TIME = None
MANUAL_TOP_TERMINAL_FATES = 5
MANUAL_PERTURB_START_TIME = None
MANUAL_PERTURB_END_TIME = None
MANUAL_PERTURB_TARGET_LABEL = None
MANUAL_PERTURB_N_TIMEPOINTS = 25
MANUAL_PERTURB_N_REPEATS = 4
MANUAL_PERTURB_MAX_START_CELLS = 96
MANUAL_PERTURB_SCALE = 2.0
MANUAL_POTENTIAL_REDUCTION = "pca"
MANUAL_POTENTIAL_COORDS_OBSM_KEY = None
MANUAL_POTENTIAL_TIME_MODE = "obs"
MANUAL_TRAJECTORY_N_PARTICLES = 2000
MANUAL_TRAJECTORY_N_TRAJECTORIES = 96
MANUAL_TRAJECTORY_N_DENSE_TIMEPOINTS = 50
MANUAL_TRAJECTORY_PROJECTION = "pca"


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return slug or "dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run downstream analysis with manual path settings.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-name", default=MANUAL_RUN_NAME)
    parser.add_argument("--seed", type=int, default=MANUAL_SEED)
    parser.add_argument("--data-path", type=Path, default=Path(MANUAL_DATA_PATH))
    parser.add_argument("--embedding-key", default=MANUAL_EMBEDDING_KEY)
    parser.add_argument("--raw-time-key", default=MANUAL_RAW_TIME_KEY)
    parser.add_argument("--checkpoint", default=MANUAL_CHECKPOINT)
    parser.add_argument("--output-label", default=MANUAL_OUTPUT_LABEL)
    parser.add_argument("--output-slug", default=MANUAL_OUTPUT_SLUG)
    parser.add_argument("--state-key", default=MANUAL_STATE_KEY)
    parser.add_argument("--fate-key", default=MANUAL_FATE_KEY)
    parser.add_argument("--analysis-device", default=MANUAL_ANALYSIS_DEVICE)
    parser.add_argument("--perturb-device", default=MANUAL_PERTURB_DEVICE)
    parser.add_argument("--critical-indicator", default=MANUAL_CRITICAL_INDICATOR)
    parser.add_argument("--critical-window-start", type=float, default=MANUAL_CRITICAL_WINDOW_START)
    parser.add_argument("--critical-window-end", type=float, default=MANUAL_CRITICAL_WINDOW_END)
    parser.add_argument("--anchor-min-time", type=float, default=MANUAL_ANCHOR_MIN_TIME)
    parser.add_argument("--normalize-start-time", type=float, default=MANUAL_NORMALIZE_START_TIME)
    parser.add_argument("--top-terminal-fates", type=int, default=MANUAL_TOP_TERMINAL_FATES)
    parser.add_argument("--perturb-start-time", type=float, default=MANUAL_PERTURB_START_TIME)
    parser.add_argument("--perturb-end-time", type=float, default=MANUAL_PERTURB_END_TIME)
    parser.add_argument("--perturb-target-label", default=MANUAL_PERTURB_TARGET_LABEL)
    parser.add_argument("--perturb-n-timepoints", type=int, default=MANUAL_PERTURB_N_TIMEPOINTS)
    parser.add_argument("--perturb-n-repeats", type=int, default=MANUAL_PERTURB_N_REPEATS)
    parser.add_argument("--perturb-max-start-cells", type=int, default=MANUAL_PERTURB_MAX_START_CELLS)
    parser.add_argument("--perturb-scale", type=float, default=MANUAL_PERTURB_SCALE)
    parser.add_argument("--potential-reduction", choices=["pca", "first2"], default=MANUAL_POTENTIAL_REDUCTION)
    parser.add_argument("--potential-coords-obsm-key", default=MANUAL_POTENTIAL_COORDS_OBSM_KEY)
    parser.add_argument("--potential-time-mode", choices=["obs", "fixed", "start", "end", "mid"], default=MANUAL_POTENTIAL_TIME_MODE)
    parser.add_argument("--potential-fixed-time", type=float, default=None)
    parser.add_argument("--trajectory-n-particles", type=int, default=MANUAL_TRAJECTORY_N_PARTICLES)
    parser.add_argument("--trajectory-n-trajectories", type=int, default=MANUAL_TRAJECTORY_N_TRAJECTORIES)
    parser.add_argument("--trajectory-n-dense-timepoints", type=int, default=MANUAL_TRAJECTORY_N_DENSE_TIMEPOINTS)
    parser.add_argument("--trajectory-projection", choices=["pca", "first2"], default=MANUAL_TRAJECTORY_PROJECTION)
    parser.add_argument("--skip-potential-map", action="store_true")
    parser.add_argument("--skip-perturbation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    py = str(Path(args.python_bin).resolve())

    run(
        [
            py,
            str(PIUOT_ROOT / "plot.py"),
            "--run_name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--checkpoint_epoch",
            args.checkpoint,
            "--output_label",
            args.output_label,
        ]
    )

    run(
        [
            py,
            str(PIUOT_ROOT / "diagnose.py"),
            "--run_name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--epoch_tag",
            args.checkpoint,
            "--output_label",
            args.output_label,
        ]
    )

    output_slug = slugify(args.output_slug)
    trajectory_dir = DOWNSTREAM_OUTPUT_ROOT / f"{output_slug}_trajectory_points"
    trajectory_points_csv = trajectory_dir / f"{output_slug}_trajectory_points.csv"
    run(
        [
            py,
            str(PIUOT_ROOT / "export_trajectory_points.py"),
            "--run-name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--checkpoint-epoch",
            args.checkpoint,
            "--output-label",
            args.output_label,
            "--output-prefix",
            output_slug,
            "--output-dir",
            str(trajectory_dir),
            "--n-particles",
            str(args.trajectory_n_particles),
            "--n-trajectories",
            str(args.trajectory_n_trajectories),
            "--n-dense-timepoints",
            str(args.trajectory_n_dense_timepoints),
            "--projection",
            args.trajectory_projection,
            "--device",
            args.analysis_device,
        ]
    )
    figure_b_dir = DOWNSTREAM_OUTPUT_ROOT / f"{output_slug}_figure_b_continuous_dynamics"
    run(
        [
            py,
            str(PROJECT_ROOT / "downstream" / "figure_b_continuous_dynamics.py"),
            "--points-csv",
            str(trajectory_points_csv),
            "--output-dir",
            str(figure_b_dir),
            "--prefix",
            output_slug,
        ]
    )

    potential_dir = DOWNSTREAM_OUTPUT_ROOT / f"{output_slug}_potential_landscape"
    potential_points_csv = potential_dir / f"{output_slug}_potential_landscape_points.csv"
    if not args.skip_potential_map:
        export_cmd = [
            py,
            str(PROJECT_ROOT / "downstream" / "export_potential_landscape_points.py"),
            "--run-name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--checkpoint",
            args.checkpoint,
            "--data-path",
            str(Path(args.data_path).expanduser().resolve()),
            "--embedding-key",
            args.embedding_key,
            "--time-obs-key",
            args.raw_time_key,
            "--reduction",
            args.potential_reduction,
            "--time-mode",
            args.potential_time_mode,
            "--device",
            args.analysis_device,
            "--output-prefix",
            output_slug,
            "--output-dir",
            str(potential_dir),
        ]
        if str(args.state_key).strip():
            export_cmd += ["--annotation-key", str(args.state_key).strip()]
        if args.potential_coords_obsm_key:
            export_cmd += ["--coords-obsm-key", str(args.potential_coords_obsm_key)]
        if args.potential_fixed_time is not None:
            export_cmd += ["--fixed-time", str(args.potential_fixed_time)]
        run(export_cmd)

        run(
            [
                py,
                str(PROJECT_ROOT / "downstream" / "figure_c_umap_potential_landscape.py"),
                "--points-csv",
                str(potential_points_csv),
                "--x-column",
                "plot_x",
                "--y-column",
                "plot_y",
                "--time-column",
                "time_obs",
                "--z-column",
                "surface_z",
                "--prefix",
                output_slug,
                "--output-dir",
                str(potential_dir),
            ]
        )

    analysis_cmd = [
        py,
        str(PROJECT_ROOT / "downstream" / "analyze_manifold_physics_fates.py"),
        "--run-name",
        args.run_name,
        "--seed",
        str(args.seed),
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.analysis_device,
        "--data-path",
        str(Path(args.data_path).expanduser().resolve()),
        "--embedding-key",
        args.embedding_key,
        "--label",
        args.output_label,
        "--critical-indicator",
        args.critical_indicator,
    ]
    if str(args.state_key).strip():
        analysis_cmd += ["--state-key", str(args.state_key).strip()]
    if str(args.fate_key).strip():
        analysis_cmd += ["--fate-key", str(args.fate_key).strip()]
    if args.critical_window_start is not None:
        analysis_cmd += ["--critical-window-start", str(args.critical_window_start)]
    if args.critical_window_end is not None:
        analysis_cmd += ["--critical-window-end", str(args.critical_window_end)]
    if args.anchor_min_time is not None:
        analysis_cmd += ["--anchor-min-time", str(args.anchor_min_time)]
    if args.normalize_start_time is not None:
        analysis_cmd += ["--normalize-start-time", str(args.normalize_start_time)]
    run(analysis_cmd)

    if not args.skip_perturbation:
        perturb_cmd = [
            py,
            str(PROJECT_ROOT / "downstream" / "build_perturbation_dynamic_fraction.py"),
            "--run-name",
            args.run_name,
            "--seed",
            str(args.seed),
            "--checkpoint",
            args.checkpoint,
            "--device",
            args.perturb_device,
            "--data-path",
            str(Path(args.data_path).expanduser().resolve()),
            "--embedding-key",
            args.embedding_key,
            "--raw-time-key",
            args.raw_time_key,
            "--fate-key",
            args.fate_key,
            "--output-label",
            args.output_label,
            "--n-timepoints",
            str(args.perturb_n_timepoints),
            "--n-repeats",
            str(args.perturb_n_repeats),
            "--max-start-cells",
            str(args.perturb_max_start_cells),
            "--scale",
            str(args.perturb_scale),
            "--top-terminal-fates",
            str(args.top_terminal_fates),
        ]
        if args.perturb_start_time is not None:
            perturb_cmd += ["--start-time", str(args.perturb_start_time)]
        if args.perturb_end_time is not None:
            perturb_cmd += ["--end-time", str(args.perturb_end_time)]
        if args.perturb_target_label:
            perturb_cmd += ["--target-label", str(args.perturb_target_label)]
        run(perturb_cmd)

    manifest = {
        "run_name": args.run_name,
        "output_label": args.output_label,
        "output_slug": args.output_slug,
        "embedding_key": args.embedding_key,
        "trajectory_dir": str(PIUOT_ROOT / "output" / "figs" / args.output_label),
        "trajectory_points_dir": str(trajectory_dir),
        "trajectory_points_csv": str(trajectory_points_csv),
        "figure_b_dir": str(figure_b_dir),
        "potential_landscape_dir": str(potential_dir),
        "potential_landscape_points_csv": str(potential_points_csv),
        "perturbation_dir": str(DOWNSTREAM_OUTPUT_ROOT / f"{args.output_label}_perturbation_dynamic_fraction"),
    }
    DOWNSTREAM_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (DOWNSTREAM_OUTPUT_ROOT / f"{args.output_label}_downstream_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
