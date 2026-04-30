from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


METHOD_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = METHOD_ROOT / "output"

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


def _load_runtime_modules():
    src_pkg = importlib.import_module("core")
    sys.modules["src"] = src_pkg
    config_mod = importlib.import_module("src.config_model")
    model_mod = importlib.import_module("src.model")
    train_mod = importlib.import_module("src.train")
    return config_mod, model_mod, train_mod


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return slug or "dataset"


def _format_time_label(value) -> str:
    value = float(value)
    text = str(int(round(value))) if abs(value - round(value)) < 1e-8 else f"{value:g}"
    return f"{text} d"


def _resolve_run_dir(run_name: str, seed: int) -> Path:
    matches = sorted((OUTPUT_ROOT / run_name).glob(f"*/seed_{seed}/alltime"))
    if not matches:
        raise FileNotFoundError(f"Could not find run '{run_name}' for seed {seed}.")
    return matches[-1]


def _latest_epoch_from_checkpoints(run_dir: Path) -> str:
    matches = sorted(run_dir.glob("train.epoch_*.pt"))
    if not matches:
        raise FileNotFoundError(f"No epoch checkpoints found under {run_dir}")
    latest = max(matches, key=lambda path: int(path.stem.split("epoch_")[1]))
    return latest.stem.split("train.", 1)[1]


def _best_epoch_from_eval(run_dir: Path) -> str | None:
    eval_path = run_dir / "interpolate-mioemd2.log"
    if not eval_path.exists():
        return None
    eval_df = pd.read_csv(eval_path, sep="\t")
    if not {"epoch", "loss"}.issubset(eval_df.columns):
        return None
    mean_eval = eval_df.groupby("epoch", as_index=False)["loss"].mean()
    if mean_eval.empty:
        return None
    return str(mean_eval.loc[mean_eval["loss"].idxmin(), "epoch"])


def _resolve_epoch_tag(run_dir: Path, epoch_selector: str) -> str:
    if epoch_selector == "auto":
        best = _best_epoch_from_eval(run_dir)
        if best is not None and (run_dir / f"train.{best}.pt").exists():
            return best
        if (run_dir / "train.best.pt").exists():
            return "best"
        return _latest_epoch_from_checkpoints(run_dir)
    if epoch_selector == "final":
        return _latest_epoch_from_checkpoints(run_dir)
    return str(epoch_selector)


def _checkpoint_path(run_dir: Path, epoch_tag: str) -> Path:
    path = run_dir / "train.best.pt" if epoch_tag == "best" else run_dir / f"train.{epoch_tag}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def _move_time_series_to_device(x, device: torch.device):
    return [x_i.to(device) if x_i.device != device else x_i for x_i in x]


def _rollout(model, train_mod, config, x0: torch.Tensor, times: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    states = []
    chunk_size = max(1, int(config.ns))
    for start in range(0, x0.shape[0], chunk_size):
        x0_chunk = x0[start : start + chunk_size]
        if x0_chunk.shape[0] == 0:
            continue
        init = train_mod.build_initial_state(
            x0_chunk,
            bool(config.use_growth),
            clip_value=float(getattr(config, "mass_clip_value", 30.0)),
        )
        state = model([np.float64(t) for t in times.tolist()], init).detach().cpu()
        states.append(state)
    if not states:
        raise ValueError("No particles were available for rollout.")
    rollout_state = torch.cat(states, dim=1)
    pred_x = rollout_state[..., : int(config.x_dim)]
    if bool(config.use_growth):
        logw = rollout_state[..., int(config.x_dim) + 1 : int(config.x_dim) + 2]
        masses = []
        for time_idx in range(logw.shape[0]):
            masses.append(train_mod.normalized_mass_from_logw(logw[time_idx]).detach().cpu())
        mass = torch.stack(masses, dim=0)
    else:
        mass = torch.full(
            (pred_x.shape[0], pred_x.shape[1]),
            1.0 / max(pred_x.shape[1], 1),
            dtype=pred_x.dtype,
        )
    return pred_x.cpu(), mass.cpu()


def _projection_matrix(
    observed: list[np.ndarray],
    predicted: np.ndarray,
    trajectory: np.ndarray,
    *,
    method: str,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, str]:
    observed_flat = np.concatenate(observed, axis=0)
    pred_flat = predicted.reshape(-1, predicted.shape[-1])
    traj_flat = trajectory.reshape(-1, trajectory.shape[-1])
    blocks = [observed_flat, pred_flat, traj_flat]
    counts = [block.shape[0] for block in blocks]
    all_points = np.concatenate(blocks, axis=0)

    if method == "first2":
        if all_points.shape[1] < 2:
            raise ValueError("Cannot use first2 projection with fewer than two latent dimensions.")
        projected = all_points[:, :2]
        projection_name = "first2"
    else:
        reducer = PCA(n_components=2, random_state=int(seed)).fit(observed_flat)
        projected = reducer.transform(all_points)
        projection_name = "pca"

    start = 0
    observed_proj_flat = projected[start : start + counts[0]]
    start += counts[0]
    predicted_proj = projected[start : start + counts[1]].reshape(predicted.shape[0], predicted.shape[1], 2)
    start += counts[1]
    trajectory_proj = projected[start : start + counts[2]].reshape(trajectory.shape[0], trajectory.shape[1], 2)

    observed_proj = []
    offset = 0
    for block in observed:
        observed_proj.append(observed_proj_flat[offset : offset + block.shape[0]])
        offset += block.shape[0]
    return observed_proj, predicted_proj, trajectory_proj, projection_name


def _latent_columns(prefix: str, values: np.ndarray) -> dict[str, np.ndarray]:
    return {f"{prefix}_{idx}": values[:, idx].astype(float) for idx in range(values.shape[1])}


def _observed_rows(x, y, observed_proj) -> list[pd.DataFrame]:
    frames = []
    for time_index, (x_t, time_value, xy_t) in enumerate(zip(x, y, observed_proj)):
        latent = x_t.detach().cpu().numpy()
        frame = pd.DataFrame(
            {
                "kind": "observed",
                "time_index": int(time_index),
                "time": float(time_value),
                "time_label": _format_time_label(time_value),
                "point_id": np.arange(latent.shape[0], dtype=np.int64),
                "trajectory_id": -1,
                "mass": 1.0 / max(latent.shape[0], 1),
                "plot_x": xy_t[:, 0].astype(float),
                "plot_y": xy_t[:, 1].astype(float),
            }
        )
        for name, values in _latent_columns("latent", latent).items():
            frame[name] = values
        frames.append(frame)
    return frames


def _predicted_rows(pred_x: np.ndarray, pred_mass: np.ndarray, times: np.ndarray) -> list[pd.DataFrame]:
    frames = []
    for time_index, time_value in enumerate(times):
        latent = pred_x[time_index]
        frame = pd.DataFrame(
            {
                "kind": "predicted",
                "time_index": int(time_index),
                "time": float(time_value),
                "time_label": _format_time_label(time_value),
                "point_id": np.arange(latent.shape[0], dtype=np.int64),
                "trajectory_id": -1,
                "mass": pred_mass[time_index].astype(float),
                "plot_x": np.nan,
                "plot_y": np.nan,
            }
        )
        for name, values in _latent_columns("latent", latent).items():
            frame[name] = values
        frames.append(frame)
    return frames


def _trajectory_rows(traj_x: np.ndarray, traj_mass: np.ndarray, times: np.ndarray) -> list[pd.DataFrame]:
    frames = []
    for time_index, time_value in enumerate(times):
        latent = traj_x[time_index]
        frame = pd.DataFrame(
            {
                "kind": "trajectory",
                "time_index": int(time_index),
                "time": float(time_value),
                "time_label": _format_time_label(time_value),
                "point_id": np.arange(latent.shape[0], dtype=np.int64),
                "trajectory_id": np.arange(latent.shape[0], dtype=np.int64),
                "mass": traj_mass[time_index].astype(float),
                "plot_x": np.nan,
                "plot_y": np.nan,
            }
        )
        for name, values in _latent_columns("latent", latent).items():
            frame[name] = values
        frames.append(frame)
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PIUOT reconstructed trajectory points for downstream figures.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-epoch", "--checkpoint", dest="checkpoint_epoch", default="auto")
    parser.add_argument("--output-label", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-particles", type=int, default=2000)
    parser.add_argument("--n-trajectories", type=int, default=96)
    parser.add_argument("--n-dense-timepoints", type=int, default=50)
    parser.add_argument("--projection", choices=["pca", "first2"], default="pca")
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_mod, model_mod, train_mod = _load_runtime_modules()

    run_dir = _resolve_run_dir(args.run_name, int(args.seed))
    epoch_tag = _resolve_epoch_tag(run_dir, args.checkpoint_epoch)
    checkpoint_path = _checkpoint_path(run_dir, epoch_tag)

    cfg_dict = torch.load(run_dir / "config.pt", map_location="cpu")
    config = SimpleNamespace(**cfg_dict)
    x, y, config = config_mod.load_data(config)
    device = torch.device(str(args.device))
    x_device = _move_time_series_to_device(x, device)

    model = model_mod.ForwardSDE(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    start_x, _ = train_mod.p_samp(x_device[0], int(args.n_particles))
    observed_times = np.asarray(y, dtype=np.float64)
    dense_times = np.linspace(float(observed_times[0]), float(observed_times[-1]), int(args.n_dense_timepoints))

    pred_x_t, pred_mass_t = _rollout(model, train_mod, config, start_x, observed_times)
    traj_count = min(int(args.n_trajectories), int(start_x.shape[0]))
    traj_x_t, traj_mass_t = _rollout(model, train_mod, config, start_x[:traj_count], dense_times)

    observed_np = [x_t.detach().cpu().numpy() for x_t in x]
    observed_proj, pred_proj, traj_proj, projection_name = _projection_matrix(
        observed_np,
        pred_x_t.numpy(),
        traj_x_t.numpy(),
        method=args.projection,
        seed=int(args.seed),
    )

    pred_np = pred_x_t.numpy()
    traj_np = traj_x_t.numpy()
    pred_mass_np = pred_mass_t.numpy()
    traj_mass_np = traj_mass_t.numpy()

    predicted_frames = _predicted_rows(pred_np, pred_mass_np, observed_times)
    trajectory_frames = _trajectory_rows(traj_np, traj_mass_np, dense_times)
    for time_index, frame in enumerate(predicted_frames):
        frame["plot_x"] = pred_proj[time_index, :, 0].astype(float)
        frame["plot_y"] = pred_proj[time_index, :, 1].astype(float)
    for time_index, frame in enumerate(trajectory_frames):
        frame["plot_x"] = traj_proj[time_index, :, 0].astype(float)
        frame["plot_y"] = traj_proj[time_index, :, 1].astype(float)

    label = str(args.output_label or args.run_name)
    prefix = _slugify(args.output_prefix or label)
    output_dir = args.output_dir or (OUTPUT_ROOT / "figs" / label / "trajectory_points")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points_csv = output_dir / f"{prefix}_trajectory_points.csv"
    observed_csv = output_dir / f"{prefix}_observed_points.csv"
    predicted_csv = output_dir / f"{prefix}_predicted_points.csv"
    trajectory_csv = output_dir / f"{prefix}_dense_trajectory_points.csv"
    manifest_path = output_dir / f"{prefix}_trajectory_points_manifest.json"

    observed_df = pd.concat(_observed_rows(x, y, observed_proj), ignore_index=True)
    predicted_df = pd.concat(predicted_frames, ignore_index=True)
    trajectory_df = pd.concat(trajectory_frames, ignore_index=True)
    all_df = pd.concat([observed_df, predicted_df, trajectory_df], ignore_index=True)

    all_df.to_csv(points_csv, index=False)
    observed_df.to_csv(observed_csv, index=False)
    predicted_df.to_csv(predicted_csv, index=False)
    trajectory_df.to_csv(trajectory_csv, index=False)

    manifest = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "selected_epoch": epoch_tag,
        "checkpoint_path": str(checkpoint_path),
        "output_label": label,
        "projection": projection_name,
        "observed_times": [float(v) for v in observed_times.tolist()],
        "dense_times": [float(v) for v in dense_times.tolist()],
        "n_particles": int(args.n_particles),
        "n_trajectories": int(traj_count),
        "artifacts": {
            "points_csv": str(points_csv),
            "observed_csv": str(observed_csv),
            "predicted_csv": str(predicted_csv),
            "trajectory_csv": str(trajectory_csv),
            "manifest_json": str(manifest_path),
        },
        "columns": {
            "kind": "observed | predicted | trajectory",
            "x": "plot_x",
            "y": "plot_y",
            "time": "time",
            "trajectory_id": "trajectory_id",
            "mass": "mass",
            "latent_prefix": "latent_",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
