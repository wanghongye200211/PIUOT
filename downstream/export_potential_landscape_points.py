#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIUOT_ROOT = PROJECT_ROOT / "piuot"
PIUOT_OUTPUT_ROOT = PIUOT_ROOT / "output"
DOWNSTREAM_OUTPUT_ROOT = PROJECT_ROOT / "downstream" / "output"

if str(PIUOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PIUOT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the points CSV required by build_potential_state_map.py.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory containing config.pt and train.*.pt.")
    parser.add_argument("--run-name", default=None, help="PIUOT run name under piuot/output.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="auto", help="auto, best, final, or epoch_000100 style tag.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--embedding-key", default=None)
    parser.add_argument("--time-obs-key", default=None)
    parser.add_argument("--annotation-key", default=None)
    parser.add_argument("--reduction", choices=["pca", "first2"], default="pca")
    parser.add_argument("--coords-obsm-key", default=None)
    parser.add_argument("--time-mode", choices=["obs", "fixed", "start", "end", "mid"], default="obs")
    parser.add_argument("--fixed-time", type=float, default=None)
    parser.add_argument("--orientation", choices=["auto_time_descend", "none", "flip"], default="auto_time_descend")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed-value", type=int, default=0)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _load_runtime_modules():
    src_pkg = importlib.import_module("core")
    sys.modules["src"] = src_pkg
    config_mod = importlib.import_module("src.config_model")
    model_mod = importlib.import_module("src.model")
    return config_mod, model_mod


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return slug or "piuot_run"


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run directory: {run_dir}")
        return run_dir
    if not args.run_name:
        raise ValueError("Provide either --run-dir or --run-name.")
    matches = sorted((PIUOT_OUTPUT_ROOT / args.run_name).glob(f"*/seed_{args.seed}/alltime"))
    if not matches:
        raise FileNotFoundError(f"Could not find run '{args.run_name}' with seed {args.seed}.")
    return matches[-1].resolve()


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


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> tuple[str, Path]:
    selector = str(checkpoint)
    if selector == "auto":
        epoch_tag = _best_epoch_from_eval(run_dir)
        if epoch_tag is not None and (run_dir / f"train.{epoch_tag}.pt").exists():
            return epoch_tag, (run_dir / f"train.{epoch_tag}.pt").resolve()
        if (run_dir / "train.best.pt").exists():
            return "best", (run_dir / "train.best.pt").resolve()
        selector = "final"
    if selector == "final":
        selector = _latest_epoch_from_checkpoints(run_dir)
    if selector == "best":
        path = run_dir / "train.best.pt"
    else:
        path = run_dir / f"train.{selector}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return selector, path.resolve()


def _infer_run_name(run_dir: Path) -> str:
    if run_dir.name == "alltime" and run_dir.parent.name.startswith("seed_") and len(run_dir.parents) >= 3:
        return run_dir.parents[2].name
    return run_dir.name


def _load_dict(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, SimpleNamespace):
        return dict(vars(payload))
    raise TypeError(f"Unsupported config payload in {path}: {type(payload)!r}")


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint)!r}")


def _to_dense_float32(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def _load_embedding(adata: ad.AnnData, embedding_key: str) -> np.ndarray:
    if embedding_key == "X":
        return _to_dense_float32(adata.X)
    if embedding_key not in adata.obsm:
        raise KeyError(f"{adata.filename or 'h5ad'} does not contain obsm['{embedding_key}'].")
    return _to_dense_float32(adata.obsm[embedding_key])


def _infer_k_dims(state_dict: dict[str, torch.Tensor]) -> list[int]:
    dims: list[int] = []
    for idx in range(1, 50):
        key = f"_func.net.linear{idx}.weight"
        if key not in state_dict:
            break
        dims.append(int(state_dict[key].shape[0]))
    return dims


def _complete_config(
    cfg: dict[str, Any],
    *,
    run_name: str,
    embedding_dim: int,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], list[str]]:
    filled: list[str] = []

    def set_default(name: str, value: Any) -> None:
        if name not in cfg or cfg[name] is None:
            cfg[name] = value
            filled.append(name)

    set_default("run_name", run_name)
    set_default("x_dim", int(embedding_dim))
    if "k_dims" not in cfg or cfg["k_dims"] is None:
        inferred = _infer_k_dims(state_dict)
        cfg["k_dims"] = inferred if inferred else [400, 400]
        filled.append("k_dims")
    if isinstance(cfg["k_dims"], tuple):
        cfg["k_dims"] = list(cfg["k_dims"])
    cfg["k_dims"] = [int(v) for v in cfg["k_dims"]]
    set_default("layers", len(cfg["k_dims"]))
    set_default("activation", "softplus")
    if "sigma_type" not in cfg or cfg["sigma_type"] is None:
        if any(key.startswith("_func.sigma._model") for key in state_dict):
            cfg["sigma_type"] = "Mlp"
        elif "_func.sigma" in state_dict:
            cfg["sigma_type"] = "const"
        else:
            cfg["sigma_type"] = "const"
        filled.append("sigma_type")
    if "sigma_const" not in cfg or cfg["sigma_const"] is None:
        sigma_tensor = state_dict.get("_func.sigma")
        cfg["sigma_const"] = float(sigma_tensor.reshape(-1)[0]) if sigma_tensor is not None else 0.1
        filled.append("sigma_const")
    set_default("solver_dt", 0.1)
    if "use_growth" not in cfg or cfg["use_growth"] is None:
        cfg["use_growth"] = any("growth_net" in key for key in state_dict)
        filled.append("use_growth")
    set_default("growth_mode", "bounded")
    set_default("growth_scale", 0.05)
    set_default("hjb_growth_coeff", 2.0)
    set_default("mass_clip_value", 30.0)
    return cfg, filled


def _format_time_label(value: Any) -> str:
    value = float(value)
    text = str(int(round(value))) if abs(value - round(value)) < 1e-8 else f"{value:g}"
    if 0.0 <= value <= 10.0:
        return f"Day{text}"
    return f"t={text}"


def _resolve_eval_times(time_obs: np.ndarray, *, mode: str, fixed_time: float | None) -> np.ndarray:
    time_obs = np.asarray(time_obs, dtype=np.float32)
    if mode == "obs":
        return time_obs.copy()
    t_min = float(np.min(time_obs))
    t_max = float(np.max(time_obs))
    if mode == "fixed":
        if fixed_time is None:
            raise ValueError("--fixed-time is required when --time-mode fixed.")
        value = float(fixed_time)
    elif mode == "start":
        value = t_min
    elif mode == "end":
        value = t_max
    elif mode == "mid":
        value = 0.5 * (t_min + t_max)
    else:
        raise ValueError(f"Unsupported time mode: {mode}")
    return np.full(time_obs.shape[0], value, dtype=np.float32)


def _evaluate_potential(
    model: torch.nn.Module,
    x: np.ndarray,
    t_eval: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], int(batch_size)):
            end = min(start + int(batch_size), x.shape[0])
            xb = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            tb = torch.tensor(t_eval[start:end, None], dtype=torch.float32, device=device)
            xt = torch.cat([xb, tb], dim=1)
            values = model._func.net(xt).detach().cpu().numpy().reshape(-1)
            outputs.append(values)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _orient_potential(phi_raw: np.ndarray, time_obs: np.ndarray, mode: str) -> tuple[np.ndarray, float, dict[str, float]]:
    phi = np.asarray(phi_raw, dtype=np.float32).copy()
    time_obs = np.asarray(time_obs, dtype=np.float32)
    by_time: dict[str, float] = {}
    for tv in np.sort(np.unique(time_obs)):
        mask = np.isclose(time_obs, tv, atol=1e-6)
        by_time[f"{float(tv):g}"] = float(np.mean(phi[mask]))
    if mode == "flip":
        return -phi, -1.0, by_time
    if mode == "none":
        return phi, 1.0, by_time
    time_means = np.asarray(list(by_time.values()), dtype=np.float32)
    if time_means.size >= 2 and time_means[-1] > time_means[0]:
        return -phi, -1.0, by_time
    return phi, 1.0, by_time


def _plot_coordinates(
    adata: ad.AnnData,
    x: np.ndarray,
    *,
    coords_obsm_key: str | None,
    reduction: str,
    seed: int,
) -> tuple[np.ndarray, str]:
    if coords_obsm_key:
        if coords_obsm_key not in adata.obsm:
            raise KeyError(f"h5ad does not contain obsm['{coords_obsm_key}'].")
        coords = _to_dense_float32(adata.obsm[coords_obsm_key])
        if coords.shape[1] < 2:
            raise ValueError(f"obsm['{coords_obsm_key}'] must have at least two columns.")
        plot_space = coords_obsm_key.lower().removeprefix("x_")
        return coords[:, :2], plot_space
    if reduction == "first2":
        if x.shape[1] < 2:
            raise ValueError("The selected embedding has fewer than two dimensions.")
        return x[:, :2].astype(np.float32), "embedding"
    coords = PCA(n_components=2, random_state=int(seed)).fit_transform(x)
    return coords.astype(np.float32), "pca"


def main() -> None:
    args = build_parser().parse_args()
    config_mod, model_mod = _load_runtime_modules()

    run_dir = _resolve_run_dir(args)
    epoch_tag, checkpoint_path = _resolve_checkpoint(run_dir, args.checkpoint)
    cfg = _load_dict(run_dir / "config.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)

    run_name = str(args.run_name or cfg.get("run_name") or _infer_run_name(run_dir))
    data_path = Path(args.data_path or cfg.get("data_path", "")).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing input h5ad: {data_path}. Pass --data-path if this run was moved."
        )
    data_path = data_path.resolve()
    embedding_key = str(args.embedding_key or cfg.get("embedding_key", "X_latent"))

    adata = ad.read_h5ad(data_path)
    x = _load_embedding(adata, embedding_key)
    cfg, filled_fields = _complete_config(
        cfg,
        run_name=run_name,
        embedding_dim=int(x.shape[1]),
        state_dict=state_dict,
    )
    config = SimpleNamespace(**cfg)

    time_key = str(args.time_obs_key or cfg.get("raw_time_key") or cfg.get("time_key", "t"))
    if time_key not in adata.obs.columns:
        raise KeyError(f"{data_path} does not contain obs['{time_key}'].")
    time_obs = np.asarray(adata.obs[time_key], dtype=np.float32)
    eval_times = _resolve_eval_times(time_obs, mode=args.time_mode, fixed_time=args.fixed_time)

    device = torch.device(str(args.device))
    model = model_mod.ForwardSDE(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    phi_raw = _evaluate_potential(
        model,
        x,
        eval_times,
        device=device,
        batch_size=int(args.batch_size),
    )
    phi_oriented, orientation_sign, time_means_before = _orient_potential(phi_raw, time_obs, args.orientation)
    phi_shifted = phi_oriented - float(np.min(phi_oriented))

    coords, plot_space = _plot_coordinates(
        adata,
        x,
        coords_obsm_key=args.coords_obsm_key,
        reduction=args.reduction,
        seed=args.seed_value,
    )

    time_labels = np.asarray([_format_time_label(v) for v in time_obs], dtype=object)
    if args.annotation_key and args.annotation_key in adata.obs.columns:
        annotation = np.asarray(adata.obs[args.annotation_key].astype(str), dtype=object)
        annotation_key = args.annotation_key
    else:
        annotation = time_labels
        annotation_key = None

    output_prefix = _slugify(args.output_prefix or run_name)
    output_dir = args.output_dir or (DOWNSTREAM_OUTPUT_ROOT / f"{output_prefix}_potential_landscape")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"{output_prefix}_potential_landscape_points.csv"
    df = pd.DataFrame(
        {
            "obs_name": np.asarray(adata.obs_names, dtype=object),
            "time_obs": time_obs.astype(float),
            "time_eval": eval_times.astype(float),
            "time_label": time_labels,
            "annotation": annotation,
            f"{plot_space}_1": coords[:, 0].astype(float),
            f"{plot_space}_2": coords[:, 1].astype(float),
            "plot_x": coords[:, 0].astype(float),
            "plot_y": coords[:, 1].astype(float),
            "potential_raw": phi_raw.astype(float),
            "potential_oriented": phi_oriented.astype(float),
            "potential_plot": phi_shifted.astype(float),
            "surface_z": phi_shifted.astype(float),
        }
    )
    df.to_csv(out_csv, index=False)

    summary = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "selected_epoch": epoch_tag,
        "selected_checkpoint": str(checkpoint_path),
        "input_h5ad": str(data_path),
        "embedding_key": embedding_key,
        "time_obs_key": time_key,
        "annotation_key": annotation_key,
        "plot_space": plot_space,
        "columns": {
            "x": "plot_x",
            "y": "plot_y",
            "z": "surface_z",
            "time": "time_label",
            "state": "annotation",
        },
        "filled_config_fields": filled_fields,
        "orientation_mode": args.orientation,
        "orientation_sign": float(orientation_sign),
        "time_mean_potential_before_orientation": time_means_before,
        "potential_stats": {
            "raw_min": float(np.min(phi_raw)),
            "raw_median": float(np.median(phi_raw)),
            "raw_max": float(np.max(phi_raw)),
            "plot_min": float(np.min(phi_shifted)),
            "plot_median": float(np.median(phi_shifted)),
            "plot_max": float(np.max(phi_shifted)),
        },
        "artifacts": {
            "points_csv": str(out_csv),
            "summary_json": str(output_dir / f"{output_prefix}_potential_landscape_summary.json"),
        },
    }
    summary_path = output_dir / f"{output_prefix}_potential_landscape_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
