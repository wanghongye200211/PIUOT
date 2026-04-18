from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from piuot.yaml_config import (
    device_from_config,
    embedding_key_from_config,
    load_yaml_config,
    reduction_epoch_from_config,
)


def to_dense_float32(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def load_embedding_config(config_path: str | Path | None) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    config.setdefault("embedding", {})
    return config


def load_adata_from_config(config: dict[str, Any]) -> tuple[ad.AnnData, Path]:
    data_path = Path(config["data"]["path"]).expanduser().resolve()
    return ad.read_h5ad(data_path), data_path


def embedding_input_key_from_config(config: dict[str, Any]) -> str:
    return str(config.get("embedding", {}).get("input_key", "X") or "X").strip()


def embedding_output_key_from_config(config: dict[str, Any]) -> str:
    return embedding_key_from_config(config)


def embedding_output_path_from_config(
    config: dict[str, Any],
    input_path: Path,
    output_key: str,
) -> Path:
    explicit = config.get("embedding", {}).get("output_path")
    if explicit not in (None, ""):
        return Path(explicit).expanduser().resolve()
    return input_path


def latent_dim_from_config(config: dict[str, Any]) -> int:
    explicit = config.get("embedding", {}).get("latent_dim")
    if explicit not in (None, ""):
        return int(explicit)
    return reduction_epoch_from_config(config)


def hidden_dims_from_config(config: dict[str, Any], input_dim: int, latent_dim: int) -> list[int]:
    raw = config.get("embedding", {}).get("hidden_dims")
    if isinstance(raw, list) and raw:
        return [int(v) for v in raw if int(v) > latent_dim]

    first = max(128, min(512, max(input_dim // 2, latent_dim * 8)))
    second = max(64, min(256, max(first // 2, latent_dim * 4)))
    dims = []
    for value in (first, second):
        if value > latent_dim and (not dims or dims[-1] != value):
            dims.append(value)
    return dims or [max(latent_dim * 4, 64)]


def batch_size_from_config(config: dict[str, Any]) -> int:
    return int(config.get("embedding", {}).get("batch_size", 256))


def learning_rate_from_config(config: dict[str, Any]) -> float:
    return float(config.get("embedding", {}).get("learning_rate", 1e-3))


def weight_decay_from_config(config: dict[str, Any]) -> float:
    return float(config.get("embedding", {}).get("weight_decay", 0.0))


def train_epochs_from_config(config: dict[str, Any]) -> int:
    return int(config.get("embedding", {}).get("train_epochs", 150))


def standardize_from_config(config: dict[str, Any]) -> bool:
    return bool(config.get("embedding", {}).get("standardize", True))


def noise_std_from_config(config: dict[str, Any]) -> float:
    return float(config.get("embedding", {}).get("noise_std", 0.0))


def distance_weight_from_config(config: dict[str, Any]) -> float:
    return float(config.get("embedding", {}).get("distance_weight", 1.0))


def reconstruction_weight_from_config(config: dict[str, Any]) -> float:
    return float(config.get("embedding", {}).get("reconstruction_weight", 1.0))


def device_name_from_config(config: dict[str, Any]) -> str:
    return device_from_config(config, fallback="cpu")


def choose_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device.type is 'cuda' but CUDA is not available.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device.type is 'mps' but MPS is not available.")
    return torch.device(device_name)


def load_feature_matrix(adata: ad.AnnData, config: dict[str, Any]) -> tuple[np.ndarray, str]:
    input_key = embedding_input_key_from_config(config)
    if input_key == "X":
        matrix = to_dense_float32(adata.X)
        source_desc = "adata.X"
    elif input_key in adata.obsm:
        matrix = to_dense_float32(adata.obsm[input_key])
        source_desc = f"adata.obsm['{input_key}']"
    elif input_key in adata.layers:
        matrix = to_dense_float32(adata.layers[input_key])
        source_desc = f"adata.layers['{input_key}']"
    else:
        raise KeyError(
            f"Embedding input key '{input_key}' not found in adata.X / adata.obsm / adata.layers."
        )

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix for embedding input, got shape {matrix.shape}.")
    if matrix.shape[0] < 2:
        raise ValueError("Need at least two cells to build an embedding.")
    return matrix, source_desc


def maybe_standardize(matrix: np.ndarray, enabled: bool) -> tuple[np.ndarray, dict[str, Any]]:
    if not enabled:
        return matrix.astype(np.float32, copy=False), {"standardized": False}

    scaler = StandardScaler(with_mean=True, with_std=True)
    transformed = scaler.fit_transform(matrix).astype(np.float32, copy=False)
    meta = {
        "standardized": True,
        "feature_mean": scaler.mean_.astype(np.float32),
        "feature_scale": scaler.scale_.astype(np.float32),
    }
    return transformed, meta


def save_embedding_result(
    *,
    adata: ad.AnnData,
    output_path: Path,
    output_key: str,
    latent: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    adata_out = adata.copy()
    adata_out.obsm[output_key] = np.asarray(latent, dtype=np.float32)
    adata_out.uns.setdefault("embedding_runs", {})
    adata_out.uns["embedding_runs"][output_key] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_out.write_h5ad(output_path)
