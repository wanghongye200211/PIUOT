from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


METHOD_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = METHOD_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding.common import (
    batch_size_from_config,
    choose_device,
    device_name_from_config,
    embedding_output_key_from_config,
    embedding_output_path_from_config,
    hidden_dims_from_config,
    latent_dim_from_config,
    learning_rate_from_config,
    load_adata_from_config,
    load_embedding_config,
    load_feature_matrix,
    maybe_standardize,
    noise_std_from_config,
    save_embedding_result,
    standardize_from_config,
    train_epochs_from_config,
    weight_decay_from_config,
)
from embedding.models import Autoencoder
from project_paths import DEFAULT_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a generic GAE-style embedding from a .h5ad file.")
    parser.add_argument("--config", "--yaml-config", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def train_gae_from_config(config_path: str | Path | None) -> Path:
    config = load_embedding_config(config_path)
    adata, input_path = load_adata_from_config(config)
    raw_matrix, source_desc = load_feature_matrix(adata, config)
    matrix, scaler_meta = maybe_standardize(raw_matrix, standardize_from_config(config))

    device = choose_device(device_name_from_config(config))
    latent_dim = latent_dim_from_config(config)
    hidden_dims = hidden_dims_from_config(config, input_dim=matrix.shape[1], latent_dim=latent_dim)
    output_key = embedding_output_key_from_config(config)
    output_path = embedding_output_path_from_config(config, input_path, output_key)

    model = Autoencoder(input_dim=matrix.shape[1], latent_dim=latent_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_from_config(config),
        weight_decay=weight_decay_from_config(config),
    )
    dataset = TensorDataset(torch.from_numpy(matrix))
    loader = DataLoader(dataset, batch_size=batch_size_from_config(config), shuffle=True, drop_last=False)

    noise_std = noise_std_from_config(config)
    epochs = train_epochs_from_config(config)
    last_loss = None
    model.train()
    for epoch in range(epochs):
        running = 0.0
        seen = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            input_batch = batch
            if noise_std > 0:
                input_batch = batch + noise_std * torch.randn_like(batch)
            reconstruction, _ = model(input_batch)
            loss = F.mse_loss(reconstruction, batch)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * batch.shape[0]
            seen += int(batch.shape[0])
        last_loss = running / max(seen, 1)
        print(f"[GAE] epoch {epoch + 1:03d}/{epochs:03d} loss={last_loss:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        all_tensor = torch.from_numpy(matrix).to(device)
        latent = model.encode(all_tensor).cpu().numpy().astype(np.float32, copy=False)

    metadata = {
        "method": "gae",
        "input_path": str(input_path),
        "input_source": source_desc,
        "output_key": output_key,
        "latent_dim": int(latent_dim),
        "hidden_dims": [int(v) for v in hidden_dims],
        "train_epochs": int(epochs),
        "learning_rate": float(learning_rate_from_config(config)),
        "noise_std": float(noise_std),
        "final_loss": None if last_loss is None else float(last_loss),
        "device": str(device),
    }
    metadata.update(scaler_meta)
    save_embedding_result(
        adata=adata,
        output_path=output_path,
        output_key=output_key,
        latent=latent,
        metadata=metadata,
    )
    print(f"[GAE] saved {output_key} to {output_path}", flush=True)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    train_gae_from_config(args.config_path)


if __name__ == "__main__":
    main()

