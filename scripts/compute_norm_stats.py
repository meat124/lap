"""Compute normalization statistics for a LAP training config.

Standalone implementation — no openpi runtime imports.

Usage:
    HF_LEROBOT_HOME=/path/to/dataset_root \\
    uv run scripts/compute_norm_stats.py lap_rby1

Stats are saved under:
    ./assets/<config_name>/<repo_id>/norm_stats.json

The norm_stats file is then automatically loaded during training via
DataConfigFactory._load_norm_stats().
"""

from __future__ import annotations

import functools
import json
import pathlib

from typing import Annotated

import numpy as np
import torch
import torch.utils.data
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

# Import LAP's config (not openpi's) so LAP-specific configs are available.
import lap.training.config as _config


# ---------------------------------------------------------------------------
# Minimal running-stats implementation (mirrors openpi.shared.normalize)
# ---------------------------------------------------------------------------

class RunningStats:
    """Online mean/std/quantile accumulator (Welford-style)."""

    _NUM_BINS = 5000

    def __init__(self) -> None:
        self._count = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None   # sum of squared deviations
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._hists: list[np.ndarray] | None = None
        self._edges: list[np.ndarray] | None = None

    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, batch.shape[-1]).astype(np.float64)
        n, d = batch.shape
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)

        if self._count == 0:
            self._mean = batch_mean.copy()
            self._m2 = (batch_var * n).copy()
            self._min = batch.min(axis=0)
            self._max = batch.max(axis=0)
            self._hists = [np.zeros(self._NUM_BINS) for _ in range(d)]
            self._edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._NUM_BINS + 1)
                for i in range(d)
            ]
        else:
            new_min = batch.min(axis=0)
            new_max = batch.max(axis=0)
            changed = np.any(new_min < self._min) or np.any(new_max > self._max)
            self._min = np.minimum(self._min, new_min)
            self._max = np.maximum(self._max, new_max)
            if changed:
                for i in range(d):
                    new_edges = np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._NUM_BINS + 1)
                    new_hist, _ = np.histogram(self._edges[i][:-1], bins=new_edges, weights=self._hists[i])
                    self._hists[i] = new_hist
                    self._edges[i] = new_edges

            # Parallel (Chan's) update for mean and M2
            delta = batch_mean - self._mean
            total = self._count + n
            self._mean += delta * (n / total)
            self._m2 += batch_var * n + delta ** 2 * (self._count * n / total)

        self._count += n
        for i in range(d):
            h, _ = np.histogram(batch[:, i], bins=self._edges[i])
            self._hists[i] += h

    def _quantile(self, q: float) -> np.ndarray:
        result = []
        target = q * self._count
        for hist, edges in zip(self._hists, self._edges):
            idx = np.searchsorted(np.cumsum(hist), target)
            result.append(edges[min(idx, len(edges) - 1)])
        return np.array(result)

    def get_statistics(self) -> dict:
        if self._count < 2:
            raise ValueError("Need at least 2 samples to compute statistics.")
        std = np.sqrt(np.maximum(0, self._m2 / self._count))
        return {
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "q01": self._quantile(0.01).tolist(),
            "q99": self._quantile(0.99).tolist(),
        }


def save_norm_stats(directory: pathlib.Path, norm_stats: dict[str, dict]) -> None:
    """Save norm stats as JSON in the format expected by openpi's normalize.load()."""
    path = directory / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"norm_stats": norm_stats}
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _compose(transforms):
    """Apply a list of transform callables left-to-right."""
    def apply(x):
        for t in transforms:
            x = t(x)
        return x
    return apply


class _TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn):
        self._dataset = dataset
        self._transform = transform_fn

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._transform(self._dataset[idx])


def _strip_strings(sample: dict) -> dict:
    """Remove string-valued keys (not needed for stats, break np.stack)."""
    return {
        k: v for k, v in sample.items()
        if not (isinstance(v, str) or (hasattr(v, "dtype") and np.issubdtype(np.asarray(v).dtype, np.str_)))
    }


def _numpy_collate(batch: list[dict]) -> dict:
    """Collate a list of sample dicts into a batched dict of numpy arrays."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        try:
            out[k] = np.stack([np.asarray(v) for v in vals])
        except Exception:
            pass  # skip non-stackable keys (strings etc.)
    return out


def build_dataloader(
    data_config,
    action_horizon: int,
    batch_size: int,
    num_workers: int,
    max_frames: int | None,
) -> tuple[torch.utils.data.DataLoader, int]:
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("data_config.repo_id is not set")

    meta = LeRobotDatasetMetadata(repo_id)

    # Episode filtering (num_episodes support)
    episodes = None
    num_episodes = getattr(data_config, 'num_episodes', None)
    if num_episodes is not None:
        total = meta.total_episodes
        n = min(num_episodes, total)
        episodes = list(range(n))
        print(f"num_episodes={num_episodes}: using first {n} of {total} episodes")

    delta_timestamps = {
        key: [t / meta.fps for t in range(action_horizon)]
        for key in data_config.action_sequence_keys
    }

    base_dataset = LeRobotDataset(repo_id, episodes=episodes, delta_timestamps=delta_timestamps, video_backend="pyav")

    # PromptFromLeRobotTask if needed
    transforms = []
    if data_config.prompt_from_task:
        from openpi.transforms import PromptFromLeRobotTask
        transforms.append(PromptFromLeRobotTask(meta.tasks))

    # repack + data transforms (no normalization, no model transforms)
    transforms += [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _strip_strings,
    ]

    dataset = _TransformedDataset(base_dataset, _compose(transforms))

    total_frames = len(dataset)
    if max_frames is not None and max_frames < total_frames:
        num_batches = max_frames // batch_size
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=max_frames)
    else:
        num_batches = total_frames // batch_size
        sampler = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=_numpy_collate,
        drop_last=True,
    )
    return loader, num_batches


# ---------------------------------------------------------------------------
# Fast parquet-based stats computation
# ---------------------------------------------------------------------------

def _compute_stats_from_parquet(
    data_dir: pathlib.Path,
    arm_indices: list[int] | None,
    max_frames: int | None,
) -> dict[str, RunningStats]:
    """Read parquet files directly — avoids video decoding entirely."""
    import glob
    import pandas as pd

    parquet_files = sorted(glob.glob(str(data_dir / "data/**/*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir / 'data'}")

    stats: dict[str, RunningStats] = {"state": RunningStats(), "actions": RunningStats()}
    total_seen = 0

    for fpath in tqdm.tqdm(parquet_files, desc="Reading parquets"):
        df = pd.read_parquet(fpath, columns=["state", "actions"])
        state_arr = np.stack(df["state"].values).astype(np.float32)
        action_arr = np.stack(df["actions"].values).astype(np.float32)

        if arm_indices is not None:
            state_arr = state_arr[:, arm_indices]
            action_arr = action_arr[:, arm_indices]

        stats["state"].update(state_arr)
        stats["actions"].update(action_arr)
        total_seen += len(state_arr)

        if max_frames is not None and total_seen >= max_frames:
            break

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    config_name: Annotated[str, tyro.conf.Positional],
    data_dir: pathlib.Path | None = None,
    max_frames: int | None = None,
    repo_id: str | None = None,
) -> None:
    """Compute normalization statistics for a LAP training config.

    Args:
        config_name: LAP config name (e.g. lap_rby1).
        data_dir: Root directory containing the LeRobot dataset (the folder that
            holds <repo_id>/data/*.parquet).  Overrides HF_LEROBOT_HOME.
        max_frames: If set, stop after this many frames.
        repo_id: Override the repo_id from the config.  Useful for computing
            stats on a dataset variant (e.g. PuttingCupintotheDishV2_50_eef)
            while reusing the same config (lap_rby1_eef).  Stats are saved
            under assets/<config_name>/<repo_id>/norm_stats.json.
    """
    config = _config.get_config(config_name)
    data_config_factory = config.data
    data_config = data_config_factory.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        raise ValueError(
            "compute_norm_stats.py only supports the LeRobot (non-RLDS) path. "
            "RLDS datasets handle normalization internally."
        )

    if repo_id is None:
        repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("data_config.repo_id is not set")

    # Locate the LeRobot dataset root
    import os
    if data_dir is not None:
        dataset_dir = data_dir / repo_id
    else:
        hf_home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
        dataset_dir = hf_home / repo_id

    # Determine arm dimension indices (for rby1: select right arm+gripper from 16-DOF)
    arm_indices: list[int] | None = None
    if getattr(data_config_factory, "right_arm_only", False):
        # modality order: 0-6 right_arm, 7-13 left_arm, 14 right_gripper, 15 left_gripper
        arm_indices = [0, 1, 2, 3, 4, 5, 6, 14]

    stats = _compute_stats_from_parquet(dataset_dir, arm_indices=arm_indices, max_frames=max_frames)

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = config.assets_dirs / repo_id
    save_norm_stats(output_path, norm_stats)
    print("Done.")


if __name__ == "__main__":
    tyro.cli(main)
