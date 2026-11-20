"""Compute pi0.5 normalization statistics for a local ERSIBench G2 mix."""

import argparse
import pathlib
import tempfile

import numpy as np
import pyarrow.parquet as pq
import tqdm

from openpi import transforms
from openpi.ersi import dataset
from openpi.ersi import g2
from openpi.shared import normalize


def compute(root: pathlib.Path, assets_dir: pathlib.Path) -> None:
    metadata = dataset.load_metadata(root)
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    delta = transforms.DeltaActions(g2.DELTA_MASK)
    offsets = np.arange(g2.ACTION_HORIZON)
    frames_per_batch = 256

    # The normalization uses only the two numeric columns. Match LeRobot's
    # episode-boundary padding without decoding any video frames.
    for episode_index, episode in tqdm.tqdm(sorted(metadata.episodes.items()), unit="episode"):
        path = root / metadata.get_data_file_path(episode_index)
        table = pq.read_table(path, columns=[g2.STATE_KEY, g2.ACTION_KEY])
        states = np.asarray(table[g2.STATE_KEY].to_pylist(), dtype=np.float32)
        actions = np.asarray(table[g2.ACTION_KEY].to_pylist(), dtype=np.float32)
        length = episode["length"]
        if length < 1 or states.shape != (length, g2.JOINT_DIM) or actions.shape != states.shape:
            raise ValueError(f"{path}: unexpected shape for the recorded G2 state or action.")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"{path}: state and action must be finite.")

        state_stats.update(states)
        for start in range(0, length, frames_per_batch):
            indices = np.arange(start, min(start + frames_per_batch, length))
            chunk_indices = np.minimum(indices[:, None] + offsets, length - 1)
            batch = delta({"state": states[indices], "actions": actions[chunk_indices]})
            action_stats.update(batch["actions"])

    stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    output_dir = assets_dir / g2.ASSET_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        normalize.save(tmp, stats)
        output = output_dir / "norm_stats.json"
        (pathlib.Path(tmp) / output.name).replace(output)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--assets-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    compute(args.dataset.resolve(), args.assets_dir.resolve())


if __name__ == "__main__":
    main()
