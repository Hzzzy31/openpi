"""Keep full training state for the newest checkpoint published by this run."""

import logging
import pathlib
import shutil

import jax
import orbax.checkpoint as ocp

from openpi.training import checkpoints
from openpi.training import data_loader as _data_loader
from openpi.training import utils as training_utils


class TrainStateRetention:
    def __init__(self, manager: ocp.CheckpointManager, asset_id: str):
        self._manager = manager
        self._asset_id = asset_id
        self._latest_step: int | None = None
        self._pending_step: int | None = None

    def save(self, state: training_utils.TrainState, data_loader: _data_loader.DataLoader, step: int) -> None:
        if self._pending_step is not None:
            # Orbax already waits for any previous save before starting another one.
            self.wait_until_finished()
        if checkpoints.save_state(self._manager, state, data_loader, step):
            self._pending_step = step

    def poll(self) -> None:
        if self._pending_step is not None and not self._manager.is_saving_in_progress():
            self.wait_until_finished()

    def wait_until_finished(self) -> None:
        self._manager.wait_until_finished()
        if self._pending_step is None:
            return

        root = pathlib.Path(self._manager.directory)
        published = root / str(self._pending_step)
        required = (
            published / "_CHECKPOINT_METADATA",
            published / "params" / "_METADATA",
            published / "train_state" / "_METADATA",
            published / "assets" / self._asset_id / "norm_stats.json",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"Missing items in finalized ERSI checkpoint {published}")

        if self._latest_step is not None and jax.process_index() == 0:
            old_state = root / str(self._latest_step) / "train_state"
            try:
                shutil.rmtree(old_state)
            except FileNotFoundError:
                if old_state.exists():
                    raise
            else:
                logging.info("Removed superseded training state at %s", old_state)
        self._latest_step = self._pending_step
        self._pending_step = None
