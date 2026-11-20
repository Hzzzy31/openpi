"""Report checked ERSI optimizer updates independently of the training log."""

import json
import pathlib
import tempfile
import time

import jax
import numpy as np

from openpi.training import utils as training_utils

FILENAME = "openpi_progress.json"
_MAX_REPORT_SECONDS = 15.0


class TrainingProgress:
    def __init__(self, path: pathlib.Path, log_interval: int):
        self._path = path
        self._log_interval = log_interval
        self._last_report: float | None = None
        self._pending: list[dict[str, jax.Array]] = []
        self._action_loss: float | None = None
        self._lr: float | None = None

    def record(
        self,
        step: int,
        info: dict[str, jax.Array],
        state: training_utils.TrainState,
        *,
        checkpoint_due: bool,
    ) -> dict[str, float] | None:
        """Check the first update, each reporting window and every checkpoint boundary."""
        self._pending.append(info)
        if (
            not checkpoint_due
            and self._last_report is not None
            and len(self._pending) < self._log_interval
            and time.monotonic() - self._last_report < _MAX_REPORT_SECONDS
        ):
            return None

        # A dispatched JAX step can still be running. Await the optimizer state and transfer
        # just the scalar diagnostics in this window. Inspect each update before any averaging.
        jax.block_until_ready(state)
        keys = ("loss", *(key for key in info if key != "loss"))
        rows = np.asarray([[row[key] for key in keys] for row in jax.device_get(self._pending)], dtype=np.float64)
        self._pending.clear()
        invalid = np.argwhere(~np.isfinite(rows))
        error = None
        if invalid.size:
            index, column = map(int, invalid[0])
            step = step - len(rows) + index
            error = f"Non-finite {keys[column]}={rows[index, column]} at optimizer update {step + 1}"
            rows = rows[:index]

        means = None
        if len(rows):
            means = dict(zip(keys, rows.mean(axis=0).tolist(), strict=True))
            # The rate is the one used by the last completed update in this window.
            means["lr"] = float(rows[-1, keys.index("lr")])
            self._action_loss = means["loss"]
            self._lr = means["lr"]

        snapshot = {"step": step, "action_loss": self._action_loss, "lr": self._lr}
        if error is not None:
            self._write({**snapshot, "failed_step": step + 1, "error": error})
            raise FloatingPointError(error)
        self._write(snapshot)
        self._last_report = time.monotonic()
        return means

    def _write(self, snapshot: dict) -> None:
        if jax.process_index() != 0:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{self._path.name}.", dir=self._path.parent) as tmp:
            path = pathlib.Path(tmp) / self._path.name
            path.write_text(json.dumps(snapshot, allow_nan=False) + "\n", encoding="utf-8")
            path.replace(self._path)
