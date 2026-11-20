import dataclasses
import json
from types import SimpleNamespace
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.ersi import checkpoints as ersi_checkpoints
from openpi.ersi import g2
from openpi.ersi import progress as ersi_progress
from openpi.models import model
from openpi.shared import normalize
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import optimizer
from openpi.training import utils as training_utils
from openpi.training import weight_loaders
from scripts import train
from scripts import train_ersi


@pytest.fixture
def toy_training(tmp_path, monkeypatch):
    previous_cache = jax.config.jax_compilation_cache_dir
    cache = str(tmp_path / "explicit-jax-cache")
    jax.config.update("jax_compilation_cache_dir", cache)
    stats = normalize.NormStats(np.zeros(g2.JOINT_DIM), np.ones(g2.JOINT_DIM))
    toy = SimpleNamespace(failed_step=None, nonfinite=np.nan, progress_file=None, snapshots=[], cache=cache)

    def create_data_loader(train_config, *, sharding, shuffle):
        del shuffle
        n = train_config.batch_size
        obs = model.Observation(
            images={"camera": np.zeros((n, 4, 5, 3), dtype=np.float32)},
            image_masks={"camera": np.ones(n, dtype=bool)},
            state=np.zeros((n, g2.JOINT_DIM), dtype=np.float32),
        )
        batch = jax.device_put((obs, np.zeros((n, 1, g2.JOINT_DIM), dtype=np.float32)), sharding)

        class ToyDataLoader:
            def __iter__(self):
                for index in range(train_config.num_train_steps):
                    if index and toy.progress_file is not None:
                        toy.snapshots.append(json.loads(toy.progress_file.read_text()))
                    yield batch

            def data_config(self):
                return config.DataConfig(asset_id=train_config.norm_asset_id, norm_stats={"state": stats})

        return ToyDataLoader()

    def init_state(train_config, init_rng, mesh, *, resume, learning_rate):
        assert not resume
        model_def, params = nnx.split(nnx.Linear(1, 1, rngs=nnx.Rngs(init_rng)))
        params.replace_by_pure_dict(train_config.weight_loader.load(params.to_pure_dict()))
        tx = optax.adam(learning_rate)
        state = training_utils.TrainState(
            step=jnp.asarray(0), params=params, model_def=model_def, opt_state=tx.init(params), tx=tx, ema_decay=None
        )
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        return jax.device_put(state, replicated), replicated

    def step(_config, _rng, state, _batch):
        completed = state.step + 1
        loss = completed.astype(jnp.float32)
        if toy.failed_step is not None:
            loss = jnp.where(completed == toy.failed_step, toy.nonfinite, loss)
        return state.replace(step=completed), {"loss": loss}

    monkeypatch.setattr(train, "init_logging", lambda: None)
    monkeypatch.setattr(train, "init_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.wandb, "Image", lambda *_: None)
    monkeypatch.setattr(train.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "init_train_state", init_state)
    monkeypatch.setattr(train, "train_step", step)
    monkeypatch.setattr("openpi.training.data_loader.create_data_loader", create_data_loader)
    try:
        yield toy
    finally:
        jax.config.update("jax_compilation_cache_dir", previous_cache)


def _training_config(output, save_interval):
    return dataclasses.replace(
        config.get_ersi_g2_config(),
        output_dir=str(output),
        batch_size=jax.device_count(),
        num_train_steps=5,
        save_interval=save_interval,
        keep_period=save_interval,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, decay_steps=5),
        weight_loader=weight_loaders.NoOpWeightLoader(),
    )


def _saved_steps(output):
    return sorted(int(path.name) for path in output.iterdir() if path.is_dir() and path.name.isdecimal())


def test_short_run_progress_and_checkpoint_retention(tmp_path, monkeypatch, toy_training):
    output = tmp_path / "run" / "checkpoints"
    selected = _training_config(output, save_interval=2)
    toy_training.progress_file = output.parent / ersi_progress.FILENAME
    train.main(selected, progress_file=toy_training.progress_file)

    first = toy_training.snapshots[0]
    schedule = selected.lr_schedule.create()
    assert first == {"step": 1, "action_loss": 1.0, "lr": pytest.approx(float(schedule(0)))}
    final = json.loads(toy_training.progress_file.read_text())
    assert final == {"step": 5, "action_loss": 5.0, "lr": pytest.approx(float(schedule(4)))}
    assert jax.config.jax_compilation_cache_dir == toy_training.cache
    assert _saved_steps(output) == [2, 4, 5]
    for step in (2, 4, 5):
        saved = output / str(step)
        assert (saved / "params" / "_METADATA").is_file()
        stats = normalize.load(saved / "assets" / selected.norm_asset_id)
        np.testing.assert_array_equal(stats["state"].mean, np.zeros(g2.JOINT_DIM))
        assert (saved / "train_state").is_dir() == (step == 5)

    # Warm-start from the params-only checkpoint through the real CLI and weight loader.
    earlier = output / "2"
    warmed = tmp_path / "warmed"
    monkeypatch.setattr(train_ersi.dataset, "load_metadata", lambda _: SimpleNamespace(total_frames=selected.batch_size))
    argv = [
        str(tmp_path / "data"),
        "--assets-dir", str(earlier / "assets"),
        "--init-from", str(earlier),
        "--output-dir", str(warmed),
        "--steps", "1",
        "--batch-size", str(selected.batch_size),
        "--seed", "876",
    ]
    train_ersi.main(argv)
    assert _saved_steps(warmed) == [1]
    warm_progress = json.loads((warmed / ersi_progress.FILENAME).read_text())
    assert warm_progress["step"] == 1
    base_rate = config.get_ersi_g2_config().lr_schedule.create()
    assert warm_progress["lr"] == pytest.approx(float(base_rate(0)))
    old_params = model.restore_params(earlier / "params", restore_type=np.ndarray)
    new_params = model.restore_params(warmed / "1" / "params", restore_type=np.ndarray)
    np.testing.assert_array_equal(old_params["kernel"], new_params["kernel"])
    assert (output / "5" / "train_state" / "_METADATA").is_file()
    assert not (earlier / "train_state").exists()


@pytest.mark.parametrize(
    ("failed_step", "nonfinite", "save_interval", "saved_steps"),
    [(1, np.nan, 2, []), (3, np.inf, 2, [2]), (5, np.nan, 2, [2, 4]), (3, np.nan, 5, [])],
)
def test_nonfinite_update_never_reaches_a_checkpoint(
    tmp_path, toy_training, failed_step, nonfinite, save_interval, saved_steps
):
    toy_training.failed_step = failed_step
    toy_training.nonfinite = nonfinite
    output = tmp_path / "output"
    selected = _training_config(output, save_interval)
    with pytest.raises(FloatingPointError, match=rf"Non-finite loss=.* at optimizer update {failed_step}"):
        train.main(selected)

    assert _saved_steps(output) == saved_steps
    actual = json.loads((output / ersi_progress.FILENAME).read_text())
    assert actual["step"] == failed_step - 1
    assert actual["failed_step"] == failed_step
    assert "Non-finite loss=" in actual["error"]
    if failed_step == 1:
        assert actual["action_loss"] is None
        assert actual["lr"] is None
    else:
        assert np.isfinite(actual["action_loss"])
        assert actual["lr"] == pytest.approx(float(selected.lr_schedule.create()(failed_step - 2)))
    if saved_steps:
        assert (output / str(saved_steps[-1]) / "train_state" / "_METADATA").is_file()


def _publish_fake_checkpoint(path):
    for name in ("_CHECKPOINT_METADATA", "train_state/_METADATA", "params/_METADATA", "assets/g2_rel14/norm_stats.json"):
        entry = path / name
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.touch()


def test_retention_waits_for_atomic_publication_and_can_retry(tmp_path, monkeypatch):
    manager = mock.Mock(directory=tmp_path)
    monkeypatch.setattr(checkpoints, "save_state", lambda *_: True)
    retention = ersi_checkpoints.TrainStateRetention(manager, config.get_ersi_g2_config().norm_asset_id)
    foreign = tmp_path / "1"
    first, second = tmp_path / "2", tmp_path / "4"
    _publish_fake_checkpoint(foreign)
    retention.save(None, None, 2)
    _publish_fake_checkpoint(first)
    retention.wait_until_finished()

    retention.save(None, None, 4)
    incomplete = tmp_path / "4.orbax-checkpoint-tmp-0"
    _publish_fake_checkpoint(incomplete)
    manager.is_saving_in_progress.return_value = True
    retention.poll()
    assert (first / "train_state" / "_METADATA").is_file()
    manager.wait_until_finished.side_effect = RuntimeError("save interrupted")
    with pytest.raises(RuntimeError, match="save interrupted"):
        retention.wait_until_finished()
    assert (first / "train_state" / "_METADATA").is_file()

    manager.wait_until_finished.side_effect = None
    with pytest.raises(RuntimeError, match="Missing items"):
        retention.wait_until_finished()
    incomplete.replace(second)
    retention.wait_until_finished()
    assert not (first / "train_state").exists()
    assert (first / "params" / "_METADATA").is_file()
    assert (first / "assets" / "g2_rel14" / "norm_stats.json").is_file()
    assert (second / "train_state" / "_METADATA").is_file()
    assert (foreign / "train_state" / "_METADATA").is_file()
