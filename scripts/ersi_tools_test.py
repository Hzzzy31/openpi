import dataclasses
import pathlib
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi import transforms
from openpi.ersi import dataset
from openpi.ersi import g2
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader
from openpi.training import optimizer
from openpi.training import weight_loaders
from scripts import compute_ersi_norm_stats
from scripts import train
from scripts import train_ersi


def test_g2_stats_match_the_training_pipeline(tmp_path, monkeypatch):
    base = config.get_ersi_g2_config()
    model_config = dataclasses.replace(base.model, action_horizon=3)
    data_config = dataclasses.replace(base.data, assets=dataclasses.replace(base.data.assets, asset_id="g2_test"))
    train_config = dataclasses.replace(base, model=model_config, data=data_config)
    monkeypatch.setattr(config, "get_ersi_g2_config", lambda _: train_config)
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *_: transforms.Group())

    episodes = {0: {"length": 2}, 1: {"length": 4}}
    metadata = SimpleNamespace(episodes=episodes, get_data_file_path=lambda index: f"episode_{index}.parquet")
    monkeypatch.setattr(dataset, "load_metadata", lambda _: metadata)
    samples = []
    for episode_id, episode in episodes.items():
        length = episode["length"]
        states = np.arange(length * g2.JOINT_DIM, dtype=np.float32).reshape(length, g2.JOINT_DIM) / 8 + episode_id
        actions = states * 0.75 + np.arange(g2.JOINT_DIM, dtype=np.float32)
        vector = pa.list_(pa.float32(), g2.JOINT_DIM)
        table = pa.table({
            g2.STATE_KEY: pa.array(states.tolist(), type=vector),
            g2.ACTION_KEY: pa.array(actions.tolist(), type=vector),
        })
        pq.write_table(table, tmp_path / metadata.get_data_file_path(episode_id))
        for i, state in enumerate(states):
            frames = [min(frame, length - 1) for frame in range(i, i + model_config.action_horizon)]
            samples.append({
                g2.STATE_KEY: state,
                g2.ACTION_KEY: actions[frames],
                "prompt": "test",
                **{feature: np.zeros((3, 4, 5), dtype=np.float32) for _, feature, _ in g2.CAMERAS},
            })

    assets = tmp_path / "assets"
    compute_ersi_norm_stats.main([str(tmp_path), "--config", base.name, "--assets-dir", str(assets)])
    actual = normalize.load(assets / train_config.norm_asset_id)
    transformed = data_loader.transform_dataset(samples, data_config.create(assets, model_config), skip_norm_stats=True)
    for name in ("state", "actions"):
        expected = normalize.RunningStats()
        offset = 0
        for episode in episodes.values():
            stop = offset + episode["length"]
            expected.update(np.stack([transformed[i][name] for i in range(offset, stop)]))
            offset = stop
        statistics = expected.get_statistics()
        for field in ("mean", "std", "q01", "q99"):
            np.testing.assert_allclose(getattr(actual[name], field), getattr(statistics, field), atol=1e-6)


@pytest.mark.parametrize("params_argument", [False, True])
def test_train_ersi_cli_maps_g2_recipe(tmp_path, monkeypatch, params_argument):
    base = config.get_ersi_g2_config()
    assets = tmp_path / "assets"
    (assets / base.norm_asset_id).mkdir(parents=True)
    (assets / base.norm_asset_id / "norm_stats.json").touch()
    # A superseded checkpoint still has the weights used to start a new run.
    checkpoint = tmp_path / "previous" / "2"
    params = checkpoint / "params"
    params.mkdir(parents=True)
    (params / "_METADATA").touch()
    monkeypatch.setattr(dataset, "load_metadata", lambda _: SimpleNamespace(total_frames=16))
    calls = []
    monkeypatch.setattr(train, "main", lambda cfg, *, progress_file: calls.append((cfg, progress_file)))
    output = tmp_path / "new" / "checkpoints"
    progress = output.parent / "openpi_progress.json"
    argv = [
        str(tmp_path / "data"),
        "--assets-dir", str(assets),
        "--output-dir", str(output),
        "--progress-file", str(progress),
        "--init-from", str(params if params_argument else checkpoint),
        "--steps", "5",
        "--save-interval", "2",
        "--batch-size", "4",
        "--warmup-steps", "1",
        "--peak-lr", "0.02",
        "--weight-decay", "0.003",
        "--clip-grad", "0.5",
        "--adam-beta1", "0.8",
        "--adam-beta2", "0.91",
        "--num-workers", "0",
        "--seed", "13",
    ]
    if params_argument:
        argv += ["--config", base.name]
    train_ersi.main(argv)
    [(cfg, progress_file)] = calls
    assert cfg.model == base.model
    assert pathlib.Path(cfg.data.lerobot_root) == tmp_path / "data"
    assert cfg.data.assets == config.AssetsConfig(assets_dir=str(assets), asset_id=base.norm_asset_id)
    assert cfg.checkpoint_dir == output
    assert progress_file == progress
    assert cfg.weight_loader == weight_loaders.CheckpointWeightLoader(str(params))
    assert isinstance(cfg.lr_schedule, optimizer.CosineDecaySchedule)
    assert cfg.lr_schedule == optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=0.02, decay_steps=5, decay_lr=0.002)
    assert isinstance(cfg.optimizer, optimizer.AdamW)
    assert cfg.optimizer == optimizer.AdamW(b1=0.8, b2=0.91, weight_decay=0.003, clip_gradient_norm=0.5)
    assert (cfg.num_train_steps, cfg.save_interval, cfg.keep_period) == (5, 2, 2)
    assert (cfg.batch_size, cfg.num_workers, cfg.seed) == (4, 0, 13)
    assert not cfg.resume


@pytest.mark.parametrize("entry", ["stats", "train"])
def test_ersi_cli_rejects_other_openpi_configs(entry, capsys):
    argv = ["no-dataset", "--assets-dir", "no-assets", "--config", "debug"]
    with pytest.raises(SystemExit) as error:
        if entry == "stats":
            compute_ersi_norm_stats.main(argv)
        else:
            train_ersi.main([*argv, "--output-dir", "no-output", "--init-from", "no-weights", "--steps", "1"])
    assert error.value.code == 2
    assert "argument --config: invalid choice" in capsys.readouterr().err
