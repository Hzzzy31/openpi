import dataclasses

import pytest

from openpi.ersi import g2
from openpi.training import config


@pytest.mark.parametrize(
    ("config_name", "num_steps", "start_step", "save_interval", "expected"),
    [
        (g2.CONFIG_NAME, 5, 0, 2, [2, 4, 5]),
        (g2.CONFIG_NAME, 5, 2, 2, [4, 5]),
        (g2.CONFIG_NAME, 3, 0, 1, [1, 2, 3]),
        (g2.CONFIG_NAME, 1, 0, 1, [1]),
        (g2.CONFIG_NAME, 5, 0, 5, [5]),
        ("debug", 5, 0, 2, [2, 4]),
        ("debug", 5, 2, 2, [4]),
        ("debug", 1, 0, 1, [0]),
    ],
)
def test_checkpoint_schedule(config_name, num_steps, start_step, save_interval, expected):
    train_config = dataclasses.replace(
        config.get_config(config_name),
        num_train_steps=num_steps,
        save_interval=save_interval,
    )

    saved = []
    for step in range(start_step, num_steps):
        if (checkpoint_step := train_config.checkpoint_step_to_save(step, start_step)) is not None:
            saved.append(checkpoint_step)

    assert saved == expected
