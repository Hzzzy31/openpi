import numpy as np

from openpi import transforms
from openpi.ersi import g2


def test_g2_maps_bench_state_and_camera_formats():
    wire_state = np.arange(44, dtype=np.float32)
    wire_state[:2] = [-0.15, -0.25]
    expected_state = np.r_[wire_state[30:44], wire_state[:2]]
    images = {
        "top_head": np.full((4, 5, 3), 80, dtype=np.uint8),
        "hand_left": np.broadcast_to(np.array([0.0, 0.5, 1.0])[:, None, None], (3, 4, 5)),
        "hand_right": np.full((4, 5, 3), 120, dtype=np.uint8),
    }

    for state in (wire_state, expected_state):
        sample = g2.G2Inputs()({"state": state, "images": images, "prompt": "test"})
        np.testing.assert_array_equal(sample["state"], expected_state)
        assert sample["state"].dtype == np.float32
        assert sample["prompt"] == "test"
        assert sample["image_mask"] == {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": True}
        np.testing.assert_array_equal(sample["image"]["base_0_rgb"], images["top_head"])
        np.testing.assert_array_equal(sample["image"]["left_wrist_0_rgb"], np.broadcast_to([0, 128, 255], (4, 5, 3)))
        assert sample["image"]["left_wrist_0_rgb"].dtype == np.uint8
        np.testing.assert_array_equal(sample["image"]["right_wrist_0_rgb"], images["hand_right"])


def test_g2_round_trips_arm_deltas_and_absolute_grippers():
    state = np.arange(1, 17, dtype=np.float32)
    actions = np.stack([state + 1, state + 2])
    actions[:, 14:] = [[-0.25, -0.5], [-0.1, -0.8]]
    original = actions.copy()
    images = {wire: np.zeros((4, 5, 3), dtype=np.uint8) for wire, _, _ in g2.CAMERAS}
    obs = g2.G2Inputs()({"state": state, "actions": actions, "images": images})
    relative = transforms.DeltaActions(g2.DELTA_MASK)(obs)["actions"]
    np.testing.assert_array_equal(actions, original)
    np.testing.assert_array_equal(relative[:, :14], original[:, :14] - state[:14])
    np.testing.assert_array_equal(relative[:, 14:], original[:, 14:])

    output = transforms.compose([transforms.AbsoluteActions(g2.DELTA_MASK), g2.G2Outputs()])
    padded = {"state": np.pad(state, (0, 16)), "actions": np.pad(relative, ((0, 0), (0, 16)))}
    np.testing.assert_array_equal(output(padded)["actions"], original)
