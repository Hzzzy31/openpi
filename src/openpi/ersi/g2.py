"""Transforms for the ERSIBench G2 joint dataset and policy interface."""

import dataclasses

import numpy as np

from openpi import transforms

CONFIG_NAME = "openpi_pi05_g2_rel14"
ASSET_ID = "g2_rel14"
LOCAL_REPO_ID = "local/ersi-g2"
ACTION_HORIZON = 50
STATE_KEY = "observation.state"
ACTION_KEY = "action"

# Joint names in ERSIBench's g2_omnipicker embodiment and LeRobot exporter order.
ARM_JOINTS = (
    *(f"idx2{joint}_arm_l_joint{joint}" for joint in range(1, 8)),
    *(f"idx6{joint}_arm_r_joint{joint}" for joint in range(1, 8)),
)
JOINT_NAMES = (*ARM_JOINTS, "left_gripper", "right_gripper")
ARM_DIM = len(ARM_JOINTS)
JOINT_DIM = len(JOINT_NAMES)
DELTA_MASK = transforms.make_bool_mask(ARM_DIM, -(JOINT_DIM - ARM_DIM))

# Wire camera name, LeRobot feature and pi0.5 input name.
CAMERAS = (
    ("top_head", "observation.images.cam0_head", "base_0_rgb"),
    ("hand_left", "observation.images.cam1_hand_left", "left_wrist_0_rgb"),
    ("hand_right", "observation.images.cam2_hand_right", "right_wrist_0_rgb"),
)

DATASET_REPACK = transforms.RepackTransform(
    {
        "images": {wire: feature for wire, feature, _ in CAMERAS},
        "state": STATE_KEY,
        "actions": ACTION_KEY,
        "prompt": "prompt",
    }
)


def _rgb_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a three-dimensional G2 image, got {image.shape}.")
    if image.dtype == np.uint8 and image.shape[-1] == 3:
        return image
    if np.issubdtype(image.dtype, np.floating) and image.shape[0] == 3:
        # LeRobot decoders yield float CHW in [0, 1]. The policy receives RGB uint8 HWC.
        pixels = np.rint(255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
        return np.ascontiguousarray(np.moveaxis(pixels, 0, -1))
    raise ValueError(f"Expected RGB uint8 HWC or float CHW G2 image, got {image.dtype} {image.shape}.")


@dataclasses.dataclass(frozen=True)
class G2Inputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape == (44,):
            # The bench's agibot_g2 wire layout puts grippers first and arm joints last.
            state = np.concatenate((state[30:44], state[:2]))
        if state.shape != (JOINT_DIM,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite G2 state in 16D joint or 44D agibot_g2 layout.")

        images = {model: _rgb_image(data["images"][wire]) for wire, _, model in CAMERAS}
        sample = {"state": state, "image": images, "image_mask": dict.fromkeys(images, np.True_)}
        if "prompt" in data:
            sample["prompt"] = data["prompt"]
        if "actions" in data:
            # DeltaActions mutates its input. The dataset may expose the underlying action buffer.
            actions = np.array(data["actions"], dtype=np.float32, copy=True)
            if actions.ndim != 2 or actions.shape[-1] != JOINT_DIM or not np.isfinite(actions).all():
                raise ValueError("Expected a finite G2 action chunk with 16 values per timestep.")
            sample["actions"] = actions
        return sample


@dataclasses.dataclass(frozen=True)
class G2Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :JOINT_DIM], dtype=np.float32)}
