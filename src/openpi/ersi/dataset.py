"""Read an ERSIBench G2 mix from its local LeRobot v2.1 directory."""

import json
import pathlib

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from openpi.ersi import g2


def load_metadata(root: pathlib.Path) -> LeRobotDatasetMetadata:
    # Required local files are checked before invoking LeRobot's Hub-aware loader.
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        if not (root / "meta" / name).is_file():
            raise FileNotFoundError(root / "meta" / name)
    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1" or info.get("robot_type") != "g2a":
        raise ValueError(f"{root}: expected a LeRobot v2.1 dataset exported for the ERSIBench G2.")

    features = info["features"]
    state, action = features[g2.STATE_KEY], features[g2.ACTION_KEY]
    joints = state.get("names") or []
    if (
        state["shape"] != [g2.JOINT_DIM]
        or action["shape"] != [g2.JOINT_DIM]
        or state["dtype"] != "float32"
        or action["dtype"] != "float32"
        or len(joints) != g2.JOINT_DIM
        or joints != action.get("names")
        or joints[-2:] != ["left_gripper", "right_gripper"]
    ):
        raise ValueError(f"{root}: expected the G2 joint layout (14 joints and two grippers).")
    camera_features = {feature for _, feature, _ in g2.CAMERAS}
    if {key for key, feature in features.items() if feature["dtype"] == "video"} != camera_features:
        raise ValueError(f"{root}: expected the three G2 camera video features.")

    metadata = LeRobotDatasetMetadata(g2.LOCAL_REPO_ID, root=root)
    if metadata.total_frames < 2 or sorted(metadata.episodes) != list(range(metadata.total_episodes)):
        raise ValueError(f"{root}: expected at least two G2 frames and contiguous episode indices.")
    for episode_index in metadata.episodes:
        files = [metadata.get_data_file_path(episode_index)]
        files.extend(metadata.get_video_file_path(episode_index, key) for key in camera_features)
        for file in files:
            if not (root / file).is_file():
                raise FileNotFoundError(root / file)
    return metadata
