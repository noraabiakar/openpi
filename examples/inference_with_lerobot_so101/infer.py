from dataclasses import dataclass
import dataclasses
import os
from pathlib import Path

import numpy as np
from PIL import Image

from openpi.models import pi0_fast
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from lerobot.common.datasets.utils import build_dataset_frame
from lerobot.common.robots.config import RobotConfig
from lerobot.common.robots.so101_follower import SO101Follower
from lerobot.configs import parser
from lerobot.record import DatasetRecordConfig


@dataclass
class RecordConfig:
    robot: RobotConfig
    checkpoint_dir: Path
    prompt: str
    lerobot_repo_id: str
    wrist_image_key: str
    secondary_image_key: str
    total_inference_steps: int = 1000

@parser.wrap()
def main(cfg: RecordConfig):
    # Pi0 FAST
    base_config = _config.get_config("pi0_fast_custom")

    base_data_config = base_config.data
    data_config = dataclasses.replace(
        base_data_config,
        repo_id=cfg.lerobot_repo_id,
        wrist_image_key=cfg.wrist_image_key,
        secondary_image_key=cfg.secondary_image_key,
    )
    custom_config = dataclasses.replace(base_config, data=data_config)

    # Create a trained policy.
    policy = _policy_config.create_trained_policy(custom_config, cfg.checkpoint_dir)

    # Create a robot instance and connect to it.
    robot = SO101Follower(cfg.robot)
    robot.connect()

    # Prepare features that match the original lerobot dataset.
    dataset_features = {
        'action': {
            'dtype': 'float32', 
            'shape': (6,), 
            'names': [
                'shoulder_pan.pos', 
                'shoulder_lift.pos', 
                'elbow_flex.pos', 
                'wrist_flex.pos', 
                'wrist_roll.pos', 
                'gripper.pos',
            ],
        }, 
        'observation.state': {
            'dtype': 'float32', 
            'shape': (6,), 
            'names': [
                'shoulder_pan.pos', 
                'shoulder_lift.pos', 
                'elbow_flex.pos', 
                'wrist_flex.pos', 
                'wrist_roll.pos', 
                'gripper.pos',
            ],
        }, 
        'observation.images.front': {
            'dtype': 'video', 
            'shape': (480, 640, 3), 
            'names': [
                'height',
                'width',
                'channels',
            ],
        }, 
        'observation.images.side': {
            'dtype': 'video',
            'shape': (480, 640, 3),
            'names': [
                'height',
                'width',
                'channels',
            ],
        },
    }

    action_queue = []
    for i in range(cfg.total_inference_steps):
        # Get the robot obervation and state
        observation = robot.get_observation()
        observation_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
        
        # Rename the keys to match the expected input of the policy
        renamings = {
            "observation/image": "observation.images.side",
            "observation/wrist_image": "observation.images.front",
            "observation/state": "observation.state",
            "prompt": "prompt",
        }
        observation_frame["prompt"] = cfg.prompt
        observation_frame = {k: observation_frame[v] for k,v in renamings.items()}

        if not action_queue:
            print(f"Run inference for step {i}")
            action_values = policy.infer(observation_frame)["actions"]
            # Assume action_values is a 2D array: (batch, action_dim)
            # Fill the queue with all actions in the batch
            for av in action_values:
                action_queue.append({key: av[i] for i, key in enumerate(robot.action_features)})
        
        # Pop the next action from the queue
        action = action_queue.pop(0)
        print(f"ACTION {i}: {action}")

        robot.send_action(action)
    
    print("Done")
    return

if __name__ == "__main__":
    main()