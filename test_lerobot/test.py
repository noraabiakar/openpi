import dataclasses
import os
from pathlib import Path

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import cv2
import numpy as np

def load_image_as_array_cv2(image_path, target_size=(224, 224)):
    # Read the image using OpenCV (Note: OpenCV reads images in BGR format)
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image
    img = cv2.resize(img, target_size)
    
    # Ensure it's the right dtype (should be already, but just to be safe)
    img = img.astype(np.uint8)
    
    # Ensure the shape is (224, 224, 3)
    assert img.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {img.shape}"
    return img

# Pi0 FAST
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
print("here 0")

from openpi.models import pi0_fast
custom_config = _config.TrainConfig(
    name="pi0_fast_custom",
    model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
    data=_config.LeRobotV2DataConfig(
        repo_id="noraabk/so101-goat-picking-v3",
        base_config=_config.DataConfig(prompt_from_task=True),
    ),
)
checkpoint_dir = Path("/home/azureuser/localfiles/openpi/checkpoint/")
print("here 1")

# Create a trained policy.
policy = _policy_config.create_trained_policy(custom_config, checkpoint_dir)
print("here 2")

example = {
    "observation/image": load_image_as_array_cv2("/home/azureuser/localfiles/openpi/test_lerobot/side.png"),
    "observation/wrist_image": load_image_as_array_cv2("/home/azureuser/localfiles/openpi/test_lerobot/wrist.png"),
    "observation/state": np.array([4.194260597229004,-39.18918991088867,49.68383026123047,32.617103576660156,-49.6099853515625,9.313077926635742, 0]),
    "prompt": "Put the red goat toy in the bowl",
}
result = policy.infer(example)
print(result)
print("here 3") 
