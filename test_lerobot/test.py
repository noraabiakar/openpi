import os
from pathlib import Path

import numpy as np
from PIL import Image

from openpi.models import pi0_fast
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def load_image_as_array_pil(image_path, target_size=(224, 224)):
    # Read the image using PIL
    try:
        img = Image.open(image_path)
    except OSError as err:
        raise ValueError(f"Failed to load image: {image_path}") from err

    # Convert to RGB mode (in case it's grayscale or has alpha channel)
    img = img.convert("RGB")

    # Check current size before resizing
    current_size = img.size  # Returns (width, height)
    if current_size != target_size:
        # Resize the image only if needed
        img = img.resize(target_size, Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Ensure it's the right dtype
    img_array = img_array.astype(np.uint8)

    # Ensure the shape is (224, 224, 3)
    assert img_array.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {img_array.shape}"
    return img_array


# Pi0 FAST
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
print("here 0")

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
    "observation/image": load_image_as_array_pil("/home/azureuser/localfiles/openpi/test_lerobot/side.png"),
    "observation/wrist_image": load_image_as_array_pil("/home/azureuser/localfiles/openpi/test_lerobot/wrist.png"),
    "observation/state": np.array(
        [
            4.194260597229004,
            -39.18918991088867,
            49.68383026123047,
            32.617103576660156,
            -49.6099853515625,
            9.313077926635742,
            0,
        ]
    ),
    "prompt": "Put the red goat toy in the bowl",
}
result = policy.infer(example)
print(result)
print("here 3")
