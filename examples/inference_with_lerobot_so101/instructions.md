# Installation 

Install openpi as usual 
```
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
Install extra feetech-servo-sdk
```
uv add feetech-servo-sdk>=1.0.0
```

# Run inference on so101
```
uv run ./examples/inference_with_lerobot_so101/infer.py \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
--robot.id=follower_arm \
--prompt="Put the red goat toy in the bowl" \
--checkpoint_dir="/storage/models/openpi0fast_200_episodes_PI_impl" \ # <-- path to checkpoint, should contain assets, _CHECKPOINT_METADATA, params
--lerobot_repo_id="noraabk/so101-goat-picking-v3" \ # <-- name of the original dataset, used to load the norm stats which will be under the assets folder in your checkpoint
--wrist_image_key="observation.images.front" \
--secondary_image_key="observation.images.side"
```

> Note: if you have strange errors try disconnecting and reconnetcing the robot from power. 