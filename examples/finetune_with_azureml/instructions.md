# Normalize data
az ml job create --file examples/finetune_with_azureml/data-norm-stats.yaml \
--resource-group robotics-ch-north-secure \
--workspace-name robotics-ch-north-secure \
--set compute=nabk-e4ds-v4 \
--set environment="azureml:openpi0-env:6" \
--set outputs.HF_LEROBOT_HOME.path="azureml://datastores/nora_datasets/paths/pi0_fast_custom/" \
--set inputs.repo_id="noraabk/dummy" \
--set inputs.wrist_image_key="observation.images.front" \
--set inputs.secondary_image_key="observation.images.side"


# Start finetuning
az ml job create --file examples/finetune_with_azureml/train.yaml \
--resource-group robotics-ch-north-secure \
--workspace-name robotics-ch-north-secure \
--set inputs.pi0_experiment_name="openpi0fast_dummy" \
--set compute="nc80adis-h100-v5-single" \
--set environment="azureml:openpi0-env:6" \
--set outputs.ASSETS_BASE.path="azureml://datastores/nora_datasets/paths/" \
--set outputs.HF_LEROBOT_HOME.path="azureml://datastores/nora_datasets/paths/pi0_fast_custom/" \
--set outputs.MODEL_CHECKPOINTS.path="azureml://datastores/nora_datasets/paths/finetuned_model/" \
--set inputs.extra_flags="--data.repo_id noraabk/dummy --data.wrist_image_key='observation.images.front' --data.secondary_image_key='observation.images.side' --overwrite"
