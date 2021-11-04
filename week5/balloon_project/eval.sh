export MODEL=ssd_mobilenet_v2_320x320_coco17_tpu-8
export CUDA_VISIBLE_DEVICES=""
export PIPELINE_CONFIG_PATH=/home/ubuntu/balloon_project/models/${MODEL}/pipeline.config
export MODEL_DIR=/home/ubuntu/balloon_project/models/${MODEL}/
export CHECKPOINT_DIR=/home/ubuntu/balloon_project/models/${MODEL}/
python model_main_tf2.py \
    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
    --model_dir="${MODEL_DIR}" \
    --checkpoint_dir="${CHECKPOINT_DIR}" \
    --alsologtostderr
