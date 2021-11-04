MODEL=ssd_mobilenet_v2_320x320_coco17_tpu-8
CUDA_VISIBLE_DEVICES=0
PIPELINE_CONFIG_PATH=/home/ubuntu/balloon_project/models/${MODEL}/pipeline.config
MODEL_DIR=/home/ubuntu/balloon_project/models/${MODEL}

python model_main_tf2.py \
    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
    --model_dir="${MODEL_DIR}" \
    --checkpoint_every_n=100 \
    --alsologtostderr

