MODEL=ssd_mobilenet_v2_320x320_coco17_tpu-8
PIPELINE_CONFIG_PATH=/home/ubuntu/balloon_project/models/${MODEL}/pipeline.config
MODEL_DIR=/home/ubuntu/balloon_project/models/${MODEL}/
TRAIN_CHECKPOINT_DIR=/home/ubuntu/balloon_project/models/${MODEL}/
EXPORT_DIR=/home/ubuntu/balloon_project/exported-models/${MODEL}/

python exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_dir ${TRAIN_CHECKPOINT_DIR} \
    --output_directory ${EXPORT_DIR}
    
