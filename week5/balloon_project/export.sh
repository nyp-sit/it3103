MODEL=ssd_mobilenet_v2_320x320_coco17_tpu-8
EXPERIMENT=run1
PIPELINE_CONFIG_PATH=/home/ubuntu/balloon_project/models/${MODEL}/${EXPERIMENT}/pipeline.config
MODEL_DIR=/home/ubuntu/balloon_project/models/${MODEL}/${EXPERIMENT}/
TRAIN_CHECKPOINT_DIR=/home/ubuntu/balloon_project/models/${MODEL}/${EXPERIMENT}/
EXPORT_DIR=/home/ubuntu/balloon_project/exported_models/${MODEL}/${EXPERIMENT}/

python exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_dir ${TRAIN_CHECKPOINT_DIR} \
    --output_directory ${EXPORT_DIR}
    
