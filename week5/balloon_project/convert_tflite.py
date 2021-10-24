import tensorflow as tf 

saved_model_dir = "/home/markk/balloon_project/tflite/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#     tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]
tflite_model = converter.convert()

# Save the model.
with open('/home/markk/model.tflite', 'wb') as f:
  f.write(tflite_model)