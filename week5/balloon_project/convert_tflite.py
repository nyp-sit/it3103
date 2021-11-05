import tensorflow as tf 

# this points to the directory where you export the model 
saved_model_dir = "/home/markk/balloon_project/exported_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/run1/saved_model"
tflite_model_dir = "/home/markk/balloon_project"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory

tflite_model = converter.convert()

# Save the TF-lite model.
with open('/home/markk/model.tflite', 'wb') as f:
  f.write(tflite_model)