import tensorflow as tf
import numpy as np
from PIL import Image
from six import BytesIO

def load_image_into_numpy_array(path):

    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    image = image.resize((300,300))
    (im_width, im_height) = image.size
        
    return np.array(image.getdata())[:,:3].reshape(
      (1, im_height, im_width, 3)).astype(np.float32)


img = load_image_into_numpy_array("/home/markk/git/it3103/week5/balloon_project/test_samples/2.png")
print(img.shape)
# saved_model_dir = "/home/markk/balloon_project/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
tflite_model_dir = "/home/markk/model.tflite"

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(tflite_model_dir)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
print(output_details[0])
print(output_details[1])
print(output_details[2])
print(output_details[3])

names = ['detection scores', "detection boxes", "num boxes", "detection classes" ]

for i in range(4):
  output_data = interpreter.get_tensor(output_details[i]['index'])
  print(f"{names[i]} = \n {output_data}")

# my_signature = interpreter.get_signature_runner("detect")
# print(my_signature)
# my_signature is callable with input as arguments.
#output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# output = my_signature(x=img)
# print(output)