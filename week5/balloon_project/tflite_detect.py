#from tflite_runtime.interpreter import Interpreter, load_delegate
import tensorflow as tf
import argparse
import time
import cv2
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from six import BytesIO

def draw_image(image, results, labels, size):
    result_size = len(results)
    for idx, obj in enumerate(results):
        print(obj)
        # Prepare image for drawing
        draw = ImageDraw.Draw(image)

        # Prepare boundary box
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * size[0])
        xmax = int(xmax * size[0])
        ymin = int(ymin * size[1])
        ymax = int(ymax * size[1])

        # Draw rectangle to desired thickness
        for x in range( 0, 4 ):
            draw.rectangle((ymin, xmin, ymax, xmax), outline=(255, 255, 0))

        # Annotate image with label and confidence score
        display_str = labels[obj['class_id']] + ": " + str(round(obj['score']*100, 2)) + "%"
        draw.text((ymin,xmin), display_str, font=ImageFont.truetype("/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf", 20))

        displayImage = np.asarray( image )
        cv2.imshow('Coral Live Object Detection', displayImage)


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def make_interpreter(model_file):
    #model_file, *device = model_file.split('@')
    return tf.lite.Interpreter(model_path=model_file)


def load_image(path):

    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    image = image.convert("RGB")
    image = image.resize((300,300))
    return image
    # (im_width, im_height) = image.size

    # return np.array(image.getdata())[:,:3].reshape(
    #   (1, im_height, im_width, 3)).astype(np.float32)


img = load_image("/home/ubuntu/git/it3103/week5/balloon_project/test_samples/sample_balloon.jpeg")
model_file = "/home/ubuntu/model.tflite"
labels = load_labels("/home/ubuntu/labels.txt")
interpreter = make_interpreter(model_file)
interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
results = detect_objects(interpreter, img, 0.4)
print(results)
draw_image(img, results, labels, img.size)

