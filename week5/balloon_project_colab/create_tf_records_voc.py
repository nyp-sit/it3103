import fnmatch
import hashlib
import io
import logging
import os
import random
import re
import sys
import contextlib2
import numpy as np
import PIL.Image
#import tensorflow as tf
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from lxml import etree
import argparse

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util, label_map_util

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TFRecord converter")
parser.add_argument("-d",
                    "--data_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str)
parser.add_argument("-o",
                    "--output_dir",
                    help="Path to the folder where the tfrecords will be written to.",
                    type=str)
parser.add_argument("-l",
                    "--label_map",
                    help="Path to the labels (.pbtxt) file.", 
                    type=str)
parser.add_argument("-r",
                    "--test_ratio",
                    help="Test and train ratio ",
                    type=float, default=0.2)

args = parser.parse_args()


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory):
  
    img_path = os.path.join(image_subdirectory, data['filename'])
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    masks = []
    if 'object' in data:
        for obj in data['object']:

            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])

            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)
            class_name = obj['name']
            classes_text.append(class_name.encode('utf8'))
            ## this is the class id
            classes.append(label_map_dict[class_name])

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     xml_files,
                     faces_only=True,
                     mask_type='png'):
    
    output_tfrecords = tf.python_io.TFRecordWriter(output_filename)

    for idx, example in enumerate(xml_files):
        print(example)
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(xml_files))
        xml_path = os.path.join(annotations_dir, example)

        with tf.io.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        try:
            tf_example = dict_to_tf_example(
                data,
                label_map_dict,
                image_dir)

            if tf_example:
                output_tfrecords.write(tf_example.SerializeToString())

        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', xml_path)

# the script assuming the following directory structure:
# under the data_dir, there should be two subdirectories: images and annotations
# all the xml files are in the annotations directory.


def main(_):
    data_dir = args.data_dir
   
    label_map_dict = label_map_util.get_label_map_dict(args.label_map)
    output_dir = args.output_dir 
    test_ratio = args.test_ratio

    logging.info('Reading from dataset')
    image_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    xml_dir = annotations_dir
    xml_files = fnmatch.filter(os.listdir(xml_dir), '*.xml')
    random.seed(42)
    random.shuffle(xml_files)
    num_xml_files = len(xml_files)
    num_test = int(test_ratio * num_xml_files)
    test_annot_files = xml_files[:num_test]
    train_annot_files = xml_files[num_test:]

    logging.info('%d training and %d validation examples.',
                 len(train_annot_files), len(test_annot_files))

    train_output_path = os.path.join(output_dir, 'train.record')
    test_output_path = os.path.join(output_dir, 'val.record')

    create_tf_record(
        train_output_path,
        label_map_dict,
        annotations_dir,
        image_dir,
        train_annot_files)

    create_tf_record(
        test_output_path,
        label_map_dict,
        annotations_dir,
        image_dir,
        test_annot_files)

if __name__ == '__main__':
    tf.app.run()
