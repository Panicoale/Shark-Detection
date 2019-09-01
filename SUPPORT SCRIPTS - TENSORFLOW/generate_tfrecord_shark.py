"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python3 generate_tfrecord_shark.py --csv_input=/home/alessandro/Shark/tf/workspace/training_demo2/annotation/train_labels.csv  --output_path=/home/alessandro/Shark/tf/workspace/training_demo2/annotation/train.record --image_dir=/home/alessandro/Shark/tf/workspace/training_demo2/images/train2

  python3 generate_tfrecord_shark.py --csv_input=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/testing_labels.csv  --output_path=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/testing.record --image_dir=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/testing



  python3 generate_tfrecord_shark.py --csv_input=/home/alessandro/Shark/tf/workspace/training_demo2/annotation/train_labels.csv  --output_path=/home/alessandro/Shark/tf/workspace/training_demo2/annotation/train.record --image_dir=/home/alessandro/Shark/tf/workspace/training_demo2/images/train2



python3 generate_tfrecord_shark.py --csv_input=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/train_labels3.csv  --output_path=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/train2.record --image_dir=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/train

python3 generate_tfrecord_shark.py --csv_input=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/test_labels3.csv  --output_path=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/test2.record --image_dir=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/test

python3 generate_tfrecord_shark.py --csv_input=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/validation_labels3.csv  --output_path=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/validation2.record --image_dir=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/validation



python3 generate_tfrecord_shark.py --csv_input=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/train_labels2.csv  --output_path=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/train.record --image_dir=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/train

python3 generate_tfrecord_shark.py --csv_input=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/test_labels2.csv  --output_path=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/test.record --image_dir=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/test

python3 generate_tfrecord_shark.py --csv_input=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/validation_labels2.csv  --output_path=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/validation.record --image_dir=/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/validation

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test.csv  --output_path=images/test.record

python3 generate_tfrecord2.py --csv_input=/home/alessandro/Shark/tf/workspace/training_demo2/annotation/test_labels.csv  --output_path=/home/alessandro/Shark/tf/workspace/training_demo2/annotation/test.record --image_dir=/home/alessandro/Shark/tf/workspace/training_demo2/images/test2/

python3 generate_tfrecord_shark.py --csv_input=/home/alessandro/Shark/tf/WS_shark/COM/annotation/test_labels.csv  --output_path=/home/alessandro/Shark/tf/WS_shark/COM/annotation/test.record --image_dir=/home/alessandro/Shark/tf/WS_shark/COM/test




"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '/home/alessandro/Shark/tf/workspace/training_demo/images/train/', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'shark':
        return 1
    elif row_label == 'whale':
        return 2
    elif row_label == 'dolphin':
        return 3
    elif row_label == 'turtle':
        return 4
    elif row_label == 'ray':
        return 5
    elif row_label == 'swimmer':
        return 6
    elif row_label == 'surfer':
        return 7
    elif row_label == 'boat':
        return 8
    elif row_label == 'rubbish':
        return 9
    elif row_label == 'buoy':
        return 10
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
