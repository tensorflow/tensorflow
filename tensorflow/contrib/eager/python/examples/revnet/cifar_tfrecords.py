# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

from absl import flags
from six.moves import cPickle as pickle
from six.moves import urllib
import tensorflow as tf

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'


def download_and_extract(data_dir):
  """Download CIFAR-10 if not already downloaded."""
  filepath = os.path.join(data_dir, CIFAR_FILENAME)
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  urllib.request.urlretrieve(CIFAR_DOWNLOAD_URL, filepath)
  tarfile.open(os.path.join(filepath), 'r:gz').extractall(data_dir)
  return filepath


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
  file_names['validation'] = ['data_batch_5']
  file_names['test'] = ['test_batch']
  return file_names


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict


def convert_to_tfrecord(input_files, output_file):
  """Converts files with pickled data to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      labels = data_dict[b'labels']
      num_entries_in_batch = len(labels)

      for i in range(num_entries_in_batch):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': _bytes_feature(data[i].tobytes()),
                    'label': _int64_feature(labels[i])
                }))
        record_writer.write(example.SerializeToString())


def main(_):
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  download_and_extract(FLAGS.data_dir)
  file_names = _get_file_names()
  input_dir = os.path.join(FLAGS.data_dir, CIFAR_LOCAL_FOLDER)

  for mode, files in file_names.items():
    input_files = [os.path.join(input_dir, f) for f in files]
    output_file = os.path.join(FLAGS.data_dir, mode + '.tfrecords')
    try:
      os.remove(output_file)
    except OSError:
      pass
    convert_to_tfrecord(input_files, output_file)
  print('Done!')


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_string(
      'data_dir',
      default=None,
      help='Directory to download and extract CIFAR-10 to.')

  tf.app.run(main)
