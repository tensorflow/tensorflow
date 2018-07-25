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
"""Read CIFAR data from pickled numpy arrays and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR dataset downloaded from
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

BASE_URL = 'https://www.cs.toronto.edu/~kriz/'
CIFAR_FILE_NAMES = ['cifar-10-python.tar.gz', 'cifar-100-python.tar.gz']
CIFAR_DOWNLOAD_URLS = [BASE_URL + name for name in CIFAR_FILE_NAMES]
CIFAR_LOCAL_FOLDERS = ['cifar-10', 'cifar-100']
EXTRACT_FOLDERS = ['cifar-10-batches-py', 'cifar-100-python']


def download_and_extract(data_dir, file_name, url):
  """Download CIFAR if not already downloaded."""
  filepath = os.path.join(data_dir, file_name)
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  urllib.request.urlretrieve(url, filepath)
  tarfile.open(os.path.join(filepath), 'r:gz').extractall(data_dir)
  return filepath


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names(folder):
  """Returns the file names expected to exist in the input_dir."""
  assert folder in ['cifar-10', 'cifar-100']

  file_names = {}
  if folder == 'cifar-10':
    file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
    file_names['validation'] = ['data_batch_5']
    file_names['train_all'] = ['data_batch_%d' % i for i in range(1, 6)]
    file_names['test'] = ['test_batch']
  else:
    file_names['train_all'] = ['train']
    file_names['test'] = ['test']
    # Split in `convert_to_tfrecord` function
    file_names['train'] = ['train']
    file_names['validation'] = ['train']
  return file_names


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict


def convert_to_tfrecord(input_files, output_file, folder):
  """Converts files with pickled data to TFRecords."""
  assert folder in ['cifar-10', 'cifar-100']

  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      try:
        labels = data_dict[b'labels']
      except KeyError:
        labels = data_dict[b'fine_labels']

      if folder == 'cifar-100' and input_file.endswith('train.tfrecords'):
        data = data[:40000]
        labels = labels[:40000]
      elif folder == 'cifar-100' and input_file.endswith(
          'validation.tfrecords'):
        data = data[40000:]
        labels = labels[40000:]

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
  for file_name, url, folder, extract_folder in zip(
      CIFAR_FILE_NAMES, CIFAR_DOWNLOAD_URLS, CIFAR_LOCAL_FOLDERS,
      EXTRACT_FOLDERS):
    print('Download from {} and extract.'.format(url))
    data_dir = os.path.join(FLAGS.data_dir, folder)
    download_and_extract(data_dir, file_name, url)
    file_names = _get_file_names(folder)
    input_dir = os.path.join(data_dir, extract_folder)

    for mode, files in file_names.items():
      input_files = [os.path.join(input_dir, f) for f in files]
      output_file = os.path.join(data_dir, mode + '.tfrecords')
      try:
        os.remove(output_file)
      except OSError:
        pass
      convert_to_tfrecord(input_files, output_file, folder)

  print('Done!')


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_string(
      'data_dir',
      default=None,
      help='Directory to download, extract and store TFRecords.')

  tf.app.run(main)
