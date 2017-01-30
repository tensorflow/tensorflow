# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for slim.data.dataset_data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from tensorflow.contrib.slim.python.slim import queues
from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.contrib.slim.python.slim.data import test_utils
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


def _resize_image(image, height, width):
  image = array_ops.expand_dims(image, 0)
  image = image_ops.resize_bilinear(image, [height, width])
  return array_ops.squeeze(image, [0])


def _create_tfrecord_dataset(tmpdir):
  if not gfile.Exists(tmpdir):
    gfile.MakeDirs(tmpdir)

  data_sources = test_utils.create_tfrecord_files(tmpdir, num_files=1)

  keys_to_features = {
      'image/encoded':
          parsing_ops.FixedLenFeature(
              shape=(), dtype=dtypes.string, default_value=''),
      'image/format':
          parsing_ops.FixedLenFeature(
              shape=(), dtype=dtypes.string, default_value='jpeg'),
      'image/class/label':
          parsing_ops.FixedLenFeature(
              shape=[1],
              dtype=dtypes.int64,
              default_value=array_ops.zeros(
                  [1], dtype=dtypes.int64))
  }

  items_to_handlers = {
      'image': tfexample_decoder.Image(),
      'label': tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                               items_to_handlers)

  return dataset.Dataset(
      data_sources=data_sources,
      reader=io_ops.TFRecordReader,
      decoder=decoder,
      num_samples=100,
      items_to_descriptions=None)


class DatasetDataProviderTest(test.TestCase):

  def testTFRecordDataset(self):
    dataset_dir = tempfile.mkdtemp(prefix=os.path.join(self.get_temp_dir(),
                                                       'tfrecord_dataset'))

    height = 300
    width = 280

    with self.test_session():
      provider = dataset_data_provider.DatasetDataProvider(
          _create_tfrecord_dataset(dataset_dir))
      image, label = provider.get(['image', 'label'])
      image = _resize_image(image, height, width)

      with session.Session('') as sess:
        with queues.QueueRunners(sess):
          image, label = sess.run([image, label])
      self.assertListEqual([height, width, 3], list(image.shape))
      self.assertListEqual([1], list(label.shape))

  def testTFRecordSeparateGetDataset(self):
    dataset_dir = tempfile.mkdtemp(prefix=os.path.join(self.get_temp_dir(),
                                                       'tfrecord_separate_get'))

    height = 300
    width = 280

    with self.test_session():
      provider = dataset_data_provider.DatasetDataProvider(
          _create_tfrecord_dataset(dataset_dir))
    [image] = provider.get(['image'])
    [label] = provider.get(['label'])
    image = _resize_image(image, height, width)

    with session.Session('') as sess:
      with queues.QueueRunners(sess):
        image, label = sess.run([image, label])
      self.assertListEqual([height, width, 3], list(image.shape))
      self.assertListEqual([1], list(label.shape))


if __name__ == '__main__':
  test.main()
