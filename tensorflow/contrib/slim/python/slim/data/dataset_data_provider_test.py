# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.data import test_utils


def _resize_image(image, height, width):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [height, width])
  return tf.squeeze(image, [0])


def _create_tfrecord_dataset(tmpdir):
  data_sources = test_utils.create_tfrecord_files(
      tmpdir,
      num_files=1)

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          shape=(), dtype=tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          shape=(), dtype=tf.string, default_value='jpeg'),
      'image/class/label': tf.FixedLenFeature(
          shape=[1], dtype=tf.int64,
          default_value=tf.zeros([1], dtype=tf.int64))
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
      data_sources=data_sources,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=100,
      items_to_descriptions=None)


class DatasetDataProviderTest(tf.test.TestCase):

  def testTFRecordDataset(self):
    height = 300
    width = 280

    with self.test_session():
      provider = slim.dataset_data_provider.DatasetDataProvider(
          _create_tfrecord_dataset(self.get_temp_dir()))
      image, label = provider.get(['image', 'label'])
      image = _resize_image(image, height, width)

      with tf.Session('') as sess:
        with slim.queues.QueueRunners(sess):
          image, label = sess.run([image, label])
      self.assertListEqual([height, width, 3], list(image.shape))
      self.assertListEqual([1], list(label.shape))

  def testTFRecordSeparateGetDataset(self):
    height = 300
    width = 280

    with self.test_session():
      provider = slim.dataset_data_provider.DatasetDataProvider(
          _create_tfrecord_dataset(self.get_temp_dir()))
    [image] = provider.get(['image'])
    [label] = provider.get(['label'])
    image = _resize_image(image, height, width)

    with tf.Session('') as sess:
      with slim.queues.QueueRunners(sess):
        image, label = sess.run([image, label])
      self.assertListEqual([height, width, 3], list(image.shape))
      self.assertListEqual([1], list(label.shape))


if __name__ == '__main__':
  tf.test.main()
