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
"""Script for reading and loading CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# Global constants describing the CIFAR data set.
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CHANNEL = 3
NUM_TRAIN_IMG = 50000
NUM_TEST_IMG = 10000


def get_ds_from_tfrecords(data_dir,
                          split,
                          data_aug=True,
                          batch_size=100,
                          epochs=None,
                          shuffle=True,
                          data_format="channels_first",
                          num_parallel_calls=4,
                          prefetch=True,
                          div255=True,
                          dtype=tf.float32):
  """Returns a tf.train.Dataset object from reading tfrecords.

  Args:
      data_dir: Directory of tfrecords
      split: "train", "validation", or "test"
      data_aug: Apply data augmentation if True
      batch_size: Batch size of dataset object
      epochs: Number of epochs to repeat the dataset
      shuffle: Shuffle the dataset if True
      data_format: `channels_first` or `channels_last`
      num_parallel_calls: Number of threads for dataset preprocess
      prefetch: Apply prefetch for the dataset if True
      div255: Divide the images by 255 if True
      dtype: Data type of images
  Returns:
      A tf.train.Dataset object

  Raises:
      ValueError: Unknown split
  """

  if split not in ["train", "validation", "test"]:
    raise ValueError("Unknown split {}".format(split))

  def _parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features["image"], tf.uint8)
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNEL])
    image = tf.cast(image, dtype)
    label = tf.cast(features["label"], tf.int32)

    if data_aug:
      image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEIGHT + 4,
                                                     IMAGE_WIDTH + 4)
      image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNEL])
      image = tf.image.random_flip_left_right(image)

    if data_format == "channels_first":
      image = tf.transpose(image, [2, 0, 1])

    if div255:
      image /= 255.

    return image, label

  filename = os.path.join(data_dir, split + ".tfrecords")
  dataset = tf.data.TFRecordDataset(filename).repeat(epochs)
  dataset = dataset.map(_parser, num_parallel_calls=num_parallel_calls)

  if prefetch:
    dataset = dataset.prefetch(batch_size)
  if shuffle:
    dataset = dataset.shuffle(NUM_TRAIN_IMG)
  dataset = dataset.batch(batch_size)

  return dataset
