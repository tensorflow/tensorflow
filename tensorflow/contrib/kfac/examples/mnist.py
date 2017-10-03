# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for loading MNIST into TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

__all__ = [
    'load_mnist',
]


def load_mnist(data_dir,
               num_epochs,
               batch_size,
               flatten_images=True,
               use_fake_data=False):
  """Loads MNIST dataset into memory.

  Args:
    data_dir: string. Directory to read MNIST examples from.
    num_epochs: int. Number of passes to make over the dataset.
    batch_size: int. Number of examples per minibatch.
    flatten_images: bool. If True, [28, 28, 1]-shaped images are flattened into
      [784]-shaped vectors.
    use_fake_data: bool. If True, generate a synthetic dataset rather than
      reading MNIST in.

  Returns:
    examples: Tensor of shape [batch_size, 784] if 'flatten_images' is
      True, else [batch_size, 28, 28, 1]. Each row is one example.
      Values in [0, 1].
    labels: Tensor of shape [batch_size]. Indices of integer corresponding to
      each example. Values in {0...9}.
  """
  if use_fake_data:
    rng = np.random.RandomState(42)
    num_examples = batch_size * 4
    images = rng.rand(num_examples, 28 * 28)
    if not flatten_images:
      images = np.reshape(images, [num_examples, 28, 28, 1])
    labels = rng.randint(10, size=num_examples)
  else:
    mnist_data = tf.contrib.learn.datasets.mnist.read_data_sets(
        data_dir, reshape=flatten_images)
    num_examples = len(mnist_data.train.labels)
    images = mnist_data.train.images
    labels = mnist_data.train.labels

  dataset = tf.contrib.data.Dataset.from_tensor_slices((np.asarray(
      images, dtype=np.float32), np.asarray(labels, dtype=np.int64)))
  return (dataset.repeat(num_epochs).shuffle(num_examples).batch(batch_size)
          .make_one_shot_iterator().get_next())
