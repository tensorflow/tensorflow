# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Convenience wrapper around Keras' MNIST and Fashion MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10


def _load_random_data(num_train_and_test):
  return ((np.random.randint(0, 256, (num, 28, 28), dtype=np.uint8),
           np.random.randint(0, 10, (num,), dtype=np.int64))
          for num in num_train_and_test)


def load_reshaped_data(use_fashion_mnist=False, fake_tiny_data=False):
  """Returns MNIST or Fashion MNIST or fake train and test data."""
  load = ((lambda: _load_random_data([16, 128])) if fake_tiny_data else
          tf.keras.datasets.fashion_mnist.load_data if use_fashion_mnist else
          tf.keras.datasets.mnist.load_data)
  (x_train, y_train), (x_test, y_test) = load()
  return ((_prepare_image(x_train), _prepare_label(y_train)),
          (_prepare_image(x_test), _prepare_label(y_test)))


def _prepare_image(x):
  """Converts images to [n,h,w,c] format in range [0,1]."""
  return x[..., None].astype('float32') / 255.


def _prepare_label(y):
  """Conerts labels to one-hot encoding."""
  return tf.keras.utils.to_categorical(y, NUM_CLASSES)
