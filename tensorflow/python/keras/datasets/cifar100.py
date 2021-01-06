# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""CIFAR100 small images classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.datasets.cifar100.load_data')
def load_data(label_mode='fine'):
  """Loads [CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

  This is a dataset of 50,000 32x32 color training images and
  10,000 test images, labeled over 100 fine-grained classes that are
  grouped into 20 coarse-grained classes. See more info at the
  [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

  Args:
      label_mode: one of "fine", "coarse". If it is "fine" the category labels
      are the fine-grained labels, if it is "coarse" the output labels are the
      coarse-grained superclasses.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

      **x_train, x_test**: uint8 arrays of RGB image data with shape
        `(num_samples, 3, 32, 32)` if `tf.keras.backend.image_data_format()` is
        `'channels_first'`, or `(num_samples, 32, 32, 3)` if the data format
        is `'channels_last'`.

      **y_train, y_test**: uint8 arrays of category labels with shape
        (num_samples, 1).

  Raises:
      ValueError: in case of invalid `label_mode`.
  """
  if label_mode not in ['fine', 'coarse']:
    raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

  dirname = 'cifar-100-python'
  origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
  path = get_file(
      dirname,
      origin=origin,
      untar=True,
      file_hash=
      '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7')

  fpath = os.path.join(path, 'train')
  x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

  fpath = os.path.join(path, 'test')
  x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  return (x_train, y_train), (x_test, y_test)
