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
"""CIFAR10 small image classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.datasets.cifar import load_batch
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


def load_data():
  """Loads CIFAR10 dataset.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  dirname = 'cifar-10-batches-py'
  origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  path = get_file(dirname, origin=origin, untar=True)

  num_train_samples = 50000

  x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.zeros((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    data, labels = load_batch(fpath)
    x_train[(i - 1) * 10000:i * 10000, :, :, :] = data
    y_train[(i - 1) * 10000:i * 10000] = labels

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  return (x_train, y_train), (x_test, y_test)
