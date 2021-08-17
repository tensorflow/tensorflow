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
"""CIFAR10 small images classification dataset."""

import os

import numpy as np

from tensorflow.python.keras import backend
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.datasets.cifar10.load_data')
def load_data():
  """Loads the CIFAR10 dataset.

  This is a dataset of 50,000 32x32 color training images and 10,000 test
  images, labeled over 10 categories. See more info at the
  [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

  The classes are:

  | Label | Description |
  |:-----:|-------------|
  |   0   | airplane    |
  |   1   | automobile  |
  |   2   | bird        |
  |   3   | cat         |
  |   4   | deer        |
  |   5   | dog         |
  |   6   | frog        |
  |   7   | horse       |
  |   8   | ship        |
  |   9   | truck       |

  Returns:
    Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

  **x_train**: uint8 NumPy array of grayscale image data with shapes
    `(50000, 32, 32, 3)`, containing the training data. Pixel values range
    from 0 to 255.

  **y_train**: uint8 NumPy array of labels (integers in range 0-9)
    with shape `(50000, 1)` for the training data.

  **x_test**: uint8 NumPy array of grayscale image data with shapes
    (10000, 32, 32, 3), containing the test data. Pixel values range
    from 0 to 255.

  **y_test**: uint8 NumPy array of labels (integers in range 0-9)
    with shape `(10000, 1)` for the test data.

  Example:

  ```python
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  assert x_train.shape == (50000, 32, 32, 3)
  assert x_test.shape == (10000, 32, 32, 3)
  assert y_train.shape == (50000, 1)
  assert y_test.shape == (10000, 1)
  ```
  """
  dirname = 'cifar-10-batches-py'
  origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  path = get_file(
      dirname,
      origin=origin,
      untar=True,
      file_hash=
      '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce')

  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if backend.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  x_test = x_test.astype(x_train.dtype)
  y_test = y_test.astype(y_train.dtype)

  return (x_train, y_train), (x_test, y_test)
