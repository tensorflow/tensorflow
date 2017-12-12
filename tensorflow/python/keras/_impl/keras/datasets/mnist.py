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
"""MNIST handwritten digits classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl.keras.utils.data_utils import get_file


def load_data(path='mnist.npz'):
  """Loads the MNIST dataset.

  Arguments:
      path: path where to cache the dataset locally
          (relative to ~/.keras/datasets).

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  path = get_file(
      path,
      origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
      file_hash='8a61469f7ea1b51cbae51d4f78837e45')
  f = np.load(path)
  x_train = f['x_train']
  y_train = f['y_train']
  x_test = f['x_test']
  y_test = f['y_test']
  f.close()
  return (x_train, y_train), (x_test, y_test)
