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
"""Boston housing price regression dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.datasets.boston_housing.load_data')
def load_data(path='boston_housing.npz', test_split=0.2, seed=113):
  """Loads the Boston Housing dataset.

  Arguments:
      path: path where to cache the dataset locally
          (relative to ~/.keras/datasets).
      test_split: fraction of the data to reserve as test set.
      seed: Random seed for shuffling the data
          before computing the test split.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  assert 0 <= test_split < 1
  path = get_file(
      path,
      origin='https://s3.amazonaws.com/keras-datasets/boston_housing.npz',
      file_hash=
      'f553886a1f8d56431e820c5b82552d9d95cfcb96d1e678153f8839538947dff5')
  f = np.load(path)
  x = f['x']
  y = f['y']
  f.close()

  np.random.seed(seed)
  indices = np.arange(len(x))
  np.random.shuffle(indices)
  x = x[indices]
  y = y[indices]

  x_train = np.array(x[:int(len(x) * (1 - test_split))])
  y_train = np.array(y[:int(len(x) * (1 - test_split))])
  x_test = np.array(x[int(len(x) * (1 - test_split)):])
  y_test = np.array(y[int(len(x) * (1 - test_split)):])
  return (x_train, y_train), (x_test, y_test)
