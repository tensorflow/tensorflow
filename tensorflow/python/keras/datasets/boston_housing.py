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

from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.datasets.boston_housing.load_data')
def load_data(path='boston_housing.npz', test_split=0.2, seed=113):
  """Loads the Boston Housing dataset.

  This is a dataset taken from the StatLib library which is maintained at
  Carnegie Mellon University.

  Samples contain 13 attributes of houses at different locations around the
  Boston suburbs in the late 1970s. Targets are the median values of
  the houses at a location (in k$).

  The attributes themselves are defined in the
  [StatLib website](http://lib.stat.cmu.edu/datasets/boston).

  Args:
      path: path where to cache the dataset locally
          (relative to `~/.keras/datasets`).
      test_split: fraction of the data to reserve as test set.
      seed: Random seed for shuffling the data
          before computing the test split.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

      **x_train, x_test**: numpy arrays with shape `(num_samples, 13)`
        containing either the training samples (for x_train),
        or test samples (for y_train).

      **y_train, y_test**: numpy arrays of shape `(num_samples,)` containing the
        target scalars. The targets are float scalars typically between 10 and
        50 that represent the home prices in k$.
  """
  assert 0 <= test_split < 1
  origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
  path = get_file(
      path,
      origin=origin_folder + 'boston_housing.npz',
      file_hash=
      'f553886a1f8d56431e820c5b82552d9d95cfcb96d1e678153f8839538947dff5')
  with np.load(path, allow_pickle=True) as f:
    x = f['x']
    y = f['y']

  rng = np.random.RandomState(seed)
  indices = np.arange(len(x))
  rng.shuffle(indices)
  x = x[indices]
  y = y[indices]

  x_train = np.array(x[:int(len(x) * (1 - test_split))])
  y_train = np.array(y[:int(len(x) * (1 - test_split))])
  x_test = np.array(x[int(len(x) * (1 - test_split)):])
  y_test = np.array(y[int(len(x) * (1 - test_split)):])
  return (x_train, y_train), (x_test, y_test)
