# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for io_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


def create_dataset(h5_path='test.h5'):
  x = np.random.randn(200, 10).astype('float32')
  y = np.random.randint(0, 2, size=(200, 1))
  f = h5py.File(h5_path, 'w')
  # Creating dataset to store features
  x_dset = f.create_dataset('my_data', (200, 10), dtype='f')
  x_dset[:] = x
  # Creating dataset to store labels
  y_dset = f.create_dataset('my_labels', (200, 1), dtype='i')
  y_dset[:] = y
  f.close()


class TestIOUtils(test.TestCase):

  def test_HDF5Matrix(self):
    if h5py is None:
      return

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)

    h5_path = os.path.join(temp_dir, 'test.h5')
    create_dataset(h5_path)

    # Instantiating HDF5Matrix for the training set,
    # which is a slice of the first 150 elements
    x_train = keras.utils.io_utils.HDF5Matrix(
        h5_path, 'my_data', start=0, end=150)
    y_train = keras.utils.io_utils.HDF5Matrix(
        h5_path, 'my_labels', start=0, end=150)

    # Likewise for the test set
    x_test = keras.utils.io_utils.HDF5Matrix(
        h5_path, 'my_data', start=150, end=200)
    y_test = keras.utils.io_utils.HDF5Matrix(
        h5_path, 'my_labels', start=150, end=200)

    # HDF5Matrix behave more or less like Numpy matrices
    # with regard to indexing
    self.assertEqual(y_train.shape, (150, 1))
    # But they do not support negative indices, so don't try print(x_train[-1])

    self.assertEqual(y_train.dtype, np.dtype('i'))
    self.assertEqual(y_train.ndim, 2)
    self.assertEqual(y_train.size, 150)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, input_shape=(10,), activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd')

    # Note: you have to use shuffle='batch' or False with HDF5Matrix
    model.fit(x_train, y_train, batch_size=32, shuffle='batch', verbose=False)
    # test that evalutation and prediction
    # don't crash and return reasonable results
    out_pred = model.predict(x_test, batch_size=32, verbose=False)
    out_eval = model.evaluate(x_test, y_test, batch_size=32, verbose=False)

    self.assertEqual(out_pred.shape, (50, 1))
    self.assertEqual(out_eval.shape, ())
    self.assertGreater(out_eval, 0)


if __name__ == '__main__':
  test.main()
