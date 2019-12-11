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
import six

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.utils import io_utils
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


class TestIOUtils(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_HDF5Matrix(self):
    if h5py is None:
      return

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)

    h5_path = os.path.join(temp_dir, 'test.h5')
    create_dataset(h5_path)

    # Instantiating HDF5Matrix for the training set,
    # which is a slice of the first 150 elements
    x_train = io_utils.HDF5Matrix(h5_path, 'my_data', start=0, end=150)
    y_train = io_utils.HDF5Matrix(h5_path, 'my_labels', start=0, end=150)

    # Likewise for the test set
    x_test = io_utils.HDF5Matrix(h5_path, 'my_data', start=150, end=200)
    y_test = io_utils.HDF5Matrix(h5_path, 'my_labels', start=150, end=200)

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
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    # Note: you have to use shuffle='batch' or False with HDF5Matrix
    model.fit(x_train, y_train, batch_size=32, shuffle='batch', verbose=False)
    # test that evalutation and prediction
    # don't crash and return reasonable results
    out_pred = model.predict(x_test, batch_size=32, verbose=False)
    out_eval = model.evaluate(x_test, y_test, batch_size=32, verbose=False)

    self.assertEqual(out_pred.shape, (50, 1))
    self.assertEqual(out_eval.shape, ())
    self.assertGreater(out_eval, 0)

    # test slicing for shortened array
    self.assertEqual(len(x_train[0:]), len(x_train))

    # test __getitem__ invalid use cases
    with self.assertRaises(IndexError):
      _ = x_train[1000]
    with self.assertRaises(IndexError):
      _ = x_train[1000: 1001]
    with self.assertRaises(IndexError):
      _ = x_train[[1000, 1001]]
    with self.assertRaises(IndexError):
      _ = x_train[six.moves.range(1000, 1001)]
    with self.assertRaises(IndexError):
      _ = x_train[np.array([1000])]
    with self.assertRaises(TypeError):
      _ = x_train[None]

    # test normalizer
    normalizer = lambda x: x + 1
    normalized_x_train = io_utils.HDF5Matrix(
        h5_path, 'my_data', start=0, end=150, normalizer=normalizer)
    self.assertAllClose(normalized_x_train[0][0], x_train[0][0] + 1)

  def test_ask_to_proceed_with_overwrite(self):
    with test.mock.patch.object(six.moves, 'input') as mock_log:
      mock_log.return_value = 'y'
      self.assertTrue(io_utils.ask_to_proceed_with_overwrite('/tmp/not_exists'))

      mock_log.return_value = 'n'
      self.assertFalse(
          io_utils.ask_to_proceed_with_overwrite('/tmp/not_exists'))

      mock_log.side_effect = ['m', 'y']
      self.assertTrue(io_utils.ask_to_proceed_with_overwrite('/tmp/not_exists'))

      mock_log.side_effect = ['m', 'n']
      self.assertFalse(
          io_utils.ask_to_proceed_with_overwrite('/tmp/not_exists'))


if __name__ == '__main__':
  test.main()
