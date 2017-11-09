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
"""Tests for Keras loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test


ALL_LOSSES = [keras.losses.mean_squared_error,
              keras.losses.mean_absolute_error,
              keras.losses.mean_absolute_percentage_error,
              keras.losses.mean_squared_logarithmic_error,
              keras.losses.squared_hinge,
              keras.losses.hinge,
              keras.losses.categorical_crossentropy,
              keras.losses.binary_crossentropy,
              keras.losses.kullback_leibler_divergence,
              keras.losses.poisson,
              keras.losses.cosine_proximity,
              keras.losses.logcosh,
              keras.losses.categorical_hinge]


class KerasLossesTest(test.TestCase):

  def test_objective_shapes_3d(self):
    with self.test_session():
      y_a = keras.backend.variable(np.random.random((5, 6, 7)))
      y_b = keras.backend.variable(np.random.random((5, 6, 7)))
      for obj in ALL_LOSSES:
        objective_output = obj(y_a, y_b)
        self.assertListEqual(objective_output.get_shape().as_list(), [5, 6])

  def test_objective_shapes_2d(self):
    with self.test_session():
      y_a = keras.backend.variable(np.random.random((6, 7)))
      y_b = keras.backend.variable(np.random.random((6, 7)))
      for obj in ALL_LOSSES:
        objective_output = obj(y_a, y_b)
        self.assertListEqual(objective_output.get_shape().as_list(), [6,])

  def test_cce_one_hot(self):
    with self.test_session():
      y_a = keras.backend.variable(np.random.randint(0, 7, (5, 6)))
      y_b = keras.backend.variable(np.random.random((5, 6, 7)))
      objective_output = keras.losses.sparse_categorical_crossentropy(y_a, y_b)
      assert keras.backend.eval(objective_output).shape == (5, 6)

      y_a = keras.backend.variable(np.random.randint(0, 7, (6,)))
      y_b = keras.backend.variable(np.random.random((6, 7)))
      objective_output = keras.losses.sparse_categorical_crossentropy(y_a, y_b)
      assert keras.backend.eval(objective_output).shape == (6,)

  def test_serialization(self):
    fn = keras.losses.get('mse')
    config = keras.losses.serialize(fn)
    new_fn = keras.losses.deserialize(config)
    self.assertEqual(fn, new_fn)

  def test_categorical_hinge(self):
    y_pred = keras.backend.variable(np.array([[0.3, 0.2, 0.1],
                                              [0.1, 0.2, 0.7]]))
    y_true = keras.backend.variable(np.array([[0, 1, 0], [1, 0, 0]]))
    expected_loss = ((0.3 - 0.2 + 1) + (0.7 - 0.1 + 1)) / 2.0
    loss = keras.backend.eval(keras.losses.categorical_hinge(y_true, y_pred))
    self.assertAllClose(expected_loss, np.mean(loss))


if __name__ == '__main__':
  test.main()
