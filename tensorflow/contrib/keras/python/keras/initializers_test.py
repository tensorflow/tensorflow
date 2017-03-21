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
"""Tests for Keras initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.python.platform import test


def _runner(init, shape, target_mean=None, target_std=None,
            target_max=None, target_min=None):
  variable = keras.backend.variable(init(shape))
  output = keras.backend.get_value(variable)
  lim = 3e-2
  if target_std is not None:
    assert abs(output.std() - target_std) < lim, output.std()
  if target_mean is not None:
    assert abs(output.mean() - target_mean) < lim, output.mean()
  if target_max is not None:
    assert abs(output.max() - target_max) < lim, output.max()
  if target_min is not None:
    assert abs(output.min() - target_min) < lim, output.min()


class KerasInitializersTest(test.TestCase):

  def test_uniform(self):
    tensor_shape = (9, 6, 7)
    with self.test_session():
      _runner(keras.initializers.RandomUniform(minval=-1, maxval=1, seed=124),
              tensor_shape,
              target_mean=0., target_max=1, target_min=-1)

  def test_normal(self):
    tensor_shape = (8, 12, 99)
    with self.test_session():
      _runner(keras.initializers.RandomNormal(mean=0, stddev=1, seed=153),
              tensor_shape,
              target_mean=0., target_std=1)

  def test_truncated_normal(self):
    tensor_shape = (12, 99, 7)
    with self.test_session():
      _runner(keras.initializers.TruncatedNormal(mean=0, stddev=1, seed=126),
              tensor_shape,
              target_mean=0., target_std=None, target_max=2)

  def test_constant(self):
    tensor_shape = (5, 6, 4)
    with self.test_session():
      _runner(keras.initializers.Constant(2), tensor_shape,
              target_mean=2, target_max=2, target_min=2)

  def test_lecun_uniform(self):
    tensor_shape = (5, 6, 4, 2)
    with self.test_session():
      fan_in, _ = keras.initializers._compute_fans(tensor_shape)
      scale = np.sqrt(3. / fan_in)
      _runner(keras.initializers.lecun_uniform(seed=123), tensor_shape,
              target_mean=0., target_max=scale, target_min=-scale)

  def test_glorot_uniform(self):
    tensor_shape = (5, 6, 4, 2)
    with self.test_session():
      fan_in, fan_out = keras.initializers._compute_fans(tensor_shape)
      scale = np.sqrt(6. / (fan_in + fan_out))
      _runner(keras.initializers.glorot_uniform(seed=123), tensor_shape,
              target_mean=0., target_max=scale, target_min=-scale)

  def test_he_uniform(self):
    tensor_shape = (5, 6, 4, 2)
    with self.test_session():
      fan_in, _ = keras.initializers._compute_fans(tensor_shape)
      scale = np.sqrt(6. / fan_in)
      _runner(keras.initializers.he_uniform(seed=123), tensor_shape,
              target_mean=0., target_max=scale, target_min=-scale)

  def test_glorot_normal(self):
    tensor_shape = (5, 6, 4, 2)
    with self.test_session():
      fan_in, fan_out = keras.initializers._compute_fans(tensor_shape)
      scale = np.sqrt(2. / (fan_in + fan_out))
      _runner(keras.initializers.glorot_normal(seed=123), tensor_shape,
              target_mean=0., target_std=None, target_max=2 * scale)

  def test_he_normal(self):
    tensor_shape = (5, 6, 4, 2)
    with self.test_session():
      fan_in, _ = keras.initializers._compute_fans(tensor_shape)
      scale = np.sqrt(2. / fan_in)
      _runner(keras.initializers.he_normal(seed=123), tensor_shape,
              target_mean=0., target_std=None, target_max=2 * scale)

  def test_orthogonal(self):
    tensor_shape = (7, 8)
    with self.test_session():
      _runner(keras.initializers.orthogonal(seed=123), tensor_shape,
              target_mean=0.)

  def test_identity(self):
    with self.test_session():
      tensor_shape = (3, 4, 5)
      with self.assertRaises(ValueError):
        _runner(keras.initializers.identity(), tensor_shape,
                target_mean=1. / tensor_shape[0], target_max=1.)

      tensor_shape = (3, 3)
      _runner(keras.initializers.identity(), tensor_shape,
              target_mean=1. / tensor_shape[0], target_max=1.)

  def test_zero(self):
    tensor_shape = (4, 5)
    with self.test_session():
      _runner(keras.initializers.zeros(), tensor_shape,
              target_mean=0., target_max=0.)

  def test_one(self):
    tensor_shape = (4, 5)
    with self.test_session():
      _runner(keras.initializers.ones(), tensor_shape,
              target_mean=1., target_max=1.)


if __name__ == '__main__':
  test.main()
