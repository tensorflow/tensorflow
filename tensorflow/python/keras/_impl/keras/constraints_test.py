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
"""Tests for Keras weights constraints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test


def get_test_values():
  return [0.1, 0.5, 3, 8, 1e-7]


def get_example_array():
  np.random.seed(3537)
  example_array = np.random.random((100, 100)) * 100. - 50.
  example_array[0, 0] = 0.  # 0 could possibly cause trouble
  return example_array


class KerasConstraintsTest(test.TestCase):

  def test_serialization(self):
    all_activations = ['max_norm', 'non_neg',
                       'unit_norm', 'min_max_norm']
    for name in all_activations:
      fn = keras.constraints.get(name)
      ref_fn = getattr(keras.constraints, name)()
      assert fn.__class__ == ref_fn.__class__
      config = keras.constraints.serialize(fn)
      fn = keras.constraints.deserialize(config)
      assert fn.__class__ == ref_fn.__class__

  def test_max_norm(self):
    with self.test_session():
      array = get_example_array()
      for m in get_test_values():
        norm_instance = keras.constraints.max_norm(m)
        normed = norm_instance(keras.backend.variable(array))
        assert np.all(keras.backend.eval(normed) < m)

      # a more explicit example
      norm_instance = keras.constraints.max_norm(2.0)
      x = np.array([[0, 0, 0], [1.0, 0, 0], [3, 0, 0], [3, 3, 3]]).T
      x_normed_target = np.array([[0, 0, 0], [1.0, 0, 0],
                                  [2.0, 0, 0],
                                  [2. / np.sqrt(3),
                                   2. / np.sqrt(3),
                                   2. / np.sqrt(3)]]).T
      x_normed_actual = keras.backend.eval(
          norm_instance(keras.backend.variable(x)))
      self.assertAllClose(x_normed_actual, x_normed_target, rtol=1e-05)

  def test_non_neg(self):
    with self.test_session():
      non_neg_instance = keras.constraints.non_neg()
      normed = non_neg_instance(keras.backend.variable(get_example_array()))
      assert np.all(np.min(keras.backend.eval(normed), axis=1) == 0.)

  def test_unit_norm(self):
    with self.test_session():
      unit_norm_instance = keras.constraints.unit_norm()
      normalized = unit_norm_instance(
          keras.backend.variable(get_example_array()))
      norm_of_normalized = np.sqrt(
          np.sum(keras.backend.eval(normalized) ** 2, axis=0))
      # In the unit norm constraint, it should be equal to 1.
      difference = norm_of_normalized - 1.
      largest_difference = np.max(np.abs(difference))
      assert np.abs(largest_difference) < 10e-5

  def test_min_max_norm(self):
    with self.test_session():
      array = get_example_array()
      for m in get_test_values():
        norm_instance = keras.constraints.min_max_norm(min_value=m,
                                                       max_value=m * 2)
        normed = norm_instance(keras.backend.variable(array))
        value = keras.backend.eval(normed)
        l2 = np.sqrt(np.sum(np.square(value), axis=0))
        assert not l2[l2 < m]
        assert not l2[l2 > m * 2 + 1e-5]


if __name__ == '__main__':
  test.main()
