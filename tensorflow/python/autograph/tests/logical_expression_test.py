# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Basic logical expressions that are not autoboxed to TF."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def composite_ors_with_callable(x, y, z):
  z1 = lambda: z
  return x or y or z1()


def composite_ors(x, y, z):
  return x or y or z


def composite_ands(x, y, z):
  return x and y and z


def composite_mixed(x, y, z):
  return x or y or z and y and z


def equality(x, y):
  return x == y


def inequality(x, y):
  return x != y


def multiple_equality(x, y, z):
  return x == y == z


def comparison(x, y, z):
  return x < y and y < z


class ReferenceTest(reference_test_base.TestCase):

  def test_basic(self):
    self.assertFunctionMatchesEager(composite_ors, False, True, False)
    self.assertFunctionMatchesEager(composite_ors, False, False, False)
    self.assertFunctionMatchesEager(composite_ands, True, True, True)
    self.assertFunctionMatchesEager(composite_ands, True, False, True)
    self.assertFunctionMatchesEager(composite_mixed, False, True, True)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, True,
                                    False)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, False,
                                    True)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, False,
                                    False)

    self.assertFunctionMatchesEager(comparison, 1, 2, 3)
    self.assertFunctionMatchesEager(comparison, 2, 1, 3)
    self.assertFunctionMatchesEager(comparison, 3, 2, 1)
    self.assertFunctionMatchesEager(comparison, 3, 1, 2)
    self.assertFunctionMatchesEager(comparison, 1, 3, 2)
    self.assertFunctionMatchesEager(comparison, 2, 3, 1)

  def test_basic_tensor(self):
    self.all_inputs_tensors = True
    self.assertFunctionMatchesEager(composite_ors, False, True, False)
    self.assertFunctionMatchesEager(composite_ors, False, False, False)
    self.assertFunctionMatchesEager(composite_ands, True, True, True)
    self.assertFunctionMatchesEager(composite_ands, True, False, True)
    self.assertFunctionMatchesEager(composite_mixed, False, True, True)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, True,
                                    False)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, False,
                                    True)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, False,
                                    False)

    self.assertFunctionMatchesEager(comparison, 1, 2, 3)
    self.assertFunctionMatchesEager(comparison, 2, 1, 3)
    self.assertFunctionMatchesEager(comparison, 3, 2, 1)
    self.assertFunctionMatchesEager(comparison, 3, 1, 2)
    self.assertFunctionMatchesEager(comparison, 1, 3, 2)
    self.assertFunctionMatchesEager(comparison, 2, 3, 1)

  def test_equality(self):
    self.assertFunctionMatchesEager(equality, 1, 1)
    self.assertFunctionMatchesEager(equality, 1, 2)
    self.assertFunctionMatchesEager(inequality, 1, 1)
    self.assertFunctionMatchesEager(inequality, 1, 2)
    self.assertFunctionMatchesEager(multiple_equality, 1, 1, 2)
    self.assertFunctionMatchesEager(multiple_equality, 1, 1, 1)

  def test_equality_tensor(self):
    self.autograph_opts = tf.autograph.experimental.Feature.EQUALITY_OPERATORS
    self.all_inputs_tensors = True

    self.assertFunctionMatchesEager(equality, 1, 1)
    self.assertFunctionMatchesEager(equality, 1, 2)
    self.assertFunctionMatchesEager(inequality, 1, 1)
    self.assertFunctionMatchesEager(inequality, 1, 2)
    self.assertFunctionMatchesEager(multiple_equality, 1, 1, 2)
    self.assertFunctionMatchesEager(multiple_equality, 1, 1, 1)


if __name__ == '__main__':
  tf.test.main()
