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
"""Tests for training_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import training_util


class GlobalStepTest(test.TestCase):

  def _assert_global_step(self, global_step, expected_dtype=dtypes.int64):
    self.assertEqual('%s:0' % ops.GraphKeys.GLOBAL_STEP, global_step.name)
    self.assertEqual(expected_dtype, global_step.dtype.base_dtype)
    self.assertEqual([], global_step.get_shape().as_list())

  def test_invalid_dtype(self):
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      variables.Variable(
          0.0,
          trainable=False,
          dtype=dtypes.float32,
          name=ops.GraphKeys.GLOBAL_STEP)
      self.assertRaisesRegexp(TypeError, 'does not have integer type',
                              training_util.get_global_step)
    self.assertRaisesRegexp(TypeError, 'does not have integer type',
                            training_util.get_global_step, g)

  def test_invalid_shape(self):
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      variables.Variable(
          [0],
          trainable=False,
          dtype=dtypes.int32,
          name=ops.GraphKeys.GLOBAL_STEP)
      self.assertRaisesRegexp(TypeError, 'not scalar',
                              training_util.get_global_step)
    self.assertRaisesRegexp(TypeError, 'not scalar',
                            training_util.get_global_step, g)

  def test_create_global_step(self):
    self.assertIsNone(training_util.get_global_step())
    with ops.Graph().as_default() as g:
      global_step = training_util.create_global_step()
      self._assert_global_step(global_step)
      self.assertRaisesRegexp(ValueError, 'already exists',
                              training_util.create_global_step)
      self.assertRaisesRegexp(ValueError, 'already exists',
                              training_util.create_global_step, g)
      self._assert_global_step(training_util.create_global_step(ops.Graph()))

  def test_get_global_step(self):
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      variables.Variable(
          0,
          trainable=False,
          dtype=dtypes.int32,
          name=ops.GraphKeys.GLOBAL_STEP)
      self._assert_global_step(
          training_util.get_global_step(), expected_dtype=dtypes.int32)
    self._assert_global_step(
        training_util.get_global_step(g), expected_dtype=dtypes.int32)

  def test_get_or_create_global_step(self):
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      self._assert_global_step(training_util.get_or_create_global_step())
      self._assert_global_step(training_util.get_or_create_global_step(g))


if __name__ == '__main__':
  test.main()
