# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for manip_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import manip_ops
from tensorflow.python.platform import test as test_lib

# pylint: disable=g-import-not-at-top
try:
  from distutils.version import StrictVersion as Version
  # numpy.roll for multiple shifts was introduced in numpy version 1.12.0
  NP_ROLL_CAN_MULTISHIFT = Version(np.version.version) >= Version("1.12.0")
except ImportError:
  NP_ROLL_CAN_MULTISHIFT = False
# pylint: enable=g-import-not-at-top


class RollTest(test_util.TensorFlowTestCase):

  def _testRoll(self, np_input, shift, axis):
    expected_roll = np.roll(np_input, shift, axis)
    with self.test_session():
      roll = manip_ops.roll(np_input, shift, axis)
      self.assertAllEqual(roll.eval(), expected_roll)

  def _testGradient(self, np_input, shift, axis):
    with self.test_session():
      inx = constant_op.constant(np_input.tolist())
      xs = list(np_input.shape)
      y = manip_ops.roll(inx, shift, axis)
      # Expected y's shape to be the same
      ys = xs
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, y, ys, x_init_value=np_input)
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testAll(self, np_input, shift, axis):
    self._testRoll(np_input, shift, axis)
    if np_input.dtype == np.float32:
      self._testGradient(np_input, shift, axis)

  def testIntTypes(self):
    for t in [np.int32, np.int64]:
      self._testAll(np.random.randint(-100, 100, (5)).astype(t), 3, 0)
      if NP_ROLL_CAN_MULTISHIFT:
        self._testAll(
            np.random.randint(-100, 100, (4, 4, 3)).astype(t), [1, -2, 3],
            [0, 1, 2])
        self._testAll(
            np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t), [0, 1, -2],
            [1, 2, 3])

  def testFloatTypes(self):
    for t in [np.float32, np.float64]:
      self._testAll(np.random.rand(5).astype(t), 2, 0)
      if NP_ROLL_CAN_MULTISHIFT:
        self._testAll(np.random.rand(3, 4).astype(t), [1, 2], [1, 0])
        self._testAll(np.random.rand(1, 3, 4).astype(t), [1, 0, -3], [0, 1, 2])

  def testComplexTypes(self):
    for t in [np.complex64, np.complex128]:
      x = np.random.rand(4, 4).astype(t)
      self._testAll(x + 1j * x, 2, 0)
      if NP_ROLL_CAN_MULTISHIFT:
        x = np.random.rand(2, 5).astype(t)
        self._testAll(x + 1j * x, [1, 2], [1, 0])
        x = np.random.rand(3, 2, 1, 1).astype(t)
        self._testAll(x + 1j * x, [2, 1, 1, 0], [0, 3, 1, 2])

  def testRollInputMustVectorHigherRaises(self):
    tensor = 7
    shift = 1
    axis = 0
    with self.test_session():
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "input must be 1-D or higher"):
        manip_ops.roll(tensor, shift, axis).eval()

  def testRollAxisMustBeScalarOrVectorRaises(self):
    tensor = [[1, 2], [3, 4]]
    shift = 1
    axis = [[0, 1]]
    with self.test_session():
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "axis must be a scalar or a 1-D vector"):
        manip_ops.roll(tensor, shift, axis).eval()

  def testRollShiftMustBeScalarOrVectorRaises(self):
    tensor = [[1, 2], [3, 4]]
    shift = [[0, 1]]
    axis = 1
    with self.test_session():
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "shift must be a scalar or a 1-D vector"):
        manip_ops.roll(tensor, shift, axis).eval()

  def testRollShiftAndAxisMustBeSameSizeRaises(self):
    tensor = [[1, 2], [3, 4]]
    shift = [1]
    axis = [0, 1]
    with self.test_session():
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "shift and axis must have the same size"):
        manip_ops.roll(tensor, shift, axis).eval()

  def testRollAxisOutOfRangeRaises(self):
    tensor = [1, 2]
    shift = 1
    axis = 1
    with self.test_session():
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "is out of range"):
        manip_ops.roll(tensor, shift, axis).eval()


if __name__ == "__main__":
  test_lib.main()
