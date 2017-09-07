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
"""Test cases for ternary operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class TernaryOpsTest(XLATestCase):

  def _testTernary(self, op, a, b, c, expected):
    with self.test_session() as session:
      with self.test_scope():
        pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name="b")
        pc = array_ops.placeholder(dtypes.as_dtype(c.dtype), c.shape, name="c")
        output = op(pa, pb, pc)
      result = session.run(output, {pa: a, pb: b, pc: c})
      self.assertAllClose(result, expected, rtol=1e-3)

  def testLinspace(self):
    self._testTernary(
        math_ops.linspace,
        np.float32(1),
        np.float32(2),
        np.int32(1),
        expected=np.array([1], dtype=np.float32))
    self._testTernary(
        math_ops.linspace,
        np.float32(1),
        np.float32(4),
        np.int32(3),
        expected=np.array([1, 2.5, 4], dtype=np.float32))

  def testRange(self):
    self._testTernary(
        math_ops.range,
        np.int32(1),
        np.int32(2),
        np.int32(1),
        expected=np.array([1], dtype=np.int32))
    self._testTernary(
        math_ops.range,
        np.int32(1),
        np.int32(7),
        np.int32(2),
        expected=np.array([1, 3, 5], dtype=np.int32))

  def testSelect(self):
    self._testTernary(
        array_ops.where,
        np.array(0, dtype=np.bool),
        np.array(2, dtype=np.float32),
        np.array(7, dtype=np.float32),
        expected=np.array(7, dtype=np.float32))

    self._testTernary(
        array_ops.where,
        np.array(1, dtype=np.bool),
        np.array([1, 2, 3, 4], dtype=np.float32),
        np.array([5, 6, 7, 8], dtype=np.float32),
        expected=np.array([1, 2, 3, 4], dtype=np.float32))

    self._testTernary(
        array_ops.where,
        np.array(0, dtype=np.bool),
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32),
        expected=np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32))

    self._testTernary(
        array_ops.where,
        np.array([0, 1, 1, 0], dtype=np.bool),
        np.array([1, 2, 3, 4], dtype=np.float32),
        np.array([5, 6, 7, 8], dtype=np.float32),
        expected=np.array([5, 2, 3, 8], dtype=np.float32))

    self._testTernary(
        array_ops.where,
        np.array([0, 1, 0], dtype=np.bool),
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32),
        expected=np.array([[7, 8], [3, 4], [11, 12]], dtype=np.float32))

  def testSlice(self):
    for dtype in self.numeric_types:
      self._testTernary(
          array_ops.slice,
          np.array([[], [], []], dtype=dtype),
          np.array([1, 0], dtype=np.int32),
          np.array([2, 0], dtype=np.int32),
          expected=np.array([[], []], dtype=dtype))

      self._testTernary(
          array_ops.slice,
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype),
          np.array([0, 1], dtype=np.int32),
          np.array([2, 1], dtype=np.int32),
          expected=np.array([[2], [5]], dtype=dtype))


if __name__ == "__main__":
  googletest.main()
