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
"""Test cases for operators with > 3 or arbitrary numbers of arguments."""

import unittest

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class NAryOpsTest(xla_test.XLATestCase):

  def _testNAry(self, op, args, expected, equality_fn=None):
    with self.session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(placeholders)
      result = session.run(output, feeds)
      if not equality_fn:
        equality_fn = self.assertAllClose
      equality_fn(result, expected, rtol=1e-3)

  def _nAryListCheck(self, results, expected, **kwargs):
    self.assertEqual(len(results), len(expected))
    for (r, e) in zip(results, expected):
      self.assertAllClose(r, e, **kwargs)

  def _testNAryLists(self, op, args, expected):
    self._testNAry(op, args, expected, equality_fn=self._nAryListCheck)

  def testFloat(self):
    self._testNAry(math_ops.add_n,
                   [np.array([[1, 2, 3]], dtype=np.float32)],
                   expected=np.array([[1, 2, 3]], dtype=np.float32))

    self._testNAry(math_ops.add_n,
                   [np.array([1, 2], dtype=np.float32),
                    np.array([10, 20], dtype=np.float32)],
                   expected=np.array([11, 22], dtype=np.float32))
    self._testNAry(math_ops.add_n,
                   [np.array([-4], dtype=np.float32),
                    np.array([10], dtype=np.float32),
                    np.array([42], dtype=np.float32)],
                   expected=np.array([48], dtype=np.float32))

  def testComplex(self):
    for dtype in self.complex_types:
      self._testNAry(
          math_ops.add_n, [np.array([[1 + 2j, 2 - 3j, 3 + 4j]], dtype=dtype)],
          expected=np.array([[1 + 2j, 2 - 3j, 3 + 4j]], dtype=dtype))

      self._testNAry(
          math_ops.add_n, [
              np.array([1 + 2j, 2 - 3j], dtype=dtype),
              np.array([10j, 20], dtype=dtype)
          ],
          expected=np.array([1 + 12j, 22 - 3j], dtype=dtype))
      self._testNAry(
          math_ops.add_n, [
              np.array([-4, 5j], dtype=dtype),
              np.array([2 + 10j, -2], dtype=dtype),
              np.array([42j, 3 + 3j], dtype=dtype)
          ],
          expected=np.array([-2 + 52j, 1 + 8j], dtype=dtype))

  @unittest.skip("IdentityN is temporarily CompilationOnly as workaround")
  def testIdentityN(self):
    self._testNAryLists(array_ops.identity_n,
                        [np.array([[1, 2, 3]], dtype=np.float32)],
                        expected=[np.array([[1, 2, 3]], dtype=np.float32)])
    self._testNAryLists(array_ops.identity_n,
                        [np.array([[1, 2], [3, 4]], dtype=np.float32),
                         np.array([[3, 2, 1], [6, 5, 1]], dtype=np.float32)],
                        expected=[
                            np.array([[1, 2], [3, 4]], dtype=np.float32),
                            np.array([[3, 2, 1], [6, 5, 1]], dtype=np.float32)])
    self._testNAryLists(array_ops.identity_n,
                        [np.array([[1], [2], [3], [4]], dtype=np.int32),
                         np.array([[3, 2, 1], [6, 5, 1]], dtype=np.float32)],
                        expected=[
                            np.array([[1], [2], [3], [4]], dtype=np.int32),
                            np.array([[3, 2, 1], [6, 5, 1]], dtype=np.float32)])

  def testConcat(self):
    self._testNAry(
        lambda x: array_ops.concat(x, 0), [
            np.array(
                [[1, 2, 3], [4, 5, 6]], dtype=np.float32), np.array(
                    [[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        ],
        expected=np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32))

    self._testNAry(
        lambda x: array_ops.concat(x, 1), [
            np.array(
                [[1, 2, 3], [4, 5, 6]], dtype=np.float32), np.array(
                    [[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        ],
        expected=np.array(
            [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], dtype=np.float32))

  def testOneHot(self):
    with self.session() as session, self.test_scope():
      indices = array_ops.constant(np.array([[2, 3], [0, 1]], dtype=np.int32))
      op = array_ops.one_hot(indices,
                             np.int32(4),
                             on_value=np.float32(7), off_value=np.float32(3))
      output = session.run(op)
      expected = np.array([[[3, 3, 7, 3], [3, 3, 3, 7]],
                           [[7, 3, 3, 3], [3, 7, 3, 3]]],
                          dtype=np.float32)
      self.assertAllEqual(output, expected)

      op = array_ops.one_hot(indices,
                             np.int32(4),
                             on_value=np.int32(2), off_value=np.int32(1),
                             axis=1)
      output = session.run(op)
      expected = np.array([[[1, 1], [1, 1], [2, 1], [1, 2]],
                           [[2, 1], [1, 2], [1, 1], [1, 1]]],
                          dtype=np.int32)
      self.assertAllEqual(output, expected)

  def testSplitV(self):
    with self.session() as session:
      with self.test_scope():
        output = session.run(
            array_ops.split(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],
                                     dtype=np.float32),
                            [2, 2], 1))
        expected = [np.array([[1, 2], [5, 6], [9, 0]], dtype=np.float32),
                    np.array([[3, 4], [7, 8], [1, 2]], dtype=np.float32)]
        self.assertAllEqual(output, expected)

  def testStridedSlice(self):
    self._testNAry(lambda x: array_ops.strided_slice(*x),
                   [np.array([[], [], []], dtype=np.float32),
                    np.array([1, 0], dtype=np.int32),
                    np.array([3, 0], dtype=np.int32),
                    np.array([1, 1], dtype=np.int32)],
                   expected=np.array([[], []], dtype=np.float32))

    if np.int64 in self.int_types:
      self._testNAry(
          lambda x: array_ops.strided_slice(*x), [
              np.array([[], [], []], dtype=np.float32), np.array(
                  [1, 0], dtype=np.int64), np.array([3, 0], dtype=np.int64),
              np.array([1, 1], dtype=np.int64)
          ],
          expected=np.array([[], []], dtype=np.float32))

    self._testNAry(lambda x: array_ops.strided_slice(*x),
                   [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                             dtype=np.float32),
                    np.array([1, 1], dtype=np.int32),
                    np.array([3, 3], dtype=np.int32),
                    np.array([1, 1], dtype=np.int32)],
                   expected=np.array([[5, 6], [8, 9]], dtype=np.float32))

    self._testNAry(lambda x: array_ops.strided_slice(*x),
                   [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                             dtype=np.float32),
                    np.array([0, 2], dtype=np.int32),
                    np.array([2, 0], dtype=np.int32),
                    np.array([1, -1], dtype=np.int32)],
                   expected=np.array([[3, 2], [6, 5]], dtype=np.float32))

    self._testNAry(lambda x: x[0][0:2, array_ops.newaxis, ::-1],
                   [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                             dtype=np.float32)],
                   expected=np.array([[[3, 2, 1]], [[6, 5, 4]]],
                                     dtype=np.float32))

    self._testNAry(lambda x: x[0][1, :, array_ops.newaxis],
                   [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                             dtype=np.float32)],
                   expected=np.array([[4], [5], [6]], dtype=np.float32))

  def testStridedSliceGrad(self):
    # Tests cases where input shape is empty.
    self._testNAry(lambda x: array_ops.strided_slice_grad(*x),
                   [np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    np.float32(0.5)],
                   expected=np.array(np.float32(0.5), dtype=np.float32))

    # Tests case where input shape is non-empty, but gradients are empty.
    self._testNAry(lambda x: array_ops.strided_slice_grad(*x),
                   [np.array([3], dtype=np.int32),
                    np.array([0], dtype=np.int32),
                    np.array([0], dtype=np.int32),
                    np.array([1], dtype=np.int32),
                    np.array([], dtype=np.float32)],
                   expected=np.array([0, 0, 0], dtype=np.float32))

    self._testNAry(lambda x: array_ops.strided_slice_grad(*x),
                   [np.array([3, 0], dtype=np.int32),
                    np.array([1, 0], dtype=np.int32),
                    np.array([3, 0], dtype=np.int32),
                    np.array([1, 1], dtype=np.int32),
                    np.array([[], []], dtype=np.float32)],
                   expected=np.array([[], [], []], dtype=np.float32))

    self._testNAry(lambda x: array_ops.strided_slice_grad(*x),
                   [np.array([3, 3], dtype=np.int32),
                    np.array([1, 1], dtype=np.int32),
                    np.array([3, 3], dtype=np.int32),
                    np.array([1, 1], dtype=np.int32),
                    np.array([[5, 6], [8, 9]], dtype=np.float32)],
                   expected=np.array([[0, 0, 0], [0, 5, 6], [0, 8, 9]],
                                     dtype=np.float32))

    def ssg_test(x):
      return array_ops.strided_slice_grad(*x, shrink_axis_mask=0x4,
                                          new_axis_mask=0x1)

    self._testNAry(ssg_test,
                   [np.array([3, 1, 3], dtype=np.int32),
                    np.array([0, 0, 0, 2], dtype=np.int32),
                    np.array([0, 3, 1, -4], dtype=np.int32),
                    np.array([1, 2, 1, -3], dtype=np.int32),
                    np.array([[[1], [2]]], dtype=np.float32)],
                   expected=np.array([[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 2]]],
                                     dtype=np.float32))

    ssg_test2 = lambda x: array_ops.strided_slice_grad(*x, new_axis_mask=0x15)
    self._testNAry(ssg_test2,
                   [np.array([4, 4], dtype=np.int32),
                    np.array([0, 0, 0, 1, 0], dtype=np.int32),
                    np.array([0, 3, 0, 4, 0], dtype=np.int32),
                    np.array([1, 2, 1, 2, 1], dtype=np.int32),
                    np.array([[[[[1], [2]]], [[[3], [4]]]]], dtype=np.float32)],
                   expected=np.array([[0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4],
                                      [0, 0, 0, 0]], dtype=np.float32))

    self._testNAry(lambda x: array_ops.strided_slice_grad(*x),
                   [np.array([3, 3], dtype=np.int32),
                    np.array([0, 2], dtype=np.int32),
                    np.array([2, 0], dtype=np.int32),
                    np.array([1, -1], dtype=np.int32),
                    np.array([[1, 2], [3, 4]], dtype=np.float32)],
                   expected=np.array([[0, 2, 1], [0, 4, 3], [0, 0, 0]],
                                     dtype=np.float32))

    self._testNAry(lambda x: array_ops.strided_slice_grad(*x),
                   [np.array([3, 3], dtype=np.int32),
                    np.array([2, 2], dtype=np.int32),
                    np.array([0, 1], dtype=np.int32),
                    np.array([-1, -2], dtype=np.int32),
                    np.array([[1], [2]], dtype=np.float32)],
                   expected=np.array([[0, 0, 0], [0, 0, 2], [0, 0, 1]],
                                     dtype=np.float32))

if __name__ == "__main__":
  googletest.main()
