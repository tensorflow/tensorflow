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
"""Functional tests for Tile op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test

class TileTest(test.TestCase):

  def _np_tile(self, x, multiples):
    ret = np.tile(x, multiples)
    return ret

  def _testTile(self, x, mult, use_gpu):
    np_ans = self._np_tile(x, mult)
    with self.test_session(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      y = array_ops.tile(inx, mult)
      tf_ans = y.eval()
      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

  def _test1D(self, dtype):
    x = np.asarray([1, 2, 3], dtype=dtype)
    mult = np.asarray([2])
    self._testTile(x, mult, False)
    self._testTile(x, mult, True)

  def _test2D(self, dtype):
    x = np.asarray([[1, 2, 3],[4, 5, 6]], dtype=dtype)
    mult = np.asarray([2, 2])
    self._testTile(x, mult, False)
    self._testTile(x, mult, True)

  def _test2DZero1(self, dtype):
    x = np.asarray([[1, 2, 3],[4, 5, 6]], dtype=dtype)
    mult = np.asarray([0, 2])
    self._testTile(x, mult, False)
    self._testTile(x, mult, True)

  def _test2DZero2(self, dtype):
    x = np.asarray([[1, 2, 3],[4, 5, 6]], dtype=dtype)
    mult = np.asarray([2, 0])
    self._testTile(x, mult, False)
    self._testTile(x, mult, True)

  def _test3D(self, dtype):
    x = np.asarray([[[1, 2, 3],[4, 5, 6]]], dtype=dtype)
    mult = np.asarray([2, 2, 3])
    self._testTile(x, mult, False)
    self._testTile(x, mult, True)

  def _test9D(self, dtype):
    x = np.asarray([[[[[[[[[1, 2]]]]]]]]], dtype=dtype)
    mult = np.asarray([2, 1, 1, 1, 1, 1, 1, 2, 1])
    self._testTile(x, mult, False)
    self._testTile(x, mult, True)

  def test1D(self):
    self._test1D(np.int16)
    self._test1D(np.int32)
    self._test1D(np.int64)
    self._test1D(np.float16)
    self._test1D(np.float32)
    self._test1D(np.float64)
    self._test1D(np.complex64)
    self._test1D(np.complex128)

  def test2D(self):
    self._test2D(np.int16)
    self._test2D(np.int32)
    self._test2D(np.int64)
    self._test2D(np.float16)
    self._test2D(np.float32)
    self._test2D(np.float64)
    self._test2D(np.complex64)
    self._test2D(np.complex128)

  def test3D(self):
    self._test3D(np.int16)
    self._test3D(np.int32)
    self._test3D(np.int64)
    self._test3D(np.float16)
    self._test3D(np.float32)
    self._test3D(np.float64)
    self._test3D(np.complex64)
    self._test3D(np.complex128)

  def test9D(self):
    self._test9D(np.int16)
    self._test9D(np.int32)
    self._test9D(np.int64)
    self._test9D(np.float16)
    self._test9D(np.float32)
    self._test9D(np.float64)
    self._test9D(np.complex64)
    self._test9D(np.complex128)

  def testString(self):
    x = np.asarray(['a', 'b', 'c'])
    mult = np.asarray([2])
    np_ans = self._np_tile(x, mult)
    with self.test_session(use_gpu=False):
      inx = ops.convert_to_tensor(x)
      y = array_ops.tile(inx, mult)
      tf_ans = y.eval()
      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

  def testShapeEqualError(self):
    x = np.asarray([[0], [1]])
    mult = np.asarray([2])
    with self.assertRaisesRegexp(ValueError, "Shape must be rank 2 but is rank 1 for"):
      array_ops.tile(x, mult)

  def testShape1DError(self):
    x = np.asarray([[0], [1]])
    mult = np.asarray([[[2]]])
    with self.assertRaisesRegexp(ValueError, "Shape must be rank 1 but is rank 3 for"):
      array_ops.tile(x, mult)

  def testNegativeError(self):
    x = np.asarray([1])
    mult = np.asarray([-2])
    with self.assertRaisesRegexp(ValueError, "Dimension -2 must be >= 0"):
      array_ops.tile(x, mult)

if __name__ == "__main__":
  test.main()
