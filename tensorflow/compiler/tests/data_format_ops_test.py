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
"""Tests for the DataFormatVecPermute operator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class XlaDataFormatDimMapTest(xla_test.XLATestCase):

  def _test(self, input_data, src_format, dst_format, expected):
    for dtype in {np.int32, np.int64}:
      x = np.array(input_data, dtype=dtype)
      with self.session() as session:
        with self.test_scope():
          placeholder = array_ops.placeholder(dtypes.as_dtype(x.dtype), x.shape)
          param = {placeholder: x}
          output = nn_ops.data_format_dim_map(
              placeholder, src_format=src_format, dst_format=dst_format)
        result = session.run(output, param)
      self.assertAllEqual(result, expected)

  def test(self):
    self._test(0, "NHWC", "NCHW", 0)
    self._test(1, "NHWC", "NCHW", 2)
    self._test(2, "NHWC", "NCHW", 3)
    self._test(3, "NHWC", "NCHW", 1)
    self._test(-1, "NHWC", "NCHW", 1)
    self._test(-2, "NHWC", "NCHW", 3)
    self._test(-3, "NHWC", "NCHW", 2)
    self._test(-4, "NHWC", "NCHW", 0)
    self._test([1, 3], "NHWC", "NCHW", [2, 1])
    self._test([1, 3, -2], "NHWC", "NCHW", [2, 1, 3])
    self._test([1, -3, -2], "NHWC", "NCHW", [2, 2, 3])
    self._test([[1, -3], [1, -1]], "NHWC", "NCHW", [[2, 2], [2, 1]])

    self._test([1, -3, -2], "NHWC", "NCHW", [2, 2, 3])
    self._test([-4, -3, -2, -1, 0, 1, 2, 3], "NHWC", "HWNC",
               [2, 0, 1, 3, 2, 0, 1, 3])
    self._test([-4, -3, -2, -1, 0, 1, 2, 3], "NHWC", "WHCN",
               [3, 1, 0, 2, 3, 1, 0, 2])
    self._test([-4, -3, -2, -1, 0, 1, 2, 3], "qwer", "rewq",
               [3, 2, 1, 0, 3, 2, 1, 0])


class XlaPermuteOpTest(xla_test.XLATestCase):

  def _runPermuteAndCompare(self, x, src_format, dst_format, expected):
    with self.session() as session:
      with self.test_scope():
        placeholder = array_ops.placeholder(dtypes.as_dtype(x.dtype), x.shape)
        param = {placeholder: x}
        output = nn_ops.data_format_vec_permute(
            placeholder, src_format=src_format, dst_format=dst_format)
      result = session.run(output, param)
    self.assertAllEqual(result, expected)

  def testNHWCToNCHW(self):
    for dtype in {np.int32, np.int64}:
      x = np.array([7, 4, 9, 3], dtype=dtype)
      self._runPermuteAndCompare(x, "NHWC", "NCHW", [7, 3, 4, 9])

  def testNCHWToNHWC(self):
    for dtype in {np.int32, np.int64}:
      x = np.array([7, 4, 9, 3], dtype=dtype)
      self._runPermuteAndCompare(x, "NCHW", "NHWC", [7, 9, 3, 4])

  def testNHWCToHWNC(self):
    for dtype in {np.int32, np.int64}:
      x = np.array([7, 4, 9, 3], dtype=dtype)
      self._runPermuteAndCompare(x, "NHWC", "HWNC", [4, 9, 7, 3])

  def testHWNCToNHWC(self):
    for dtype in {np.int32, np.int64}:
      x = np.array([7, 4, 9, 3], dtype=dtype)
      self._runPermuteAndCompare(x, "HWNC", "NHWC", [9, 7, 4, 3])

  def testNHWCToNCHW2D(self):
    for dtype in {np.int32, np.int64}:
      x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
      self._runPermuteAndCompare(x, "NHWC", "NCHW",
                                 [[7, 4], [5, 1], [9, 3], [4, 5]])

  def testNHWCToHWNC2D(self):
    for dtype in {np.int32, np.int64}:
      x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
      self._runPermuteAndCompare(x, "NHWC", "HWNC",
                                 [[9, 3], [4, 5], [7, 4], [5, 1]])

  def testHWNCToNHWC2D(self):
    for dtype in {np.int32, np.int64}:
      x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
      self._runPermuteAndCompare(x, "HWNC", "NHWC",
                                 [[4, 5], [7, 4], [9, 3], [5, 1]])

  def testNCHWToNHWC2D(self):
    for dtype in {np.int32, np.int64}:
      x = np.array([[7, 4], [9, 3], [4, 5], [5, 1]], dtype=dtype)
      self._runPermuteAndCompare(x, "NCHW", "NHWC",
                                 [[7, 4], [4, 5], [5, 1], [9, 3]])


if __name__ == "__main__":
  test.main()
