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
"""Tests for tensorflow.contrib.reduce_slice_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from tensorflow.contrib.reduce_slice_ops.python.ops import reduce_slice_ops
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.platform import googletest

class ReduceSliceTest(TensorFlowTestCase):

  def testReduceSliceSum1D(self):
    x = np.array([1, 40, 700], dtype=np.int32)
    indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]], dtype=np.int32)
    result = np.array([1, 741, 40, 740, 41], dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceSum2D(self):
    x = np.array([[1, 2, 3], [40, 50, 60], [700, 800, 900]], dtype=np.int32)
    indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]], dtype=np.int32)
    result = np.array([[1, 2, 3], [741, 852, 963], [40, 50, 60],
                       [740, 850, 960], [41, 52, 63]], dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceSum3D(self):
    x = np.array([[[1, 2], [3, 4]], [[50, 60], [70, 80]],
                  [[600, 700], [800, 900]]], dtype=np.int32)
    indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]], dtype=np.int32)
    result = np.array([[[1, 2], [3, 4]],
                       [[651, 762], [873, 984]],
                       [[50, 60], [70, 80]],
                       [[650, 760], [870, 980]],
                       [[51, 62], [73, 84]]], dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceSumAxis1(self):
    x = np.transpose(np.array([[1, 2, 3], [40, 50, 60],
                               [700, 800, 900]], dtype=np.int32))
    indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]], dtype=np.int32)
    result = np.transpose(np.array([[1, 2, 3],
                                    [741, 852, 963],
                                    [40, 50, 60],
                                    [740, 850, 960],
                                    [41, 52, 63]], dtype=np.int32))
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 1).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceSum1DIndices(self):
    x = np.array([[1, 2, 3], [40, 50, 60], [700, 800, 900],
                  [1000, 2000, 3000], [40000, 50000, 60000]], dtype=np.int32)
    indices = np.array([0, 0, 2, 5], dtype=np.int32)
    result = np.array([[0, 0, 0], [41, 52, 63],
                       [41700, 52800, 63900]], dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceProd(self):
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]], dtype=np.int32)
    result = np.array([[1, 2, 3], [28, 80, 162], [4, 5, 6],
                       [28, 40, 54], [4, 10, 18]], dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_prod(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceMax(self):
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]], dtype=np.int32)
    result = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6],
                       [7, 8, 9], [4, 5, 6]], dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_max(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceMin(self):
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]], dtype=np.int32)
    result = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6],
                       [4, 5, 6], [1, 2, 3]], dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_min(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceEmptyDataRows(self):
    x = np.empty((0, 1, 2, 3, 4, 5, 6), dtype=np.int32)
    indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]], dtype=np.int32)
    result = np.zeros((5, 1, 2, 3, 4, 5, 6), dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceEmptyDataCols(self):
    x = np.empty((100, 0, 2, 3, 4, 5, 6), dtype=np.int32)
    indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]], dtype=np.int32)
    result = np.empty((5, 0, 2, 3, 4, 5, 6), dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceEmptyIndicesRows(self):
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.empty((0, 2), dtype=np.int32)
    result = np.empty((0, 3), dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceEmpty0Indices1D(self):
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.empty((0,), dtype=np.int32)
    result = np.empty((0, 3), dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)

  def testReduceSliceEmpty1Indices1D(self):
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    result = np.empty((0, 3), dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = reduce_slice_ops.reduce_slice_sum(x, indices, 0).eval()
      self.assertAllEqual(y_tf, result)


if __name__ == "__main__":
  googletest.main()
