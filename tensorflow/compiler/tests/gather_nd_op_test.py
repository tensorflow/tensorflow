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
"""Tests for tensorflow.ops.tf.gather_nd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class GatherNdTest(xla_test.XLATestCase):

  def _runGather(self, params, indices):
    with self.session():
      paramsp = array_ops.placeholder(params.dtype)
      indicesp = array_ops.placeholder(indices.dtype)
      with self.test_scope():
        gather_nd_t = array_ops.gather_nd(paramsp, indicesp)
      feed_dict = {paramsp: params, indicesp: indices}
      return gather_nd_t.eval(feed_dict=feed_dict)

  def testSimpleDtype(self):
    for dtype in self.numeric_types:
      self.assertAllEqual(
          np.array([7, 7, 8], dtype=dtype),
          self._runGather(
              np.array([8, 1, 2, 3, 7, 5], dtype=dtype),
              np.array([[4], [4], [0]], np.int32)))

  @test_util.disable_mlir_bridge("Error handling")
  def testEmptyIndicesAndParamsOKButJustEmptyParamsFails(self):
    with self.session():
      params = np.ones((3, 3), dtype=np.float32)

      indices_empty = np.empty((0, 2), dtype=np.int32)
      gather_nd_ok_val = self._runGather(params, indices_empty)
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

      indices_empty = np.empty((0, 1), dtype=np.int32)
      gather_nd_ok_val = self._runGather(params, indices_empty)
      self.assertAllClose(np.empty((0, 3), dtype=np.float32), gather_nd_ok_val)

      params_empty = np.empty((0, 3), dtype=np.float32)
      indices_empty = np.empty((0, 2), dtype=np.int32)
      gather_nd_ok_val = self._runGather(params_empty, indices_empty)
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

      params_empty = np.empty((0, 3), dtype=np.float32)
      indices_nonempty = np.zeros((1, 2), dtype=np.int32)
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError, r"Gather dimension 0 is of size zero"):
        self._runGather(params_empty, indices_nonempty)

  def testIndexScalar(self):
    params = np.array(
        [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
    indices = np.array([4, 1], dtype=np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(np.array(7), gather_nd_val)

  def testParamsRankLargerThanIndexIndexScalarSlices(self):
    params = np.array(
        [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
    indices = np.array(
        [
            4,
        ], dtype=np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(np.array([-7, 7]), gather_nd_val)

  def testParamsRankLargerThanIndexSlices(self):
    params = np.array(
        [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
    indices = np.array([[4], [4], [0]], np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(np.array([[-7, 7], [-7, 7], [-8, 8]]), gather_nd_val)

  def testHigherRankParamsLargerThanIndexSlices(self):
    params = np.array(
        [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
         [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
        dtype=np.float32).T
    indices = np.array([[4], [4], [0]], np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(params[[4, 4, 0]], gather_nd_val)

  def testEmptyIndicesLastRankMeansCopyEntireTensor(self):
    params = np.array(
        [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
         [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
        dtype=np.float32).T
    indices = np.array([[], []], dtype=np.int32)  # Size (2, 0)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(
        np.vstack((params[np.newaxis, :], params[np.newaxis, :])),
        gather_nd_val)

  def testHigherRankParamsAndIndicesLargerThanIndexSlices(self):
    params = np.array(
        [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
         [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
        dtype=np.float32).T
    indices = np.array([[[3], [2], [1]], [[4], [4], [0]]], np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(params[[3, 2, 1, 4, 4, 0]].reshape(2, 3, 2, 2),
                        gather_nd_val)

  def testHigherRankParams(self):
    shape = (10, 20, 5, 1, 17)
    params = np.random.rand(*shape).astype(np.float32)
    indices = np.vstack(
        [np.random.randint(0, s, size=2000, dtype=np.int32) for s in shape]).T
    gather_nd_val = self._runGather(params, indices)

    expected = params[tuple(indices.T)]
    self.assertAllEqual(expected, gather_nd_val)

  def testHigherRankParamsAndIndices(self):
    shape = (10, 20, 5, 1, 17)
    params = np.random.rand(*shape).astype(np.float32)
    indices = np.vstack(
        [np.random.randint(0, s, size=2000, dtype=np.int32) for s in shape]).T
    indices_reshaped = indices.reshape([10, 10, 20, 5])
    gather_nd_val = self._runGather(params, indices_reshaped)
    expected = params[tuple(indices.T)]
    self.assertAllEqual(expected.reshape([10, 10, 20]), gather_nd_val)


if __name__ == "__main__":
  test.main()
