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
"""Functional tests for reduction ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BaseReductionTest(test.TestCase):

  def _tf_reduce(self, x, reduction_axes, keep_dims):
    raise NotImplementedError()


class SumReductionTest(BaseReductionTest):

  def _tf_reduce(self, x, reduction_axes, keep_dims):
    return math_ops.reduce_sum(x, reduction_axes, keep_dims)

  def testFloat32(self):
    # make sure we test all possible kernel invocations
    # logic is the same for all ops, test just float32 for brevity
    for size_x in range(1, 4105, 27):
      for size_y in range(1, 4105, 27):
        arr = np.ones([size_x, size_y], dtype=np.float32)
        col_sum = np.ones([size_y], dtype=np.float32) * size_x
        row_sum = np.ones([size_x], dtype=np.float32) * size_y
        full_sum = np.ones([], dtype=np.float32) * size_x * size_y

        with self.test_session(graph=ops.Graph(), use_gpu=True) as sess:
          tf_row_sum = self._tf_reduce(arr, 1, False)
          tf_col_sum = self._tf_reduce(arr, 0, False)
          tf_full_sum = self._tf_reduce(arr, [0, 1], False)
          tf_out_row, tf_out_col, tf_out_full = sess.run(
              [tf_row_sum, tf_col_sum, tf_full_sum])
        self.assertAllClose(col_sum, tf_out_col)
        self.assertAllClose(row_sum, tf_out_row)
        self.assertAllClose(full_sum, tf_out_full)

    for size_x in range(1, 130, 3):
      for size_y in range(1, 130, 3):
        for size_z in range(1, 130, 3):
          arr = np.ones([size_x, size_y, size_z], dtype=np.float32)
          sum_y = np.sum(arr, axis=1)
          sum_xz = np.sum(arr, axis=(0, 2))

          with self.test_session(graph=ops.Graph(), use_gpu=True) as sess:
            tf_sum_xz = self._tf_reduce(arr, [0, 2], False)
            tf_sum_y = self._tf_reduce(arr, 1, False)
            tf_out_sum_xz, tf_out_sum_y = sess.run([tf_sum_xz, tf_sum_y])
          self.assertAllClose(sum_y, tf_out_sum_y)
          self.assertAllClose(sum_xz, tf_out_sum_xz)


if __name__ == "__main__":
  test.main()
