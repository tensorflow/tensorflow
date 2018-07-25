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
"""Tests for XLA dynamic slicing ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DynamicUpdateSliceOpsTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self, op, args, expected):
    with self.test_session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(*placeholders)
      result = session.run(output, feeds)
      self.assertAllClose(result, expected, rtol=1e-3)

  def testUpdateSlice(self):
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          xla.dynamic_update_slice, [
              np.array([], dtype=dtype),
              np.array([], dtype=dtype),
              np.array([0], dtype=np.int32)
          ],
          expected=np.array([], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          xla.dynamic_update_slice, [
              np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype),
              np.array([-1, -2, -3], dtype=dtype),
              np.array([6], dtype=np.int32)
          ],
          expected=np.array([1, 2, 3, 4, 5, 6, -1, -2, -3, 10], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          xla.dynamic_update_slice, [
              np.array(
                  [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=dtype),
              np.array([[42, 43], [44, 45]], dtype=dtype),
              np.array([1, 2], dtype=np.int32)
          ],
          expected=np.array(
              [[1, 2, 3, 4], [5, 6, 42, 43], [9, 10, 44, 45]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          xla.dynamic_update_slice, [
              np.array(
                  [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=dtype),
              np.array([[], []], dtype=dtype),
              np.array([1, 2], dtype=np.int32)
          ],
          expected=np.array(
              [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          xla.dynamic_update_slice, [
              np.array(
                  [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=dtype),
              np.ones([3, 4], dtype=dtype),
              np.array([0, 0], dtype=np.int32)
          ],
          expected=np.ones([3, 4], dtype=dtype))


if __name__ == '__main__':
  test.main()
