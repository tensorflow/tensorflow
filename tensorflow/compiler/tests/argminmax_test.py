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
"""Functional tests for ArgMin and ArgMax Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ArgMinMaxTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self, op, inp, expected):
    """Verifies that 'op' produces 'expected' when fed input 'inp' .

    Args:
      op: operator to test
      inp: numpy input array to use as input to 'op'.
      expected: numpy array representing the expected output of 'op'.
    """
    with self.test_session() as session:
      with self.test_scope():
        pinp = array_ops.placeholder(
            dtypes.as_dtype(inp.dtype), inp.shape, name="a")
        output = op(pinp)
      result = session.run(output, {pinp: inp})
      self.assertAllEqual(result, expected)

  def testArgMinMax(self):
    # Complex numbers do not support argmin/argmax.
    minmax_types = set(self.numeric_types) - set(self.complex_types)
    for dtype in minmax_types:
      self._assertOpOutputMatchesExpected(
          lambda x: math_ops.argmax(x, axis=0, output_type=dtypes.int32),
          np.array([1, 10, 27, 3, 3, 4], dtype=dtype),
          expected=np.int32(2))
      self._assertOpOutputMatchesExpected(
          lambda x: math_ops.argmax(x, axis=0, output_type=dtypes.int32),
          np.array([[4, 1, 7], [3, 2, 4]], dtype=dtype),
          expected=np.array([0, 1, 0], dtype=np.int32))
      self._assertOpOutputMatchesExpected(
          lambda x: math_ops.argmax(x, axis=1, output_type=dtypes.int32),
          np.array([[4, 1], [3, 2]], dtype=dtype),
          expected=np.array([0, 0], dtype=np.int32))

      self._assertOpOutputMatchesExpected(
          lambda x: math_ops.argmin(x, axis=0, output_type=dtypes.int32),
          np.array([3, 10, 27, 3, 2, 4], dtype=dtype),
          expected=np.int32(4))
      self._assertOpOutputMatchesExpected(
          lambda x: math_ops.argmin(x, axis=0, output_type=dtypes.int32),
          np.array([[4, 1, 7], [3, 2, 4]], dtype=dtype),
          expected=np.array([1, 0, 1], dtype=np.int32))
      self._assertOpOutputMatchesExpected(
          lambda x: math_ops.argmin(x, axis=1, output_type=dtypes.int32),
          np.array([[4, 1], [3, 2]], dtype=dtype),
          expected=np.array([1, 1], dtype=np.int32))


if __name__ == "__main__":
  test.main()
