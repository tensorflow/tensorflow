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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ArgMinMaxTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self, op, axis, output_type, op_input,
                                     expected):
    """Verifies that 'op' produces 'expected' when fed input 'op_input' .

    Args:
      op: argmin or argmax operator to test.
      axis: integer axis to reduce across.
      output_type: numpy datatype of the output to produce.
      op_input: numpy input array to use as input to 'op'.
      expected: numpy array representing the expected output of 'op'.
    """
    with self.session() as session:
      with self.test_scope():
        pinp = array_ops.placeholder(
            dtypes.as_dtype(op_input.dtype), op_input.shape, name="a")
        output = op(pinp, axis=axis, output_type=output_type)
      result = session.run(output, {pinp: op_input})
      self.assertAllEqual(result, expected)

  def testArgMinMax(self):
    # Complex numbers do not support argmin/argmax.
    minmax_types = self.all_types & {np.int32, np.int64}
    for dtype in self.int_types | self.float_types:
      # output_type is a numpy data type that is used to specify the desired
      # output type of the op as well as to convert the Python number to the
      # array scalar of the type.
      for output_type in minmax_types:
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array([1, 10, 27, 3, 3, 4], dtype=dtype),
            expected=output_type(2))
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array([[4, 1, 7], [3, 2, 4]], dtype=dtype),
            expected=np.array([0, 1, 0], dtype=output_type))
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=1,
            output_type=output_type,
            op_input=np.array([[4, 1], [3, 2]], dtype=dtype),
            expected=np.array([0, 0], dtype=output_type))

        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array([3, 10, 27, 3, 2, 4], dtype=dtype),
            expected=output_type(4))
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array([[4, 1, 7], [3, 2, 4]], dtype=dtype),
            expected=np.array([1, 0, 1], dtype=output_type))
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=1,
            output_type=output_type,
            op_input=np.array([[4, 1], [3, 2]], dtype=dtype),
            expected=np.array([1, 1], dtype=output_type))

  def testArgMinMaxNaN(self):
    # IEEE-754 comparisons against NaN are always false, which used to make XLA
    # report the index of a NaN instead of the index of the finite extremum.
    # Eager execution ignores NaNs whenever a finite value is present, and
    # returns the lowest index when every value is NaN.
    cases = [
        # A NaN must not mask a finite extremum, wherever it appears.
        (math_ops.argmax, [4.0, 5.5, np.nan], 1),
        (math_ops.argmax, [np.nan, 4.0, 5.5], 2),
        (math_ops.argmax, [4.0, np.nan, 5.5], 2),
        (math_ops.argmin, [4.0, 5.5, np.nan], 0),
        (math_ops.argmin, [np.nan, 4.0, 5.5], 1),
        (math_ops.argmin, [4.0, np.nan, 5.5], 0),
        # With no finite value to select, the lowest index wins.
        (math_ops.argmax, [np.nan, np.nan], 0),
        (math_ops.argmin, [np.nan, np.nan], 0),
        # The reduction is seeded with -inf for argmax and +inf for argmin, so
        # an infinity in the data ties with the init value. The NaN must still
        # lose, and the lowest tied index must win.
        (math_ops.argmax, [-np.inf, np.nan, -np.inf], 0),
        (math_ops.argmin, [np.inf, np.nan, np.inf], 0),
    ]
    for dtype in self.float_types:
      for op, op_input, expected in cases:
        self._assertOpOutputMatchesExpected(
            op,
            axis=0,
            output_type=np.int32,
            op_input=np.array(op_input, dtype=dtype),
            expected=np.int32(expected),
        )
      # Reducing along an axis of a 2-D input exercises several independent
      # reduction lanes, including one that is entirely NaN.
      self._assertOpOutputMatchesExpected(
          math_ops.argmax,
          axis=1,
          output_type=np.int32,
          op_input=np.array(
              [[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]], dtype=dtype
          ),
          expected=np.array([2, 0], dtype=np.int32),
      )


if __name__ == "__main__":
  test.main()
