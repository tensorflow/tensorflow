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
  """Test cases for argmin and argmax."""
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
    minmax_types = self.all_types & {np.int32, np.int64}
    for dtype in self.int_types | self.float_types | self.complex_types:
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

  def testNaN(self):
    minmax_types = self.all_types & {np.int32, np.int64}
    for dtype in self.float_types:
      # output_type is a numpy data type that is used to specify the desired
      # output type of the op as well as to convert the Python number to the
      # array scalar of the type.
      for output_type in minmax_types:
        # argmax NAN tests (ignore nan in tensor)
        self._assertOpOutputMatchesExpected(
          math_ops.argmax,
          axis=0,
          output_type=output_type,
          op_input=np.array(
              [np.nan, 6.0, 7.0, -1.0, 4.0, np.nan, -50.0],
              dtype=dtype),
          expected=output_type(2))
        self._assertOpOutputMatchesExpected(
          math_ops.argmax,
          axis=0,
          output_type=output_type,
          op_input=np.array([np.nan, np.nan, 1.0], dtype=dtype),
          expected=output_type(2))

        # argmin NAN tests (ignore nan in tensor)
        self._assertOpOutputMatchesExpected(
          math_ops.argmin,
          axis=0,
          output_type=output_type,
          op_input=np.array(
            [np.nan, 6.0, 7.0, -1.0, 4.0, np.nan, -50.0],
            dtype=dtype),
          expected=output_type(6))
        self._assertOpOutputMatchesExpected(
          math_ops.argmin,
          axis=0,
          output_type=output_type,
          op_input=np.array([np.nan, np.nan], dtype=dtype),
          expected=output_type(0))

  def testInf(self):
    minmax_types = self.all_types & {np.int32, np.int64}
    for dtype in self.float_types:
      # output_type is a numpy data type that is used to specify the desired
      # output type of the op as well as to convert the Python number to the
      # array scalar of the type.
      for output_type in minmax_types:
        # Argmax Inf tests
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array([-np.inf, np.inf, np.nan], dtype=dtype),
            expected=output_type(1))

        # Argmin Inf tests
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array([-np.inf, np.inf, np.nan], dtype=dtype),
            expected=output_type(0))

  def testComplex(self):
    # Test complex numbers support of argmin/argmax.
    minmax_types = self.all_types & {np.int32, np.int64}
    for dtype in self.complex_types:
      # output_type is a numpy data type that is used to specify the desired
      # output type of the op as well as to convert the Python number to the
      # array scalar of the type.
      for output_type in minmax_types:
        # Argmax tests
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [1+2j, 10+3j, 27-2j, 27+0j, 3+30j, 4+5j], dtype=dtype),
            expected=output_type(3))
        # Axis is 0
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [[4+5j, 1-6j, 7+3j], [3-7j, 2+1j, 7+8j]], dtype=dtype),
            expected=np.array([0, 1, 1], dtype=output_type))
        # Axis is 1
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=1,
            output_type=output_type,
            op_input=np.array([[4-5j, 1+6j], [4-32j, 2+2j]], dtype=dtype),
            expected=np.array([0, 0], dtype=output_type))
        # Same real
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array([3+32j, 3+3j], dtype=dtype),
            expected=output_type(0))
        # Same Image
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array([12+32j, 3+32j], dtype=dtype),
            expected=output_type(0))
        # NAN test
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [complex(2, np.nan), complex(1, np.nan)], dtype=dtype),
            expected=output_type(0))
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [complex(2, np.nan), complex(np.nan, 1)], dtype=dtype),
            expected=output_type(0))
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [complex(np.nan, np.nan), complex(np.nan, 1)], dtype=dtype),
            expected=output_type(0))
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [complex(np.nan, 1), complex(1, 1)], dtype=dtype),
            expected=output_type(1))
        # Inf test
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array([np.complex(np.inf, np.inf),
                               np.complex(0, 1)], dtype=dtype),
            expected=output_type(0))
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [np.complex(0, np.inf), np.complex(1, 1)], dtype=dtype),
            expected=output_type(1))
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [np.complex(np.inf, 3), np.complex(np.inf, 1)], dtype=dtype),
            expected=output_type(0))

        # Argmin tests
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [3+32j, 10+3j, 27-2j, 27+0j, 3+30j, 4+5j], dtype=dtype),
            expected=output_type(4))
        # Axis is 0
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [[4+5j, 1-6j, 7+3j], [3-7j, 2+1j, 7+8j]], dtype=dtype),
            expected=np.array([1, 0, 0], dtype=output_type))
        # Axis is 1
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=1,
            output_type=output_type,
            op_input=np.array([[4-5j, 1+6j], [4-32j, 2+2j]], dtype=dtype),
            expected=np.array([1, 1], dtype=output_type))
        # Same real
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array([3+32j, 3+3j], dtype=dtype),
            expected=output_type(1))
        # Same Image
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array([12+32j, 3+32j], dtype=dtype),
            expected=output_type(1))
        # NAN test
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [np.complex(2, np.nan), np.complex(1, np.nan)], dtype=dtype),
            expected=output_type(0))
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [np.complex(2, np.nan), np.complex(np.nan, 1)], dtype=dtype),
            expected=output_type(0))
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array([np.complex(np.nan, np.nan),
                               np.complex(np.nan, 1)], dtype=dtype),
            expected=output_type(0))
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [np.complex(np.nan, 1), np.complex(1, 1)], dtype=dtype),
            expected=output_type(1))
        # Inf test
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array([np.complex(np.inf, np.inf),
                               np.complex(0, 1)], dtype=dtype),
            expected=output_type(1))
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [np.complex(3, np.inf), np.complex(0, 1)], dtype=dtype),
            expected=output_type(1))
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array(
                [np.complex(np.inf, 3), np.complex(np.inf, 1)], dtype=dtype),
            expected=output_type(1))


if __name__ == "__main__":
  test.main()
