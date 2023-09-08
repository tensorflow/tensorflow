# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for stochastic cast op generation ops."""

import math

from absl.testing import parameterized

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import stochastic_cast_op
from tensorflow.python.platform import test


def _allowed_from_types():
  return {
      dtypes.float64,
      dtypes.float32,
      dtypes.float16,
      dtypes.bfloat16,
      dtypes.half,
  }


def _return_saturate_value(is_negative=False, dtype=dtypes.int32):
  if dtype is dtypes.int32:
    if is_negative:
      return -(2**31)
    return 2**31 - 1
  if dtype is dtypes.int16:
    if is_negative:
      return -(2**15)
    return 2**15 - 1
  if dtype is dtypes.int8:
    if is_negative:
      return -(2**7)
    return 2**7 - 1


class StochasticCastOpTest(xla_test.XLATestCase, parameterized.TestCase):
  """Test cases for stochastic cast operator."""

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(
      (
          f"_{value}_from_{from_dtype.name}_to_{to_dtype.name}",
          value,
          from_dtype,
          to_dtype,
      )
      for value in [0.0025, 0.125, 0.625, 8.5]
      for from_dtype in _allowed_from_types()
      for to_dtype in stochastic_cast_op.allowed_to_types(is_integer=True)
  )
  # pylint: enable=g-complex-comprehension
  def testStochasticCastOpResultProbability(self, value, from_dtype, to_dtype):
    test_value = value
    with self.session() as sess, self.test_scope():
      input_t = constant_op.constant(test_value, from_dtype, [1000, 1000])
      op = stochastic_cast_op.stochastic_cast(
          input_t, to_dtype, [12345, 12345], "auto_select"
      )
      result = sess.run(op)
      expected = (test_value - math.floor(test_value)) / (
          math.ceil(test_value) - test_value
      )
      actual = (result == math.ceil(test_value)).sum() / (
          result == math.floor(test_value)
      ).sum()
      self.assertNear(expected, actual, 0.05)

    # pylint: disable=g-complex-comprehension

  @parameterized.named_parameters(
      (
          f"_{value}_from_{from_dtype.name}_to_{to_dtype.name}",
          value,
          from_dtype,
          to_dtype,
      )
      for value in [2**33, -(2**34), 2**35, -(2**36)]
      for from_dtype in _allowed_from_types()
      for to_dtype in stochastic_cast_op.allowed_to_types(is_integer=True)
  )
  # pylint: enable=g-complex-comprehension
  def testStochasticCastOpSaturateOutOfRange(self, value, from_dtype, to_dtype):
    test_value = value
    with self.session() as sess, self.test_scope():
      input_t = constant_op.constant(test_value, from_dtype, [])
      op = stochastic_cast_op.stochastic_cast(
          input_t, to_dtype, [12345, 12345], "auto_select"
      )
      result = sess.run(op)
      self.assertEqual(result, _return_saturate_value(value < 0, to_dtype))


if __name__ == "__main__":
  test.main()
