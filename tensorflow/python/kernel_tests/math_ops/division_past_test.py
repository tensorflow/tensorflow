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
"""Tests for division with division imported from __future__.

This file should be exactly the same as division_future_test.py except
for the __future__ division line.
"""

# from __future__ import division  # Intentionally skip this import
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class DivisionTestCase(test.TestCase):

  def testDivision(self):
    """Test all the different ways to divide."""
    values = [1, 2, 7, 11]
    functions = (lambda x: x), constant_op.constant
    test_dtypes = np.int8, np.int16, np.int32, np.int64, np.float32, np.float64

    tensors = []
    checks = []

    def check(x, y):
      x = ops.convert_to_tensor(x)
      y = ops.convert_to_tensor(y)
      tensors.append((x, y))
      def f(x, y):
        self.assertEqual(x.dtype, y.dtype)
        self.assertAllClose(x, y)
      checks.append(f)

    with self.cached_session() as sess:
      for dtype in test_dtypes:
        for x in map(dtype, values):
          for y in map(dtype, values):
            for fx in functions:
              for fy in functions:
                tf_x = fx(x)
                tf_y = fy(y)
                div = x / y
                tf_div = tf_x / tf_y
                # In NumPy 1.23, np.int16(x) / np.int16(y) now has np.float64
                # type, which disagrees with TF.
                if x.dtype not in (np.int8, np.int16):
                  check(div, tf_div)
                floordiv = x // y
                tf_floordiv = tf_x // tf_y
                check(floordiv, tf_floordiv)
      # Do only one sess.run for speed
      for f, (x, y) in zip(checks, self.evaluate(tensors)):
        f(x, y)

  @test_util.run_in_graph_and_eager_modes
  def testSignedIntegerDivisionOverflow(self):
    # Signed integer division INT_MIN / -1 overflows the signed integer
    # range and wraps around to INT_MIN in two's complement arithmetic
    # without crashing.
    cases = [
        (dtypes.int8, -128, -1, -128),
        (dtypes.int16, -32768, -1, -32768),
        (dtypes.int32, -2147483648, -1, -2147483648),
        (dtypes.int64, -9223372036854775808, -1, -9223372036854775808),
    ]
    for dtype, min_val, div_val, expected in cases:
      x = constant_op.constant([min_val], dtype=dtype)
      y = constant_op.constant([div_val], dtype=dtype)
      # Test math_ops.floordiv and operator //
      self.assertAllEqual(self.evaluate(math_ops.floordiv(x, y)), [expected])
      self.assertAllEqual(self.evaluate(x // y), [expected])
      # Test raw_ops FloorDiv, TruncateDiv, Div
      self.assertAllEqual(
          self.evaluate(gen_math_ops.floor_div(x, y)), [expected]
      )
      self.assertAllEqual(
          self.evaluate(gen_math_ops.truncate_div(x, y)), [expected]
      )
      self.assertAllEqual(self.evaluate(gen_math_ops.div(x, y)), [expected])
      # Test mod operations INT_MIN % -1 == 0
      self.assertAllEqual(self.evaluate(gen_math_ops.floor_mod(x, y)), [0])
      if dtype in (dtypes.int32, dtypes.int64):
        self.assertAllEqual(self.evaluate(gen_math_ops.truncate_mod(x, y)), [0])


if __name__ == "__main__":
  test.main()
