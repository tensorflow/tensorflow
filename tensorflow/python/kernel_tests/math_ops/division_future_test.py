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

This file should be exactly the same as division_past_test.py except
for the __future__ division line.
"""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class DivisionTestCase(test.TestCase):

  def testDivision(self):
    """Test all the different ways to divide."""
    values = [1, 2, 7, 11]
    functions = (lambda x: x), constant_op.constant
    # TODO(geoffreyi): Test int8, int16 once we support casts for those.
    dtypes = np.int32, np.int64, np.float32, np.float64

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
      for dtype in dtypes:
        for x in map(dtype, values):
          for y in map(dtype, values):
            for fx in functions:
              for fy in functions:
                tf_x = fx(x)
                tf_y = fy(y)
                div = x / y
                tf_div = tf_x / tf_y
                check(div, tf_div)
                floordiv = x // y
                tf_floordiv = tf_x // tf_y
                check(floordiv, tf_floordiv)
      # Do only one sess.run for speed
      for f, (x, y) in zip(checks, self.evaluate(tensors)):
        f(x, y)


if __name__ == "__main__":
  test.main()
