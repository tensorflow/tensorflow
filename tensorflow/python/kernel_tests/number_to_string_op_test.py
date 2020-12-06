# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for NumberToString op from parsing_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class NumberToStringOpTest(test.TestCase):

  def _test(self, tf_type, test_cases):
    with self.cached_session():
      # Build a small testing graph.
      input_value = array_ops.placeholder(tf_type)
      output = parsing_ops.number_to_string(input_value)

      for test_case in test_cases:
        invalue = test_case[0]
        result, = output.eval(feed_dict={input_value: [invalue]})
        outnum = test_case[1] if len(test_case) > 1 else str(invalue)
        self.assertAllEqual(result.decode("ascii"), outnum)

  @test_util.run_deprecated_v1
  def testFromFloat(self):
    self._test(dtypes.float32,
               [(0, "0"), (3, "3"), (-1, "-1"),
                (1.12, "1.12"),
                (3.40282e+38, "3.40282e+38"),
                # # Greater than max value of float.
                (3.40283e+38, "inf"),
                (-3.40283e+38, "-inf"),
                # Less than min value of float.
                (float("nan"), "nan"),
                (float("inf"), "inf")])

  @test_util.run_deprecated_v1
  def testFromDouble(self):
    self._test(dtypes.float64,
               [(0, "0"), (3, "3"), (-1, "-1"),
                (1.12,), (0xF, "15"),
                (3.40282e+38, "3.40282e+38"),
                # Greater than max value of float.
                (3.40283e+38, "3.40283e+38"),
                # Less than min value of float.
                (-3.40283e+38, "-3.40283e+38"),
                (float("nan"), "nan"),
                (float("inf"), "inf")])

  @test_util.run_deprecated_v1
  def testFromInt32(self):
    self._test(dtypes.int32,
               [(0, "0"), (3, "3"), (-1, "-1"),
                (-2147483648, "-2147483648"),
                (2147483647, "2147483647")])

  @test_util.run_deprecated_v1
  def testFromInt64(self):
    self._test(dtypes.int64,
               [(0, "0"), (3, "3"), (-1, "-1"),
                (-2147483648, "-2147483648"),
                (2147483647, "2147483647"),
                # Less than min value of int32.
                (-2147483649, "-2147483649"),
                # Greater than max value of int32.
                (2147483648, "2147483648")])


if __name__ == "__main__":
  test.main()
