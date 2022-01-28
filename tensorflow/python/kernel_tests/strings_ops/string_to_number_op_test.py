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
"""Tests for StringToNumber op from parsing_ops."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test

_ERROR_MESSAGE = "StringToNumberOp could not correctly convert string: "


class StringToNumberOpTest(test.TestCase):

  def _test(self, tf_type, good_pairs, bad_pairs):
    with self.cached_session():
      # Build a small testing graph.
      input_string = array_ops.placeholder(dtypes.string)
      output = parsing_ops.string_to_number(
          input_string, out_type=tf_type)

      # Check all the good input/output pairs.
      for instr, outnum in good_pairs:
        result, = output.eval(feed_dict={input_string: [instr]})
        self.assertAllClose([outnum], [result])

      # Check that the bad inputs produce the right errors.
      for instr, outstr in bad_pairs:
        with self.assertRaisesOpError(outstr):
          output.eval(feed_dict={input_string: [instr]})

  @test_util.run_deprecated_v1
  def testToFloat(self):
    self._test(dtypes.float32,
               [("0", 0), ("3", 3), ("-1", -1),
                ("1.12", 1.12), ("0xF", 15), ("   -10.5", -10.5),
                ("3.40282e+38", 3.40282e+38),
                # Greater than max value of float.
                ("3.40283e+38", float("INF")),
                ("-3.40283e+38", float("-INF")),
                # Less than min value of float.
                ("NAN", float("NAN")),
                ("INF", float("INF"))],
               [("10foobar", _ERROR_MESSAGE + "10foobar")])

  @test_util.run_deprecated_v1
  def testToDouble(self):
    self._test(dtypes.float64,
               [("0", 0), ("3", 3), ("-1", -1),
                ("1.12", 1.12), ("0xF", 15), ("   -10.5", -10.5),
                ("3.40282e+38", 3.40282e+38),
                # Greater than max value of float.
                ("3.40283e+38", 3.40283e+38),
                # Less than min value of float.
                ("-3.40283e+38", -3.40283e+38),
                ("NAN", float("NAN")),
                ("INF", float("INF"))],
               [("10foobar", _ERROR_MESSAGE + "10foobar")])

  @test_util.run_deprecated_v1
  def testToInt32(self):
    self._test(dtypes.int32,
               [("0", 0), ("3", 3), ("-1", -1),
                ("    -10", -10),
                ("-2147483648", -2147483648),
                ("2147483647", 2147483647)],
               [   # Less than min value of int32.
                   ("-2147483649", _ERROR_MESSAGE + "-2147483649"),
                   # Greater than max value of int32.
                   ("2147483648", _ERROR_MESSAGE + "2147483648"),
                   ("2.9", _ERROR_MESSAGE + "2.9"),
                   ("10foobar", _ERROR_MESSAGE + "10foobar")])

  @test_util.run_deprecated_v1
  def testToInt64(self):
    self._test(dtypes.int64,
               [("0", 0), ("3", 3), ("-1", -1),
                ("    -10", -10),
                ("-2147483648", -2147483648),
                ("2147483647", 2147483647),
                ("-2147483649", -2147483649),  # Less than min value of int32.
                ("2147483648", 2147483648)],  # Greater than max value of int32.
               [("2.9", _ERROR_MESSAGE + "2.9"),
                ("10foobar", _ERROR_MESSAGE + "10foobar")])


if __name__ == "__main__":
  test.main()
