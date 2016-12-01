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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


_ERROR_MESSAGE = "StringToNumberOp could not correctly convert string: "


class StringToNumberOpTest(tf.test.TestCase):

  def testToFloat(self):
    with self.test_session():
      input_string = tf.placeholder(tf.string)
      output = tf.string_to_number(
          input_string,
          out_type=tf.float32)

      result = output.eval(feed_dict={
          input_string: ["0",
                         "3",
                         "-1",
                         "1.12",
                         "0xF",
                         "   -10.5",
                         "3.40282e+38",
                         # The next two exceed maximum value for float, so we
                         # expect +/-INF to be returned instead.
                         "3.40283e+38",
                         "-3.40283e+38",
                         "NAN",
                         "INF"]
      })

      self.assertAllClose([0, 3, -1, 1.12, 0xF, -10.5, 3.40282e+38,
                           float("INF"), float("-INF"), float("NAN"),
                           float("INF")], result)

      with self.assertRaisesOpError(_ERROR_MESSAGE + "10foobar"):
        output.eval(feed_dict={input_string: ["10foobar"]})

  def testToInt32(self):
    with self.test_session():
      input_string = tf.placeholder(tf.string)
      output = tf.string_to_number(
          input_string,
          out_type=tf.int32)

      result = output.eval(feed_dict={
          input_string: ["0", "3", "-1", "    -10", "-2147483648", "2147483647"]
      })

      self.assertAllEqual([0, 3, -1, -10, -2147483648, 2147483647], result)

      with self.assertRaisesOpError(_ERROR_MESSAGE + "2.9"):
        output.eval(feed_dict={input_string: ["2.9"]})

      # The next two exceed maximum value of int32.
      for in_string in ["-2147483649", "2147483648"]:
        with self.assertRaisesOpError(_ERROR_MESSAGE + in_string):
          output.eval(feed_dict={input_string: [in_string]})


if __name__ == "__main__":
  tf.test.main()
