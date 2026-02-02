# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for string_join_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class StringJoinOpTest(tf.test.TestCase):

  def testStringJoin(self):
    input0 = ["a", "b"]
    input1 = "a"
    input2 = [["b"], ["c"]]

    with self.test_session():
      output = tf.string_join([input0, input1])
      self.assertAllEqual(output.eval(), [b"aa", b"ba"])

      output = tf.string_join([input0, input1], separator="--")
      self.assertAllEqual(output.eval(), [b"a--a", b"b--a"])

      output = tf.string_join([input0, input1, input0], separator="--")
      self.assertAllEqual(output.eval(), [b"a--a--a", b"b--a--b"])

      output = tf.string_join([input1] * 4, separator="!")
      self.assertEqual(output.eval(), b"a!a!a!a")

      output = tf.string_join([input2] * 2, separator="")
      self.assertAllEqual(output.eval(), [[b"bb"], [b"cc"]])

      with self.assertRaises(ValueError):  # Inconsistent shapes
        tf.string_join([input0, input2]).eval()


if __name__ == "__main__":
  tf.test.main()
