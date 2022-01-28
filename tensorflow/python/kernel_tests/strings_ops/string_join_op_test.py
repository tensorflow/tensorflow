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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringJoinOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testStringJoin(self):
    input0 = ["a", "b"]
    input1 = "a"
    input2 = [["b"], ["c"]]

    with self.cached_session():
      output = string_ops.string_join([input0, input1])
      self.assertAllEqual(output, [b"aa", b"ba"])

      output = string_ops.string_join([input0, input1], separator="--")
      self.assertAllEqual(output, [b"a--a", b"b--a"])

      output = string_ops.string_join([input0, input1, input0], separator="--")
      self.assertAllEqual(output, [b"a--a--a", b"b--a--b"])

      output = string_ops.string_join([input1] * 4, separator="!")
      self.assertEqual(self.evaluate(output), b"a!a!a!a")

      output = string_ops.string_join([input2] * 2, separator="")
      self.assertAllEqual(output, [[b"bb"], [b"cc"]])

      with self.assertRaises(ValueError):  # Inconsistent shapes
        string_ops.string_join([input0, input2]).eval()


if __name__ == "__main__":
  test.main()
