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

"""Test for version 3 of the zero_out op."""

import tensorflow as tf
from tensorflow.examples.adding_an_op import zero_out_op_3


class ZeroOut3Test(tf.test.TestCase):

  def test(self):
    result = zero_out_op_3.zero_out([5, 4, 3, 2, 1])
    self.assertAllEqual(result, [5, 0, 0, 0, 0])

  def test_attr(self):
    result = zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=3)
    self.assertAllEqual(result, [0, 0, 0, 2, 0])

  def test_negative(self):
    with self.assertRaisesOpError("Need preserve_index >= 0, got -1"):
      self.evaluate(zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=-1))

  def test_large(self):
    with self.assertRaisesOpError("preserve_index out of range"):
      self.evaluate(zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=17))


if __name__ == "__main__":
  tf.test.main()
