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
"""Tests for IdentityNOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class IdentityNOpTest(test.TestCase):

  def testInt32String_6(self):
    with self.test_session() as sess:
      [value0, value1] = sess.run(
          array_ops.identity_n([[1, 2, 3, 4, 5, 6],
                                [b"a", b"b", b"C", b"d", b"E", b"f", b"g"]]))
    self.assertAllEqual(np.array([1, 2, 3, 4, 5, 6]), value0)
    self.assertAllEqual(
        np.array([b"a", b"b", b"C", b"d", b"E", b"f", b"g"]), value1)

  def testInt32_shapes(self):
    with self.test_session() as sess:
      inp0 = constant_op.constant([10, 20, 30, 40, 50, 60], shape=[2, 3])
      inp1 = constant_op.constant([11, 21, 31, 41, 51, 61], shape=[3, 2])
      inp2 = constant_op.constant(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], shape=[5, 3])
      [value0, value1,
       value2] = sess.run(array_ops.identity_n([inp0, inp1, inp2]))
    self.assertAllEqual(np.array([[10, 20, 30], [40, 50, 60]]), value0)
    self.assertAllEqual(np.array([[11, 21], [31, 41], [51, 61]]), value1)
    self.assertAllEqual(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
        value2)

  def testString(self):
    source = [b"A", b"b", b"C", b"d", b"E", b"f"]
    with self.test_session() as sess:
      [value] = sess.run(array_ops.identity_n([source]))
    self.assertAllEqual(source, value)

  def testIdentityShape(self):
    with self.test_session():
      shape = [2, 3]
      array_2x3 = [[1, 2, 3], [6, 5, 4]]
      tensor = constant_op.constant(array_2x3)
      self.assertEquals(shape, tensor.get_shape())
      self.assertEquals(shape, array_ops.identity_n([tensor])[0].get_shape())
      self.assertEquals(shape, array_ops.identity_n([array_2x3])[0].get_shape())


if __name__ == "__main__":
  test.main()
