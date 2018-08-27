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
"""Tests for IdentityOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class IdentityOpTest(test.TestCase):

  def testInt32_6(self):
    with self.test_session():
      value = array_ops.identity([1, 2, 3, 4, 5, 6]).eval()
    self.assertAllEqual(np.array([1, 2, 3, 4, 5, 6]), value)

  def testInt32_2_3(self):
    with self.test_session():
      inp = constant_op.constant([10, 20, 30, 40, 50, 60], shape=[2, 3])
      value = array_ops.identity(inp).eval()
    self.assertAllEqual(np.array([[10, 20, 30], [40, 50, 60]]), value)

  def testString(self):
    source = [b"A", b"b", b"C", b"d", b"E", b"f"]
    with self.test_session():
      value = array_ops.identity(source).eval()
    self.assertAllEqual(source, value)

  def testIdentityShape(self):
    with self.test_session():
      shape = [2, 3]
      array_2x3 = [[1, 2, 3], [6, 5, 4]]
      tensor = constant_op.constant(array_2x3)
      self.assertEquals(shape, tensor.get_shape())
      self.assertEquals(shape, array_ops.identity(tensor).get_shape())
      self.assertEquals(shape, array_ops.identity(array_2x3).get_shape())
      self.assertEquals(shape,
                        array_ops.identity(np.array(array_2x3)).get_shape())

  def testRefIdentityShape(self):
    with self.test_session():
      shape = [2, 3]
      tensor = variables.Variable(
          constant_op.constant(
              [[1, 2, 3], [6, 5, 4]], dtype=dtypes.int32))
      self.assertEquals(shape, tensor.get_shape())
      self.assertEquals(shape, gen_array_ops.ref_identity(tensor).get_shape())


if __name__ == "__main__":
  test.main()
