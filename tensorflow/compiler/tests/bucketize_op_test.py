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
"""Tests for bucketize_op."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class BucketizationOpTest(xla_test.XLATestCase):

  def testInt(self):
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = math_ops._bucketize(p, boundaries=[0, 3, 8, 11])
      expected_out = [0, 1, 1, 2, 2, 3, 3, 4, 4]
      self.assertAllEqual(expected_out,
                          sess.run(op, {p: [-5, 0, 2, 3, 5, 8, 10, 11, 12]}))

  def testFloat(self):
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.float32)
      with self.test_scope():
        op = math_ops._bucketize(p, boundaries=[0., 3., 8., 11.])
      expected_out = [0, 1, 1, 2, 2, 3, 3, 4, 4]
      self.assertAllEqual(
          expected_out,
          sess.run(op, {p: [-5., 0., 2., 3., 5., 8., 10., 11., 12.]}))

  def test2DInput(self):
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.float32)
      with self.test_scope():
        op = math_ops._bucketize(p, boundaries=[0, 3, 8, 11])
      expected_out = [[0, 1, 1, 2, 2], [3, 3, 4, 4, 1]]
      self.assertAllEqual(
          expected_out, sess.run(op,
                                 {p: [[-5, 0, 2, 3, 5], [8, 10, 11, 12, 0]]}))

  @test_util.disable_mlir_bridge("Error handling")
  def testInvalidBoundariesOrder(self):
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = math_ops._bucketize(p, boundaries=[0, 8, 3, 11])
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Expected sorted boundaries"):
        sess.run(op, {p: [-5, 0]})

  def testBoundariesNotList(self):
    with self.session():
      with self.assertRaisesRegex(TypeError, "Expected list.*"):
        p = array_ops.placeholder(dtypes.int32)
        with self.test_scope():
          math_ops._bucketize(p, boundaries=0)


if __name__ == "__main__":
  test.main()
