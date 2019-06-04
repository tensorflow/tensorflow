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
"""Tests for broadcast_to ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test as test_lib


class BroadcastToTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testBroadcastToBasic(self):
    for dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]:
      with self.session(use_gpu=True):
        x = np.array([1, 2, 3], dtype=dtype)
        v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToString(self):
    with self.session(use_gpu=True):
      x = np.array([b"1", b"2", b"3"])
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToBool(self):
    with self.session(use_gpu=True):
      x = np.array([True, False, True], dtype=np.bool)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShape(self):
    for input_dim in range(1, 6):
      for output_dim in range(input_dim, 6):
        with self.cached_session(use_gpu=True):
          input_shape = [2] * input_dim
          output_shape = [2] * output_dim
          x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
          v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
          v_np = np.broadcast_to(x, output_shape)
          self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeInnerDim(self):
    input_shape = [2, 1, 3]
    output_shape = [2, 5, 3]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeLargerDim(self):
    input_shape = [2, 1, 3, 2, 2, 2]
    output_shape = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 15, 3, 2, 2, 2]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeLargerDim2(self):
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToScalar(self):
    with self.session(use_gpu=True):
      x = np.array(1, dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastScalarToNonScalar(self):
    with self.session(use_gpu=True):
      x = np.array(1.0, dtype=np.float)
      v_tf = array_ops.broadcast_to(constant_op.constant(1.0), [2, 3, 4,
                                                                1, 1, 1])
      v_np = np.broadcast_to(x, [2, 3, 4, 1, 1, 1])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeTypeAndInference(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.cached_session(use_gpu=True):
        x = np.array([1, 2, 3])
        v_tf = array_ops.broadcast_to(
            constant_op.constant(x),
            constant_op.constant([3, 3], dtype=dtype))
        shape = v_tf.get_shape().as_list()
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf.eval(), v_np)
        # check shape inference when shape input is constant
        self.assertAllEqual(shape, v_np.shape)

  def testBroadcastToBadOutputShape(self):
    with context.eager_mode():
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Unable to broadcast tensor of shape"):
        self.evaluate(
            array_ops.broadcast_to(
                constant_op.constant([0, 1]), constant_op.constant([2, 1])))

  @test_util.run_deprecated_v1
  def testGradientForScalar(self):
    x = constant_op.constant(1, dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [2, 4, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(), out,
                                                    out.get_shape())
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithSameRank(self):
    x = constant_op.constant(np.reshape(np.arange(6), (2, 1, 3)),
                             dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [2, 5, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(),
                                                    out, out.get_shape())
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithIncreasingRank(self):
    x = constant_op.constant([[1], [2]],
                             dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [5, 2, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(),
                                                    out, out.get_shape())
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithBroadcastAllDimensions(self):
    x = constant_op.constant([1], dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [5, 2, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(),
                                                    out, out.get_shape())
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithLargeDim(self):
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    x = constant_op.constant(np.array(np.random.randn(*input_shape),
                                      dtype=np.float32))
    v = array_ops.broadcast_to(x, output_shape)
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(x, x.get_shape(),
                                                    out, out.get_shape())
    self.assertLess(err, 1e-4)

if __name__ == "__main__":
  test_lib.main()
