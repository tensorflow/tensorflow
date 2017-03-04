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

"""Tests for tensorflow.ops.math_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

exp = np.exp
log = np.log


class ReduceTest(test_util.TensorFlowTestCase):

  def testReduceAllDims(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with self.test_session(use_gpu=True):
      y_tf = math_ops.reduce_sum(x).eval()
      self.assertEqual(y_tf, 21)

  def testReduceExplicitAxes(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with self.test_session(use_gpu=True):
      for axis in (0, -2, (0, 0), (0, -2)):
        self.assertAllEqual(math_ops.reduce_sum(x, axis=axis).eval(), [5, 7, 9])
      for axis in (1, -1, (1, 1), (1, -1)):
        self.assertAllEqual(math_ops.reduce_sum(x, axis=axis).eval(), [6, 15])
      for axis in (None, (0, 1), (-1, -2), (-2, -1, 0, 1)):
        self.assertEqual(math_ops.reduce_sum(x, axis=axis).eval(), 21)

  def testReduceInvalidAxis(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([[0], [1]])
    with self.assertRaisesRegexp(ValueError, "must be at most rank 1"):
      math_ops.reduce_sum(x, axis)


class LogSumExpTest(test_util.TensorFlowTestCase):

  def testReduceLogSumExp(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with self.test_session(use_gpu=True):
        y_tf_np = math_ops.reduce_logsumexp(x_np).eval()
        y_np = log(np.sum(exp(x_np)))
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with self.test_session(use_gpu=True):
        y_tf = math_ops.reduce_logsumexp(x_np, reduction_indices=[0])
        y_np = log(np.sum(exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = y_tf.eval()
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices2(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with self.test_session(use_gpu=True):
        y_tf = math_ops.reduce_logsumexp(x_np, reduction_indices=0)
        y_np = log(np.sum(exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = y_tf.eval()
        self.assertAllClose(y_tf_np, y_np)

  def testKeepDims(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with self.test_session(use_gpu=True):
        y_tf_np = math_ops.reduce_logsumexp(x_np, keep_dims=True).eval()
        self.assertEqual(y_tf_np.ndim, x_np.ndim)
        y_np = log(np.sum(exp(x_np), keepdims=True))
        self.assertAllClose(y_tf_np, y_np)

  def testOverflow(self):
    x = [1000, 1001, 1002, 1003]
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegexp(RuntimeWarning,
                                   "overflow encountered in exp"):
        out = log(np.sum(exp(x_np)))
        if out == np.inf:
          raise RuntimeWarning("overflow encountered in exp")

      with self.test_session(use_gpu=True):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf).eval()
        y_np = log(np.sum(exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testUnderflow(self):
    x = [-1000, -1001, -1002, -1003]
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegexp(RuntimeWarning,
                                   "divide by zero encountered in log"):
        out = log(np.sum(exp(x_np)))
        if out == -np.inf:
          raise RuntimeWarning("divide by zero encountered in log")

      with self.test_session(use_gpu=True):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf).eval()
        y_np = log(np.sum(exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)


class RoundTest(test_util.TensorFlowTestCase):

  def testRounding(self):
    x = [0.49, 0.7, -0.3, -0.8]
    # TODO(nolivia): Remove this when RoundOp is forwards compatible
    # x = np.arange(-5.0, 5.0, .25)
    for dtype in [np.float32, np.double, np.int32]:
      x_np = np.array(x, dtype=dtype)
      with self.test_session(use_gpu=True):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf = math_ops.round(x_tf)
        y_tf_np = y_tf.eval()
        y_np = np.round(x_np)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)


class ModTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    x = [0.5, 0.7, 0.3]
    for dtype in [np.float32, np.double]:
      # Test scalar and vector versions.
      for denom in [x[0], [x[0]] * 3]:
        x_np = np.array(x, dtype=dtype)
        with self.test_session(use_gpu=True):
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = y_tf.eval()
          y_np = np.fmod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)

  def testFixed(self):
    x = [5, 10, 23]
    for dtype in [np.int32, np.int64]:
      # Test scalar and vector versions.
      for denom in [x[0], x]:
        x_np = np.array(x, dtype=dtype)
        with self.test_session(use_gpu=True):
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = y_tf.eval()
          y_np = np.mod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np)


class SquaredDifferenceTest(test_util.TensorFlowTestCase):

  def testSquaredDifference(self):
    for dtype in [np.int32, np.float16]:
      x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
      y = np.array([-3, -2, -1], dtype=dtype)
      z = (x - y)*(x - y)
      with self.test_session(use_gpu=True):
        z_tf = math_ops.squared_difference(x, y).eval()
        self.assertAllClose(z, z_tf)


class ScalarMulTest(test_util.TensorFlowTestCase):

  def testAcceptsRefs(self):
    var = variables.Variable(10)
    result = math_ops.scalar_mul(3, var)
    init = variables.global_variables_initializer()
    with self.test_session(use_gpu=True) as sess:
      sess.run(init)
      self.assertEqual(30, result.eval())

  def testAcceptsConstant(self):
    const = constant_op.constant(10)
    result = math_ops.scalar_mul(3, const)
    with self.test_session(use_gpu=True):
      self.assertEqual(30, result.eval())

  def testAcceptsTensor(self):
    tensor = array_ops.ones([10, 10])
    result = math_ops.scalar_mul(3, tensor)
    expected = array_ops.ones([10, 10]) * 3

    with self.test_session(use_gpu=True):
      self.assertAllEqual(expected.eval(), result.eval())

  def testAcceptsIndexedSlices(self):
    values = constant_op.constant([2, 3, 5, 7, 0, -1], shape=[3, 2])
    indices = constant_op.constant([0, 2, 5])
    x = math_ops.scalar_mul(-3, ops.IndexedSlices(values, indices))
    with self.test_session(use_gpu=True):
      self.assertAllEqual(x.values.eval(), [[-6, -9], [-15, -21], [0, 3]])
      self.assertAllEqual(x.indices.eval(), [0, 2, 5])


class AccumulateNTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    np.random.seed(12345)
    x = [np.random.random((1, 2, 3, 4, 5)) - 0.5 for _ in range(5)]
    tf_x = ops.convert_n_to_tensor(x)
    with self.test_session(use_gpu=True):
      self.assertAllClose(sum(x), math_ops.accumulate_n(tf_x).eval())
      self.assertAllClose(x[0] * 5, math_ops.accumulate_n([tf_x[0]] * 5).eval())

  def testInt(self):
    np.random.seed(54321)
    x = [np.random.randint(-128, 128, (5, 4, 3, 2, 1)) for _ in range(6)]
    tf_x = ops.convert_n_to_tensor(x)
    with self.test_session(use_gpu=True):
      self.assertAllEqual(sum(x), math_ops.accumulate_n(tf_x).eval())
      self.assertAllEqual(x[0] * 6, math_ops.accumulate_n([tf_x[0]] * 6).eval())


class DivAndModTest(test_util.TensorFlowTestCase):
  # TODO(aselle): Test more types before exposing new division operators.

  def intTestData(self):
    nums = np.arange(-10, 10, 1).reshape(20, 1)
    divs = np.arange(-3, 4, 2).reshape(1, 4)
    return nums, divs

  def floatTestData(self):
    nums = np.arange(-10, 10, .25).reshape(80, 1)
    divs = np.arange(-3, 0, .25).reshape(1, 12)
    return nums, divs

  def testFloorModInt(self):
    nums, divs = self.intTestData()
    with self.test_session():
      # TODO(aselle): Change test to use % after switch
      # tf_result = math_ops.floor_mod(nums, divs).eval()
      tf_result = math_ops.floormod(nums, divs).eval()
      np_result = nums % divs
      self.assertAllEqual(tf_result, np_result)

  def testFloorModFloat(self):
    nums, divs = self.floatTestData()
    with self.test_session():
      tf_result = math_ops.floormod(nums, divs).eval()
      np_result = nums % divs
      self.assertAllEqual(tf_result, np_result)
      # TODO(aselle): put this test in once % switched to floormod
      # tf2_result = (array_ops.constant(nums)
      #               % array_ops.constant(divs)).eval()
      # self.assertAllEqual(tf2_result, tf_result)

  def testTruncateModInt(self):
    nums, divs = self.intTestData()
    with self.test_session():
      tf_result = math_ops.truncatemod(nums, divs).eval()
      np_result = np.fmod(nums, divs)
      self.assertAllEqual(tf_result, np_result)

  def testTruncateModFloat(self):
    nums, divs = self.floatTestData()
    with self.test_session():
      tf_result = math_ops.truncatemod(nums, divs).eval()
      np_result = np.fmod(nums, divs)
      self.assertAllEqual(tf_result, np_result)

  def testDivideInt(self):
    nums, divs = self.intTestData()
    with self.test_session():
      tf_result = math_ops.floor_div(nums, divs).eval()
      np_result = nums // divs
      self.assertAllEqual(tf_result, np_result)
      # TODO(aselle): Put this test in once // is switched to floordiv
      # tf2_result = (array_ops.constant(nums)
      #               // array_ops.constant(divs)).eval()
      # self.assertAllEqual(tf2_result, tf_result)

  def testDivideName(self):
    with self.test_session():
      op = math_ops.divide(array_ops.constant(3),
                           array_ops.constant(4), name="my_cool_divide")
      self.assertEqual(op.name, "my_cool_divide:0")

  def testRealDiv(self):
    nums, divs = self.floatTestData()
    with self.test_session():
      tf_result = math_ops.realdiv(nums, divs).eval()
      np_result = np.divide(nums, divs)
      self.assertAllEqual(tf_result, np_result)

  def testComplexDiv(self):
    foo = array_ops.constant([1.+3.j])
    with self.test_session():
      _ = math_ops.divide(foo, 1.).eval()
      _ = math_ops.div(foo, 2.).eval()

  def testFloorDivGrad(self):
    with self.test_session():
      a = variables.Variable(2.)
      b = variables.Variable(4.)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        c_grad = gradients.gradients(math_ops.divide(a, b), [a, b])
        self.assertAllEqual([x.eval() for x in c_grad], [.25, -.125])
        c_grad = gradients.gradients(math_ops.div(a, b), [a, b])
        self.assertAllEqual([x.eval() for x in c_grad], [.25, -.125])
        c_grad = gradients.gradients(math_ops.floordiv(a, b), [a, b])
        self.assertAllEqual([None if x is None else x.eval() for x in c_grad],
                            [None, None])

  def testConsistent(self):
    nums, divs = self.intTestData()
    with self.test_session():
      tf_result = (
          math_ops.floor_div(nums, divs) * divs + math_ops.floormod(nums, divs)
      ).eval()
      tf_nums = array_ops.constant(nums)
      tf_divs = array_ops.constant(divs)
      tf2_result = (tf_nums // tf_divs * tf_divs + tf_nums % tf_divs).eval()
      np_result = (nums // divs) * divs + (nums % divs)
      # consistentcy with numpy
      self.assertAllEqual(tf_result, np_result)
      # consistentcy with two forms of divide
      self.assertAllEqual(tf_result, tf2_result)
      # consistency for truncation form
      tf3_result = (
          math_ops.truncatediv(nums, divs) * divs
          + math_ops.truncatemod(nums, divs)
      ).eval()
      expanded_nums = np.reshape(np.tile(nums, divs.shape[1]),
                                 (nums.shape[0], divs.shape[1]))
      # Consistent with desire to get numerator
      self.assertAllEqual(tf3_result, expanded_nums)
      # Consistent with desire to get numerator
      self.assertAllEqual(tf_result, expanded_nums)


if __name__ == "__main__":
  googletest.main()
