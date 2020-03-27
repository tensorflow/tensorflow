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

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class ReduceTest(test_util.TensorFlowTestCase):

  def testReduceAllDims(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with test_util.device(use_gpu=True):
      y_tf = self.evaluate(math_ops.reduce_sum(x))
      self.assertEqual(y_tf, 21)

  def testReduceExplicitAxes(self):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with test_util.device(use_gpu=True):
      for axis in (0, -2, (0, 0), (0, -2)):
        self.assertAllEqual(self.evaluate(math_ops.reduce_sum(x, axis=axis)),
                            [5, 7, 9])
      for axis in (1, -1, (1, 1), (1, -1)):
        self.assertAllEqual(self.evaluate(math_ops.reduce_sum(x, axis=axis)),
                            [6, 15])
      for axis in (None, (0, 1), (-1, -2), (-2, -1, 0, 1)):
        self.assertEqual(self.evaluate(math_ops.reduce_sum(x, axis=axis)), 21)

  def testReduceInvalidAxis(self):
    if context.executing_eagerly():
      # The shape check is in run a graph construction time. In eager mode,
      # it misses the check, magically return result given wrong shape.
      return
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([[0], [1]])
    with self.assertRaisesRegexp(ValueError, "must be at most rank 1"):
      math_ops.reduce_sum(x, axis)

  def testReduceVar(self):
    x = np.array([[0, 0, 0], [0, 0, 0]], "float32")
    self.assertAllClose(self.evaluate(math_ops.reduce_variance(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_variance(x, axis=0)), [0, 0, 0])

    x = np.array([[0, 2, 1, 1], [1, 2, 0, 1]], "float32")
    self.assertAllClose(self.evaluate(math_ops.reduce_variance(x)), 0.5)

  def testReduceStd(self):
    x = np.array([[0, 0, 0], [0, 0, 0]], "float32")
    self.assertAllClose(self.evaluate(math_ops.reduce_std(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_std(x, axis=0)), [0, 0, 0])

    x = np.array([[1, 2, 1, 1], [1, 1, 0, 1]], "float32")
    self.assertAllClose(self.evaluate(math_ops.reduce_std(x)), 0.5)


@test_util.run_all_in_graph_and_eager_modes
class LogSumExpTest(test_util.TensorFlowTestCase):

  def testReduceLogSumExp(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf_np = math_ops.reduce_logsumexp(x_np)
        y_np = np.log(np.sum(np.exp(x_np)))
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf = math_ops.reduce_logsumexp(x_np, axis=[0])
        y_np = np.log(np.sum(np.exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = self.evaluate(y_tf)
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices2(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf = math_ops.reduce_logsumexp(x_np, axis=0)
        y_np = np.log(np.sum(np.exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = self.evaluate(y_tf)
        self.assertAllClose(y_tf_np, y_np)

  def testKeepDims(self):
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf_np = math_ops.reduce_logsumexp(x_np, keepdims=True)
        self.assertEqual(y_tf_np.shape.rank, x_np.ndim)
        y_np = np.log(np.sum(np.exp(x_np), keepdims=True))
        self.assertAllClose(y_tf_np, y_np)

  def testOverflow(self):
    x = [1000, 1001, 1002, 1003]
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegexp(RuntimeWarning,
                                   "overflow encountered in exp"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == np.inf:
          raise RuntimeWarning("overflow encountered in exp")

      with test_util.use_gpu():
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testUnderflow(self):
    x = [-1000, -1001, -1002, -1003]
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegexp(RuntimeWarning,
                                   "divide by zero encountered in log"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == -np.inf:
          raise RuntimeWarning("divide by zero encountered in log")

      with test_util.use_gpu():
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testInfinity(self):
    with test_util.use_gpu():
      res = math_ops.reduce_logsumexp(-np.inf)
      self.assertEqual(-np.inf, self.evaluate(res))


@test_util.run_all_in_graph_and_eager_modes
class RoundTest(test_util.TensorFlowTestCase):

  def testRounding(self):
    x = np.arange(-5.0, 5.0, .25)
    for dtype in [np.float32, np.double, np.int32]:
      x_np = np.array(x, dtype=dtype)
      with test_util.device(use_gpu=True):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf = math_ops.round(x_tf)
        y_tf_np = self.evaluate(y_tf)
        y_np = np.round(x_np)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)


@test_util.run_all_in_graph_and_eager_modes
class ModTest(test_util.TensorFlowTestCase):

  def testFloat(self):
    x = [0.5, 0.7, 0.3]
    for dtype in [np.float32, np.double]:
      # Test scalar and vector versions.
      for denom in [x[0], [x[0]] * 3]:
        x_np = np.array(x, dtype=dtype)
        with test_util.use_gpu():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = self.evaluate(y_tf)
          y_np = np.fmod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)

  def testFixed(self):
    x = [5, 10, 23]
    for dtype in [np.int32, np.int64]:
      # Test scalar and vector versions.
      for denom in [x[0], x]:
        x_np = np.array(x, dtype=dtype)
        with test_util.use_gpu():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = self.evaluate(y_tf)
          y_np = np.mod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np)


@test_util.run_all_in_graph_and_eager_modes
class SquaredDifferenceTest(test_util.TensorFlowTestCase):

  def testSquaredDifference(self):
    for dtype in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
      y = np.array([-3, -2, -1], dtype=dtype)
      z = (x - y) * (x - y)
      with test_util.device(use_gpu=True):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)

  def testComplexSquaredDifference(self):
    for dtype in [np.complex64, np.complex128]:
      x = np.array([[1 + 3j, 2 + 2j, 3 + 1j], [4 - 1j, 5 - 2j, 6 - 3j]],
                   dtype=dtype)
      y = np.array([-3 + 1j, -2 + 2j, -1 + 3j], dtype=dtype)
      z = np.conj(x - y) * (x - y)
      with test_util.device(use_gpu=False):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)


@test_util.run_all_in_graph_and_eager_modes
class ApproximateEqualTest(test_util.TensorFlowTestCase):

  def testApproximateEqual(self):
    for dtype in [np.float32, np.double]:
      x = dtype(1)
      y = dtype(1.00009)
      z = False
      with test_util.device(use_gpu=True):
        # Default tolerance is 0.00001
        z_tf = self.evaluate(math_ops.approximate_equal(x, y))
        self.assertAllEqual(z, z_tf)

    for dtype in [np.float32, np.double]:
      x = dtype(1)
      y = dtype(1.000009)
      z = True
      with test_util.device(use_gpu=True):
        # Default tolerance is 0.00001
        z_tf = self.evaluate(math_ops.approximate_equal(x, y))
        self.assertAllEqual(z, z_tf)

    for dtype in [np.float32, np.double]:
      x = np.array([[[[-1, 2.00009999], [-3, 4.01]]]], dtype=dtype)
      y = np.array([[[[-1.001, 2], [-3.00009, 4]]]], dtype=dtype)
      z = np.array([[[[False, True], [True, False]]]], dtype=np.bool)
      with test_util.device(use_gpu=True):
        z_tf = self.evaluate(math_ops.approximate_equal(x, y, tolerance=0.0001))
        self.assertAllEqual(z, z_tf)

  def testApproximateEqualShape(self):
    for dtype in [np.float32, np.double]:
      x = np.array([1, 2], dtype=dtype)
      y = np.array([[1, 2]], dtype=dtype)
      # The inputs 'x' and 'y' must have the same shape.
      with self.assertRaisesRegexp(
          (ValueError, errors.InvalidArgumentError),
          "Shapes must be equal rank|must be of the same shape"):
        math_ops.approximate_equal(x, y)


@test_util.run_all_in_graph_and_eager_modes
class ScalarMulTest(test_util.TensorFlowTestCase):

  def testAcceptsRefs(self):
    if context.executing_eagerly():
      var = resource_variable_ops.ResourceVariable(10, name="var")
    else:
      var = variables.Variable(10)
    result = math_ops.scalar_mul(3, var)
    init = variables.global_variables_initializer()
    with test_util.device(use_gpu=True):
      self.evaluate(init)
      self.assertEqual(30, self.evaluate(result))

  def testAcceptsConstant(self):
    const = constant_op.constant(10)
    result = math_ops.scalar_mul(3, const)
    with test_util.device(use_gpu=True):
      self.assertEqual(30, self.evaluate(result))

  def testAcceptsTensor(self):
    tensor = array_ops.ones([10, 10])
    result = math_ops.scalar_mul(3, tensor)
    expected = array_ops.ones([10, 10]) * 3

    with test_util.device(use_gpu=True):
      self.assertAllEqual(self.evaluate(expected), self.evaluate(result))

  def testAcceptsIndexedSlices(self):
    values = constant_op.constant([2, 3, 5, 7, 0, -1], shape=[3, 2])
    indices = constant_op.constant([0, 2, 5])
    x = math_ops.scalar_mul(-3, ops.IndexedSlices(values, indices))
    with test_util.device(use_gpu=True):
      self.assertAllEqual(self.evaluate(x.values),
                          [[-6, -9], [-15, -21], [0, 3]])
      self.assertAllEqual(self.evaluate(x.indices), [0, 2, 5])


@test_util.run_all_in_graph_and_eager_modes
class AddNTest(test_util.TensorFlowTestCase):

  def testPartials(self):
    """Test that previously revealed a bug in buffer forwarding for AddN."""
    partials = []
    for _ in range(98):
      partials.append(math_ops.add_n([constant_op.constant(1)]))
    partials.append(
        math_ops.add_n([constant_op.constant(1),
                        constant_op.constant(1)]))

    res = math_ops.add_n(partials) + constant_op.constant(0)
    with test_util.use_gpu():
      self.assertAllEqual(res, 100)

  def testFloat(self):
    np.random.seed(12345)
    for num_inputs in range(1, 10):
      x = [np.random.random((1, 2, 3, 4, 5)) - 0.5 for _ in range(num_inputs)]
      tf_x = ops.convert_n_to_tensor(x)
      with test_util.use_gpu():
        self.assertAllClose(sum(x), math_ops.add_n(tf_x))
        self.assertAllClose(x[0] * num_inputs,
                            math_ops.add_n([tf_x[0]] * num_inputs))

  def testInt(self):
    np.random.seed(54321)
    for num_inputs in range(1, 10):
      x = [
          np.random.randint(-128, 128, (5, 4, 3, 2, 1))
          for _ in range(num_inputs)
      ]
      tf_x = ops.convert_n_to_tensor(x)
      with test_util.use_gpu():
        self.assertAllEqual(sum(x), math_ops.add_n(tf_x))
        self.assertAllEqual(x[0] * num_inputs,
                            math_ops.add_n([tf_x[0]] * num_inputs))

  @test_util.deprecated_graph_mode_only
  def testGrad(self):
    np.random.seed(42)
    for num_inputs in range(1, 10):
      with test_util.use_gpu():
        input_vars = [
            variables.Variable(10.0 * np.random.random())
            for _ in range(0, num_inputs)
        ]
        addn = math_ops.add_n(input_vars)
        self.evaluate(variables.global_variables_initializer())
        add_n_grad = gradients.gradients(addn, input_vars)
        self.assertAllEqual(
            np.repeat(1.0, num_inputs),  # d/dx (x + y + ...) = 1
            [self.evaluate(g) for g in add_n_grad])

  def testIndexedSlices(self):
    slc = ops.IndexedSlices(
        array_ops.constant([1, 2], shape=[1, 2]), array_ops.constant([1]),
        array_ops.constant([2, 2]))
    slc_as_dense = np.array([[0, 0], [1, 2]])
    with test_util.use_gpu():
      # add_n currently always converts IndexedSlices to dense
      self.assertAllEqual(slc_as_dense, math_ops.add_n([slc]))
      self.assertAllEqual(2 * slc_as_dense, math_ops.add_n([slc, slc]))


@test_util.run_all_in_graph_and_eager_modes
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
    # TODO(aselle): Change test to use % after switch
    # tf_result = math_ops.floor_mod(nums, divs)
    tf_result = math_ops.floormod(nums, divs)
    np_result = nums % divs
    self.assertAllEqual(tf_result, np_result)

  def testFloorModFloat(self):
    nums, divs = self.floatTestData()
    tf_result = math_ops.floormod(nums, divs)
    np_result = nums % divs
    self.assertAllEqual(tf_result, np_result)
    # TODO(aselle): put this test in once % switched to floormod
    # tf2_result = (array_ops.constant(nums)
    #               % array_ops.constant(divs))
    # self.assertAllEqual(tf2_result, tf_result)

  def testTruncateModInt(self):
    nums, divs = self.intTestData()
    tf_result = math_ops.truncatemod(nums, divs)
    np_result = np.fmod(nums, divs)
    self.assertAllEqual(tf_result, np_result)

  def testTruncateModFloat(self):
    nums, divs = self.floatTestData()
    tf_result = math_ops.truncatemod(nums, divs)
    np_result = np.fmod(nums, divs)
    self.assertAllEqual(tf_result, np_result)

  def testDivideInt(self):
    nums, divs = self.intTestData()
    tf_result = math_ops.floor_div(nums, divs)
    np_result = nums // divs
    self.assertAllEqual(tf_result, np_result)
    # TODO(aselle): Put this test in once // is switched to floordiv
    # tf2_result = (array_ops.constant(nums)
    #               // array_ops.constant(divs))
    # self.assertAllEqual(tf2_result, tf_result)

  @test_util.deprecated_graph_mode_only
  def testDivideName(self):
    op = math_ops.divide(
        array_ops.constant(3), array_ops.constant(4), name="my_cool_divide")
    self.assertEqual(op.name, "my_cool_divide:0")

  def testRealDiv(self):
    nums, divs = self.floatTestData()
    tf_result = math_ops.realdiv(nums, divs)
    np_result = np.divide(nums, divs)
    self.assertAllClose(tf_result, np_result)

  def testDivideType(self):
    a = array_ops.constant([2], dtype=dtypes.int32)
    # Since __future__.division is effect, we should always upgrade to float64
    b = math_ops.divide(a, 1)
    self.assertEqual(b.dtype, dtypes.float64)
    self.assertEqual(2.0, self.evaluate(b))
    c = math_ops.divide(a, 4)
    self.assertEqual(c.dtype, dtypes.float64)
    self.assertEqual(0.5, self.evaluate(c))

  def testComplexDiv(self):
    foo = array_ops.constant([1. + 3.j])
    _ = math_ops.divide(foo, 1.)
    _ = math_ops.div(foo, 2.)

  @test_util.deprecated_graph_mode_only
  def testFloorDivGrad(self):
    a = variables.Variable(2.)
    b = variables.Variable(4.)
    self.evaluate(variables.global_variables_initializer())
    c_grad = gradients.gradients(math_ops.divide(a, b), [a, b])
    self.assertAllEqual([self.evaluate(x) for x in c_grad], [.25, -.125])
    c_grad = gradients.gradients(math_ops.div(a, b), [a, b])
    self.assertAllEqual([self.evaluate(x) for x in c_grad], [.25, -.125])
    c_grad = gradients.gradients(math_ops.floordiv(a, b), [a, b])
    self.assertAllEqual(
        [None if x is None else self.evaluate(x) for x in c_grad], [None, None])

  def testConsistent(self):
    nums, divs = self.intTestData()
    tf_result = (
        math_ops.floor_div(nums, divs) * divs + math_ops.floormod(nums, divs))
    tf_nums = array_ops.constant(nums)
    tf_divs = array_ops.constant(divs)
    tf2_result = (tf_nums // tf_divs * tf_divs + tf_nums % tf_divs)
    np_result = (nums // divs) * divs + (nums % divs)
    # Consistent with numpy
    self.assertAllEqual(tf_result, np_result)
    # Consistent with two forms of divide
    self.assertAllEqual(tf_result, tf2_result)
    # consistency for truncation form
    tf3_result = (
        math_ops.truncatediv(nums, divs) * divs +
        math_ops.truncatemod(nums, divs))
    expanded_nums = np.reshape(
        np.tile(nums, divs.shape[1]), (nums.shape[0], divs.shape[1]))
    # Consistent with desire to get numerator
    self.assertAllEqual(tf3_result, expanded_nums)
    # Consistent with desire to get numerator
    self.assertAllEqual(tf_result, expanded_nums)


@test_util.run_all_in_graph_and_eager_modes
class DivNoNanTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    for dtype in [np.float32, np.float64]:
      nums = np.arange(-10, 10, .25, dtype=dtype).reshape(80, 1)
      divs = np.arange(-3, 3, .25, dtype=dtype).reshape(1, 24)

      np_result = np.true_divide(nums, divs)
      np_result[:, divs[0] == 0] = 0

      with test_util.use_gpu():
        tf_result = math_ops.div_no_nan(nums, divs)
        self.assertAllClose(tf_result, np_result)


@test_util.run_all_in_graph_and_eager_modes
class MultiplyNoNanTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    for dtype in [np.float32, np.float64]:
      values = [0, 1, np.nan, np.inf, np.NINF]
      x = constant_op.constant(values, dtype=dtype)
      zeros = constant_op.constant(np.zeros((5,)), dtype=dtype)
      ones = constant_op.constant(np.ones((5,)), dtype=dtype)
      with test_util.use_gpu():
        tf_result_zeros = math_ops.multiply_no_nan(x, zeros)
        self.assertAllEqual(tf_result_zeros, zeros)
        tf_result_ones = math_ops.multiply_no_nan(x, ones)
        self.assertAllEqual(tf_result_ones, x)


@test_util.run_all_in_graph_and_eager_modes
class XlogyTest(test_util.TensorFlowTestCase):

  def testXlogyNoZero(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.1, 0.2, 3.5], [-2., -5., 30.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [3.1, 4., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlogy = self.evaluate(math_ops.xlogy(x, y))
        xtimeslogy = self.evaluate(x * math_ops.log(y))
        self.assertAllClose(xlogy, xtimeslogy)

  def testXlogyWithZero(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(np.zeros((2, 3)), dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlogy_tf_np = self.evaluate(math_ops.xlogy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y))
        self.assertAllClose(xlogy_tf_np, zeros_np)

  def testXlogyWithZeroBroadcast(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.], [1.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlogy_tf_np = self.evaluate(math_ops.xlogy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y[0]))
        xtimes_logy = self.evaluate(math_ops.log(y[1]))
        self.assertAllClose(zeros_np, xlogy_tf_np[0])
        self.assertAllClose(xtimes_logy, xlogy_tf_np[1])


@test_util.run_all_in_graph_and_eager_modes
class Xlog1pyTest(test_util.TensorFlowTestCase):

  def testXlog1pyNoNeg1(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.1, 0.2, 3.5], [-2., -5., 30.]], dtype=dtype)
      y = constant_op.constant([[-0.1, -0.2, 3.5], [3.1, -0.9, 2.]],
                               dtype=dtype)
      with test_util.use_gpu():
        xlog1py = self.evaluate(math_ops.xlog1py(x, y))
        xtimeslog1py = self.evaluate(x * math_ops.log1p(y))
        self.assertAllClose(xlog1py, xtimeslog1py)

  def testXlog1pyWithNegOne(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(np.zeros((2, 3)), dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [-1., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlog1py_tf_np = self.evaluate(math_ops.xlog1py(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y))
        self.assertAllClose(xlog1py_tf_np, zeros_np)

  def testXlog1pyWithZeroBroadcast(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.], [1.]], dtype=dtype)
      y = constant_op.constant([[-0.1, -0.2, -1.], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlog1py_tf_np = self.evaluate(math_ops.xlog1py(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y[0]))
        xtimes_log1py = self.evaluate(math_ops.log1p(y[1]))
        self.assertAllClose(zeros_np, xlog1py_tf_np[0])
        self.assertAllClose(xtimes_log1py, xlog1py_tf_np[1])


@test_util.run_all_in_graph_and_eager_modes
class XdivyTest(test_util.TensorFlowTestCase):

  def testXdivyNoZero(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.1, 0.2, 3.5], [-2., -5., 30.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [3.1, 4., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xdivy = self.evaluate(math_ops.xdivy(x, y))
        x_over_y = self.evaluate(x / y)
        self.assertAllClose(xdivy, x_over_y)

  def testXdivyWithZero(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(np.zeros((2, 3)), dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xdivy_tf_np = self.evaluate(math_ops.xdivy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y))
        self.assertAllClose(xdivy_tf_np, zeros_np)

  def testXdivyWithZeroBroadcast(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.], [1.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xdivy_tf_np = self.evaluate(math_ops.xdivy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y[0]))
        x_over_y = self.evaluate(1 / y[1])
        self.assertAllClose(zeros_np, xdivy_tf_np[0])
        self.assertAllClose(x_over_y, xdivy_tf_np[1])


@test_util.run_all_in_graph_and_eager_modes
class NextAfterTest(test_util.TensorFlowTestCase):

  # Basic NextAfter tests that replicate numpy nextafter tests.
  def testBasic(self):

    for dtype in [dtypes.float32, dtypes.float64]:
      one = constant_op.constant([1], dtype=dtype)
      two = constant_op.constant([2], dtype=dtype)
      zero = constant_op.constant([0], dtype=dtype)
      nan = constant_op.constant([np.nan], dtype=dtype)

      eps = constant_op.constant([np.finfo(dtype.as_numpy_dtype).eps],
                                 dtype=dtype)

      self.assertAllEqual(math_ops.nextafter(one, two) - one, eps)
      self.assertAllLess(math_ops.nextafter(one, zero) - one, 0)
      self.assertAllEqual(
          math_ops.is_nan(math_ops.nextafter(nan, one)), [True])
      self.assertAllEqual(
          math_ops.is_nan(math_ops.nextafter(one, nan)), [True])
      self.assertAllEqual(math_ops.nextafter(one, one), one)

  def testBroadcasting(self):

    for dtype in [dtypes.float32, dtypes.float64]:
      one = constant_op.constant([1, 1], dtype=dtype)
      two = constant_op.constant([2], dtype=dtype)

      eps = np.finfo(dtype.as_numpy_dtype).eps

      eps_const = constant_op.constant([eps, eps], dtype=dtype)

      self.assertAllEqual(math_ops.nextafter(one, two) - one, eps_const)


@test_util.run_all_in_graph_and_eager_modes
class BinaryOpsTest(test_util.TensorFlowTestCase):

  def testErrorReceivedIfDtypeMismatchFromOp(self):
    if context.executing_eagerly():
      error = errors_impl.InvalidArgumentError
      error_message = (
          r"cannot compute Add(V2)? as input #1\(zero-based\) was expected to "
          r"be a int32 tensor but is a float tensor \[Op:Add(V2)?\]")
    else:
      error = TypeError
      error_message = (
          "Input 'y' of 'Add(V2)?' Op has type float32 that does not "
          "match type int32 of argument 'x'.")
    with self.assertRaisesRegexp(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + 1.0
      self.evaluate(a)


class SignTest(test_util.TensorFlowTestCase):

  def test_complex_sign_gradient(self):
    with context.eager_mode():
      x = math_ops.complex(1., 1.)
      with backprop.GradientTape() as t:
        t.watch(x)
        y = math_ops.sign(x)
      self.assertAllClose(
          t.gradient(y, x), math_ops.complex(0.353553, -0.353553))


@test_util.run_all_in_graph_and_eager_modes
class ReciprocalNoNanTest(test_util.TensorFlowTestCase):

  allowed_dtypes = [
      dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64,
      dtypes.complex128
  ]

  def testBasic(self):
    for dtype in self.allowed_dtypes:
      x = constant_op.constant([1.0, 2.0, 0.0, 4.0], dtype=dtype)

      y = math_ops.reciprocal_no_nan(x)

      target = constant_op.constant([1.0, 0.5, 0.0, 0.25], dtype=dtype)

      self.assertAllEqual(y, target)
      self.assertEqual(y.dtype.base_dtype, target.dtype.base_dtype)

  def testInverse(self):
    for dtype in self.allowed_dtypes:
      x = np.random.choice([0, 1, 2, 4, 5], size=(5, 5, 5))
      x = constant_op.constant(x, dtype=dtype)

      y = math_ops.reciprocal_no_nan(math_ops.reciprocal_no_nan(x))

      self.assertAllClose(y, x)
      self.assertEqual(y.dtype.base_dtype, x.dtype.base_dtype)


@test_util.run_all_in_graph_and_eager_modes
class EqualityTest(test_util.TensorFlowTestCase):

  def testEqualityNone(self):
    x = constant_op.constant([1.0, 2.0, 0.0, 4.0], dtype=dtypes.float32)
    self.assertNotEqual(x, None)
    self.assertNotEqual(None, x)
    self.assertFalse(math_ops.tensor_equals(x, None))
    self.assertTrue(math_ops.tensor_not_equals(x, None))


@test_util.run_all_in_graph_and_eager_modes
class RangeTest(test_util.TensorFlowTestCase):

  def testConvertToTensorRange(self):
    values = range(5)
    tensor = ops.convert_to_tensor(values)
    self.assertAllEqual((5,), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))


if __name__ == "__main__":
  googletest.main()
