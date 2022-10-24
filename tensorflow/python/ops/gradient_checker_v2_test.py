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
"""Tests for compute_gradient."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import \
gradient_checker_v2 as gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
# needs this to register gradient for SoftmaxCrossEntropyWithLogits:
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _random_complex(shape, dtype):
  data = np.random.random_sample(shape).astype(dtype.as_numpy_dtype)
  if dtype.is_complex:
    data.imag = np.random.random_sample(shape)
  return data


@test_util.run_all_in_graph_and_eager_modes
class GradientCheckerTest(test.TestCase):

  def testSparseTensorReshape(self):
    x = constant_op.constant(2.0, shape=(2,))

    def sparse_tensor_reshape(values):
      sparse = sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]], values=values, dense_shape=[3, 4])
      sparse = sparse_ops.sparse_reshape(sparse, shape=(12,))
      return sparse.values

    error = gradient_checker.max_error(
        *gradient_checker.compute_gradient(sparse_tensor_reshape, [x]))

    self.assertLess(error, 1e-4)

  def testWithStaticShape(self):
    size = (2, 3)
    constant = constant_op.constant(2.0, shape=size, name="const")

    def add_constant_with_static_shape_check(x):
      self.assertAllEqual(x.shape.as_list(), constant.shape.as_list())
      return x + constant

    x = constant_op.constant(3.0, shape=size, name="x")

    error = gradient_checker.max_error(*gradient_checker.compute_gradient(
        add_constant_with_static_shape_check, [x]))

    self.assertLess(error, 1e-4)

  def testWithArgumentsAsTuple(self):
    size = (2, 3)
    x1 = constant_op.constant(2.0, shape=size, name="x1")
    x2 = constant_op.constant(3.0, shape=size, name="x2")

    error = gradient_checker.max_error(*gradient_checker.compute_gradient(
        lambda x1: math_ops.add(x1, x2), (x1,)))

    tf_logging.info("x1 error = %f", error)
    self.assertLess(error, 1e-4)

  def testAddSimple(self):
    size = (2, 3)
    x1 = constant_op.constant(2.0, shape=size, name="x1")
    x2 = constant_op.constant(3.0, shape=size, name="x2")
    error = gradient_checker.max_error(*gradient_checker.compute_gradient(
        lambda x1: math_ops.add(x1, x2), [x1]))
    tf_logging.info("x1 error = %f", error)
    self.assertLess(error, 1e-4)

  def testBfloat16(self):
    x1 = constant_op.constant(2.0, dtype="bfloat16")
    x2 = constant_op.constant(3.0, dtype="bfloat16")
    # bfloat16 is very imprecise, so we use very large delta and error bar here.
    error = gradient_checker.max_error(*gradient_checker.compute_gradient(
        lambda x1: math_ops.add(x1, x2), [x1], delta=0.1))
    tf_logging.info("x1 error = %f", error)
    self.assertLess(error, 0.07)

  def testAddCustomized(self):
    size = (2, 3)
    x1 = constant_op.constant(2.0, shape=size, dtype=dtypes.float64, name="x1")
    x2 = np.asarray(np.arange(6, dtype=np.float64).reshape(2, 3))
    # checkint gradients for x2 using a special delta
    error = gradient_checker.max_error(*gradient_checker.compute_gradient(
        lambda x2: math_ops.add(x1, x2), [x2], delta=1e-2))
    tf_logging.info("x2 error = %f", error)
    self.assertLess(error, 1e-10)

  def testGather(self):

    def f(params):
      index_values = [1, 3]
      indices = constant_op.constant(index_values, name="i")
      return array_ops.gather(params, indices, name="y")

    p_shape = (4, 2)
    p_size = 8
    params = constant_op.constant(
        np.arange(p_size).astype(np.float64), shape=p_shape, name="p")
    error = gradient_checker.max_error(
        *gradient_checker.compute_gradient(f, [params]))
    tf_logging.info("gather error = %f", error)
    self.assertLess(error, 1e-4)

  def testNestedGather(self):

    def f(params):
      index_values = [1, 3, 5, 6]
      indices = constant_op.constant(index_values, name="i")
      y = array_ops.gather(params, indices, name="y")
      index_values2 = [0, 2]
      indices2 = constant_op.constant(index_values2, name="i2")
      return array_ops.gather(y, indices2, name="y2")

    p_shape = (8, 2)
    p_size = 16
    params = constant_op.constant(
        np.arange(p_size).astype(np.float64), shape=p_shape, name="p")
    error = gradient_checker.max_error(
        *gradient_checker.compute_gradient(f, [params]))
    tf_logging.info("nested gather error = %f", error)
    self.assertLess(error, 1e-4)

  def testComplexMul(self):
    c = constant_op.constant(5 + 7j, dtype=dtypes.complex64)

    def f(x):
      return c * x

    x_shape = c.shape
    x_dtype = c.dtype
    x = constant_op.constant(_random_complex(x_shape, x_dtype))
    analytical, numerical = gradient_checker.compute_gradient(f, [x])
    correct = np.array([[5, -7], [7, 5]])
    self.assertAllEqual(correct, analytical[0])
    self.assertAllClose(correct, numerical[0], rtol=1e-4)
    x = constant_op.constant(_random_complex(x_shape, x_dtype))
    self.assertLess(
        gradient_checker.max_error(*gradient_checker.compute_gradient(f, [x])),
        3e-4)

  def testComplexConj(self):

    def f(x):
      return math_ops.conj(x)

    x_shape = ()
    x_dtype = dtypes.complex64
    x = constant_op.constant(_random_complex(x_shape, x_dtype))
    analytical, numerical = gradient_checker.compute_gradient(f, [x])
    correct = np.array([[1, 0], [0, -1]])
    self.assertAllEqual(correct, analytical[0])
    self.assertAllClose(correct, numerical[0], rtol=2e-5)
    x = constant_op.constant(_random_complex(x_shape, x_dtype))
    self.assertLess(
        gradient_checker.max_error(*gradient_checker.compute_gradient(f, [x])),
        2e-5)

  def testEmptySucceeds(self):

    def f(x):
      return array_ops.identity(x)

    x = constant_op.constant(
        np.random.random_sample((0, 3)), dtype=dtypes.float32)
    for grad in gradient_checker.compute_gradient(f, [x]):
      self.assertEqual(grad[0].shape, (0, 0))
    error = gradient_checker.max_error(
        *gradient_checker.compute_gradient(f, [x]))
    self.assertEqual(error, 0)

  def testEmptyMatMul(self):

    def f(x, y):
      return math_ops.matmul(x, y)

    x = constant_op.constant(
        np.random.random_sample((0, 3)), dtype=dtypes.float32)
    y = constant_op.constant(
        np.random.random_sample((3, 4)), dtype=dtypes.float32)
    for grad in gradient_checker.compute_gradient(f, [x, y]):
      self.assertEqual(grad[0].shape, (0, 0))
      self.assertEqual(grad[1].shape, (0, 12))
    error = gradient_checker.max_error(
        *gradient_checker.compute_gradient(f, [x, y]))
    self.assertEqual(error, 0)

  def testEmptyFails(self):

    @custom_gradient.custom_gradient
    def id_bad_grad(x):
      y = array_ops.identity(x)

      def grad_fn(dy):
        # dx = constant_op.constant(np.zeros((1, 4)), dtype=dtypes.float32)
        dx = array_ops.transpose(dy)
        return dx

      return y, grad_fn

    def f(x):
      return id_bad_grad(x)

    x = constant_op.constant(
        np.random.random_sample((0, 3)), dtype=dtypes.float32)
    bad = r"Empty gradient has wrong shape: expected \(0, 3\), got \(3, 0\)"
    with self.assertRaisesRegex(ValueError, bad):
      gradient_checker.compute_gradient(f, [x])

  def testNaNGradFails(self):

    @custom_gradient.custom_gradient
    def id_nan_grad(x):
      y = array_ops.identity(x)

      def grad_fn(dy):
        dx = np.nan * dy
        # dx = dy
        return dx

      return y, grad_fn

    def f(x):
      return id_nan_grad(x)

    x = constant_op.constant(
        np.random.random_sample((1, 1)), dtype=dtypes.float32)
    error = gradient_checker.max_error(
        *gradient_checker.compute_gradient(f, [x]))
    # Typical test would assert error < max_err, so assert this test would
    # raise AssertionError, since NaN is not < 1.0.
    with self.assertRaisesRegex(AssertionError, "nan not less than 1.0"):
      self.assertLess(error, 1.0)

  def testGradGrad(self):

    def f(x):
      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = math_ops.square(x)
        z = math_ops.square(y)
      return tape.gradient(z, x)

    analytical, numerical = gradient_checker.compute_gradient(f, [2.0])
    self.assertAllEqual([[[48.]]], analytical)
    self.assertAllClose([[[48.]]], numerical, rtol=1e-4)


@test_util.run_all_in_graph_and_eager_modes
class MiniMNISTTest(test.TestCase):

  # Gradient checker for MNIST.
  def _BuildAndTestMiniMNIST(self, param_index, tag):
    # Fix seed to avoid occasional flakiness
    np.random.seed(6)

    # Hyperparameters
    batch = 3
    inputs = 16
    features = 32
    classes = 10

    # Define the parameters
    inp_data = np.random.random_sample(inputs * batch)
    hidden_weight_data = np.random.randn(inputs * features) / np.sqrt(inputs)
    hidden_bias_data = np.random.random_sample(features)
    sm_weight_data = np.random.randn(features * classes) / np.sqrt(features)
    sm_bias_data = np.random.random_sample(classes)

    # special care for labels since they need to be normalized per batch
    label_data = np.random.random(batch * classes).reshape((batch, classes))
    s = label_data.sum(axis=1)
    label_data /= s[:, None]

    # We treat the inputs as "parameters" here
    inp = constant_op.constant(
        inp_data.tolist(),
        shape=[batch, inputs],
        dtype=dtypes.float64,
        name="inp")
    hidden_weight = constant_op.constant(
        hidden_weight_data.tolist(),
        shape=[inputs, features],
        dtype=dtypes.float64,
        name="hidden_weight")
    hidden_bias = constant_op.constant(
        hidden_bias_data.tolist(),
        shape=[features],
        dtype=dtypes.float64,
        name="hidden_bias")
    softmax_weight = constant_op.constant(
        sm_weight_data.tolist(),
        shape=[features, classes],
        dtype=dtypes.float64,
        name="softmax_weight")
    softmax_bias = constant_op.constant(
        sm_bias_data.tolist(),
        shape=[classes],
        dtype=dtypes.float64,
        name="softmax_bias")

    # List all the parameter so that we can test them one at a time
    all_params = [inp, hidden_weight, hidden_bias, softmax_weight, softmax_bias]

    # Now, Building MNIST
    def f(inp, hidden_weight, hidden_bias, softmax_weight, softmax_bias):
      features = nn_ops.relu(
          nn_ops.xw_plus_b(inp, hidden_weight, hidden_bias), name="features")
      logits = nn_ops.xw_plus_b(
          features, softmax_weight, softmax_bias, name="logits")
      labels = constant_op.constant(
          label_data.tolist(),
          shape=[batch, classes],
          dtype=dtypes.float64,
          name="labels")
      cost = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name="cost")
      return cost

    def f_restricted(x):
      xs = all_params
      i = param_index
      # use x for the i-th parameter
      xs = xs[0:i] + [x] + xs[i + 1:]
      return f(*xs)

    # Test the gradients.
    err = gradient_checker.max_error(*gradient_checker.compute_gradient(
        f_restricted, [all_params[param_index]], delta=1e-5))

    tf_logging.info("Mini MNIST: %s gradient error = %g", tag, err)
    return err

  def testInputGradient(self):
    self.assertLess(self._BuildAndTestMiniMNIST(0, "input"), 1e-8)

  def testHiddenWeightGradient(self):
    self.assertLess(self._BuildAndTestMiniMNIST(1, "hidden_weight"), 1e-8)

  def testHiddenBiasGradient(self):
    self.assertLess(self._BuildAndTestMiniMNIST(2, "hidden_bias"), 1e-8)

  def testSoftmaxWeightGradient(self):
    self.assertLess(self._BuildAndTestMiniMNIST(3, "softmax_weight"), 1e-8)

  def testSoftmaxBiasGradient(self):
    self.assertLess(self._BuildAndTestMiniMNIST(4, "softmax_bias"), 1e-8)


if __name__ == "__main__":
  test.main()
