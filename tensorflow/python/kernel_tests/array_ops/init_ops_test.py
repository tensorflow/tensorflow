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
"""Tests for tensorflow.ops.ops."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util.numpy_compat import np_where


# Returns true iff the two initializers produce the same tensor to
# within a tiny tolerance.
def identicaltest(tc, init1, init2, shape=None):
  """Tests if two initializations are identical to within tiny tolerances.

  Args:
    tc: An instance of TensorFlowTestCase.
    init1: An Initializer that generates a tensor of a given shape
    init2: An Initializer that generates a tensor of a given shape
    shape: Shape of the tensor to initialize or `None` to use a vector of length
      100.

  Returns:
    True or False as determined by test.
  """
  if shape is None:
    shape = [100]
  with tc.test_session(graph=ops.Graph()):
    t1 = init1(shape).eval()
  with tc.test_session(graph=ops.Graph()):
    t2 = init2(shape).eval()
  return np.allclose(t1, t2, rtol=1e-15, atol=1e-15)


def duplicated_initializer(tc, init, graph_seed, shape=None):
  """Tests duplicated random initializer within the same graph.

  This test generates two random kernels from the same initializer to the same
  graph, and checks if the results are close enough. Even given the same global,
  seed, two different instances of random kernels should generate different
  results.

  Args:
    tc: An instance of TensorFlowTestCase.
    init: An Initializer that generates a tensor of a given shape
    graph_seed: A graph-level seed to use.
    shape: Shape of the tensor to initialize or `None` to use a vector of length
      100.

  Returns:
    True or False as determined by test.
  """
  if shape is None:
    shape = [100]
  with tc.test_session(graph=ops.Graph()):
    random_seed.set_random_seed(graph_seed)
    t1 = init(shape).eval()
    t2 = init(shape).eval()
    return np.allclose(t1, t2, rtol=1e-15, atol=1e-15)


def _init_sampler(tc, init, num):
  """Returns a func to generate a random tensor of shape [num].

  Args:
    tc: An instance of TensorFlowTestCase.
    init: An Initializer that generates a tensor of a given shape
    num: Size of 1D tensor to create.

  Returns:
    Function to generate a random tensor.
  """

  def func():
    with tc.test_session():
      return init([num]).eval()

  return func


class ConstantInitializersTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testZerosInitializer(self):
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.zeros_initializer())
      self.evaluate(x.initializer)
      self.assertAllEqual(x, np.zeros(shape))

  @test_util.run_deprecated_v1
  def testOnesInitializer(self):
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.ones_initializer())
      self.evaluate(x.initializer)
      self.assertAllEqual(x, np.ones(shape))

  @test_util.run_deprecated_v1
  def testConstantZeroInitializer(self):
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.constant_initializer(0.0))
      self.evaluate(x.initializer)
      self.assertAllEqual(x, np.zeros(shape))

  @test_util.run_deprecated_v1
  def testConstantOneInitializer(self):
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.constant_initializer(1.0))
      self.evaluate(x.initializer)
      self.assertAllEqual(x, np.ones(shape))

  @test_util.run_deprecated_v1
  def testConstantIntInitializer(self):
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x",
          shape=shape,
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer(7))
      self.evaluate(x.initializer)
      self.assertEqual(x.dtype.base_dtype, dtypes.int32)
      self.assertAllEqual(x, 7 * np.ones(shape, dtype=np.int32))

  @test_util.run_deprecated_v1
  def testConstantTupleInitializer(self):
    with self.session():
      shape = [3]
      x = variable_scope.get_variable(
          "x",
          shape=shape,
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer((10, 20, 30)))
      self.evaluate(x.initializer)
      self.assertEqual(x.dtype.base_dtype, dtypes.int32)
      self.assertAllEqual(x, [10, 20, 30])

  def _testNDimConstantInitializer(self, name, value, shape, expected):
    with self.cached_session():
      init = init_ops.constant_initializer(value, dtype=dtypes.int32)
      x = variable_scope.get_variable(name, shape=shape, initializer=init)
      self.evaluate(x.initializer)

      actual = array_ops.reshape(x, [-1]).eval()
      self.assertEqual(len(actual), len(expected))
      for a, e in zip(actual, expected):
        self.assertEqual(a, e)

  @test_util.run_deprecated_v1
  def testNDimConstantInitializer(self):
    value = [0, 1, 2, 3, 4, 5]
    shape = [2, 3]
    expected = list(value)

    self._testNDimConstantInitializer("list", value, shape, expected)
    self._testNDimConstantInitializer("ndarray", np.asarray(value), shape,
                                      expected)
    self._testNDimConstantInitializer("2D-ndarray",
                                      np.asarray(value).reshape(tuple(shape)),
                                      shape, expected)

  def _testNDimConstantInitializerLessValues(self, name, value, shape,
                                             expected):
    with self.cached_session():
      init = init_ops.constant_initializer(value, dtype=dtypes.int32)
      x = variable_scope.get_variable(name, shape=shape, initializer=init)
      self.evaluate(x.initializer)

      actual = array_ops.reshape(x, [-1]).eval()
      self.assertGreater(len(actual), len(expected))
      for i in range(len(actual)):
        a = actual[i]
        e = expected[i] if i < len(expected) else expected[-1]
        self.assertEqual(a, e)

  @test_util.run_deprecated_v1
  def testNDimConstantInitializerLessValues(self):
    value = [0, 1, 2, 3, 4, 5]
    shape = [2, 4]
    expected = list(value)

    self._testNDimConstantInitializerLessValues("list", value, shape, expected)
    self._testNDimConstantInitializerLessValues("ndarray", np.asarray(value),
                                                shape, expected)
    self._testNDimConstantInitializerLessValues(
        "2D-ndarray",
        np.asarray(value).reshape(tuple([2, 3])), shape, expected)

  def _testNDimConstantInitializerMoreValues(self, value, shape):
    ops.reset_default_graph()
    with self.cached_session():
      init = init_ops.constant_initializer(value, dtype=dtypes.int32)
      self.assertRaises(
          ValueError,
          variable_scope.get_variable,
          "x",
          shape=shape,
          initializer=init)

  @test_util.run_deprecated_v1
  def testNDimConstantInitializerMoreValues(self):
    value = [0, 1, 2, 3, 4, 5, 6, 7]
    shape = [2, 3]
    self._testNDimConstantInitializerMoreValues(value, shape)
    self._testNDimConstantInitializerMoreValues(np.asarray(value), shape)
    self._testNDimConstantInitializerMoreValues(
        np.asarray(value).reshape(tuple([2, 4])), shape)

  def testInvalidValueTypeForConstantInitializerCausesTypeError(self):
    c = constant_op.constant([1.0, 2.0, 3.0])
    with self.assertRaisesRegex(TypeError,
                                r"Invalid type for initial value=.*Tensor.*"):
      init_ops.constant_initializer(c, dtype=dtypes.float32)
    v = variables.Variable([3.0, 2.0, 1.0])
    with self.assertRaisesRegex(
        TypeError, r"Invalid type for initial value=.*Variable.*"):
      init_ops.constant_initializer(v, dtype=dtypes.float32)


class RandomNormalInitializationTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      init2 = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      init2 = init_ops.random_normal_initializer(0.0, 1.0, seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    init = init_ops.random_normal_initializer(0.0, 1.0)
    self.assertFalse(duplicated_initializer(self, init, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError,
        init_ops.random_normal_initializer,
        0.0,
        1.0,
        dtype=dtypes.string)


class TruncatedNormalInitializationTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.truncated_normal_initializer(
          0.0, 1.0, seed=1, dtype=dtype)
      init2 = init_ops.truncated_normal_initializer(
          0.0, 1.0, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.truncated_normal_initializer(
          0.0, 1.0, seed=1, dtype=dtype)
      init2 = init_ops.truncated_normal_initializer(
          0.0, 1.0, seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    init = init_ops.truncated_normal_initializer(0.0, 1.0)
    self.assertFalse(duplicated_initializer(self, init, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError,
        init_ops.truncated_normal_initializer,
        0.0,
        1.0,
        dtype=dtypes.string)


class RandomUniformInitializationTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    for dtype in [dtypes.float32, dtypes.float64, dtypes.int64]:
      init1 = init_ops.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
      init2 = init_ops.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    for dtype in [dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64]:
      init1 = init_ops.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
      init2 = init_ops.random_uniform_initializer(0, 7, seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    init = init_ops.random_uniform_initializer(0.0, 1.0)
    self.assertFalse(duplicated_initializer(self, init, 1))


class UniformUnitScalingInitializationTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
      init2 = init_ops.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))
      init3 = init_ops.uniform_unit_scaling_initializer(
          1.5, seed=1, dtype=dtype)
      init4 = init_ops.uniform_unit_scaling_initializer(
          1.5, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init3, init4))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
      init2 = init_ops.uniform_unit_scaling_initializer(seed=2, dtype=dtype)
      init3 = init_ops.uniform_unit_scaling_initializer(
          1.5, seed=1, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))
      self.assertFalse(identicaltest(self, init1, init3))
      self.assertFalse(identicaltest(self, init2, init3))

  @test_util.run_deprecated_v1
  def testZeroSize(self):
    shape = [0, 2]
    with self.cached_session():
      x = variable_scope.get_variable(
          "x",
          shape=shape,
          initializer=init_ops.uniform_unit_scaling_initializer())
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(shape, self.evaluate(x).shape)

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    init = init_ops.uniform_unit_scaling_initializer()
    self.assertFalse(duplicated_initializer(self, init, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError,
        init_ops.uniform_unit_scaling_initializer,
        dtype=dtypes.string)


class VarianceScalingInitializationTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testTruncatedNormalDistribution(self):
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops.variance_scaling_initializer(
        distribution="truncated_normal")

    with self.session(), \
      test.mock.patch.object(
          random_ops, "truncated_normal", wraps=random_ops.truncated_normal) \
          as mock_truncated_normal:
      x = init(shape).eval()
      self.assertTrue(mock_truncated_normal.called)

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_deprecated_v1
  def testNormalDistribution(self):
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops.variance_scaling_initializer(distribution="normal")

    with self.session(), \
      test.mock.patch.object(
          random_ops, "truncated_normal", wraps=random_ops.truncated_normal) \
          as mock_truncated_normal:
      x = init(shape).eval()
      self.assertTrue(mock_truncated_normal.called)

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_deprecated_v1
  def testUntruncatedNormalDistribution(self):
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops.variance_scaling_initializer(
        distribution="untruncated_normal")

    with self.session(), \
      test.mock.patch.object(
          random_ops, "random_normal", wraps=random_ops.random_normal) \
          as mock_random_normal:
      x = init(shape).eval()
      self.assertTrue(mock_random_normal.called)

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_deprecated_v1
  def testUniformDistribution(self):
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops.variance_scaling_initializer(distribution="uniform")

    with self.session():
      x = init(shape).eval()

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)


# TODO(vrv): move to sequence_ops_test?
class RangeTest(test.TestCase):

  def _Range(self, start, limit, delta):
    with self.cached_session():
      tf_ans = math_ops.range(start, limit, delta, name="range")
      self.assertEqual([len(np.arange(start, limit, delta))],
                       tf_ans.get_shape())
      return self.evaluate(tf_ans)

  def testBasic(self):
    self.assertTrue(
        np.array_equal(self._Range(0, 5, 1), np.array([0, 1, 2, 3, 4])))
    self.assertTrue(np.array_equal(self._Range(0, 5, 2), np.array([0, 2, 4])))
    self.assertTrue(np.array_equal(self._Range(0, 6, 2), np.array([0, 2, 4])))
    self.assertTrue(
        np.array_equal(self._Range(13, 32, 7), np.array([13, 20, 27])))
    self.assertTrue(
        np.array_equal(
            self._Range(100, 500, 100), np.array([100, 200, 300, 400])))
    self.assertEqual(math_ops.range(0, 5, 1).dtype, dtypes.int32)

  @test_util.run_deprecated_v1
  def testLimitOnly(self):
    with self.session():
      self.assertAllEqual(np.arange(5), math_ops.range(5))

  def testEmpty(self):
    for start in 0, 5:
      self.assertTrue(np.array_equal(self._Range(start, start, 1), []))

  def testNonInteger(self):
    self.assertTrue(
        np.allclose(self._Range(0, 2, 0.5), np.array([0, 0.5, 1, 1.5])))
    self.assertTrue(np.allclose(self._Range(0, 5, 2.5), np.array([0, 2.5])))
    self.assertTrue(
        np.allclose(self._Range(0, 3, 0.9), np.array([0, 0.9, 1.8, 2.7])))
    self.assertTrue(
        np.allclose(
            self._Range(100., 500., 100.), np.array([100, 200, 300, 400])))
    self.assertEqual(math_ops.range(0., 5., 1.).dtype, dtypes.float32)

  def testNegativeDelta(self):
    self.assertTrue(
        np.array_equal(self._Range(5, -1, -1), np.array([5, 4, 3, 2, 1, 0])))
    self.assertTrue(
        np.allclose(self._Range(2.5, 0, -0.5), np.array([2.5, 2, 1.5, 1, 0.5])))
    self.assertTrue(
        np.array_equal(self._Range(-5, -10, -3), np.array([-5, -8])))

  def testDType(self):
    zero_int32 = math_ops.cast(0, dtypes.int32)
    zero_int64 = math_ops.cast(0, dtypes.int64)
    zero_float32 = math_ops.cast(0, dtypes.float32)
    zero_float64 = math_ops.cast(0, dtypes.float64)

    self.assertEqual(math_ops.range(zero_int32, 0, 1).dtype, dtypes.int32)
    self.assertEqual(math_ops.range(zero_int64, 0, 1).dtype, dtypes.int64)
    self.assertEqual(math_ops.range(zero_float32, 0, 1).dtype, dtypes.float32)
    self.assertEqual(math_ops.range(zero_float64, 0, 1).dtype, dtypes.float64)

    self.assertEqual(
        math_ops.range(zero_int32, zero_int64, 1).dtype, dtypes.int64)
    self.assertEqual(
        math_ops.range(zero_int64, zero_float32, 1).dtype, dtypes.float32)
    self.assertEqual(
        math_ops.range(zero_float32, zero_float64, 1).dtype, dtypes.float64)
    self.assertEqual(
        math_ops.range(zero_float64, zero_int32, 1).dtype, dtypes.float64)

    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.int32).dtype, dtypes.int32)
    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.int64).dtype, dtypes.int64)
    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.float32).dtype, dtypes.float32)
    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.float64).dtype, dtypes.float64)

  def testMixedDType(self):
    # Test case for GitHub issue 35710
    tf_ans = math_ops.range(
        constant_op.constant(4, dtype=dtypes.int32), dtype=dtypes.int64)
    self.assertAllEqual(self.evaluate(tf_ans), np.array([0, 1, 2, 3]))

  def testLargeStarts(self):
    # Test case for GitHub issue 46899.
    with self.session():
      with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
        v = math_ops.range(start=-1e+38, limit=1)
        self.evaluate(v)


# TODO(vrv): move to sequence_ops_test?
class LinSpaceTest(test.TestCase):

  def _gpu_modes(self):
    if test.is_gpu_available():
      return [False, True]
    else:
      return [False]

  def _LinSpace(self, start, stop, num):
    with ops.Graph().as_default() as graph:
      with self.session(graph=graph, force_gpu=self.force_gpu):
        tf_ans = math_ops.linspace(start, stop, num, name="linspace")
        self.assertEqual([num], tf_ans.get_shape())
        return self.evaluate(tf_ans)

  def testPositive(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(1., 5., 1), np.array([1.]), 1e-5)
      self.assertArrayNear(self._LinSpace(1., 5., 2), np.array([1., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 3), np.array([1., 3., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 4), np.array([1., 7. / 3., 11. / 3., 5.]),
          1e-5)

  def testNegative(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., -5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 2), np.array([-1., -5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 3), np.array([-1., -3., -5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 4), np.array([-1., -7. / 3., -11. / 3.,
                                                 -5.]), 1e-5)

  def testNegativeToPositive(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., 5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 2), np.array([-1., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 3), np.array([-1., 2., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 4), np.array([-1., 1., 3., 5.]), 1e-5)

  def testPoint(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(5., 5., 1), np.array([5.]), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 2), np.array([5.] * 2), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 3), np.array([5.] * 3), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 4), np.array([5.] * 4), 1e-5)

  def testEndpointsAreExact(self):
    for self.force_gpu in self._gpu_modes():
      # Test some cases that produce last values not equal to "stop" when
      # computed via start + (num - 1) * ((stop - start) / (num - 1)), since
      # float arithmetic will introduce error through precision loss.
      self.assertAllEqual(
          self._LinSpace(0., 1., 42)[[0, -1]], np.array([0., 1.], np.float32))
      self.assertAllEqual(
          self._LinSpace(-1., 0., 42)[[0, -1]], np.array([-1., 0.], np.float32))
      self.assertAllEqual(
          self._LinSpace(.1, .2, 4)[[0, -1]], np.array([.1, .2], np.float32))
      # Check a case for float64 error too.
      self.assertAllEqual(
          self._LinSpace(np.array(0., np.float64), .1, 12)[[0, -1]],
          np.array([0., .1], np.float64))


class LinSpaceNdTest(test.TestCase):

  def _gpu_modes(self):
    if test.is_gpu_available():
      return [False, True]
    else:
      return [False]

  def _LinSpace(self, start, stop, num, axis=0):
    with ops.Graph().as_default() as graph:
      with self.session(graph=graph, force_gpu=self.force_gpu):
        tf_ans = math_ops.linspace_nd(start, stop, num, axis=axis)
        return self.evaluate(tf_ans)

  def _LinSpaceNumConstant(self, start, stop, num, axis=0):
    with ops.Graph().as_default() as graph:
      num_constant = constant_op.constant(num)
      with self.session(graph=graph, force_gpu=self.force_gpu):
        tf_ans = math_ops.linspace_nd(start, stop, num_constant, axis=axis)
        return self.evaluate(tf_ans)

  def _LinspaceNoneShape(self, start, stop, num, graph_shape=None, axis=0):
    with ops.Graph().as_default() as graph:
      num_tensor = array_ops.placeholder(dtypes.int32)
      start_t = array_ops.placeholder(dtypes.float32, shape=graph_shape)
      stop_t = array_ops.placeholder(dtypes.float32, shape=graph_shape)
      ans_tensor = math_ops.linspace_nd(start_t, stop_t, num_tensor, axis=axis)

      with self.session(graph=graph, force_gpu=self.force_gpu) as sess:
        feed_dict = {start_t: start, stop_t: stop, num_tensor: num}
        return sess.run(ans_tensor, feed_dict=feed_dict)

  def testPositive(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(1., 5., 1), np.array([1.]), 1e-5)
      self.assertArrayNear(self._LinSpace(1., 5., 2), np.array([1., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 3), np.array([1., 3., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 4), np.array([1., 7. / 3., 11. / 3., 5.]),
          1e-5)

  def testNegative(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., -5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 2), np.array([-1., -5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 3), np.array([-1., -3., -5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 4), np.array([-1., -7. / 3., -11. / 3.,
                                                 -5.]), 1e-5)

  def testNegativeToPositive(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., 5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 2), np.array([-1., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 3), np.array([-1., 2., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 4), np.array([-1., 1., 3., 5.]), 1e-5)

  def testPoint(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(5., 5., 1), np.array([5.]), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 2), np.array([5.] * 2), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 3), np.array([5.] * 3), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 4), np.array([5.] * 4), 1e-5)

  def testEndpointsAreExact(self):
    for self.force_gpu in self._gpu_modes():
      # Test some cases that produce last values not equal to "stop" when
      # computed via start + (num - 1) * ((stop - start) / (num - 1)), since
      # float arithmetic will introduce error through precision loss.
      self.assertAllEqual(
          self._LinSpace(0., 1., 42)[[0, -1]], np.array([0., 1.], np.float32))
      self.assertAllEqual(
          self._LinSpace(-1., 0., 42)[[0, -1]], np.array([-1., 0.], np.float32))
      self.assertAllEqual(
          self._LinSpace(.1, .2, 4)[[0, -1]], np.array([.1, .2], np.float32))
      # Check a case for float64 error too.
      self.assertAllEqual(
          self._LinSpace(np.array(0., np.float64), .1, 12)[[0, -1]],
          np.array([0., .1], np.float64))

  def testScalarsCompareToNumpy(self):
    for self.force_gpu in self._gpu_modes():
      actual = self._LinSpace(0., 1., 32)
      expected = np.linspace(0., 1., 32)
      self.assertArrayNear(expected, actual, 1e-5)

  def _baseNDArrayCompareToNumpy(self, axis):
    for self.force_gpu in self._gpu_modes():
      a, b, expected, num = self.create_nd_inputs_and_expected_output(axis)
      actual = self._LinSpace(a, b, num, axis=axis)
      self.assert_close(actual, expected)

  def assert_close(self, actual, expected):
    wrong_indices = np_where(~np.allclose(actual, expected))
    mess = "Wrong float answer. Wrong indices: {}".format(wrong_indices)
    self.assertTrue(np.allclose(actual, expected), mess)

  def create_nd_inputs_and_expected_output(self, axis):
    a = np.arange(2, dtype=np.float32)
    b = a * 5
    num = 5

    res = np.array([[0., 0., 0., 0., 0.], [1., 2., 3., 4., 5.]])
    expected = res if axis != 0 else res.T
    return a, b, expected, num

  def testNDArrayCompareToNumpyDefaultAxis(self):
    self._baseNDArrayCompareToNumpy(0)

  def testNDArrayAxisStrictlyPositive(self):
    self._baseNDArrayCompareToNumpy(1)

  def testNDArrayAxisStrictlyNegative(self):
    self._baseNDArrayCompareToNumpy(-1)

  def testNumConstant(self):
    for self.force_gpu in self._gpu_modes():
      actual = self._LinSpaceNumConstant(0., 1., 32)
      expected = np.linspace(0., 1., 32)
      self.assertArrayNear(expected, actual, 1e-5)

  def testUnknownShapeAtGraphCreationTime(self):
    self.base_test_unknown_shape((2))

  def testNoneValuesInShapeAtGraphCreationTime(self):
    self.base_test_unknown_shape((None))

  def testNoneShapeAtGraphCreationTime(self):
    self.base_test_unknown_shape(None)

  def base_test_unknown_shape(self, graph_shape):
    for self.force_gpu in self._gpu_modes():
      axis = 1
      a, b, expected, num = self.create_nd_inputs_and_expected_output(axis)
      actual = self._LinspaceNoneShape(a, b, num, graph_shape, axis)
      self.assert_close(actual, expected)


class DeviceTest(test.TestCase):

  def testNoDevice(self):
    with ops.Graph().as_default():
      var = variables.Variable([[1.0, 1.0]])
    self.assertDeviceEqual(None, var.device)
    self.assertDeviceEqual(None, var.initializer.device)

  def testDevice(self):
    with ops.Graph().as_default():
      with ops.device("/job:ps"):
        var = variables.Variable([[1.0, 1.0]])
    self.assertDeviceEqual("/job:ps", var.device)
    self.assertDeviceEqual("/job:ps", var.initializer.device)


class OrthogonalInitializerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
      init2 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
      init2 = init_ops.orthogonal_initializer(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    init = init_ops.orthogonal_initializer()
    self.assertFalse(duplicated_initializer(self, init, 1, (10, 10)))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError, init_ops.orthogonal_initializer, dtype=dtypes.string)

  def testInvalidShape(self):
    init1 = init_ops.orthogonal_initializer()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[5])

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
      init2 = init_ops.orthogonal_initializer(gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testShapesValues(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      for shape in [(10, 10), (10, 9, 8), (100, 5, 5), (50, 40), (40, 50)]:
        init = init_ops.orthogonal_initializer(dtype=dtype)
        tol = 1e-5 if dtype == dtypes.float32 else 1e-12
        with self.session(graph=ops.Graph(), use_gpu=True):
          # Check the shape
          t = init(shape).eval()
          self.assertAllEqual(shape, t.shape)
          # Check orthogonality by computing the inner product
          t = t.reshape((np.prod(t.shape[:-1]), t.shape[-1]))
          if t.shape[0] > t.shape[1]:
            self.assertAllClose(
                np.dot(t.T, t), np.eye(t.shape[1]), rtol=tol, atol=tol)
          else:
            self.assertAllClose(
                np.dot(t, t.T), np.eye(t.shape[0]), rtol=tol, atol=tol)


class ConvolutionDeltaOrthogonalInitializerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_delta_orthogonal(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    init = init_ops.convolutional_delta_orthogonal()
    self.assertFalse(duplicated_initializer(self, init, 1, (3, 3, 10, 10)))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError,
        init_ops.convolutional_delta_orthogonal,
        dtype=dtypes.string)

  def testInvalidShape(self):
    init1 = init_ops.convolutional_delta_orthogonal()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (3, 3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_delta_orthogonal(
          gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testNonuniformity(self):
    value = 0
    abs_value = 0
    shape = [3, 3, 10, 10]
    count = 70
    tol = 1e-5
    with self.session():
      for i in range(count):
        x = variable_scope.get_variable(
            "{}".format(i),
            shape=shape,
            initializer=init_ops.convolutional_delta_orthogonal)
        self.evaluate(x.initializer)
        y = self.evaluate(x)[1, 1, :, :]
        determinant = np.linalg.det(y)
        value += determinant
        abs_value += np.abs(determinant)

      # Check there is some variation in the signs of the determinants
      self.assertLess(value, count - tol)
      self.assertLess(-count + tol, value)
      # Check all determinants have absolute value 1
      # Compute the sum of the absolute values of 'count' determinants
      self.assertAllClose(abs_value, count, rtol=tol, atol=tol)


@test_util.run_all_without_tensor_float_32(
    "Tests convolutional_orthogonal_1d, which calls matmul")
class ConvolutionOrthogonal1dInitializerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (3, 10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_1d(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (3, 10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    init = init_ops.convolutional_orthogonal_1d()
    self.assertFalse(duplicated_initializer(self, init, 1, (3, 10, 10)))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError, init_ops.convolutional_orthogonal_1d, dtype=dtypes.string)

  def testInvalidShape(self):
    init1 = init_ops.convolutional_orthogonal_1d()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_1d(
          gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testNonuniformity(self):
    value = 0
    abs_value = 0
    shape = [3, 10, 10]
    count = 70
    tol = 1e-5
    with self.session():
      for i in range(count):
        x = variable_scope.get_variable(
            "{}".format(i),
            shape=shape,
            initializer=init_ops.convolutional_orthogonal_1d)
        self.evaluate(x.initializer)
        y = np.sum(self.evaluate(x), axis=0)
        determinant = np.linalg.det(y)
        value += determinant
        abs_value += np.abs(determinant)

      # Check there is some variation in the signs of the determinants.
      self.assertLess(value, count - tol)
      self.assertLess(-count + tol, value)
      # Check all determinants have absolute value 1
      # Compute the sum of the absolute values of 'count' determinants
      self.assertAllClose(abs_value, count, rtol=tol, atol=tol)


class ConvolutionOrthogonal2dInitializerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_2d(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    init = init_ops.convolutional_orthogonal_2d()
    self.assertFalse(duplicated_initializer(self, init, 1, (3, 3, 10, 10)))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError, init_ops.convolutional_orthogonal_2d, dtype=dtypes.string)

  def testInvalidShape(self):
    init1 = init_ops.convolutional_orthogonal_2d()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (3, 3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_2d(
          gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)


@test_util.run_all_without_tensor_float_32(
    "Tests convolutional_orthogonal_3d, which calls matmul")
class ConvolutionOrthogonal3dInitializerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (3, 3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_3d(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (3, 3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    init = init_ops.convolutional_orthogonal_3d()
    self.assertFalse(duplicated_initializer(self, init, 1, (3, 3, 3, 10, 10)))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError, init_ops.convolutional_orthogonal_3d, dtype=dtypes.string)

  def testInvalidShape(self):
    init1 = init_ops.convolutional_orthogonal_3d()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 3, 3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (3, 3, 3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_3d(
          gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testNonuniformity(self):
    value = 0
    abs_value = 0
    shape = [3, 3, 3, 5, 5]
    count = 20
    tol = 1e-5
    with self.session():
      for i in range(count):
        x = variable_scope.get_variable(
            "{}".format(i),
            shape=shape,
            initializer=init_ops.convolutional_orthogonal_3d)
        self.evaluate(x.initializer)
        y = np.sum(self.evaluate(x), axis=(0, 1, 2))
        determinant = np.linalg.det(y)
        value += determinant
        abs_value += np.abs(determinant)

      # Check there is some variation in the signs of the determinants
      self.assertLess(value, count - tol)
      self.assertLess(-count + tol, value)
      # Check all determinants have absolute value 1
      # Compute the sum of the absolute values of 'count' determinants
      self.assertAllClose(abs_value, count, rtol=tol, atol=tol)


class IdentityInitializerTest(test.TestCase):

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError, init_ops.orthogonal_initializer, dtype=dtypes.string)

  def testInvalidShape(self):
    init = init_ops.identity_initializer()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init, shape=[5, 7, 7])
      self.assertRaises(ValueError, init, shape=[5])
      self.assertRaises(ValueError, init, shape=[])

  @test_util.run_deprecated_v1
  def testNonSquare(self):
    init = init_ops.identity_initializer()
    shape = (10, 5)
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertAllClose(init(shape), np.eye(*shape))

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init_default = init_ops.identity_initializer(dtype=dtype)
      init_custom = init_ops.identity_initializer(gain=0.9, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        self.assertAllClose(init_default(shape), np.eye(*shape))
      with self.session(graph=ops.Graph(), use_gpu=True):
        self.assertAllClose(init_custom(shape), np.eye(*shape) * 0.9)

  @test_util.run_deprecated_v1
  def testPartitions(self):
    shape = (10, 10)
    init = init_ops.identity_initializer()
    partitioner = partitioned_variables.variable_axis_size_partitioner(1)
    with self.session(graph=ops.Graph(), use_gpu=True):
      with variable_scope.variable_scope(
          "foo", partitioner=partitioner, initializer=init):
        v = array_ops.identity(variable_scope.get_variable("bar", shape=shape))
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(v, np.eye(*shape))


if __name__ == "__main__":
  test.main()
