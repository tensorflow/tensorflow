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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


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
    with tc.test_session(use_gpu=True):
      return init([num]).eval()

  return func


class ConstantInitializersTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testZerosInitializer(self):
    with self.session(use_gpu=True):
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.zeros_initializer())
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.zeros(shape))

  @test_util.run_deprecated_v1
  def testOnesInitializer(self):
    with self.session(use_gpu=True):
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.ones_initializer())
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.ones(shape))

  @test_util.run_deprecated_v1
  def testConstantZeroInitializer(self):
    with self.session(use_gpu=True):
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.constant_initializer(0.0))
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.zeros(shape))

  @test_util.run_deprecated_v1
  def testConstantOneInitializer(self):
    with self.session(use_gpu=True):
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.constant_initializer(1.0))
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.ones(shape))

  @test_util.run_deprecated_v1
  def testConstantIntInitializer(self):
    with self.session(use_gpu=True):
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x",
          shape=shape,
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer(7))
      x.initializer.run()
      self.assertEqual(x.dtype.base_dtype, dtypes.int32)
      self.assertAllEqual(x.eval(), 7 * np.ones(shape, dtype=np.int32))

  @test_util.run_deprecated_v1
  def testConstantTupleInitializer(self):
    with self.session(use_gpu=True):
      shape = [3]
      x = variable_scope.get_variable(
          "x",
          shape=shape,
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer((10, 20, 30)))
      x.initializer.run()
      self.assertEqual(x.dtype.base_dtype, dtypes.int32)
      self.assertAllEqual(x.eval(), [10, 20, 30])

  def _testNDimConstantInitializer(self, name, value, shape, expected):
    with self.cached_session(use_gpu=True):
      init = init_ops.constant_initializer(value, dtype=dtypes.int32)
      x = variable_scope.get_variable(name, shape=shape, initializer=init)
      x.initializer.run()

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
    self._testNDimConstantInitializer("ndarray",
                                      np.asarray(value), shape, expected)
    self._testNDimConstantInitializer("2D-ndarray",
                                      np.asarray(value).reshape(tuple(shape)),
                                      shape, expected)

  def _testNDimConstantInitializerLessValues(self, name, value, shape,
                                             expected):
    with self.cached_session(use_gpu=True):
      init = init_ops.constant_initializer(value, dtype=dtypes.int32)
      x = variable_scope.get_variable(name, shape=shape, initializer=init)
      x.initializer.run()

      actual = array_ops.reshape(x, [-1]).eval()
      self.assertGreater(len(actual), len(expected))
      for i in xrange(len(actual)):
        a = actual[i]
        e = expected[i] if i < len(expected) else expected[-1]
        self.assertEqual(a, e)

  @test_util.run_deprecated_v1
  def testNDimConstantInitializerLessValues(self):
    value = [0, 1, 2, 3, 4, 5]
    shape = [2, 4]
    expected = list(value)

    self._testNDimConstantInitializerLessValues("list", value, shape, expected)
    self._testNDimConstantInitializerLessValues("ndarray",
                                                np.asarray(value), shape,
                                                expected)
    self._testNDimConstantInitializerLessValues(
        "2D-ndarray", np.asarray(value).reshape(tuple([2, 3])), shape, expected)

  def _testNDimConstantInitializerMoreValues(self, value, shape):
    ops.reset_default_graph()
    with self.cached_session(use_gpu=True):
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
    with self.assertRaisesRegexp(
        TypeError, r"Invalid type for initial value: .*Tensor.*"):
      init_ops.constant_initializer(c, dtype=dtypes.float32)
    v = variables.Variable([3.0, 2.0, 1.0])
    with self.assertRaisesRegexp(
        TypeError, r"Invalid type for initial value: .*Variable.*"):
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
      variables.global_variables_initializer().run()
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
        distribution='truncated_normal')

    with self.session(use_gpu=True), \
      test.mock.patch.object(
          random_ops, 'truncated_normal', wraps=random_ops.truncated_normal) \
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
    init = init_ops.variance_scaling_initializer(distribution='normal')

    with self.session(use_gpu=True), \
      test.mock.patch.object(
          random_ops, 'truncated_normal', wraps=random_ops.truncated_normal) \
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
        distribution='untruncated_normal')

    with self.session(use_gpu=True), \
      test.mock.patch.object(
          random_ops, 'random_normal', wraps=random_ops.random_normal) \
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
    init = init_ops.variance_scaling_initializer(distribution='uniform')

    with self.session(use_gpu=True):
      x = init(shape).eval()

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)


# TODO(vrv): move to sequence_ops_test?
class RangeTest(test.TestCase):

  def _Range(self, start, limit, delta):
    with self.cached_session(use_gpu=True):
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
    with self.session(use_gpu=True):
      self.assertAllEqual(np.arange(5), math_ops.range(5).eval())

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
        math_ops.range(
            0, 0, 1, dtype=dtypes.int32).dtype, dtypes.int32)
    self.assertEqual(
        math_ops.range(
            0, 0, 1, dtype=dtypes.int64).dtype, dtypes.int64)
    self.assertEqual(
        math_ops.range(
            0, 0, 1, dtype=dtypes.float32).dtype, dtypes.float32)
    self.assertEqual(
        math_ops.range(
            0, 0, 1, dtype=dtypes.float64).dtype, dtypes.float64)


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
          self._LinSpace(-1., -5., 4),
          np.array([-1., -7. / 3., -11. / 3., -5.]), 1e-5)

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
        ValueError, init_ops.convolutional_delta_orthogonal,
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
      init2 = init_ops.convolutional_delta_orthogonal(gain=3.14,
                                                      seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testShapesValues(self):
    gain = 3.14
    for dtype in [dtypes.float32]:
      for kernel_size in [[3], [8], [3, 5], [2, 4], [3, 3, 3], [2, 2, 2]]:
        tol = 1e-2
        # Check orthogonality by computing ratio between
        # the 2-norms of the inputs and outputs.
        if len(kernel_size) == 1:
          shape = [4, 32, 64]
          convolution = convolutional.conv1d
        elif len(kernel_size) == 2:
          convolution = convolutional.conv2d
          shape = [4, 32, 32, 64]
        else:
          shape = [4, 16, 16, 16, 64]
          convolution = convolutional.conv3d

          if test.is_built_with_rocm():
            # This subtest triggers a known bug in ROCm runtime code
            # The bug has been fixed and will be available in ROCm 2.7
            # Re-enable this test once ROCm 2.7 is released
            continue

        inputs = random_ops.random_normal(shape, dtype=dtype)
        inputs_2norm = linalg_ops.norm(inputs)
        outputs = convolution(
            inputs, padding="same", filters=128,
            kernel_size=kernel_size, use_bias=False,
            kernel_initializer=init_ops.convolutional_delta_orthogonal(
                gain=gain))
        outputs_shape = shape[0:-1] + [128]
        outputs_2norm = linalg_ops.norm(outputs)
        ratio = outputs_2norm / inputs_2norm
        my_ops = variables.global_variables_initializer()
        with self.session(use_gpu=True) as sess:
          self.evaluate(my_ops)
          # Check the shape of the outputs
          t = self.evaluate(outputs)
          self.assertAllEqual(t.shape, outputs_shape)
          # Check isometry of the delta-orthogonal kernel.
          self.assertAllClose(self.evaluate(ratio), gain, rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testNonuniformity(self):
    value = 0
    abs_value = 0
    shape = [3, 3, 10, 10]
    count = 70
    tol = 1e-5
    with self.session(use_gpu=True):
      for i in range(count):
        x = variable_scope.get_variable("{}".format(i), shape=shape,
                                        initializer=
                                        init_ops.convolutional_delta_orthogonal)
        x.initializer.run()
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
        ValueError, init_ops.convolutional_orthogonal_1d,
        dtype=dtypes.string)

  def testInvalidShape(self):
    init1 = init_ops.convolutional_orthogonal_1d()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_1d(gain=3.14,
                                                   seed=1, dtype=dtype)
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
    with self.session(use_gpu=True):
      for i in range(count):
        x = variable_scope.get_variable("{}".format(i), shape=shape,
                                        initializer=
                                        init_ops.convolutional_orthogonal_1d)
        x.initializer.run()
        y = np.sum(x.eval(), axis=0)
        determinant = np.linalg.det(y)
        value += determinant
        abs_value += np.abs(determinant)

      # Check there is some variation in the signs of the determinants.
      self.assertLess(value, count - tol)
      self.assertLess(-count + tol, value)
      # Check all determinants have absolute value 1
      # Compute the sum of the absolute values of 'count' determinants
      self.assertAllClose(abs_value, count, rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testShapesValues(self):
    def circular_pad(input_, width, kernel_size):
      """Pad input_ for computing (circular) convolution.

      Args:
        input_: the input tensor
        width: the width of the tensor.
        kernel_size: the kernel size of the filter.
      Returns:
        a tensor whose width is (width + kernel_size - 1).
      """

      beginning = kernel_size // 2
      end = kernel_size - 1 - beginning

      tmp_up = array_ops.slice(input_, [0, width - beginning, 0],
                               [-1, beginning, -1])
      tmp_down = array_ops.slice(input_, [0, 0, 0], [-1, end, -1])
      tmp = array_ops.concat([tmp_up, input_, tmp_down], 1)

      return tmp

    cout = 64
    shape = [10, 20, 32]
    outputs_shape = shape[0:-1] + [cout]
    dtype = dtypes.float32
    tol = 1e-3
    gain = 3.14
    # Check orthogonality/isometry by computing the ratio between
    # the 2-norms of the inputs and outputs.
    for kernel_size in [[1], [2], [3], [4], [5], [6]]:
      convolution = convolutional.conv1d
      inputs = random_ops.random_normal(shape, dtype=dtype)
      inputs_2norm = linalg_ops.norm(inputs)
      input_with_circular_pad = circular_pad(inputs, shape[1], kernel_size[0])
      outputs = convolution(
          input_with_circular_pad, padding="valid", filters=cout,
          kernel_size=kernel_size[0], use_bias=False,
          kernel_initializer=init_ops.convolutional_orthogonal_1d(gain=gain))
      outputs_2norm = linalg_ops.norm(outputs)
      ratio = outputs_2norm / inputs_2norm
      my_ops = variables.global_variables_initializer()
      with self.session(use_gpu=True) as sess:
        self.evaluate(my_ops)
        # Check the shape of the outputs
        t = self.evaluate(outputs)
        self.assertAllEqual(t.shape, outputs_shape)
        # Check isometry of the orthogonal kernel.
        self.assertAllClose(self.evaluate(ratio), gain, rtol=tol, atol=tol)


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
        ValueError, init_ops.convolutional_orthogonal_2d,
        dtype=dtypes.string)

  def testInvalidShape(self):
    init1 = init_ops.convolutional_orthogonal_2d()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (3, 3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_2d(gain=3.14,
                                                   seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testShapesValues(self):
    def circular_pad(input_, width, kernel_size):
      """Pad input_ for computing (circular) convolution.

      Args:
        input_: the input tensor
        width: the width of the tensor.
        kernel_size: the kernel size of the filter.
      Returns:
        a tensor whose width is (width + kernel_size - 1).
      """
      beginning = kernel_size // 2
      end = kernel_size - 1 - beginning

      tmp_up = array_ops.slice(input_, [0, width - beginning, 0, 0],
                               [-1, beginning, width, -1])
      tmp_down = array_ops.slice(input_, [0, 0, 0, 0], [-1, end, width, -1])
      tmp = array_ops.concat([tmp_up, input_, tmp_down], 1)

      new_width = width + kernel_size - 1
      tmp_left = array_ops.slice(tmp, [0, 0, width - beginning, 0],
                                 [-1, new_width, beginning, -1])
      tmp_right = array_ops.slice(tmp, [0, 0, 0, 0], [-1, new_width, end, -1])

      final = array_ops.concat([tmp_left, tmp, tmp_right], 2)
      return final

    cout = 45
    shape = [64, 28, 28, 32]
    outputs_shape = shape[0:-1] + [cout]
    dtype = dtypes.float32
    tol = 1e-3
    gain = 3.14
    # Check orthogonality/isometry by computing the ratio between
    # the 2-norms of the inputs and outputs.
    for kernel_size in [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]:
      convolution = convolutional.conv2d
      inputs = random_ops.random_normal(shape, dtype=dtype)
      inputs_2norm = linalg_ops.norm(inputs)
      input_with_circular_pad = circular_pad(inputs, shape[1], kernel_size[0])
      outputs = convolution(
          input_with_circular_pad, padding="valid", filters=cout,
          kernel_size=kernel_size, use_bias=False,
          kernel_initializer=init_ops.convolutional_orthogonal_2d(gain=gain))
      outputs_2norm = linalg_ops.norm(outputs)
      ratio = outputs_2norm / inputs_2norm
      my_ops = variables.global_variables_initializer()
      with self.session(use_gpu=True) as sess:
        self.evaluate(my_ops)
        # Check the shape of the outputs
        t = self.evaluate(outputs)
        self.assertAllEqual(t.shape, outputs_shape)
        # Check isometry of the orthogonal kernel.
        self.assertAllClose(self.evaluate(ratio), gain, rtol=tol, atol=tol)


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
        ValueError, init_ops.convolutional_orthogonal_3d,
        dtype=dtypes.string)

  def testInvalidShape(self):
    init1 = init_ops.convolutional_orthogonal_3d()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 3, 3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (3, 3, 3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_3d(gain=3.14,
                                                   seed=1, dtype=dtype)
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
    with self.session(use_gpu=True):
      for i in range(count):
        x = variable_scope.get_variable("{}".format(i), shape=shape,
                                        initializer=
                                        init_ops.convolutional_orthogonal_3d)
        x.initializer.run()
        y = np.sum(x.eval(), axis=(0, 1, 2))
        determinant = np.linalg.det(y)
        value += determinant
        abs_value += np.abs(determinant)

      # Check there is some variation in the signs of the determinants
      self.assertLess(value, count - tol)
      self.assertLess(-count + tol, value)
      # Check all determinants have absolute value 1
      # Compute the sum of the absolute values of 'count' determinants
      self.assertAllClose(abs_value, count, rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testShapesValues(self):
    def circular_pad(input_, width, kernel_size):
      """Padding input_ for computing circular convolution.

      Args:
        input_: the input tensor
        width: the width of the tensor.
        kernel_size: the kernel size of the filter.

      Returns:
        a tensor whose width is (width + kernel_size - 1).
      """

      beginning = kernel_size // 2
      end = kernel_size - 1 - beginning

      tmp_up = array_ops.slice(input_, [0, width - beginning, 0, 0, 0],
                               [-1, beginning, -1, -1, -1])
      tmp_down = array_ops.slice(input_, [0, 0, 0, 0, 0],
                                 [-1, end, -1, -1, -1])
      tmp = array_ops.concat([tmp_up, input_, tmp_down], 1)

      tmp_left = array_ops.slice(tmp, [0, 0, width - beginning, 0, 0],
                                 [-1, -1, beginning, -1, -1])
      tmp_right = array_ops.slice(tmp, [0, 0, 0, 0, 0],
                                  [-1, -1, end, -1, -1])
      tmp = array_ops.concat([tmp_left, tmp, tmp_right], 2)

      tmp_front = array_ops.slice(tmp, [0, 0, 0, width - beginning, 0],
                                  [-1, -1, -1, beginning, -1])
      tmp_back = array_ops.slice(tmp, [0, 0, 0, 0, 0], [-1, -1, -1, end, -1])
      return array_ops.concat([tmp_front, tmp, tmp_back], 3)

    cout = 32
    shape = [1, 7, 7, 7, 16]
    outputs_shape = shape[0:-1] + [cout]
    dtype = dtypes.float32
    tol = 1e-3
    gain = 3.14
    # Check orthogonality/isometry by computing the ratio between
    # the 2-norms of the inputs and outputs.
    for kernel_size in [[1, 1, 1], [2, 2, 2], [3, 3, 3]]:
      convolution = convolutional.conv3d
      inputs = random_ops.random_normal(shape, dtype=dtype)
      inputs_2norm = linalg_ops.norm(inputs)
      input_with_circular_pad = circular_pad(inputs, shape[1], kernel_size[0])
      outputs = convolution(
          input_with_circular_pad, padding="valid", filters=cout,
          kernel_size=kernel_size[0], use_bias=False,
          kernel_initializer=init_ops.convolutional_orthogonal_3d(gain=gain))
      outputs_2norm = linalg_ops.norm(outputs)
      ratio = outputs_2norm / inputs_2norm
      my_ops = variables.global_variables_initializer()
      with self.cached_session(use_gpu=True) as sess:
        self.evaluate(my_ops)
        # Check the shape of the outputs
        t = self.evaluate(outputs)
        self.assertAllEqual(t.shape, outputs_shape)
        # Check isometry of the orthogonal kernel.
        self.assertAllClose(self.evaluate(ratio), gain, rtol=tol, atol=tol)


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
      self.assertAllClose(init(shape).eval(), np.eye(*shape))

  @test_util.run_deprecated_v1
  def testGain(self):
    shape = (10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init_default = init_ops.identity_initializer(dtype=dtype)
      init_custom = init_ops.identity_initializer(gain=0.9, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        self.assertAllClose(init_default(shape).eval(), np.eye(*shape))
      with self.session(graph=ops.Graph(), use_gpu=True):
        self.assertAllClose(init_custom(shape).eval(), np.eye(*shape) * 0.9)

  @test_util.run_deprecated_v1
  def testPartitions(self):
    shape = (10, 10)
    init = init_ops.identity_initializer()
    partitioner = partitioned_variables.variable_axis_size_partitioner(1)
    with self.session(graph=ops.Graph(), use_gpu=True):
      with variable_scope.variable_scope(
          "foo", partitioner=partitioner, initializer=init):
        v = array_ops.identity(variable_scope.get_variable("bar", shape=shape))
      variables.global_variables_initializer().run()
      self.assertAllClose(v.eval(), np.eye(*shape))


if __name__ == "__main__":
  test.main()
