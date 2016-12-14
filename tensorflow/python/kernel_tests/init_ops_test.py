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
import tensorflow as tf

from tensorflow.python.framework import random_seed
from tensorflow.python.ops import init_ops


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
  with tc.test_session(graph=tf.Graph()):
    t1 = init1(shape).eval()
  with tc.test_session(graph=tf.Graph()):
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
  with tc.test_session(graph=tf.Graph()):
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


class ConstantInitializersTest(tf.test.TestCase):

  def testZerosInitializer(self):
    with self.test_session(use_gpu=True):
      shape = [2, 3]
      x = tf.get_variable("x", shape=shape, initializer=tf.zeros_initializer())
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.zeros(shape))

  def testOnesInitializer(self):
    with self.test_session(use_gpu=True):
      shape = [2, 3]
      x = tf.get_variable("x", shape=shape, initializer=tf.ones_initializer())
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.ones(shape))

  def testConstantZeroInitializer(self):
    with self.test_session(use_gpu=True):
      shape = [2, 3]
      x = tf.get_variable(
          "x", shape=shape, initializer=tf.constant_initializer(0.0))
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.zeros(shape))

  def testConstantOneInitializer(self):
    with self.test_session(use_gpu=True):
      shape = [2, 3]
      x = tf.get_variable(
          "x", shape=shape, initializer=tf.constant_initializer(1.0))
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.ones(shape))

  def testConstantIntInitializer(self):
    with self.test_session(use_gpu=True):
      shape = [2, 3]
      x = tf.get_variable(
          "x",
          shape=shape,
          dtype=tf.int32,
          initializer=tf.constant_initializer(7))
      x.initializer.run()
      self.assertEqual(x.dtype.base_dtype, tf.int32)
      self.assertAllEqual(x.eval(), 7 * np.ones(shape, dtype=np.int32))

  def _testNDimConstantInitializer(self, name, value, shape, expected):
    with self.test_session(use_gpu=True):
      init = tf.constant_initializer(value, dtype=tf.int32)
      x = tf.get_variable(name, shape=shape, initializer=init)
      x.initializer.run()

      actual = tf.reshape(x, [-1]).eval()
      self.assertEqual(len(actual), len(expected))
      for a, e in zip(actual, expected):
        self.assertEqual(a, e)

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
    with self.test_session(use_gpu=True):
      init = tf.constant_initializer(value, dtype=tf.int32)
      x = tf.get_variable(name, shape=shape, initializer=init)
      x.initializer.run()

      actual = tf.reshape(x, [-1]).eval()
      self.assertGreater(len(actual), len(expected))
      for i in xrange(len(actual)):
        a = actual[i]
        e = expected[i] if i < len(expected) else expected[-1]
        self.assertEqual(a, e)

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
    tf.reset_default_graph()
    with self.test_session(use_gpu=True):
      init = tf.constant_initializer(value, dtype=tf.int32)
      self.assertRaises(
          ValueError, tf.get_variable, "x", shape=shape, initializer=init)

  def testNDimConstantInitializerMoreValues(self):
    value = [0, 1, 2, 3, 4, 5, 6, 7]
    shape = [2, 3]
    self._testNDimConstantInitializerMoreValues(value, shape)
    self._testNDimConstantInitializerMoreValues(np.asarray(value), shape)
    self._testNDimConstantInitializerMoreValues(
        np.asarray(value).reshape(tuple([2, 4])), shape)


class RandomNormalInitializationTest(tf.test.TestCase):

  def testInitializerIdentical(self):
    for dtype in [tf.float32, tf.float64]:
      init1 = tf.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      init2 = tf.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))

  def testInitializerDifferent(self):
    for dtype in [tf.float32, tf.float64]:
      init1 = tf.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      init2 = tf.random_normal_initializer(0.0, 1.0, seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))

  def testDuplicatedInitializer(self):
    init = tf.random_normal_initializer(0.0, 1.0)
    self.assertFalse(duplicated_initializer(self, init, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError, tf.random_normal_initializer, 0.0, 1.0, dtype=tf.string)


class TruncatedNormalInitializationTest(tf.test.TestCase):

  def testInitializerIdentical(self):
    for dtype in [tf.float32, tf.float64]:
      init1 = tf.truncated_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      init2 = tf.truncated_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))

  def testInitializerDifferent(self):
    for dtype in [tf.float32, tf.float64]:
      init1 = tf.truncated_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      init2 = tf.truncated_normal_initializer(0.0, 1.0, seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))

  def testDuplicatedInitializer(self):
    init = tf.truncated_normal_initializer(0.0, 1.0)
    self.assertFalse(duplicated_initializer(self, init, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError, tf.truncated_normal_initializer, 0.0, 1.0, dtype=tf.string)


class RandomUniformInitializationTest(tf.test.TestCase):

  def testInitializerIdentical(self):
    for dtype in [tf.float32, tf.float64, tf.int64]:
      init1 = tf.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
      init2 = tf.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))

  def testInitializerDifferent(self):
    for dtype in [tf.float32, tf.float64, tf.int32, tf.int64]:
      init1 = tf.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
      init2 = tf.random_uniform_initializer(0, 7, seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))

  def testDuplicatedInitializer(self):
    init = tf.random_uniform_initializer(0.0, 1.0)
    self.assertFalse(duplicated_initializer(self, init, 1))


class UniformUnitScalingInitializationTest(tf.test.TestCase):

  def testInitializerIdentical(self):
    for dtype in [tf.float32, tf.float64]:
      init1 = tf.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
      init2 = tf.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))
      init3 = tf.uniform_unit_scaling_initializer(1.5, seed=1, dtype=dtype)
      init4 = tf.uniform_unit_scaling_initializer(1.5, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init3, init4))

  def testInitializerDifferent(self):
    for dtype in [tf.float32, tf.float64]:
      init1 = tf.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
      init2 = tf.uniform_unit_scaling_initializer(seed=2, dtype=dtype)
      init3 = tf.uniform_unit_scaling_initializer(1.5, seed=1, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))
      self.assertFalse(identicaltest(self, init1, init3))
      self.assertFalse(identicaltest(self, init2, init3))

  def testZeroSize(self):
    shape = [0, 2]
    with self.test_session():
      x = tf.get_variable(
          "x", shape=shape, initializer=tf.uniform_unit_scaling_initializer())
      self.assertAllEqual(shape, x.eval().shape)

  def testDuplicatedInitializer(self):
    init = tf.uniform_unit_scaling_initializer()
    self.assertFalse(duplicated_initializer(self, init, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError, tf.uniform_unit_scaling_initializer, dtype=tf.string)


class RandomWalkShapeTest(tf.test.TestCase):

  def testRandomWalk(self):
    # Fully known shape.
    rnd1 = init_ops._random_walk([1, 2], tf.nn.relu)
    self.assertEqual([1, 2], rnd1.get_shape())


# TODO(vrv): move to sequence_ops_test?
class RangeTest(tf.test.TestCase):

  def _Range(self, start, limit, delta):
    with self.test_session(use_gpu=True):
      tf_ans = tf.range(start, limit, delta, name="range")
      self.assertEqual([len(np.arange(start, limit, delta))],
                       tf_ans.get_shape())
      return tf_ans.eval()

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
    self.assertEqual(tf.range(0, 5, 1).dtype, tf.int32)

  def testLimitOnly(self):
    with self.test_session(use_gpu=True):
      self.assertAllEqual(np.arange(5), tf.range(5).eval())

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
    self.assertEqual(tf.range(0., 5., 1.).dtype, tf.float32)

  def testNegativeDelta(self):
    self.assertTrue(
        np.array_equal(self._Range(5, -1, -1), np.array([5, 4, 3, 2, 1, 0])))
    self.assertTrue(
        np.allclose(
            self._Range(2.5, 0, -0.5), np.array([2.5, 2, 1.5, 1, 0.5])))
    self.assertTrue(
        np.array_equal(self._Range(-5, -10, -3), np.array([-5, -8])))

  def testDType(self):
    zero_int32 = tf.cast(0, tf.int32)
    zero_int64 = tf.cast(0, tf.int64)
    zero_float32 = tf.cast(0, tf.float32)
    zero_float64 = tf.cast(0, tf.float64)

    self.assertEqual(tf.range(zero_int32, 0, 1).dtype, tf.int32)
    self.assertEqual(tf.range(zero_int64, 0, 1).dtype, tf.int64)
    self.assertEqual(tf.range(zero_float32, 0, 1).dtype, tf.float32)
    self.assertEqual(tf.range(zero_float64, 0, 1).dtype, tf.float64)

    self.assertEqual(tf.range(zero_int32, zero_int64, 1).dtype, tf.int64)
    self.assertEqual(tf.range(zero_int64, zero_float32, 1).dtype, tf.float32)
    self.assertEqual(tf.range(zero_float32, zero_float64, 1).dtype, tf.float64)
    self.assertEqual(tf.range(zero_float64, zero_int32, 1).dtype, tf.float64)

    self.assertEqual(tf.range(0, 0, 1, dtype=tf.int32).dtype, tf.int32)
    self.assertEqual(tf.range(0, 0, 1, dtype=tf.int64).dtype, tf.int64)
    self.assertEqual(tf.range(0, 0, 1, dtype=tf.float32).dtype, tf.float32)
    self.assertEqual(tf.range(0, 0, 1, dtype=tf.float64).dtype, tf.float64)


# TODO(vrv): move to sequence_ops_test?
class LinSpaceTest(tf.test.TestCase):

  def _gpu_modes(self):
    if tf.test.is_gpu_available():
      return [False, True]
    else:
      return [False]

  def _LinSpace(self, start, stop, num):
    # NOTE(touts): Needs to pass a graph to get a new session each time.
    with tf.Graph().as_default() as graph:
      with self.test_session(graph=graph, force_gpu=self.force_gpu):
        tf_ans = tf.linspace(start, stop, num, name="linspace")
        self.assertEqual([num], tf_ans.get_shape())
        return tf_ans.eval()

  def testPositive(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(1., 5., 1), np.array([1.]), 1e-5)
      self.assertArrayNear(self._LinSpace(1., 5., 2), np.array([1., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 3), np.array([1., 3., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 4),
          np.array([1., 7. / 3., 11. / 3., 5.]), 1e-5)

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


class DeviceTest(tf.test.TestCase):

  def testNoDevice(self):
    with tf.Graph().as_default():
      var = tf.Variable([[1.0, 1.0]])
    self.assertDeviceEqual(None, var.device)
    self.assertDeviceEqual(None, var.initializer.device)

  def testDevice(self):
    with tf.Graph().as_default():
      with tf.device("/job:ps"):
        var = tf.Variable([[1.0, 1.0]])
    self.assertDeviceEqual("/job:ps", var.device)
    self.assertDeviceEqual("/job:ps", var.initializer.device)


class OrthogonalInitializerTest(tf.test.TestCase):

  def testInitializerIdentical(self):
    for dtype in [tf.float32, tf.float64]:
      init1 = tf.orthogonal_initializer(seed=1, dtype=dtype)
      init2 = tf.orthogonal_initializer(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (10, 10)))

  def testInitializerDifferent(self):
    for dtype in [tf.float32, tf.float64]:
      init1 = tf.orthogonal_initializer(seed=1, dtype=dtype)
      init2 = tf.orthogonal_initializer(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (10, 10)))

  def testDuplicatedInitializer(self):
    init = tf.orthogonal_initializer()
    self.assertFalse(duplicated_initializer(self, init, 1, (10, 10)))

  def testInvalidDataType(self):
    self.assertRaises(ValueError, tf.orthogonal_initializer, dtype=tf.string)

  def testInvalidShape(self):
    init1 = tf.orthogonal_initializer()
    with self.test_session(graph=tf.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[5])

  def testGain(self):
    shape = (10, 10)
    for dtype in [tf.float32, tf.float64]:
      init1 = tf.orthogonal_initializer(seed=1, dtype=dtype)
      init2 = tf.orthogonal_initializer(gain=3.14, seed=1, dtype=dtype)
      with self.test_session(graph=tf.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
      with self.test_session(graph=tf.Graph(), use_gpu=True):
        t2 = init2(shape).eval()
      return np.allclose(t1, t2 / 3.14, rtol=1e-15, atol=1e-15)

  def testShapesValues(self):
    for dtype in [tf.float32, tf.float64]:
      for shape in [(10, 10), (10, 9, 8), (100, 5, 5), (50, 40), (40, 50)]:
        init = tf.orthogonal_initializer(dtype=dtype)
        tol = 1e-5 if dtype == tf.float32 else 1e-12
        with self.test_session(graph=tf.Graph(), use_gpu=True):
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


if __name__ == "__main__":
  tf.test.main()
