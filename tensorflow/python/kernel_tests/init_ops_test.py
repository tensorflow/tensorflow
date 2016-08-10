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
import tensorflow as tf

from tensorflow.python.framework import random_seed
from tensorflow.python.ops import init_ops


# Returns true iff the two initializers produce the same tensor to
# within a tiny tolerance.
def identicaltest(tc, init1, init2, use_gpu):
  """Tests if two initializations are identical to within tiny tolerances.

  Args:
    tc: An instance of TensorFlowTestCase.
    init1: An Initializer that generates a tensor of a given shape
    init2: An Initializer that generates a tensor of a given shape
    use_gpu: Use gpu if true.
  Returns:
    True or False as determined by test.
  """
  num = 100
  with tc.test_session(use_gpu=use_gpu, graph=tf.Graph()):
    t1 = init1([num]).eval()
  with tc.test_session(use_gpu=use_gpu, graph=tf.Graph()):
    t2 = init2([num]).eval()
  return np.allclose(t1, t2, rtol=1e-15, atol=1e-15)


def duplicated_initializer(tc, init, use_gpu, graph_seed):
  """Tests duplicated random initializer within the same graph.

  This test generates two random kernels from the same initializer to the same
  graph, and checks if the results are close enough. Even given the same global,
  seed, two different instances of random kernels should generate different
  results.

  Args:
    tc: An instance of TensorFlowTestCase.
    init: An Initializer that generates a tensor of a given shape
    use_gpu: Use gpu if true.
    graph_seed: A graph-level seed to use.
  Returns:
    True or False as determined by test.
  """
  num = 100
  with tc.test_session(use_gpu=use_gpu, graph=tf.Graph()):
    random_seed.set_random_seed(graph_seed)
    t1 = init([num]).eval()
    t2 = init([num]).eval()
    return np.allclose(t1, t2, rtol=1e-15, atol=1e-15)


def _init_sampler(tc, init, num, use_gpu):
  """Returns a func to generate a random tensor of shape [num].

  Args:
    tc: An instance of TensorFlowTestCase.
    init: An Initializer that generates a tensor of a given shape
    num: Size of 1D tensor to create.
    use_gpu: Use gpu if true.
  Returns:
    Function to generate a random tensor.
  """
  def func():
    with tc.test_session(use_gpu=use_gpu):
      return init([num]).eval()
  return func


class ConstantInitializersTest(tf.test.TestCase):

  def testZerosInitializer(self):
    with self.test_session():
      shape = [2, 3]
      x = tf.get_variable("x", shape=shape, initializer=tf.zeros_initializer)
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.zeros(shape))

  def testOnesInitializer(self):
    with self.test_session():
      shape = [2, 3]
      x = tf.get_variable("x", shape=shape, initializer=tf.ones_initializer)
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.ones(shape))

  def testConstantZeroInitializer(self):
    with self.test_session():
      shape = [2, 3]
      x = tf.get_variable("x", shape=shape,
                          initializer=tf.constant_initializer(0.0))
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.zeros(shape))

  def testConstantOneInitializer(self):
    with self.test_session():
      shape = [2, 3]
      x = tf.get_variable("x", shape=shape,
                          initializer=tf.constant_initializer(1.0))
      x.initializer.run()
      self.assertAllEqual(x.eval(), np.ones(shape))


class RandomNormalInitializationTest(tf.test.TestCase):

  def testInitializerIdentical(self):
    for use_gpu in [False, True]:
      for dtype in [tf.float32, tf.float64]:
        init1 = tf.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
        init2 = tf.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
        self.assertTrue(identicaltest(self, init1, init2, use_gpu))

  def testInitializerDifferent(self):
    for use_gpu in [False, True]:
      for dtype in [tf.float32, tf.float64]:
        init1 = tf.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
        init2 = tf.random_normal_initializer(0.0, 1.0, seed=2, dtype=dtype)
        self.assertFalse(identicaltest(self, init1, init2, use_gpu=use_gpu))

  def testDuplicatedInitializer(self):
    for use_gpu in [False, True]:
      init = tf.random_normal_initializer(0.0, 1.0)
      self.assertFalse(duplicated_initializer(self, init, use_gpu, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError,
        tf.random_normal_initializer, 0.0, 1.0, dtype=tf.string)


class TruncatedNormalInitializationTest(tf.test.TestCase):

  def testInitializerIdentical(self):
    for use_gpu in [False, True]:
      for dtype in [tf.float32, tf.float64]:
        init1 = tf.truncated_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
        init2 = tf.truncated_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
        self.assertTrue(identicaltest(self, init1, init2, use_gpu))

  def testInitializerDifferent(self):
    for use_gpu in [False, True]:
      for dtype in [tf.float32, tf.float64]:
        init1 = tf.truncated_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
        init2 = tf.truncated_normal_initializer(0.0, 1.0, seed=2, dtype=dtype)
        self.assertFalse(identicaltest(self, init1, init2, use_gpu=use_gpu))

  def testDuplicatedInitializer(self):
    for use_gpu in [False, True]:
      init = tf.truncated_normal_initializer(0.0, 1.0)
      self.assertFalse(duplicated_initializer(self, init, use_gpu, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError,
        tf.truncated_normal_initializer, 0.0, 1.0, dtype=tf.string)


class RandomUniformInitializationTest(tf.test.TestCase):

  def testInitializerIdentical(self):
    for use_gpu in [False, True]:
      for dtype in [tf.float32, tf.float64]:
        init1 = tf.random_uniform_initializer(0.0, 1.0, seed=1, dtype=dtype)
        init2 = tf.random_uniform_initializer(0.0, 1.0, seed=1, dtype=dtype)
        self.assertTrue(identicaltest(self, init1, init2, use_gpu))

  def testInitializerDifferent(self):
    for use_gpu in [False, True]:
      for dtype in [tf.float32, tf.float64]:
        init1 = tf.random_uniform_initializer(0.0, 1.0, seed=1, dtype=dtype)
        init2 = tf.random_uniform_initializer(0.0, 1.0, seed=2, dtype=dtype)
        self.assertFalse(identicaltest(self, init1, init2, use_gpu))

  def testDuplicatedInitializer(self):
    for use_gpu in [False, True]:
      init = tf.random_uniform_initializer(0.0, 1.0)
      self.assertFalse(duplicated_initializer(self, init, use_gpu, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError,
        tf.random_uniform_initializer, 0.0, 1.0, dtype=tf.string)


class UniformUnitScalingInitializationTest(tf.test.TestCase):

  def testInitializerIdentical(self):
    for use_gpu in [False, True]:
      for dtype in [tf.float32, tf.float64]:
        init1 = tf.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
        init2 = tf.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
        self.assertTrue(identicaltest(self, init1, init2, use_gpu))
        init3 = tf.uniform_unit_scaling_initializer(1.5, seed=1, dtype=dtype)
        init4 = tf.uniform_unit_scaling_initializer(1.5, seed=1, dtype=dtype)
        self.assertTrue(identicaltest(self, init3, init4, use_gpu))

  def testInitializerDifferent(self):
    for use_gpu in [False, True]:
      for dtype in [tf.float32, tf.float64]:
        init1 = tf.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
        init2 = tf.uniform_unit_scaling_initializer(seed=2, dtype=dtype)
        init3 = tf.uniform_unit_scaling_initializer(1.5, seed=1, dtype=dtype)
        self.assertFalse(identicaltest(self, init1, init2, use_gpu))
        self.assertFalse(identicaltest(self, init1, init3, use_gpu))
        self.assertFalse(identicaltest(self, init2, init3, use_gpu))

  def testDuplicatedInitializer(self):
    for use_gpu in [False, True]:
      init = tf.uniform_unit_scaling_initializer()
      self.assertFalse(duplicated_initializer(self, init, use_gpu, 1))

  def testInvalidDataType(self):
    self.assertRaises(
        ValueError,
        tf.uniform_unit_scaling_initializer, dtype=tf.string)


class RandomWalkShapeTest(tf.test.TestCase):

  def testRandomWalk(self):
    # Fully known shape.
    rnd1 = init_ops._random_walk([1, 2], tf.nn.relu)
    self.assertEqual([1, 2], rnd1.get_shape())


# TODO(vrv): move to sequence_ops_test?
class RangeTest(tf.test.TestCase):

  def _Range(self, start, limit, delta):
    with self.test_session():
      tf_ans = tf.range(start, limit, delta, name="range")
      self.assertEqual([len(range(start, limit, delta))], tf_ans.get_shape())
      return tf_ans.eval()

  def testBasic(self):
    self.assertTrue(np.array_equal(
        self._Range(0, 5, 1), np.array([0, 1, 2, 3, 4])))
    self.assertTrue(np.array_equal(
        self._Range(0, 5, 2), np.array([0, 2, 4])))
    self.assertTrue(np.array_equal(
        self._Range(0, 6, 2), np.array([0, 2, 4])))
    self.assertTrue(np.array_equal(
        self._Range(13, 32, 7), np.array([13, 20, 27])))
    self.assertTrue(np.array_equal(
        self._Range(100, 500, 100), np.array([100, 200, 300, 400])))
    self.assertEqual(tf.range(0, 5, 1).dtype, tf.int32)

  def testLimitOnly(self):
    with self.test_session():
      self.assertAllEqual(np.arange(5), tf.range(5).eval())

  def testEmpty(self):
    for start in 0, 5:
      self.assertTrue(np.array_equal(self._Range(start, start, 1), []))


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
      self.assertArrayNear(self._LinSpace(1., 5., 3),
                           np.array([1., 3., 5.]), 1e-5)
      self.assertArrayNear(self._LinSpace(1., 5., 4),
                           np.array([1., 7. / 3., 11. / 3., 5.]), 1e-5)

  def testNegative(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., -5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(self._LinSpace(-1., -5., 2),
                           np.array([-1., -5.]), 1e-5)
      self.assertArrayNear(self._LinSpace(-1., -5., 3),
                           np.array([-1., -3., -5.]), 1e-5)
      self.assertArrayNear(self._LinSpace(-1., -5., 4),
                           np.array([-1., -7. / 3., -11. / 3., -5.]), 1e-5)

  def testNegativeToPositive(self):
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., 5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(self._LinSpace(-1., 5., 2), np.array([-1., 5.]),
                           1e-5)
      self.assertArrayNear(self._LinSpace(-1., 5., 3),
                           np.array([-1., 2., 5.]), 1e-5)
      self.assertArrayNear(self._LinSpace(-1., 5., 4),
                           np.array([-1., 1., 3., 5.]), 1e-5)

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


if __name__ == "__main__":
  tf.test.main()
