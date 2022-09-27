# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for initializers in init_ops_v2."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class InitializersTest(test.TestCase):

  def _identical_test(self,
                      init1,
                      init2,
                      assertion,
                      shape=None,
                      dtype=dtypes.float32):
    if shape is None:
      shape = [100]
    t1 = self.evaluate(init1(shape, dtype))
    t2 = self.evaluate(init2(shape, dtype))
    self.assertEqual(tensor_shape.as_shape(shape), t1.shape)
    self.assertEqual(tensor_shape.as_shape(shape), t2.shape)
    self.assertEqual(assertion, np.allclose(t1, t2, rtol=1e-15, atol=1e-15))

  def _duplicated_test(self, init, shape=None, dtype=dtypes.float32):
    if shape is None:
      shape = [100]
    t1 = self.evaluate(init(shape, dtype))
    t2 = self.evaluate(init(shape, dtype))
    self.assertEqual(tensor_shape.as_shape(shape), t1.shape)
    self.assertEqual(tensor_shape.as_shape(shape), t2.shape)
    self.assertFalse(np.allclose(t1, t2, rtol=1e-15, atol=1e-15))

  def _range_test(self,
                  init,
                  shape,
                  target_mean=None,
                  target_std=None,
                  target_max=None,
                  target_min=None):
    output = self.evaluate(init(shape))
    self.assertEqual(output.shape, shape)
    lim = 3e-2
    if target_std is not None:
      self.assertGreater(lim, abs(output.std() - target_std))
    if target_mean is not None:
      self.assertGreater(lim, abs(output.mean() - target_mean))
    if target_max is not None:
      self.assertGreater(lim, abs(output.max() - target_max))
    if target_min is not None:
      self.assertGreater(lim, abs(output.min() - target_min))

  def _partition_test(self, init):
    full_shape = (4, 2)
    partition_shape = (2, 2)
    partition_offset = (0, 0)
    full_value = self.evaluate(init(full_shape, dtype=dtypes.float32))
    got = self.evaluate(
        init(
            full_shape,
            dtype=dtypes.float32,
            partition_shape=partition_shape,
            partition_offset=partition_offset))
    self.assertEqual(got.shape, partition_shape)
    self.assertAllClose(
        got, array_ops.slice(full_value, partition_offset, partition_shape))


class ConstantInitializersTest(InitializersTest):

  @test_util.run_in_graph_and_eager_modes
  def testZeros(self):
    self._range_test(
        init_ops_v2.Zeros(), shape=(4, 5), target_mean=0., target_max=0.)

  @test_util.run_in_graph_and_eager_modes
  def testZerosPartition(self):
    init = init_ops_v2.Zeros()
    self._partition_test(init)

  @test_util.run_in_graph_and_eager_modes
  def testZerosInvalidKwargs(self):
    init = init_ops_v2.Zeros()
    with self.assertRaisesRegex(
        TypeError, r"Keyword argument should be one of .* Received: dtpye"):
      init((2, 2), dtpye=dtypes.float32)

  @test_util.run_in_graph_and_eager_modes
  def testOnes(self):
    self._range_test(
        init_ops_v2.Ones(), shape=(4, 5), target_mean=1., target_max=1.)

  @test_util.run_in_graph_and_eager_modes
  def testOnesPartition(self):
    init = init_ops_v2.Ones()
    self._partition_test(init)

  @test_util.run_in_graph_and_eager_modes
  def testConstantInt(self):
    self._range_test(
        init_ops_v2.Constant(2),
        shape=(5, 6, 4),
        target_mean=2,
        target_max=2,
        target_min=2)

  @test_util.run_in_graph_and_eager_modes
  def testConstantPartition(self):
    init = init_ops_v2.Constant([1, 2, 3, 4])
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        r"Constant initializer doesn't support partition-related arguments"):
      init((4, 2), dtype=dtypes.float32, partition_shape=(2, 2))

  @test_util.run_in_graph_and_eager_modes
  def testConstantTuple(self):
    init = init_ops_v2.constant_initializer((10, 20, 30))
    tensor = init(shape=[3])
    self.assertAllEqual(self.evaluate(tensor), [10, 20, 30])
    self.assertEqual(tensor.shape, [3])

  @test_util.run_in_graph_and_eager_modes
  def testConstantInvalidValue(self):
    c = constant_op.constant([1.0, 2.0, 3.0])
    with self.assertRaisesRegex(TypeError,
                                r"Invalid type for initial value: .*Tensor.*"):
      init_ops_v2.constant_initializer(c)
    v = variables.Variable([3.0, 2.0, 1.0])
    with self.assertRaisesRegex(
        TypeError, r"Invalid type for initial value: .*Variable.*"):
      init_ops_v2.constant_initializer(v)

  def _testNDimConstantInitializer(self, value, shape, expected):
    with test_util.use_gpu():
      init = init_ops_v2.constant_initializer(value)
      x = init(shape)

      actual = self.evaluate(array_ops.reshape(x, [-1]))
      self.assertEqual(len(actual), len(expected))
      for a, e in zip(actual, expected):
        self.assertEqual(a, e)

  @test_util.run_in_graph_and_eager_modes
  def testNDimConstantInitializer(self):
    value = [0, 1, 2, 3, 4, 5]
    shape = [2, 3]
    expected = list(value)

    self._testNDimConstantInitializer(value, shape, expected)
    self._testNDimConstantInitializer(np.asarray(value), shape, expected)
    self._testNDimConstantInitializer(
        np.asarray(value).reshape(tuple(shape)), shape, expected)

  def _testNDimConstantInitializerIncorrectNumberValues(self, value, shape):
    with test_util.use_gpu():
      init = init_ops_v2.constant_initializer(value)
      self.assertRaises(TypeError, init, shape=shape)

  @test_util.run_in_graph_and_eager_modes
  def testNDimConstantInitializerIncorrectNumberValues(self):
    value = [0, 1, 2, 3, 4, 5]

    for shape in [[2, 4], [2, 2]]:
      self._testNDimConstantInitializerIncorrectNumberValues(value, shape)
      self._testNDimConstantInitializerIncorrectNumberValues(
          np.asarray(value), shape)
      self._testNDimConstantInitializerIncorrectNumberValues(
          np.asarray(value).reshape(tuple([2, 3])), shape)


class RandomUniformInitializerTest(InitializersTest):

  @test_util.run_in_graph_and_eager_modes
  def testRangeInitializer(self):
    shape = (20, 6, 7)
    self._range_test(
        init_ops_v2.RandomUniform(minval=-1, maxval=1, seed=124),
        shape,
        target_mean=0.,
        target_max=1,
        target_min=-1)

  @test_util.run_in_graph_and_eager_modes
  def testInitializerIdentical(self):
    self.skipTest("Doesn't work without the graphs")
    init1 = init_ops_v2.RandomUniform(0, 7, seed=1)
    init2 = init_ops_v2.RandomUniform(0, 7, seed=1)
    self._identical_test(init1, init2, True)

  @test_util.run_in_graph_and_eager_modes
  def testInitializerDifferent(self):
    init1 = init_ops_v2.RandomUniform(0, 7, seed=1)
    init2 = init_ops_v2.RandomUniform(0, 7, seed=2)
    self._identical_test(init1, init2, False)

  @test_util.run_in_graph_and_eager_modes
  def testDuplicatedInitializer(self):
    init = init_ops_v2.RandomUniform(0.0, 1.0)
    self._duplicated_test(init)

  @test_util.run_in_graph_and_eager_modes
  def testInitializePartition(self):
    init = init_ops_v2.RandomUniform(0, 7, seed=1)
    self._partition_test(init)


class RandomNormalInitializerTest(InitializersTest):

  @test_util.run_in_graph_and_eager_modes
  def testRangeInitializer(self):
    self._range_test(
        init_ops_v2.RandomNormal(mean=0, stddev=1, seed=153),
        shape=(8, 12, 99),
        target_mean=0.,
        target_std=1)

  @test_util.run_in_graph_and_eager_modes
  def testInitializerIdentical(self):
    self.skipTest("Doesn't work without the graphs")
    init1 = init_ops_v2.RandomNormal(0, 7, seed=1)
    init2 = init_ops_v2.RandomNormal(0, 7, seed=1)
    self._identical_test(init1, init2, True)

  @test_util.run_in_graph_and_eager_modes
  def testInitializerDifferent(self):
    init1 = init_ops_v2.RandomNormal(0, 7, seed=1)
    init2 = init_ops_v2.RandomNormal(0, 7, seed=2)
    self._identical_test(init1, init2, False)

  @test_util.run_in_graph_and_eager_modes
  def testDuplicatedInitializer(self):
    init = init_ops_v2.RandomNormal(0.0, 1.0)
    self._duplicated_test(init)

  @test_util.run_in_graph_and_eager_modes
  def testInitializePartition(self):
    if test_util.is_xla_enabled():
      self.skipTest(
          "XLA ignores seeds for RandomNormal, skip xla-enabled test.")
    init = init_ops_v2.RandomNormal(0, 7, seed=1)
    self._partition_test(init)


class TruncatedNormalInitializerTest(InitializersTest):

  @test_util.run_in_graph_and_eager_modes
  def testRangeInitializer(self):
    self._range_test(
        init_ops_v2.TruncatedNormal(mean=0, stddev=1, seed=126),
        shape=(12, 99, 7),
        target_mean=0.,
        target_max=2,
        target_min=-2)

  @test_util.run_in_graph_and_eager_modes
  def testInitializerIdentical(self):
    self.skipTest("Not seeming to work in Eager mode")
    init1 = init_ops_v2.TruncatedNormal(0.0, 1.0, seed=1)
    init2 = init_ops_v2.TruncatedNormal(0.0, 1.0, seed=1)
    self._identical_test(init1, init2, True)

  @test_util.run_in_graph_and_eager_modes
  def testInitializerDifferent(self):
    init1 = init_ops_v2.TruncatedNormal(0.0, 1.0, seed=1)
    init2 = init_ops_v2.TruncatedNormal(0.0, 1.0, seed=2)
    self._identical_test(init1, init2, False)

  @test_util.run_in_graph_and_eager_modes
  def testDuplicatedInitializer(self):
    init = init_ops_v2.TruncatedNormal(0.0, 1.0)
    self._duplicated_test(init)

  @test_util.run_in_graph_and_eager_modes
  def testInitializePartition(self):
    init = init_ops_v2.TruncatedNormal(0.0, 1.0, seed=1)
    self._partition_test(init)

  @test_util.run_in_graph_and_eager_modes
  def testInvalidDataType(self):
    init = init_ops_v2.TruncatedNormal(0.0, 1.0)
    with self.assertRaises(ValueError):
      init([1], dtype=dtypes.int32)


class VarianceScalingInitializerTest(InitializersTest):

  @test_util.run_in_graph_and_eager_modes
  def testTruncatedNormalDistribution(self):
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops_v2.VarianceScaling(distribution="truncated_normal")

    with test_util.use_gpu(), test.mock.patch.object(
        random_ops, "truncated_normal",
        wraps=random_ops.truncated_normal) as mock_truncated_normal:
      x = self.evaluate(init(shape))
      self.assertTrue(mock_truncated_normal.called)

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def testNormalDistribution(self):
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops_v2.VarianceScaling(distribution="truncated_normal")

    with test_util.use_gpu(), test.mock.patch.object(
        random_ops, "truncated_normal",
        wraps=random_ops.truncated_normal) as mock_truncated_normal:
      x = self.evaluate(init(shape))
      self.assertTrue(mock_truncated_normal.called)

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def testUntruncatedNormalDistribution(self):
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops_v2.VarianceScaling(distribution="untruncated_normal")

    with test_util.use_gpu(), test.mock.patch.object(
        random_ops, "random_normal",
        wraps=random_ops.random_normal) as mock_random_normal:
      x = self.evaluate(init(shape))
      self.assertTrue(mock_random_normal.called)

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def testUniformDistribution(self):
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops_v2.VarianceScaling(distribution="uniform")

    with test_util.use_gpu():
      x = self.evaluate(init(shape))

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def testInitializePartition(self):
    partition_shape = (100, 100)
    shape = [1000, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops_v2.VarianceScaling(distribution="untruncated_normal")

    with test_util.use_gpu(), test.mock.patch.object(
        random_ops, "random_normal",
        wraps=random_ops.random_normal) as mock_random_normal:
      x = self.evaluate(init(shape, partition_shape=partition_shape))
      self.assertTrue(mock_random_normal.called)

    self.assertEqual(x.shape, partition_shape)
    self.assertNear(np.mean(x), expect_mean, err=2e-3)
    self.assertNear(np.var(x), expect_var, err=2e-3)


class OrthogonalInitializerTest(InitializersTest):

  @test_util.run_in_graph_and_eager_modes
  def testRangeInitializer(self):
    self._range_test(
        init_ops_v2.Orthogonal(seed=123), shape=(20, 20), target_mean=0.)

  @test_util.run_in_graph_and_eager_modes
  def testInitializerIdentical(self):
    self.skipTest("Doesn't work without the graphs")
    init1 = init_ops_v2.Orthogonal(seed=1)
    init2 = init_ops_v2.Orthogonal(seed=1)
    self._identical_test(init1, init2, True, (10, 10))

  @test_util.run_in_graph_and_eager_modes
  def testInitializerDifferent(self):
    init1 = init_ops_v2.Orthogonal(seed=1)
    init2 = init_ops_v2.Orthogonal(seed=2)
    self._identical_test(init1, init2, False, (10, 10))

  @test_util.run_in_graph_and_eager_modes
  def testDuplicatedInitializer(self):
    init = init_ops_v2.Orthogonal()
    self._duplicated_test(init, (10, 10))

  @test_util.run_in_graph_and_eager_modes
  def testInvalidDataType(self):
    init = init_ops_v2.Orthogonal()
    self.assertRaises(ValueError, init, shape=(10, 10), dtype=dtypes.string)

  @test_util.run_in_graph_and_eager_modes
  def testInvalidShape(self):
    init = init_ops_v2.Orthogonal()
    with test_util.use_gpu():
      self.assertRaises(ValueError, init, shape=[5])

  @test_util.run_in_graph_and_eager_modes
  def testGain(self):
    self.skipTest("Doesn't work without the graphs")
    init1 = init_ops_v2.Orthogonal(seed=1)
    init2 = init_ops_v2.Orthogonal(gain=3.14, seed=1)
    with test_util.use_gpu():
      t1 = self.evaluate(init1(shape=(10, 10)))
      t2 = self.evaluate(init2(shape=(10, 10)))
    self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_in_graph_and_eager_modes
  def testShapesValues(self):
    for shape in [(10, 10), (10, 9, 8), (100, 5, 5), (50, 40), (40, 50)]:
      init = init_ops_v2.Orthogonal()
      tol = 1e-5
      with test_util.use_gpu():
        # Check the shape
        t = self.evaluate(init(shape))
        self.assertAllEqual(shape, t.shape)
        # Check orthogonality by computing the inner product
        t = t.reshape((np.prod(t.shape[:-1]), t.shape[-1]))
        if t.shape[0] > t.shape[1]:
          self.assertAllClose(
              np.dot(t.T, t), np.eye(t.shape[1]), rtol=tol, atol=tol)
        else:
          self.assertAllClose(
              np.dot(t, t.T), np.eye(t.shape[0]), rtol=tol, atol=tol)

  @test_util.run_in_graph_and_eager_modes
  def testPartition(self):
    init = init_ops_v2.Orthogonal(seed=1)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        r"Orthogonal initializer doesn't support partition-related arguments"):
      init((4, 2), dtype=dtypes.float32, partition_shape=(2, 2))


class IdentityInitializerTest(InitializersTest):

  @test_util.run_in_graph_and_eager_modes
  def testRange(self):
    with self.assertRaises(ValueError):
      shape = (3, 4, 5)
      self._range_test(
          init_ops_v2.Identity(),
          shape=shape,
          target_mean=1. / shape[0],
          target_max=1.)

    shape = (3, 3)
    self._range_test(
        init_ops_v2.Identity(),
        shape=shape,
        target_mean=1. / shape[0],
        target_max=1.)

  @test_util.run_in_graph_and_eager_modes
  def testInvalidDataType(self):
    init = init_ops_v2.Identity()
    self.assertRaises(ValueError, init, shape=[10, 5], dtype=dtypes.int32)

  @test_util.run_in_graph_and_eager_modes
  def testInvalidShape(self):
    init = init_ops_v2.Identity()
    with test_util.use_gpu():
      self.assertRaises(ValueError, init, shape=[5, 7, 7])
      self.assertRaises(ValueError, init, shape=[5])
      self.assertRaises(ValueError, init, shape=[])

  @test_util.run_in_graph_and_eager_modes
  def testNonSquare(self):
    init = init_ops_v2.Identity()
    shape = (10, 5)
    with test_util.use_gpu():
      self.assertAllClose(self.evaluate(init(shape)), np.eye(*shape))

  @test_util.run_in_graph_and_eager_modes
  def testGain(self):
    shape = (10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init_default = init_ops_v2.Identity()
      init_custom = init_ops_v2.Identity(gain=0.9)
      with test_util.use_gpu():
        self.assertAllClose(
            self.evaluate(init_default(shape, dtype=dtype)), np.eye(*shape))
      with test_util.use_gpu():
        self.assertAllClose(
            self.evaluate(init_custom(shape, dtype=dtype)),
            np.eye(*shape) * 0.9)

  @test_util.run_in_graph_and_eager_modes
  def testPartition(self):
    init = init_ops_v2.Identity()
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        r"Identity initializer doesn't support partition-related arguments"):
      init((4, 2), dtype=dtypes.float32, partition_shape=(2, 2))


class GlorotInitializersTest(InitializersTest):

  @test_util.run_in_graph_and_eager_modes
  def testGlorotUniform(self):
    shape = (5, 6, 4, 2)
    fan_in, fan_out = init_ops_v2._compute_fans(shape)
    std = np.sqrt(2. / (fan_in + fan_out))
    self._range_test(
        init_ops_v2.GlorotUniform(seed=123),
        shape,
        target_mean=0.,
        target_std=std)

  @test_util.run_in_graph_and_eager_modes
  def test_GlorotNormal(self):
    shape = (5, 6, 4, 2)
    fan_in, fan_out = init_ops_v2._compute_fans(shape)
    std = np.sqrt(2. / (fan_in + fan_out))
    self._range_test(
        init_ops_v2.GlorotNormal(seed=123),
        shape,
        target_mean=0.,
        target_std=std)


class MethodInitializers(InitializersTest):

  @test_util.run_in_graph_and_eager_modes
  def testLecunUniform(self):
    shape = (5, 6, 4, 2)
    fan_in, _ = init_ops_v2._compute_fans(shape)
    std = np.sqrt(1. / fan_in)
    self._range_test(
        init_ops_v2.lecun_uniform(seed=123),
        shape,
        target_mean=0.,
        target_std=std)

  @test_util.run_in_graph_and_eager_modes
  def testHeUniform(self):
    shape = (5, 6, 4, 2)
    fan_in, _ = init_ops_v2._compute_fans(shape)
    std = np.sqrt(2. / fan_in)
    self._range_test(
        init_ops_v2.he_uniform(seed=123), shape, target_mean=0., target_std=std)

  @test_util.run_in_graph_and_eager_modes
  def testLecunNormal(self):
    shape = (5, 6, 4, 2)
    fan_in, _ = init_ops_v2._compute_fans(shape)
    std = np.sqrt(1. / fan_in)
    self._range_test(
        init_ops_v2.lecun_normal(seed=123),
        shape,
        target_mean=0.,
        target_std=std)

  @test_util.run_in_graph_and_eager_modes
  def testHeNormal(self):
    shape = (5, 6, 4, 2)
    fan_in, _ = init_ops_v2._compute_fans(shape)
    std = np.sqrt(2. / fan_in)
    self._range_test(
        init_ops_v2.he_normal(seed=123), shape, target_mean=0., target_std=std)


if __name__ == "__main__":
  test.main()
