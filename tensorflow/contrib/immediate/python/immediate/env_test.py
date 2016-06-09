"""Test of basic immediate Env functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.immediate.python.immediate import test_util

class EnvTest(test_util.ImmediateTestCase):

  def testInit(self):
    with self.test_env(tf) as _unused_env:
      self.assertTrue(True)

  def testNN(self):
    with self.test_env(tf) as env:
      val = env.numpy_to_itensor(-1)
      self.assertEqual(env.tf.nn.relu(val), env.numpy_to_itensor(0))

  def testNumpyConversion(self):
    def testForDtype(dtype):
      a = np.array([[1, 2], [3, 4]], dtype=dtype)
      itensor_handle = env.numpy_to_handle(a)
      b = env.handle_to_numpy(itensor_handle)
      self.assertAllEqual(a, b)

    with self.test_env(tf) as env:
      testForDtype(np.float32)
      testForDtype(np.float64)
      testForDtype(np.int32)
      testForDtype(np.int64)

  def testNumpySingleton(self):
    def testForDtype(dtype):
      a = np.array(1, dtype=dtype)
      itensor_handle = env.numpy_to_handle(a)
      b = env.handle_to_numpy(itensor_handle)
      self.assertAllEqual(a, b)

    with self.test_env(tf) as env:
      testForDtype(np.float32)
      testForDtype(np.float64)
      testForDtype(np.int32)
      testForDtype(np.int64)

  def testReshapeOpWithConversion(self):
    """Try reshape op where arguments are implicitly converted to Tensors"""

    with self.test_env(tf) as env:
      val1 = env.numpy_to_itensor([[1], [2]])
      val2 = env.tf.reshape(val1, [-1])
      self.assertAllEqual(val2.as_numpy(), [1, 2])

  def testNumpyBoolConversion(self):
    with self.test_env(tf) as env:
      itensor = env.numpy_to_itensor(False)
      self.assertEqual(itensor, False)

  def testAdd(self):
    with self.test_env(tf) as env:
      val = np.ones(())
      tensor1 = env.numpy_to_itensor(val)
      tensor2 = env.numpy_to_itensor(val)
      tensor3 = env.tf.add(tensor1, tensor2)
      tensor4 = env.tf.add(tensor3, tensor2)
      self.assertAllEqual(tensor4.as_numpy(), 3*val)

  def testSub(self):
    with self.test_env(tf) as env:
      val = np.ones(())
      tensor1 = env.numpy_to_itensor(val)
      tensor2 = env.numpy_to_itensor(val)
      tensor3 = env.tf.sub(tensor1, tensor2)
      tensor4 = env.tf.sub(tensor3, tensor2)
      self.assertAllEqual(tensor4.as_numpy(), -1*val)

  def testPowOp(self):
    """Try a simple non-native op."""

    with self.test_env(tf) as env:
      val1 = env.numpy_to_itensor(2)
      val2 = env.numpy_to_itensor(3)
      self.assertEqual(env.tf.pow(val1, val2), env.numpy_to_itensor(8))

  def testOnes(self):
    with self.test_env(tf) as env:
      val1 = env.tf.ones(shape=(3, 3))
      self.assertAllEqual(val1.as_numpy(), np.ones((3, 3)))

  def testRank(self):
    with self.test_env(tf) as env:
      val1 = env.numpy_to_itensor([[1], [2]])
      self.assertEqual(env.tf.rank(val1), 2)

  def testRange(self):
    with self.test_env(tf) as env:
      val = env.tf.range(3)
      self.assertAllEqual(val.as_numpy(), [0, 1, 2])

  def testReduceSum(self):
    """Try a simple non-native op."""
    with self.test_env(tf) as env:
      val1 = env.numpy_to_itensor([1, 2, 3])
      self.assertEqual(env.tf.reduce_sum(val1), 6)

  def testConstant(self):
    with self.test_env(tf) as env:
      val1 = env.constant(1.5, shape=[2, 2])
      self.assertAllEqual(val1.as_numpy(), [[1.5, 1.5], [1.5, 1.5]])

      val2 = env.constant([1, 2, 3, 4])
      self.assertAllEqual(val2.as_numpy(), [1, 2, 3, 4])

      val3 = env.constant(7, dtype=tf.int32)
      self.assertAllEqual(val3.as_numpy(), 7)

  def testRandomUniform(self):
    with self.test_env(tf) as env:
      n = 3
      val = env.tf.random_uniform([n, n], -2, 2)
      sum_ = env.tf.reduce_sum(val)
      self.assertTrue(sum_ < n*n*2+1.)

  def testShape(self):
    with self.test_env(tf) as env:
      val0 = env.numpy_to_itensor([[1, 2, 3], [4, 5, 6]])
      self.assertAllEqual(env.tf.shape(val0).as_numpy(), [2, 3])

  def testSplit(self):
    with self.test_env(tf) as env:
      value = env.tf.ones((1, 3))
      split0, _unused_split1, _unused_split2 = env.tf.split(1, 3, value)
      self.assertAllEqual(env.tf.shape(split0).as_numpy(), [1, 1])
      split0, split1 = env.tf.split(0, 2, env.numpy_to_itensor([1, 2, 3, 4]))
      self.assertAllEqual(split0.as_numpy(), [1, 2])
      self.assertAllEqual(split1.as_numpy(), [3, 4])
      split0, split1 = env.tf.split(0, 2, env.numpy_to_itensor([1, 2]))
      self.assertAllEqual(split0.as_numpy(), [1])
      self.assertAllEqual(split1.as_numpy(), [2])

  def testConcat(self):
    with self.test_env(tf) as env:
      self.assertEqual(env.tf.concat(0, 5), 5)
      val1 = env.numpy_to_itensor([1, 2])
      val2 = env.numpy_to_itensor([3, 4])
      val3 = env.tf.concat(0, [val1, val2])
      self.assertAllEqual(val3.as_numpy(), [1, 2, 3, 4])

  def testGetItem(self):
    with self.test_env(tf) as env:
      val0 = env.numpy_to_itensor([1, 2, 3])
      self.assertEqual(val0[0], 1)
      self.assertEqual(val0[1], 2)
      self.assertEqual(val0[2], 3)


if __name__ == "__main__":
  tf.test.main()
