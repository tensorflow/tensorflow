"""Tests for tensorflow.ops.random_ops."""

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class RandomNormalTest(tf.test.TestCase):

  def _Sampler(self, num, mu, sigma, dtype, use_gpu, seed=None):
    def func():
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
        rng = tf.random_normal(
            [num], mean=mu, stddev=sigma, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in xrange(10):
          ret[i, :] = sess.run(rng)
      return ret
    return func

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    for use_gpu in [False, True]:
      for dt in tf.float32, tf.float64:
        sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu)
        x = sampler()
        y = sampler()
        # Number of different samples.
        count = (x == y).sum()
        if count >= 10:
          print "x = ", x
          print "y = ", y
          print "count = ", count
        self.assertTrue(count < 10)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  def testCPUGPUMatch(self):
    for dt in tf.float32, tf.float64:
      results = {}
      for use_gpu in [False, True]:
        sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      self.assertAllClose(results[False], results[True], rtol=1e-6, atol=1e-6)

  def testSeed(self):
    for use_gpu in [False, True]:
      for dt in tf.float32, tf.float64:
        sx = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=345)
        sy = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=345)
        self.assertAllEqual(sx(), sy())

  def testNoCSE(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        shape = [2, 3, 4]
        rnd1 = tf.random_normal(shape, 0.0, 1.0, tf.float32)
        rnd2 = tf.random_normal(shape, 0.0, 1.0, tf.float32)
        diff = rnd2 - rnd1
        self.assertTrue(np.linalg.norm(diff.eval()) > 0.1)


class TruncatedNormalTest(tf.test.TestCase):

  def _Sampler(self, num, mu, sigma, dtype, use_gpu, seed=None):
    def func():
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
        rng = tf.truncated_normal(
            [num], mean=mu, stddev=sigma, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in xrange(10):
          ret[i, :] = sess.run(rng)
      return ret
    return func

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    # NOTE: RandomParameters on GPU is not supported.
    for use_gpu in [False]:
      for dt in tf.float32, tf.float64:
        sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu)
        x = sampler()
        y = sampler()
        # Number of different samples.
        count = (x == y).sum()
        if count >= 10:
          print "x = ", x
          print "y = ", y
          print "count = ", count
        self.assertTrue(count < 10)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  def testCPUGPUMatch(self):
    for dt in tf.float32, tf.float64:
      results = {}
      for use_gpu in [False, True]:
        # We need a particular larger number of samples to test multiple rounds
        # on GPU
        sampler = self._Sampler(1000000, 0.0, 1.0, dt, use_gpu=use_gpu,
                                seed=12345)
        results[use_gpu] = sampler()
      self.assertAllClose(results[False], results[True], rtol=1e-6, atol=1e-6)

  def testSeed(self):
    for use_gpu in [False, True]:
      for dt in tf.float32, tf.float64:
        sx = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=345)
        sy = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=345)
        self.assertAllEqual(sx(), sy())

  # The effective standard deviation of truncated normal is 85% of the
  # requested one.
  def testStdDev(self):
    for use_gpu in [False, True]:
      for dt in tf.float32, tf.float64:
        stddev = 3.0
        sampler = self._Sampler(100000, 0.0, stddev, dt, use_gpu=use_gpu)
        x = sampler()
        print "std(x)", np.std(x), abs(np.std(x) / stddev - 0.85)
        self.assertTrue(abs(np.std(x) / stddev - 0.85) < 0.04)

  def testNoCSE(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        shape = [2, 3, 4]
        rnd1 = tf.truncated_normal(shape, 0.0, 1.0, tf.float32)
        rnd2 = tf.truncated_normal(shape, 0.0, 1.0, tf.float32)
        diff = rnd2 - rnd1
        self.assertTrue(np.linalg.norm(diff.eval()) > 0.1)


class RandomUniformTest(tf.test.TestCase):

  def _Sampler(self, num, minv, maxv, dtype, use_gpu, seed=None):
    def func():
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
        rng = tf.random_uniform(
            [num], minval=minv, maxval=maxv, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in xrange(10):
          ret[i, :] = sess.run(rng)
      return ret
    return func

  def testRange(self):
    for use_gpu in [False, True]:
      for dt in tf.float32, tf.float64:
        sampler = self._Sampler(1000, -2., 8., dt, use_gpu=use_gpu)
        x = sampler()
        self.assertTrue(-2 <= np.min(x))
        self.assertTrue(np.max(x) <= 8)

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    for use_gpu in [False, True]:
      for dt in tf.float32, tf.float64:
        sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu)
        x = sampler()
        y = sampler()
        count = (x == y).sum()
        if count >= 10:
          print "x = ", x
          print "y = ", y
          print "count = ", count
        self.assertTrue(count < 10)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  def testCPUGPUMatch(self):
    for dt in tf.float32, tf.float64:
      results = {}
      for use_gpu in [False, True]:
        sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      self.assertAllClose(results[False], results[True], rtol=1e-6, atol=1e-6)

  def testSeed(self):
    for use_gpu in [False, True]:
      for dt in tf.float32, tf.float64:
        sx = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=345)
        sy = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=345)
        self.assertAllEqual(sx(), sy())

  def testNoCSE(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        shape = [2, 3, 4]
        rnd1 = tf.random_uniform(shape, 0.0, 1.0,
                                         dtype=tf.float32)
        rnd2 = tf.random_uniform(shape, 0.0, 1.0,
                                         dtype=tf.float32)
        diff = (rnd2 - rnd1).eval()
        self.assertTrue(np.linalg.norm(diff) > 0.1)


class RandomShapeTest(tf.test.TestCase):

  def testRandomParameters(self):
    # Fully known shape.
    rnd1 = tf.truncated_normal([1, 2, 3])
    self.assertEqual([1, 2, 3], rnd1.get_shape())
    # Partially known shape.
    rnd2 = tf.truncated_normal(tf.placeholder(tf.int32, shape=(3,)))
    self.assertEqual([None, None, None], rnd2.get_shape().as_list())
    # Unknown shape.
    rnd3 = tf.truncated_normal(tf.placeholder(tf.int32))
    self.assertIs(None, rnd3.get_shape().ndims)

  def testRandomNormal(self):
    # Fully known shape.
    rnd1 = tf.random_normal([1, 2, 3])
    self.assertEqual([1, 2, 3], rnd1.get_shape())
    # Partially known shape.
    rnd2 = tf.random_normal(tf.placeholder(tf.int32, shape=(3,)))
    self.assertEqual([None, None, None], rnd2.get_shape().as_list())
    # Unknown shape.
    rnd3 = tf.random_normal(tf.placeholder(tf.int32))
    self.assertIs(None, rnd3.get_shape().ndims)

  def testRandomUniform(self):
    # Fully known shape.
    rnd1 = tf.random_uniform([1, 2, 3])
    self.assertEqual([1, 2, 3], rnd1.get_shape())
    # Partially known shape.
    rnd2 = tf.random_uniform(
        tf.placeholder(tf.int32, shape=(3,)))
    self.assertEqual([None, None, None], rnd2.get_shape().as_list())
    # Unknown shape.
    rnd3 = tf.random_uniform(tf.placeholder(tf.int32))
    self.assertIs(None, rnd3.get_shape().ndims)


if __name__ == "__main__":
  tf.test.main()
