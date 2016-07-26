"""Functional tests for immediate Graph caching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import immediate
from tensorflow.contrib.immediate.python.immediate import test_util

import tensorflow as tf

class EnvCacheTest(test_util.ImmediateTestCase):

  def testAddCacheCpu(self):
    env = immediate.Env(tf)
    is_graph_changed(env)
    env.disable_gc()
    with env.g.device("cpu:0"):
      val1 = env.numpy_to_itensor(1)
      val2 = env.numpy_to_itensor(2)
      self.assertTrue(is_graph_changed(env))
      val3 = val1 + val2
      self.assertTrue(is_graph_changed(env))
      _unused_val4 = val2 + val3
      self.assertFalse(is_graph_changed(env))

  def testAddCacheGpu(self):
    if not tf.test.is_built_with_cuda():
      return True
    self._assertHaveGpu0()

    env = immediate.Env(tf)
    is_graph_changed(env)
    env.disable_gc()

    val1 = env.numpy_to_itensor(1)
    val2 = env.numpy_to_itensor(2)
    with env.g.device("gpu:0"):
      # move tensors onto GPU
      val1 = env.tf.identity(val1)
      val2 = env.tf.identity(val2)
      val3 = val1 + val2
      self.assertTrue(is_graph_changed(env))
      val2 + val3
      self.assertFalse(is_graph_changed(env))

  def testAddCacheMixed(self):
    if not tf.test.is_built_with_cuda():
      return True
    self._assertHaveGpu0()

    env = immediate.Env(tf)
    is_graph_changed(env)
    env.disable_gc()

    val1 = env.numpy_to_itensor(1)
    val2 = env.numpy_to_itensor(2)
    with env.g.device("cpu:0"):
      cpu_val1 = env.tf.identity(val1)
      cpu_val2 = env.tf.identity(val2)

    with env.g.device("gpu:0"):
      gpu_val1 = env.tf.identity(val1)
      gpu_val2 = env.tf.identity(val2)

    cpu_plus_gpu1 = cpu_val1 + gpu_val1
    self.assertTrue(is_graph_changed(env))

    gpu_plus_cpu1 = gpu_val1 + cpu_val1
    self.assertTrue(is_graph_changed(env))

    cpu_plus_cpu1 = cpu_val1 + cpu_val1
    self.assertTrue(is_graph_changed(env))

    gpu_plus_gpu1 = gpu_val1 + gpu_val1
    self.assertTrue(is_graph_changed(env))

    cpu_plus_gpu2 = cpu_val2 + gpu_val2
    self.assertFalse(is_graph_changed(env))

    gpu_plus_cpu2 = gpu_val2 + cpu_val2
    self.assertFalse(is_graph_changed(env))

    cpu_plus_cpu2 = cpu_val2 + cpu_val2
    self.assertFalse(is_graph_changed(env))

    gpu_plus_gpu2 = gpu_val2 + gpu_val2
    self.assertFalse(is_graph_changed(env))

    self.assertEqual(cpu_plus_gpu1, 2)
    self.assertEqual(gpu_plus_cpu1, 2)
    self.assertEqual(cpu_plus_cpu1, 2)
    self.assertEqual(gpu_plus_gpu1, 2)
    self.assertEqual(cpu_plus_gpu2, 4)
    self.assertEqual(gpu_plus_cpu2, 4)
    self.assertEqual(cpu_plus_cpu2, 4)
    self.assertEqual(gpu_plus_gpu2, 4)

  def testConcatCache(self):
    env = immediate.Env(tf)
    is_graph_changed(env)
    env.disable_gc()

    val1 = env.numpy_to_itensor([1, 2])
    val2 = env.numpy_to_itensor([3, 4])
    concat_dim = env.numpy_to_itensor(0)

    with env.g.device("cpu:0"):
      cpu_val1 = env.tf.identity(val1)
      cpu_val2 = env.tf.identity(val2)

    val1 = env.tf.concat(concat_dim, [cpu_val1, cpu_val1])
    self.assertTrue(is_graph_changed(env))
    val2 = env.tf.concat(concat_dim, [cpu_val2, cpu_val2])
    self.assertFalse(is_graph_changed(env))

    if not tf.test.is_built_with_cuda():
      return True
    self._assertHaveGpu0()

    with env.g.device("gpu:0"):
      gpu_val1 = env.tf.identity(val1)
      gpu_val2 = env.tf.identity(val2)

    env.tf.concat(concat_dim, [cpu_val1, gpu_val1])
    self.assertTrue(is_graph_changed(env))
    env.tf.concat(concat_dim, [cpu_val2, gpu_val2])
    self.assertFalse(is_graph_changed(env))
    env.tf.concat(concat_dim, [gpu_val1, cpu_val1])
    self.assertTrue(is_graph_changed(env))
    env.tf.concat(concat_dim, [gpu_val2, cpu_val2])
    self.assertFalse(is_graph_changed(env))

  def testSplitCache(self):
    env = immediate.Env(tf)
    env.disable_gc()

    env.tf.split(1, 3, env.tf.ones((1, 3)))
    self.assertTrue(is_graph_changed(env))
    env.tf.split(1, 3, env.tf.ones((1, 9)))
    self.assertFalse(is_graph_changed(env))

  def testNumpyToTensorCache(self):
    env = immediate.Env(tf)
    is_graph_changed(env)
    env.disable_gc()

    env.numpy_to_itensor(1)
    self.assertTrue(is_graph_changed(env))
    env.numpy_to_itensor(2)
    self.assertFalse(is_graph_changed(env))

    with env.device("cpu:0"):
      env.numpy_to_itensor(3)
      self.assertTrue(is_graph_changed(env))
      env.numpy_to_itensor(4)
      self.assertFalse(is_graph_changed(env))

    if not tf.test.is_built_with_cuda():
      return True
    self._assertHaveGpu0()

    with env.device("gpu:0"):
      env.numpy_to_itensor(5)
      self.assertTrue(is_graph_changed(env))
      env.numpy_to_itensor(6)
      self.assertFalse(is_graph_changed(env))

  def testTensorToNumpyCache(self):
    env = immediate.Env(tf)
    is_graph_changed(env)
    env.disable_gc()

    val1 = env.numpy_to_itensor(1)
    val2 = env.numpy_to_itensor(2)
    with env.device("cpu:0"):
      cpu_val1 = env.tf.identity(val1)
      cpu_val2 = env.tf.identity(val2)

    env.itensor_to_numpy(cpu_val1)
    self.assertTrue(is_graph_changed(env))
    env.itensor_to_numpy(cpu_val2)
    self.assertFalse(is_graph_changed(env))

    if not tf.test.is_built_with_cuda():
      return True
    self._assertHaveGpu0()

    with env.device("gpu:0"):
      gpu_val1 = env.tf.identity(val1)
      gpu_val2 = env.tf.identity(val2)

    env.itensor_to_numpy(gpu_val1)
    self.assertTrue(is_graph_changed(env))
    env.itensor_to_numpy(gpu_val2)
    self.assertFalse(is_graph_changed(env))

    with env.device("gpu:0"):
      env.itensor_to_numpy(gpu_val1)
      self.assertTrue(is_graph_changed(env))
      env.itensor_to_numpy(gpu_val2)
      self.assertFalse(is_graph_changed(env))


_cached_graph_version = 0
def is_graph_changed(env):
  global _cached_graph_version
  is_changed = (env._graph_version != _cached_graph_version)
  _cached_graph_version = env._graph_version
  return is_changed


if __name__ == "__main__":
  tf.test.main()
