import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

from tensorflow.contrib.immediate.python.immediate import test_util

from tensorflow.python.framework import ops as ops

import threading

# TODO(yaroslavvb): make these tests compatible with non-GPU machines
# by testing if GPU is available

def print_gdef_diff(gdef1, gdef2):
  print("GraphDef difference")
  print("-"*80)
  dict1 = {node.name: node for node in gdef1.node}
  dict2 = {node.name: node for node in gdef2.node}
  names1 = set(dict1.keys())
  names2 = set(dict2.keys())
  if names1 == names2:
    return
  for name in sorted(names2.difference(names1)):
    print dict2[name]


_cached_graph_version = 0
def _is_graph_changed(env):
  global _cached_graph_version
  is_changed = (env._graph_version != _cached_graph_version)
  _cached_graph_version = env._graph_version
  return is_changed

class ExtraEnvTest(test_util.ImmediateTestCase):

  def testAddCacheCpu(self):
    env = immediate.Env(tf)
    env.disable_gc()
    with env.g.device("cpu:0"):
      val1 = env.numpy_to_itensor(1)
      val2 = env.numpy_to_itensor(2)
      self.assertTrue(_is_graph_changed(env))
      val3 = val1 + val2
      self.assertTrue(_is_graph_changed(env))
      val4 = val2 + val3
      self.assertFalse(_is_graph_changed(env))

  def testAddCacheGpu(self):
    env = immediate.Env(tf)
    env.disable_gc()
    val1 = env.numpy_to_itensor(1)
    val2 = env.numpy_to_itensor(2)
    with env.g.device("gpu:0"):
      # move tensors onto GPU
      val1 = env.tf.identity(val1)
      val2 = env.tf.identity(val2)
      val3 = val1 + val2
      self.assertTrue(_is_graph_changed(env))
      val4 = val2 + val3
      self.assertFalse(_is_graph_changed(env))

  def testAddCacheMixed(self):
    env = immediate.Env(tf)
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
    self.assertTrue(_is_graph_changed(env))

    gpu_plus_cpu1 = gpu_val1 + cpu_val1
    self.assertTrue(_is_graph_changed(env))

    cpu_plus_cpu1 = cpu_val1 + cpu_val1
    self.assertTrue(_is_graph_changed(env))

    gpu_plus_gpu1 = gpu_val1 + gpu_val1
    self.assertTrue(_is_graph_changed(env))

    cpu_plus_gpu2 = cpu_val2 + gpu_val2
    self.assertFalse(_is_graph_changed(env))

    gpu_plus_cpu2 = gpu_val2 + cpu_val2
    self.assertFalse(_is_graph_changed(env))

    cpu_plus_cpu2 = cpu_val2 + cpu_val2
    self.assertFalse(_is_graph_changed(env))

    gpu_plus_gpu2 = gpu_val2 + gpu_val2
    self.assertFalse(_is_graph_changed(env))

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
    env.disable_gc()

    val1 = env.numpy_to_itensor([1, 2])
    val2 = env.numpy_to_itensor([3, 4])
    concat_dim = env.numpy_to_itensor(0)
    
    with env.g.device("cpu:0"):
      cpu_val1 = env.tf.identity(val1)
      cpu_val2 = env.tf.identity(val2)

    with env.g.device("gpu:0"):
      gpu_val1 = env.tf.identity(val1)
      gpu_val2 = env.tf.identity(val2)

    val3 = env.tf.concat(concat_dim, [cpu_val1, gpu_val1])
    self.assertTrue(_is_graph_changed(env))
    val4 = env.tf.concat(concat_dim, [cpu_val2, gpu_val2])
    self.assertFalse(_is_graph_changed(env))
    val5 = env.tf.concat(concat_dim, [gpu_val1, cpu_val1])
    self.assertTrue(_is_graph_changed(env))
    val6 = env.tf.concat(concat_dim, [gpu_val2, cpu_val2])
    self.assertFalse(_is_graph_changed(env))

  def testNumpyToTensorCache(self):
    env = immediate.Env(tf)
    env.disable_gc()

    val1 = env.numpy_to_itensor(1)
    self.assertTrue(_is_graph_changed(env))
    val2 = env.numpy_to_itensor(2)
    self.assertFalse(_is_graph_changed(env))
    
    with env.device("cpu:0"):
      val1 = env.numpy_to_itensor(3)
      self.assertTrue(_is_graph_changed(env))
      val2 = env.numpy_to_itensor(4)
      self.assertFalse(_is_graph_changed(env))

    with env.device("gpu:0"):
      val1 = env.numpy_to_itensor(5)
      self.assertTrue(_is_graph_changed(env))
      val2 = env.numpy_to_itensor(6)
      self.assertFalse(_is_graph_changed(env))

  def testTensorToNumpyCache(self):
    env = immediate.Env(tf)
    env.disable_gc()

    val1 = env.numpy_to_itensor(1)
    val2 = env.numpy_to_itensor(2)
    with env.device("cpu:0"):
      cpu_val1 = env.tf.identity(val1)
      cpu_val2 = env.tf.identity(val2)
    with env.device("gpu:0"):
      gpu_val1 = env.tf.identity(val1)
      gpu_val2 = env.tf.identity(val2)

    np_val1 = env.itensor_to_numpy(cpu_val1)
    self.assertTrue(_is_graph_changed(env))
    np_val2 = env.itensor_to_numpy(cpu_val2)
    self.assertFalse(_is_graph_changed(env))
    np_val3 = env.itensor_to_numpy(gpu_val1)
    self.assertTrue(_is_graph_changed(env))
    np_val4 = env.itensor_to_numpy(gpu_val2)
    self.assertFalse(_is_graph_changed(env))

    with env.device("gpu:0"):
      np_val5 = env.itensor_to_numpy(gpu_val1)
      self.assertTrue(_is_graph_changed(env))
      np_val5 = env.itensor_to_numpy(gpu_val2)
      self.assertFalse(_is_graph_changed(env))
    
if __name__ == "__main__":
  tf.test.main()
