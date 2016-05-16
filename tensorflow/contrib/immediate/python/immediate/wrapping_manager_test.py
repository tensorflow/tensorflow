import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

class WrappingManagerTest(tf.test.TestCase):

  def testBasic(self):
    env = immediate.Env(tf)
    wrapping_manager = immediate.WrappingManager(env, tf)
    print wrapping_manager.wrapped_modules

  def atestConvertToTensor(self):
    env = immediate.Env(tf)
    wrapping_manager = immediate.WrappingManager(env, tf)

    from tensorflow.python.framework import ops
    wrapping_manager.wrap_module(ops)
    module = wrapping_manager["tensorflow/python/framework/ops.py"]
    print module.convert_to_tensor([[1,2],[3,4]])

  def atestConvertNToTensor(self):
    """Test for a wrapped function that calls another wrapped function."""
    env = immediate.Env(tf)
    wrapping_manager = immediate.WrappingManager(env, tf)
    module = wrapping_manager.get_module["tensorflow/python/framework/ops.py"]
    print module.convert_n_to_tensor([[1,2],[3,4]])

  def atestAdd(self):
    env = immediate.Env(tf)
    wrapping_manager = immediate.WrappingManager(env, tf)
    module = wrapping_manager["tensorflow/python/ops/gen_math_ops.py"]
    val1 = env.numpy_to_tensor([1,2])
    val2 = env.numpy_to_tensor([1,2])
    val3 = module.add(val1, val2)
    print val3

  
  def atestSlice(self):
    env = immediate.Env(tf)
    wrapping_manager = immediate.WrappingManager(env, tf)
    module = wrapping_manager["tensorflow/python/ops/gen_array_ops.py"]
    val = env.numpy_to_tensor([1,2])
    start = env.numpy_to_tensor([0])
    size = env.numpy_to_tensor([1])
    print module._slice(val, start, size)
    
  # TODO(yaroslavvb): allow handling list as input
  def atestPack(self):
    env = immediate.Env(tf)
    wrapping_manager = immediate.WrappingManager(env, tf)
    module = wrapping_manager["tensorflow/python/ops/gen_array_ops.py"]
    val1 = env.convert_to_tensor([1,2,3])
    val2 = env.convert_to_tensor([1,2,3])
    print module._pack([val1, val2], name=name)


if __name__ == "__main__":
  tf.test.main()
