# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class EnvTest(tf.test.TestCase):

  def atestInit(self):
    env = immediate.Env()
    self.assertTrue(True)

  def atestNumpyConversion(self):
    def testForDtype(dtype):
      a = np.array([[1,2],[3,4]], dtype=dtype)
      tensor_handle = env.numpy_to_handle(a)
      b = env.handle_to_numpy(tensor_handle)
      self.assertAllEqual(a, b)

    env = immediate.Env()
    testForDtype(np.float32)
    testForDtype(np.float64)
    testForDtype(np.int32)
    testForDtype(np.int64)

  def atestAdd(self):
    env = immediate.Env()
    val = np.ones(())
    tensor1 = immediate.Tensor.numpy_to_tensor(env, val)
    tensor2 = immediate.Tensor.numpy_to_tensor(env, val)
    tensor3 = env.add(tensor1, tensor2)
    tensor4 = env.add(tensor3, tensor2)
    self.assertAllEqual(tensor4.as_numpy(), 3*val)

  def atestSub(self):
    env = immediate.Env()
    val = np.ones(())
    tensor1 = immediate.Tensor.numpy_to_tensor(env, val)
    tensor2 = immediate.Tensor.numpy_to_tensor(env, val)
    tensor3 = env.sub(tensor1, tensor2)
    tensor4 = env.sub(tensor3, tensor2)
    self.assertAllEqual(tensor4.as_numpy(), -1*val)


  def testAddCaching(self):
    # make sure that graph is not modified in a loop
    env = immediate.Env()
    val = np.ones(())
    tensor0 = immediate.Tensor.numpy_to_tensor(env, val)
    tensor1 = immediate.Tensor.numpy_to_tensor(env, np.zeros(()))

    # the first loop has to be long enough to trigger tensor
    # garbage collection since that modifies the graph
    for i in range(20):
      tensor1+=tensor0

    number_of_graph_modifications = env.g.version

    for i in range(10):
      tensor1+=tensor0

    # check that graph hasn't been modified by checking its
    # graph version
    self.assertEqual(number_of_graph_modifications, env.g.version)
    self.assertEqual(tensor1.as_numpy(), 30)

if __name__ == "__main__":
  tf.test.main()
