# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class EnvTest(tf.test.TestCase):

  def testInit(self):
    env = immediate.Env(tf)
    self.assertTrue(True)

  def testNamespacePassthrough(self):
    # test to make sure that non-ops pass-through the namespace without
    # being wrapped
    env = immediate.Env(tf)
    from tensorflow.python.framework import ops

    namespace = immediate.Namespace(env, "ops", ops, tf_root=False)
    self.assertEqual(ops._convert_stack, namespace._convert_stack)


  def testNN(self):
    env = immediate.Env(tf)
    val = env.numpy_to_tensor(-1)
    self.assertEqual(env.tf.nn.relu(val), env.numpy_to_tensor(0))

  def testNumpyConversion(self):
    def testForDtype(dtype):
      a = np.array([[1,2],[3,4]], dtype=dtype)
      tensor_handle = env.numpy_to_handle(a)
      b = env.handle_to_numpy(tensor_handle)
      self.assertAllEqual(a, b)

    env = immediate.Env(tf)
    testForDtype(np.float32)
    testForDtype(np.float64)
    testForDtype(np.int32)
    testForDtype(np.int64)

  def testNumpySingleton(self):
    def testForDtype(dtype):
      a = np.array(1, dtype=dtype)
      tensor_handle = env.numpy_to_handle(a)
      b = env.handle_to_numpy(tensor_handle)
      self.assertAllEqual(a, b)

    env = immediate.Env(tf)
    testForDtype(np.float32)
    testForDtype(np.float64)
    testForDtype(np.int32)
    testForDtype(np.int64)

  def testNumpyBoolConversion(self):
    env = immediate.Env(tf)
    tensor = env.numpy_to_tensor(False)

  def testAdd(self):
    env = immediate.Env(tf)
    val = np.ones(())
    tensor1 = env.numpy_to_tensor(val)
    tensor2 = env.numpy_to_tensor(val)
    tensor3 = env.tf.add(tensor1, tensor2)
    tensor4 = env.tf.add(tensor3, tensor2)
    self.assertAllEqual(tensor4.as_numpy(), 3*val)

  def testSub(self):
    env = immediate.Env(tf)
    val = np.ones(())
    tensor1 = env.numpy_to_tensor(val)
    tensor2 = env.numpy_to_tensor(val)
    tensor3 = env.tf.sub(tensor1, tensor2)
    tensor4 = env.tf.sub(tensor3, tensor2)
    self.assertAllEqual(tensor4.as_numpy(), -1*val)

  def testPowOp(self):
    """Try a simple non-native op."""
    env = immediate.Env(tf)
    val1 = env.numpy_to_tensor(2)
    val2 = env.numpy_to_tensor(3)
    self.assertEqual(env.tf.pow(val1, val2), env.numpy_to_tensor(8))

  # TODO(yaroslavvb): test tf.ones for empty shape (ndarray.empty)
  def testOnes(self):
fill(shape, constant(1, dtype=dtype), name=name)
    env = immediate.Env(tf)
    val1 = env.tf.ones((3, 3))
    self.assertAllEqual(val1.as_numpy(), np.ones((3, 3)))

  def testReshapeOpWithConversion(self):
    """Try reshape op where arguments are implicitly converted to Tensors"""
    env = immediate.Env(tf)
    val1 = env.numpy_to_tensor([[1],[2]])
    val2 = env.tf.reshape(val1, [-1])
    # TODO(yaroslavvb): implement slicing and get rid of numpy conversion
    self.assertAllEqual(env.tensor_to_numpy(val2), [1, 2])

  def testRank(self):
    env = immediate.Env(tf)
    val1 = env.numpy_to_tensor([[1],[2]])
    self.assertEqual(env.tf.rank(val1), 2)

  def testRange(self):
    env = immediate.Env(tf)
    val = env.tf.range(3)
    self.assertAllEqual(env.tensor_to_numpy(val), [0, 1, 2])
    
  def testReduceSum(self):
    """Try a simple non-native op."""
    env = immediate.Env(tf)
    val1 = env.numpy_to_tensor([1,2,3])
    self.assertEqual(env.tf.reduce_sum(val1), 6)

  def testConstant(self):
    env = immediate.Env(tf)
    val1 = env.constant(1.5, shape=[2, 2])
    self.assertAllEqual(val1.as_numpy(), [[1.5, 1.5], [1.5, 1.5]])

    val2 = env.constant([1, 2, 3, 4])
    self.assertAllEqual(val2.as_numpy(), [1, 2, 3, 4])

    val3 = env.constant(7, dtype=tf.int32)
    self.assertAllEqual(val3.as_numpy(), 7)

  def testRandomUniform(self):
    env = immediate.Env(tf)
    n = 3
    val = env.tf.random_uniform([n, n], -2, 2)
    sum_ = env.tf.reduce_sum(val)
    self.assertTrue(sum_ < n*n*2+1.)
    

  def testShape(self):
    env = immediate.Env(tf)
    val0 = env.numpy_to_tensor([[1,2,3],[4,5,6]])
    self.assertAllEqual(env.tf.shape(val0).as_numpy(), [2, 3])
        

  def testAddCaching(self):
    # make sure that graph is not modified in a loop
    env = immediate.Env(tf)
    val = np.ones(())
    tensor0 = env.numpy_to_tensor(val)
    tensor1 = env.numpy_to_tensor(np.zeros(()))

    # the first loop needs to be long enough to trigger tensor
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
