# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class EnvTest(tf.test.TestCase):

  def testInit(self):
    env = immediate.Env(tf)
    self.assertTrue(True)

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
    #fill(shape, constant(1, dtype=dtype), name=name)
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
        
  def testSplit(self):
    env = immediate.Env(tf)
    value = env.tf.ones((1, 3))
    split0, split1, split2 = env.tf.split(1, 3, value)
    self.assertAllEqual(env.tf.shape(split0).as_numpy(), [1, 1])
    split0, split1 = env.tf.split(0, 2, env.numpy_to_tensor([1, 2, 3, 4]))
    self.assertAllEqual(split1.as_numpy(), [3, 4])
    split0, split1 = env.tf.split(0, 2, env.numpy_to_tensor([1, 2]))
    self.assertAllEqual(split0.as_numpy(), [1])
    

  def testConcat(self):
    env = immediate.Env(tf)
    val0 = env.numpy_to_tensor(0)
    # test special degenerate case
    self.assertEqual(env.tf.concat(0, 5), 5)

    val1 = env.numpy_to_tensor([1,2])
    val2 = env.numpy_to_tensor([3,4])
    val3 = env.tf.concat(0, [val1, val2]) 
    self.assertAllEqual(val3.as_numpy(), [1,2,3,4])


## TODO(yaroslavvb): move to env_test2
#### Tests below are for development/checking the Graph Caching system
#### They rely on private details of session_ops implementation
#### (such as, how often deletion graphs are created).
#### As such they should be removed after Graph Caching is shown to work
#### reliably


  def atestAddCaching(self):
    # make sure that graph is not modified in a loop
    env = immediate.Env(tf)
    val = np.ones(())
    tensor0 = env.numpy_to_tensor(val)
    tensor1 = env.numpy_to_tensor(np.zeros(()))

    # the first loop needs to be long enough to trigger tensor
    # garbage collection since that modifies the graph
    for i in range(20):
      tensor1+=tensor0

    number_of_graph_modifications = env.graph_version

    for i in range(10):
      tensor1+=tensor0

    # check that graph hasn't been modified by checking its
    # graph version
    self.assertEqual(number_of_graph_modifications, env.graph_version)
    self.assertEqual(tensor1.as_numpy(), 30)

  def atestSplitCaching(self):
    # TODO(yaroslavvb): remove to_split_value1/2 when numpy conversion
    # ops are properly cached
    env = immediate.Env(tf)
    value = env.tf.ones((1, 3))
    to_split_value1 = env.numpy_to_tensor([1, 2])
    to_split_value2 = env.numpy_to_tensor([1, 2, 3])
    split0, split1, split2 = env.tf.split(1, 3, value)
    self.assertAllEqual(env.tf.shape(split0).as_numpy(), [1, 1])
    split0, split1 = env.tf.split(0, 2, env.numpy_to_tensor([1, 2, 3, 4]))
    version1 = env.graph_version
    split0, split1 = env.tf.split(0, 2, to_split_value1)
    version2 = env.graph_version
    splits = env.tf.split(0, 3, to_split_value2)
    version3 = env.graph_version
    self.assertEqual(version1, version2)
    self.assertNotEqual(version2, version3)

    

if __name__ == "__main__":
  tf.test.main()
