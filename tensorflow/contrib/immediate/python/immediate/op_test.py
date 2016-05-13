# Tests for immediate.Op

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class OpTest(tf.test.TestCase):

  def testOpFactory(self):
    env = immediate.Env()
    op_factory = env.op_factory

    val = np.ones((), dtype=np.float32)
    tensor1 = immediate.Tensor.numpy_to_tensor(env, val)
    tensor2 = immediate.Tensor.numpy_to_tensor(env, val)
    op = op_factory("add", tensor1, tensor2)
    self.assertEqual(str(op), "Op('add', tf.float32, tf.float32)")
    tensor3 = op(tensor1, tensor2)
    self.assertEqual(tensor3.as_numpy(), 2.)

  def testOpWrapper(self):
    env = immediate.Env()
    val = np.ones((), dtype=np.float32)
    tensor1 = immediate.Tensor.numpy_to_tensor(env, val)
    tensor2 = immediate.Tensor.numpy_to_tensor(env, val)
    op_wrapper = immediate.OpWrapper(env, "add")
    print op_wrapper(tensor1, tensor2)
    
if __name__ == "__main__":
  tf.test.main()
