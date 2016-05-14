# Tests for immediate.Op

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class OpTest(tf.test.TestCase):

  def testOpFactory(self):
    env = immediate.Env(tf)
    op_factory = env.op_factory

    val = np.ones((), dtype=np.float32)
    tensor1 = env.numpy_to_tensor(val)
    tensor2 = env.numpy_to_tensor(val)
    op = op_factory("add", tf.add, tensor1, tensor2)
    self.assertEqual(str(op), "Op('add', tf.float32, tf.float32)")
    tensor3 = op(tensor1, tensor2)
    self.assertEqual(tensor3.as_numpy(), 2.)

  def testOpWrapper(self):
    env = immediate.Env(tf)
    val = np.ones((), dtype=np.float32)
    tensor1 = env.numpy_to_tensor(val)
    tensor2 = env.numpy_to_tensor(val)
    op_wrapper = immediate.OpWrapper(None, env, "add", tf.add)
    print op_wrapper(tensor1, tensor2)
    
if __name__ == "__main__":
  tf.test.main()
