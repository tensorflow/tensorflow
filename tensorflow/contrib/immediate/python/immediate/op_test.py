# Tests for immediate.Op

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class OpTest(tf.test.TestCase):

  def testOpWrapper(self):
    env = immediate.Env(tf)
    val = np.ones((), dtype=np.float32)
    #tensor1 = env.numpy_to_tensor(val)
    #tensor2 = env.numpy_to_tensor(val)
    tensor1 = tf.constant(1)
    tensor2 = tf.constant(2)
    def my_add(x, y):
      return tf.constant(42)

    op_wrapper = immediate.OpWrapper(env, my_add)
    print op_wrapper(x=tensor1, y=tensor2)
    
if __name__ == "__main__":
  tf.test.main()
