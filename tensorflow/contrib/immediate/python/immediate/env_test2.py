# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

# TODO(yaroslavvb): overload identity in array_ops

class EnvTest(tf.test.TestCase):

  def atestConstant(self):
    env = immediate.Env(tf)
    val1 = env.constant(1.5, shape=[2, 2])
    self.assertAllEqual(val1.as_numpy(), [[1.5, 1.5], [1.5, 1.5]])

    val2 = env.constant([1, 2, 3, 4])
    self.assertAllEqual(val2.as_numpy(), [1, 2, 3, 4])


  def testSplit(self):
    env = immediate.Env(tf)
    value = env.tf.ones((1, 3))
    split0, split1, split2 = env.tf.split(1, 3, value)
    self.assertAllEqual(env.tf.shape(split0).as_numpy(), [1, 1])
    split0, split1 = env.tf.split(0, 2, env.numpy_to_tensor([1, 2, 3, 4]))
    self.assertAllEqual(split1.as_numpy(), [3, 4])
    

  def atestConcat(self):
    env = immediate.Env(tf)
    val0 = env.numpy_to_tensor(0)
    # test special degenerate case
    self.assertEqual(env.tf.concat(0, 5), 5)

    val1 = env.numpy_to_tensor([1,2])
    val2 = env.numpy_to_tensor([3,4])
    val3 = env.tf.concat(0, [val1, val2]) 
    print val3
   


#  def testRandomUniform(self):
#    env = immediate.Env(tf)
#    val = env.tf.random_uniform([3, 3], -2, 2)
#    print val

if __name__ == "__main__":
  tf.test.main()
