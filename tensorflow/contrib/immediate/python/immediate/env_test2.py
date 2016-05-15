# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class EnvTest(tf.test.TestCase):

  """ Call ones with empty shape. That calls constant(1) with default dtype
  (float32), 
  """
  def testOnes(self):
    env = immediate.Env(tf)
    val1 = env.tf.ones((3, 3))
    self.assertAllEqual(val1.as_numpy(), np.ones((3, 3)))

#  def testRandomUniform(self):
#    env = immediate.Env(tf)
#    val = env.tf.random_uniform([3, 3], -2, 2)
#    print val

if __name__ == "__main__":
  tf.test.main()
