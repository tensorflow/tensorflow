# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class EnvTest(tf.test.TestCase):


  def testRandomUniform(self):
    env = immediate.Env(tf)
    val = env.tf.random_uniform([3, 3], -2, 2)
    print val

if __name__ == "__main__":
  tf.test.main()
