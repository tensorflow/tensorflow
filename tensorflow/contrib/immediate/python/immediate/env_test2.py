# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate


class EnvTest(tf.test.TestCase):
  def testReduceSum(self):
    """Try a simple non-native op."""
    env = immediate.Env(tf)
    val1 = env.numpy_to_tensor([1,2,3])
    self.assertEqual(env.tf.reduce_sum(val1), 6)


if __name__ == "__main__":
  tf.test.main()
