# Small set of broken tests for env (to be moved into env_test.py after they
# work )

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

from tensorflow.contrib.immediate.python.immediate import test_util

from tensorflow.python.framework import ops as ops

import threading

class TinyEnvTest(test_util.ImmediateTestCase):

  def testAdd(self):
    with self.test_env(tf) as env:
      val1 = env.numpy_to_tensor(1)
      val2 = env.numpy_to_tensor(2)
      val3 = env.tf.add(val1, val2)
      result = val3.as_numpy()
      self.assertEqual(result, 3)

      
if __name__ == "__main__":
  tf.test.main()
