# Small set of broken tests for env (to be moved into env_test.py after they
# work )

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

from tensorflow.contrib.immediate.python.immediate import test_util

from tensorflow.python.framework import ops as ops

import threading

class EnvTest2(test_util.ImmediateTestCase):

  def testAdd(self):
    with self.test_env(tf) as env:
      self.assertEqual(env.tf.add(1, 2), 3)

      
if __name__ == "__main__":
  tf.test.main()
