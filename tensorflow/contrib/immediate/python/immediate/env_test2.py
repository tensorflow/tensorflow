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
      x_dim = 40
      y_dim = 30
      keep_prob = 0.5
      t = env.tf.constant(1.0, shape=[x_dim, y_dim], dtype=tf.float32)
      env.tf.nn.dropout(t, keep_prob, noise_shape=[x_dim, y_dim + 10])
      
if __name__ == "__main__":
  tf.test.main()
