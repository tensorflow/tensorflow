"""Tests for tensorflow.kernels.unique_op."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class UniqueTest(tf.test.TestCase):

  def testInt32(self):
    x = list(np.random.randint(2, high=10, size=7000))
    with self.test_session() as sess:
      y, idx = tf.unique(x)
      tf_y, tf_idx = sess.run([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

if __name__ == "__main__":
  tf.test.main()
