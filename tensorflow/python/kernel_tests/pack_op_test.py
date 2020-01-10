"""Functional tests for Pack Op."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker


class PackOpTest(tf.test.TestCase):

  def testSimple(self):
    np.random.seed(7)
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
          data = np.random.randn(*shape)
          # Convert [data[0], data[1], ...] separately to tensorflow
          xs = map(tf.constant, data)
          # Pack back into a single tensorflow tensor
          c = tf.pack(xs)
          self.assertAllEqual(c.eval(), data)

  def testGradients(self):
    np.random.seed(7)
    for use_gpu in False, True:
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        data = np.random.randn(*shape)
        shapes = [shape[1:]] * shape[0]
        with self.test_session(use_gpu=use_gpu):
          xs = map(tf.constant, data)
          c = tf.pack(xs)
          err = gradient_checker.ComputeGradientError(xs, shapes, c, shape)
          self.assertLess(err, 1e-6)

  def testZeroSize(self):
    # Verify that pack doesn't crash for zero size inputs
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        for shape in (0,), (3,0), (0, 3):
          x = np.zeros((2,) + shape)
          p = tf.pack(list(x)).eval()
          self.assertAllEqual(p, x)


if __name__ == "__main__":
  tf.test.main()
