"""Functional tests for Unpack Op."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker


class UnpackOpTest(tf.test.TestCase):

  def testSimple(self):
    np.random.seed(7)
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
          data = np.random.randn(*shape)
          # Convert data to a single tensorflow tensor
          x = tf.constant(data)
          # Unpack into a list of tensors
          cs = tf.unpack(x, num=shape[0])
          self.assertEqual(type(cs), list)
          self.assertEqual(len(cs), shape[0])
          cs = [c.eval() for c in cs]
          self.assertAllEqual(cs, data)

  def testGradients(self):
    for use_gpu in False, True:
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        data = np.random.randn(*shape)
        shapes = [shape[1:]] * shape[0]
        for i in xrange(shape[0]):
          with self.test_session(use_gpu=use_gpu):
            x = tf.constant(data)
            cs = tf.unpack(x, num=shape[0])
            err = gradient_checker.ComputeGradientError(x, shape, cs[i],
                                                        shapes[i])
            self.assertLess(err, 1e-6)

  def testInferNum(self):
    with self.test_session():
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        x = tf.placeholder(np.float32, shape=shape)
        cs = tf.unpack(x)
        self.assertEqual(type(cs), list)
        self.assertEqual(len(cs), shape[0])

  def testCannotInferNum(self):
    x = tf.placeholder(np.float32)
    with self.assertRaisesRegexp(
        ValueError, r'Cannot infer num from shape TensorShape\(None\)'):
      tf.unpack(x)


if __name__ == '__main__':
  tf.test.main()
