"""Tests for Relu and ReluGrad."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc


class ReluTest(tf.test.TestCase):

  def _npRelu(self, np_features):
    return np.maximum(np_features, np.zeros(np_features.shape))

  def testNpRelu(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 0.0],
                  [0.1, 0.0, 0.5, 0.0, 0.9]]),
        self._npRelu(np.array([[-0.9, 0.7, -0.5, 0.3, -0.1],
                               [0.1, -0.3, 0.5, -0.7, 0.9]])))

  def _testRelu(self, np_features, use_gpu=False):
    np_relu = self._npRelu(np_features)
    with self.test_session(use_gpu=use_gpu):
      relu = tf.nn.relu(np_features)
      tf_relu = relu.eval()
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, relu)

  def testNumbers(self):
    for t in [np.int32, np.int64, np.float, np.double]:
      self._testRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      if t in [np.float, np.double]:
        self._testRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            use_gpu=True)

  # The gradient test for ReLU is a bit tricky as the derivative is not well
  # defined at around zero and we want to avoid that in terms of input values.
  def testGradientFloat(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], name="x")
      y = tf.nn.relu(x, name="relu")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32, order="F")
      err = gc.ComputeGradientError(x, [2, 5], y, [2, 5], x_init_value=x_init)
    print "relu (float) gradient err = ", err
    self.assertLess(err, 1e-4)

  def testGradientNaN(self):
    with self.test_session():
      # Note the NaN is injected as an input to the gradient calculation.
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, np.nan, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], name="x")
      y = tf.nn.relu(x, name="relu")
      grad_ys = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], name="ys")
      g_op = tf.gradients(
          [y], [x], grad_ys=[grad_ys], name="gradients")[0]
      try:
        g_op.op.run()
        assert False, "ReluGrad should have failed due to CheckNumerics."
      except Exception as e:  # pylint: disable=broad-except
        assert "ReluGrad input is not finite." in str(e)

  def testGradientDouble(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], dtype=tf.float64, name="x")
      y = tf.nn.relu(x, name="relu")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64, order="F")
      err = gc.ComputeGradientError(x, [2, 5], y, [2, 5], x_init_value=x_init)
    print "relu (double) gradient err = ", err
    self.assertLess(err, 1e-10)

  def testGradGradFloat(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], name="x")
      y = tf.nn.relu(x, name="relu")
      z = tf.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32, order="F")
      err = gc.ComputeGradientError(x, [2, 5], z[0], [2, 5],
                                    x_init_value=x_init)
    print "relu (float) gradient of gradient err = ", err
    self.assertLess(err, 1e-4)

  def testGradGradDouble(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], dtype=tf.float64, name="x")
      y = tf.nn.relu(x, name="relu")
      z = tf.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64, order="F")
      err = gc.ComputeGradientError(x, [2, 5], z[0], [2, 5],
                                    x_init_value=x_init)
    print "relu (double) gradient of gradient err = ", err
    self.assertLess(err, 1e-10)


class Relu6Test(tf.test.TestCase):

  def _npRelu6(self, np_features):
    sixes = np.copy(np_features)
    sixes.fill(6.0)
    return np.minimum(np.maximum(np_features, np.zeros(np_features.shape)),
                      sixes)

  def testNpRelu6(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 6.0],
                  [0.1, 0.0, 6.0, 0.0, 0.9]]),
        self._npRelu6(np.array([[-0.9, 0.7, -0.5, 0.3, 6.0],
                                [0.1, -0.3, 6.5, -0.7, 0.9]])))

  def _testRelu6(self, np_features, use_gpu=False):
    np_relu6 = self._npRelu6(np_features)
    with self.test_session(use_gpu=use_gpu):
      relu6 = tf.nn.relu6(np_features)
      tf_relu6 = relu6.eval()
    self.assertAllClose(np_relu6, tf_relu6)
    self.assertShapeEqual(np_relu6, relu6)

  def testNumbers(self):
    for t in [np.int32, np.int64, np.float, np.double]:
      self._testRelu6(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      if t in [np.float, np.double]:
        self._testRelu6(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            use_gpu=True)

  # The gradient test for ReLU6 is a bit tricky as the derivative is
  # not well defined at around zero and six and we want to avoid that
  # in terms of input values.
  def testGradientFloat(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 6.1, 6.3, 6.5, 6.7, 6.9],
          shape=[2, 5], name="x")
      y = tf.nn.relu6(x, name="relu6")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float32, order="F")
      err = gc.ComputeGradientError(x, [2, 5], y, [2, 5], x_init_value=x_init)
    print "relu6 (float) gradient err = ", err
    self.assertLess(err, 1e-4)

  def testGradientDouble(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 6.1, 6.3, 6.5, 6.7, 6.9],
          shape=[2, 5], dtype=tf.float64, name="x")
      y = tf.nn.relu6(x, name="relu6")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float64, order="F")
      err = gc.ComputeGradientError(x, [2, 5], y, [2, 5], x_init_value=x_init)
    print "relu6 (double) gradient err = ", err
    self.assertLess(err, 1e-10)


if __name__ == "__main__":
  tf.test.main()
