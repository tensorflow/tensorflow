"""Tests for local response normalization."""
import copy

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests.gradient_checker import ComputeGradientError



class LRNOpTest(tf.test.TestCase):

  def _LRN(self, input_image, lrn_depth_radius=5, bias=1.0,
           alpha=1.0, beta=0.5):
    """Compute expected result."""
    output = copy.deepcopy(input_image)
    batch_size = input_image.shape[0]
    rows = input_image.shape[1]
    cols = input_image.shape[2]
    depth = input_image.shape[3]
    for b in range(batch_size):
      for r in range(rows):
        for c in range(cols):
          for d in range(depth):
            begin = max(0, d - lrn_depth_radius)
            end = min(depth, d + lrn_depth_radius + 1)
            patch = input_image[b, r, c, begin:end]
            output[b, r, c, d] /= (
                np.power(bias + alpha * np.sum(patch * patch), beta))
    return output

  def _RunAndVerify(self):
    with self.test_session():
      # random shape
      shape = np.random.randint(1, 16, size=4)
      # Make depth at least 2 to make it meaningful
      shape[3] += 1
      p = tf.placeholder(tf.float32, shape=shape)
      # random depth_radius, bias, alpha, beta
      lrn_depth_radius = np.random.randint(1, shape[3])
      bias = 1.0 + np.random.rand()
      alpha = 2.0 * np.random.rand()
      beta = 2.0 * np.random.rand()
      lrn_t = tf.nn.local_response_normalization(
          p, name="lrn", depth_radius=lrn_depth_radius, bias=bias,
          alpha=alpha, beta=beta)
      params = {p: np.random.rand(*shape).astype("f")}
      result = lrn_t.eval(feed_dict=params)
    expected = self._LRN(
        params[p], lrn_depth_radius=lrn_depth_radius, bias=bias, alpha=alpha,
        beta=beta)
    self.assertTrue(np.amax(np.abs(result - expected)) < 1e-4)
    self.assertShapeEqual(expected, lrn_t)

  def testCompute(self):
    for _ in range(2):
      self._RunAndVerify()

  def testGradientsZeroInput(self):
    with self.test_session():
      shape = [4, 4, 4, 4]
      p = tf.placeholder(tf.float32, shape=shape)
      inp_array = np.zeros(shape).astype("f")
      lrn_op = tf.nn.local_response_normalization(p, 2, 1.0, 0.0,
                                                  1.0, name="lrn")
      grad = tf.gradients([lrn_op], [p])[0]
      params = {p: inp_array}
      r = grad.eval(feed_dict=params)
    expected = np.ones(shape).astype("f")
    self.assertAllClose(r, expected)
    self.assertShapeEqual(expected, grad)

  def _RunAndVerifyGradients(self):
    with self.test_session():
      # random shape
      shape = np.random.randint(1, 5, size=4)
      # Make depth at least 2 to make it meaningful
      shape[3] += 1
      # random depth_radius, bias, alpha, beta
      lrn_depth_radius = np.random.randint(1, shape[3])
      bias = 1.0 + np.random.rand()
      alpha = 1.0 * np.random.rand()
      beta = 1.0 * np.random.rand()
      inp_array = np.random.rand(*shape).astype("f")
      inp = tf.constant(list(inp_array.ravel(order="C")), shape=shape)
      lrn_op = tf.nn.local_response_normalization(
          inp, name="lrn", depth_radius=lrn_depth_radius, bias=bias,
          alpha=alpha, beta=beta)
      err = ComputeGradientError(inp, shape, lrn_op, shape)
    print "LRN Gradient error ", err
    self.assertLess(err, 1e-4)

  def testGradients(self):
    for _ in range(2):
      self._RunAndVerifyGradients()


if __name__ == "__main__":
  tf.test.main()
