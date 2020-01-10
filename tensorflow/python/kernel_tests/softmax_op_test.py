"""Tests for SoftmaxOp."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class SoftmaxTest(tf.test.TestCase):

  def _npSoftmax(self, features):
    batch_dim = 0
    class_dim = 1
    batch_size = features.shape[batch_dim]
    e = np.exp(features -
               np.reshape(np.amax(features, axis=class_dim), [batch_size, 1]))
    return e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])

  def _testSoftmax(self, np_features, use_gpu=False):
    np_softmax = self._npSoftmax(np_features)
    with self.test_session(use_gpu=use_gpu):
      tf_softmax = tf.nn.softmax(np_features)
      out = tf_softmax.eval()
    self.assertAllClose(np_softmax, out)
    self.assertShapeEqual(np_softmax, tf_softmax)
    # Bonus check: the softmaxes should add to one in each
    # batch element.
    self.assertAllClose(np.ones(out.shape[0]),
                        np.sum(out, axis=1))

  def _testAll(self, features):
    self._testSoftmax(features, use_gpu=False)
    self._testSoftmax(features, use_gpu=True)

  def testNpSoftmax(self):
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    # Batch 0: All exps are 1.  The expected result is
    # [0.25, 0.25, 0.25, 0.25]
    #
    # Batch 1:
    # exps = [1., 2.718, 7.389, 20.085]
    # sum = 31.192
    # Softmaxes = exps / sum = [0.0320586, 0.08714432, 0.23688282, 0.64391426]
    np_sm = self._npSoftmax(np.array(features))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.0320586, 0.08714432, 0.23688282, 0.64391426]]),
        np_sm,
        rtol=1.e-5, atol=1.e-5)

  def testShapeMismatch(self):
    with self.assertRaises(ValueError):
      tf.nn.softmax([0., 1., 2., 3.])

  def testFloat(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32))

  def testDouble(self):
    self._testSoftmax(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64),
        use_gpu=False)


if __name__ == "__main__":
  tf.test.main()
