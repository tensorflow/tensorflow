"""Tests for SoftmaxCrossEntropyWithLogits op."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc


class XentTest(tf.test.TestCase):

  def _npXent(self, features, labels):
    batch_dim = 0
    class_dim = 1
    batch_size = features.shape[batch_dim]
    e = np.exp(features -
               np.reshape(np.amax(features, axis=class_dim), [batch_size, 1]))
    probs = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    bp = (probs - labels)
    l = -np.sum(labels * np.log(probs + 1.0e-20), axis=1)
    return l, bp

  def _testXent(self, np_features, np_labels, use_gpu=False):
    np_loss, np_backprop = self._npXent(np_features, np_labels)
    with self.test_session(use_gpu=use_gpu) as sess:
      loss = tf.nn.softmax_cross_entropy_with_logits(np_features, np_labels)
      backprop = loss.op.outputs[1]
      tf_loss, tf_backprop = sess.run([loss, backprop])
    self.assertAllClose(np_loss, tf_loss)
    self.assertAllClose(np_backprop, tf_backprop)

  def _testAll(self, features, labels):
    self._testXent(features, labels, use_gpu=False)
    self._testXent(features, labels, use_gpu=True)

  def testNpXent(self):
    # We create 2 batches of logits for testing.
    # batch 0 is the boring uniform distribution: 1, 1, 1, 1, with target 3.
    # batch 1 has a bit of difference: 1, 2, 3, 4, with soft targets (1, 2).
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    labels = [[0., 0., 0., 1.], [0., .5, .5, 0.]]

    # For batch 0, we expect the uniform distribution: 0.25, 0.25, 0.25, 0.25
    # With a hard target 3, the backprop is [0.25, 0.25, 0.25, -0.75]
    # The loss for this batch is -log(0.25) = 1.386
    #
    # For batch 1, we have:
    # exp(0) = 1
    # exp(1) = 2.718
    # exp(2) = 7.389
    # exp(3) = 20.085
    # SUM = 31.192
    # So we have as probabilities:
    # exp(0) / SUM = 0.032
    # exp(1) / SUM = 0.087
    # exp(2) / SUM = 0.237
    # exp(3) / SUM = 0.644
    # With a soft target (1, 2), the backprop is
    # [0.032, 0.087 - 0.5 = -0.413, 0.237 - 0.5 = -0.263, 0.644]
    # The loss for this batch is [0.5 * -log(0.087), 0.5 * -log(0.237)]
    # = [1.3862, 1.9401]
    np_loss, np_backprop = self._npXent(np.array(features), np.array(labels))
    self.assertAllClose(np.array([[0.25, 0.25, 0.25, -0.75],
                                  [0.0321, -0.4129, -0.2632, 0.6439]]),
                        np_backprop,
                        rtol=1.e-3, atol=1.e-3)
    self.assertAllClose(np.array([1.3862, 1.9401]), np_loss,
                        rtol=1.e-3, atol=1.e-3)

  def testShapeMismatch(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.nn.softmax_cross_entropy_with_logits(
            [[0., 1.], [2., 3.]], [[0., 1., 0.], [1., 0., 0.]])

  def testNotMatrix(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.nn.softmax_cross_entropy_with_logits([0., 1., 2., 3.],
                                                [0., 1., 0., 1.])

  def testFloat(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float32))

  def testDouble(self):
    self._testXent(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float64),
        use_gpu=False)

  def testGradient(self):
    with self.test_session():
      l = tf.constant([0.0, 0.0, 1.0, 0.0,
                       1.0, 0.0, 0.0, 0.0,
                       0.0, 0.5, 0.0, 0.5], shape=[3, 4],
                      dtype=tf.float64, name="l")
      f = tf.constant([0.1, 0.2, 0.3, 0.4,
                       0.1, 0.4, 0.9, 1.6,
                       0.1, 0.8, 2.7, 6.4], shape=[3, 4],
                      dtype=tf.float64, name="f")
      x = tf.nn.softmax_cross_entropy_with_logits(f, l, name="xent")
      err = gc.ComputeGradientError(f, [3, 4], x, [3])
    print "cross entropy gradient err = ", err
    self.assertLess(err, 5e-8)


if __name__ == "__main__":
  tf.test.main()
