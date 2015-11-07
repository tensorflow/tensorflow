"""Tests for tensorflow.ops.tf.gather."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class GatherTest(tf.test.TestCase):

  def testScalar1D(self):
    with self.test_session():
      params = tf.constant([0, 1, 2, 3, 7, 5])
      indices = tf.constant(4)
      gather_t = tf.gather(params, indices)
      gather_val = gather_t.eval()
    self.assertAllEqual(7, gather_val)
    self.assertEqual([], gather_t.get_shape())

  def testScalar2D(self):
    with self.test_session():
      params = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                     [9, 10, 11], [12, 13, 14]])
      indices = tf.constant(2)
      gather_t = tf.gather(params, indices)
      gather_val = gather_t.eval()
    self.assertAllEqual([6, 7, 8], gather_val)
    self.assertEqual([3], gather_t.get_shape())

  def testSimpleTwoD32(self):
    with self.test_session():
      params = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                     [9, 10, 11], [12, 13, 14]])
      indices = tf.constant([0, 4, 0, 2])
      gather_t = tf.gather(params, indices)
      gather_val = gather_t.eval()
    self.assertAllEqual([[0, 1, 2], [12, 13, 14], [0, 1, 2], [6, 7, 8]],
                        gather_val)
    self.assertEqual([4, 3], gather_t.get_shape())

  def testHigherRank(self):
    np.random.seed(1)
    shape = (4, 3, 2)
    params = np.random.randn(*shape)
    indices = np.random.randint(shape[0], size=15).reshape(3, 5)
    with self.test_session():
      tf_params = tf.constant(params)
      tf_indices = tf.constant(indices)
      gather = tf.gather(tf_params, tf_indices)
      self.assertAllEqual(params[indices], gather.eval())
      self.assertEqual(indices.shape + params.shape[1:], gather.get_shape())
      # Test gradients
      gather_grad = np.random.randn(*gather.get_shape().as_list())
      params_grad, indices_grad = tf.gradients(gather, [tf_params, tf_indices],
                                               gather_grad)
      self.assertEqual(indices_grad, None)
      self.assertEqual(type(params_grad), tf.IndexedSlices)
      params_grad = tf.convert_to_tensor(params_grad)
      correct_params_grad = np.zeros(shape)
      for i, g in zip(indices.ravel(), gather_grad.reshape((15,) + shape[1:])):
        correct_params_grad[i] += g
      self.assertAllEqual(correct_params_grad, params_grad.eval())

  def testUnknownIndices(self):
    params = tf.constant([[0, 1, 2]])
    indices = tf.placeholder(tf.int32)
    gather_t = tf.gather(params, indices)
    self.assertEqual(None, gather_t.get_shape())


if __name__ == "__main__":
  tf.test.main()
