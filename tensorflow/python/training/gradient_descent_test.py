"""Functional test for GradientDescent."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class GradientDescentOptimizerTest(tf.test.TestCase):

  def testBasic(self):
    with self.test_session():
      var0 = tf.Variable([1.0, 2.0])
      var1 = tf.Variable([3.0, 4.0])
      grads0 = tf.constant([0.1, 0.1])
      grads1 = tf.constant([0.01, 0.01])
      sgd_op = tf.train.GradientDescentOptimizer(3.0).apply_gradients(
          zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())
      # Run 1 step of sgd
      sgd_op.run()
      # Validate updated params
      self.assertAllClose([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], var0.eval())
      self.assertAllClose([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], var1.eval())

  def testFloat64(self):
    with self.test_session():
      opt = tf.train.GradientDescentOptimizer(3.0)

      # compute_gradients.
      values = [1.0, 3.0]
      good_vars = [tf.Variable([v]) for v in values]
      bad_loss = tf.constant(2.0, tf.float64, name="bad_loss")
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_loss.*expected.*float32",
          opt.compute_gradients, bad_loss, good_vars)
      bad_vars = [
          tf.Variable(np.array([v], np.float64), name="bad_var")
          for v in values]
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_var.*expected.*float32",
          opt.compute_gradients, tf.cast(bad_vars[0] + bad_vars[1], tf.float32),
          bad_vars)
      opt.compute_gradients(good_vars[0] + good_vars[1], good_vars)

      # apply_gradients.
      bad_grads = [
          tf.constant([0.1], dtype=np.float64, name="bad_grad"),
          tf.constant([0.01])]
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_grad.*expected.*float32",
          opt.apply_gradients, zip(bad_grads, good_vars))
      good_grads = [tf.constant([0.01]), tf.constant([0.02])]
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_var.*expected.*float32",
          opt.apply_gradients, zip(good_grads, bad_vars))
      opt.apply_gradients(zip(good_grads, good_vars))

  def testWithGlobalStep(self):
    with self.test_session():
      global_step = tf.Variable(0, trainable=False)
      var0 = tf.Variable([1.0, 2.0])
      var1 = tf.Variable([3.0, 4.0])
      grads0 = tf.constant([0.1, 0.1])
      grads1 = tf.constant([0.01, 0.01])
      sgd_op = tf.train.GradientDescentOptimizer(3.0).apply_gradients(
          zip([grads0, grads1], [var0, var1]), global_step=global_step)
      tf.initialize_all_variables().run()
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())
      # Run 1 step of sgd
      sgd_op.run()
      # Validate updated params and global_step
      self.assertAllClose([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], var0.eval())
      self.assertAllClose([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], var1.eval())
      self.assertAllClose(1, global_step.eval())

  def testSparseBasic(self):
    with self.test_session():
      var0 = tf.Variable([[1.0], [2.0]])
      var1 = tf.Variable([[3.0], [4.0]])
      grads0 = tf.IndexedSlices(tf.constant([0.1], shape=[1, 1]),
                                tf.constant([0]),
                                tf.constant([2, 1]))
      grads1 = tf.IndexedSlices(tf.constant([0.01], shape=[1, 1]),
                                tf.constant([1]),
                                tf.constant([2, 1]))
      sgd_op = tf.train.GradientDescentOptimizer(3.0).apply_gradients(
          zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()
      # Fetch params to validate initial values
      self.assertAllClose([[1.0], [2.0]], var0.eval())
      self.assertAllClose([[3.0], [4.0]], var1.eval())
      # Run 1 step of sgd
      sgd_op.run()
      # Validate updated params
      self.assertAllClose([[1.0 - 3.0 * 0.1], [2.0]], var0.eval())
      self.assertAllClose([[3.0], [4.0 - 3.0 * 0.01]], var1.eval())


if __name__ == "__main__":
  tf.test.main()
