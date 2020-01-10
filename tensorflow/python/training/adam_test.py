"""Tests for Adam."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


def adam_update_numpy(param, g_t, t, m, v, alpha=0.001, beta1=0.9, beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class AdamOptimizerTest(tf.test.TestCase):

  def testSparse(self):
    with self.test_session():
      # Initialize variables for numpy implementation.
      m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
      var0_np = np.array([1.0, 2.0], dtype=np.float32)
      grads0_np = np.array([0.1, 0.1], dtype=np.float32)
      var1_np = np.array([3.0, 4.0], dtype=np.float32)
      grads1_np = np.array([0.01, 0.01], dtype=np.float32)

      var0 = tf.Variable(var0_np)
      var1 = tf.Variable(var1_np)
      grads0_np_indices = np.array([0, 1], dtype=np.int32)
      grads0 = tf.IndexedSlices(tf.constant(grads0_np),
                                tf.constant(grads0_np_indices),
                                tf.constant([2]))
      grads1_np_indices = np.array([0, 1], dtype=np.int32)
      grads1 = tf.IndexedSlices(tf.constant(grads1_np),
                                tf.constant(grads1_np_indices),
                                tf.constant([2]))
      opt = tf.train.AdamOptimizer()
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())

      beta1_power, beta2_power = opt._get_beta_accumulators()

      # Run 3 steps of Adam
      for t in range(1, 4):
        self.assertAllClose(0.9 ** t, beta1_power.eval())
        self.assertAllClose(0.999 ** t, beta2_power.eval())
        update.run()

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        self.assertAllClose(var0_np, var0.eval())
        self.assertAllClose(var1_np, var1.eval())

  def testBasic(self):
    with self.test_session():
      # Initialize variables for numpy implementation.
      m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
      var0_np = np.array([1.0, 2.0], dtype=np.float32)
      grads0_np = np.array([0.1, 0.1], dtype=np.float32)
      var1_np = np.array([3.0, 4.0], dtype=np.float32)
      grads1_np = np.array([0.01, 0.01], dtype=np.float32)

      var0 = tf.Variable(var0_np)
      var1 = tf.Variable(var1_np)
      grads0 = tf.constant(grads0_np)
      grads1 = tf.constant(grads1_np)
      opt = tf.train.AdamOptimizer()
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())

      beta1_power, beta2_power = opt._get_beta_accumulators()

      # Run 3 steps of Adam
      for t in range(1, 4):
        self.assertAllClose(0.9 ** t, beta1_power.eval())
        self.assertAllClose(0.999 ** t, beta2_power.eval())
        update.run()

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        self.assertAllClose(var0_np, var0.eval())
        self.assertAllClose(var1_np, var1.eval())

  def testFloat64(self):
    with self.test_session():
      opt = tf.train.AdamOptimizer()

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

  def testSharing(self):
    with self.test_session():
      # Initialize variables for numpy implementation.
      m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
      var0_np = np.array([1.0, 2.0], dtype=np.float32)
      grads0_np = np.array([0.1, 0.1], dtype=np.float32)
      var1_np = np.array([3.0, 4.0], dtype=np.float32)
      grads1_np = np.array([0.01, 0.01], dtype=np.float32)

      var0 = tf.Variable(var0_np)
      var1 = tf.Variable(var1_np)
      grads0 = tf.constant(grads0_np)
      grads1 = tf.constant(grads1_np)
      opt = tf.train.AdamOptimizer()
      update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      beta1_power, beta2_power = opt._get_beta_accumulators()

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())

      # Run 3 steps of intertwined Adam1 and Adam2.
      for t in range(1, 4):
        self.assertAllClose(0.9 ** t, beta1_power.eval())
        self.assertAllClose(0.999 ** t, beta2_power.eval())
        if t % 2 == 0:
          update1.run()
        else:
          update2.run()

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        self.assertAllClose(var0_np, var0.eval())
        self.assertAllClose(var1_np, var1.eval())


if __name__ == "__main__":
  tf.test.main()
