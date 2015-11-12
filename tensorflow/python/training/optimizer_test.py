"""Functional test for optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.python.platform

import tensorflow as tf


class OptimizerTest(tf.test.TestCase):

  def testBasic(self):
    with self.test_session():
      var0 = tf.Variable([1.0, 2.0])
      var1 = tf.Variable([3.0, 4.0])
      cost = 5 * var0 + 3 * var1
      global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step')
      sgd_op = tf.train.GradientDescentOptimizer(3.0)
      opt_op = sgd_op.minimize(cost, global_step, [var0, var1])

      tf.initialize_all_variables().run()
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())
      # Run 1 step of sgd through optimizer
      opt_op.run()
      # Validate updated params
      self.assertAllClose([-14., -13.], var0.eval())
      self.assertAllClose([-6., -5.], var1.eval())

  def testAggregationMethod(self):
    with self.test_session():
      var0 = tf.Variable([1.0, 2.0])
      var1 = tf.Variable([3.0, 4.0])
      cost = 5 * var0 + 3 * var1
      global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step')
      sgd_op = tf.train.GradientDescentOptimizer(3.0)
      opt_op = sgd_op.minimize(
          cost, global_step, [var0, var1], aggregation_method=
          tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

      tf.initialize_all_variables().run()
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())
      # Run 1 step of sgd through optimizer
      opt_op.run()
      # Validate updated params
      self.assertAllClose([-14., -13.], var0.eval())
      self.assertAllClose([-6., -5.], var1.eval())


if __name__ == "__main__":
  tf.test.main()
