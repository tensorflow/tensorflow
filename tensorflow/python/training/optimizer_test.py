# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functional test for optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class OptimizerTest(tf.test.TestCase):

  def testBasic(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        cost = 5 * var0 + 3 * var1
        global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step')
        sgd_op = tf.train.GradientDescentOptimizer(3.0)
        opt_op = sgd_op.minimize(cost, global_step, [var0, var1])

        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Run 1 step of sgd through optimizer
        opt_op.run()
        # Validate updated params
        self.assertAllClose([-14., -13.], var0.eval())
        self.assertAllClose([-6., -5.], var1.eval())

  def testAggregationMethod(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        cost = 5 * var0 + 3 * var1
        global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step')
        sgd_op = tf.train.GradientDescentOptimizer(3.0)
        opt_op = sgd_op.minimize(
            cost,
            global_step, [var0, var1],
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Run 1 step of sgd through optimizer
        opt_op.run()
        # Validate updated params
        self.assertAllClose([-14., -13.], var0.eval())
        self.assertAllClose([-6., -5.], var1.eval())

  def testPrecomputedGradient(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        cost = 5 * var0 + 3 * var1
        grad_loss = tf.constant([42, -42], dtype=dtype)
        global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step')
        sgd_op = tf.train.GradientDescentOptimizer(3.0)
        opt_op = sgd_op.minimize(
            cost, global_step, [var0, var1], grad_loss=grad_loss)

        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Run 1 step of sgd through optimizer
        opt_op.run()
        # Validate updated params
        self.assertAllClose([1.0 - 3 * 5 * 42.0, 2.0 - 3 * 5 * (-42.0)],
                            var0.eval())
        self.assertAllClose([3.0 - 3 * 3 * 42.0, 4.0 - 3 * 3 * (-42.0)],
                            var1.eval())

  def testNoVariables(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype, trainable=False)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype, trainable=False)
        cost = 5 * var0 + var1
        sgd_op = tf.train.GradientDescentOptimizer(3.0)
        with self.assertRaisesRegexp(ValueError, 'No variables'):
          sgd_op.minimize(cost)

  def testNoGradients(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        cost = 5 * var0
        global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step')
        sgd_op = tf.train.GradientDescentOptimizer(3.0)
        with self.assertRaisesRegexp(ValueError, 'No gradients'):
          # var1 has no gradient
          sgd_op.minimize(cost, global_step, [var1])

  def testNoGradientsForAnyVariables_Minimize(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        cost = tf.constant(5.0)
        global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step')
        sgd_op = tf.train.GradientDescentOptimizer(3.0)
        with self.assertRaisesRegexp(ValueError,
                                     'No gradients provided for any variable'):
          sgd_op.minimize(cost, global_step, [var0, var1])

  def testNoGradientsForAnyVariables_ApplyGradients(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        sgd_op = tf.train.GradientDescentOptimizer(3.0)
        with self.assertRaisesRegexp(ValueError,
                                     'No gradients provided for any variable'):
          sgd_op.apply_gradients([(None, var0), (None, var1)])

  def testGradientsAsVariables(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session() as sess:
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        cost = 5 * var0 + 3 * var1
        global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step')
        sgd_op = tf.train.GradientDescentOptimizer(3.0)
        grads_and_vars = sgd_op.compute_gradients(cost, [var0, var1])
        # Convert gradients to tf.Variables
        converted_grads = [
            tf.Variable(tf.zeros([2], dtype)) for i in grads_and_vars
        ]
        convert_ops = [
            tf.assign(converted_grads[i], gv[0])
            for i, gv in enumerate(grads_and_vars)
        ]

        converted_grads_and_vars = list(zip(converted_grads, [var0, var1]))
        opt_op = sgd_op.apply_gradients(converted_grads_and_vars, global_step)

        tf.global_variables_initializer().run()
        # Run convert_ops to achieve the gradietns converting
        sess.run(convert_ops)
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Run 1 step of sgd through optimizer
        opt_op.run()
        # Validate updated params
        self.assertAllClose([-14., -13.], var0.eval())
        self.assertAllClose([-6., -5.], var1.eval())

  def testTrainOp(self):
    with self.test_session():
      var0 = tf.Variable([1.0, 2.0])
      var1 = tf.Variable([3.0, 4.0])
      cost = 5 * var0 + 3 * var1
      global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step')
      sgd_op = tf.train.GradientDescentOptimizer(3.0)
      opt_op = sgd_op.minimize(cost, global_step, [var0, var1])
      self.assertTrue(opt_op in tf.get_collection(tf.GraphKeys.TRAIN_OP))


if __name__ == '__main__':
  tf.test.main()
