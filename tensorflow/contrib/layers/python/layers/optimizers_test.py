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
"""Tests for optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _setup_model():
  x = tf.placeholder(tf.float32, [])
  var = tf.get_variable("test", [], initializer=tf.constant_initializer(10))
  loss = tf.abs(var * x)
  global_step = tf.get_variable(
      "global_step", [], trainable=False, dtype=tf.int64,
      initializer=tf.constant_initializer(0, dtype=tf.int64))
  return x, var, loss, global_step


def _no_op_learning_rate_decay_fn(lr, global_step):
  assert lr is not None
  assert global_step is not None
  return lr


class OptimizersTest(tf.test.TestCase):

  def testSGDOptimizer(self):
    optimizers = [
        "SGD", tf.train.GradientDescentOptimizer,
        tf.train.GradientDescentOptimizer(learning_rate=0.1),
        lambda lr: tf.train.GradientDescentOptimizer(learning_rate=lr)]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g:
        with self.test_session(graph=g) as session:
          x, var, loss, global_step = _setup_model()
          train = tf.contrib.layers.optimize_loss(loss,
                                                  global_step,
                                                  learning_rate=0.1,
                                                  optimizer=optimizer)
          tf.global_variables_initializer().run()
          session.run(train, feed_dict={x: 5})
          var_value, global_step_value = session.run([var, global_step])
          self.assertEqual(var_value, 9.5)
          self.assertEqual(global_step_value, 1)

  def testNoLrCallable(self):
    def optimizer_fn():
      return tf.train.GradientDescentOptimizer(learning_rate=0.1)
    with tf.Graph().as_default() as g:
      with self.test_session(graph=g) as session:
        x, var, loss, global_step = _setup_model()
        train = tf.contrib.layers.optimize_loss(loss,
                                                global_step,
                                                learning_rate=None,
                                                optimizer=optimizer_fn)
        tf.global_variables_initializer().run()
        session.run(train, feed_dict={x: 5})
        var_value, global_step_value = session.run([var, global_step])
        self.assertEqual(var_value, 9.5)
        self.assertEqual(global_step_value, 1)

  def testWrongOptimizer(self):
    optimizers = ["blah", tf.Variable, object(), lambda x: None]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g:
        with self.test_session(graph=g):
          _, _, loss, global_step = _setup_model()
          with self.assertRaises(ValueError):
            tf.contrib.layers.optimize_loss(loss,
                                            global_step,
                                            learning_rate=0.1,
                                            optimizer=optimizer)

  def testInvalidLoss(self):
    with tf.Graph().as_default() as g, self.test_session(graph=g):
      _, _, _, global_step = _setup_model()
      with self.assertRaises(ValueError):
        tf.contrib.layers.optimize_loss(None,
                                        global_step,
                                        learning_rate=0.1,
                                        optimizer="SGD")
      with self.assertRaises(ValueError):
        tf.contrib.layers.optimize_loss([[1.0]],
                                        global_step,
                                        learning_rate=0.1,
                                        optimizer="SGD")

  def testInvalidGlobalStep(self):
    with tf.Graph().as_default() as g, self.test_session(graph=g):
      x = tf.placeholder(tf.float32, [])
      var = tf.get_variable("test", [], initializer=tf.constant_initializer(10))
      loss = tf.abs(var * x)
      with self.assertRaises(TypeError):
        tf.contrib.layers.optimize_loss(
            loss, global_step=tf.constant(43, dtype=tf.int64),
            learning_rate=0.1, optimizer="SGD")
      with self.assertRaises(TypeError):
        tf.contrib.layers.optimize_loss(
            loss,
            global_step=tf.get_variable(
                "global_step", [], trainable=False, dtype=tf.float64,
                initializer=tf.constant_initializer(0.0, dtype=tf.float64)),
            learning_rate=0.1, optimizer="SGD")
      with self.assertRaises(ValueError):
        tf.contrib.layers.optimize_loss(
            loss,
            global_step=tf.get_variable(
                "global_step", [1], trainable=False, dtype=tf.int64,
                initializer=tf.constant_initializer([0], dtype=tf.int64)),
            learning_rate=0.1, optimizer="SGD")

  def testInvalidLearningRate(self):
    with tf.Graph().as_default() as g, self.test_session(graph=g):
      _, _, loss, global_step = _setup_model()
      with self.assertRaises(ValueError):
        tf.contrib.layers.optimize_loss(loss,
                                        global_step,
                                        learning_rate=-0.1,
                                        optimizer="SGD")

  def testGradientNoise(self):
    tf.set_random_seed(42)
    with self.test_session() as session:
      x, var, loss, global_step = _setup_model()
      train = tf.contrib.layers.optimize_loss(loss,
                                              global_step,
                                              learning_rate=0.1,
                                              optimizer="SGD",
                                              gradient_noise_scale=10.0)
      tf.global_variables_initializer().run()
      session.run(train, feed_dict={x: 5})
      var_value, global_step_value = session.run([var, global_step])
      # Due to randomness the following number may change if graph is different.
      self.assertAlmostEqual(var_value, 8.5591021, 4)
      self.assertEqual(global_step_value, 1)

  def testGradientNoiseWithClipping(self):
    tf.set_random_seed(42)
    with self.test_session() as session:
      x, var, loss, global_step = _setup_model()
      train = tf.contrib.layers.optimize_loss(loss,
                                              global_step,
                                              learning_rate=0.1,
                                              optimizer="SGD",
                                              gradient_noise_scale=10.0,
                                              clip_gradients=10.0)
      tf.global_variables_initializer().run()
      session.run(train, feed_dict={x: 5})
      var_value, global_step_value = session.run([var, global_step])
      self.assertAlmostEqual(var_value, 9.0, 4)
      self.assertEqual(global_step_value, 1)

  def testGradientClip(self):
    with self.test_session() as session:
      x, var, loss, global_step = _setup_model()
      train = tf.contrib.layers.optimize_loss(loss,
                                              global_step,
                                              learning_rate=0.1,
                                              optimizer="SGD",
                                              clip_gradients=0.1)
      tf.global_variables_initializer().run()
      session.run(train, feed_dict={x: 5})
      var_value, global_step_value = session.run([var, global_step])
      self.assertAlmostEqual(var_value, 9.98999, 4)
      self.assertEqual(global_step_value, 1)

  def testAdaptiveGradientClip(self):
    with self.test_session() as session:
      x, var, loss, global_step = _setup_model()
      clip_gradients = tf.contrib.layers.adaptive_clipping_fn()
      train = tf.contrib.layers.optimize_loss(loss,
                                              global_step,
                                              learning_rate=0.1,
                                              optimizer="SGD",
                                              clip_gradients=clip_gradients)
      tf.global_variables_initializer().run()
      session.run(train, feed_dict={x: 5})
      var_value, global_step_value = session.run([var, global_step])
      self.assertAlmostEqual(var_value, 9.8916, 4)
      self.assertEqual(global_step_value, 1)
      var_count = 0
      for var in tf.all_variables():
        if var.name.startswith("OptimizeLoss/AdaptiveMaxNorm"):
          var_count += 1
      self.assertEqual(2, var_count)

  def testGradientMultiply(self):
    with self.test_session() as session:
      x, var, loss, global_step = _setup_model()
      train = tf.contrib.layers.optimize_loss(loss,
                                              global_step,
                                              learning_rate=0.1,
                                              optimizer="SGD",
                                              gradient_multipliers={var: 7.})
      tf.global_variables_initializer().run()
      session.run(train, feed_dict={x: 5})
      var_value, global_step_value = session.run([var, global_step])
      # var(0) = 10, x = 5, var(0)/dx = 5,
      # var(1) = var(0) - learning_rate * gradient_multiplier * var(0)/dx
      self.assertAlmostEqual(var_value, 6.5, 4)
      self.assertEqual(global_step_value, 1)

  def testIgnoreVariablesWithNoGradients(self):
    _, _, loss, global_step = _setup_model()

    unused_variable = tf.get_variable("ignore_me", [])

    tf.contrib.layers.optimize_loss(
        loss, global_step, learning_rate=0.1, optimizer="SGD",
        gradient_noise_scale=10.0,
        gradient_multipliers={unused_variable: 1.},
        clip_gradients=10.0)

  def testNoGlobalStep(self):
    optimizers = ["SGD", tf.train.GradientDescentOptimizer,
                  tf.train.GradientDescentOptimizer(learning_rate=0.1)]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
        x = tf.placeholder(tf.float32, [])
        var = tf.get_variable(
            "test", [], initializer=tf.constant_initializer(10))
        loss = tf.abs(var * x)
        update_var = tf.get_variable(
            "update", [], initializer=tf.constant_initializer(10))
        update_op = tf.assign(update_var, 20)
        train = tf.contrib.layers.optimize_loss(loss,
                                                global_step=None,
                                                learning_rate=0.1,
                                                optimizer=optimizer,
                                                update_ops=[update_op])
        tf.global_variables_initializer().run()
        session.run(train, feed_dict={x: 5})
        self.assertEqual(9.5, var.eval())
        self.assertEqual(20, update_var.eval())

  def testNoGlobalStepWithDecay(self):
    optimizers = ["SGD", tf.train.GradientDescentOptimizer,
                  tf.train.GradientDescentOptimizer(learning_rate=0.1)]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g, self.test_session(graph=g):
        x = tf.placeholder(tf.float32, [])
        var = tf.get_variable(
            "test", [], initializer=tf.constant_initializer(10))
        loss = tf.abs(var * x)
        update_var = tf.get_variable(
            "update", [], initializer=tf.constant_initializer(10))
        update_op = tf.assign(update_var, 20)
        with self.assertRaisesRegexp(
            ValueError, "global_step is required for learning_rate_decay_fn"):
          tf.contrib.layers.optimize_loss(
              loss,
              global_step=None,
              learning_rate=0.1,
              learning_rate_decay_fn=_no_op_learning_rate_decay_fn,
              optimizer=optimizer,
              update_ops=[update_op])

  def testNoGlobalStepArg(self):
    optimizers = ["SGD", tf.train.GradientDescentOptimizer,
                  tf.train.GradientDescentOptimizer(learning_rate=0.1)]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
        x, var, loss, global_step = _setup_model()
        update_var = tf.get_variable(
            "update", [], initializer=tf.constant_initializer(10))
        update_op = tf.assign(update_var, 20)
        train = tf.contrib.layers.optimize_loss(loss,
                                                global_step=None,
                                                learning_rate=0.1,
                                                optimizer=optimizer,
                                                update_ops=[update_op])
        tf.global_variables_initializer().run()
        session.run(train, feed_dict={x: 5})
        self.assertEqual(9.5, var.eval())
        self.assertEqual(20, update_var.eval())
        self.assertEqual(1, global_step.eval())

  def testUpdateOp(self):
    optimizers = ["SGD", tf.train.GradientDescentOptimizer,
                  tf.train.GradientDescentOptimizer(learning_rate=0.1)]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
        x, var, loss, global_step = _setup_model()
        update_var = tf.get_variable(
            "update", [], initializer=tf.constant_initializer(10))
        update_op = tf.assign(update_var, 20)
        train = tf.contrib.layers.optimize_loss(loss,
                                                global_step,
                                                learning_rate=0.1,
                                                optimizer=optimizer,
                                                update_ops=[update_op])
        tf.global_variables_initializer().run()
        session.run(train, feed_dict={x: 5})
        self.assertEqual(9.5, var.eval())
        self.assertEqual(20, update_var.eval())
        self.assertEqual(1, global_step.eval())

  def testUpdateOpWithNoOpDecay(self):
    optimizers = ["SGD", tf.train.GradientDescentOptimizer,
                  tf.train.GradientDescentOptimizer(learning_rate=0.1)]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
        x, var, loss, global_step = _setup_model()
        update_var = tf.get_variable(
            "update", [], initializer=tf.constant_initializer(10))
        update_op = tf.assign(update_var, 20)
        train = tf.contrib.layers.optimize_loss(
            loss,
            global_step,
            learning_rate=0.1,
            learning_rate_decay_fn=_no_op_learning_rate_decay_fn,
            optimizer=optimizer,
            update_ops=[update_op])
        tf.global_variables_initializer().run()
        session.run(train, feed_dict={x: 5})
        self.assertEqual(9.5, var.eval())
        self.assertEqual(20, update_var.eval())
        self.assertEqual(1, global_step.eval())

  def testUpdateOpFromCollection(self):
    optimizers = ["SGD", tf.train.GradientDescentOptimizer,
                  tf.train.GradientDescentOptimizer(learning_rate=0.1)]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g, self.test_session(graph=g) as session:
        x, var, loss, global_step = _setup_model()
        update_var = tf.get_variable(
            "update", [], initializer=tf.constant_initializer(10))
        update_op = tf.assign(update_var, 20)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
        train = tf.contrib.layers.optimize_loss(
            loss, global_step, learning_rate=0.1, optimizer=optimizer)
        tf.global_variables_initializer().run()
        session.run(train, feed_dict={x: 5})
        var_value, update_var_value, global_step_value = session.run(
            [var, update_var, global_step])
        self.assertEqual(var_value, 9.5)
        self.assertEqual(update_var_value, 20)
        self.assertEqual(global_step_value, 1)


class AdaptiveClipping(tf.test.TestCase):

  def testAverages(self):
    with self.test_session() as session:
      scale = 2.
      grad = tf.ones([3, 4]) * scale
      log_norm = np.log(np.sqrt(scale**2 * grad.get_shape().num_elements()))
      grads_and_vars = [(grad, grad)]
      grads_and_vars = tf.contrib.layers.adaptive_clipping_fn(
          decay=0.5)(grads_and_vars)

      var_dict = {}
      for var in tf.all_variables():
        if var.name.startswith("AdaptiveMaxNorm"):
          var_dict[var.name.split(":")[0]] = var
      self.assertEqual(2, len(var_dict))
      moving_mean = var_dict["AdaptiveMaxNorm/mean"]
      moving_sq_mean = var_dict["AdaptiveMaxNorm/sq_mean"]
      tf.global_variables_initializer().run()
      mean, sq_mean = session.run([moving_mean, moving_sq_mean])
      self.assertEqual([0], mean)
      self.assertEqual([0], sq_mean)
      for i in range(20):
        mean, sq_mean, _ = session.run(
            [moving_mean, moving_sq_mean, grads_and_vars[0][0]])
        if i == 0:
          self.assertLess(mean, 0.9 * log_norm)
          self.assertLess(sq_mean, 0.9 * log_norm**2)

      self.assertAlmostEqual(float(mean), log_norm, places=4)
      self.assertAlmostEqual(float(sq_mean), log_norm**2, places=4)

  def testClip(self):
    with self.test_session() as session:
      spike = 1000.
      multiplier = tf.placeholder(tf.float32, [], "multiplier")
      step = tf.placeholder(tf.int32, [], "step")

      grad = tf.ones([3, 4]) * multiplier
      grads_and_vars = [(grad, grad)]
      grads_and_vars = tf.contrib.layers.adaptive_clipping_fn(
          decay=0.9, global_step=step)(grads_and_vars)

      tf.global_variables_initializer().run()
      def run(scale, i):
        return session.run(grads_and_vars[0][0],
                           feed_dict={multiplier: scale, step: i})

      for i in range(20):
        scale = [1., -2.][i % 2]
        clipped_grad = run(scale, i)
        if i > 3:
          self.assertAllClose(np.ones(clipped_grad.shape)*scale, clipped_grad)

      # assert that the spike will have low influence.
      clipped_grad = run(spike, 20)
      self.assertTrue((clipped_grad < 25.).all())

      # assert that a repeated spike will converge to this new value.
      for i in range(10):
        clipped_grad = run(spike, i + 21)

      self.assertAllClose(np.ones(clipped_grad.shape)*spike, clipped_grad)

if __name__ == "__main__":
  tf.test.main()
