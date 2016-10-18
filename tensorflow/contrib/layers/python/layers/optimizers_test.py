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
          tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
      tf.initialize_all_variables().run()
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
      tf.initialize_all_variables().run()
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
      tf.initialize_all_variables().run()
      session.run(train, feed_dict={x: 5})
      var_value, global_step_value = session.run([var, global_step])
      self.assertAlmostEqual(var_value, 9.98999, 4)
      self.assertEqual(global_step_value, 1)

  def testGradientMultiply(self):
    with self.test_session() as session:
      x, var, loss, global_step = _setup_model()
      train = tf.contrib.layers.optimize_loss(loss,
                                              global_step,
                                              learning_rate=0.1,
                                              optimizer="SGD",
                                              gradient_multipliers={var: 7.})
      tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
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
        tf.initialize_all_variables().run()
        session.run(train, feed_dict={x: 5})
        var_value, update_var_value, global_step_value = session.run(
            [var, update_var, global_step])
        self.assertEqual(var_value, 9.5)
        self.assertEqual(update_var_value, 20)
        self.assertEqual(global_step_value, 1)

if __name__ == "__main__":
  tf.test.main()
