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
  global_step = tf.get_variable("global_step",
                                [],
                                trainable=False,
                                initializer=tf.constant_initializer(0))
  return x, var, loss, global_step


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

  def testUpdateOp(self):
    optimizers = ["SGD", tf.train.GradientDescentOptimizer,
                  tf.train.GradientDescentOptimizer(learning_rate=0.1)]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g:
        with self.test_session(graph=g) as session:
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
          var_value, update_var_value, global_step_value = session.run(
              [var, update_var, global_step])
          self.assertEqual(var_value, 9.5)
          self.assertEqual(update_var_value, 20)
          self.assertEqual(global_step_value, 1)

  def testUpdateOpFromCollection(self):
    optimizers = ["SGD", tf.train.GradientDescentOptimizer,
                  tf.train.GradientDescentOptimizer(learning_rate=0.1)]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g:
        with self.test_session(graph=g) as session:
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
