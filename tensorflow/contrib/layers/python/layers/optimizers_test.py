# Copyright 2015 Google Inc. All Rights Reserved.
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
    optimizers = ["SGD", tf.train.GradientDescentOptimizer,
                  tf.train.GradientDescentOptimizer(learning_rate=0.1)]
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

  def testWrongOptimizer(self):
    optimizers = ["blah", tf.Variable, object()]
    for optimizer in optimizers:
      with tf.Graph().as_default() as g:
        with self.test_session(graph=g):
          _, _, loss, global_step = _setup_model()
          with self.assertRaises(ValueError):
            tf.contrib.layers.optimize_loss(loss,
                                            global_step,
                                            learning_rate=0.1,
                                            optimizer=optimizer)

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

  def testIgnoreVariablesWithNoGradients(self):
    _, _, loss, global_step = _setup_model()

    unused_variable = tf.get_variable("ignore me", [])

    tf.contrib.layers.optimize_loss(
        loss, global_step, learning_rate=0.1, optimizer="SGD")


if __name__ == "__main__":
  tf.test.main()

