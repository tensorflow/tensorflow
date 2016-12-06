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
"""Tests for tf.layers.core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.layers import core as core_layers


class DenseTest(tf.test.TestCase):

  def testDenseProperties(self):
    dense = core_layers.Dense(2, activation=tf.nn.relu, name='my_dense')
    self.assertEqual(dense.units, 2)
    self.assertEqual(dense.activation, tf.nn.relu)
    self.assertEqual(dense.weights_regularizer, None)
    self.assertEqual(dense.bias_regularizer, None)
    self.assertEqual(dense.activity_regularizer, None)
    self.assertEqual(dense.use_bias, True)
    self.assertEqual(dense.name, 'my_dense')

    # Test auto-naming
    dense = core_layers.Dense(2, activation=tf.nn.relu)
    self.assertEqual(dense.name, 'dense')
    dense = core_layers.Dense(2, activation=tf.nn.relu)
    self.assertEqual(dense.name, 'dense_1')

  def testCall(self):
    dense = core_layers.Dense(2, activation=tf.nn.relu, name='my_dense')
    inputs = tf.random_uniform((5, 2), seed=1)
    _ = dense(inputs)
    self.assertListEqual(dense.weights, [dense.w, dense.bias])
    self.assertListEqual(dense.trainable_weights, [dense.w, dense.bias])
    self.assertListEqual(dense.non_trainable_weights, [])
    self.assertListEqual(dense._trainable_weights, [dense.w, dense.bias])
    self.assertListEqual(dense._non_trainable_weights, [])
    self.assertEqual(
        len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)), 2)
    self.assertEqual(dense.w.name, 'my_dense/weights:0')
    self.assertEqual(dense.bias.name, 'my_dense/bias:0')

  def testNoBias(self):
    dense = core_layers.Dense(2, use_bias=False, name='my_dense')
    inputs = tf.random_uniform((5, 2), seed=1)
    _ = dense(inputs)
    self.assertListEqual(dense.weights, [dense.w])
    self.assertListEqual(dense.trainable_weights, [dense.w])
    self.assertListEqual(dense.non_trainable_weights, [])
    self.assertEqual(
        len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)), 1)
    self.assertEqual(dense.w.name, 'my_dense/weights:0')
    self.assertEqual(dense.bias, None)

  def testNonTrainable(self):
    dense = core_layers.Dense(2, trainable=False, name='my_dense')
    inputs = tf.random_uniform((5, 2), seed=1)
    _ = dense(inputs)
    self.assertListEqual(dense.weights, [dense.w, dense.bias])
    self.assertListEqual(dense.non_trainable_weights, [dense.w, dense.bias])
    self.assertListEqual(dense.trainable_weights, [])
    self.assertListEqual(dense._trainable_weights, [dense.w, dense.bias])
    self.assertListEqual(dense._non_trainable_weights, [])
    self.assertEqual(
        len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)), 0)

  def testOutputShape(self):
    dense = core_layers.Dense(7, activation=tf.nn.relu, name='my_dense')
    inputs = tf.random_uniform((5, 3), seed=1)
    outputs = dense.apply(inputs)
    self.assertEqual(outputs.get_shape().as_list(), [5, 7])

    inputs = tf.random_uniform((5, 2, 3), seed=1)
    outputs = dense(inputs)
    self.assertEqual(outputs.get_shape().as_list(), [5, 2, 7])

    inputs = tf.random_uniform((1, 2, 4, 3), seed=1)
    outputs = dense.apply(inputs)
    self.assertEqual(outputs.get_shape().as_list(), [1, 2, 4, 7])

  def testCallOnPlaceHolder(self):
    inputs = tf.placeholder(dtype=tf.float32)
    dense = core_layers.Dense(4, name='my_dense')
    with self.assertRaises(ValueError):
      dense(inputs)

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, None])
    dense = core_layers.Dense(4, name='my_dense')
    with self.assertRaises(ValueError):
      dense(inputs)

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
    dense = core_layers.Dense(4, name='my_dense')
    with self.assertRaises(ValueError):
      dense(inputs)

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    dense = core_layers.Dense(4, name='my_dense')
    dense(inputs)

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
    dense = core_layers.Dense(4, name='my_dense')
    dense(inputs)

  def testActivation(self):
    dense = core_layers.Dense(2, activation=tf.nn.relu, name='dense1')
    inputs = tf.random_uniform((5, 3), seed=1)
    outputs = dense(inputs)
    self.assertEqual(outputs.op.name, 'dense1/Relu')

    dense = core_layers.Dense(2, name='dense2')
    inputs = tf.random_uniform((5, 3), seed=1)
    outputs = dense(inputs)
    self.assertEqual(outputs.op.name, 'dense2/BiasAdd')

  def testActivityRegularizer(self):
    regularizer = lambda x: tf.reduce_sum(x) * 1e-3
    dense = core_layers.Dense(2, name='my_dense',
                              activity_regularizer=regularizer)
    inputs = tf.random_uniform((5, 3), seed=1)
    _ = dense(inputs)
    loss_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(dense.losses, loss_keys)

  def testWeightsRegularizer(self):
    regularizer = lambda x: tf.reduce_sum(x) * 1e-3
    dense = core_layers.Dense(2, name='my_dense',
                              weights_regularizer=regularizer)
    inputs = tf.random_uniform((5, 3), seed=1)
    _ = dense(inputs)
    loss_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(dense.losses, loss_keys)

  def testBiasRegularizer(self):
    regularizer = lambda x: tf.reduce_sum(x) * 1e-3
    dense = core_layers.Dense(2, name='my_dense',
                              bias_regularizer=regularizer)
    inputs = tf.random_uniform((5, 3), seed=1)
    _ = dense(inputs)
    loss_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(loss_keys), 1)
    self.assertListEqual(dense.losses, loss_keys)

  def testFunctionalDense(self):
    inputs = tf.random_uniform((5, 3), seed=1)
    outputs = core_layers.dense(
        inputs, 2, activation=tf.nn.relu, name='my_dense')
    self.assertEqual(
        len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)), 2)
    self.assertEqual(outputs.op.name, 'my_dense/Relu')
    self.assertEqual(outputs.get_shape().as_list(), [5, 2])

  def testFunctionalDenseTwice(self):
    inputs = tf.random_uniform((5, 3), seed=1)
    core_layers.dense(inputs, 2)
    vars1 = tf.trainable_variables()
    core_layers.dense(inputs, 2)
    vars2 = tf.trainable_variables()
    self.assertEqual(len(vars1), 2)
    self.assertEqual(len(vars2), 4)

  def testFunctionalDenseTwiceReuse(self):
    inputs = tf.random_uniform((5, 3), seed=1)
    core_layers.dense(inputs, 2, name='my_dense')
    vars1 = tf.trainable_variables()
    core_layers.dense(inputs, 2, name='my_dense', reuse=True)
    vars2 = tf.trainable_variables()
    self.assertEqual(vars1, vars2)

  def testFunctionalDenseWithCustomGetter(self):
    called = [0]
    def custom_getter(getter, *args, **kwargs):
      called[0] += 1
      return getter(*args, **kwargs)
    with tf.variable_scope('test', custom_getter=custom_getter):
      inputs = tf.random_uniform((5, 3), seed=1)
      core_layers.dense(inputs, 2)
    self.assertEqual(called[0], 2)

  def testFunctionalDenseInScope(self):
    with tf.variable_scope('test'):
      inputs = tf.random_uniform((5, 3), seed=1)
      core_layers.dense(inputs, 2, name='my_dense')
      var = tf.trainable_variables()[0]
      self.assertEqual(var.name, 'test/my_dense/weights:0')
    with tf.variable_scope('test1') as scope:
      inputs = tf.random_uniform((5, 3), seed=1)
      core_layers.dense(inputs, 2, name=scope)
      var = tf.trainable_variables()[2]
      self.assertEqual(var.name, 'test1/weights:0')
    with tf.variable_scope('test2'):
      inputs = tf.random_uniform((5, 3), seed=1)
      core_layers.dense(inputs, 2)
      var = tf.trainable_variables()[4]
      self.assertEqual(var.name, 'test2/dense/weights:0')


class DropoutTest(tf.test.TestCase):

  def testDropoutProperties(self):
    dp = core_layers.Dropout(0.5)
    self.assertEqual(dp.rate, 0.5)
    self.assertEqual(dp.name, 'dropout')
    self.assertEqual(dp.noise_shape, None)

  def testBooleanLearningPhase(self):
    with self.test_session() as sess:
      dp = core_layers.Dropout(0.5)
      inputs = tf.ones((5, 3))
      dropped = dp.apply(inputs, training=True)
      sess.run(tf.global_variables_initializer())
      np_output = sess.run(dropped)
      self.assertAlmostEqual(0., np_output.min())
      dropped = dp.apply(inputs, training=False)
      np_output = sess.run(dropped)
      self.assertAllClose(np.ones((5, 3)), np_output)

  def testDynamicLearningPhase(self):
    with self.test_session() as sess:
      dp = core_layers.Dropout(0.5, seed=1)
      inputs = tf.ones((5, 5))
      training = tf.placeholder(dtype='bool')
      dropped = dp.apply(inputs, training=training)
      sess.run(tf.global_variables_initializer())
      np_output = sess.run(dropped, feed_dict={training: True})
      self.assertAlmostEqual(0., np_output.min())
      np_output = sess.run(dropped, feed_dict={training: False})
      self.assertAllClose(np.ones((5, 5)), np_output)

  def testCustomNoiseShape(self):
    with self.test_session() as sess:
      inputs = tf.ones((5, 3, 2))
      noise_shape = [5, 1, 2]
      dp = core_layers.Dropout(0.5, noise_shape=noise_shape, seed=1)
      dropped = dp.apply(inputs, training=True)
      sess.run(tf.global_variables_initializer())
      np_output = sess.run(dropped)
      self.assertAlmostEqual(0., np_output.min())
      self.assertAllClose(np_output[:, 0, :], np_output[:, 1, :])

  def testFunctionalDropout(self):
    with self.test_session() as sess:
      inputs = tf.ones((5, 5))
      training = tf.placeholder(dtype='bool')
      dropped = core_layers.dropout(inputs, 0.5, training=training, seed=1)
      self.assertEqual(dropped.op.name, 'dropout/cond/Merge')

      sess.run(tf.global_variables_initializer())
      np_output = sess.run(dropped, feed_dict={training: True})
      self.assertAlmostEqual(0., np_output.min())
      np_output = sess.run(dropped, feed_dict={training: False})
      self.assertAllClose(np.ones((5, 5)), np_output)


if __name__ == '__main__':
  tf.test.main()
