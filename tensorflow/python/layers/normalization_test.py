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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class BNTest(test.TestCase):

  def testCreateBN(self):
    # Call layer.
    bn = normalization_layers.BatchNormalization(axis=1)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    # Verify shape.
    self.assertListEqual(outputs.get_shape().as_list(), [5, 4, 3])

    # Verify layer attributes.
    self.assertEqual(len(bn.updates), 2)
    self.assertEqual(len(bn.variables), 4)
    self.assertEqual(len(bn.trainable_variables), 2)
    self.assertEqual(len(bn.non_trainable_variables), 2)

    # Test that updates were created and added to UPDATE_OPS.
    self.assertEqual(len(bn.updates), 2)
    self.assertListEqual(
        ops.get_collection(ops.GraphKeys.UPDATE_OPS), bn.updates)

    # Test that weights were created and added to TRAINABLE_VARIABLES.
    self.assertListEqual(
        ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES),
        bn.trainable_variables)

  def test3DInputAxis1(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=1, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())

      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 4, 1))
      np_beta = np.reshape(np_beta, (1, 4, 1))

      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 2))
      std = np.std(np_inputs, axis=(0, 2))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test3DInputAxis2(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=2, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 3))
      np_beta = np.reshape(np_beta, (1, 1, 3))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1))
      std = np.std(np_inputs, axis=(0, 1))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test4DInputAxis1(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=1, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 4, 1, 1))
      np_beta = np.reshape(np_beta, (1, 4, 1, 1))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 2, 3))
      std = np.std(np_inputs, axis=(0, 2, 3))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test4DInputAxis2(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=2, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 3, 1))
      np_beta = np.reshape(np_beta, (1, 1, 3, 1))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1, 3))
      std = np.std(np_inputs, axis=(0, 1, 3))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def test4DInputAxis3(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=3, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1, 2))
      std = np.std(np_inputs, axis=(0, 1, 2))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testNegativeAxis(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=-1, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + bn.updates,
                                   feed_dict={training: True})

        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1, 2))
      std = np.std(np_inputs, axis=(0, 1, 2))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testBooleanLearningPhase(self):
    epsilon = 1e-3
    bn = normalization_layers.BatchNormalization(
        axis=-1, epsilon=epsilon, momentum=0.9)
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)) + 100, dtype=dtypes.float32)
    outputs_training = bn.apply(inputs, training=True)
    outputs_infer = bn.apply(inputs, training=False)

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([bn.gamma, bn.beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs_training] + bn.updates)
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=2)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      moving_mean, moving_var = sess.run([bn.moving_mean, bn.moving_variance])
      np_inputs = sess.run(inputs)
      mean = np.mean(np_inputs, axis=(0, 1, 2))
      std = np.std(np_inputs, axis=(0, 1, 2))
      variance = np.square(std)
      self.assertAllClose(mean, moving_mean, atol=1e-2)
      self.assertAllClose(variance, moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs_infer)

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testFunctionalNoReuse(self):
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)), dtype=dtypes.float32)
    epsilon = 1e-3
    training = array_ops.placeholder(dtype='bool')
    outputs = normalization_layers.batch_norm(
        inputs,
        axis=-1,
        momentum=0.9,
        epsilon=epsilon,
        training=training,
        name='bn')

    updates = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    all_vars = dict([(v.name, v) for v in variables.global_variables()])
    moving_mean = all_vars['bn/moving_mean:0']
    moving_variance = all_vars['bn/moving_variance:0']
    beta = all_vars['bn/beta:0']
    gamma = all_vars['bn/gamma:0']

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      np_gamma, np_beta = sess.run([gamma, beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      for _ in range(100):
        np_output, _, _ = sess.run([outputs] + updates,
                                   feed_dict={training: True})
        # Verify that the axis is normalized during training.
        normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
        self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
        self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Verify that the statistics are updated during training.
      np_moving_mean, np_moving_var = sess.run([moving_mean, moving_variance])
      np_inputs = sess.run(inputs)
      np_mean = np.mean(np_inputs, axis=(0, 1, 2))
      np_std = np.std(np_inputs, axis=(0, 1, 2))
      np_variance = np.square(np_std)
      self.assertAllClose(np_mean, np_moving_mean, atol=1e-2)
      self.assertAllClose(np_variance, np_moving_var, atol=1e-2)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=1)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testFunctionalReuse(self):
    inputs1 = variables.Variable(
        np.random.random((5, 4, 3, 6)), dtype=dtypes.float32)
    inputs2 = variables.Variable(
        np.random.random((5, 4, 3, 6)), dtype=dtypes.float32)
    epsilon = 1e-3
    training = array_ops.placeholder(dtype='bool')
    _ = normalization_layers.batch_norm(
        inputs1,
        axis=-1,
        momentum=0.9,
        epsilon=epsilon,
        training=training,
        name='bn')
    outputs2 = normalization_layers.batch_norm(
        inputs2,
        axis=-1,
        momentum=0.9,
        epsilon=epsilon,
        training=training,
        name='bn',
        reuse=True)

    # Last 2 update ops
    updates = ops.get_collection(ops.GraphKeys.UPDATE_OPS)[-2:]
    all_vars = dict([(v.name, v) for v in variables.global_variables()])
    moving_mean = all_vars['bn/moving_mean:0']
    moving_variance = all_vars['bn/moving_variance:0']
    beta = all_vars['bn/beta:0']
    gamma = all_vars['bn/gamma:0']

    with self.test_session() as sess:
      # Test training with placeholder learning phase.
      sess.run(variables.global_variables_initializer())
      for _ in range(100):
        np_output, _, _ = sess.run([outputs2] + updates,
                                   feed_dict={training: True})

      # Verify that the statistics are updated during training.
      np_moving_mean, np_moving_var = sess.run([moving_mean, moving_variance])
      np_inputs = sess.run(inputs2)
      np_mean = np.mean(np_inputs, axis=(0, 1, 2))
      np_std = np.std(np_inputs, axis=(0, 1, 2))
      np_variance = np.square(np_std)
      self.assertAllClose(np_mean, np_moving_mean, atol=1e-2)
      self.assertAllClose(np_variance, np_moving_var, atol=1e-2)

      # Verify that the axis is normalized during training.
      np_gamma, np_beta = sess.run([gamma, beta])
      np_gamma = np.reshape(np_gamma, (1, 1, 1, 6))
      np_beta = np.reshape(np_beta, (1, 1, 1, 6))
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=2)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

      # Test inference with placeholder learning phase.
      np_output = sess.run(outputs2, feed_dict={training: False})

      # Verify that the axis is normalized during inference.
      normed_np_output = ((np_output - epsilon) * np_gamma) + np_beta
      self.assertAlmostEqual(np.mean(normed_np_output), 0., places=2)
      self.assertAlmostEqual(np.std(normed_np_output), 1., places=1)

  def testFunctionalReuseFromScope(self):
    inputs = variables.Variable(
        np.random.random((5, 4, 3, 6)), dtype=dtypes.float32)
    epsilon = 1e-3
    training = array_ops.placeholder(dtype='bool')
    with variable_scope.variable_scope('scope'):
      _ = normalization_layers.batch_norm(
          inputs, axis=-1, momentum=0.9, epsilon=epsilon, training=training)
      self.assertEqual(len(variables.global_variables()), 5)
    with variable_scope.variable_scope('scope', reuse=True):
      _ = normalization_layers.batch_norm(
          inputs, axis=-1, momentum=0.9, epsilon=epsilon, training=training)
      self.assertEqual(len(variables.global_variables()), 5)

  def testNoCenter(self):
    bn = normalization_layers.BatchNormalization(axis=1, center=False)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    # Verify shape.
    self.assertListEqual(outputs.get_shape().as_list(), [5, 4, 3])

    # Verify layer attributes.
    self.assertEqual(len(bn.updates), 2)
    self.assertEqual(len(bn.variables), 3)
    self.assertEqual(len(bn.trainable_variables), 1)
    self.assertEqual(len(bn.non_trainable_variables), 2)

  def testNoScale(self):
    bn = normalization_layers.BatchNormalization(axis=1, scale=False)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    outputs = bn.apply(inputs, training=training)

    # Verify shape.
    self.assertListEqual(outputs.get_shape().as_list(), [5, 4, 3])

    # Verify layer attributes.
    self.assertEqual(len(bn.updates), 2)
    self.assertEqual(len(bn.variables), 3)
    self.assertEqual(len(bn.trainable_variables), 1)
    self.assertEqual(len(bn.non_trainable_variables), 2)

  def testRegularizers(self):
    reg = lambda x: 0.1 * math_ops.reduce_sum(x)
    bn = normalization_layers.BatchNormalization(axis=1, beta_regularizer=reg)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    _ = bn.apply(inputs, training=training)
    self.assertEqual(len(bn.losses), 1)

    bn = normalization_layers.BatchNormalization(axis=1, gamma_regularizer=reg)
    inputs = random_ops.random_uniform((5, 4, 3), seed=1)
    training = array_ops.placeholder(dtype='bool')
    _ = bn.apply(inputs, training=training)
    self.assertEqual(len(bn.losses), 1)


if __name__ == '__main__':
  test.main()
