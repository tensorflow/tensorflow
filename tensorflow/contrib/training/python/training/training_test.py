# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tf.contrib.training.training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf


def logistic_classifier(inputs):
  return tf.contrib.layers.fully_connected(
      inputs, 1, activation_fn=tf.sigmoid)


def batchnorm_classifier(inputs):
  inputs = tf.contrib.layers.batch_norm(inputs, decay=0.1)
  return tf.contrib.layers.fully_connected(inputs, 1, activation_fn=tf.sigmoid)


class CreateTrainOpTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(0)

    # Create an easy training set:
    self._inputs = np.random.rand(16, 4).astype(np.float32)
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

  def testUseUpdateOps(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      expected_mean = np.mean(self._inputs, axis=(0))
      expected_var = np.var(self._inputs, axis=(0))

      tf_predictions = batchnorm_classifier(tf_inputs)
      tf.contrib.losses.log_loss(tf_predictions, tf_labels)
      total_loss = tf.contrib.losses.get_total_loss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = tf.contrib.training.create_train_op(total_loss, optimizer)

      moving_mean = tf.contrib.framework.get_variables_by_name('moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables_by_name(
          'moving_variance')[0]

      with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        mean, variance = sess.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

        for _ in range(10):
          sess.run([train_op])
        mean = moving_mean.eval()
        variance = moving_variance.eval()
        # After 10 updates with decay 0.1 moving_mean == expected_mean and
        # moving_variance == expected_var.
        self.assertAllClose(mean, expected_mean)
        self.assertAllClose(variance, expected_var)

  def testEmptyUpdateOps(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      tf.contrib.losses.log_loss(tf_predictions, tf_labels)
      total_loss = tf.contrib.losses.get_total_loss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = tf.contrib.training.create_train_op(
          total_loss, optimizer, update_ops=[])

      moving_mean = tf.contrib.framework.get_variables_by_name('moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables_by_name(
          'moving_variance')[0]

      with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        mean, variance = sess.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

        for _ in range(10):
          sess.run([train_op])
        mean = moving_mean.eval()
        variance = moving_variance.eval()

        # Since we skip update_ops the moving_vars are not updated.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)


class TrainBNClassifierTest(tf.test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)
    self._logdir = os.path.join(self.get_temp_dir(), 'tmp_bnlogs/')

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testTrainWithNoInitAssignCanAchieveZeroLoss(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      tf.contrib.losses.log_loss(tf_predictions, tf_labels)
      total_loss = tf.contrib.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = tf.contrib.training.create_train_op(
          total_loss, optimizer)

      loss = tf.contrib.training.train(
          train_op, self._logdir, hooks=[
              tf.train.StopAtStepHook(num_steps=300)
          ])
      self.assertLess(loss, .1)


class TrainTest(tf.test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testCanAchieveZeroLoss(self):
    logdir = os.path.join(self.get_temp_dir(), 'can_achieve_zero_loss')

    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = logistic_classifier(tf_inputs)
      tf.contrib.losses.log_loss(tf_predictions, tf_labels)
      total_loss = tf.contrib.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = tf.contrib.training.create_train_op(total_loss, optimizer)

      loss = tf.contrib.training.train(
          train_op, logdir, hooks=[
              tf.train.StopAtStepHook(num_steps=300)
          ])
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testTrainWithLocalVariable(self):
    logdir = os.path.join(self.get_temp_dir(), 'train_with_local_variable')

    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      local_multiplier = tf.contrib.framework.local_variable(1.0)

      tf_predictions = logistic_classifier(tf_inputs) * local_multiplier
      tf.contrib.losses.log_loss(tf_predictions, tf_labels)
      total_loss = tf.contrib.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = tf.contrib.training.create_train_op(
          total_loss, optimizer)

      loss = tf.contrib.training.train(
          train_op, logdir, hooks=[
              tf.train.StopAtStepHook(num_steps=300)
          ])
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testResumeTrainAchievesRoughlyTheSameLoss(self):
    number_of_steps = [300, 1, 5]
    logdir = os.path.join(self.get_temp_dir(), 'resume_train_same_loss')

    for i in range(len(number_of_steps)):
      with tf.Graph().as_default():
        tf.set_random_seed(i)
        tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
        tf_labels = tf.constant(self._labels, dtype=tf.float32)

        tf_predictions = logistic_classifier(tf_inputs)
        tf.contrib.losses.log_loss(tf_predictions, tf_labels)
        total_loss = tf.contrib.losses.get_total_loss()

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        train_op = tf.contrib.training.create_train_op(
            total_loss, optimizer)

        saver = tf.train.Saver()

        loss = tf.contrib.training.train(
            train_op, logdir, hooks=[
                tf.train.StopAtStepHook(num_steps=number_of_steps[i]),
                tf.train.CheckpointSaverHook(
                    logdir, save_steps=50, saver=saver),
            ])
        self.assertIsNotNone(loss)
        self.assertLess(loss, .015)

  def create_train_op(self, learning_rate=1.0, gradient_multiplier=1.0):
    tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
    tf_labels = tf.constant(self._labels, dtype=tf.float32)

    tf_predictions = logistic_classifier(tf_inputs)
    tf.contrib.losses.log_loss(tf_predictions, tf_labels)
    total_loss = tf.contrib.losses.get_total_loss()

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)

    def transform_grads_fn(grads):
      if gradient_multiplier != 1.0:
        variables = tf.trainable_variables()
        gradient_multipliers = {var: gradient_multiplier for var in variables}

        with tf.name_scope('multiply_grads'):
          return tf.contrib.training.multiply_gradients(
              grads, gradient_multipliers)
      else:
        return grads

    return tf.contrib.training.create_train_op(
        total_loss, optimizer, transform_grads_fn=transform_grads_fn)

  def testTrainWithInitFromCheckpoint(self):
    logdir1 = os.path.join(self.get_temp_dir(), 'tmp_logs1/')
    logdir2 = os.path.join(self.get_temp_dir(), 'tmp_logs2/')

    if tf.gfile.Exists(logdir1):  # For running on jenkins.
      tf.gfile.DeleteRecursively(logdir1)
    if tf.gfile.Exists(logdir2):  # For running on jenkins.
      tf.gfile.DeleteRecursively(logdir2)

    # First, train the model one step (make sure the error is high).
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      train_op = self.create_train_op()
      saver = tf.train.Saver()
      loss = tf.contrib.training.train(
          train_op, logdir1, hooks=[
              tf.train.CheckpointSaverHook(logdir1, save_steps=1, saver=saver),
              tf.train.StopAtStepHook(num_steps=1),
          ], save_checkpoint_secs=None)
      self.assertGreater(loss, .5)

    # Next, train the model to convergence.
    with tf.Graph().as_default():
      tf.set_random_seed(1)
      train_op = self.create_train_op()
      saver = tf.train.Saver()
      loss = tf.contrib.training.train(
          train_op, logdir1, hooks=[
              tf.train.CheckpointSaverHook(logdir1, save_steps=1, saver=saver),
              tf.train.StopAtStepHook(num_steps=300),
          ], save_checkpoint_secs=None)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

    # Finally, advance the model a single step and validate that the loss is
    # still low.
    with tf.Graph().as_default():
      tf.set_random_seed(2)
      train_op = self.create_train_op()

      model_variables = tf.global_variables()
      model_path = os.path.join(logdir1, 'model.ckpt-300')

      assign_fn = tf.contrib.framework.assign_from_checkpoint_fn(
          model_path, model_variables)
      def init_fn(_, session):
        assign_fn(session)

      loss = tf.contrib.training.train(
          train_op,
          logdir2,
          scaffold=tf.train.Scaffold(init_fn=init_fn),
          hooks=[tf.train.StopAtStepHook(num_steps=1)])

      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

  def ModelLoss(self):
    tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
    tf_labels = tf.constant(self._labels, dtype=tf.float32)

    tf_predictions = logistic_classifier(tf_inputs)
    tf.contrib.losses.log_loss(tf_predictions, tf_labels)
    return tf.contrib.losses.get_total_loss()

  def testTrainAllVarsHasLowerLossThanTrainSubsetOfVars(self):
    logdir = os.path.join(self.get_temp_dir(), 'tmp_logs3/')
    if tf.gfile.Exists(logdir):  # For running on jenkins.
      tf.gfile.DeleteRecursively(logdir)

    # First, train only the weights of the model.
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      weights = tf.contrib.framework.get_variables_by_name('weights')

      train_op = tf.contrib.training.create_train_op(
          total_loss,
          optimizer,
          variables_to_train=weights)

      saver = tf.train.Saver()
      loss = tf.contrib.training.train(
          train_op, logdir, hooks=[
              tf.train.CheckpointSaverHook(logdir, save_steps=1, saver=saver),
              tf.train.StopAtStepHook(num_steps=200),
          ])
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Next, train the biases of the model.
    with tf.Graph().as_default():
      tf.set_random_seed(1)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      biases = tf.contrib.framework.get_variables_by_name('biases')

      train_op = tf.contrib.training.create_train_op(
          total_loss,
          optimizer,
          variables_to_train=biases)

      saver = tf.train.Saver()
      loss = tf.contrib.training.train(
          train_op, logdir, hooks=[
              tf.train.CheckpointSaverHook(logdir, save_steps=1, saver=saver),
              tf.train.StopAtStepHook(num_steps=300),
          ])
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Finally, train both weights and bias to get lower loss.
    with tf.Graph().as_default():
      tf.set_random_seed(2)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = tf.contrib.training.create_train_op(total_loss, optimizer)
      saver = tf.train.Saver()
      loss = tf.contrib.training.train(
          train_op, logdir, hooks=[
              tf.train.CheckpointSaverHook(logdir, save_steps=1, saver=saver),
              tf.train.StopAtStepHook(num_steps=400),
          ])
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testTrainingSubsetsOfVariablesOnlyUpdatesThoseVariables(self):
    # First, train only the weights of the model.
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      weights, biases = tf.contrib.framework.get_variables()

      train_op = tf.contrib.training.create_train_op(total_loss, optimizer)
      train_weights = tf.contrib.training.create_train_op(
          total_loss, optimizer, variables_to_train=[weights])
      train_biases = tf.contrib.training.create_train_op(
          total_loss, optimizer, variables_to_train=[biases])

      with tf.Session() as sess:
        # Initialize the variables.
        sess.run(tf.global_variables_initializer())

        # Get the intial weights and biases values.
        weights_values, biases_values = sess.run([weights, biases])
        self.assertGreater(np.linalg.norm(weights_values), 0)
        self.assertAlmostEqual(np.linalg.norm(biases_values), 0)

        # Update weights and biases.
        loss = sess.run(train_op)
        self.assertGreater(loss, .5)
        new_weights, new_biases = sess.run([weights, biases])

        # Check that the weights and biases have been updated.
        self.assertGreater(np.linalg.norm(weights_values - new_weights), 0)
        self.assertGreater(np.linalg.norm(biases_values - new_biases), 0)

        weights_values, biases_values = new_weights, new_biases

        # Update only weights.
        loss = sess.run(train_weights)
        self.assertGreater(loss, .5)
        new_weights, new_biases = sess.run([weights, biases])

        # Check that the weights have been updated, but biases have not.
        self.assertGreater(np.linalg.norm(weights_values - new_weights), 0)
        self.assertAlmostEqual(np.linalg.norm(biases_values - new_biases), 0)
        weights_values = new_weights

        # Update only biases.
        loss = sess.run(train_biases)
        self.assertGreater(loss, .5)
        new_weights, new_biases = sess.run([weights, biases])

        # Check that the biases have been updated, but weights have not.
        self.assertAlmostEqual(np.linalg.norm(weights_values - new_weights), 0)
        self.assertGreater(np.linalg.norm(biases_values - new_biases), 0)

  def testTrainWithAlteredGradients(self):
    # Use the same learning rate but different gradient multipliers
    # to train two models. Model with equivalently larger learning
    # rate (i.e., learning_rate * gradient_multiplier) has smaller
    # training loss.
    logdir1 = os.path.join(self.get_temp_dir(), 'tmp_logs6/')
    logdir2 = os.path.join(self.get_temp_dir(), 'tmp_logs7/')

    if tf.gfile.Exists(logdir1):
      tf.gfile.DeleteRecursively(logdir1)
    if tf.gfile.Exists(logdir2):
      tf.gfile.DeleteRecursively(logdir2)

    multipliers = [1., 1000.]
    number_of_steps = 10
    losses = []
    learning_rate = 0.001

    # First, train the model with equivalently smaller learning rate.
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      train_op = self.create_train_op(
          learning_rate=learning_rate,
          gradient_multiplier=multipliers[0])

      saver = tf.train.Saver()

      loss = tf.contrib.training.train(
          train_op, logdir1, hooks=[
              tf.train.StopAtStepHook(num_steps=number_of_steps),
              tf.train.CheckpointSaverHook(logdir1, save_steps=50, saver=saver),
          ])

      losses.append(loss)
      self.assertGreater(loss, .5)

    # Second, train the model with equivalently larger learning rate.
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      train_op = self.create_train_op(
          learning_rate=learning_rate,
          gradient_multiplier=multipliers[1])
      saver = tf.train.Saver()

      loss = tf.contrib.training.train(
          train_op, logdir2, hooks=[
              tf.train.StopAtStepHook(num_steps=number_of_steps),
              tf.train.CheckpointSaverHook(logdir2, save_steps=50, saver=saver),
          ])

      losses.append(loss)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .5)

    # The loss of the model trained with larger learning rate should
    # be smaller.
    self.assertGreater(losses[0], losses[1])


if __name__ == '__main__':
  tf.test.main()
