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

from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.training.python.training import training
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_lib2
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
# pylint: enable=g-import-not-at-top


def logistic_classifier(inputs):
  return layers.fully_connected(inputs, 1, activation_fn=math_ops.sigmoid)


def batchnorm_classifier(inputs):
  inputs = layers.batch_norm(inputs, decay=0.1)
  return layers.fully_connected(inputs, 1, activation_fn=math_ops.sigmoid)


class CreateTrainOpTest(test.TestCase):

  def setUp(self):
    np.random.seed(0)

    # Create an easy training set:
    self._inputs = np.random.rand(16, 4).astype(np.float32)
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

  def testTrainOpInCollection(self):
    with ops.Graph().as_default():
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss = losses.log_loss(tf_labels, tf_predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = training.create_train_op(loss, optimizer)

      # Make sure the training op was recorded in the proper collection
      self.assertTrue(train_op in ops.get_collection(ops.GraphKeys.TRAIN_OP))

  def testUseUpdateOps(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      expected_mean = np.mean(self._inputs, axis=(0))
      expected_var = np.var(self._inputs, axis=(0))

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss = losses.log_loss(tf_labels, tf_predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(loss, optimizer)

      moving_mean = variables_lib.get_variables_by_name('moving_mean')[0]
      moving_variance = variables_lib.get_variables_by_name('moving_variance')[
          0]

      with self.test_session() as session:
        # Initialize all variables
        session.run(variables_lib2.global_variables_initializer())
        mean, variance = session.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

        for _ in range(10):
          session.run(train_op)

        mean = moving_mean.eval()
        variance = moving_variance.eval()
        # After 10 updates with decay 0.1 moving_mean == expected_mean and
        # moving_variance == expected_var.
        self.assertAllClose(mean, expected_mean)
        self.assertAllClose(variance, expected_var)

  def testEmptyUpdateOps(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss = losses.log_loss(tf_labels, tf_predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = training.create_train_op(loss, optimizer, update_ops=[])

      moving_mean = variables_lib.get_variables_by_name('moving_mean')[0]
      moving_variance = variables_lib.get_variables_by_name('moving_variance')[
          0]

      with self.test_session() as session:
        # Initialize all variables
        session.run(variables_lib2.global_variables_initializer())
        mean, variance = session.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

        for _ in range(10):
          session.run(train_op)

        mean = moving_mean.eval()
        variance = moving_variance.eval()

        # Since we skip update_ops the moving_vars are not updated.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

  def testGlobalStepIsIncrementedByDefault(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss = losses.log_loss(tf_labels, tf_predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = training.create_train_op(loss, optimizer)

      global_step = variables_lib.get_or_create_global_step()

      with self.test_session() as session:
        # Initialize all variables
        session.run(variables_lib2.global_variables_initializer())

        for _ in range(10):
          session.run(train_op)

        # After 10 updates global_step should be 10.
        self.assertAllClose(global_step.eval(), 10)

  def testGlobalStepNotIncrementedWhenSetToNone(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss = losses.log_loss(tf_labels, tf_predictions)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = training.create_train_op(loss, optimizer, global_step=None)

      global_step = variables_lib.get_or_create_global_step()

      with self.test_session() as session:
        # Initialize all variables
        session.run(variables_lib2.global_variables_initializer())

        for _ in range(10):
          session.run(train_op)

        # Since train_op don't use global_step it shouldn't change.
        self.assertAllClose(global_step.eval(), 0)


class TrainBatchNormClassifierTest(test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testTrainWithNoInitAssignCanAchieveZeroLoss(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      losses.log_loss(tf_labels, tf_predictions)
      total_loss = losses.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(total_loss, optimizer)

      loss = training.train(
          train_op,
          None,
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps=300)],
          save_summaries_steps=None,
          save_checkpoint_secs=None)
      self.assertLess(loss, .1)


class TrainTest(test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testCanAchieveZeroLoss(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = logistic_classifier(tf_inputs)
      losses.log_loss(tf_labels, tf_predictions)
      total_loss = losses.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = training.create_train_op(total_loss, optimizer)

      loss = training.train(
          train_op,
          None,
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps=300)],
          save_summaries_steps=None,
          save_checkpoint_secs=None)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testTrainWithLocalVariable(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      local_multiplier = variables_lib.local_variable(1.0)

      tf_predictions = logistic_classifier(tf_inputs) * local_multiplier
      losses.log_loss(tf_labels, tf_predictions)
      total_loss = losses.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = training.create_train_op(total_loss, optimizer)

      loss = training.train(
          train_op,
          None,
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps=300)],
          save_summaries_steps=None,
          save_checkpoint_secs=None)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testResumeTrainAchievesRoughlyTheSameLoss(self):
    number_of_steps = [300, 1, 5]
    logdir = os.path.join(self.get_temp_dir(), 'resume_train_same_loss')

    for i in range(len(number_of_steps)):
      with ops.Graph().as_default():
        random_seed.set_random_seed(i)
        tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
        tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

        tf_predictions = logistic_classifier(tf_inputs)
        losses.log_loss(tf_labels, tf_predictions)
        total_loss = losses.get_total_loss()

        optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

        train_op = training.create_train_op(total_loss, optimizer)

        saver = saver_lib.Saver()

        loss = training.train(
            train_op,
            logdir,
            hooks=[
                basic_session_run_hooks.StopAtStepHook(
                    num_steps=number_of_steps[i]),
                basic_session_run_hooks.CheckpointSaverHook(
                    logdir, save_steps=50, saver=saver),
            ],
            save_checkpoint_secs=None,
            save_summaries_steps=None)
        self.assertIsNotNone(loss)
        self.assertLess(loss, .015)

  def create_train_op(self, learning_rate=1.0, gradient_multiplier=1.0):
    tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

    tf_predictions = logistic_classifier(tf_inputs)
    losses.log_loss(tf_labels, tf_predictions)
    total_loss = losses.get_total_loss()

    optimizer = gradient_descent.GradientDescentOptimizer(
        learning_rate=learning_rate)

    def transform_grads_fn(grads):
      if gradient_multiplier != 1.0:
        variables = variables_lib2.trainable_variables()
        gradient_multipliers = {var: gradient_multiplier for var in variables}

        with ops.name_scope('multiply_grads'):
          return training.multiply_gradients(grads, gradient_multipliers)
      else:
        return grads

    return training.create_train_op(
        total_loss, optimizer, transform_grads_fn=transform_grads_fn)

  def testTrainWithInitFromCheckpoint(self):
    logdir1 = os.path.join(self.get_temp_dir(), 'tmp_logs1/')
    logdir2 = os.path.join(self.get_temp_dir(), 'tmp_logs2/')

    if gfile.Exists(logdir1):  # For running on jenkins.
      gfile.DeleteRecursively(logdir1)
    if gfile.Exists(logdir2):  # For running on jenkins.
      gfile.DeleteRecursively(logdir2)

    # First, train the model one step (make sure the error is high).
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      train_op = self.create_train_op()
      saver = saver_lib.Saver()
      loss = training.train(
          train_op,
          logdir1,
          hooks=[
              basic_session_run_hooks.CheckpointSaverHook(
                  logdir1, save_steps=1, saver=saver),
              basic_session_run_hooks.StopAtStepHook(num_steps=1),
          ],
          save_checkpoint_secs=None,
          save_summaries_steps=None)
      self.assertGreater(loss, .5)

    # Next, train the model to convergence.
    with ops.Graph().as_default():
      random_seed.set_random_seed(1)
      train_op = self.create_train_op()
      saver = saver_lib.Saver()
      loss = training.train(
          train_op,
          logdir1,
          hooks=[
              basic_session_run_hooks.CheckpointSaverHook(
                  logdir1, save_steps=300, saver=saver),
              basic_session_run_hooks.StopAtStepHook(num_steps=300),
          ],
          save_checkpoint_secs=None,
          save_summaries_steps=None)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

    # Finally, advance the model a single step and validate that the loss is
    # still low.
    with ops.Graph().as_default():
      random_seed.set_random_seed(2)
      train_op = self.create_train_op()

      model_variables = variables_lib2.global_variables()
      model_path = saver_lib.latest_checkpoint(logdir1)

      assign_fn = variables_lib.assign_from_checkpoint_fn(
          model_path, model_variables)

      def init_fn(_, session):
        assign_fn(session)

      loss = training.train(
          train_op,
          None,
          scaffold=monitored_session.Scaffold(init_fn=init_fn),
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps=1)],
          save_checkpoint_secs=None,
          save_summaries_steps=None)

      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

  def ModelLoss(self):
    tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

    tf_predictions = logistic_classifier(tf_inputs)
    losses.log_loss(tf_labels, tf_predictions)
    return losses.get_total_loss()

  def testTrainAllVarsHasLowerLossThanTrainSubsetOfVars(self):
    logdir = os.path.join(self.get_temp_dir(), 'tmp_logs3/')
    if gfile.Exists(logdir):  # For running on jenkins.
      gfile.DeleteRecursively(logdir)

    # First, train only the weights of the model.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      total_loss = self.ModelLoss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      weights = variables_lib.get_variables_by_name('weights')

      train_op = training.create_train_op(
          total_loss, optimizer, variables_to_train=weights)

      saver = saver_lib.Saver()
      loss = training.train(
          train_op,
          logdir,
          hooks=[
              basic_session_run_hooks.CheckpointSaverHook(
                  logdir, save_steps=200, saver=saver),
              basic_session_run_hooks.StopAtStepHook(num_steps=200),
          ],
          save_checkpoint_secs=None,
          save_summaries_steps=None)
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Next, train the biases of the model.
    with ops.Graph().as_default():
      random_seed.set_random_seed(1)
      total_loss = self.ModelLoss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      biases = variables_lib.get_variables_by_name('biases')

      train_op = training.create_train_op(
          total_loss, optimizer, variables_to_train=biases)

      saver = saver_lib.Saver()
      loss = training.train(
          train_op,
          logdir,
          hooks=[
              basic_session_run_hooks.CheckpointSaverHook(
                  logdir, save_steps=300, saver=saver),
              basic_session_run_hooks.StopAtStepHook(num_steps=300),
          ],
          save_checkpoint_secs=None,
          save_summaries_steps=None)
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Finally, train both weights and bias to get lower loss.
    with ops.Graph().as_default():
      random_seed.set_random_seed(2)
      total_loss = self.ModelLoss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(total_loss, optimizer)
      saver = saver_lib.Saver()
      loss = training.train(
          train_op,
          logdir,
          hooks=[
              basic_session_run_hooks.StopAtStepHook(num_steps=400),
          ],
          save_checkpoint_secs=None,
          save_summaries_steps=None)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testTrainingSubsetsOfVariablesOnlyUpdatesThoseVariables(self):
    # First, train only the weights of the model.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      total_loss = self.ModelLoss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      weights, biases = variables_lib.get_variables()

      train_op = training.create_train_op(total_loss, optimizer)
      train_weights = training.create_train_op(
          total_loss, optimizer, variables_to_train=[weights])
      train_biases = training.create_train_op(
          total_loss, optimizer, variables_to_train=[biases])

      with self.test_session() as session:
        # Initialize the variables.
        session.run(variables_lib2.global_variables_initializer())

        # Get the intial weights and biases values.
        weights_values, biases_values = session.run([weights, biases])
        self.assertGreater(np.linalg.norm(weights_values), 0)
        self.assertAlmostEqual(np.linalg.norm(biases_values), 0)

        # Update weights and biases.
        loss = session.run(train_op)
        self.assertGreater(loss, .5)
        new_weights, new_biases = session.run([weights, biases])

        # Check that the weights and biases have been updated.
        self.assertGreater(np.linalg.norm(weights_values - new_weights), 0)
        self.assertGreater(np.linalg.norm(biases_values - new_biases), 0)

        weights_values, biases_values = new_weights, new_biases

        # Update only weights.
        loss = session.run(train_weights)
        self.assertGreater(loss, .5)
        new_weights, new_biases = session.run([weights, biases])

        # Check that the weights have been updated, but biases have not.
        self.assertGreater(np.linalg.norm(weights_values - new_weights), 0)
        self.assertAlmostEqual(np.linalg.norm(biases_values - new_biases), 0)
        weights_values = new_weights

        # Update only biases.
        loss = session.run(train_biases)
        self.assertGreater(loss, .5)
        new_weights, new_biases = session.run([weights, biases])

        # Check that the biases have been updated, but weights have not.
        self.assertAlmostEqual(np.linalg.norm(weights_values - new_weights), 0)
        self.assertGreater(np.linalg.norm(biases_values - new_biases), 0)

  def testTrainWithAlteredGradients(self):
    # Use the same learning rate but different gradient multipliers
    # to train two models. Model with equivalently larger learning
    # rate (i.e., learning_rate * gradient_multiplier) has smaller
    # training loss.
    multipliers = [1., 1000.]
    number_of_steps = 10
    learning_rate = 0.001

    # First, train the model with equivalently smaller learning rate.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      train_op = self.create_train_op(
          learning_rate=learning_rate, gradient_multiplier=multipliers[0])

      loss0 = training.train(
          train_op,
          None,
          hooks=[
              basic_session_run_hooks.StopAtStepHook(num_steps=number_of_steps),
          ],
          save_checkpoint_secs=None,
          save_summaries_steps=None)
      self.assertIsNotNone(loss0)
      self.assertGreater(loss0, .5)

    # Second, train the model with equivalently larger learning rate.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      train_op = self.create_train_op(
          learning_rate=learning_rate, gradient_multiplier=multipliers[1])

      loss1 = training.train(
          train_op,
          None,
          hooks=[
              basic_session_run_hooks.StopAtStepHook(num_steps=number_of_steps),
          ],
          save_checkpoint_secs=None,
          save_summaries_steps=None)
      self.assertIsNotNone(loss1)
      self.assertLess(loss1, .5)

    # The loss of the model trained with larger learning rate should
    # be smaller.
    self.assertGreater(loss0, loss1)


if __name__ == '__main__':
  test.main()
