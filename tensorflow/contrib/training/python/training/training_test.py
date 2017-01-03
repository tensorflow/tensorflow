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
import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.contrib.training.python.training import training
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_lib2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib


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

  def testUseUpdateOps(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      expected_mean = np.mean(self._inputs, axis=(0))
      expected_var = np.var(self._inputs, axis=(0))

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(total_loss, optimizer)

      moving_mean = variables_lib.get_variables_by_name('moving_mean')[0]
      moving_variance = variables_lib.get_variables_by_name('moving_variance')[
          0]

      with session_lib.Session() as sess:
        # Initialize all variables
        sess.run(variables_lib2.global_variables_initializer())
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
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(total_loss, optimizer, update_ops=[])

      moving_mean = variables_lib.get_variables_by_name('moving_mean')[0]
      moving_variance = variables_lib.get_variables_by_name('moving_variance')[
          0]

      with session_lib.Session() as sess:
        # Initialize all variables
        sess.run(variables_lib2.global_variables_initializer())
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

  def testUseGlobalStep(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(total_loss, optimizer)

      global_step = variables_lib.get_or_create_global_step()

      with session_lib.Session() as sess:
        # Initialize all variables
        sess.run(variables_lib2.global_variables_initializer())

        for _ in range(10):
          sess.run([train_op])
        global_step = global_step.eval()
        # After 10 updates global_step should be 10.
        self.assertAllClose(global_step, 10)

  def testNoneGlobalStep(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(
          total_loss, optimizer, global_step=None)

      global_step = variables_lib.get_or_create_global_step()

      with session_lib.Session() as sess:
        # Initialize all variables
        sess.run(variables_lib2.global_variables_initializer())

        for _ in range(10):
          sess.run([train_op])
        global_step = global_step.eval()
        # Since train_op don't use global_step it shouldn't change.
        self.assertAllClose(global_step, 0)


class TrainBNClassifierTest(test.TestCase):

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
    g = ops.Graph()
    with g.as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(total_loss, optimizer)

      loss = training.train(
          train_op,
          self._logdir,
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps=300)])
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
    logdir = os.path.join(self.get_temp_dir(), 'can_achieve_zero_loss')

    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = logistic_classifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(total_loss, optimizer)

      loss = training.train(
          train_op,
          logdir,
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps=300)])
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testTrainWithLocalVariable(self):
    logdir = os.path.join(self.get_temp_dir(), 'train_with_local_variable')

    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      local_multiplier = variables_lib.local_variable(1.0)

      tf_predictions = logistic_classifier(tf_inputs) * local_multiplier
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = training.create_train_op(total_loss, optimizer)

      loss = training.train(
          train_op,
          logdir,
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps=300)])
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
        loss_ops.log_loss(tf_predictions, tf_labels)
        total_loss = loss_ops.get_total_loss()

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
            ])
        self.assertIsNotNone(loss)
        self.assertLess(loss, .015)

  def create_train_op(self, learning_rate=1.0, gradient_multiplier=1.0):
    tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

    tf_predictions = logistic_classifier(tf_inputs)
    loss_ops.log_loss(tf_predictions, tf_labels)
    total_loss = loss_ops.get_total_loss()

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
          save_checkpoint_secs=None)
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
                  logdir1, save_steps=1, saver=saver),
              basic_session_run_hooks.StopAtStepHook(num_steps=300),
          ],
          save_checkpoint_secs=None)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

    # Finally, advance the model a single step and validate that the loss is
    # still low.
    with ops.Graph().as_default():
      random_seed.set_random_seed(2)
      train_op = self.create_train_op()

      model_variables = variables_lib2.global_variables()
      model_path = os.path.join(logdir1, 'model.ckpt-300')

      assign_fn = variables_lib.assign_from_checkpoint_fn(model_path,
                                                          model_variables)

      def init_fn(_, session):
        assign_fn(session)

      loss = training.train(
          train_op,
          logdir2,
          scaffold=monitored_session.Scaffold(init_fn=init_fn),
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps=1)])

      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

  def ModelLoss(self):
    tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

    tf_predictions = logistic_classifier(tf_inputs)
    loss_ops.log_loss(tf_predictions, tf_labels)
    return loss_ops.get_total_loss()

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
                  logdir, save_steps=1, saver=saver),
              basic_session_run_hooks.StopAtStepHook(num_steps=200),
          ])
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
                  logdir, save_steps=1, saver=saver),
              basic_session_run_hooks.StopAtStepHook(num_steps=300),
          ])
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
              basic_session_run_hooks.CheckpointSaverHook(
                  logdir, save_steps=1, saver=saver),
              basic_session_run_hooks.StopAtStepHook(num_steps=400),
          ])
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

      with session_lib.Session() as sess:
        # Initialize the variables.
        sess.run(variables_lib2.global_variables_initializer())

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

    if gfile.Exists(logdir1):
      gfile.DeleteRecursively(logdir1)
    if gfile.Exists(logdir2):
      gfile.DeleteRecursively(logdir2)

    multipliers = [1., 1000.]
    number_of_steps = 10
    losses = []
    learning_rate = 0.001

    # First, train the model with equivalently smaller learning rate.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      train_op = self.create_train_op(
          learning_rate=learning_rate, gradient_multiplier=multipliers[0])

      saver = saver_lib.Saver()

      loss = training.train(
          train_op,
          logdir1,
          hooks=[
              basic_session_run_hooks.StopAtStepHook(num_steps=number_of_steps),
              basic_session_run_hooks.CheckpointSaverHook(
                  logdir1, save_steps=50, saver=saver),
          ])

      losses.append(loss)
      self.assertGreater(loss, .5)

    # Second, train the model with equivalently larger learning rate.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      train_op = self.create_train_op(
          learning_rate=learning_rate, gradient_multiplier=multipliers[1])
      saver = saver_lib.Saver()

      loss = training.train(
          train_op,
          logdir2,
          hooks=[
              basic_session_run_hooks.StopAtStepHook(num_steps=number_of_steps),
              basic_session_run_hooks.CheckpointSaverHook(
                  logdir2, save_steps=50, saver=saver),
          ])

      losses.append(loss)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .5)

    # The loss of the model trained with larger learning rate should
    # be smaller.
    self.assertGreater(losses[0], losses[1])


if __name__ == '__main__':
  test.main()
