# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for slim.learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import numpy as np
from numpy import testing as np_testing
import tensorflow as tf

import tensorflow.contrib.slim as slim


class ClipGradientNormsTest(tf.test.TestCase):

  def clip_values(self, arr):
    norm = np.sqrt(np.sum(arr**2))
    if norm > self._max_norm:
      return self._max_norm * arr / np.sqrt(np.sum(arr**2))
    return arr

  def setUp(self):
    np.random.seed(0)

    self._max_norm = 1.0
    self._grad_vec = np.array([1., 2., 3.])
    self._clipped_grad_vec = self.clip_values(self._grad_vec)
    self._zero_vec = np.zeros(self._grad_vec.size)

  def testOrdinaryGradIsClippedCorrectly(self):
    gradient = tf.constant(self._grad_vec, dtype=tf.float32)
    variable = tf.Variable(self._zero_vec, dtype=tf.float32)
    gradients_to_variables = (gradient, variable)
    [gradients_to_variables] = slim.learning.clip_gradient_norms(
        [gradients_to_variables], self._max_norm)

    # Ensure the variable passed through.
    self.assertEqual(gradients_to_variables[1], variable)

    with self.test_session() as sess:
      actual_gradient = sess.run(gradients_to_variables[0])
    np_testing.assert_almost_equal(actual_gradient, self._clipped_grad_vec)

  def testNoneGradPassesThroughCorrectly(self):
    gradient = None
    variable = tf.Variable(self._zero_vec, dtype=tf.float32)

    gradients_to_variables = (gradient, variable)
    [gradients_to_variables] = slim.learning.clip_gradient_norms(
        [gradients_to_variables], self._max_norm)

    self.assertEqual(gradients_to_variables[0], None)
    self.assertEqual(gradients_to_variables[1], variable)

  def testIndexedSlicesGradIsClippedCorrectly(self):
    sparse_grad_indices = np.array([0, 1, 4])
    sparse_grad_dense_shape = [self._grad_vec.size]

    values = tf.constant(self._grad_vec, dtype=tf.float32)
    indices = tf.constant(sparse_grad_indices, dtype=tf.int32)
    dense_shape = tf.constant(sparse_grad_dense_shape, dtype=tf.int32)

    gradient = tf.IndexedSlices(values, indices, dense_shape)
    variable = tf.Variable(self._zero_vec, dtype=tf.float32)

    gradients_to_variables = (gradient, variable)
    gradients_to_variables = slim.learning.clip_gradient_norms(
        [gradients_to_variables], self._max_norm)[0]

    # Ensure the built IndexedSlice has the right form.
    self.assertEqual(gradients_to_variables[1], variable)
    self.assertEqual(gradients_to_variables[0].indices, indices)
    self.assertEqual(gradients_to_variables[0].dense_shape, dense_shape)

    with tf.Session() as sess:
      actual_gradient = sess.run(gradients_to_variables[0].values)
    np_testing.assert_almost_equal(actual_gradient, self._clipped_grad_vec)


class MultiplyGradientsTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(0)
    self._multiplier = 3.7
    self._grad_vec = np.array([1., 2., 3.])
    self._multiplied_grad_vec = np.multiply(self._grad_vec, self._multiplier)

  def testNonListGradsRaisesError(self):
    gradient = tf.constant(self._grad_vec, dtype=tf.float32)
    variable = tf.Variable(tf.zeros_like(gradient))
    grad_to_var = (gradient, variable)
    gradient_multipliers = {variable: self._multiplier}
    with self.assertRaises(ValueError):
      slim.learning.multiply_gradients(grad_to_var, gradient_multipliers)

  def testEmptyMultiplesRaisesError(self):
    gradient = tf.constant(self._grad_vec, dtype=tf.float32)
    variable = tf.Variable(tf.zeros_like(gradient))
    grad_to_var = (gradient, variable)
    with self.assertRaises(ValueError):
      slim.learning.multiply_gradients([grad_to_var], {})

  def testNonDictMultiplierRaisesError(self):
    gradient = tf.constant(self._grad_vec, dtype=tf.float32)
    variable = tf.Variable(tf.zeros_like(gradient))
    grad_to_var = (gradient, variable)
    with self.assertRaises(ValueError):
      slim.learning.multiply_gradients([grad_to_var], 3)

  def testMultipleOfNoneGradRaisesError(self):
    gradient = tf.constant(self._grad_vec, dtype=tf.float32)
    variable = tf.Variable(tf.zeros_like(gradient))
    grad_to_var = (None, variable)
    gradient_multipliers = {variable: self._multiplier}
    with self.assertRaises(ValueError):
      slim.learning.multiply_gradients(grad_to_var, gradient_multipliers)

  def testMultipleGradientsWithVariables(self):
    gradient = tf.constant(self._grad_vec, dtype=tf.float32)
    variable = tf.Variable(tf.zeros_like(gradient))
    grad_to_var = (gradient, variable)
    gradient_multipliers = {variable: self._multiplier}

    [grad_to_var] = slim.learning.multiply_gradients(
        [grad_to_var],
        gradient_multipliers)

    # Ensure the variable passed through.
    self.assertEqual(grad_to_var[1], variable)

    with self.test_session() as sess:
      actual_gradient = sess.run(grad_to_var[0])
    np_testing.assert_almost_equal(actual_gradient,
                                   self._multiplied_grad_vec, 5)

  def testIndexedSlicesGradIsMultiplied(self):
    values = tf.constant(self._grad_vec, dtype=tf.float32)
    indices = tf.constant([0, 1, 2], dtype=tf.int32)
    dense_shape = tf.constant([self._grad_vec.size], dtype=tf.int32)

    gradient = tf.IndexedSlices(values, indices, dense_shape)
    variable = tf.Variable(tf.zeros((1, 3)))
    grad_to_var = (gradient, variable)
    gradient_multipliers = {variable: self._multiplier}

    [grad_to_var] = slim.learning.multiply_gradients(
        [grad_to_var],
        gradient_multipliers)

    # Ensure the built IndexedSlice has the right form.
    self.assertEqual(grad_to_var[1], variable)
    self.assertEqual(grad_to_var[0].indices, indices)
    self.assertEqual(grad_to_var[0].dense_shape, dense_shape)

    with self.test_session() as sess:
      actual_gradient = sess.run(grad_to_var[0].values)
    np_testing.assert_almost_equal(actual_gradient,
                                   self._multiplied_grad_vec, 5)


def LogisticClassifier(inputs):
  return slim.fully_connected(
      inputs, 1, activation_fn=tf.sigmoid)


class TrainTest(tf.test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)
    self._logdir = os.path.join(self.get_temp_dir(), 'tmp_logs/')

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testTrainWithNoInitAssignCanAchieveZeroLoss(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      log_loss = slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss([log_loss])

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(
          total_loss, optimizer)

      loss = slim.learning.train(
          train_op, self._logdir, number_of_steps=300)
      self.assertLess(loss, .015)

  def testResumeTrainAchievesRoughlyTheSameLoss(self):
    number_of_steps = [300, 301, 305]

    for i in range(len(number_of_steps)):
      g = tf.Graph()
      with g.as_default():
        tf.set_random_seed(i)
        tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
        tf_labels = tf.constant(self._labels, dtype=tf.float32)

        tf_predictions = LogisticClassifier(tf_inputs)
        log_loss = slim.losses.log_loss(tf_predictions, tf_labels)
        total_loss = slim.losses.get_total_loss([log_loss])

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        train_op = slim.learning.create_train_op(
            total_loss, optimizer)

        loss = slim.learning.train(
            train_op, self._logdir, number_of_steps=number_of_steps[i])
        self.assertLess(loss, .015)

  def create_train_op(self):
    tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
    tf_labels = tf.constant(self._labels, dtype=tf.float32)

    tf_predictions = LogisticClassifier(tf_inputs)
    log_loss = slim.losses.log_loss(tf_predictions, tf_labels)
    total_loss = slim.losses.get_total_loss([log_loss])

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

    return slim.learning.create_train_op(total_loss, optimizer)

  def testTrainWithInitFromCheckpoint(self):
    logdir1 = os.path.join(self.get_temp_dir(), 'tmp_logs1/')
    logdir2 = os.path.join(self.get_temp_dir(), 'tmp_logs2/')
    if tf.gfile.Exists(logdir1):  # For running on jenkins.
      tf.gfile.DeleteRecursively(logdir1)
    if tf.gfile.Exists(logdir2):  # For running on jenkins.
      tf.gfile.DeleteRecursively(logdir2)

    # First, train the model one step (make sure the error is high).
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      train_op = self.create_train_op()
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=1)
      self.assertGreater(loss, .5)

    # Next, train the model to convergence.
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(1)
      train_op = self.create_train_op()
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=300)
      self.assertLess(loss, .02)

    # Finally, advance the model a single step and validate that the loss is
    # still low.
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(2)
      train_op = self.create_train_op()

      model_variables = tf.all_variables()
      model_path = os.path.join(logdir1, 'model.ckpt-300')

      init_op = tf.initialize_all_variables()
      op, init_feed_dict = slim.assign_from_checkpoint(
          model_path, model_variables)

      def InitAssignFn(sess):
        sess.run(op, init_feed_dict)

      loss = slim.learning.train(
          train_op,
          logdir2,
          number_of_steps=1,
          init_op=init_op,
          init_fn=InitAssignFn)

      self.assertLess(loss, .02)

  def testTrainWithInitFromFn(self):
    logdir1 = os.path.join(self.get_temp_dir(), 'tmp_logs4/')
    logdir2 = os.path.join(self.get_temp_dir(), 'tmp_logs5/')
    if tf.gfile.Exists(logdir1):  # For running on jenkins.
      tf.gfile.DeleteRecursively(logdir1)
    if tf.gfile.Exists(logdir2):  # For running on jenkins.
      tf.gfile.DeleteRecursively(logdir2)

    # First, train the model one step (make sure the error is high).
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      train_op = self.create_train_op()
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=1)
      self.assertGreater(loss, .5)

    # Next, train the model to convergence.
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(1)
      train_op = self.create_train_op()
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=300)
      self.assertLess(loss, .015)

    # Finally, advance the model a single step and validate that the loss is
    # still low.
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(2)
      train_op = self.create_train_op()

      model_variables = tf.all_variables()
      model_path = os.path.join(logdir1, 'model.ckpt-300')
      saver = tf.train.Saver(model_variables)
      def RestoreFn(sess):
        saver.restore(sess, model_path)
      loss = slim.learning.train(
          train_op,
          logdir2,
          number_of_steps=1,
          init_fn=RestoreFn)

      self.assertLess(loss, .015)

  def ModelLoss(self):
    tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
    tf_labels = tf.constant(self._labels, dtype=tf.float32)

    tf_predictions = LogisticClassifier(tf_inputs)
    log_loss = slim.losses.log_loss(tf_predictions, tf_labels)
    return slim.losses.get_total_loss([log_loss])

  def  testTrainAllVarsHasLowerLossThanTrainSubsetOfVars(self):
    logdir1 = os.path.join(self.get_temp_dir(), 'tmp_logs3/')
    if tf.gfile.Exists(logdir1):  # For running on jenkins.
      tf.gfile.DeleteRecursively(logdir1)

    # First, train only the weights of the model.
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      weights = slim.get_variables_by_name('weights')

      train_op = slim.learning.create_train_op(
          total_loss,
          optimizer,
          variables_to_train=weights)

      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=200)
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Next, train the biases of the model.
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(1)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      biases = slim.get_variables_by_name('biases')

      train_op = slim.learning.create_train_op(
          total_loss,
          optimizer,
          variables_to_train=biases)

      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=300)
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Finally, train both weights and bias to get lower loss.
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(2)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=400)

      self.assertLess(loss, .015)

  def testTrainingSubsetsOfVariablesOnlyUpdatesThoseVariables(self):
    # First, train only the weights of the model.
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      weights, biases = slim.get_variables()

      train_op = slim.learning.create_train_op(total_loss, optimizer)
      train_weights = slim.learning.create_train_op(
          total_loss, optimizer, variables_to_train=[weights])
      train_biases = slim.learning.create_train_op(
          total_loss, optimizer, variables_to_train=[biases])

      with tf.Session() as sess:
        # Initialize the variables.
        sess.run(tf.initialize_all_variables())

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


if __name__ == '__main__':
  tf.test.main()
