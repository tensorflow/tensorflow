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
"""Tests for slim.learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
from numpy import testing as np_testing
import tensorflow as tf

slim = tf.contrib.slim


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


def BatchNormClassifier(inputs):
  inputs = slim.batch_norm(inputs, decay=0.1)
  return slim.fully_connected(inputs, 1, activation_fn=tf.sigmoid)


class TrainBNClassifierTest(tf.test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testTrainWithNoInitAssignCanAchieveZeroLoss(self):
    logdir = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                          'tmp_logs')
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = BatchNormClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(
          total_loss, optimizer)

      loss = slim.learning.train(
          train_op, logdir, number_of_steps=300, log_every_n_steps=10)
      self.assertLess(loss, .1)


class CreateTrainOpTest(tf.test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)
    self._inputs = np.random.rand(16, 4).astype(np.float32)
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

  def testUseUpdateOps(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      expected_mean = np.mean(self._inputs, axis=(0))
      expected_var = np.var(self._inputs, axis=(0))

      tf_predictions = BatchNormClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)

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

      tf_predictions = BatchNormClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer,
                                               update_ops=[])

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

  def testUseGlobalStep(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = BatchNormClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)

      global_step = slim.get_or_create_global_step()

      with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        for _ in range(10):
          sess.run([train_op])
        global_step = global_step.eval()
        # After 10 updates global_step should be 10.
        self.assertAllClose(global_step, 10)

  def testNoneGlobalStep(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = BatchNormClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss,
                                               optimizer,
                                               global_step=None)

      global_step = slim.get_or_create_global_step()

      with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        for _ in range(10):
          sess.run([train_op])
        global_step = global_step.eval()
        # Since train_op don't use global_step it shouldn't change.
        self.assertAllClose(global_step, 0)

  def testRecordTrainOpInCollection(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = slim.learning.create_train_op(total_loss, optimizer)

      # Make sure the training op was recorded in the proper collection
      self.assertTrue(train_op in tf.get_collection(tf.GraphKeys.TRAIN_OP))


class TrainTest(tf.test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testTrainWithNonDefaultGraph(self):
    logdir = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                          'tmp_logs')
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)

    loss = slim.learning.train(
        train_op, logdir, number_of_steps=300, log_every_n_steps=10, graph=g)
    self.assertIsNotNone(loss)
    self.assertLess(loss, .015)

  def testTrainWithNoneAsLogdir(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)

      loss = slim.learning.train(
          train_op, None, number_of_steps=300, log_every_n_steps=10)
    self.assertIsNotNone(loss)
    self.assertLess(loss, .015)

  def testTrainWithSessionConfig(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)

      session_config = tf.ConfigProto(allow_soft_placement=True)
      loss = slim.learning.train(
          train_op,
          None,
          number_of_steps=300,
          log_every_n_steps=10,
          session_config=session_config)
    self.assertIsNotNone(loss)
    self.assertLess(loss, .015)

  def testTrainWithTrace(self):
    logdir = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                          'tmp_logs')
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()
      tf.summary.scalar('total_loss', total_loss)

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)

      loss = slim.learning.train(
          train_op,
          logdir,
          number_of_steps=300,
          log_every_n_steps=10,
          trace_every_n_steps=100)
    self.assertIsNotNone(loss)
    for trace_step in [1, 101, 201]:
      trace_filename = 'tf_trace-%d.json' % trace_step
      self.assertTrue(
          os.path.isfile(os.path.join(logdir, trace_filename)))

  def testTrainWithNoneAsLogdirWhenUsingSummariesRaisesError(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()
      tf.summary.scalar('total_loss', total_loss)

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)
      summary_op = tf.summary.merge_all()

      with self.assertRaises(ValueError):
        slim.learning.train(
            train_op, None, number_of_steps=300, summary_op=summary_op)

  def testTrainWithNoneAsLogdirWhenUsingTraceRaisesError(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)

      with self.assertRaises(ValueError):
        slim.learning.train(
            train_op, None, number_of_steps=300, trace_every_n_steps=10)

  def testTrainWithNoneAsLogdirWhenUsingSaverRaisesError(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)
      saver = tf.train.Saver()

      with self.assertRaises(ValueError):
        slim.learning.train(
            train_op, None, init_op=None, number_of_steps=300, saver=saver)

  def testTrainWithNoneAsInitWhenUsingVarsRaisesError(self):
    logdir = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                          'tmp_logs')
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(
          total_loss, optimizer)

      with self.assertRaises(RuntimeError):
        slim.learning.train(
            train_op, logdir, init_op=None, number_of_steps=300)

  def testTrainWithNoInitAssignCanAchieveZeroLoss(self):
    logdir = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                          'tmp_logs')
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)

      loss = slim.learning.train(
          train_op, logdir, number_of_steps=300, log_every_n_steps=10)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testTrainWithLocalVariable(self):
    logdir = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                          'tmp_logs')
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      local_multiplier = slim.local_variable(1.0)

      tf_predictions = LogisticClassifier(tf_inputs) * local_multiplier
      slim.losses.log_loss(tf_predictions, tf_labels)
      total_loss = slim.losses.get_total_loss()

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(
          total_loss, optimizer)

      loss = slim.learning.train(
          train_op, logdir, number_of_steps=300, log_every_n_steps=10)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testResumeTrainAchievesRoughlyTheSameLoss(self):
    logdir = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                          'tmp_logs')
    number_of_steps = [300, 301, 305]

    for i in range(len(number_of_steps)):
      with tf.Graph().as_default():
        tf.set_random_seed(i)
        tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
        tf_labels = tf.constant(self._labels, dtype=tf.float32)

        tf_predictions = LogisticClassifier(tf_inputs)
        slim.losses.log_loss(tf_predictions, tf_labels)
        total_loss = slim.losses.get_total_loss()

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        train_op = slim.learning.create_train_op(
            total_loss, optimizer)

        loss = slim.learning.train(
            train_op, logdir, number_of_steps=number_of_steps[i],
            log_every_n_steps=10)
        self.assertIsNotNone(loss)
        self.assertLess(loss, .015)

  def create_train_op(self, learning_rate=1.0, gradient_multiplier=1.0):
    tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
    tf_labels = tf.constant(self._labels, dtype=tf.float32)

    tf_predictions = LogisticClassifier(tf_inputs)
    slim.losses.log_loss(tf_predictions, tf_labels)
    total_loss = slim.losses.get_total_loss()

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)

    if gradient_multiplier != 1.0:
      variables = tf.trainable_variables()
      gradient_multipliers = {var: gradient_multiplier for var in variables}
    else:
      gradient_multipliers = None

    return slim.learning.create_train_op(
        total_loss, optimizer,
        gradient_multipliers=gradient_multipliers)

  def testTrainWithInitFromCheckpoint(self):
    logdir1 = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                           'tmp_logs1')
    logdir2 = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                           'tmp_logs2')

    # First, train the model one step (make sure the error is high).
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      train_op = self.create_train_op()
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=1)
      self.assertGreater(loss, .5)

    # Next, train the model to convergence.
    with tf.Graph().as_default():
      tf.set_random_seed(1)
      train_op = self.create_train_op()
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=300, log_every_n_steps=10)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

    # Finally, advance the model a single step and validate that the loss is
    # still low.
    with tf.Graph().as_default():
      tf.set_random_seed(2)
      train_op = self.create_train_op()

      model_variables = tf.global_variables()
      model_path = os.path.join(logdir1, 'model.ckpt-300')

      init_op = tf.global_variables_initializer()
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

      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

  def testTrainWithInitFromFn(self):
    logdir1 = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                           'tmp_logs1')
    logdir2 = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                           'tmp_logs2')

    # First, train the model one step (make sure the error is high).
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      train_op = self.create_train_op()
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=1)
      self.assertGreater(loss, .5)

    # Next, train the model to convergence.
    with tf.Graph().as_default():
      tf.set_random_seed(1)
      train_op = self.create_train_op()
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=300, log_every_n_steps=10)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

    # Finally, advance the model a single step and validate that the loss is
    # still low.
    with tf.Graph().as_default():
      tf.set_random_seed(2)
      train_op = self.create_train_op()

      model_variables = tf.global_variables()
      model_path = os.path.join(logdir1, 'model.ckpt-300')
      saver = tf.train.Saver(model_variables)
      def RestoreFn(sess):
        saver.restore(sess, model_path)
      loss = slim.learning.train(
          train_op,
          logdir2,
          number_of_steps=1,
          init_fn=RestoreFn)

      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def ModelLoss(self):
    tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
    tf_labels = tf.constant(self._labels, dtype=tf.float32)

    tf_predictions = LogisticClassifier(tf_inputs)
    slim.losses.log_loss(tf_predictions, tf_labels)
    return slim.losses.get_total_loss()

  def testTrainAllVarsHasLowerLossThanTrainSubsetOfVars(self):
    logdir1 = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                           'tmp_logs1')

    # First, train only the weights of the model.
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      weights = slim.get_variables_by_name('weights')

      train_op = slim.learning.create_train_op(
          total_loss,
          optimizer,
          variables_to_train=weights)

      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=200, log_every_n_steps=10)
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Next, train the biases of the model.
    with tf.Graph().as_default():
      tf.set_random_seed(1)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      biases = slim.get_variables_by_name('biases')

      train_op = slim.learning.create_train_op(
          total_loss,
          optimizer,
          variables_to_train=biases)

      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=300, log_every_n_steps=10)
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Finally, train both weights and bias to get lower loss.
    with tf.Graph().as_default():
      tf.set_random_seed(2)
      total_loss = self.ModelLoss()
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = slim.learning.create_train_op(total_loss, optimizer)
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=400, log_every_n_steps=10)

      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testTrainingSubsetsOfVariablesOnlyUpdatesThoseVariables(self):
    # First, train only the weights of the model.
    with tf.Graph().as_default():
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
    logdir1 = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                           'tmp_logs1')
    logdir2 = os.path.join(tempfile.mkdtemp(prefix=self.get_temp_dir()),
                           'tmp_logs2')

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
      loss = slim.learning.train(
          train_op, logdir1, number_of_steps=number_of_steps)
      losses.append(loss)
      self.assertGreater(loss, .5)

    # Second, train the model with equivalently larger learning rate.
    with tf.Graph().as_default():
      tf.set_random_seed(0)
      train_op = self.create_train_op(
          learning_rate=learning_rate,
          gradient_multiplier=multipliers[1])
      loss = slim.learning.train(
          train_op, logdir2, number_of_steps=number_of_steps)
      losses.append(loss)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .5)

    # The loss of the model trained with larger learning rate should
    # be smaller.
    self.assertGreater(losses[0], losses[1])


if __name__ == '__main__':
  tf.test.main()
