# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for slim.learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tempfile

import numpy as np
from numpy import testing as np_testing

from tensorflow.contrib.framework.python.ops import variables as variables_lib2
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.contrib.slim.python.slim import learning
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import dumping_wrapper as dumping_wrapper_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import saver as saver_lib


class ClipGradientNormsTest(test.TestCase):

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
    gradient = constant_op.constant(self._grad_vec, dtype=dtypes.float32)
    variable = variables_lib.Variable(self._zero_vec, dtype=dtypes.float32)
    gradients_to_variables = (gradient, variable)
    [gradients_to_variables] = learning.clip_gradient_norms(
        [gradients_to_variables], self._max_norm)

    # Ensure the variable passed through.
    self.assertEqual(gradients_to_variables[1], variable)

    with self.test_session() as sess:
      actual_gradient = sess.run(gradients_to_variables[0])
    np_testing.assert_almost_equal(actual_gradient, self._clipped_grad_vec)

  def testNoneGradPassesThroughCorrectly(self):
    gradient = None
    variable = variables_lib.Variable(self._zero_vec, dtype=dtypes.float32)

    gradients_to_variables = (gradient, variable)
    [gradients_to_variables] = learning.clip_gradient_norms(
        [gradients_to_variables], self._max_norm)

    self.assertEqual(gradients_to_variables[0], None)
    self.assertEqual(gradients_to_variables[1], variable)

  def testIndexedSlicesGradIsClippedCorrectly(self):
    sparse_grad_indices = np.array([0, 1, 4])
    sparse_grad_dense_shape = [self._grad_vec.size]

    values = constant_op.constant(self._grad_vec, dtype=dtypes.float32)
    indices = constant_op.constant(sparse_grad_indices, dtype=dtypes.int32)
    dense_shape = constant_op.constant(
        sparse_grad_dense_shape, dtype=dtypes.int32)

    gradient = ops.IndexedSlices(values, indices, dense_shape)
    variable = variables_lib.Variable(self._zero_vec, dtype=dtypes.float32)

    gradients_to_variables = (gradient, variable)
    gradients_to_variables = learning.clip_gradient_norms(
        [gradients_to_variables], self._max_norm)[0]

    # Ensure the built IndexedSlice has the right form.
    self.assertEqual(gradients_to_variables[1], variable)
    self.assertEqual(gradients_to_variables[0].indices, indices)
    self.assertEqual(gradients_to_variables[0].dense_shape, dense_shape)

    with session.Session() as sess:
      actual_gradient = sess.run(gradients_to_variables[0].values)
    np_testing.assert_almost_equal(actual_gradient, self._clipped_grad_vec)


class MultiplyGradientsTest(test.TestCase):

  def setUp(self):
    np.random.seed(0)
    self._multiplier = 3.7
    self._grad_vec = np.array([1., 2., 3.])
    self._multiplied_grad_vec = np.multiply(self._grad_vec, self._multiplier)

  def testNonListGradsRaisesError(self):
    gradient = constant_op.constant(self._grad_vec, dtype=dtypes.float32)
    variable = variables_lib.Variable(array_ops.zeros_like(gradient))
    grad_to_var = (gradient, variable)
    gradient_multipliers = {variable: self._multiplier}
    with self.assertRaises(ValueError):
      learning.multiply_gradients(grad_to_var, gradient_multipliers)

  def testEmptyMultiplesRaisesError(self):
    gradient = constant_op.constant(self._grad_vec, dtype=dtypes.float32)
    variable = variables_lib.Variable(array_ops.zeros_like(gradient))
    grad_to_var = (gradient, variable)
    with self.assertRaises(ValueError):
      learning.multiply_gradients([grad_to_var], {})

  def testNonDictMultiplierRaisesError(self):
    gradient = constant_op.constant(self._grad_vec, dtype=dtypes.float32)
    variable = variables_lib.Variable(array_ops.zeros_like(gradient))
    grad_to_var = (gradient, variable)
    with self.assertRaises(ValueError):
      learning.multiply_gradients([grad_to_var], 3)

  def testMultipleOfNoneGradRaisesError(self):
    gradient = constant_op.constant(self._grad_vec, dtype=dtypes.float32)
    variable = variables_lib.Variable(array_ops.zeros_like(gradient))
    grad_to_var = (None, variable)
    gradient_multipliers = {variable: self._multiplier}
    with self.assertRaises(ValueError):
      learning.multiply_gradients(grad_to_var, gradient_multipliers)

  def testMultipleGradientsWithVariables(self):
    gradient = constant_op.constant(self._grad_vec, dtype=dtypes.float32)
    variable = variables_lib.Variable(array_ops.zeros_like(gradient))
    grad_to_var = (gradient, variable)
    gradient_multipliers = {variable: self._multiplier}

    [grad_to_var] = learning.multiply_gradients([grad_to_var],
                                                gradient_multipliers)

    # Ensure the variable passed through.
    self.assertEqual(grad_to_var[1], variable)

    with self.test_session() as sess:
      actual_gradient = sess.run(grad_to_var[0])
    np_testing.assert_almost_equal(actual_gradient, self._multiplied_grad_vec,
                                   5)

  def testIndexedSlicesGradIsMultiplied(self):
    values = constant_op.constant(self._grad_vec, dtype=dtypes.float32)
    indices = constant_op.constant([0, 1, 2], dtype=dtypes.int32)
    dense_shape = constant_op.constant(
        [self._grad_vec.size], dtype=dtypes.int32)

    gradient = ops.IndexedSlices(values, indices, dense_shape)
    variable = variables_lib.Variable(array_ops.zeros((1, 3)))
    grad_to_var = (gradient, variable)
    gradient_multipliers = {variable: self._multiplier}

    [grad_to_var] = learning.multiply_gradients([grad_to_var],
                                                gradient_multipliers)

    # Ensure the built IndexedSlice has the right form.
    self.assertEqual(grad_to_var[1], variable)
    self.assertEqual(grad_to_var[0].indices, indices)
    self.assertEqual(grad_to_var[0].dense_shape, dense_shape)

    with self.test_session() as sess:
      actual_gradient = sess.run(grad_to_var[0].values)
    np_testing.assert_almost_equal(actual_gradient, self._multiplied_grad_vec,
                                   5)

  def testTensorMultiplierOfGradient(self):
    gradient = constant_op.constant(self._grad_vec, dtype=dtypes.float32)
    variable = variables_lib.Variable(array_ops.zeros_like(gradient))
    multiplier_flag = variables_lib.Variable(True)
    tensor_multiplier = array_ops.where(multiplier_flag, self._multiplier, 1.0)
    grad_to_var = (gradient, variable)
    gradient_multipliers = {variable: tensor_multiplier}

    [grad_to_var] = learning.multiply_gradients([grad_to_var],
                                                gradient_multipliers)

    with self.test_session() as sess:
      sess.run(variables_lib.global_variables_initializer())
      gradient_true_flag = sess.run(grad_to_var[0])
      sess.run(multiplier_flag.assign(False))
      gradient_false_flag = sess.run(grad_to_var[0])
    np_testing.assert_almost_equal(gradient_true_flag,
                                   self._multiplied_grad_vec, 5)
    np_testing.assert_almost_equal(gradient_false_flag, self._grad_vec, 5)


def LogisticClassifier(inputs):
  return layers.fully_connected(inputs, 1, activation_fn=math_ops.sigmoid)


def BatchNormClassifier(inputs):
  inputs = layers.batch_norm(inputs, decay=0.1, fused=True)
  return layers.fully_connected(inputs, 1, activation_fn=math_ops.sigmoid)


class TrainBNClassifierTest(test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testTrainWithNoInitAssignCanAchieveZeroLoss(self):
    logdir = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs')
    g = ops.Graph()
    with g.as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = BatchNormClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      loss = learning.train(
          train_op, logdir, number_of_steps=300, log_every_n_steps=10)
      self.assertLess(loss, .1)


class CreateTrainOpTest(test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)
    self._inputs = np.random.rand(16, 4).astype(np.float32)
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

  def _addBesselsCorrection(self, sample_size, expected_var):
    correction_factor = sample_size / (sample_size - 1)
    expected_var *= correction_factor
    return expected_var

  def testUseUpdateOps(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      expected_mean = np.mean(self._inputs, axis=(0))
      expected_var = np.var(self._inputs, axis=(0))
      expected_var = self._addBesselsCorrection(16, expected_var)

      tf_predictions = BatchNormClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      moving_mean = variables_lib2.get_variables_by_name('moving_mean')[0]
      moving_variance = variables_lib2.get_variables_by_name('moving_variance')[
          0]

      with session.Session() as sess:
        # Initialize all variables
        sess.run(variables_lib.global_variables_initializer())
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

      tf_predictions = BatchNormClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer, update_ops=[])

      moving_mean = variables_lib2.get_variables_by_name('moving_mean')[0]
      moving_variance = variables_lib2.get_variables_by_name('moving_variance')[
          0]

      with session.Session() as sess:
        # Initialize all variables
        sess.run(variables_lib.global_variables_initializer())
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

      tf_predictions = BatchNormClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      global_step = variables_lib2.get_or_create_global_step()

      with session.Session() as sess:
        # Initialize all variables
        sess.run(variables_lib.global_variables_initializer())

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

      tf_predictions = BatchNormClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(
          total_loss, optimizer, global_step=None)

      global_step = variables_lib2.get_or_create_global_step()

      with session.Session() as sess:
        # Initialize all variables
        sess.run(variables_lib.global_variables_initializer())

        for _ in range(10):
          sess.run([train_op])
        global_step = global_step.eval()
        # Since train_op don't use global_step it shouldn't change.
        self.assertAllClose(global_step, 0)

  def testRecordTrainOpInCollection(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = learning.create_train_op(total_loss, optimizer)

      # Make sure the training op was recorded in the proper collection
      self.assertTrue(train_op in ops.get_collection(ops.GraphKeys.TRAIN_OP))


class TrainTest(test.TestCase):

  def setUp(self):
    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def testTrainWithNonDefaultGraph(self):
    logdir = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs')
    g = ops.Graph()
    with g.as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

    loss = learning.train(
        train_op, logdir, number_of_steps=300, log_every_n_steps=10, graph=g)
    self.assertIsNotNone(loss)
    self.assertLess(loss, .015)

  def testTrainWithNoneAsLogdir(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      loss = learning.train(
          train_op, None, number_of_steps=300, log_every_n_steps=10)
    self.assertIsNotNone(loss)
    self.assertLess(loss, .015)

  def testTrainWithSessionConfig(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      session_config = config_pb2.ConfigProto(allow_soft_placement=True)
      loss = learning.train(
          train_op,
          None,
          number_of_steps=300,
          log_every_n_steps=10,
          session_config=session_config)
    self.assertIsNotNone(loss)
    self.assertLess(loss, .015)

  def testTrainWithSessionWrapper(self):
    """Test that slim.learning.train can take `session_wrapper` args.

    One of the applications of `session_wrapper` is the wrappers of TensorFlow
    Debugger (tfdbg), which intercept methods calls to `tf.Session` (e.g., run)
    to achieve debugging. `DumpingDebugWrapperSession` is used here for testing
    purpose.
    """
    dump_root = tempfile.mkdtemp()

    def dumping_wrapper(sess):  # pylint: disable=invalid-name
      return dumping_wrapper_lib.DumpingDebugWrapperSession(sess, dump_root)

    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      loss = learning.train(
          train_op, None, number_of_steps=1, session_wrapper=dumping_wrapper)
    self.assertIsNotNone(loss)

    run_root = glob.glob(os.path.join(dump_root, 'run_*'))[-1]
    dump = debug_data.DebugDumpDir(run_root)

  def testTrainWithTrace(self):
    logdir = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs')
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      summary.scalar('total_loss', total_loss)

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      loss = learning.train(
          train_op,
          logdir,
          number_of_steps=300,
          log_every_n_steps=10,
          trace_every_n_steps=100)
    self.assertIsNotNone(loss)
    for trace_step in [0, 100, 200]:
      trace_filename = 'tf_trace-%d.json' % trace_step
      self.assertTrue(os.path.isfile(os.path.join(logdir, trace_filename)))

  def testTrainWithNoneAsLogdirWhenUsingSummariesRaisesError(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()
      summary.scalar('total_loss', total_loss)

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)
      summary_op = summary.merge_all()

      with self.assertRaises(ValueError):
        learning.train(
            train_op, None, number_of_steps=300, summary_op=summary_op)

  def testTrainWithNoneAsLogdirWhenUsingTraceRaisesError(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      with self.assertRaises(ValueError):
        learning.train(
            train_op, None, number_of_steps=300, trace_every_n_steps=10)

  def testTrainWithNoneAsLogdirWhenUsingSaverRaisesError(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)
      saver = saver_lib.Saver()

      with self.assertRaises(ValueError):
        learning.train(
            train_op, None, init_op=None, number_of_steps=300, saver=saver)

  def testTrainWithNoneAsInitWhenUsingVarsRaisesError(self):
    logdir = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs')
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      with self.assertRaises(RuntimeError):
        learning.train(train_op, logdir, init_op=None, number_of_steps=300)

  def testTrainWithNoInitAssignCanAchieveZeroLoss(self):
    logdir = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs')
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = LogisticClassifier(tf_inputs)
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      loss = learning.train(
          train_op, logdir, number_of_steps=300, log_every_n_steps=10)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testTrainWithLocalVariable(self):
    logdir = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs')
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      local_multiplier = variables_lib2.local_variable(1.0)

      tf_predictions = LogisticClassifier(tf_inputs) * local_multiplier
      loss_ops.log_loss(tf_predictions, tf_labels)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      loss = learning.train(
          train_op, logdir, number_of_steps=300, log_every_n_steps=10)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testResumeTrainAchievesRoughlyTheSameLoss(self):
    logdir = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs')
    number_of_steps = [300, 301, 305]

    for i in range(len(number_of_steps)):
      with ops.Graph().as_default():
        random_seed.set_random_seed(i)
        tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
        tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

        tf_predictions = LogisticClassifier(tf_inputs)
        loss_ops.log_loss(tf_predictions, tf_labels)
        total_loss = loss_ops.get_total_loss()

        optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

        train_op = learning.create_train_op(total_loss, optimizer)

        loss = learning.train(
            train_op,
            logdir,
            number_of_steps=number_of_steps[i],
            log_every_n_steps=10)
        self.assertIsNotNone(loss)
        self.assertLess(loss, .015)

  def create_train_op(self, learning_rate=1.0, gradient_multiplier=1.0):
    tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

    tf_predictions = LogisticClassifier(tf_inputs)
    loss_ops.log_loss(tf_predictions, tf_labels)
    total_loss = loss_ops.get_total_loss()

    optimizer = gradient_descent.GradientDescentOptimizer(
        learning_rate=learning_rate)

    if gradient_multiplier != 1.0:
      variables = variables_lib.trainable_variables()
      gradient_multipliers = {var: gradient_multiplier for var in variables}
    else:
      gradient_multipliers = None

    return learning.create_train_op(
        total_loss, optimizer, gradient_multipliers=gradient_multipliers)

  def testTrainWithInitFromCheckpoint(self):
    logdir1 = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs1')
    logdir2 = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs2')

    # First, train the model one step (make sure the error is high).
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      train_op = self.create_train_op()
      loss = learning.train(train_op, logdir1, number_of_steps=1)
      self.assertGreater(loss, .5)

    # Next, train the model to convergence.
    with ops.Graph().as_default():
      random_seed.set_random_seed(1)
      train_op = self.create_train_op()
      loss = learning.train(
          train_op, logdir1, number_of_steps=300, log_every_n_steps=10)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

    # Finally, advance the model a single step and validate that the loss is
    # still low.
    with ops.Graph().as_default():
      random_seed.set_random_seed(2)
      train_op = self.create_train_op()

      model_variables = variables_lib.global_variables()
      model_path = os.path.join(logdir1, 'model.ckpt-300')

      init_op = variables_lib.global_variables_initializer()
      op, init_feed_dict = variables_lib2.assign_from_checkpoint(
          model_path, model_variables)

      def InitAssignFn(sess):
        sess.run(op, init_feed_dict)

      loss = learning.train(
          train_op,
          logdir2,
          number_of_steps=1,
          init_op=init_op,
          init_fn=InitAssignFn)

      self.assertIsNotNone(loss)
      self.assertLess(loss, .02)

  def testTrainWithInitFromFn(self):
    logdir1 = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs1')
    logdir2 = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs2')

    # First, train the model one step (make sure the error is high).
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      train_op = self.create_train_op()
      loss = learning.train(train_op, logdir1, number_of_steps=1)
      self.assertGreater(loss, .5)

    # Next, train the model to convergence.
    with ops.Graph().as_default():
      random_seed.set_random_seed(1)
      train_op = self.create_train_op()
      loss = learning.train(
          train_op, logdir1, number_of_steps=300, log_every_n_steps=10)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

    # Finally, advance the model a single step and validate that the loss is
    # still low.
    with ops.Graph().as_default():
      random_seed.set_random_seed(2)
      train_op = self.create_train_op()

      model_variables = variables_lib.global_variables()
      model_path = os.path.join(logdir1, 'model.ckpt-300')
      saver = saver_lib.Saver(model_variables)

      def RestoreFn(sess):
        saver.restore(sess, model_path)

      loss = learning.train(
          train_op, logdir2, number_of_steps=1, init_fn=RestoreFn)

      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def ModelLoss(self):
    tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

    tf_predictions = LogisticClassifier(tf_inputs)
    loss_ops.log_loss(tf_predictions, tf_labels)
    return loss_ops.get_total_loss()

  def testTrainAllVarsHasLowerLossThanTrainSubsetOfVars(self):
    logdir1 = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs1')

    # First, train only the weights of the model.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      total_loss = self.ModelLoss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      weights = variables_lib2.get_variables_by_name('weights')

      train_op = learning.create_train_op(
          total_loss, optimizer, variables_to_train=weights)

      loss = learning.train(
          train_op, logdir1, number_of_steps=200, log_every_n_steps=10)
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Next, train the biases of the model.
    with ops.Graph().as_default():
      random_seed.set_random_seed(1)
      total_loss = self.ModelLoss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      biases = variables_lib2.get_variables_by_name('biases')

      train_op = learning.create_train_op(
          total_loss, optimizer, variables_to_train=biases)

      loss = learning.train(
          train_op, logdir1, number_of_steps=300, log_every_n_steps=10)
      self.assertGreater(loss, .015)
      self.assertLess(loss, .05)

    # Finally, train both weights and bias to get lower loss.
    with ops.Graph().as_default():
      random_seed.set_random_seed(2)
      total_loss = self.ModelLoss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)
      loss = learning.train(
          train_op, logdir1, number_of_steps=400, log_every_n_steps=10)

      self.assertIsNotNone(loss)
      self.assertLess(loss, .015)

  def testTrainingSubsetsOfVariablesOnlyUpdatesThoseVariables(self):
    # First, train only the weights of the model.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      total_loss = self.ModelLoss()
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      weights, biases = variables_lib2.get_variables()

      train_op = learning.create_train_op(total_loss, optimizer)
      train_weights = learning.create_train_op(
          total_loss, optimizer, variables_to_train=[weights])
      train_biases = learning.create_train_op(
          total_loss, optimizer, variables_to_train=[biases])

      with session.Session() as sess:
        # Initialize the variables.
        sess.run(variables_lib.global_variables_initializer())

        # Get the initial weights and biases values.
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
    logdir1 = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs1')
    logdir2 = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs2')

    multipliers = [1., 1000.]
    number_of_steps = 10
    losses = []
    learning_rate = 0.001

    # First, train the model with equivalently smaller learning rate.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      train_op = self.create_train_op(
          learning_rate=learning_rate, gradient_multiplier=multipliers[0])
      loss = learning.train(train_op, logdir1, number_of_steps=number_of_steps)
      losses.append(loss)
      self.assertGreater(loss, .5)

    # Second, train the model with equivalently larger learning rate.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      train_op = self.create_train_op(
          learning_rate=learning_rate, gradient_multiplier=multipliers[1])
      loss = learning.train(train_op, logdir2, number_of_steps=number_of_steps)
      losses.append(loss)
      self.assertIsNotNone(loss)
      self.assertLess(loss, .5)

    # The loss of the model trained with larger learning rate should
    # be smaller.
    self.assertGreater(losses[0], losses[1])

  def testTrainWithEpochLimit(self):
    logdir = os.path.join(
        tempfile.mkdtemp(prefix=self.get_temp_dir()), 'tmp_logs')
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)
      tf_inputs_limited = input_lib.limit_epochs(tf_inputs, num_epochs=300)
      tf_labels_limited = input_lib.limit_epochs(tf_labels, num_epochs=300)

      tf_predictions = LogisticClassifier(tf_inputs_limited)
      loss_ops.log_loss(tf_predictions, tf_labels_limited)
      total_loss = loss_ops.get_total_loss()

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

      train_op = learning.create_train_op(total_loss, optimizer)

      loss = learning.train(train_op, logdir, log_every_n_steps=10)
    self.assertIsNotNone(loss)
    self.assertLess(loss, .015)
    self.assertTrue(os.path.isfile('{}/model.ckpt-300.index'.format(logdir)))
    self.assertTrue(
        os.path.isfile('{}/model.ckpt-300.data-00000-of-00001'.format(logdir)))


if __name__ == '__main__':
  test.main()
