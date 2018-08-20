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
"""Tests for tf.contrib.training.evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import time

import numpy as np

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.contrib.training.python.training import training
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.summary import summary_iterator
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_lib


class CheckpointIteratorTest(test.TestCase):

  def testReturnsEmptyIfNoCheckpointsFound(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'no_checkpoints_found')

    num_found = 0
    for _ in evaluation.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 0)

  def testReturnsSingleCheckpointIfOneCheckpointFound(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'one_checkpoint_found')
    if not gfile.Exists(checkpoint_dir):
      gfile.MakeDirs(checkpoint_dir)

    global_step = variables.get_or_create_global_step()
    saver = saver_lib.Saver()  # Saves the global step.

    with self.test_session() as session:
      session.run(variables_lib.global_variables_initializer())
      save_path = os.path.join(checkpoint_dir, 'model.ckpt')
      saver.save(session, save_path, global_step=global_step)

    num_found = 0
    for _ in evaluation.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 1)

  def testReturnsSingleCheckpointIfOneShardedCheckpoint(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'one_checkpoint_found_sharded')
    if not gfile.Exists(checkpoint_dir):
      gfile.MakeDirs(checkpoint_dir)

    global_step = variables.get_or_create_global_step()

    # This will result in 3 different checkpoint shard files.
    with ops.device('/cpu:0'):
      variables_lib.Variable(10, name='v0')
    with ops.device('/cpu:1'):
      variables_lib.Variable(20, name='v1')

    saver = saver_lib.Saver(sharded=True)

    with session_lib.Session(
        target='',
        config=config_pb2.ConfigProto(device_count={'CPU': 2})) as session:

      session.run(variables_lib.global_variables_initializer())
      save_path = os.path.join(checkpoint_dir, 'model.ckpt')
      saver.save(session, save_path, global_step=global_step)

    num_found = 0
    for _ in evaluation.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 1)

  def testTimeoutFn(self):
    timeout_fn_calls = [0]
    def timeout_fn():
      timeout_fn_calls[0] += 1
      return timeout_fn_calls[0] > 3

    results = list(
        evaluation.checkpoints_iterator(
            '/non-existent-dir', timeout=0.1, timeout_fn=timeout_fn))
    self.assertEqual([], results)
    self.assertEqual(4, timeout_fn_calls[0])


class WaitForNewCheckpointTest(test.TestCase):

  def testReturnsNoneAfterTimeout(self):
    start = time.time()
    ret = evaluation.wait_for_new_checkpoint(
        '/non-existent-dir', 'foo', timeout=1.0, seconds_to_sleep=0.5)
    end = time.time()
    self.assertIsNone(ret)

    # We've waited one second.
    self.assertGreater(end, start + 0.5)

    # The timeout kicked in.
    self.assertLess(end, start + 1.1)


def logistic_classifier(inputs):
  return layers.fully_connected(inputs, 1, activation_fn=math_ops.sigmoid)


class EvaluateOnceTest(test.TestCase):

  def setUp(self):
    super(EvaluateOnceTest, self).setUp()

    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def _train_model(self, checkpoint_dir, num_steps):
    """Trains a simple classification model.

    Note that the data has been configured such that after around 300 steps,
    the model has memorized the dataset (e.g. we can expect %100 accuracy).

    Args:
      checkpoint_dir: The directory where the checkpoint is written to.
      num_steps: The number of steps to train for.
    """
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = logistic_classifier(tf_inputs)
      loss = loss_ops.log_loss(tf_predictions, tf_labels)

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = training.create_train_op(loss, optimizer)

      loss = training.train(
          train_op,
          checkpoint_dir,
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps)])

      if num_steps >= 300:
        assert loss < .015

  def testEvaluatePerfectModel(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluate_perfect_model_once')

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Run
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    labels = constant_op.constant(self._labels, dtype=dtypes.float32)
    logits = logistic_classifier(inputs)
    predictions = math_ops.round(logits)

    accuracy, update_op = metrics.accuracy(
        predictions=predictions, labels=labels)

    checkpoint_path = evaluation.wait_for_new_checkpoint(checkpoint_dir)

    final_ops_values = evaluation.evaluate_once(
        checkpoint_path=checkpoint_path,
        eval_ops=update_op,
        final_ops={'accuracy': accuracy},
        hooks=[
            evaluation.StopAfterNEvalsHook(1),
        ])
    self.assertTrue(final_ops_values['accuracy'] > .99)

  def testEvalOpAndFinalOp(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'eval_ops_and_final_ops')

    # Train a model for a single step to get a checkpoint.
    self._train_model(checkpoint_dir, num_steps=1)
    checkpoint_path = evaluation.wait_for_new_checkpoint(checkpoint_dir)

    # Create the model so we have something to restore.
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    logistic_classifier(inputs)

    num_evals = 5
    final_increment = 9.0

    my_var = variables.local_variable(0.0, name='MyVar')
    eval_ops = state_ops.assign_add(my_var, 1.0)
    final_ops = array_ops.identity(my_var) + final_increment

    final_ops_values = evaluation.evaluate_once(
        checkpoint_path=checkpoint_path,
        eval_ops=eval_ops,
        final_ops={'value': final_ops},
        hooks=[
            evaluation.StopAfterNEvalsHook(num_evals),
        ])
    self.assertEqual(final_ops_values['value'], num_evals + final_increment)

  def testOnlyFinalOp(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'only_final_ops')

    # Train a model for a single step to get a checkpoint.
    self._train_model(checkpoint_dir, num_steps=1)
    checkpoint_path = evaluation.wait_for_new_checkpoint(checkpoint_dir)

    # Create the model so we have something to restore.
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    logistic_classifier(inputs)

    final_increment = 9.0

    my_var = variables.local_variable(0.0, name='MyVar')
    final_ops = array_ops.identity(my_var) + final_increment

    final_ops_values = evaluation.evaluate_once(
        checkpoint_path=checkpoint_path, final_ops={'value': final_ops})
    self.assertEqual(final_ops_values['value'], final_increment)


class EvaluateRepeatedlyTest(test.TestCase):

  def setUp(self):
    super(EvaluateRepeatedlyTest, self).setUp()

    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def _train_model(self, checkpoint_dir, num_steps):
    """Trains a simple classification model.

    Note that the data has been configured such that after around 300 steps,
    the model has memorized the dataset (e.g. we can expect %100 accuracy).

    Args:
      checkpoint_dir: The directory where the checkpoint is written to.
      num_steps: The number of steps to train for.
    """
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = logistic_classifier(tf_inputs)
      loss = loss_ops.log_loss(tf_predictions, tf_labels)

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = training.create_train_op(loss, optimizer)

      loss = training.train(
          train_op,
          checkpoint_dir,
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps)])

  def testEvaluatePerfectModel(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluate_perfect_model_repeated')

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Run
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    labels = constant_op.constant(self._labels, dtype=dtypes.float32)
    logits = logistic_classifier(inputs)
    predictions = math_ops.round(logits)

    accuracy, update_op = metrics.accuracy(
        predictions=predictions, labels=labels)

    final_values = evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        eval_ops=update_op,
        final_ops={'accuracy': accuracy},
        hooks=[
            evaluation.StopAfterNEvalsHook(1),
        ],
        max_number_of_evaluations=1)
    self.assertTrue(final_values['accuracy'] > .99)

  def testEvaluationLoopTimeout(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluation_loop_timeout')
    if not gfile.Exists(checkpoint_dir):
      gfile.MakeDirs(checkpoint_dir)

    # We need a variable that the saver will try to restore.
    variables.get_or_create_global_step()

    # Run with placeholders. If we actually try to evaluate this, we'd fail
    # since we're not using a feed_dict.
    cant_run_op = array_ops.placeholder(dtype=dtypes.float32)

    start = time.time()
    final_values = evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        eval_ops=cant_run_op,
        hooks=[evaluation.StopAfterNEvalsHook(10)],
        timeout=6)
    end = time.time()
    self.assertFalse(final_values)

    # Assert that we've waited for the duration of the timeout (minus the sleep
    # time).
    self.assertGreater(end - start, 5.0)

    # Then the timeout kicked in and stops the loop.
    self.assertLess(end - start, 7)

  def testEvaluationLoopTimeoutWithTimeoutFn(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluation_loop_timeout_with_timeout_fn')

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Run
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    labels = constant_op.constant(self._labels, dtype=dtypes.float32)
    logits = logistic_classifier(inputs)
    predictions = math_ops.round(logits)

    accuracy, update_op = metrics.accuracy(
        predictions=predictions, labels=labels)

    timeout_fn_calls = [0]
    def timeout_fn():
      timeout_fn_calls[0] += 1
      return timeout_fn_calls[0] > 3

    final_values = evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        eval_ops=update_op,
        final_ops={'accuracy': accuracy},
        hooks=[
            evaluation.StopAfterNEvalsHook(1),
        ],
        eval_interval_secs=1,
        max_number_of_evaluations=2,
        timeout=0.1,
        timeout_fn=timeout_fn)
    # We should have evaluated once.
    self.assertTrue(final_values['accuracy'] > .99)
    # And called 4 times the timeout fn
    self.assertEqual(4, timeout_fn_calls[0])

  def testEvaluateWithEvalFeedDict(self):
    # Create a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluate_with_eval_feed_dict')
    self._train_model(checkpoint_dir, num_steps=1)

    # We need a variable that the saver will try to restore.
    variables.get_or_create_global_step()

    # Create a variable and an eval op that increments it with a placeholder.
    my_var = variables.local_variable(0.0, name='my_var')
    increment = array_ops.placeholder(dtype=dtypes.float32)
    eval_ops = state_ops.assign_add(my_var, increment)

    increment_value = 3
    num_evals = 5
    expected_value = increment_value * num_evals
    final_values = evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        eval_ops=eval_ops,
        feed_dict={increment: 3},
        final_ops={'my_var': array_ops.identity(my_var)},
        hooks=[
            evaluation.StopAfterNEvalsHook(num_evals),
        ],
        max_number_of_evaluations=1)
    self.assertEqual(final_values['my_var'], expected_value)

  def _create_names_to_metrics(self, predictions, labels):
    accuracy0, update_op0 = metrics.accuracy(labels, predictions)
    accuracy1, update_op1 = metrics.accuracy(labels, predictions + 1)

    names_to_values = {'Accuracy': accuracy0, 'Another_accuracy': accuracy1}
    names_to_updates = {'Accuracy': update_op0, 'Another_accuracy': update_op1}
    return names_to_values, names_to_updates

  def _verify_summaries(self, output_dir, names_to_values):
    """Verifies that the given `names_to_values` are found in the summaries.

    Args:
      output_dir: An existing directory where summaries are found.
      names_to_values: A dictionary of strings to values.
    """
    # Check that the results were saved. The events file may have additional
    # entries, e.g. the event version stamp, so have to parse things a bit.
    output_filepath = glob.glob(os.path.join(output_dir, '*'))
    self.assertEqual(len(output_filepath), 1)

    events = summary_iterator.summary_iterator(output_filepath[0])
    summaries = [e.summary for e in events if e.summary.value]
    values = []
    for summary in summaries:
      for value in summary.value:
        values.append(value)
    saved_results = {v.tag: v.simple_value for v in values}
    for name in names_to_values:
      self.assertAlmostEqual(names_to_values[name], saved_results[name], 5)

  def testSummariesAreFlushedToDisk(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'summaries_are_flushed')
    logdir = os.path.join(self.get_temp_dir(), 'summaries_are_flushed_eval')
    if gfile.Exists(logdir):
      gfile.DeleteRecursively(logdir)

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Create the model (which can be restored).
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    logistic_classifier(inputs)

    names_to_values = {'bread': 3.4, 'cheese': 4.5, 'tomato': 2.0}

    for k in names_to_values:
      v = names_to_values[k]
      summary_lib.scalar(k, v)

    evaluation.evaluate_repeatedly(
        checkpoint_dir=checkpoint_dir,
        hooks=[
            evaluation.SummaryAtEndHook(log_dir=logdir),
        ],
        max_number_of_evaluations=1)

    self._verify_summaries(logdir, names_to_values)


if __name__ == '__main__':
  test.main()
