# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os

from tensorflow.contrib.estimator.python.estimator import hooks as hooks_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import training


def summary_step_keyword_to_value_mapping(dir_):
  writer_cache.FileWriterCache.clear()

  # Get last Event written.
  event_paths = glob.glob(os.path.join(dir_, 'events*'))
  step_keyword_to_value = {}
  for last_event in summary_iterator.summary_iterator(event_paths[-1]):
    if last_event.step not in step_keyword_to_value:
      step_keyword_to_value[last_event.step] = {}
    if last_event.summary is not None:
      for value in last_event.summary.value:
        step_keyword_to_value[last_event.step][value.tag] = value.simple_value

  return step_keyword_to_value


def get_summary_value(dir_, step, keyword):
  """Get summary value for given step and keyword."""

  writer_cache.FileWriterCache.clear()
  # Get last Event written.
  event_paths = glob.glob(os.path.join(dir_, 'events*'))
  print('XXX', event_paths)
  for last_event in summary_iterator.summary_iterator(event_paths[-1]):
    if last_event.step == step and last_event.summary is not None:
      for value in last_event.summary.value:
        if keyword in value.tag:
          return value.simple_value
  return None


class InMemoryEvaluatorHookTest(test.TestCase):

  def test_runs_eval_metrics(self):

    def model_fn(features, labels, mode):
      _ = labels
      if estimator_lib.ModeKeys.TRAIN == mode:
        with ops.control_dependencies([features]):
          train_op = state_ops.assign_add(training.get_global_step(), 1)
        return estimator_lib.EstimatorSpec(
            mode, loss=constant_op.constant(3.), train_op=train_op)
      if estimator_lib.ModeKeys.EVAL == mode:
        return estimator_lib.EstimatorSpec(
            mode,
            loss=constant_op.constant(5.),
            eval_metric_ops={'mean_of_features': metrics_lib.mean(features)})

    estimator = estimator_lib.Estimator(model_fn=model_fn)

    def input_fn():
      return dataset_ops.Dataset.range(10)

    evaluator = hooks_lib.InMemoryEvaluatorHook(
        estimator, input_fn, every_n_iter=4)
    estimator.train(input_fn, hooks=[evaluator])

    self.assertTrue(os.path.isdir(estimator.eval_dir()))
    step_keyword_to_value = summary_step_keyword_to_value_mapping(
        estimator.eval_dir())
    # 4.5 = sum(range(10))/10
    # before training
    self.assertEqual(4.5, step_keyword_to_value[0]['mean_of_features'])
    # intervals (every_n_iter=4)
    self.assertEqual(4.5, step_keyword_to_value[4]['mean_of_features'])
    self.assertEqual(4.5, step_keyword_to_value[8]['mean_of_features'])
    # end
    self.assertEqual(4.5, step_keyword_to_value[10]['mean_of_features'])

  def test_uses_latest_variable_value(self):

    def model_fn(features, labels, mode):
      _ = labels
      step = training.get_global_step()
      w = variable_scope.get_variable(
          'w',
          shape=[],
          initializer=init_ops.zeros_initializer(),
          dtype=dtypes.int64)
      if estimator_lib.ModeKeys.TRAIN == mode:
        # to consume features, we have control dependency
        with ops.control_dependencies([features]):
          step_inc = state_ops.assign_add(training.get_global_step(), 1)
        with ops.control_dependencies([step_inc]):
          assign_w_to_step_plus_2 = w.assign(step + 2)
        return estimator_lib.EstimatorSpec(
            mode,
            loss=constant_op.constant(3.),
            train_op=assign_w_to_step_plus_2)
      if estimator_lib.ModeKeys.EVAL == mode:
        # to consume features, we have control dependency
        with ops.control_dependencies([features]):
          loss = constant_op.constant(5.)
        return estimator_lib.EstimatorSpec(
            mode,
            loss=loss,
            # w is constant in each step, so the mean.
            # w = 0 if step==0 else step+2
            eval_metric_ops={'mean_of_const': metrics_lib.mean(w)})

    estimator = estimator_lib.Estimator(model_fn=model_fn)

    def input_fn():
      return dataset_ops.Dataset.range(10)

    evaluator = hooks_lib.InMemoryEvaluatorHook(
        estimator, input_fn, every_n_iter=4)
    estimator.train(input_fn, hooks=[evaluator])

    self.assertTrue(os.path.isdir(estimator.eval_dir()))
    step_keyword_to_value = summary_step_keyword_to_value_mapping(
        estimator.eval_dir())
    # w = 0 if step==0 else step+2
    self.assertEqual(0, step_keyword_to_value[0]['mean_of_const'])
    self.assertEqual(5, step_keyword_to_value[4]['mean_of_const'])
    self.assertEqual(11, step_keyword_to_value[10]['mean_of_const'])

  def test_dnn_classifier(self):
    embedding = feature_column_lib.embedding_column(
        feature_column_lib.categorical_column_with_vocabulary_list(
            'wire_cast', ['kima', 'omar', 'stringer']), 8)
    dnn = estimator_lib.DNNClassifier(
        feature_columns=[embedding], hidden_units=[3, 1])

    def train_input_fn():
      return dataset_ops.Dataset.from_tensors(({
          'wire_cast': [['omar'], ['kima']]
      }, [[0], [1]])).repeat(3)

    def eval_input_fn():
      return dataset_ops.Dataset.from_tensors(({
          'wire_cast': [['stringer'], ['kima']]
      }, [[0], [1]])).repeat(2)

    evaluator = hooks_lib.InMemoryEvaluatorHook(
        dnn, eval_input_fn, name='in-memory')
    dnn.train(train_input_fn, hooks=[evaluator])
    self.assertTrue(os.path.isdir(dnn.eval_dir('in-memory')))
    step_keyword_to_value = summary_step_keyword_to_value_mapping(
        dnn.eval_dir('in-memory'))

    final_metrics = dnn.evaluate(eval_input_fn)
    step = final_metrics[ops.GraphKeys.GLOBAL_STEP]
    for summary_tag in final_metrics:
      if summary_tag == ops.GraphKeys.GLOBAL_STEP:
        continue
      self.assertEqual(final_metrics[summary_tag],
                       step_keyword_to_value[step][summary_tag])

  def test_raise_error_with_multi_worker(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host0:0'],
            run_config_lib.TaskType.WORKER: ['host3:3', 'host4:4', 'host5:5']
        },
        'task': {
            'type': run_config_lib.TaskType.CHIEF,
            'index': 0
        }
    }
    with test.mock.patch.dict('os.environ',
                              {'TF_CONFIG': json.dumps(tf_config)}):
      dnn = estimator_lib.DNNClassifier(
          feature_columns=[feature_column_lib.numeric_column('x')],
          hidden_units=[3, 1])

    def eval_input_fn():
      pass

    with self.assertRaisesRegexp(ValueError, 'supports only single machine'):
      hooks_lib.InMemoryEvaluatorHook(dnn, eval_input_fn)

  def test_raise_error_with_ps(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host0:0'],
            run_config_lib.TaskType.PS: ['host1:1'],
        },
        'task': {
            'type': run_config_lib.TaskType.CHIEF,
            'index': 0
        }
    }
    with test.mock.patch.dict('os.environ',
                              {'TF_CONFIG': json.dumps(tf_config)}):
      dnn = estimator_lib.DNNClassifier(
          feature_columns=[feature_column_lib.numeric_column('x')],
          hidden_units=[3, 1])

    def eval_input_fn():
      pass

    with self.assertRaisesRegexp(ValueError, 'supports only single machine'):
      hooks_lib.InMemoryEvaluatorHook(dnn, eval_input_fn)

  def test_raise_error_with_custom_saver_in_eval(self):

    def model_fn(features, labels, mode):
      _, _ = features, labels
      return estimator_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(3.),
          scaffold=training.Scaffold(saver=training.Saver()),
          train_op=constant_op.constant(5.),
          eval_metric_ops={
              'mean_of_features': metrics_lib.mean(constant_op.constant(2.))
          })

    estimator = estimator_lib.Estimator(model_fn=model_fn)

    def input_fn():
      return dataset_ops.Dataset.range(10)

    evaluator = hooks_lib.InMemoryEvaluatorHook(estimator, input_fn)
    with self.assertRaisesRegexp(ValueError, 'does not support custom saver'):
      evaluator.begin()

  def test_raise_error_with_custom_init_fn_in_eval(self):

    def model_fn(features, labels, mode):
      _, _ = features, labels

      def init_fn(scaffold, session):
        _, _ = scaffold, session

      return estimator_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(3.),
          scaffold=training.Scaffold(init_fn=init_fn),
          train_op=constant_op.constant(5.),
          eval_metric_ops={
              'mean_of_features': metrics_lib.mean(constant_op.constant(2.))
          })

    estimator = estimator_lib.Estimator(model_fn=model_fn)

    def input_fn():
      return dataset_ops.Dataset.range(10)

    evaluator = hooks_lib.InMemoryEvaluatorHook(estimator, input_fn)
    with self.assertRaisesRegexp(ValueError, 'does not support custom init_fn'):
      evaluator.begin()

  def test_raise_error_with_saveables_other_than_global_variables(self):

    def model_fn(features, labels, mode):
      _, _ = features, labels
      w = variables.Variable(
          initial_value=[0.],
          trainable=False,
          collections=[ops.GraphKeys.SAVEABLE_OBJECTS])
      init_op = control_flow_ops.group(
          [w.initializer, training.get_global_step().initializer])
      return estimator_lib.EstimatorSpec(
          mode,
          loss=constant_op.constant(3.),
          scaffold=training.Scaffold(init_op=init_op),
          train_op=constant_op.constant(5.),
          eval_metric_ops={
              'mean_of_features': metrics_lib.mean(constant_op.constant(2.))
          })

    estimator = estimator_lib.Estimator(model_fn=model_fn)

    def input_fn():
      return dataset_ops.Dataset.range(10)

    evaluator = hooks_lib.InMemoryEvaluatorHook(estimator, input_fn)
    with self.assertRaisesRegexp(ValueError, 'does not support saveables'):
      estimator.train(input_fn, hooks=[evaluator])


if __name__ == '__main__':
  test.main()
