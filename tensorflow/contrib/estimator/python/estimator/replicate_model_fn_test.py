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
"""Tests for utilities that replicate `Estimator.model_fn` over GPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import shutil
import tempfile
import numpy as np
import six

from tensorflow.contrib.estimator.python.estimator import replicate_model_fn
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.export import export_output
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import gradient_descent


class DNNClassifierIntegrationTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def test_complete_flow(self):
    n_classes = 3
    input_dimension = 2
    batch_size = 12

    data = np.linspace(
        0., n_classes - 1., batch_size * input_dimension, dtype=np.float32)
    x_data = data.reshape(batch_size, input_dimension)
    categorical_data = np.random.random_integers(
        0, len(x_data), size=len(x_data))
    y_data = np.reshape(self._as_label(data[:batch_size]), (batch_size, 1))
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data,
           'categories': categorical_data},
        y=y_data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data,
           'categories': categorical_data},
        y=y_data,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data,
           'categories': categorical_data},
        batch_size=batch_size,
        shuffle=False)

    feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,)),
        feature_column.embedding_column(
            feature_column.categorical_column_with_vocabulary_list(
                'categories',
                vocabulary_list=np.linspace(
                    0., len(x_data), len(x_data), dtype=np.int64)), 1)
    ]

    estimator = dnn.DNNClassifier(
        hidden_units=(2, 2),
        feature_columns=feature_columns,
        n_classes=n_classes,
        model_dir=self._model_dir)

    def optimizer_fn():
      return optimizers.get_optimizer_instance('Adagrad', learning_rate=0.05)

    estimator = estimator_lib.Estimator(
        model_fn=replicate_model_fn.replicate_model_fn(
            estimator.model_fn,
            optimizer_fn,
            devices=['/gpu:0', '/gpu:1', '/gpu:2']),
        model_dir=estimator.model_dir,
        config=estimator.config,
        params=estimator.params)

    num_steps = 10
    estimator.train(train_input_fn, steps=num_steps)

    scores = estimator.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[ops_lib.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))

    predicted_proba = np.array([
        x[prediction_keys.PredictionKeys.PROBABILITIES]
        for x in estimator.predict(predict_input_fn)
    ])
    self.assertAllEqual((batch_size, n_classes), predicted_proba.shape)

    feature_spec = feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = estimator.export_savedmodel(tempfile.mkdtemp(),
                                             serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))

  def _as_label(self, data_in_float):
    return np.rint(data_in_float).astype(np.int64)

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)


class ReplicateModelTest(test_util.TensorFlowTestCase):

  def model_fn(self, mode, features, labels, params):
    c = variable_scope.get_variable(
        'c',
        initializer=constant_op.constant(10, dtype=dtypes.float64),
        dtype=dtypes.float64)

    predictions = math_ops.multiply(features, c)

    loss = None
    if mode is not model_fn_lib.ModeKeys.PREDICT:
      loss = losses.absolute_difference(
          labels=labels,
          predictions=predictions,
          reduction=losses.Reduction.SUM)
      loss = math_ops.reduce_sum(loss)

    metrics = {
        'accuracy': metrics_lib.accuracy(labels, predictions),
        'auc': metrics_lib.auc(labels, predictions)
    }

    return model_fn_lib.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=metrics,
        predictions={'probabilities': predictions},
        train_op=control_flow_ops.no_op())  # This train_op isn't actually used.

  def optimizer_fn(self, params):
    return gradient_descent.GradientDescentOptimizer(params['learning_rate'])

  @property
  def params(self):
    params = {}
    params['learning_rate'] = 1.0
    return params

  def test_train(self):
    features = np.array([[1.0], [2.0]])
    labels = np.array([[1.0], [2.0]])

    with self.test_session() as session:
      replicated_model_fn = replicate_model_fn.replicate_model_fn(
          self.model_fn, self.optimizer_fn, devices=['/gpu:0', '/gpu:1'])
      estimator_spec = replicated_model_fn(
          features, labels, model_fn_lib.ModeKeys.TRAIN, self.params)
      session.run(variables.global_variables_initializer())

      # loss = feature * c - label
      total_loss = (1.0 * 10 - 1.0) + (2.0 * 10 - 2.0)
      self.assertEqual(total_loss, session.run(estimator_spec.loss))

      # loss' of c is 3.
      # new value of c = 10 - learning rate * 3 = 7.0.
      session.run(estimator_spec.train_op)
      with variable_scope.variable_scope('', reuse=True):
        c = variable_scope.get_variable('c', dtype=dtypes.float64)
        self.assertEqual(7.0, session.run(c))

  def test_train_spec_with_optimizer_without_params(self):

    def optimizer_fn_without_params():
      return gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

    features = np.array([[1.0], [2.0]])
    labels = np.array([[1.0], [2.0]])

    with self.test_session() as session:  # pylint: disable=unused-variable
      replicated_model_fn = replicate_model_fn.replicate_model_fn(
          self.model_fn,
          optimizer_fn_without_params,
          devices=['/gpu:0', '/gpu:1'])
      # This call is going to fail if `replicated_model_fn` is still passing
      # `params` inside `optimizer_fn`, even though the latter doesn't take any:
      estimator_spec = replicated_model_fn(
          features, labels, model_fn_lib.ModeKeys.TRAIN, self.params)
      del estimator_spec

  def test_eval(self):
    features = np.array([[0.01], [0.002]])
    labels = np.array([[0.01], [0.02]])

    with self.test_session() as session:
      replicated_model_fn = replicate_model_fn.replicate_model_fn(
          self.model_fn, self.optimizer_fn, devices=['/gpu:0', '/gpu:1'])
      estimator_spec = replicated_model_fn(
          features, labels, model_fn_lib.ModeKeys.EVAL, self.params)
      session.run(variables.local_variables_initializer())
      session.run(variables.global_variables_initializer())

      accuracy, a = estimator_spec.eval_metric_ops['accuracy']
      auc, b = estimator_spec.eval_metric_ops['auc']

      session.run([a, b])
      accuracy = session.run(accuracy)
      auc = session.run(auc)

      # loss[i] = features[i] * 10 - labels[i].
      # Accuracy is 0.0 (no match) in the first tower.
      # Accuracy is 1.0 (match) in the second tower, since the feature
      # times weight "c" happened to be equal to the label.
      total_loss = ((0.01 * 10 - 0.01) + (0.002 * 10 - 0.02))

      self.assertNear((0.0 + 1.0) / 2.0, accuracy, 0.01)
      self.assertEqual(0, auc)
      self.assertNear(total_loss, session.run(estimator_spec.loss), 0.01)

  def test_predict(self):
    features = np.array([[0.01], [0.002]])
    labels = np.array([[0.01], [0.02]])

    with self.test_session() as session:
      replicated_model_fn = replicate_model_fn.replicate_model_fn(
          self.model_fn, self.optimizer_fn, devices=['/gpu:0', '/gpu:1'])
      estimator_spec = replicated_model_fn(
          features, labels, model_fn_lib.ModeKeys.PREDICT, self.params)
      session.run(variables.global_variables_initializer())

      self.assertAllClose({
          'probabilities': np.array([[0.1], [0.02]])
      }, session.run(estimator_spec.predictions))

  def test_train_single_tower(self):
    features = np.array([[1.0], [2.0]])
    labels = np.array([[1.0], [2.0]])

    with self.test_session() as session:
      replicated_model_fn = replicate_model_fn.replicate_model_fn(
          self.model_fn, self.optimizer_fn)
      estimator_spec = replicated_model_fn(
          features, labels, model_fn_lib.ModeKeys.TRAIN, self.params)
      session.run(variables.global_variables_initializer())

      # loss = feature * c - label
      total_loss = (1.0 * 10 - 1.0) + (2.0 * 10 - 2.0)
      self.assertEqual(total_loss, session.run(estimator_spec.loss))

      # loss' of c is 3.
      # new value of c = 10 - learning rate * 3 = 7.0.
      session.run(estimator_spec.train_op)
      with variable_scope.variable_scope('', reuse=True):
        c = variable_scope.get_variable('c', dtype=dtypes.float64)
        self.assertEqual(7.0, session.run(c))

  def test_eval_single_tower(self):
    features = np.array([[0.01], [0.002]])
    labels = np.array([[0.01], [0.02]])

    with self.test_session() as session:
      replicated_model_fn = replicate_model_fn.replicate_model_fn(
          self.model_fn, self.optimizer_fn, devices=['/gpu:0'])
      estimator_spec = replicated_model_fn(
          features, labels, model_fn_lib.ModeKeys.EVAL, self.params)
      session.run(variables.local_variables_initializer())
      session.run(variables.global_variables_initializer())

      accuracy, a = estimator_spec.eval_metric_ops['accuracy']
      auc, b = estimator_spec.eval_metric_ops['auc']

      session.run([a, b])
      accuracy = session.run(accuracy)
      auc = session.run(auc)

      # Accuracy is 0.0 (no match) in the first tower.
      # Accuracy is 1.0 (match) in the second tower, since the feature
      # times weight "c" happened to be equal to the label.
      total_loss = ((0.01 * 10 - 0.01) + (0.002 * 10 - 0.02))

      self.assertNear((0.0 + 1.0) / 2.0, accuracy, 0.01)
      self.assertEqual(0, auc)
      self.assertNear(total_loss, session.run(estimator_spec.loss), 0.01)

  def test_predict_single_tower(self):
    features = np.array([[0.01], [0.002]])
    labels = np.array([[0.01], [0.02]])

    with self.test_session() as session:
      replicated_model_fn = replicate_model_fn.replicate_model_fn(
          self.model_fn, self.optimizer_fn, devices=['/gpu:0'])
      estimator_spec = replicated_model_fn(
          features, labels, model_fn_lib.ModeKeys.PREDICT, self.params)
      session.run(variables.global_variables_initializer())

      self.assertAllClose({
          'probabilities': np.array([[0.1], [0.02]])
      }, session.run(estimator_spec.predictions))


class GetLossTowersTest(test_util.TensorFlowTestCase):

  def model_fn(self, mode, features, labels, params):
    c = variable_scope.get_variable(
        'c',
        initializer=constant_op.constant(0.25, dtype=dtypes.float64),
        dtype=dtypes.float64)

    predictions = math_ops.add(np.array([0.1, 0.2, 0.3, features[0]]), c)
    labels = np.array([0.1, 0.2, 0.3, labels[0]])

    loss = losses.absolute_difference(
        labels=labels, predictions=predictions, reduction=losses.Reduction.SUM)

    return model_fn_lib.EstimatorSpec(mode=mode, loss=math_ops.reduce_sum(loss))

  def test_gradients_are_computed(self):
    with self.test_session() as session:
      tower_specs = replicate_model_fn._get_loss_towers(
          self.model_fn,
          mode=None,
          features=[[0.6], [1.6]],
          labels=[[0.6], [0.6]],
          params=None,
          config=None,
          devices=['/gpu:0', '/gpu:1'],
          local_ps_device='/gpu:0',
          name_scope_pattern='test_tower_{}')
      session.run(variables.global_variables_initializer())

      self.assertEqual(len(tower_specs), 2)

      self.assertEqual('/device:GPU:0', tower_specs[0].loss.device)
      self.assertEqual('Sum:0', tower_specs[0].loss.name)
      self.assertEqual(1.0, session.run(tower_specs[0].loss))

      self.assertEqual('/device:GPU:1', tower_specs[1].loss.device)
      self.assertEqual('test_tower_1/Sum:0', tower_specs[1].loss.name)
      # The input batch for the second tower had a loss that is 1.0
      # bigger: 0.6 vs 1.6.
      self.assertEqual(2.0, session.run(tower_specs[1].loss))

      self.assertEqual(1, len(variables.global_variables()))
      self.assertEqual(1, len(variables.trainable_variables()))

      with variable_scope.variable_scope('', reuse=True):
        c = variable_scope.get_variable('c', dtype=dtypes.float64)
        self.assertEqual(0.25, session.run(c))


class SplitBatchTest(test_util.TensorFlowTestCase):

  def evaluate_shards(self, first_list, second_list):
    evaluate_items = lambda x: x.eval()
    return list(map(evaluate_items, first_list)), list(
        map(evaluate_items, second_list))

  def test_simple_half_split(self):
    with self.test_session() as session:  # pylint: disable=unused-variable
      features = [0.0, 1.0, 2.0, 3.0]
      labels = [10.0, 11.0, 12.0, 13.0]
      feature_shards, label_shards = replicate_model_fn._split_batch(
          features, labels, 2, device='/gpu:0')

      feature_shards, label_shards = self.evaluate_shards(
          feature_shards, label_shards)

      self.assertAllEqual([[0.0, 1.0], [2.0, 3.0]], feature_shards)
      self.assertAllEqual([[10.0, 11.0], [12.0, 13.0]], label_shards)

  def test_to_each_their_own(self):
    with self.test_session() as session:  # pylint: disable=unused-variable
      features = [0.0, 1.0, 2.0, 3.0]
      labels = [10.0, 11.0, 12.0, 13.0]
      feature_shards, label_shards = replicate_model_fn._split_batch(
          features, labels, 4, device='/gpu:0')

      feature_shards, label_shards = self.evaluate_shards(
          feature_shards, label_shards)

      self.assertAllEqual([[0.0], [1.0], [2.0], [3.0]], feature_shards)
      self.assertAllEqual([[10.0], [11.0], [12.0], [13.0]], label_shards)

  def test_one_batch(self):
    with self.test_session() as session:  # pylint: disable=unused-variable
      features = [0.0, 1.0, 2.0, 3.0]
      labels = [10.0, 11.0, 12.0, 13.0]
      feature_shards, label_shards = replicate_model_fn._split_batch(
          features, labels, 1, device='/gpu:0')

      feature_shards, label_shards = self.evaluate_shards(
          feature_shards, label_shards)

      self.assertAllEqual([[0.0, 1.0, 2.0, 3.0]], feature_shards)
      self.assertAllEqual([[10.0, 11.0, 12.0, 13.0]], label_shards)

  def test_half_split_in_dictionary(self):
    with self.test_session() as session:  # pylint: disable=unused-variable
      features = {'first': [0.0, 1.0, 2.0, 3.0], 'second': [4.0, 5.0, 6.0, 7.0]}
      labels = [10.0, 11.0, 12.0, 13.0]

      feature_shards, label_shards = replicate_model_fn._split_batch(
          features, labels, 2, device='/gpu:0')

      self.assertAllEqual([0.0, 1.0], feature_shards[0]['first'].eval())
      self.assertAllEqual([4.0, 5.0], feature_shards[0]['second'].eval())
      self.assertAllEqual([2.0, 3.0], feature_shards[1]['first'].eval())
      self.assertAllEqual([6.0, 7.0], feature_shards[1]['second'].eval())
      self.assertAllEqual([10.0, 11.0], label_shards[0].eval())
      self.assertAllEqual([12.0, 13.0], label_shards[1].eval())

  def test_one_batch_in_dictionary(self):
    with self.test_session() as session:  # pylint: disable=unused-variable
      features = {'first': [0.0, 1.0, 2.0, 3.0], 'second': [4.0, 5.0, 6.0, 7.0]}
      labels = [10.0, 11.0, 12.0, 13.0]

      feature_shards, label_shards = replicate_model_fn._split_batch(
          features, labels, 1, device='/gpu:0')

      self.assertAllEqual([0.0, 1.0, 2.0, 3.0],
                          feature_shards[0]['first'].eval())
      self.assertAllEqual([4.0, 5.0, 6.0, 7.0],
                          feature_shards[0]['second'].eval())
      self.assertAllEqual([10.0, 11.0, 12.0, 13.0], label_shards[0].eval())

  def test_feature_and_label_dictionaries(self):
    with self.test_session() as session:  # pylint: disable=unused-variable
      features = {'first': [0.0, 1.0, 2.0, 3.0], 'second': [4.0, 5.0, 6.0, 7.0]}
      labels = {'first': [10.0, 11.0], 'second': [12.0, 13.0]}

      feature_shards, label_shards = replicate_model_fn._split_batch(
          features, labels, 2, device='/gpu:0')

      self.assertAllEqual([0.0, 1.0], feature_shards[0]['first'].eval())
      self.assertAllEqual([4.0, 5.0], feature_shards[0]['second'].eval())
      self.assertAllEqual([2.0, 3.0], feature_shards[1]['first'].eval())
      self.assertAllEqual([6.0, 7.0], feature_shards[1]['second'].eval())
      self.assertAllEqual([10.0], label_shards[0]['first'].eval())
      self.assertAllEqual([12.0], label_shards[0]['second'].eval())
      self.assertAllEqual([11], label_shards[1]['first'].eval())
      self.assertAllEqual([13.0], label_shards[1]['second'].eval())


class TrainSpecTest(test_util.TensorFlowTestCase):

  expected_predictions = {}

  def create_estimator_spec(self, loss):
    return model_fn_lib.EstimatorSpec(
        mode=model_fn_lib.ModeKeys.TRAIN,
        loss=loss,
        train_op=loss,  # Not used; currently required.
        predictions=self.expected_predictions)

  def create_constant_loss(self, loss_value):
    return constant_op.constant(loss_value, dtype=dtypes.float64)

  def test_example(self):
    with self.test_session() as session:
      tower_losses = list(map(self.create_constant_loss, [2, 4, 6]))
      tower_specs = list(map(self.create_estimator_spec, tower_losses))

      expected_train_op = tower_losses[1]

      estimator_spec = replicate_model_fn._train_spec(
          tower_specs, expected_train_op, aggregation_device='/gpu:0')

      self.assertEqual(expected_train_op, estimator_spec.train_op)
      self.assertEqual(2 + 4 + 6, session.run(estimator_spec.loss))
      self.assertEqual(self.expected_predictions, estimator_spec.predictions)


class EvalSpecTest(test_util.TensorFlowTestCase):

  def create_estimator_spec(self, loss, metrics):
    return model_fn_lib.EstimatorSpec(
        mode=model_fn_lib.ModeKeys.EVAL, loss=loss, eval_metric_ops=metrics)

  def create_constant_loss(self, loss_value):
    return constant_op.constant(loss_value, dtype=dtypes.float64)

  def create_eval_metrics(self, noise):
    predictions = np.array([0.1, 0.2, 0.3, 0.6 + noise])
    labels = np.array([0.1, 0.2, 0.3, 0.6])

    metrics = {
        'accuracy': metrics_lib.accuracy(labels, predictions),
        'auc': metrics_lib.auc(labels, predictions)
    }
    return metrics

  def test_example(self):
    with self.test_session() as session:
      tower_losses = map(self.create_constant_loss, [2, 4, 6])
      tower_metrics = map(self.create_eval_metrics, [0, 0.2, 0.3])
      tower_specs = [
          self.create_estimator_spec(l, m)
          for l, m in zip(tower_losses, tower_metrics)
      ]
      session.run(variables.local_variables_initializer())

      estimator_spec = replicate_model_fn._eval_spec(
          tower_specs, aggregation_device='/device:GPU:0')

      accuracy, a = estimator_spec.eval_metric_ops['accuracy']
      auc, b = estimator_spec.eval_metric_ops['auc']

      self.assertEqual('/device:CPU:0', accuracy.device)
      self.assertEqual('/device:CPU:0', auc.device)

      session.run([a, b])
      accuracy, auc = session.run([accuracy, auc])

      self.assertNear((12 - 2) / 12, accuracy, 0.01)
      self.assertEqual(0, auc)
      self.assertEqual(2 + 4 + 6, session.run(estimator_spec.loss))

  def test_handles_single_tower(self):
    with self.test_session() as session:
      tower_losses = map(self.create_constant_loss, [5])
      tower_metrics = map(self.create_eval_metrics, [0.2])
      tower_specs = [
          self.create_estimator_spec(l, m)
          for l, m in zip(tower_losses, tower_metrics)
      ]
      session.run(variables.local_variables_initializer())

      estimator_spec = replicate_model_fn._eval_spec(
          tower_specs, aggregation_device='/device:GPU:0')

      accuracy, a = estimator_spec.eval_metric_ops['accuracy']
      auc, b = estimator_spec.eval_metric_ops['auc']

      self.assertEqual('/device:CPU:0', accuracy.device)
      self.assertEqual('/device:CPU:0', auc.device)

      session.run([a, b])
      accuracy = session.run(accuracy)
      auc = session.run(auc)

      self.assertNear((4 - 1) / 4, accuracy, 0.01)
      self.assertEqual(0, auc)
      self.assertEqual(5, session.run(estimator_spec.loss))


class PredictSpecTest(test_util.TensorFlowTestCase):

  def model_fn(self, mode, features, labels, params):
    c = variable_scope.get_variable(
        'c',
        initializer=constant_op.constant(0.25, dtype=dtypes.float64),
        dtype=dtypes.float64)

    predictions = math_ops.add(np.array([features[0], features[0]]), c)

    return model_fn_lib.EstimatorSpec(
        mode=model_fn_lib.ModeKeys.PREDICT,
        predictions={
            'probabilities': predictions
        })

  def test_example(self):
    with self.test_session() as session:
      tower_specs = replicate_model_fn._get_loss_towers(
          self.model_fn,
          mode=None,
          features=[[0.1], [0.2]],
          labels=[[], []],
          params=None,
          config=None,
          devices=['/gpu:0', '/gpu:1'],
          local_ps_device='/gpu:0',
      )
      session.run(variables.global_variables_initializer())

      estimator_spec = replicate_model_fn._predict_spec(
          tower_specs, aggregation_device='/gpu:0')

      self.assertEqual('/device:GPU:0',
                       estimator_spec.predictions['probabilities'].device)
      self.assertAllClose({
          'probabilities': np.array([0.35, 0.35, 0.45, 0.45])
      }, session.run(estimator_spec.predictions))


class ReduceMetricVariablesTest(test_util.TensorFlowTestCase):

  def create_metric_variable(self, initial_value, name):
    return variable_scope.variable(
        initial_value,
        trainable=False,
        collections=[ops_lib.GraphKeys.METRIC_VARIABLES],
        validate_shape=True,
        name=name)

  def create_tower_metrics(self, tower_id):
    with variable_scope.variable_scope('', reuse=(tower_id != 0)):
      self.create_metric_variable(1.3 * (tower_id + 1), 'total')
      self.create_metric_variable(2.3 * (tower_id + 1), 'count')
      self.create_metric_variable(
          np.array([3.3, 3.5, 3.7]) * (tower_id + 1), 'total')

  def test_example(self):
    with self.test_session() as session:
      for tower_id in range(3):
        self.create_tower_metrics(tower_id)

      session.run(
          variables.variables_initializer(
              ops_lib.get_collection(ops_lib.GraphKeys.METRIC_VARIABLES)))

      session.run(
          replicate_model_fn._reduce_metric_variables(number_of_towers=3))

      # 1st tower = 1.3, 2.3,  [3.3, 3.5, 3.7]
      # 2nd tower = 2.6, 4.6,  [6.6, 7.0, 7.4]
      # 3rd tower = 3.9, 6.9,  [9.9, 10.5, 11.1]
      # Reduced =   7.8, 13.8, [19.8, 21.0, 22.2]
      # Towers are accumulated in the first tower.
      local_metrics = session.run(
          ops_lib.get_collection(ops_lib.GraphKeys.METRIC_VARIABLES))

      self.assertNear(7.8, local_metrics[0], 0.01)
      self.assertNear(13.8, local_metrics[1], 0.01)
      self.assertAllClose([19.8, 21., 22.1], local_metrics[2], 0.01)
      self.assertNear(0.0, local_metrics[3], 0.01)
      self.assertNear(0.0, local_metrics[4], 0.01)
      self.assertAllClose([0.0, 0.0, 0.0], local_metrics[5], 0.01)
      self.assertNear(0.0, local_metrics[6], 0.01)
      self.assertNear(0.0, local_metrics[7], 0.01)
      self.assertAllClose([0.0, 0.0, 0.0], local_metrics[8], 0.01)

  def test_reduce_is_idempotent(self):
    with self.test_session() as session:
      for tower_id in range(3):
        self.create_tower_metrics(tower_id)

      session.run(
          variables.variables_initializer(
              ops_lib.get_collection(ops_lib.GraphKeys.METRIC_VARIABLES)))

      for _ in range(20):
        session.run(
            replicate_model_fn._reduce_metric_variables(number_of_towers=3))

      local_metrics = session.run(
          ops_lib.get_collection(ops_lib.GraphKeys.METRIC_VARIABLES))

      self.assertNear(7.8, local_metrics[0], 0.01)
      self.assertNear(13.8, local_metrics[1], 0.01)
      self.assertAllClose([19.8, 21., 22.1], local_metrics[2], 0.01)
      self.assertNear(0.0, local_metrics[3], 0.01)
      self.assertNear(0.0, local_metrics[4], 0.01)
      self.assertAllClose([0.0, 0.0, 0.0], local_metrics[5], 0.01)
      self.assertNear(0.0, local_metrics[6], 0.01)
      self.assertNear(0.0, local_metrics[7], 0.01)
      self.assertAllClose([0.0, 0.0, 0.0], local_metrics[8], 0.01)

  def test_handles_single_tower(self):
    with self.test_session() as session:
      self.create_tower_metrics(0)
      session.run(
          variables.variables_initializer(
              ops_lib.get_collection(ops_lib.GraphKeys.METRIC_VARIABLES)))

      session.run(
          replicate_model_fn._reduce_metric_variables(number_of_towers=1))

      local_metrics = session.run(
          ops_lib.get_collection(ops_lib.GraphKeys.METRIC_VARIABLES))

      self.assertNear(1.3, local_metrics[0], 0.01)
      self.assertNear(2.3, local_metrics[1], 0.01)
      self.assertAllClose([3.3, 3.5, 3.7], local_metrics[2], 0.01)

  def test_doesnt_accept_uneven_number_of_variables(self):
    with self.test_session() as session:
      for tower_id in range(3):
        self.create_tower_metrics(tower_id)
      self.create_metric_variable(-1.0, 'oddball')

      session.run(
          variables.variables_initializer(
              ops_lib.get_collection(ops_lib.GraphKeys.METRIC_VARIABLES)))

      with self.assertRaisesRegexp(ValueError, ''):
        session.run(
            replicate_model_fn._reduce_metric_variables(number_of_towers=3))


class MergeExportOutputsTest(test_util.TensorFlowTestCase):

  def optimizer_fn(self):
    return gradient_descent.GradientDescentOptimizer(1.0)

  def model_fn(self, mode, features, labels, params):
    c = variable_scope.get_variable(
        'c',
        initializer=constant_op.constant(10, dtype=dtypes.float64),
        dtype=dtypes.float64)

    predictions = {'probabilities': math_ops.multiply(features, c)}
    loss = losses.absolute_difference(
        labels=labels,
        predictions=predictions['probabilities'],
        reduction=losses.Reduction.SUM)

    metrics = {
        'accuracy': metrics_lib.accuracy(labels, predictions['probabilities']),
        'auc': metrics_lib.auc(labels, predictions['probabilities'])
    }
    tensor_string_repr = str(features)
    classes = constant_op.constant(
        re.search('(split_inputs/split:[0-9])', tensor_string_repr).group(1),
        dtype=dtypes.string)

    export_outputs = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            export_output.PredictOutput(predictions),
        'classification_output':
            export_output.ClassificationOutput(predictions['probabilities'],
                                               classes),
        'classification_scores':
            export_output.ClassificationOutput(
                scores=predictions['probabilities']),
        'classification_classes':
            export_output.ClassificationOutput(classes=classes),
        'regression_output':
            export_output.RegressionOutput(predictions['probabilities']),
    }

    return model_fn_lib.EstimatorSpec(
        mode=mode,
        loss=math_ops.reduce_sum(loss),
        eval_metric_ops=metrics,
        predictions=predictions,
        train_op=loss,  # This train_op isn't actually used.
        export_outputs=export_outputs)

  def replicate_estimator_spec(self, session):
    features = np.array([0.01, 0.002])
    labels = np.array([0.01, 0.02])

    replicated_model_fn = replicate_model_fn.replicate_model_fn(
        self.model_fn, self.optimizer_fn, devices=['/gpu:0', '/gpu:1'])
    estimator_spec = replicated_model_fn(features, labels,
                                         model_fn_lib.ModeKeys.PREDICT, {})
    session.run(variables.global_variables_initializer())
    return estimator_spec

  def test_merde_predict_output(self):
    with self.test_session() as session:
      estimator_spec = self.replicate_estimator_spec(session)
      self.assertAllClose(
          {
              'probabilities': np.array([0.1, 0.02])
          },
          session.run(estimator_spec.export_outputs[
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs))

  def test_merge_classification_output_scores_classes(self):
    with self.test_session() as session:
      estimator_spec = self.replicate_estimator_spec(session)
      self.assertAllClose(
          [0.1, 0.02],
          session.run(
              estimator_spec.export_outputs['classification_output'].scores))
      self.assertAllEqual(
          [b'split_inputs/split:0', b'split_inputs/split:1'],
          session.run(
              estimator_spec.export_outputs['classification_output'].classes))

  def test_merge_classification_output_scores(self):
    with self.test_session() as session:
      estimator_spec = self.replicate_estimator_spec(session)
      self.assertAllClose(
          [0.1, 0.02],
          session.run(
              estimator_spec.export_outputs['classification_scores'].scores))
      self.assertEqual(
          None, estimator_spec.export_outputs['classification_scores'].classes)

  def test_merge_classification_output_classes(self):
    with self.test_session() as session:
      estimator_spec = self.replicate_estimator_spec(session)
      self.assertAllEqual(
          [b'split_inputs/split:0', b'split_inputs/split:1'],
          session.run(
              estimator_spec.export_outputs['classification_classes'].classes))
      self.assertEqual(
          None, estimator_spec.export_outputs['classification_classes'].scores)

  def test_merge_regression_output(self):
    with self.test_session() as session:
      estimator_spec = self.replicate_estimator_spec(session)
      self.assertAllClose(
          [0.1, 0.02],
          session.run(estimator_spec.export_outputs['regression_output'].value))


class GetLocalDevicesTest(test_util.TensorFlowTestCase):

  def test_there_is_at_least_a_cpu(self):
    self.assertTrue(replicate_model_fn._get_local_devices('CPU'))

  def test_there_is_no_xpu(self):
    self.assertFalse(
        replicate_model_fn._get_local_devices('XPU'))  # XPU doesn't exist.

  def test_whether_there_is_a_gpu(self):
    if test.is_gpu_available():
      self.assertTrue(len(replicate_model_fn._get_local_devices('GPU')))


class LocalDeviceSetterTest(test_util.TensorFlowTestCase):

  def test_vars_are_on_ps_but_ops_are_on_workers(self):
    local_device_setter = replicate_model_fn._local_device_setter(
        ps_device='/device:GPU:3', worker_device='/device:GPU:2')

    with ops_lib.device(local_device_setter):
      c = variables.Variable(0.01)
      self.assertEqual('/device:GPU:3', c.device)

      cc = variables.Variable(0.02)
      self.assertEqual('/device:GPU:3', cc.device)

      ccc = variables.Variable(0.03)
      self.assertEqual('/device:GPU:3', ccc.device)

      c_op = array_ops.concat(c, axis=0)
      self.assertEqual('/device:GPU:2', c_op.device)

      cc_op = array_ops.concat(cc, axis=0)
      self.assertEqual('/device:GPU:2', cc_op.device)


class ComputeSumWithDevicePlacementTest(test_util.TensorFlowTestCase):

  def test_vectors(self):
    with self.test_session() as session:
      total = replicate_model_fn._compute_sum_on_device(
          [1.0, 2.0, 3.0, 4.0], device='/device:GPU:0', name='test_sum')

      self.assertEqual('/device:GPU:0', total.device)
      self.assertEqual('test_sum', total.op.name)
      self.assertEqual(10.0, session.run(total))

  def test_tensors(self):
    with self.test_session() as session:
      total = replicate_model_fn._compute_sum_on_device(
          [[1.0, 2.0], [3.0, 4.0]], device='/device:GPU:0', name='test_sum')

      self.assertEqual('/device:GPU:0', total.device)
      self.assertEqual('test_sum', total.op.name)
      self.assertAllEqual([4.0, 6.0], session.run(total))

  def test_indexedslices(self):
    with self.test_session() as session:
      a = ops_lib.IndexedSlices(
          constant_op.constant([1.0, 2.0]), [0, 1],
          dense_shape=constant_op.constant([2]))
      b = ops_lib.IndexedSlices(constant_op.constant([3.0, 4.0]), [0, 1])

      total = replicate_model_fn._compute_sum_on_device(
          [a, b], device='/device:GPU:0')

      self.assertEqual('/device:GPU:0', total.device)
      self.assertAllEqual([4.0, 6.0],
                          session.run(ops_lib.convert_to_tensor(total)))

  def test_indexedslices_higher_dimensions(self):
    with self.test_session() as session:
      a = ops_lib.IndexedSlices(
          constant_op.constant([[1.0, 5.0], [2.0, 6.0]]), [0, 1],
          dense_shape=constant_op.constant([2, 4]))
      b = ops_lib.IndexedSlices(
          constant_op.constant([[3.0, 7.0], [4.0, 8.0]]), [0, 1])

      total = replicate_model_fn._compute_sum_on_device(
          [a, b], device='/device:GPU:0')

      self.assertEqual('/device:GPU:0', total.device)
      self.assertAllEqual([[4.0, 12.0], [6.0, 14.0]],
                          session.run(ops_lib.convert_to_tensor(total)))

  def test_indexedslices_some_dont_overlap(self):
    with self.test_session() as session:
      a = ops_lib.IndexedSlices(
          constant_op.constant([1.0, 2.0]), [0, 3],
          dense_shape=constant_op.constant([4]))
      b = ops_lib.IndexedSlices(constant_op.constant([3.0, 4.0]), [0, 1])

      total = replicate_model_fn._compute_sum_on_device(
          [a, b], device='/device:GPU:0')

      self.assertEqual('/device:GPU:0', total.device)
      self.assertAllEqual([4.0, 4.0, 0.0, 2.0],
                          session.run(ops_lib.convert_to_tensor(total)))

  def test_no_name_for_indexslices(self):
    a = ops_lib.IndexedSlices(
        constant_op.constant([1.0, 2.0]), [0, 1],
        dense_shape=constant_op.constant([2]))
    b = ops_lib.IndexedSlices(constant_op.constant([3.0, 4.0]), [0, 1])

    with self.assertRaisesRegexp(ValueError, ''):
      _ = replicate_model_fn._compute_sum_on_device(
          [a, b], device='/device:GPU:0', name='cant_name_indexslices')


class ConcatTensorDictsTest(test_util.TensorFlowTestCase):

  def test_example(self):
    tensor_dicts = [
        {
            'a': np.array([1.0, 2.0]),
            'b': np.array([11.0]),
            'c': np.array([21.0]),
        },
        {
            'a': np.array([3.0]),
            'b': np.array([12.0, 13.0]),
        },
        {
            'b': np.array([14.0]),
        },
    ]

    with self.test_session() as session:
      self.assertAllClose({
          'a': np.array([1.0, 2.0, 3.0]),
          'b': np.array([11.0, 12.0, 13.0, 14.0]),
          'c': np.array([21.0]),
      }, session.run(replicate_model_fn._concat_tensor_dicts(*tensor_dicts)))


if __name__ == '__main__':
  test.main()
