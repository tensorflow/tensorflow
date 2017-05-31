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
"""Tests for linear.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np
import six

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.estimator.inputs import pandas_io
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver
from tensorflow.python.training import session_run_hook


try:
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  HAS_PANDAS = True
except IOError:
  # Pandas writes a temporary file during import. If it fails, don't use pandas.
  HAS_PANDAS = False
except ImportError:
  HAS_PANDAS = False


# Names of variables created by model.
_AGE_WEIGHT_NAME = 'linear/linear_model/age/weights'
_HEIGHT_WEIGHT_NAME = 'linear/linear_model/height/weights'
_BIAS_NAME = 'linear/linear_model/bias_weights'
_LANGUAGE_WEIGHT_NAME = 'linear/linear_model/language/weights'


def _save_variables_to_ckpt(model_dir):
  init_all_op = [variables.global_variables_initializer()]
  with tf_session.Session() as sess:
    sess.run(init_all_op)
    saver.Saver().save(sess, os.path.join(model_dir, 'model.ckpt'))


def _queue_parsed_features(feature_map):
  tensors_to_enqueue = []
  keys = []
  for key, tensor in six.iteritems(feature_map):
    keys.append(key)
    tensors_to_enqueue.append(tensor)
  queue_dtypes = [x.dtype for x in tensors_to_enqueue]
  input_queue = data_flow_ops.FIFOQueue(capacity=100, dtypes=queue_dtypes)
  queue_runner.add_queue_runner(
      queue_runner.QueueRunner(
          input_queue,
          [input_queue.enqueue(tensors_to_enqueue)]))
  dequeued_tensors = input_queue.dequeue()
  return {keys[i]: dequeued_tensors[i] for i in range(len(dequeued_tensors))}


class _CheckPartitionerVarHook(session_run_hook.SessionRunHook):
  """A `SessionRunHook` to check a paritioned variable."""

  def __init__(self, test_case, var_name, var_dim, partitions):
    self._test_case = test_case
    self._var_name = var_name
    self._var_dim = var_dim
    self._partitions = partitions

  def begin(self):
    with variable_scope.variable_scope(
        variable_scope.get_variable_scope()) as scope:
      scope.reuse_variables()
      partitioned_weight = variable_scope.get_variable(
          self._var_name, shape=(self._var_dim, 1))
      self._test_case.assertTrue(
          isinstance(partitioned_weight, variables.PartitionedVariable))
      for part in partitioned_weight:
        self._test_case.assertEqual(self._var_dim // self._partitions,
                                    part.get_shape()[0])


class LinearRegressorPartitionerTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def testPartitioner(self):
    x_dim = 64
    partitions = 4

    def _partitioner(shape, dtype):
      del dtype  # unused; required by Fn signature.
      # Only partition the embedding tensor.
      return [partitions, 1] if shape[0] == x_dim else [1]

    regressor = linear.LinearRegressor(
        feature_columns=(
            feature_column_lib.categorical_column_with_hash_bucket(
                'language', hash_bucket_size=x_dim),),
        partitioner=_partitioner,
        model_dir=self._model_dir)

    def _input_fn():
      return {
          'language': sparse_tensor.SparseTensor(
              values=['english', 'spanish'],
              indices=[[0, 0], [0, 1]],
              dense_shape=[1, 2])
      }, [[10.]]

    hook = _CheckPartitionerVarHook(
        self, _LANGUAGE_WEIGHT_NAME, x_dim, partitions)
    regressor.train(
        input_fn=_input_fn, steps=1, hooks=[hook])

  def testDefaultPartitionerWithMultiplePsReplicas(self):
    partitions = 2
    # This results in weights larger than the default partition size of 64M,
    # so partitioned weights are created (each weight uses 4 bytes).
    x_dim = 32 << 20

    class FakeRunConfig(run_config.RunConfig):

      @property
      def num_ps_replicas(self):
        return partitions

    # Mock the device setter as ps is not available on test machines.
    with test.mock.patch.object(estimator,
                                '_get_replica_device_setter',
                                return_value=lambda _: '/cpu:0'):
      linear_regressor = linear.LinearRegressor(
          feature_columns=(
              feature_column_lib.categorical_column_with_hash_bucket(
                  'language', hash_bucket_size=x_dim),),
          config=FakeRunConfig(),
          model_dir=self._model_dir)

      def _input_fn():
        return {
            'language': sparse_tensor.SparseTensor(
                values=['english', 'spanish'],
                indices=[[0, 0], [0, 1]],
                dense_shape=[1, 2])
        }, [[10.]]

      hook = _CheckPartitionerVarHook(
          self, _LANGUAGE_WEIGHT_NAME, x_dim, partitions)
      linear_regressor.train(
          input_fn=_input_fn, steps=1, hooks=[hook])


# TODO(b/36813849): Add tests with dynamic shape inputs using placeholders.
class LinearRegressorEvaluationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_evaluation_for_simple_data(self):
    with ops.Graph().as_default():
      variables.Variable([[11.0]], name=_AGE_WEIGHT_NAME)
      variables.Variable([2.0], name=_BIAS_NAME)
      variables.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    linear_regressor = linear.LinearRegressor(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        model_dir=self._model_dir)
    eval_metrics = linear_regressor.evaluate(
        input_fn=lambda: ({'age': ((1,),)}, ((10.,),)), steps=1)

    # Logit is (1. * 11.0 + 2.0) = 13, while label is 10. Loss is 3**2 = 9.
    self.assertDictEqual({
        metric_keys.MetricKeys.LOSS: 9.,
        metric_keys.MetricKeys.LOSS_MEAN: 9.,
        ops.GraphKeys.GLOBAL_STEP: 100
    }, eval_metrics)

  def test_evaluation_batch(self):
    """Tests evaluation for batch_size==2."""
    with ops.Graph().as_default():
      variables.Variable([[11.0]], name=_AGE_WEIGHT_NAME)
      variables.Variable([2.0], name=_BIAS_NAME)
      variables.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    linear_regressor = linear.LinearRegressor(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        model_dir=self._model_dir)
    eval_metrics = linear_regressor.evaluate(
        input_fn=lambda: ({'age': ((1,), (1,))}, ((10.,), (10.,))), steps=1)

    # Logit is (1. * 11.0 + 2.0) = 13, while label is 10.
    # Loss per example is 3**2 = 9.
    # Training loss is the sum over batch = 9 + 9 = 18
    # Average loss is the average over batch = 9
    self.assertDictEqual({
        metric_keys.MetricKeys.LOSS: 18.,
        metric_keys.MetricKeys.LOSS_MEAN: 9.,
        ops.GraphKeys.GLOBAL_STEP: 100
    }, eval_metrics)

  def test_evaluation_weights(self):
    """Tests evaluation with weights."""
    with ops.Graph().as_default():
      variables.Variable([[11.0]], name=_AGE_WEIGHT_NAME)
      variables.Variable([2.0], name=_BIAS_NAME)
      variables.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    def _input_fn():
      features = {
          'age': ((1,), (1,)),
          'weights': ((1.,), (2.,))
      }
      labels = ((10.,), (10.,))
      return features, labels

    linear_regressor = linear.LinearRegressor(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        weight_feature_key='weights',
        model_dir=self._model_dir)
    eval_metrics = linear_regressor.evaluate(input_fn=_input_fn, steps=1)

    # Logit is (1. * 11.0 + 2.0) = 13, while label is 10.
    # Loss per example is 3**2 = 9.
    # Training loss is the weighted sum over batch = 9 + 2*9 = 27
    # average loss is the weighted average = 9 + 2*9 / (1 + 2) = 9
    self.assertDictEqual({
        metric_keys.MetricKeys.LOSS: 27.,
        metric_keys.MetricKeys.LOSS_MEAN: 9.,
        ops.GraphKeys.GLOBAL_STEP: 100
    }, eval_metrics)

  def test_evaluation_for_multi_dimensions(self):
    x_dim = 3
    label_dim = 2
    with ops.Graph().as_default():
      variables.Variable(
          [[1.0, 2.0],
           [3.0, 4.0],
           [5.0, 6.0]],
          name=_AGE_WEIGHT_NAME)
      variables.Variable([7.0, 8.0], name=_BIAS_NAME)
      variables.Variable(100, name='global_step', dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    linear_regressor = linear.LinearRegressor(
        feature_columns=(
            feature_column_lib.numeric_column('age', shape=(x_dim,)),),
        label_dimension=label_dim,
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={
            'age': np.array([[2., 4., 5.]]),
        },
        y=np.array([[46., 58.]]),
        batch_size=1,
        num_epochs=None,
        shuffle=False)
    eval_metrics = linear_regressor.evaluate(
        input_fn=input_fn, steps=1)

    self.assertItemsEqual((
        metric_keys.MetricKeys.LOSS,
        metric_keys.MetricKeys.LOSS_MEAN,
        ops.GraphKeys.GLOBAL_STEP
    ), eval_metrics.keys())

    # Logit is
    #   [2., 4., 5.] * [1.0, 2.0] + [7.0, 8.0] = [39, 50] + [7.0, 8.0]
    #                  [3.0, 4.0]
    #                  [5.0, 6.0]
    # which is [46, 58]
    self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])

  def test_evaluation_for_multiple_feature_columns(self):
    with ops.Graph().as_default():
      variables.Variable([[10.0]], name=_AGE_WEIGHT_NAME)
      variables.Variable([[2.0]], name=_HEIGHT_WEIGHT_NAME)
      variables.Variable([5.0], name=_BIAS_NAME)
      variables.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    batch_size = 2
    feature_columns = [
        feature_column_lib.numeric_column('age'),
        feature_column_lib.numeric_column('height')
    ]
    input_fn = numpy_io.numpy_input_fn(
        x={
            'age': np.array([20, 40]),
            'height': np.array([4, 8])
        },
        y=np.array([[213.], [421.]]),
        batch_size=batch_size,
        num_epochs=None,
        shuffle=False)

    est = linear.LinearRegressor(
        feature_columns=feature_columns,
        model_dir=self._model_dir)

    eval_metrics = est.evaluate(input_fn=input_fn, steps=1)
    self.assertItemsEqual((
        metric_keys.MetricKeys.LOSS,
        metric_keys.MetricKeys.LOSS_MEAN,
        ops.GraphKeys.GLOBAL_STEP
    ), eval_metrics.keys())

    # Logit is [(20. * 10.0 + 4 * 2.0 + 5.0), (40. * 10.0 + 8 * 2.0 + 5.0)] =
    # [213.0, 421.0], while label is [213., 421.]. Loss = 0.
    self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])


class LinearRegressorPredictTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_1d(self):
    """Tests predict when all variables are one-dimensional."""
    with ops.Graph().as_default():
      variables.Variable([[10.]], name='linear/linear_model/x/weights')
      variables.Variable([.2], name=_BIAS_NAME)
      variables.Variable(100, name='global_step', dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    linear_regressor = linear.LinearRegressor(
        feature_columns=(feature_column_lib.numeric_column('x'),),
        model_dir=self._model_dir)

    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array([[2.]])}, y=None, batch_size=1, num_epochs=1,
        shuffle=False)
    predictions = linear_regressor.predict(input_fn=predict_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    # x * weight + bias = 2. * 10. + .2 = 20.2
    self.assertAllClose([[20.2]], predicted_scores)

  def testMultiDim(self):
    """Tests predict when all variables are multi-dimenstional."""
    batch_size = 2
    label_dimension = 3
    x_dim = 4
    feature_columns = (
        feature_column_lib.numeric_column('x', shape=(x_dim,)),)
    with ops.Graph().as_default():
      variables.Variable(  # shape=[x_dim, label_dimension]
          [[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.],
           [4., 5., 6.]],
          name='linear/linear_model/x/weights')
      variables.Variable(  # shape=[label_dimension]
          [.2, .4, .6], name=_BIAS_NAME)
      variables.Variable(100, name='global_step', dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    linear_regressor = linear.LinearRegressor(
        feature_columns=feature_columns,
        label_dimension=label_dimension,
        model_dir=self._model_dir)

    predict_input_fn = numpy_io.numpy_input_fn(
        # x shape=[batch_size, x_dim]
        x={'x': np.array([[1., 2., 3., 4.],
                          [5., 6., 7., 8.]])},
        y=None, batch_size=batch_size, num_epochs=1, shuffle=False)
    predictions = linear_regressor.predict(input_fn=predict_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    # score = x * weight + bias, shape=[batch_size, label_dimension]
    self.assertAllClose(
        [[30.2, 40.4, 50.6], [70.2, 96.4, 122.6]], predicted_scores)

  def testTwoFeatureColumns(self):
    """Tests predict with two feature columns."""
    with ops.Graph().as_default():
      variables.Variable([[10.]], name='linear/linear_model/x0/weights')
      variables.Variable([[20.]], name='linear/linear_model/x1/weights')
      variables.Variable([.2], name=_BIAS_NAME)
      variables.Variable(100, name='global_step', dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    linear_regressor = linear.LinearRegressor(
        feature_columns=(
            feature_column_lib.numeric_column('x0'),
            feature_column_lib.numeric_column('x1')),
        model_dir=self._model_dir)

    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x0': np.array([[2.]]),
           'x1': np.array([[3.]])},
        y=None, batch_size=1, num_epochs=1,
        shuffle=False)
    predictions = linear_regressor.predict(input_fn=predict_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    # x0 * weight0 + x1 * weight1 + bias = 2. * 10. + 3. * 20 + .2 = 80.2
    self.assertAllClose([[80.2]], predicted_scores)


class LinearRegressorIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, input_dimension,
      label_dimension, prediction_length, batch_size):
    feature_columns = [
        feature_column_lib.numeric_column('x', shape=(input_dimension,))
    ]
    est = linear.LinearRegressor(
        feature_columns=feature_columns, label_dimension=label_dimension,
        model_dir=self._model_dir)

    # TRAIN
    # learn y = x
    est.train(train_input_fn, steps=200)

    # EVALUTE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(200, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn(metric_keys.MetricKeys.LOSS, six.iterkeys(scores))

    # PREDICT
    predictions = np.array([
        x['predictions'] for x in est.predict(predict_input_fn)])
    self.assertAllEqual((prediction_length, label_dimension), predictions.shape)

    # EXPORT
    feature_spec = feature_column_lib.make_parse_example_spec(
        feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_savedmodel(tempfile.mkdtemp(),
                                       serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))

  def test_numpy_input_fn(self):
    """Tests complete flow with numpy_input_fn."""
    label_dimension = 2
    input_dimension = label_dimension
    batch_size = 10
    prediction_length = batch_size
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=data, batch_size=batch_size, num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=data, batch_size=batch_size, num_epochs=1,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=None, batch_size=batch_size, num_epochs=1,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=input_dimension,
        label_dimension=label_dimension,
        prediction_length=prediction_length,
        batch_size=batch_size)

  def test_pandas_input_fn(self):
    """Tests complete flow with pandas_input_fn."""
    if not HAS_PANDAS:
      return

    # Pandas DataFrame natually supports 1 dim data only.
    label_dimension = 1
    input_dimension = label_dimension
    batch_size = 10
    data = np.array([1., 2., 3., 4.], dtype=np.float32)
    x = pd.DataFrame({'x': data})
    y = pd.Series(data)
    prediction_length = 4

    train_input_fn = pandas_io.pandas_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = pandas_io.pandas_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = pandas_io.pandas_input_fn(
        x=x,
        batch_size=batch_size,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=input_dimension,
        label_dimension=label_dimension,
        prediction_length=prediction_length,
        batch_size=batch_size)

  def test_input_fn_from_parse_example(self):
    """Tests complete flow with input_fn constructed from parse_example."""
    label_dimension = 2
    input_dimension = label_dimension
    batch_size = 10
    prediction_length = batch_size
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)

    serialized_examples = []
    for datum in data:
      example = example_pb2.Example(features=feature_pb2.Features(
          feature={
              'x': feature_pb2.Feature(
                  float_list=feature_pb2.FloatList(value=datum)),
              'y': feature_pb2.Feature(
                  float_list=feature_pb2.FloatList(
                      value=datum[:label_dimension])),
          }))
      serialized_examples.append(example.SerializeToString())

    feature_spec = {
        'x': parsing_ops.FixedLenFeature([input_dimension], dtypes.float32),
        'y': parsing_ops.FixedLenFeature([label_dimension], dtypes.float32),
    }

    def _train_input_fn():
      feature_map = parsing_ops.parse_example(serialized_examples, feature_spec)
      features = _queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _eval_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = _queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _predict_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = _queue_parsed_features(feature_map)
      features.pop('y')
      return features, None

    self._test_complete_flow(
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        input_dimension=input_dimension,
        label_dimension=label_dimension,
        prediction_length=prediction_length,
        batch_size=batch_size)


def _assert_close(expected, actual, rtol=1e-04, name='assert_close'):
  with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
    expected = ops.convert_to_tensor(expected, name='expected')
    actual = ops.convert_to_tensor(actual, name='actual')
    rdiff = math_ops.abs(expected - actual, 'diff') / expected
    rtol = ops.convert_to_tensor(rtol, name='rtol')
    return check_ops.assert_less(
        rdiff,
        rtol,
        data=(
            'Condition expected =~ actual did not hold element-wise:'
            'expected = ', expected,
            'actual = ', actual,
            'rdiff = ', rdiff,
            'rtol = ', rtol,
        ),
        name=scope)


class LinearRegressorTrainingTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _mockOptimizer(self, expected_loss=None):
    expected_var_names = [
        '%s/part_0:0' % _AGE_WEIGHT_NAME,
        '%s/part_0:0' % _BIAS_NAME
    ]

    def _minimize(loss, global_step):
      trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertItemsEqual(
          expected_var_names,
          [var.name for var in trainable_vars])

      # Verify loss. We can't check the value directly, so we add an assert op.
      self.assertEquals(0, loss.shape.ndims)
      if expected_loss is None:
        return state_ops.assign_add(global_step, 1).op
      assert_loss = _assert_close(
          math_ops.to_float(expected_loss, name='expected'), loss,
          name='assert_loss')
      with ops.control_dependencies((assert_loss,)):
        return state_ops.assign_add(global_step, 1).op

    mock_optimizer = test.mock.NonCallableMock(
        spec=optimizer.Optimizer,
        wraps=optimizer.Optimizer(use_locking=False, name='my_optimizer'))
    mock_optimizer.minimize = test.mock.MagicMock(wraps=_minimize)

    # NOTE: Estimator.params performs a deepcopy, which wreaks havoc with mocks.
    # So, return mock_optimizer itself for deepcopy.
    mock_optimizer.__deepcopy__ = lambda _: mock_optimizer
    return mock_optimizer

  def _assertCheckpoint(
      self, expected_global_step, expected_age_weight=None, expected_bias=None):
    shapes = {
        name: shape for (name, shape) in
        checkpoint_utils.list_variables(self._model_dir)
    }

    self.assertEqual([], shapes[ops.GraphKeys.GLOBAL_STEP])
    self.assertEqual(
        expected_global_step,
        checkpoint_utils.load_variable(
            self._model_dir, ops.GraphKeys.GLOBAL_STEP))

    self.assertEqual([1, 1], shapes[_AGE_WEIGHT_NAME])
    if expected_age_weight is not None:
      self.assertEqual(
          expected_age_weight,
          checkpoint_utils.load_variable(self._model_dir, _AGE_WEIGHT_NAME))

    self.assertEqual([1], shapes[_BIAS_NAME])
    if expected_bias is not None:
      self.assertEqual(
          expected_bias,
          checkpoint_utils.load_variable(self._model_dir, _BIAS_NAME))

  def testFromScratchWithDefaultOptimizer(self):
    # Create LinearRegressor.
    label = 5.
    age = 17
    linear_regressor = linear.LinearRegressor(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        model_dir=self._model_dir)

    # Train for a few steps, and validate final checkpoint.
    num_steps = 10
    linear_regressor.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self._assertCheckpoint(num_steps)

  def testTrainWithOneDimLabel(self):
    label_dimension = 1
    batch_size = 20
    feature_columns = [
        feature_column_lib.numeric_column('age', shape=(1,))
    ]
    est = linear.LinearRegressor(
        feature_columns=feature_columns, label_dimension=label_dimension,
        model_dir=self._model_dir)
    data_rank_1 = np.linspace(0., 2., batch_size, dtype=np.float32)
    self.assertEqual((batch_size,), data_rank_1.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1}, y=data_rank_1,
        batch_size=batch_size, num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assertCheckpoint(200)

  def testTrainWithOneDimWeight(self):
    label_dimension = 1
    batch_size = 20
    feature_columns = [
        feature_column_lib.numeric_column('age', shape=(1,))
    ]
    est = linear.LinearRegressor(
        feature_columns=feature_columns, label_dimension=label_dimension,
        weight_feature_key='w',
        model_dir=self._model_dir)

    data_rank_1 = np.linspace(0., 2., batch_size, dtype=np.float32)
    self.assertEqual((batch_size,), data_rank_1.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1, 'w': data_rank_1}, y=data_rank_1,
        batch_size=batch_size, num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assertCheckpoint(200)

  def testFromScratch(self):
    # Create LinearRegressor.
    label = 5.
    age = 17
    # loss = (logits - label)^2 = (0 - 5.)^2 = 25.
    mock_optimizer = self._mockOptimizer(expected_loss=25.)
    linear_regressor = linear.LinearRegressor(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        model_dir=self._model_dir, optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    linear_regressor.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assertCheckpoint(
        expected_global_step=num_steps,
        expected_age_weight=0.,
        expected_bias=0.)

  def testFromCheckpoint(self):
    # Create initial checkpoint.
    age_weight = 10.0
    bias = 5.0
    initial_global_step = 100
    with ops.Graph().as_default():
      variables.Variable([[age_weight]], name=_AGE_WEIGHT_NAME)
      variables.Variable([bias], name=_BIAS_NAME)
      variables.Variable(
          initial_global_step, name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    # logits = age * age_weight + bias = 17 * 10. + 5. = 175
    # loss = (logits - label)^2 = (175 - 5)^2 = 28900
    mock_optimizer = self._mockOptimizer(expected_loss=28900.)
    linear_regressor = linear.LinearRegressor(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        model_dir=self._model_dir, optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    linear_regressor.train(
        input_fn=lambda: ({'age': ((17,),)}, ((5.,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assertCheckpoint(
        expected_global_step=initial_global_step + num_steps,
        expected_age_weight=age_weight,
        expected_bias=bias)

  def testFromCheckpointMultiBatch(self):
    # Create initial checkpoint.
    age_weight = 10.0
    bias = 5.0
    initial_global_step = 100
    with ops.Graph().as_default():
      variables.Variable([[age_weight]], name=_AGE_WEIGHT_NAME)
      variables.Variable([bias], name=_BIAS_NAME)
      variables.Variable(
          initial_global_step, name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      _save_variables_to_ckpt(self._model_dir)

    # logits = age * age_weight + bias
    # logits[0] = 17 * 10. + 5. = 175
    # logits[1] = 15 * 10. + 5. = 155
    # loss = sum(logits - label)^2 = (175 - 5)^2 + (155 - 3)^2 = 52004
    mock_optimizer = self._mockOptimizer(expected_loss=52004.)
    linear_regressor = linear.LinearRegressor(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        model_dir=self._model_dir, optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    linear_regressor.train(
        input_fn=lambda: ({'age': ((17,), (15,))}, ((5.,), (3.,))),
        steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assertCheckpoint(
        expected_global_step=initial_global_step + num_steps,
        expected_age_weight=age_weight,
        expected_bias=bias)

if __name__ == '__main__':
  test.main()
