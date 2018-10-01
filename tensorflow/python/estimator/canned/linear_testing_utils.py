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
"""Utils for testing linear estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import shutil
import tempfile

import numpy as np
import six

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.estimator.inputs import pandas_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import optimizer as optimizer_lib
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

# pylint rules which are disabled by default for test files.
# pylint: disable=invalid-name,protected-access,missing-docstring

# Names of variables created by model.
AGE_WEIGHT_NAME = 'linear/linear_model/age/weights'
HEIGHT_WEIGHT_NAME = 'linear/linear_model/height/weights'
OCCUPATION_WEIGHT_NAME = 'linear/linear_model/occupation/weights'
BIAS_NAME = 'linear/linear_model/bias_weights'
LANGUAGE_WEIGHT_NAME = 'linear/linear_model/language/weights'


def assert_close(expected, actual, rtol=1e-04, name='assert_close'):
  with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
    expected = ops.convert_to_tensor(expected, name='expected')
    actual = ops.convert_to_tensor(actual, name='actual')
    rdiff = math_ops.abs(expected - actual, 'diff') / math_ops.abs(expected)
    rtol = ops.convert_to_tensor(rtol, name='rtol')
    return check_ops.assert_less(
        rdiff,
        rtol,
        data=('Condition expected =~ actual did not hold element-wise:'
              'expected = ', expected, 'actual = ', actual, 'rdiff = ', rdiff,
              'rtol = ', rtol,),
        name=scope)


def save_variables_to_ckpt(model_dir):
  init_all_op = [variables_lib.global_variables_initializer()]
  with tf_session.Session() as sess:
    sess.run(init_all_op)
    saver.Saver().save(sess, os.path.join(model_dir, 'model.ckpt'))


def queue_parsed_features(feature_map):
  tensors_to_enqueue = []
  keys = []
  for key, tensor in six.iteritems(feature_map):
    keys.append(key)
    tensors_to_enqueue.append(tensor)
  queue_dtypes = [x.dtype for x in tensors_to_enqueue]
  input_queue = data_flow_ops.FIFOQueue(capacity=100, dtypes=queue_dtypes)
  queue_runner.add_queue_runner(
      queue_runner.QueueRunner(input_queue,
                               [input_queue.enqueue(tensors_to_enqueue)]))
  dequeued_tensors = input_queue.dequeue()
  return {keys[i]: dequeued_tensors[i] for i in range(len(dequeued_tensors))}


def sorted_key_dict(unsorted_dict):
  return {k: unsorted_dict[k] for k in sorted(unsorted_dict)}


def sigmoid(x):
  return 1 / (1 + np.exp(-1.0 * x))


class CheckPartitionerVarHook(session_run_hook.SessionRunHook):
  """A `SessionRunHook` to check a partitioned variable."""

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
          isinstance(partitioned_weight, variables_lib.PartitionedVariable))
      for part in partitioned_weight:
        self._test_case.assertEqual(self._var_dim // self._partitions,
                                    part.get_shape()[0])


class BaseLinearRegressorPartitionerTest(object):

  def __init__(self, linear_regressor_fn, fc_lib=feature_column):
    self._linear_regressor_fn = linear_regressor_fn
    self._fc_lib = fc_lib

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def testPartitioner(self):
    x_dim = 64
    partitions = 4

    def _partitioner(shape, dtype):
      del dtype  # unused; required by Fn signature.
      # Only partition the embedding tensor.
      return [partitions, 1] if shape[0] == x_dim else [1]

    regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.categorical_column_with_hash_bucket(
            'language', hash_bucket_size=x_dim),),
        partitioner=_partitioner,
        model_dir=self._model_dir)

    def _input_fn():
      return {
          'language':
              sparse_tensor.SparseTensor(
                  values=['english', 'spanish'],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2])
      }, [[10.]]

    hook = CheckPartitionerVarHook(self, LANGUAGE_WEIGHT_NAME, x_dim,
                                   partitions)
    regressor.train(input_fn=_input_fn, steps=1, hooks=[hook])

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
    with test.mock.patch.object(
        estimator,
        '_get_replica_device_setter',
        return_value=lambda _: '/cpu:0'):
      linear_regressor = self._linear_regressor_fn(
          feature_columns=(self._fc_lib.categorical_column_with_hash_bucket(
              'language', hash_bucket_size=x_dim),),
          config=FakeRunConfig(),
          model_dir=self._model_dir)

      def _input_fn():
        return {
            'language':
                sparse_tensor.SparseTensor(
                    values=['english', 'spanish'],
                    indices=[[0, 0], [0, 1]],
                    dense_shape=[1, 2])
        }, [[10.]]

      hook = CheckPartitionerVarHook(self, LANGUAGE_WEIGHT_NAME, x_dim,
                                     partitions)
      linear_regressor.train(input_fn=_input_fn, steps=1, hooks=[hook])


# TODO(b/36813849): Add tests with dynamic shape inputs using placeholders.
class BaseLinearRegressorEvaluationTest(object):

  def __init__(self, linear_regressor_fn, fc_lib=feature_column):
    self._linear_regressor_fn = linear_regressor_fn
    self._fc_lib = fc_lib

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def test_evaluation_for_simple_data(self):
    with ops.Graph().as_default():
      variables_lib.Variable([[11.0]], name=AGE_WEIGHT_NAME)
      variables_lib.Variable([2.0], name=BIAS_NAME)
      variables_lib.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        model_dir=self._model_dir)
    eval_metrics = linear_regressor.evaluate(
        input_fn=lambda: ({'age': ((1,),)}, ((10.,),)), steps=1)

    # Logit is (1. * 11.0 + 2.0) = 13, while label is 10. Loss is 3**2 = 9.
    self.assertDictEqual({
        metric_keys.MetricKeys.LOSS: 9.,
        metric_keys.MetricKeys.LOSS_MEAN: 9.,
        metric_keys.MetricKeys.PREDICTION_MEAN: 13.,
        metric_keys.MetricKeys.LABEL_MEAN: 10.,
        ops.GraphKeys.GLOBAL_STEP: 100
    }, eval_metrics)

  def test_evaluation_batch(self):
    """Tests evaluation for batch_size==2."""
    with ops.Graph().as_default():
      variables_lib.Variable([[11.0]], name=AGE_WEIGHT_NAME)
      variables_lib.Variable([2.0], name=BIAS_NAME)
      variables_lib.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
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
        metric_keys.MetricKeys.PREDICTION_MEAN: 13.,
        metric_keys.MetricKeys.LABEL_MEAN: 10.,
        ops.GraphKeys.GLOBAL_STEP: 100
    }, eval_metrics)

  def test_evaluation_weights(self):
    """Tests evaluation with weights."""
    with ops.Graph().as_default():
      variables_lib.Variable([[11.0]], name=AGE_WEIGHT_NAME)
      variables_lib.Variable([2.0], name=BIAS_NAME)
      variables_lib.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    def _input_fn():
      features = {'age': ((1,), (1,)), 'weights': ((1.,), (2.,))}
      labels = ((10.,), (10.,))
      return features, labels

    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        weight_column='weights',
        model_dir=self._model_dir)
    eval_metrics = linear_regressor.evaluate(input_fn=_input_fn, steps=1)

    # Logit is (1. * 11.0 + 2.0) = 13, while label is 10.
    # Loss per example is 3**2 = 9.
    # Training loss is the weighted sum over batch = 9 + 2*9 = 27
    # average loss is the weighted average = 9 + 2*9 / (1 + 2) = 9
    self.assertDictEqual({
        metric_keys.MetricKeys.LOSS: 27.,
        metric_keys.MetricKeys.LOSS_MEAN: 9.,
        metric_keys.MetricKeys.PREDICTION_MEAN: 13.,
        metric_keys.MetricKeys.LABEL_MEAN: 10.,
        ops.GraphKeys.GLOBAL_STEP: 100
    }, eval_metrics)

  def test_evaluation_for_multi_dimensions(self):
    x_dim = 3
    label_dim = 2
    with ops.Graph().as_default():
      variables_lib.Variable(
          [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name=AGE_WEIGHT_NAME)
      variables_lib.Variable([7.0, 8.0], name=BIAS_NAME)
      variables_lib.Variable(100, name='global_step', dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('age', shape=(x_dim,)),),
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
    eval_metrics = linear_regressor.evaluate(input_fn=input_fn, steps=1)

    self.assertItemsEqual(
        (metric_keys.MetricKeys.LOSS, metric_keys.MetricKeys.LOSS_MEAN,
         metric_keys.MetricKeys.PREDICTION_MEAN,
         metric_keys.MetricKeys.LABEL_MEAN, ops.GraphKeys.GLOBAL_STEP),
        eval_metrics.keys())

    # Logit is
    #   [2., 4., 5.] * [1.0, 2.0] + [7.0, 8.0] = [39, 50] + [7.0, 8.0]
    #                  [3.0, 4.0]
    #                  [5.0, 6.0]
    # which is [46, 58]
    self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])

  def test_evaluation_for_multiple_feature_columns(self):
    with ops.Graph().as_default():
      variables_lib.Variable([[10.0]], name=AGE_WEIGHT_NAME)
      variables_lib.Variable([[2.0]], name=HEIGHT_WEIGHT_NAME)
      variables_lib.Variable([5.0], name=BIAS_NAME)
      variables_lib.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    batch_size = 2
    feature_columns = [
        self._fc_lib.numeric_column('age'),
        self._fc_lib.numeric_column('height')
    ]
    input_fn = numpy_io.numpy_input_fn(
        x={'age': np.array([20, 40]),
           'height': np.array([4, 8])},
        y=np.array([[213.], [421.]]),
        batch_size=batch_size,
        num_epochs=None,
        shuffle=False)

    est = self._linear_regressor_fn(
        feature_columns=feature_columns, model_dir=self._model_dir)

    eval_metrics = est.evaluate(input_fn=input_fn, steps=1)
    self.assertItemsEqual(
        (metric_keys.MetricKeys.LOSS, metric_keys.MetricKeys.LOSS_MEAN,
         metric_keys.MetricKeys.PREDICTION_MEAN,
         metric_keys.MetricKeys.LABEL_MEAN, ops.GraphKeys.GLOBAL_STEP),
        eval_metrics.keys())

    # Logit is [(20. * 10.0 + 4 * 2.0 + 5.0), (40. * 10.0 + 8 * 2.0 + 5.0)] =
    # [213.0, 421.0], while label is [213., 421.]. Loss = 0.
    self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])


class BaseLinearRegressorPredictTest(object):

  def __init__(self, linear_regressor_fn, fc_lib=feature_column):
    self._linear_regressor_fn = linear_regressor_fn
    self._fc_lib = fc_lib

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def test_1d(self):
    """Tests predict when all variables are one-dimensional."""
    with ops.Graph().as_default():
      variables_lib.Variable([[10.]], name='linear/linear_model/x/weights')
      variables_lib.Variable([.2], name=BIAS_NAME)
      variables_lib.Variable(100, name='global_step', dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('x'),),
        model_dir=self._model_dir)

    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array([[2.]])},
        y=None,
        batch_size=1,
        num_epochs=1,
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
    feature_columns = (self._fc_lib.numeric_column('x', shape=(x_dim,)),)
    with ops.Graph().as_default():
      variables_lib.Variable(  # shape=[x_dim, label_dimension]
          [[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.]],
          name='linear/linear_model/x/weights')
      variables_lib.Variable(  # shape=[label_dimension]
          [.2, .4, .6], name=BIAS_NAME)
      variables_lib.Variable(100, name='global_step', dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    linear_regressor = self._linear_regressor_fn(
        feature_columns=feature_columns,
        label_dimension=label_dimension,
        model_dir=self._model_dir)

    predict_input_fn = numpy_io.numpy_input_fn(
        # x shape=[batch_size, x_dim]
        x={'x': np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])},
        y=None,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)
    predictions = linear_regressor.predict(input_fn=predict_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    # score = x * weight + bias, shape=[batch_size, label_dimension]
    self.assertAllClose([[30.2, 40.4, 50.6], [70.2, 96.4, 122.6]],
                        predicted_scores)

  def testTwoFeatureColumns(self):
    """Tests predict with two feature columns."""
    with ops.Graph().as_default():
      variables_lib.Variable([[10.]], name='linear/linear_model/x0/weights')
      variables_lib.Variable([[20.]], name='linear/linear_model/x1/weights')
      variables_lib.Variable([.2], name=BIAS_NAME)
      variables_lib.Variable(100, name='global_step', dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('x0'),
                         self._fc_lib.numeric_column('x1')),
        model_dir=self._model_dir)

    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x0': np.array([[2.]]),
           'x1': np.array([[3.]])},
        y=None,
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    predictions = linear_regressor.predict(input_fn=predict_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    # x0 * weight0 + x1 * weight1 + bias = 2. * 10. + 3. * 20 + .2 = 80.2
    self.assertAllClose([[80.2]], predicted_scores)

  def testSparseCombiner(self):
    w_a = 2.0
    w_b = 3.0
    w_c = 5.0
    bias = 5.0
    with ops.Graph().as_default():
      variables_lib.Variable([[w_a], [w_b], [w_c]], name=LANGUAGE_WEIGHT_NAME)
      variables_lib.Variable([bias], name=BIAS_NAME)
      variables_lib.Variable(1, name=ops.GraphKeys.GLOBAL_STEP,
                             dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    def _input_fn():
      return dataset_ops.Dataset.from_tensors({
          'language': sparse_tensor.SparseTensor(
              values=['a', 'c', 'b', 'c'],
              indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
              dense_shape=[2, 2]),
      })

    feature_columns = (self._fc_lib.categorical_column_with_vocabulary_list(
        'language', vocabulary_list=['a', 'b', 'c']),)

    # Check prediction for each sparse_combiner.
    # With sparse_combiner = 'sum', we have
    # logits_1 = w_a + w_c + bias
    #          = 2.0 + 5.0 + 5.0 = 12.0
    # logits_2 = w_b + w_c + bias
    #          = 3.0 + 5.0 + 5.0 = 13.0
    linear_regressor = self._linear_regressor_fn(
        feature_columns=feature_columns,
        model_dir=self._model_dir)
    predictions = linear_regressor.predict(input_fn=_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    self.assertAllClose([[12.0], [13.0]], predicted_scores)

    # With sparse_combiner = 'mean', we have
    # logits_1 = 1/2 * (w_a + w_c) + bias
    #          = 1/2 * (2.0 + 5.0) + 5.0 = 8.5
    # logits_2 = 1/2 * (w_b + w_c) + bias
    #          = 1/2 * (3.0 + 5.0) + 5.0 = 9.0
    linear_regressor = self._linear_regressor_fn(
        feature_columns=feature_columns,
        model_dir=self._model_dir,
        sparse_combiner='mean')
    predictions = linear_regressor.predict(input_fn=_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    self.assertAllClose([[8.5], [9.0]], predicted_scores)

    # With sparse_combiner = 'sqrtn', we have
    # logits_1 = sqrt(2)/2 * (w_a + w_c) + bias
    #          = sqrt(2)/2 * (2.0 + 5.0) + 5.0 = 9.94974
    # logits_2 = sqrt(2)/2 * (w_b + w_c) + bias
    #          = sqrt(2)/2 * (3.0 + 5.0) + 5.0 = 10.65685
    linear_regressor = self._linear_regressor_fn(
        feature_columns=feature_columns,
        model_dir=self._model_dir,
        sparse_combiner='sqrtn')
    predictions = linear_regressor.predict(input_fn=_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    self.assertAllClose([[9.94974], [10.65685]], predicted_scores)


class BaseLinearRegressorIntegrationTest(object):

  def __init__(self, linear_regressor_fn, fc_lib=feature_column):
    self._linear_regressor_fn = linear_regressor_fn
    self._fc_lib = fc_lib

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(self, train_input_fn, eval_input_fn, predict_input_fn,
                          input_dimension, label_dimension, prediction_length):
    feature_columns = [
        self._fc_lib.numeric_column('x', shape=(input_dimension,))
    ]
    est = self._linear_regressor_fn(
        feature_columns=feature_columns,
        label_dimension=label_dimension,
        model_dir=self._model_dir)

    # TRAIN
    # learn y = x
    est.train(train_input_fn, steps=200)

    # EVALUTE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(200, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn(metric_keys.MetricKeys.LOSS, six.iterkeys(scores))

    # PREDICT
    predictions = np.array(
        [x['predictions'] for x in est.predict(predict_input_fn)])
    self.assertAllEqual((prediction_length, label_dimension), predictions.shape)

    # EXPORT
    feature_spec = self._fc_lib.make_parse_example_spec(feature_columns)
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
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=None,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=input_dimension,
        label_dimension=label_dimension,
        prediction_length=prediction_length)

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
        x=x, y=y, batch_size=batch_size, num_epochs=None, shuffle=True)
    eval_input_fn = pandas_io.pandas_input_fn(
        x=x, y=y, batch_size=batch_size, shuffle=False)
    predict_input_fn = pandas_io.pandas_input_fn(
        x=x, batch_size=batch_size, shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=input_dimension,
        label_dimension=label_dimension,
        prediction_length=prediction_length)

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
              'x':
                  feature_pb2.Feature(float_list=feature_pb2.FloatList(
                      value=datum)),
              'y':
                  feature_pb2.Feature(float_list=feature_pb2.FloatList(
                      value=datum[:label_dimension])),
          }))
      serialized_examples.append(example.SerializeToString())

    feature_spec = {
        'x': parsing_ops.FixedLenFeature([input_dimension], dtypes.float32),
        'y': parsing_ops.FixedLenFeature([label_dimension], dtypes.float32),
    }

    def _train_input_fn():
      feature_map = parsing_ops.parse_example(serialized_examples, feature_spec)
      features = queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels

    def _eval_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels

    def _predict_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = queue_parsed_features(feature_map)
      features.pop('y')
      return features, None

    self._test_complete_flow(
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        input_dimension=input_dimension,
        label_dimension=label_dimension,
        prediction_length=prediction_length)


class BaseLinearRegressorTrainingTest(object):

  def __init__(self, linear_regressor_fn, fc_lib=feature_column):
    self._linear_regressor_fn = linear_regressor_fn
    self._fc_lib = fc_lib

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _mock_optimizer(self, expected_loss=None):
    expected_var_names = [
        '%s/part_0:0' % AGE_WEIGHT_NAME,
        '%s/part_0:0' % BIAS_NAME
    ]

    def _minimize(loss, global_step=None, var_list=None):
      trainable_vars = var_list or ops.get_collection(
          ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertItemsEqual(expected_var_names,
                            [var.name for var in trainable_vars])

      # Verify loss. We can't check the value directly, so we add an assert op.
      self.assertEquals(0, loss.shape.ndims)
      if expected_loss is None:
        if global_step is not None:
          return state_ops.assign_add(global_step, 1).op
        return control_flow_ops.no_op()
      assert_loss = assert_close(
          math_ops.to_float(expected_loss, name='expected'),
          loss,
          name='assert_loss')
      with ops.control_dependencies((assert_loss,)):
        if global_step is not None:
          return state_ops.assign_add(global_step, 1).op
        return control_flow_ops.no_op()

    mock_optimizer = test.mock.NonCallableMock(
        spec=optimizer_lib.Optimizer,
        wraps=optimizer_lib.Optimizer(use_locking=False, name='my_optimizer'))
    mock_optimizer.minimize = test.mock.MagicMock(wraps=_minimize)

    # NOTE: Estimator.params performs a deepcopy, which wreaks havoc with mocks.
    # So, return mock_optimizer itself for deepcopy.
    mock_optimizer.__deepcopy__ = lambda _: mock_optimizer
    return mock_optimizer

  def _assert_checkpoint(self,
                         expected_global_step,
                         expected_age_weight=None,
                         expected_bias=None):
    shapes = {
        name: shape
        for (name, shape) in checkpoint_utils.list_variables(self._model_dir)
    }

    self.assertEqual([], shapes[ops.GraphKeys.GLOBAL_STEP])
    self.assertEqual(expected_global_step,
                     checkpoint_utils.load_variable(self._model_dir,
                                                    ops.GraphKeys.GLOBAL_STEP))

    self.assertEqual([1, 1], shapes[AGE_WEIGHT_NAME])
    if expected_age_weight is not None:
      self.assertEqual(expected_age_weight,
                       checkpoint_utils.load_variable(self._model_dir,
                                                      AGE_WEIGHT_NAME))

    self.assertEqual([1], shapes[BIAS_NAME])
    if expected_bias is not None:
      self.assertEqual(expected_bias,
                       checkpoint_utils.load_variable(self._model_dir,
                                                      BIAS_NAME))

  def testFromScratchWithDefaultOptimizer(self):
    # Create LinearRegressor.
    label = 5.
    age = 17
    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        model_dir=self._model_dir)

    # Train for a few steps, and validate final checkpoint.
    num_steps = 10
    linear_regressor.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self._assert_checkpoint(num_steps)

  def testTrainWithOneDimLabel(self):
    label_dimension = 1
    batch_size = 20
    feature_columns = [self._fc_lib.numeric_column('age', shape=(1,))]
    est = self._linear_regressor_fn(
        feature_columns=feature_columns,
        label_dimension=label_dimension,
        model_dir=self._model_dir)
    data_rank_1 = np.linspace(0., 2., batch_size, dtype=np.float32)
    self.assertEqual((batch_size,), data_rank_1.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1},
        y=data_rank_1,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(200)

  def testTrainWithOneDimWeight(self):
    label_dimension = 1
    batch_size = 20
    feature_columns = [self._fc_lib.numeric_column('age', shape=(1,))]
    est = self._linear_regressor_fn(
        feature_columns=feature_columns,
        label_dimension=label_dimension,
        weight_column='w',
        model_dir=self._model_dir)

    data_rank_1 = np.linspace(0., 2., batch_size, dtype=np.float32)
    self.assertEqual((batch_size,), data_rank_1.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1,
           'w': data_rank_1},
        y=data_rank_1,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(200)

  def testFromScratch(self):
    # Create LinearRegressor.
    label = 5.
    age = 17
    # loss = (logits - label)^2 = (0 - 5.)^2 = 25.
    mock_optimizer = self._mock_optimizer(expected_loss=25.)
    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        model_dir=self._model_dir,
        optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    linear_regressor.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        expected_global_step=num_steps,
        expected_age_weight=0.,
        expected_bias=0.)

  def testFromCheckpoint(self):
    # Create initial checkpoint.
    age_weight = 10.0
    bias = 5.0
    initial_global_step = 100
    with ops.Graph().as_default():
      variables_lib.Variable([[age_weight]], name=AGE_WEIGHT_NAME)
      variables_lib.Variable([bias], name=BIAS_NAME)
      variables_lib.Variable(
          initial_global_step,
          name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    # logits = age * age_weight + bias = 17 * 10. + 5. = 175
    # loss = (logits - label)^2 = (175 - 5)^2 = 28900
    mock_optimizer = self._mock_optimizer(expected_loss=28900.)
    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        model_dir=self._model_dir,
        optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    linear_regressor.train(
        input_fn=lambda: ({'age': ((17,),)}, ((5.,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        expected_global_step=initial_global_step + num_steps,
        expected_age_weight=age_weight,
        expected_bias=bias)

  def testFromCheckpointMultiBatch(self):
    # Create initial checkpoint.
    age_weight = 10.0
    bias = 5.0
    initial_global_step = 100
    with ops.Graph().as_default():
      variables_lib.Variable([[age_weight]], name=AGE_WEIGHT_NAME)
      variables_lib.Variable([bias], name=BIAS_NAME)
      variables_lib.Variable(
          initial_global_step,
          name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    # logits = age * age_weight + bias
    # logits[0] = 17 * 10. + 5. = 175
    # logits[1] = 15 * 10. + 5. = 155
    # loss = sum(logits - label)^2 = (175 - 5)^2 + (155 - 3)^2 = 52004
    mock_optimizer = self._mock_optimizer(expected_loss=52004.)
    linear_regressor = self._linear_regressor_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        model_dir=self._model_dir,
        optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    linear_regressor.train(
        input_fn=lambda: ({'age': ((17,), (15,))}, ((5.,), (3.,))),
        steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        expected_global_step=initial_global_step + num_steps,
        expected_age_weight=age_weight,
        expected_bias=bias)


class BaseLinearClassifierTrainingTest(object):

  def __init__(self, linear_classifier_fn, fc_lib=feature_column):
    self._linear_classifier_fn = linear_classifier_fn
    self._fc_lib = fc_lib

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _mock_optimizer(self, expected_loss=None):
    expected_var_names = [
        '%s/part_0:0' % AGE_WEIGHT_NAME,
        '%s/part_0:0' % BIAS_NAME
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
      assert_loss = assert_close(
          math_ops.to_float(expected_loss, name='expected'),
          loss,
          name='assert_loss')
      with ops.control_dependencies((assert_loss,)):
        return state_ops.assign_add(global_step, 1).op

    mock_optimizer = test.mock.NonCallableMock(
        spec=optimizer_lib.Optimizer,
        wraps=optimizer_lib.Optimizer(use_locking=False, name='my_optimizer'))
    mock_optimizer.minimize = test.mock.MagicMock(wraps=_minimize)

    # NOTE: Estimator.params performs a deepcopy, which wreaks havoc with mocks.
    # So, return mock_optimizer itself for deepcopy.
    mock_optimizer.__deepcopy__ = lambda _: mock_optimizer
    return mock_optimizer

  def _assert_checkpoint(
      self, n_classes, expected_global_step, expected_age_weight=None,
      expected_bias=None):
    logits_dimension = n_classes if n_classes > 2 else 1

    shapes = {
        name: shape for (name, shape) in
        checkpoint_utils.list_variables(self._model_dir)
    }

    self.assertEqual([], shapes[ops.GraphKeys.GLOBAL_STEP])
    self.assertEqual(
        expected_global_step,
        checkpoint_utils.load_variable(
            self._model_dir, ops.GraphKeys.GLOBAL_STEP))

    self.assertEqual([1, logits_dimension],
                     shapes[AGE_WEIGHT_NAME])
    if expected_age_weight is not None:
      self.assertAllEqual(expected_age_weight,
                          checkpoint_utils.load_variable(
                              self._model_dir,
                              AGE_WEIGHT_NAME))

    self.assertEqual([logits_dimension], shapes[BIAS_NAME])
    if expected_bias is not None:
      self.assertAllEqual(expected_bias,
                          checkpoint_utils.load_variable(
                              self._model_dir, BIAS_NAME))

  def _testFromScratchWithDefaultOptimizer(self, n_classes):
    label = 0
    age = 17
    est = linear.LinearClassifier(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        model_dir=self._model_dir)

    # Train for a few steps, and validate final checkpoint.
    num_steps = 10
    est.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self._assert_checkpoint(n_classes, num_steps)

  def testBinaryClassesFromScratchWithDefaultOptimizer(self):
    self._testFromScratchWithDefaultOptimizer(n_classes=2)

  def testMultiClassesFromScratchWithDefaultOptimizer(self):
    self._testFromScratchWithDefaultOptimizer(n_classes=4)

  def _testTrainWithTwoDimsLabel(self, n_classes):
    batch_size = 20

    est = linear.LinearClassifier(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        model_dir=self._model_dir)
    data_rank_1 = np.array([0, 1])
    data_rank_2 = np.array([[0], [1]])
    self.assertEqual((2,), data_rank_1.shape)
    self.assertEqual((2, 1), data_rank_2.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1},
        y=data_rank_2,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(n_classes, 200)

  def testBinaryClassesTrainWithTwoDimsLabel(self):
    self._testTrainWithTwoDimsLabel(n_classes=2)

  def testMultiClassesTrainWithTwoDimsLabel(self):
    self._testTrainWithTwoDimsLabel(n_classes=4)

  def _testTrainWithOneDimLabel(self, n_classes):
    batch_size = 20

    est = linear.LinearClassifier(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        model_dir=self._model_dir)
    data_rank_1 = np.array([0, 1])
    self.assertEqual((2,), data_rank_1.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1},
        y=data_rank_1,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(n_classes, 200)

  def testBinaryClassesTrainWithOneDimLabel(self):
    self._testTrainWithOneDimLabel(n_classes=2)

  def testMultiClassesTrainWithOneDimLabel(self):
    self._testTrainWithOneDimLabel(n_classes=4)

  def _testTrainWithTwoDimsWeight(self, n_classes):
    batch_size = 20

    est = linear.LinearClassifier(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        weight_column='w',
        n_classes=n_classes,
        model_dir=self._model_dir)
    data_rank_1 = np.array([0, 1])
    data_rank_2 = np.array([[0], [1]])
    self.assertEqual((2,), data_rank_1.shape)
    self.assertEqual((2, 1), data_rank_2.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1, 'w': data_rank_2}, y=data_rank_1,
        batch_size=batch_size, num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(n_classes, 200)

  def testBinaryClassesTrainWithTwoDimsWeight(self):
    self._testTrainWithTwoDimsWeight(n_classes=2)

  def testMultiClassesTrainWithTwoDimsWeight(self):
    self._testTrainWithTwoDimsWeight(n_classes=4)

  def _testTrainWithOneDimWeight(self, n_classes):
    batch_size = 20

    est = linear.LinearClassifier(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        weight_column='w',
        n_classes=n_classes,
        model_dir=self._model_dir)
    data_rank_1 = np.array([0, 1])
    self.assertEqual((2,), data_rank_1.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1, 'w': data_rank_1}, y=data_rank_1,
        batch_size=batch_size, num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(n_classes, 200)

  def testBinaryClassesTrainWithOneDimWeight(self):
    self._testTrainWithOneDimWeight(n_classes=2)

  def testMultiClassesTrainWithOneDimWeight(self):
    self._testTrainWithOneDimWeight(n_classes=4)

  def _testFromScratch(self, n_classes):
    label = 1
    age = 17
    # For binary classifier:
    #   loss = sigmoid_cross_entropy(logits, label) where logits=0 (weights are
    #   all zero initially) and label = 1 so,
    #      loss = 1 * -log ( sigmoid(logits) ) = 0.69315
    # For multi class classifier:
    #   loss = cross_entropy(logits, label) where logits are all 0s (weights are
    #   all zero initially) and label = 1 so,
    #      loss = 1 * -log ( 1.0 / n_classes )
    # For this particular test case, as logits are same, the formular
    # 1 * -log ( 1.0 / n_classes ) covers both binary and multi class cases.
    mock_optimizer = self._mock_optimizer(
        expected_loss=-1 * math.log(1.0/n_classes))

    est = linear.LinearClassifier(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    est.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        n_classes,
        expected_global_step=num_steps,
        expected_age_weight=[[0.]] if n_classes == 2 else [[0.] * n_classes],
        expected_bias=[0.] if n_classes == 2 else [.0] * n_classes)

  def testBinaryClassesFromScratch(self):
    self._testFromScratch(n_classes=2)

  def testMultiClassesFromScratch(self):
    self._testFromScratch(n_classes=4)

  def _testFromCheckpoint(self, n_classes):
    # Create initial checkpoint.
    label = 1
    age = 17
    # For binary case, the expected weight has shape (1,1). For multi class
    # case, the shape is (1, n_classes). In order to test the weights, set
    # weights as 2.0 * range(n_classes).
    age_weight = [[2.0]] if n_classes == 2 else (
        np.reshape(2.0 * np.array(list(range(n_classes)), dtype=np.float32),
                   (1, n_classes)))
    bias = [-35.0] if n_classes == 2 else [-35.0] * n_classes
    initial_global_step = 100
    with ops.Graph().as_default():
      variables_lib.Variable(age_weight, name=AGE_WEIGHT_NAME)
      variables_lib.Variable(bias, name=BIAS_NAME)
      variables_lib.Variable(
          initial_global_step,
          name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    # For binary classifier:
    #   logits = age * age_weight + bias = 17 * 2. - 35. = -1.
    #   loss = sigmoid_cross_entropy(logits, label)
    #   so, loss = 1 * -log ( sigmoid(-1) ) = 1.3133
    # For multi class classifier:
    #   loss = cross_entropy(logits, label)
    #   where logits = 17 * age_weight + bias and label = 1
    #   so, loss = 1 * -log ( soft_max(logits)[1] )
    if n_classes == 2:
      expected_loss = 1.3133
    else:
      logits = age_weight * age + bias
      logits_exp = np.exp(logits)
      softmax = logits_exp / logits_exp.sum()
      expected_loss = -1 * math.log(softmax[0, label])

    mock_optimizer = self._mock_optimizer(expected_loss=expected_loss)

    est = linear.LinearClassifier(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    est.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        n_classes,
        expected_global_step=initial_global_step + num_steps,
        expected_age_weight=age_weight,
        expected_bias=bias)

  def testBinaryClassesFromCheckpoint(self):
    self._testFromCheckpoint(n_classes=2)

  def testMultiClassesFromCheckpoint(self):
    self._testFromCheckpoint(n_classes=4)

  def _testFromCheckpointFloatLabels(self, n_classes):
    """Tests float labels for binary classification."""
    # Create initial checkpoint.
    if n_classes > 2:
      return
    label = 0.8
    age = 17
    age_weight = [[2.0]]
    bias = [-35.0]
    initial_global_step = 100
    with ops.Graph().as_default():
      variables_lib.Variable(age_weight, name=AGE_WEIGHT_NAME)
      variables_lib.Variable(bias, name=BIAS_NAME)
      variables_lib.Variable(
          initial_global_step,
          name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    # logits = age * age_weight + bias = 17 * 2. - 35. = -1.
    # loss = sigmoid_cross_entropy(logits, label)
    # => loss = -0.8 * log(sigmoid(-1)) -0.2 * log(sigmoid(+1)) = 1.1132617
    mock_optimizer = self._mock_optimizer(expected_loss=1.1132617)

    est = linear.LinearClassifier(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    est.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)

  def testBinaryClassesFromCheckpointFloatLabels(self):
    self._testFromCheckpointFloatLabels(n_classes=2)

  def testMultiClassesFromCheckpointFloatLabels(self):
    self._testFromCheckpointFloatLabels(n_classes=4)

  def _testFromCheckpointMultiBatch(self, n_classes):
    # Create initial checkpoint.
    label = [1, 0]
    age = [17, 18.5]
    # For binary case, the expected weight has shape (1,1). For multi class
    # case, the shape is (1, n_classes). In order to test the weights, set
    # weights as 2.0 * range(n_classes).
    age_weight = [[2.0]] if n_classes == 2 else (
        np.reshape(2.0 * np.array(list(range(n_classes)), dtype=np.float32),
                   (1, n_classes)))
    bias = [-35.0] if n_classes == 2 else [-35.0] * n_classes
    initial_global_step = 100
    with ops.Graph().as_default():
      variables_lib.Variable(age_weight, name=AGE_WEIGHT_NAME)
      variables_lib.Variable(bias, name=BIAS_NAME)
      variables_lib.Variable(
          initial_global_step,
          name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    # For binary classifier:
    #   logits = age * age_weight + bias
    #   logits[0] = 17 * 2. - 35. = -1.
    #   logits[1] = 18.5 * 2. - 35. = 2.
    #   loss = sigmoid_cross_entropy(logits, label)
    #   so, loss[0] = 1 * -log ( sigmoid(-1) ) = 1.3133
    #       loss[1] = (1 - 0) * -log ( 1- sigmoid(2) ) = 2.1269
    # For multi class classifier:
    #   loss = cross_entropy(logits, label)
    #   where logits = [17, 18.5] * age_weight + bias and label = [1, 0]
    #   so, loss = 1 * -log ( soft_max(logits)[label] )
    if n_classes == 2:
      expected_loss = (1.3133 + 2.1269)
    else:
      logits = age_weight * np.reshape(age, (2, 1)) + bias
      logits_exp = np.exp(logits)
      softmax_row_0 = logits_exp[0] / logits_exp[0].sum()
      softmax_row_1 = logits_exp[1] / logits_exp[1].sum()
      expected_loss_0 = -1 * math.log(softmax_row_0[label[0]])
      expected_loss_1 = -1 * math.log(softmax_row_1[label[1]])
      expected_loss = expected_loss_0 + expected_loss_1

    mock_optimizer = self._mock_optimizer(expected_loss=expected_loss)

    est = linear.LinearClassifier(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    est.train(
        input_fn=lambda: ({'age': (age)}, (label)),
        steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        n_classes,
        expected_global_step=initial_global_step + num_steps,
        expected_age_weight=age_weight,
        expected_bias=bias)

  def testBinaryClassesFromCheckpointMultiBatch(self):
    self._testFromCheckpointMultiBatch(n_classes=2)

  def testMultiClassesFromCheckpointMultiBatch(self):
    self._testFromCheckpointMultiBatch(n_classes=4)


class BaseLinearClassifierEvaluationTest(object):

  def __init__(self, linear_classifier_fn, fc_lib=feature_column):
    self._linear_classifier_fn = linear_classifier_fn
    self._fc_lib = fc_lib

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _test_evaluation_for_simple_data(self, n_classes):
    label = 1
    age = 1.

    # For binary case, the expected weight has shape (1,1). For multi class
    # case, the shape is (1, n_classes). In order to test the weights, set
    # weights as 2.0 * range(n_classes).
    age_weight = [[-11.0]] if n_classes == 2 else (
        np.reshape(-11.0 * np.array(list(range(n_classes)), dtype=np.float32),
                   (1, n_classes)))
    bias = [-30.0] if n_classes == 2 else [-30.0] * n_classes

    with ops.Graph().as_default():
      variables_lib.Variable(age_weight, name=AGE_WEIGHT_NAME)
      variables_lib.Variable(bias, name=BIAS_NAME)
      variables_lib.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    est = self._linear_classifier_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        model_dir=self._model_dir)
    eval_metrics = est.evaluate(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=1)

    if n_classes == 2:
      # Binary classes: loss = sum(corss_entropy(41)) = 41.
      expected_metrics = {
          metric_keys.MetricKeys.LOSS: 41.,
          ops.GraphKeys.GLOBAL_STEP: 100,
          metric_keys.MetricKeys.LOSS_MEAN: 41.,
          metric_keys.MetricKeys.ACCURACY: 0.,
          metric_keys.MetricKeys.PRECISION: 0.,
          metric_keys.MetricKeys.RECALL: 0.,
          metric_keys.MetricKeys.PREDICTION_MEAN: 0.,
          metric_keys.MetricKeys.LABEL_MEAN: 1.,
          metric_keys.MetricKeys.ACCURACY_BASELINE: 1,
          metric_keys.MetricKeys.AUC: 0.,
          metric_keys.MetricKeys.AUC_PR: 1.,
      }
    else:
      # Multi classes: loss = 1 * -log ( soft_max(logits)[label] )
      logits = age_weight * age + bias
      logits_exp = np.exp(logits)
      softmax = logits_exp / logits_exp.sum()
      expected_loss = -1 * math.log(softmax[0, label])

      expected_metrics = {
          metric_keys.MetricKeys.LOSS: expected_loss,
          ops.GraphKeys.GLOBAL_STEP: 100,
          metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
          metric_keys.MetricKeys.ACCURACY: 0.,
      }

    self.assertAllClose(sorted_key_dict(expected_metrics),
                        sorted_key_dict(eval_metrics), rtol=1e-3)

  def test_binary_classes_evaluation_for_simple_data(self):
    self._test_evaluation_for_simple_data(n_classes=2)

  def test_multi_classes_evaluation_for_simple_data(self):
    self._test_evaluation_for_simple_data(n_classes=4)

  def _test_evaluation_batch(self, n_classes):
    """Tests evaluation for batch_size==2."""
    label = [1, 0]
    age = [17., 18.]
    # For binary case, the expected weight has shape (1,1). For multi class
    # case, the shape is (1, n_classes). In order to test the weights, set
    # weights as 2.0 * range(n_classes).
    age_weight = [[2.0]] if n_classes == 2 else (
        np.reshape(2.0 * np.array(list(range(n_classes)), dtype=np.float32),
                   (1, n_classes)))
    bias = [-35.0] if n_classes == 2 else [-35.0] * n_classes
    initial_global_step = 100
    with ops.Graph().as_default():
      variables_lib.Variable(age_weight, name=AGE_WEIGHT_NAME)
      variables_lib.Variable(bias, name=BIAS_NAME)
      variables_lib.Variable(
          initial_global_step,
          name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    est = self._linear_classifier_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        model_dir=self._model_dir)
    eval_metrics = est.evaluate(
        input_fn=lambda: ({'age': (age)}, (label)), steps=1)

    if n_classes == 2:
      # Logits are (-1., 1.) labels are (1, 0).
      # Loss is
      #   loss for row 1: 1 * -log(sigmoid(-1)) = 1.3133
      #   loss for row 2: (1 - 0) * -log(1 - sigmoid(1)) = 1.3133
      expected_loss = 1.3133 * 2

      expected_metrics = {
          metric_keys.MetricKeys.LOSS: expected_loss,
          ops.GraphKeys.GLOBAL_STEP: 100,
          metric_keys.MetricKeys.LOSS_MEAN: expected_loss / 2,
          metric_keys.MetricKeys.ACCURACY: 0.,
          metric_keys.MetricKeys.PRECISION: 0.,
          metric_keys.MetricKeys.RECALL: 0.,
          metric_keys.MetricKeys.PREDICTION_MEAN: 0.5,
          metric_keys.MetricKeys.LABEL_MEAN: 0.5,
          metric_keys.MetricKeys.ACCURACY_BASELINE: 0.5,
          metric_keys.MetricKeys.AUC: 0.,
          metric_keys.MetricKeys.AUC_PR: 0.25,
      }
    else:
      # Multi classes: loss = 1 * -log ( soft_max(logits)[label] )
      logits = age_weight * np.reshape(age, (2, 1)) + bias
      logits_exp = np.exp(logits)
      softmax_row_0 = logits_exp[0] / logits_exp[0].sum()
      softmax_row_1 = logits_exp[1] / logits_exp[1].sum()
      expected_loss_0 = -1 * math.log(softmax_row_0[label[0]])
      expected_loss_1 = -1 * math.log(softmax_row_1[label[1]])
      expected_loss = expected_loss_0 + expected_loss_1

      expected_metrics = {
          metric_keys.MetricKeys.LOSS: expected_loss,
          ops.GraphKeys.GLOBAL_STEP: 100,
          metric_keys.MetricKeys.LOSS_MEAN: expected_loss / 2,
          metric_keys.MetricKeys.ACCURACY: 0.,
      }

    self.assertAllClose(sorted_key_dict(expected_metrics),
                        sorted_key_dict(eval_metrics), rtol=1e-3)

  def test_binary_classes_evaluation_batch(self):
    self._test_evaluation_batch(n_classes=2)

  def test_multi_classes_evaluation_batch(self):
    self._test_evaluation_batch(n_classes=4)

  def _test_evaluation_weights(self, n_classes):
    """Tests evaluation with weights."""

    label = [1, 0]
    age = [17., 18.]
    weights = [1., 2.]
    # For binary case, the expected weight has shape (1,1). For multi class
    # case, the shape is (1, n_classes). In order to test the weights, set
    # weights as 2.0 * range(n_classes).
    age_weight = [[2.0]] if n_classes == 2 else (
        np.reshape(2.0 * np.array(list(range(n_classes)), dtype=np.float32),
                   (1, n_classes)))
    bias = [-35.0] if n_classes == 2 else [-35.0] * n_classes
    initial_global_step = 100
    with ops.Graph().as_default():
      variables_lib.Variable(age_weight, name=AGE_WEIGHT_NAME)
      variables_lib.Variable(bias, name=BIAS_NAME)
      variables_lib.Variable(
          initial_global_step,
          name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    est = self._linear_classifier_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        n_classes=n_classes,
        weight_column='w',
        model_dir=self._model_dir)
    eval_metrics = est.evaluate(
        input_fn=lambda: ({'age': (age), 'w': (weights)}, (label)), steps=1)

    if n_classes == 2:
      # Logits are (-1., 1.) labels are (1, 0).
      # Loss is
      #   loss for row 1: 1 * -log(sigmoid(-1)) = 1.3133
      #   loss for row 2: (1 - 0) * -log(1 - sigmoid(1)) = 1.3133
      #   weights = [1., 2.]
      expected_loss = 1.3133 * (1. + 2.)
      loss_mean = expected_loss / (1.0 + 2.0)
      label_mean = np.average(label, weights=weights)
      logits = [-1, 1]
      logistics = sigmoid(np.array(logits))
      predictions_mean = np.average(logistics, weights=weights)

      expected_metrics = {
          metric_keys.MetricKeys.LOSS: expected_loss,
          ops.GraphKeys.GLOBAL_STEP: 100,
          metric_keys.MetricKeys.LOSS_MEAN: loss_mean,
          metric_keys.MetricKeys.ACCURACY: 0.,
          metric_keys.MetricKeys.PRECISION: 0.,
          metric_keys.MetricKeys.RECALL: 0.,
          metric_keys.MetricKeys.PREDICTION_MEAN: predictions_mean,
          metric_keys.MetricKeys.LABEL_MEAN: label_mean,
          metric_keys.MetricKeys.ACCURACY_BASELINE: (
              max(label_mean, 1-label_mean)),
          metric_keys.MetricKeys.AUC: 0.,
          metric_keys.MetricKeys.AUC_PR: 0.1668,
      }
    else:
      # Multi classes: unweighted_loss = 1 * -log ( soft_max(logits)[label] )
      logits = age_weight * np.reshape(age, (2, 1)) + bias
      logits_exp = np.exp(logits)
      softmax_row_0 = logits_exp[0] / logits_exp[0].sum()
      softmax_row_1 = logits_exp[1] / logits_exp[1].sum()
      expected_loss_0 = -1 * math.log(softmax_row_0[label[0]])
      expected_loss_1 = -1 * math.log(softmax_row_1[label[1]])
      loss_mean = np.average([expected_loss_0, expected_loss_1],
                             weights=weights)
      expected_loss = loss_mean * np.sum(weights)

      expected_metrics = {
          metric_keys.MetricKeys.LOSS: expected_loss,
          ops.GraphKeys.GLOBAL_STEP: 100,
          metric_keys.MetricKeys.LOSS_MEAN: loss_mean,
          metric_keys.MetricKeys.ACCURACY: 0.,
      }

    self.assertAllClose(sorted_key_dict(expected_metrics),
                        sorted_key_dict(eval_metrics), rtol=1e-3)

  def test_binary_classes_evaluation_weights(self):
    self._test_evaluation_weights(n_classes=2)

  def test_multi_classes_evaluation_weights(self):
    self._test_evaluation_weights(n_classes=4)


class BaseLinearClassifierPredictTest(object):

  def __init__(self, linear_classifier_fn, fc_lib=feature_column):
    self._linear_classifier_fn = linear_classifier_fn
    self._fc_lib = fc_lib

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _testPredictions(self, n_classes, label_vocabulary, label_output_fn):
    """Tests predict when all variables are one-dimensional."""
    age = 1.

    # For binary case, the expected weight has shape (1,1). For multi class
    # case, the shape is (1, n_classes). In order to test the weights, set
    # weights as 2.0 * range(n_classes).
    age_weight = [[-11.0]] if n_classes == 2 else (
        np.reshape(-11.0 * np.array(list(range(n_classes)), dtype=np.float32),
                   (1, n_classes)))
    bias = [10.0] if n_classes == 2 else [10.0] * n_classes

    with ops.Graph().as_default():
      variables_lib.Variable(age_weight, name=AGE_WEIGHT_NAME)
      variables_lib.Variable(bias, name=BIAS_NAME)
      variables_lib.Variable(100, name='global_step', dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    est = self._linear_classifier_fn(
        feature_columns=(self._fc_lib.numeric_column('age'),),
        label_vocabulary=label_vocabulary,
        n_classes=n_classes,
        model_dir=self._model_dir)

    predict_input_fn = numpy_io.numpy_input_fn(
        x={'age': np.array([[age]])},
        y=None,
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    predictions = list(est.predict(input_fn=predict_input_fn))

    if n_classes == 2:
      scalar_logits = np.asscalar(
          np.reshape(np.array(age_weight) * age + bias, (1,)))
      two_classes_logits = [0, scalar_logits]
      two_classes_logits_exp = np.exp(two_classes_logits)
      softmax = two_classes_logits_exp / two_classes_logits_exp.sum()

      expected_predictions = {
          'class_ids': [0],
          'classes': [label_output_fn(0)],
          'logistic': [sigmoid(np.array(scalar_logits))],
          'logits': [scalar_logits],
          'probabilities': softmax,
      }
    else:
      onedim_logits = np.reshape(np.array(age_weight) * age + bias, (-1,))
      class_ids = onedim_logits.argmax()
      logits_exp = np.exp(onedim_logits)
      softmax = logits_exp / logits_exp.sum()
      expected_predictions = {
          'class_ids': [class_ids],
          'classes': [label_output_fn(class_ids)],
          'logits': onedim_logits,
          'probabilities': softmax,
      }

    self.assertEqual(1, len(predictions))
    # assertAllClose cannot handle byte type.
    self.assertEqual(expected_predictions['classes'], predictions[0]['classes'])
    expected_predictions.pop('classes')
    predictions[0].pop('classes')
    self.assertAllClose(sorted_key_dict(expected_predictions),
                        sorted_key_dict(predictions[0]))

  def testBinaryClassesWithoutLabelVocabulary(self):
    n_classes = 2
    self._testPredictions(n_classes,
                          label_vocabulary=None,
                          label_output_fn=lambda x: ('%s' % x).encode())

  def testBinaryClassesWithLabelVocabulary(self):
    n_classes = 2
    self._testPredictions(
        n_classes,
        label_vocabulary=['class_vocab_{}'.format(i)
                          for i in range(n_classes)],
        label_output_fn=lambda x: ('class_vocab_%s' % x).encode())

  def testMultiClassesWithoutLabelVocabulary(self):
    n_classes = 4
    self._testPredictions(
        n_classes,
        label_vocabulary=None,
        label_output_fn=lambda x: ('%s' % x).encode())

  def testMultiClassesWithLabelVocabulary(self):
    n_classes = 4
    self._testPredictions(
        n_classes,
        label_vocabulary=['class_vocab_{}'.format(i)
                          for i in range(n_classes)],
        label_output_fn=lambda x: ('class_vocab_%s' % x).encode())

  def testSparseCombiner(self):
    w_a = 2.0
    w_b = 3.0
    w_c = 5.0
    bias = 5.0
    with ops.Graph().as_default():
      variables_lib.Variable([[w_a], [w_b], [w_c]], name=LANGUAGE_WEIGHT_NAME)
      variables_lib.Variable([bias], name=BIAS_NAME)
      variables_lib.Variable(1, name=ops.GraphKeys.GLOBAL_STEP,
                             dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    def _input_fn():
      return dataset_ops.Dataset.from_tensors({
          'language': sparse_tensor.SparseTensor(
              values=['a', 'c', 'b', 'c'],
              indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
              dense_shape=[2, 2]),
      })

    feature_columns = (self._fc_lib.categorical_column_with_vocabulary_list(
        'language', vocabulary_list=['a', 'b', 'c']),)

    # Check prediction for each sparse_combiner.
    # With sparse_combiner = 'sum', we have
    # logits_1 = w_a + w_c + bias
    #          = 2.0 + 5.0 + 5.0 = 12.0
    # logits_2 = w_b + w_c + bias
    #          = 3.0 + 5.0 + 5.0 = 13.0
    linear_classifier = self._linear_classifier_fn(
        feature_columns=feature_columns,
        model_dir=self._model_dir)
    predictions = linear_classifier.predict(input_fn=_input_fn)
    predicted_scores = list([x['logits'] for x in predictions])
    self.assertAllClose([[12.0], [13.0]], predicted_scores)

    # With sparse_combiner = 'mean', we have
    # logits_1 = 1/2 * (w_a + w_c) + bias
    #          = 1/2 * (2.0 + 5.0) + 5.0 = 8.5
    # logits_2 = 1/2 * (w_b + w_c) + bias
    #          = 1/2 * (3.0 + 5.0) + 5.0 = 9.0
    linear_classifier = self._linear_classifier_fn(
        feature_columns=feature_columns,
        model_dir=self._model_dir,
        sparse_combiner='mean')
    predictions = linear_classifier.predict(input_fn=_input_fn)
    predicted_scores = list([x['logits'] for x in predictions])
    self.assertAllClose([[8.5], [9.0]], predicted_scores)

    # With sparse_combiner = 'sqrtn', we have
    # logits_1 = sqrt(2)/2 * (w_a + w_c) + bias
    #          = sqrt(2)/2 * (2.0 + 5.0) + 5.0 = 9.94974
    # logits_2 = sqrt(2)/2 * (w_b + w_c) + bias
    #          = sqrt(2)/2 * (3.0 + 5.0) + 5.0 = 10.65685
    linear_classifier = self._linear_classifier_fn(
        feature_columns=feature_columns,
        model_dir=self._model_dir,
        sparse_combiner='sqrtn')
    predictions = linear_classifier.predict(input_fn=_input_fn)
    predicted_scores = list([x['logits'] for x in predictions])
    self.assertAllClose([[9.94974], [10.65685]], predicted_scores)


class BaseLinearClassifierIntegrationTest(object):

  def __init__(self, linear_classifier_fn, fc_lib=feature_column):
    self._linear_classifier_fn = linear_classifier_fn
    self._fc_lib = fc_lib

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(self, n_classes, train_input_fn, eval_input_fn,
                          predict_input_fn, input_dimension, prediction_length):
    feature_columns = [
        self._fc_lib.numeric_column('x', shape=(input_dimension,))
    ]
    est = self._linear_classifier_fn(
        feature_columns=feature_columns,
        n_classes=n_classes,
        model_dir=self._model_dir)

    # TRAIN
    # learn y = x
    est.train(train_input_fn, steps=200)

    # EVALUTE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(200, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn(metric_keys.MetricKeys.LOSS, six.iterkeys(scores))

    # PREDICT
    predictions = np.array(
        [x['classes'] for x in est.predict(predict_input_fn)])
    self.assertAllEqual((prediction_length, 1), predictions.shape)

    # EXPORT
    feature_spec = self._fc_lib.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_savedmodel(tempfile.mkdtemp(),
                                       serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))

  def _test_numpy_input_fn(self, n_classes):
    """Tests complete flow with numpy_input_fn."""
    input_dimension = 4
    batch_size = 10
    prediction_length = batch_size
    data = np.linspace(0., 2., batch_size * input_dimension, dtype=np.float32)
    data = data.reshape(batch_size, input_dimension)
    target = np.array([1] * batch_size)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=target,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=target,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=None,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    self._test_complete_flow(
        n_classes=n_classes,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=input_dimension,
        prediction_length=prediction_length)

  def test_binary_classes_numpy_input_fn(self):
    self._test_numpy_input_fn(n_classes=2)

  def test_multi_classes_numpy_input_fn(self):
    self._test_numpy_input_fn(n_classes=4)

  def _test_pandas_input_fn(self, n_classes):
    """Tests complete flow with pandas_input_fn."""
    if not HAS_PANDAS:
      return

    # Pandas DataFrame natually supports 1 dim data only.
    input_dimension = 1
    batch_size = 10
    data = np.array([1., 2., 3., 4.], dtype=np.float32)
    target = np.array([1, 0, 1, 0], dtype=np.int32)
    x = pd.DataFrame({'x': data})
    y = pd.Series(target)
    prediction_length = 4

    train_input_fn = pandas_io.pandas_input_fn(
        x=x, y=y, batch_size=batch_size, num_epochs=None, shuffle=True)
    eval_input_fn = pandas_io.pandas_input_fn(
        x=x, y=y, batch_size=batch_size, shuffle=False)
    predict_input_fn = pandas_io.pandas_input_fn(
        x=x, batch_size=batch_size, shuffle=False)

    self._test_complete_flow(
        n_classes=n_classes,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=input_dimension,
        prediction_length=prediction_length)

  def test_binary_classes_pandas_input_fn(self):
    self._test_pandas_input_fn(n_classes=2)

  def test_multi_classes_pandas_input_fn(self):
    self._test_pandas_input_fn(n_classes=4)

  def _test_input_fn_from_parse_example(self, n_classes):
    """Tests complete flow with input_fn constructed from parse_example."""
    input_dimension = 2
    batch_size = 10
    prediction_length = batch_size
    data = np.linspace(0., 2., batch_size * input_dimension, dtype=np.float32)
    data = data.reshape(batch_size, input_dimension)
    target = np.array([1] * batch_size, dtype=np.int64)

    serialized_examples = []
    for x, y in zip(data, target):
      example = example_pb2.Example(features=feature_pb2.Features(
          feature={
              'x':
                  feature_pb2.Feature(float_list=feature_pb2.FloatList(
                      value=x)),
              'y':
                  feature_pb2.Feature(int64_list=feature_pb2.Int64List(
                      value=[y])),
          }))
      serialized_examples.append(example.SerializeToString())

    feature_spec = {
        'x': parsing_ops.FixedLenFeature([input_dimension], dtypes.float32),
        'y': parsing_ops.FixedLenFeature([1], dtypes.int64),
    }

    def _train_input_fn():
      feature_map = parsing_ops.parse_example(serialized_examples, feature_spec)
      features = queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels

    def _eval_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels

    def _predict_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = queue_parsed_features(feature_map)
      features.pop('y')
      return features, None

    self._test_complete_flow(
        n_classes=n_classes,
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        input_dimension=input_dimension,
        prediction_length=prediction_length)

  def test_binary_classes_input_fn_from_parse_example(self):
    self._test_input_fn_from_parse_example(n_classes=2)

  def test_multi_classes_input_fn_from_parse_example(self):
    self._test_input_fn_from_parse_example(n_classes=4)


class BaseLinearLogitFnTest(object):

  def __init__(self, fc_lib=feature_column):
    self._fc_lib = fc_lib

  def test_basic_logit_correctness(self):
    """linear_logit_fn simply wraps feature_column_lib.linear_model."""
    age = self._fc_lib.numeric_column('age')
    with ops.Graph().as_default():
      logit_fn = linear._linear_logit_fn_builder(units=2, feature_columns=[age])
      logits = logit_fn(features={'age': [[23.], [31.]]})
      bias_var = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                    'linear_model/bias_weights')[0]
      age_var = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                   'linear_model/age')[0]
      with tf_session.Session() as sess:
        sess.run([variables_lib.global_variables_initializer()])
        self.assertAllClose([[0., 0.], [0., 0.]], logits.eval())
        sess.run(bias_var.assign([10., 5.]))
        self.assertAllClose([[10., 5.], [10., 5.]], logits.eval())
        sess.run(age_var.assign([[2.0, 3.0]]))
        # [2 * 23 + 10, 3 * 23 + 5] = [56, 74].
        # [2 * 31 + 10, 3 * 31 + 5] = [72, 98]
        self.assertAllClose([[56., 74.], [72., 98.]], logits.eval())

  def test_compute_fraction_of_zero(self):
    """Tests the calculation of sparsity."""
    if self._fc_lib != feature_column:
      return
    age = feature_column.numeric_column('age')
    occupation = feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=5)
    with ops.Graph().as_default():
      cols_to_vars = {}
      feature_column.linear_model(
          features={
              'age': [[23.], [31.]],
              'occupation': [['doctor'], ['engineer']]
          },
          feature_columns=[age, occupation],
          units=3,
          cols_to_vars=cols_to_vars)
      cols_to_vars.pop('bias')
      fraction_zero = linear._compute_fraction_of_zero(cols_to_vars.values())
      age_var = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                   'linear_model/age')[0]
      with tf_session.Session() as sess:
        sess.run([variables_lib.global_variables_initializer()])
        # Upon initialization, all variables will be zero.
        self.assertAllClose(1, fraction_zero.eval())

        sess.run(age_var.assign([[2.0, 0.0, -1.0]]))
        # 1 of the 3 age weights are zero, and all of the 15 (5 hash buckets
        # x 3-dim output) are zero.
        self.assertAllClose(16. / 18., fraction_zero.eval())

  def test_compute_fraction_of_zero_v2(self):
    """Tests the calculation of sparsity."""
    if self._fc_lib != feature_column_v2:
      return

    age = feature_column_v2.numeric_column('age')
    occupation = feature_column_v2.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=5)
    shared_state_manager = feature_column_v2.SharedEmbeddingStateManager()
    with ops.Graph().as_default():
      model = feature_column_v2.LinearModel(
          feature_columns=[age, occupation],
          units=3,
          shared_state_manager=shared_state_manager)
      features = {
          'age': [[23.], [31.]],
          'occupation': [['doctor'], ['engineer']]
      }
      model(features)
      variables = model.variables
      variables.remove(model.bias_variable)
      variables.extend(shared_state_manager.variables)
      fraction_zero = linear._compute_fraction_of_zero(variables)
      age_var = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                   'linear_model/age')[0]
      with tf_session.Session() as sess:
        sess.run([variables_lib.global_variables_initializer()])
        # Upon initialization, all variables will be zero.
        self.assertAllClose(1, fraction_zero.eval())

        sess.run(age_var.assign([[2.0, 0.0, -1.0]]))
        # 1 of the 3 age weights are zero, and all of the 15 (5 hash buckets
        # x 3-dim output) are zero.
        self.assertAllClose(16. / 18., fraction_zero.eval())


class BaseLinearWarmStartingTest(object):

  def __init__(self,
               _linear_classifier_fn,
               _linear_regressor_fn,
               fc_lib=feature_column):
    self._linear_classifier_fn = _linear_classifier_fn
    self._linear_regressor_fn = _linear_regressor_fn
    self._fc_lib = fc_lib

  def setUp(self):
    # Create a directory to save our old checkpoint and vocabularies to.
    self._ckpt_and_vocab_dir = tempfile.mkdtemp()

    # Make a dummy input_fn.
    def _input_fn():
      features = {
          'age': [[23.], [31.]],
          'age_in_years': [[23.], [31.]],
          'occupation': [['doctor'], ['consultant']]
      }
      return features, [0, 1]

    self._input_fn = _input_fn

  def tearDown(self):
    # Clean up checkpoint / vocab dir.
    writer_cache.FileWriterCache.clear()
    shutil.rmtree(self._ckpt_and_vocab_dir)

  def test_classifier_basic_warm_starting(self):
    """Tests correctness of LinearClassifier default warm-start."""
    age = self._fc_lib.numeric_column('age')

    # Create a LinearClassifier and train to save a checkpoint.
    linear_classifier = self._linear_classifier_fn(
        feature_columns=[age],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        optimizer='SGD')
    linear_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second LinearClassifier, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).
    warm_started_linear_classifier = self._linear_classifier_fn(
        feature_columns=[age],
        n_classes=4,
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        warm_start_from=linear_classifier.model_dir)

    warm_started_linear_classifier.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_linear_classifier.get_variable_names():
      self.assertAllClose(
          linear_classifier.get_variable_value(variable_name),
          warm_started_linear_classifier.get_variable_value(variable_name))

  def test_regressor_basic_warm_starting(self):
    """Tests correctness of LinearRegressor default warm-start."""
    age = self._fc_lib.numeric_column('age')

    # Create a LinearRegressor and train to save a checkpoint.
    linear_regressor = self._linear_regressor_fn(
        feature_columns=[age],
        model_dir=self._ckpt_and_vocab_dir,
        optimizer='SGD')
    linear_regressor.train(input_fn=self._input_fn, max_steps=1)

    # Create a second LinearRegressor, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).
    warm_started_linear_regressor = self._linear_regressor_fn(
        feature_columns=[age],
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        warm_start_from=linear_regressor.model_dir)

    warm_started_linear_regressor.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_linear_regressor.get_variable_names():
      self.assertAllClose(
          linear_regressor.get_variable_value(variable_name),
          warm_started_linear_regressor.get_variable_value(variable_name))

  def test_warm_starting_selective_variables(self):
    """Tests selecting variables to warm-start."""
    age = self._fc_lib.numeric_column('age')

    # Create a LinearClassifier and train to save a checkpoint.
    linear_classifier = self._linear_classifier_fn(
        feature_columns=[age],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        optimizer='SGD')
    linear_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second LinearClassifier, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).
    warm_started_linear_classifier = self._linear_classifier_fn(
        feature_columns=[age],
        n_classes=4,
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        # The provided regular expression will only warm-start the age variable
        # and not the bias.
        warm_start_from=estimator.WarmStartSettings(
            ckpt_to_initialize_from=linear_classifier.model_dir,
            vars_to_warm_start='.*(age).*'))

    warm_started_linear_classifier.train(input_fn=self._input_fn, max_steps=1)
    self.assertAllClose(
        linear_classifier.get_variable_value(AGE_WEIGHT_NAME),
        warm_started_linear_classifier.get_variable_value(AGE_WEIGHT_NAME))
    # Bias should still be zero from initialization.
    self.assertAllClose(
        [0.0] * 4, warm_started_linear_classifier.get_variable_value(BIAS_NAME))

  def test_warm_starting_with_vocab_remapping_and_partitioning(self):
    """Tests warm-starting with vocab remapping and partitioning."""
    vocab_list = ['doctor', 'lawyer', 'consultant']
    vocab_file = os.path.join(self._ckpt_and_vocab_dir, 'occupation_vocab')
    with open(vocab_file, 'w') as f:
      f.write('\n'.join(vocab_list))
    occupation = self._fc_lib.categorical_column_with_vocabulary_file(
        'occupation',
        vocabulary_file=vocab_file,
        vocabulary_size=len(vocab_list))

    # Create a LinearClassifier and train to save a checkpoint.
    partitioner = partitioned_variables.fixed_size_partitioner(num_shards=2)
    linear_classifier = self._linear_classifier_fn(
        feature_columns=[occupation],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        optimizer='SGD',
        partitioner=partitioner)
    linear_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second LinearClassifier, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).  Use a new FeatureColumn with a
    # different vocabulary for occupation.
    new_vocab_list = ['doctor', 'consultant', 'engineer']
    new_vocab_file = os.path.join(self._ckpt_and_vocab_dir,
                                  'new_occupation_vocab')
    with open(new_vocab_file, 'w') as f:
      f.write('\n'.join(new_vocab_list))
    new_occupation = self._fc_lib.categorical_column_with_vocabulary_file(
        'occupation',
        vocabulary_file=new_vocab_file,
        vocabulary_size=len(new_vocab_list))
    # We can create our VocabInfo object from the new and old occupation
    # FeatureColumn's.
    occupation_vocab_info = estimator.VocabInfo(
        new_vocab=new_occupation.vocabulary_file,
        new_vocab_size=new_occupation.vocabulary_size,
        num_oov_buckets=new_occupation.num_oov_buckets,
        old_vocab=occupation.vocabulary_file,
        old_vocab_size=occupation.vocabulary_size,
        # Can't use constant_initializer with load_and_remap.  In practice,
        # use a truncated normal initializer.
        backup_initializer=init_ops.random_uniform_initializer(
            minval=0.39, maxval=0.39))
    warm_started_linear_classifier = self._linear_classifier_fn(
        feature_columns=[occupation],
        n_classes=4,
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        warm_start_from=estimator.WarmStartSettings(
            ckpt_to_initialize_from=linear_classifier.model_dir,
            var_name_to_vocab_info={
                OCCUPATION_WEIGHT_NAME: occupation_vocab_info
            },
            # Explicitly providing None here will only warm-start variables
            # referenced in var_name_to_vocab_info (the bias will not be
            # warm-started).
            vars_to_warm_start=None),
        partitioner=partitioner)

    warm_started_linear_classifier.train(input_fn=self._input_fn, max_steps=1)
    # 'doctor' was ID-0 and still ID-0.
    self.assertAllClose(
        linear_classifier.get_variable_value(OCCUPATION_WEIGHT_NAME)[0, :],
        warm_started_linear_classifier.get_variable_value(
            OCCUPATION_WEIGHT_NAME)[0, :])
    # 'consultant' was ID-2 and now ID-1.
    self.assertAllClose(
        linear_classifier.get_variable_value(OCCUPATION_WEIGHT_NAME)[2, :],
        warm_started_linear_classifier.get_variable_value(
            OCCUPATION_WEIGHT_NAME)[1, :])
    # 'engineer' is a new entry and should be initialized with the
    # backup_initializer in VocabInfo.
    self.assertAllClose([0.39] * 4,
                        warm_started_linear_classifier.get_variable_value(
                            OCCUPATION_WEIGHT_NAME)[2, :])
    # Bias should still be zero (from initialization logic).
    self.assertAllClose(
        [0.0] * 4, warm_started_linear_classifier.get_variable_value(BIAS_NAME))

  def test_warm_starting_with_naming_change(self):
    """Tests warm-starting with a Tensor name remapping."""
    age_in_years = self._fc_lib.numeric_column('age_in_years')

    # Create a LinearClassifier and train to save a checkpoint.
    linear_classifier = self._linear_classifier_fn(
        feature_columns=[age_in_years],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        optimizer='SGD')
    linear_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second LinearClassifier, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).
    warm_started_linear_classifier = self._linear_classifier_fn(
        feature_columns=[self._fc_lib.numeric_column('age')],
        n_classes=4,
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        # The 'age' variable correspond to the 'age_in_years' variable in the
        # previous model.
        warm_start_from=estimator.WarmStartSettings(
            ckpt_to_initialize_from=linear_classifier.model_dir,
            var_name_to_prev_var_name={
                AGE_WEIGHT_NAME: AGE_WEIGHT_NAME.replace('age', 'age_in_years')
            }))

    warm_started_linear_classifier.train(input_fn=self._input_fn, max_steps=1)
    self.assertAllClose(
        linear_classifier.get_variable_value(
            AGE_WEIGHT_NAME.replace('age', 'age_in_years')),
        warm_started_linear_classifier.get_variable_value(AGE_WEIGHT_NAME))
    # The bias is also warm-started (with no name remapping).
    self.assertAllClose(
        linear_classifier.get_variable_value(BIAS_NAME),
        warm_started_linear_classifier.get_variable_value(BIAS_NAME))
