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
"""Tests for DNNLinearCombinedEstimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import tempfile

import numpy as np

from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn import experiment
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators import estimator_test_utils
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.estimators import test_data
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.feature_column import feature_column_lib as fc_core
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad
from tensorflow.python.training import ftrl
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util


def _assert_metrics_in_range(keys, metrics):
  epsilon = 0.00001  # Added for floating point edge cases.
  for key in keys:
    estimator_test_utils.assert_in_range(0.0 - epsilon, 1.0 + epsilon, key,
                                         metrics)


class _CheckCallsHead(head_lib.Head):
  """Head that checks whether head_ops is called."""

  def __init__(self):
    self._head_ops_called_times = 0

  @property
  def logits_dimension(self):
    return 1

  def create_model_fn_ops(
      self, mode, features, labels=None, train_op_fn=None, logits=None,
      logits_input=None, scope=None):
    """See `_Head`."""
    self._head_ops_called_times += 1
    loss = losses.mean_squared_error(labels, logits)
    return model_fn.ModelFnOps(
        mode,
        predictions={'loss': loss},
        loss=loss,
        train_op=train_op_fn(loss),
        eval_metric_ops={'loss': loss})

  @property
  def head_ops_called_times(self):
    return self._head_ops_called_times


class _StepCounterHook(session_run_hook.SessionRunHook):
  """Counts the number of training steps."""

  def __init__(self):
    self._steps = 0

  def after_run(self, run_context, run_values):
    del run_context, run_values
    self._steps += 1

  @property
  def steps(self):
    return self._steps


class EmbeddingMultiplierTest(test.TestCase):
  """dnn_model_fn tests."""

  def testRaisesNonEmbeddingColumn(self):
    one_hot_language = feature_column.one_hot_column(
        feature_column.sparse_column_with_hash_bucket('language', 10))

    params = {
        'dnn_feature_columns': [one_hot_language],
        'head': head_lib.multi_class_head(2),
        'dnn_hidden_units': [1],
        # Set lr mult to 0. to keep embeddings constant.
        'embedding_lr_multipliers': {
            one_hot_language: 0.0
        },
        'dnn_optimizer': 'Adagrad',
    }
    features = {
        'language':
            sparse_tensor.SparseTensor(
                values=['en', 'fr', 'zh'],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1]),
    }
    labels = constant_op.constant([[0], [0], [0]], dtype=dtypes.int32)
    with self.assertRaisesRegexp(ValueError,
                                 'can only be defined for embedding columns'):
      dnn_linear_combined._dnn_linear_combined_model_fn(features, labels,
                                                        model_fn.ModeKeys.TRAIN,
                                                        params)

  def testMultipliesGradient(self):
    embedding_language = feature_column.embedding_column(
        feature_column.sparse_column_with_hash_bucket('language', 10),
        dimension=1,
        initializer=init_ops.constant_initializer(0.1))
    embedding_wire = feature_column.embedding_column(
        feature_column.sparse_column_with_hash_bucket('wire', 10),
        dimension=1,
        initializer=init_ops.constant_initializer(0.1))

    params = {
        'dnn_feature_columns': [embedding_language, embedding_wire],
        'head': head_lib.multi_class_head(2),
        'dnn_hidden_units': [1],
        # Set lr mult to 0. to keep language embeddings constant, whereas wire
        # embeddings will be trained.
        'embedding_lr_multipliers': {
            embedding_language: 0.0
        },
        'dnn_optimizer': 'Adagrad',
    }
    with ops.Graph().as_default():
      features = {
          'language':
              sparse_tensor.SparseTensor(
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 1]),
          'wire':
              sparse_tensor.SparseTensor(
                  values=['omar', 'stringer', 'marlo'],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 1]),
      }
      labels = constant_op.constant([[1], [0], [0]], dtype=dtypes.int32)
      training_util.create_global_step()
      model_ops = dnn_linear_combined._dnn_linear_combined_model_fn(
          features, labels, model_fn.ModeKeys.TRAIN, params)
      with monitored_session.MonitoredSession() as sess:
        language_var = dnn_linear_combined._get_embedding_variable(
            embedding_language, 'dnn', 'dnn/input_from_feature_columns')
        language_initial_value = sess.run(language_var)
        for _ in range(2):
          _, language_value = sess.run([model_ops.train_op, language_var])

    self.assertAllClose(language_value, language_initial_value)
    # We could also test that wire_value changed, but that test would be flaky.


class DNNLinearCombinedEstimatorTest(test.TestCase):

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, dnn_linear_combined.DNNLinearCombinedEstimator)

  def testNoFeatureColumns(self):
    with self.assertRaisesRegexp(
        ValueError,
        'Either linear_feature_columns or dnn_feature_columns must be defined'):
      dnn_linear_combined.DNNLinearCombinedEstimator(
          head=_CheckCallsHead(),
          linear_feature_columns=None,
          dnn_feature_columns=None,
          dnn_hidden_units=[3, 3])

  def testCheckCallsHead(self):
    """Tests binary classification using matrix data as input."""
    head = _CheckCallsHead()
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [
        feature_column.real_valued_column('feature', dimension=4)]
    bucketized_feature = [feature_column.bucketized_column(
        cont_features[0], test_data.get_quantile_based_buckets(iris.data, 10))]

    estimator = dnn_linear_combined.DNNLinearCombinedEstimator(
        head,
        linear_feature_columns=bucketized_feature,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3])

    estimator.fit(input_fn=test_data.iris_input_multiclass_fn, steps=10)
    self.assertEqual(1, head.head_ops_called_times)

    estimator.evaluate(input_fn=test_data.iris_input_multiclass_fn, steps=10)
    self.assertEqual(2, head.head_ops_called_times)

    estimator.predict(input_fn=test_data.iris_input_multiclass_fn)
    self.assertEqual(3, head.head_ops_called_times)


class DNNLinearCombinedClassifierTest(test.TestCase):

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, dnn_linear_combined.DNNLinearCombinedClassifier)

  def testExperimentIntegration(self):
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]

    exp = experiment.Experiment(
        estimator=dnn_linear_combined.DNNLinearCombinedClassifier(
            linear_feature_columns=cont_features,
            dnn_feature_columns=cont_features,
            dnn_hidden_units=[3, 3]),
        train_input_fn=test_data.iris_input_logistic_fn,
        eval_input_fn=test_data.iris_input_logistic_fn)
    exp.test()

  def testNoFeatureColumns(self):
    with self.assertRaisesRegexp(
        ValueError,
        'Either linear_feature_columns or dnn_feature_columns must be defined'):
      dnn_linear_combined.DNNLinearCombinedClassifier(
          linear_feature_columns=None,
          dnn_feature_columns=None,
          dnn_hidden_units=[3, 3])

  def testNoDnnHiddenUnits(self):
    def _input_fn():
      return {
          'age':
              constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 100)
    age = feature_column.real_valued_column('age')

    with self.assertRaisesRegexp(
        ValueError,
        'dnn_hidden_units must be defined when dnn_feature_columns is '
        'specified'):
      classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
          dnn_feature_columns=[age, language])
      classifier.fit(input_fn=_input_fn, steps=2)

  def testSyncReplicasOptimizerUnsupported(self):
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]

    sync_optimizer = sync_replicas_optimizer.SyncReplicasOptimizer(
        opt=adagrad.AdagradOptimizer(learning_rate=0.1),
        replicas_to_aggregate=1,
        total_num_replicas=1)
    sync_hook = sync_optimizer.make_session_run_hook(is_chief=True)
    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        n_classes=3,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3],
        dnn_optimizer=sync_optimizer)

    with self.assertRaisesRegexp(
        ValueError,
        'SyncReplicasOptimizer is not supported in DNNLinearCombined model'):
      classifier.fit(
          input_fn=test_data.iris_input_multiclass_fn, steps=100,
          monitors=[sync_hook])

  def testEmbeddingMultiplier(self):
    embedding_language = feature_column.embedding_column(
        feature_column.sparse_column_with_hash_bucket('language', 10),
        dimension=1,
        initializer=init_ops.constant_initializer(0.1))
    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        dnn_feature_columns=[embedding_language],
        dnn_hidden_units=[3, 3],
        embedding_lr_multipliers={embedding_language: 0.8})
    self.assertEqual({
        embedding_language: 0.8
    }, classifier.params['embedding_lr_multipliers'])

  def testInputPartitionSize(self):
    def _input_fn_float_label(num_epochs=None):
      features = {
          'language':
              sparse_tensor.SparseTensor(
                  values=input_lib.limit_epochs(
                      ['en', 'fr', 'zh'], num_epochs=num_epochs),
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      labels = constant_op.constant([[0.8], [0.], [0.2]], dtype=dtypes.float32)
      return features, labels

    language_column = feature_column.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        feature_column.embedding_column(language_column, dimension=1),
    ]

    # Set num_ps_replica to be 10 and the min slice size to be extremely small,
    # so as to ensure that there'll be 10 partititions produced.
    config = run_config.RunConfig(tf_random_seed=1)
    config._num_ps_replicas = 10
    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        n_classes=2,
        dnn_feature_columns=feature_columns,
        dnn_hidden_units=[3, 3],
        dnn_optimizer='Adagrad',
        config=config,
        input_layer_min_slice_size=1)

    # Ensure the param is passed in.
    self.assertTrue(callable(classifier.params['input_layer_partitioner']))

    # Ensure the partition count is 10.
    classifier.fit(input_fn=_input_fn_float_label, steps=50)
    partition_count = 0
    for name in classifier.get_variable_names():
      if 'language_embedding' in name and 'Adagrad' in name:
        partition_count += 1
    self.assertEqual(10, partition_count)

  def testLogisticRegression_MatrixData(self):
    """Tests binary classification using matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]
    bucketized_feature = [
        feature_column.bucketized_column(
            cont_features[0],
            test_data.get_quantile_based_buckets(iris.data, 10))
    ]

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=bucketized_feature,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=test_data.iris_input_logistic_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_logistic_fn, steps=100)
    _assert_metrics_in_range(('accuracy', 'auc'), scores)

  def testLogisticRegression_TensorData(self):
    """Tests binary classification using Tensor data as input."""

    def _input_fn():
      iris = test_data.prepare_iris_data_for_logistic_regression()
      features = {}
      for i in range(4):
        # The following shows how to provide the Tensor data for
        # RealValuedColumns.
        features.update({
            str(i):
                array_ops.reshape(
                    constant_op.constant(
                        iris.data[:, i], dtype=dtypes.float32), [-1, 1])
        })
      # The following shows how to provide the SparseTensor data for
      # a SparseColumn.
      features['dummy_sparse_column'] = sparse_tensor.SparseTensor(
          values=['en', 'fr', 'zh'],
          indices=[[0, 0], [0, 1], [60, 0]],
          dense_shape=[len(iris.target), 2])
      labels = array_ops.reshape(
          constant_op.constant(
              iris.target, dtype=dtypes.int32), [-1, 1])
      return features, labels

    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [
        feature_column.real_valued_column(str(i)) for i in range(4)
    ]
    linear_features = [
        feature_column.bucketized_column(cont_features[i],
                                         test_data.get_quantile_based_buckets(
                                             iris.data[:, i], 10))
        for i in range(4)
    ]
    linear_features.append(
        feature_column.sparse_column_with_hash_bucket(
            'dummy_sparse_column', hash_bucket_size=100))

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=linear_features,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn, steps=100)
    _assert_metrics_in_range(('accuracy', 'auc'), scores)

  def testEstimatorWithCoreFeatureColumns(self):
    """Tests binary classification using Tensor data as input."""

    def _input_fn():
      iris = test_data.prepare_iris_data_for_logistic_regression()
      features = {}
      for i in range(4):
        # The following shows how to provide the Tensor data for
        # RealValuedColumns.
        features.update({
            str(i):
                array_ops.reshape(
                    constant_op.constant(iris.data[:, i], dtype=dtypes.float32),
                    [-1, 1])
        })
      # The following shows how to provide the SparseTensor data for
      # a SparseColumn.
      features['dummy_sparse_column'] = sparse_tensor.SparseTensor(
          values=['en', 'fr', 'zh'],
          indices=[[0, 0], [0, 1], [60, 0]],
          dense_shape=[len(iris.target), 2])
      labels = array_ops.reshape(
          constant_op.constant(iris.target, dtype=dtypes.int32), [-1, 1])
      return features, labels

    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [fc_core.numeric_column(str(i)) for i in range(4)]
    linear_features = [
        fc_core.bucketized_column(
            cont_features[i],
            sorted(set(test_data.get_quantile_based_buckets(
                iris.data[:, i], 10)))) for i in range(4)
    ]
    linear_features.append(
        fc_core.categorical_column_with_hash_bucket(
            'dummy_sparse_column', hash_bucket_size=100))

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=linear_features,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn, steps=100)
    _assert_metrics_in_range(('accuracy', 'auc'), scores)

  def testTrainWithPartitionedVariables(self):
    """Tests training with partitioned variables."""

    def _input_fn():
      features = {
          'language':
              sparse_tensor.SparseTensor(
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      labels = constant_op.constant([[1], [0], [0]])
      return features, labels

    sparse_features = [
        # The given hash_bucket_size results in variables larger than the
        # default min_slice_size attribute, so the variables are partitioned.
        feature_column.sparse_column_with_hash_bucket(
            'language', hash_bucket_size=2e7)
    ]
    embedding_features = [
        feature_column.embedding_column(
            sparse_features[0], dimension=1)
    ]

    tf_config = {
        'cluster': {
            run_config.TaskType.PS: ['fake_ps_0', 'fake_ps_1']
        }
    }
    with test.mock.patch.dict('os.environ',
                              {'TF_CONFIG': json.dumps(tf_config)}):
      config = run_config.RunConfig()
      # Because we did not start a distributed cluster, we need to pass an
      # empty ClusterSpec, otherwise the device_setter will look for
      # distributed jobs, such as "/job:ps" which are not present.
      config._cluster_spec = server_lib.ClusterSpec({})

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=sparse_features,
        dnn_feature_columns=embedding_features,
        dnn_hidden_units=[3, 3],
        config=config)

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    _assert_metrics_in_range(('accuracy', 'auc'), scores)

  def testMultiClass(self):
    """Tests multi-class classification using matrix data as input.

    Please see testLogisticRegression_TensorData() for how to use Tensor
    data as input instead.
    """
    iris = base.load_iris()
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]
    bucketized_features = [
        feature_column.bucketized_column(
            cont_features[0],
            test_data.get_quantile_based_buckets(iris.data, 10))
    ]

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        n_classes=3,
        linear_feature_columns=bucketized_features,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_multiclass_fn, steps=100)
    _assert_metrics_in_range(('accuracy',), scores)

  def testMultiClassLabelKeys(self):
    """Tests n_classes > 2 with label_keys vocabulary for labels."""
    # Byte literals needed for python3 test to pass.
    label_keys = [b'label0', b'label1', b'label2']

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[.8], [0.2], [.1]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=input_lib.limit_epochs(
                      ['en', 'fr', 'zh'], num_epochs=num_epochs),
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      labels = constant_op.constant(
          [[label_keys[1]], [label_keys[0]], [label_keys[0]]],
          dtype=dtypes.string)
      return features, labels

    language_column = feature_column.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        n_classes=3,
        linear_feature_columns=[language_column],
        dnn_feature_columns=[
            feature_column.embedding_column(
                language_column, dimension=1),
            feature_column.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        label_keys=label_keys)

    classifier.fit(input_fn=_input_fn, steps=50)

    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    _assert_metrics_in_range(('accuracy',), scores)
    self.assertIn('loss', scores)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predicted_classes = list(
        classifier.predict_classes(
            input_fn=predict_input_fn, as_iterable=True))
    self.assertEqual(3, len(predicted_classes))
    for pred in predicted_classes:
      self.assertIn(pred, label_keys)
    predictions = list(
        classifier.predict(input_fn=predict_input_fn, as_iterable=True))
    self.assertAllEqual(predicted_classes, predictions)

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The logistic prediction should be (y = 0.25).
      features = {'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),}
      labels = constant_op.constant([[1], [0], [0], [0]])
      return features, labels

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        n_classes=2,
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_train, steps=1)
    # Cross entropy = -0.25*log(0.25)-0.75*log(0.75) = 0.562
    self.assertAlmostEqual(0.562, scores['loss'], delta=0.1)

  def testLossWithWeights(self):
    """Tests loss calculation with weights."""

    def _input_fn_train():
      # 4 rows with equal weight, one of them (y = x), three of them (y=Not(x))
      # The logistic prediction should be (y = 0.25).
      features = {
          'x': array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[1.], [1.], [1.], [1.]])
      }
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      return features, labels

    def _input_fn_eval():
      # 4 rows, with different weights.
      features = {
          'x': array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[7.], [1.], [1.], [1.]])
      }
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      return features, labels

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        weight_column_name='w',
        n_classes=2,
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    # Weighted cross entropy = (-7*log(0.25)-3*log(0.75))/10 = 1.06
    self.assertAlmostEqual(1.06, scores['loss'], delta=0.1)

  def testTrainWithWeights(self):
    """Tests training with given weight column."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # First row has more weight than others. Model should fit (y=x) better
      # than (y=Not(x)) due to the relative higher weight of the first row.
      labels = constant_op.constant([[1], [0], [0], [0]])
      features = {
          'x': array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[100.], [3.], [2.], [2.]])
      }
      return features, labels

    def _input_fn_eval():
      # Create 4 rows (y = x).
      labels = constant_op.constant([[1], [1], [1], [1]])
      features = {
          'x': array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        weight_column_name='w',
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    _assert_metrics_in_range(('accuracy',), scores)

  def testCustomOptimizerByObject(self):
    """Tests binary classification using matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]
    bucketized_features = [
        feature_column.bucketized_column(
            cont_features[0],
            test_data.get_quantile_based_buckets(iris.data, 10))
    ]

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=bucketized_features,
        linear_optimizer=ftrl.FtrlOptimizer(learning_rate=0.1),
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3],
        dnn_optimizer=adagrad.AdagradOptimizer(learning_rate=0.1))

    classifier.fit(input_fn=test_data.iris_input_logistic_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_logistic_fn, steps=100)
    _assert_metrics_in_range(('accuracy',), scores)

  def testCustomOptimizerByString(self):
    """Tests binary classification using matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]
    bucketized_features = [
        feature_column.bucketized_column(
            cont_features[0],
            test_data.get_quantile_based_buckets(iris.data, 10))
    ]

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=bucketized_features,
        linear_optimizer='Ftrl',
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3],
        dnn_optimizer='Adagrad')

    classifier.fit(input_fn=test_data.iris_input_logistic_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_logistic_fn, steps=100)
    _assert_metrics_in_range(('accuracy',), scores)

  def testCustomOptimizerByFunction(self):
    """Tests binary classification using matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]
    bucketized_features = [
        feature_column.bucketized_column(
            cont_features[0],
            test_data.get_quantile_based_buckets(iris.data, 10))
    ]

    def _optimizer_exp_decay():
      global_step = training_util.get_global_step()
      learning_rate = learning_rate_decay.exponential_decay(
          learning_rate=0.1,
          global_step=global_step,
          decay_steps=100,
          decay_rate=0.001)
      return adagrad.AdagradOptimizer(learning_rate=learning_rate)

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=bucketized_features,
        linear_optimizer=_optimizer_exp_decay,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3],
        dnn_optimizer=_optimizer_exp_decay)

    classifier.fit(input_fn=test_data.iris_input_logistic_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_logistic_fn, steps=100)
    _assert_metrics_in_range(('accuracy',), scores)

  def testPredict(self):
    """Tests weight column in evaluation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = constant_op.constant([[1], [0], [0], [0]])
      features = {'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32)}
      return features, labels

    def _input_fn_predict():
      y = input_lib.limit_epochs(
          array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32), num_epochs=1)
      features = {'x': y}
      return features

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=100)

    probs = list(classifier.predict_proba(input_fn=_input_fn_predict))
    self.assertAllClose([[0.75, 0.25]] * 4, probs, 0.05)
    classes = list(classifier.predict_classes(input_fn=_input_fn_predict))
    self.assertListEqual([0] * 4, classes)

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""

    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = constant_op.constant([[1], [0], [0], [0]])
      features = {
          'x':
              input_lib.limit_epochs(
                  array_ops.ones(
                      shape=[4, 1], dtype=dtypes.float32),
                  num_epochs=num_epochs)
      }
      return features, labels

    def _my_metric_op(predictions, labels):
      # For the case of binary classification, the 2nd column of "predictions"
      # denotes the model predictions.
      labels = math_ops.cast(labels, dtypes.float32)
      predictions = array_ops.strided_slice(
          predictions, [0, 1], [-1, 2], end_mask=1)
      return math_ops.reduce_sum(math_ops.multiply(predictions, labels))

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=_input_fn,
        steps=100,
        metrics={
            'my_accuracy':
                MetricSpec(
                    metric_fn=metric_ops.streaming_accuracy,
                    prediction_key='classes'),
            'my_precision':
                MetricSpec(
                    metric_fn=metric_ops.streaming_precision,
                    prediction_key='classes'),
            'my_metric':
                MetricSpec(
                    metric_fn=_my_metric_op, prediction_key='probabilities')
        })
    self.assertTrue(
        set(['loss', 'my_accuracy', 'my_precision', 'my_metric']).issubset(
            set(scores.keys())))
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = np.array(list(classifier.predict_classes(
        input_fn=predict_input_fn)))
    self.assertEqual(
        _sklearn.accuracy_score([1, 0, 0, 0], predictions),
        scores['my_accuracy'])

    # Test the case where the 2nd element of the key is neither "classes" nor
    # "probabilities".
    with self.assertRaisesRegexp(KeyError, 'bad_type'):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=100,
          metrics={('bad_name', 'bad_type'): metric_ops.streaming_auc})

    # Test the case where the tuple of the key doesn't have 2 elements.
    with self.assertRaises(ValueError):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=100,
          metrics={
              ('bad_length_name', 'classes', 'bad_length'):
                  metric_ops.streaming_accuracy
          })

    # Test the case where the prediction_key is neither "classes" nor
    # "probabilities".
    with self.assertRaisesRegexp(KeyError, 'bad_type'):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=100,
          metrics={
              'bad_name':
                  MetricSpec(
                      metric_fn=metric_ops.streaming_auc,
                      prediction_key='bad_type')
          })

  def testVariableQuery(self):
    """Tests get_variable_names and get_variable_value."""

    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      labels = constant_op.constant([[1], [1], [1], [0]])
      features = {'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),}
      return features, labels

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=500)
    var_names = classifier.get_variable_names()
    self.assertGreater(len(var_names), 3)
    for name in var_names:
      classifier.get_variable_value(name)

  def testExport(self):
    """Tests export model for servo."""

    def input_fn():
      return {
          'age':
              constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 100)

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[
            feature_column.real_valued_column('age'),
            language,
        ],
        dnn_feature_columns=[
            feature_column.embedding_column(
                language, dimension=1),
        ],
        dnn_hidden_units=[3, 3])
    classifier.fit(input_fn=input_fn, steps=100)

    export_dir = tempfile.mkdtemp()
    input_feature_key = 'examples'

    def serving_input_fn():
      features, targets = input_fn()
      features[input_feature_key] = array_ops.placeholder(dtypes.string)
      return features, targets

    classifier.export(
        export_dir,
        serving_input_fn,
        input_feature_key,
        use_deprecated_input_fn=False)

  def testCenteredBias(self):
    """Tests bias is centered or not."""

    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      labels = constant_op.constant([[1], [1], [1], [0]])
      features = {'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),}
      return features, labels

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        enable_centered_bias=True)

    classifier.fit(input_fn=_input_fn_train, steps=1000)
    self.assertIn('binary_logistic_head/centered_bias_weight',
                  classifier.get_variable_names())
    # logodds(0.75) = 1.09861228867
    self.assertAlmostEqual(
        1.0986,
        float(classifier.get_variable_value(
            'binary_logistic_head/centered_bias_weight')[0]),
        places=2)

  def testDisableCenteredBias(self):
    """Tests bias is centered or not."""

    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      labels = constant_op.constant([[1], [1], [1], [0]])
      features = {'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),}
      return features, labels

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        enable_centered_bias=False)

    classifier.fit(input_fn=_input_fn_train, steps=500)
    self.assertNotIn('centered_bias_weight', classifier.get_variable_names())

  def testGlobalStepLinearOnly(self):
    """Tests global step update for linear-only model."""

    def input_fn():
      return {
          'age': constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 10)
    age = feature_column.real_valued_column('age')

    step_counter = _StepCounterHook()
    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=100, monitors=[step_counter])

    self.assertEqual(100, step_counter.steps)

  def testGlobalStepDNNOnly(self):
    """Tests global step update for dnn-only model."""

    def input_fn():
      return {
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 10)

    step_counter = _StepCounterHook()
    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        dnn_feature_columns=[
            feature_column.embedding_column(language, dimension=1)],
        dnn_hidden_units=[3, 3])
    classifier.fit(input_fn=input_fn, steps=100, monitors=[step_counter])

    self.assertEqual(100, step_counter.steps)

  def testGlobalStepDNNLinearCombinedBug(self):
    """Tests global step update for dnn-linear combined model."""

    def input_fn():
      return {
          'age': constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 10)
    age = feature_column.real_valued_column('age')

    step_counter = _StepCounterHook()
    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[age, language],
        dnn_feature_columns=[
            feature_column.embedding_column(language, dimension=1)],
        dnn_hidden_units=[3, 3],
        fix_global_step_increment_bug=False)
    classifier.fit(input_fn=input_fn, steps=100, monitors=[step_counter])
    global_step = classifier.get_variable_value('global_step')

    if global_step == 100:
      # Expected is 100, but because of the global step increment bug, is 50.
      # Occasionally, step increments one more time due to a race condition,
      # reaching 51 steps.
      self.assertIn(step_counter.steps, [50, 51])
    else:
      # Occasionally, training stops when global_step == 102, due to a race
      # condition. In addition, occasionally step increments one more time due
      # to a race condition reaching 52 steps.
      self.assertIn(step_counter.steps, [51, 52])

  def testGlobalStepDNNLinearCombinedBugFixed(self):
    """Tests global step update for dnn-linear combined model."""

    def input_fn():
      return {
          'age': constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 10)
    age = feature_column.real_valued_column('age')

    step_counter = _StepCounterHook()
    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[age, language],
        dnn_feature_columns=[
            feature_column.embedding_column(language, dimension=1)],
        dnn_hidden_units=[3, 3],
        fix_global_step_increment_bug=True)
    classifier.fit(input_fn=input_fn, steps=100, monitors=[step_counter])

    self.assertEqual(100, step_counter.steps)

  def testLinearOnly(self):
    """Tests that linear-only instantiation works."""

    def input_fn():
      return {
          'age':
              constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 100)
    age = feature_column.real_valued_column('age')

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=200)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)

    variable_names = classifier.get_variable_names()
    self.assertNotIn('dnn/logits/biases', variable_names)
    self.assertNotIn('dnn/logits/weights', variable_names)
    self.assertIn('linear/bias_weight', variable_names)
    self.assertIn('linear/age/weight', variable_names)
    self.assertIn('linear/language/weights', variable_names)
    self.assertEquals(
        1, len(classifier.get_variable_value('linear/age/weight')))
    self.assertEquals(
        100, len(classifier.get_variable_value('linear/language/weights')))

  def testLinearOnlyOneFeature(self):
    """Tests that linear-only instantiation works for one feature only."""

    def input_fn():
      return {
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column.sparse_column_with_hash_bucket('language', 99)

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=200)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)

    variable_names = classifier.get_variable_names()
    self.assertNotIn('dnn/logits/biases', variable_names)
    self.assertNotIn('dnn/logits/weights', variable_names)
    self.assertIn('linear/bias_weight', variable_names)
    self.assertIn('linear/language/weights', variable_names)
    self.assertEquals(
        1, len(classifier.get_variable_value('linear/bias_weight')))
    self.assertEquals(
        99, len(classifier.get_variable_value('linear/language/weights')))

  def testDNNOnly(self):
    """Tests that DNN-only instantiation works."""
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        n_classes=3, dnn_feature_columns=cont_features, dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=1000)
    classifier.evaluate(input_fn=test_data.iris_input_multiclass_fn, steps=100)

    variable_names = classifier.get_variable_names()
    self.assertIn('dnn/hiddenlayer_0/weights', variable_names)
    self.assertIn('dnn/hiddenlayer_0/biases', variable_names)
    self.assertIn('dnn/hiddenlayer_1/weights', variable_names)
    self.assertIn('dnn/hiddenlayer_1/biases', variable_names)
    self.assertIn('dnn/logits/weights', variable_names)
    self.assertIn('dnn/logits/biases', variable_names)
    self.assertNotIn('linear/bias_weight', variable_names)
    self.assertNotIn('linear/feature_BUCKETIZED/weight', variable_names)

  def testDNNWeightsBiasesNames(self):
    """Tests the names of DNN weights and biases in the checkpoints."""

    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      labels = constant_op.constant([[1], [1], [1], [0]])
      features = {'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),}
      return features, labels

    classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=5)
    variable_names = classifier.get_variable_names()
    self.assertIn('dnn/hiddenlayer_0/weights', variable_names)
    self.assertIn('dnn/hiddenlayer_0/biases', variable_names)
    self.assertIn('dnn/hiddenlayer_1/weights', variable_names)
    self.assertIn('dnn/hiddenlayer_1/biases', variable_names)
    self.assertIn('dnn/logits/weights', variable_names)
    self.assertIn('dnn/logits/biases', variable_names)


class DNNLinearCombinedRegressorTest(test.TestCase):

  def testExperimentIntegration(self):
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]

    exp = experiment.Experiment(
        estimator=dnn_linear_combined.DNNLinearCombinedRegressor(
            linear_feature_columns=cont_features,
            dnn_feature_columns=cont_features,
            dnn_hidden_units=[3, 3]),
        train_input_fn=test_data.iris_input_logistic_fn,
        eval_input_fn=test_data.iris_input_logistic_fn)
    exp.test()

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, dnn_linear_combined.DNNLinearCombinedRegressor)

  def testRegression_MatrixData(self):
    """Tests regression using matrix data as input."""
    cont_features = [feature_column.real_valued_column('feature', dimension=4)]

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=cont_features,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=test_data.iris_input_logistic_fn, steps=10)
    scores = regressor.evaluate(
        input_fn=test_data.iris_input_logistic_fn, steps=1)
    self.assertIn('loss', scores.keys())

  def testRegression_TensorData(self):
    """Tests regression using tensor data as input."""

    def _input_fn():
      # Create 4 rows of (y = x)
      labels = constant_op.constant([[100.], [3.], [2.], [2.]])
      features = {'x': constant_op.constant([[100.], [3.], [2.], [2.]])}
      return features, labels

    classifier = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=10)
    classifier.evaluate(input_fn=_input_fn, steps=1)

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The algorithm should learn (y = 0.25).
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),}
      return features, labels

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=100)
    scores = regressor.evaluate(input_fn=_input_fn_train, steps=1)
    # Average square loss = (0.75^2 + 3*0.25^2) / 4 = 0.1875
    self.assertAlmostEqual(0.1875, scores['loss'], delta=0.1)

  def testLossWithWeights(self):
    """Tests loss calculation with weights."""

    def _input_fn_train():
      # 4 rows with equal weight, one of them (y = x), three of them (y=Not(x))
      # The algorithm should learn (y = 0.25).
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    def _input_fn_eval():
      # 4 rows, with different weights.
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[7.], [1.], [1.], [1.]])
      }
      return features, labels

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        weight_column_name='w',
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=100)
    scores = regressor.evaluate(input_fn=_input_fn_eval, steps=1)
    # Weighted average square loss = (7*0.75^2 + 3*0.25^2) / 10 = 0.4125
    self.assertAlmostEqual(0.4125, scores['loss'], delta=0.1)

  def testTrainWithWeights(self):
    """Tests training with given weight column."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # First row has more weight than others. Model should fit (y=x) better
      # than (y=Not(x)) due to the relative higher weight of the first row.
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[100.], [3.], [2.], [2.]])
      }
      return features, labels

    def _input_fn_eval():
      # Create 4 rows (y = x)
      labels = constant_op.constant([[1.], [1.], [1.], [1.]])
      features = {
          'x': array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        weight_column_name='w',
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=100)
    scores = regressor.evaluate(input_fn=_input_fn_eval, steps=1)
    # The model should learn (y = x) because of the weights, so the loss should
    # be close to zero.
    self.assertLess(scores['loss'], 0.2)

  def testPredict_AsIterableFalse(self):
    """Tests predict method with as_iterable=False."""
    labels = [1., 0., 0.2]

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[0.8], [0.15], [0.]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant(labels, dtype=dtypes.float32)

    language_column = feature_column.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column, feature_column.real_valued_column('age')
        ],
        dnn_feature_columns=[
            feature_column.embedding_column(
                language_column, dimension=1),
            feature_column.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=10)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())
    regressor.predict_scores(input_fn=_input_fn, as_iterable=False)

  def testPredict_AsIterable(self):
    """Tests predict method with as_iterable=True."""
    labels = [1., 0., 0.2]

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[0.8], [0.15], [0.]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant(labels, dtype=dtypes.float32)

    language_column = feature_column.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column, feature_column.real_valued_column('age')
        ],
        dnn_feature_columns=[
            feature_column.embedding_column(
                language_column, dimension=1),
            feature_column.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=10)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    regressor.predict_scores(input_fn=predict_input_fn, as_iterable=True)

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""

    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x':
              input_lib.limit_epochs(
                  array_ops.ones(
                      shape=[4, 1], dtype=dtypes.float32),
                  num_epochs=num_epochs)
      }
      return features, labels

    def _my_metric_op(predictions, labels):
      return math_ops.reduce_sum(math_ops.multiply(predictions, labels))

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=10)
    scores = regressor.evaluate(
        input_fn=_input_fn,
        steps=1,
        metrics={
            'my_error': metric_ops.streaming_mean_squared_error,
            ('my_metric', 'scores'): _my_metric_op
        })
    self.assertIn('loss', set(scores.keys()))
    self.assertIn('my_error', set(scores.keys()))
    self.assertIn('my_metric', set(scores.keys()))
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = np.array(list(regressor.predict_scores(
        input_fn=predict_input_fn)))
    self.assertAlmostEqual(
        _sklearn.mean_squared_error(np.array([1, 0, 0, 0]), predictions),
        scores['my_error'])

    # Tests the case that the 2nd element of the key is not "scores".
    with self.assertRaises(KeyError):
      regressor.evaluate(
          input_fn=_input_fn,
          steps=1,
          metrics={
              ('my_error', 'predictions'):
                  metric_ops.streaming_mean_squared_error
          })

    # Tests the case where the tuple of the key doesn't have 2 elements.
    with self.assertRaises(ValueError):
      regressor.evaluate(
          input_fn=_input_fn,
          steps=1,
          metrics={
              ('bad_length_name', 'scores', 'bad_length'):
                  metric_ops.streaming_mean_squared_error
          })

  def testCustomMetricsWithMetricSpec(self):
    """Tests custom evaluation metrics."""

    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x':
              input_lib.limit_epochs(
                  array_ops.ones(
                      shape=[4, 1], dtype=dtypes.float32),
                  num_epochs=num_epochs)
      }
      return features, labels

    def _my_metric_op(predictions, labels):
      return math_ops.reduce_sum(math_ops.multiply(predictions, labels))

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=5)
    scores = regressor.evaluate(
        input_fn=_input_fn,
        steps=1,
        metrics={
            'my_error':
                MetricSpec(
                    metric_fn=metric_ops.streaming_mean_squared_error,
                    prediction_key='scores'),
            'my_metric':
                MetricSpec(
                    metric_fn=_my_metric_op, prediction_key='scores')
        })
    self.assertIn('loss', set(scores.keys()))
    self.assertIn('my_error', set(scores.keys()))
    self.assertIn('my_metric', set(scores.keys()))
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = np.array(list(regressor.predict_scores(
        input_fn=predict_input_fn)))
    self.assertAlmostEqual(
        _sklearn.mean_squared_error(np.array([1, 0, 0, 0]), predictions),
        scores['my_error'])

    # Tests the case where the prediction_key is not "scores".
    with self.assertRaisesRegexp(KeyError, 'bad_type'):
      regressor.evaluate(
          input_fn=_input_fn,
          steps=1,
          metrics={
              'bad_name':
                  MetricSpec(
                      metric_fn=metric_ops.streaming_auc,
                      prediction_key='bad_type')
          })

  def testExport(self):
    """Tests export model for servo."""
    labels = [1., 0., 0.2]

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[0.8], [0.15], [0.]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant(labels, dtype=dtypes.float32)

    language_column = feature_column.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column, feature_column.real_valued_column('age')
        ],
        dnn_feature_columns=[
            feature_column.embedding_column(
                language_column, dimension=1),
        ],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=10)

    export_dir = tempfile.mkdtemp()
    input_feature_key = 'examples'

    def serving_input_fn():
      features, targets = _input_fn()
      features[input_feature_key] = array_ops.placeholder(dtypes.string)
      return features, targets

    regressor.export(
        export_dir,
        serving_input_fn,
        input_feature_key,
        use_deprecated_input_fn=False)

  def testTrainSaveLoad(self):
    """Tests regression with restarting training / evaluate."""

    def _input_fn(num_epochs=None):
      # Create 4 rows of (y = x)
      labels = constant_op.constant([[100.], [3.], [2.], [2.]])
      features = {
          'x':
              input_lib.limit_epochs(
                  constant_op.constant([[100.], [3.], [2.], [2.]]),
                  num_epochs=num_epochs)
      }
      return features, labels

    model_dir = tempfile.mkdtemp()
    # pylint: disable=g-long-lambda
    new_regressor = lambda: dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        model_dir=model_dir,
        config=run_config.RunConfig(tf_random_seed=1))

    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    regressor = new_regressor()
    regressor.fit(input_fn=_input_fn, steps=10)
    predictions = list(regressor.predict_scores(input_fn=predict_input_fn))
    del regressor

    regressor = new_regressor()
    predictions2 = list(regressor.predict_scores(input_fn=predict_input_fn))
    self.assertAllClose(predictions, predictions2)

  def testTrainWithPartitionedVariables(self):
    """Tests training with partitioned variables."""

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[0.8], [0.15], [0.]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant([1., 0., 0.2], dtype=dtypes.float32)

    # The given hash_bucket_size results in variables larger than the
    # default min_slice_size attribute, so the variables are partitioned.
    language_column = feature_column.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=2e7)

    tf_config = {
        'cluster': {
            run_config.TaskType.PS: ['fake_ps_0', 'fake_ps_1']
        }
    }
    with test.mock.patch.dict('os.environ',
                              {'TF_CONFIG': json.dumps(tf_config)}):
      config = run_config.RunConfig(tf_random_seed=1)
      # Because we did not start a distributed cluster, we need to pass an
      # empty ClusterSpec, otherwise the device_setter will look for
      # distributed jobs, such as "/job:ps" which are not present.
      config._cluster_spec = server_lib.ClusterSpec({})

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column, feature_column.real_valued_column('age')
        ],
        dnn_feature_columns=[
            feature_column.embedding_column(
                language_column, dimension=1),
            feature_column.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        config=config)

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())

  def testDisableCenteredBias(self):
    """Tests that we can disable centered bias."""

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[0.8], [0.15], [0.]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant([1., 0., 0.2], dtype=dtypes.float32)

    language_column = feature_column.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column, feature_column.real_valued_column('age')
        ],
        dnn_feature_columns=[
            feature_column.embedding_column(
                language_column, dimension=1),
            feature_column.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        enable_centered_bias=False,
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())

  def testLinearOnly(self):
    """Tests linear-only instantiation and training."""

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[0.8], [0.15], [0.]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant([1., 0., 0.2], dtype=dtypes.float32)

    language_column = feature_column.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column, feature_column.real_valued_column('age')
        ],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())

  def testDNNOnly(self):
    """Tests DNN-only instantiation and training."""

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[0.8], [0.15], [0.]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant([1., 0., 0.2], dtype=dtypes.float32)

    language_column = feature_column.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        dnn_feature_columns=[
            feature_column.embedding_column(
                language_column, dimension=1),
            feature_column.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())


class FeatureEngineeringFunctionTest(test.TestCase):
  """Tests feature_engineering_fn."""

  def testNoneFeatureEngineeringFn(self):

    def input_fn():
      # Create 4 rows of (y = x)
      labels = constant_op.constant([[100.], [3.], [2.], [2.]])
      features = {'x': constant_op.constant([[100.], [3.], [2.], [2.]])}
      return features, labels

    def feature_engineering_fn(features, labels):
      _, _ = features, labels
      labels = constant_op.constant([[1000.], [30.], [20.], [20.]])
      features = {'x': constant_op.constant([[1000.], [30.], [20.], [20.]])}
      return features, labels

    estimator_with_fe_fn = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1),
        feature_engineering_fn=feature_engineering_fn)
    estimator_with_fe_fn.fit(input_fn=input_fn, steps=110)

    estimator_without_fe_fn = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[feature_column.real_valued_column('x')],
        dnn_feature_columns=[feature_column.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=run_config.RunConfig(tf_random_seed=1))
    estimator_without_fe_fn.fit(input_fn=input_fn, steps=110)

    # predictions = y
    prediction_with_fe_fn = next(
        estimator_with_fe_fn.predict_scores(
            input_fn=input_fn, as_iterable=True))
    self.assertAlmostEqual(1000., prediction_with_fe_fn, delta=10.0)
    prediction_without_fe_fn = next(
        estimator_without_fe_fn.predict_scores(
            input_fn=input_fn, as_iterable=True))
    self.assertAlmostEqual(100., prediction_without_fe_fn, delta=1.0)


if __name__ == '__main__':
  test.main()
