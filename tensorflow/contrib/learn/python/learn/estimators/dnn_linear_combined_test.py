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
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators import estimator_test_utils
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import test_data
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec


def _assert_metrics_in_range(keys, metrics):
  epsilon = 0.00001  # Added for floating point edge cases.
  for key in keys:
    estimator_test_utils.assert_in_range(
        0.0 - epsilon, 1.0 + epsilon, key, metrics)


class EmbeddingMultiplierTest(tf.test.TestCase):
  """dnn_model_fn tests."""

  def testRaisesNonEmbeddingColumn(self):
    one_hot_language = tf.contrib.layers.one_hot_column(
        tf.contrib.layers.sparse_column_with_hash_bucket('language', 10))

    params = {
        'dnn_feature_columns': [one_hot_language],
        'head': head_lib._multi_class_head(2),
        'dnn_hidden_units': [1],
        # Set lr mult to 0. to keep embeddings constant.
        'embedding_lr_multipliers': {
            one_hot_language: 0.0
        },
        'dnn_optimizer': 'Adagrad',
    }
    features = {
        'language':
            tf.SparseTensor(
                values=['en', 'fr', 'zh'],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1]),
    }
    labels = tf.constant([[0], [0], [0]], dtype=tf.int32)
    with self.assertRaisesRegexp(
        ValueError, 'can only be defined for embedding columns'):
      dnn_linear_combined._dnn_linear_combined_model_fn(
          features, labels, tf.contrib.learn.ModeKeys.TRAIN, params)

  def testMultipliesGradient(self):
    embedding_language = tf.contrib.layers.embedding_column(
        tf.contrib.layers.sparse_column_with_hash_bucket('language', 10),
        dimension=1, initializer=tf.constant_initializer(0.1))
    embedding_wire = tf.contrib.layers.embedding_column(
        tf.contrib.layers.sparse_column_with_hash_bucket('wire', 10),
        dimension=1, initializer=tf.constant_initializer(0.1))

    params = {
        'dnn_feature_columns': [embedding_language, embedding_wire],
        'head': head_lib._multi_class_head(2),
        'dnn_hidden_units': [1],
        # Set lr mult to 0. to keep embeddings constant.
        'embedding_lr_multipliers': {
            embedding_language: 0.0
        },
        'dnn_optimizer': 'Adagrad',
    }
    features = {
        'language':
            tf.SparseTensor(
                values=['en', 'fr', 'zh'],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1]),
        'wire':
            tf.SparseTensor(
                values=['omar', 'stringer', 'marlo'],
                indices=[[0, 0], [1, 0], [2, 0]],
                dense_shape=[3, 1]),
    }
    labels = tf.constant([[0], [0], [0]], dtype=tf.int32)
    model_ops = dnn_linear_combined._dnn_linear_combined_model_fn(
        features, labels, tf.contrib.learn.ModeKeys.TRAIN, params)
    with tf.train.MonitoredSession() as sess:
      language_var = dnn_linear_combined._get_embedding_variable(
          embedding_language, 'dnn', 'dnn/input_from_feature_columns')
      wire_var = dnn_linear_combined._get_embedding_variable(
          embedding_wire, 'dnn', 'dnn/input_from_feature_columns')
      for _ in range(2):
        _, language_value, wire_value = sess.run(
            [model_ops.train_op, language_var, wire_var])
      initial_value = np.full_like(language_value, 0.1)
      self.assertTrue(np.all(np.isclose(language_value, initial_value)))
      self.assertFalse(np.all(np.isclose(wire_value, initial_value)))


class DNNLinearCombinedClassifierTest(tf.test.TestCase):

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, tf.contrib.learn.DNNLinearCombinedClassifier)

  def testNoFeatureColumns(self):
    with self.assertRaisesRegexp(
        ValueError,
        'Either linear_feature_columns or dnn_feature_columns must be defined'):
      tf.contrib.learn.DNNLinearCombinedClassifier(
          linear_feature_columns=None,
          dnn_feature_columns=None,
          dnn_hidden_units=[3, 3])

  def testEmbeddingMultiplier(self):
    embedding_language = tf.contrib.layers.embedding_column(
        tf.contrib.layers.sparse_column_with_hash_bucket('language', 10),
        dimension=1, initializer=tf.constant_initializer(0.1))
    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        dnn_feature_columns=[embedding_language],
        dnn_hidden_units=[3, 3],
        embedding_lr_multipliers={embedding_language: 0.8})
    self.assertEqual(
        {embedding_language: 0.8},
        classifier._estimator.params['embedding_lr_multipliers'])

  def testLogisticRegression_MatrixData(self):
    """Tests binary classification using matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]
    bucketized_feature = [tf.contrib.layers.bucketized_column(
        cont_features[0], test_data.get_quantile_based_buckets(iris.data, 10))]

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
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
            str(i): tf.reshape(tf.constant(iris.data[:, i], dtype=tf.float32),
                               [-1, 1])})
      # The following shows how to provide the SparseTensor data for
      # a SparseColumn.
      features['dummy_sparse_column'] = tf.SparseTensor(
          values=['en', 'fr', 'zh'],
          indices=[[0, 0], [0, 1], [60, 0]],
          dense_shape=[len(iris.target), 2])
      labels = tf.reshape(tf.constant(iris.target, dtype=tf.int32), [-1, 1])
      return features, labels

    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [tf.contrib.layers.real_valued_column(str(i))
                     for i in range(4)]
    linear_features = [
        tf.contrib.layers.bucketized_column(
            cont_features[i], test_data.get_quantile_based_buckets(
                iris.data[:, i], 10)) for i in range(4)
    ]
    linear_features.append(tf.contrib.layers.sparse_column_with_hash_bucket(
        'dummy_sparse_column', hash_bucket_size=100))

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
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
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      dense_shape=[3, 2])
      }
      labels = tf.constant([[1], [0], [0]])
      return features, labels

    sparse_features = [
        # The given hash_bucket_size results in variables larger than the
        # default min_slice_size attribute, so the variables are partitioned.
        tf.contrib.layers.sparse_column_with_hash_bucket('language',
                                                         hash_bucket_size=2e7)
    ]
    embedding_features = [
        tf.contrib.layers.embedding_column(sparse_features[0], dimension=1)
    ]

    tf_config = {
        'cluster': {
            tf.contrib.learn.TaskType.PS: ['fake_ps_0', 'fake_ps_1']
        }
    }
    with tf.test.mock.patch.dict('os.environ',
                                 {'TF_CONFIG': json.dumps(tf_config)}):
      config = tf.contrib.learn.RunConfig()
      # Because we did not start a distributed cluster, we need to pass an
      # empty ClusterSpec, otherwise the device_setter will look for
      # distributed jobs, such as "/job:ps" which are not present.
      config._cluster_spec = tf.train.ClusterSpec({})

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
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
    iris = tf.contrib.learn.datasets.load_iris()
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]
    bucketized_features = [
        tf.contrib.layers.bucketized_column(
            cont_features[0],
            test_data.get_quantile_based_buckets(iris.data, 10))]

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        n_classes=3,
        linear_feature_columns=bucketized_features,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_multiclass_fn, steps=100)
    _assert_metrics_in_range(('accuracy',), scores)

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The logistic prediction should be (y = 0.25).
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
      }
      labels = tf.constant([[1], [0], [0], [0]])
      return features, labels

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        n_classes=2,
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

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
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[1.], [1.], [1.], [1.]])
      }
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      return features, labels

    def _input_fn_eval():
      # 4 rows, with different weights.
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[7.], [1.], [1.], [1.]])
      }
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      return features, labels

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        weight_column_name='w',
        n_classes=2,
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))
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
      labels = tf.constant([[1], [0], [0], [0]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[100.], [3.], [2.], [2.]])
      }
      return features, labels

    def _input_fn_eval():
      # Create 4 rows (y = x).
      labels = tf.constant([[1], [1], [1], [1]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        weight_column_name='w',
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    _assert_metrics_in_range(('accuracy',), scores)

  def testCustomOptimizerByObject(self):
    """Tests binary classification using matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]
    bucketized_features = [
        tf.contrib.layers.bucketized_column(
            cont_features[0],
            test_data.get_quantile_based_buckets(iris.data, 10))]

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=bucketized_features,
        linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3],
        dnn_optimizer=tf.train.AdagradOptimizer(learning_rate=0.1))

    classifier.fit(input_fn=test_data.iris_input_logistic_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_logistic_fn, steps=100)
    _assert_metrics_in_range(('accuracy',), scores)

  def testCustomOptimizerByString(self):
    """Tests binary classification using matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]
    bucketized_features = [
        tf.contrib.layers.bucketized_column(
            cont_features[0],
            test_data.get_quantile_based_buckets(iris.data, 10))]

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
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
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)
    ]
    bucketized_features = [
        tf.contrib.layers.bucketized_column(
            cont_features[0],
            test_data.get_quantile_based_buckets(iris.data, 10))
    ]

    def _optimizer_exp_decay():
      global_step = tf.contrib.framework.get_global_step()
      learning_rate = tf.train.exponential_decay(learning_rate=0.1,
                                                 global_step=global_step,
                                                 decay_steps=100,
                                                 decay_rate=0.001)
      return tf.train.AdagradOptimizer(learning_rate=learning_rate)

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
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
      labels = tf.constant([[1], [0], [0], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32)}
      return features, labels

    def _input_fn_predict():
      y = tf.train.limit_epochs(
          tf.ones(shape=[4, 1], dtype=tf.float32), num_epochs=1)
      features = {'x': y}
      return features

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=100)

    probs = list(classifier.predict_proba(input_fn=_input_fn_predict))
    self.assertAllClose([[0.75, 0.25]] * 4, probs, 0.05)
    classes = list(classifier.predict(input_fn=_input_fn_predict))
    self.assertListEqual([0] * 4, classes)

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""

    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = tf.constant([[1], [0], [0], [0]])
      features = {
          'x': tf.train.limit_epochs(
              tf.ones(shape=[4, 1], dtype=tf.float32), num_epochs=num_epochs)}
      return features, labels

    def _my_metric_op(predictions, labels):
      # For the case of binary classification, the 2nd column of "predictions"
      # denotes the model predictions.
      labels = tf.to_float(labels)
      predictions = tf.strided_slice(
          predictions, [0, 1], [-1, 2], end_mask=1)
      return tf.reduce_sum(tf.multiply(predictions, labels))

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=_input_fn,
        steps=100,
        metrics={
            'my_accuracy': MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key='classes'),
            'my_precision': MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_precision,
                prediction_key='classes'),
            'my_metric': MetricSpec(
                metric_fn=_my_metric_op,
                prediction_key='probabilities')
        })
    self.assertTrue(
        set(['loss', 'my_accuracy', 'my_precision', 'my_metric'
            ]).issubset(set(scores.keys())))
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = np.array(
        list(classifier.predict(input_fn=predict_input_fn)))
    self.assertEqual(_sklearn.accuracy_score([1, 0, 0, 0], predictions),
                     scores['my_accuracy'])

    # Test the case where the 2nd element of the key is neither "classes" nor
    # "probabilities".
    with self.assertRaisesRegexp(KeyError, 'bad_type'):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=100,
          metrics={('bad_name', 'bad_type'): tf.contrib.metrics.streaming_auc})

    # Test the case where the tuple of the key doesn't have 2 elements.
    with self.assertRaises(ValueError):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=100,
          metrics={
              ('bad_length_name', 'classes', 'bad_length'):
                  tf.contrib.metrics.streaming_accuracy
          })

    # Test the case where the prediction_key is neither "classes" nor
    # "probabilities".
    with self.assertRaisesRegexp(KeyError, 'bad_type'):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=100,
          metrics={
              'bad_name': MetricSpec(
                  metric_fn=tf.contrib.metrics.streaming_auc,
                  prediction_key='bad_type')})

  def testVariableQuery(self):
    """Tests bias is centered or not."""
    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      labels = tf.constant([[1], [1], [1], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, labels

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
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
          'age': tf.constant([1]),
          'language': tf.SparseTensor(values=['english'],
                                      indices=[[0, 0]],
                                      dense_shape=[1, 1])
      }, tf.constant([[1]])

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 100)

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[
            tf.contrib.layers.real_valued_column('age'),
            language,
        ],
        dnn_feature_columns=[
            tf.contrib.layers.embedding_column(language, dimension=1),
        ],
        dnn_hidden_units=[3, 3])
    classifier.fit(input_fn=input_fn, steps=100)

    export_dir = tempfile.mkdtemp()
    input_feature_key = 'examples'
    def serving_input_fn():
      features, targets = input_fn()
      features[input_feature_key] = tf.placeholder(tf.string)
      return features, targets
    classifier.export(export_dir, serving_input_fn, input_feature_key,
                      use_deprecated_input_fn=False)

  def testCenteredBias(self):
    """Tests bias is centered or not."""
    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      labels = tf.constant([[1], [1], [1], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, labels

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        enable_centered_bias=True)

    classifier.fit(input_fn=_input_fn_train, steps=1000)
    # logodds(0.75) = 1.09861228867
    self.assertAlmostEqual(
        1.0986,
        float(classifier.get_variable_value('centered_bias_weight')[0]),
        places=2)

  def testDisableCenteredBias(self):
    """Tests bias is centered or not."""
    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      labels = tf.constant([[1], [1], [1], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, labels

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        enable_centered_bias=False)

    classifier.fit(input_fn=_input_fn_train, steps=500)
    self.assertNotIn('centered_bias_weight', classifier.get_variable_names())

  def testLinearOnly(self):
    """Tests that linear-only instantiation works."""
    def input_fn():
      return {
          'age': tf.constant([1]),
          'language': tf.SparseTensor(values=['english'],
                                      indices=[[0, 0]],
                                      dense_shape=[1, 1])
      }, tf.constant([[1]])

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 100)
    age = tf.contrib.layers.real_valued_column('age')

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=200)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)

    self.assertNotIn('dnn/logits/biases', classifier.get_variable_names())
    self.assertNotIn('dnn/logits/weights', classifier.get_variable_names())
    self.assertEquals(1, len(classifier.linear_bias_))
    self.assertEquals(2, len(classifier.linear_weights_))
    self.assertEquals(1, len(classifier.linear_weights_['linear/age/weight']))
    self.assertEquals(
        100, len(classifier.linear_weights_['linear/language/weights']))

  def testLinearOnlyOneFeature(self):
    """Tests that linear-only instantiation works for one feature only."""
    def input_fn():
      return {
          'language': tf.SparseTensor(values=['english'],
                                      indices=[[0, 0]],
                                      dense_shape=[1, 1])
      }, tf.constant([[1]])

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 99)

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=200)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)

    self.assertNotIn('dnn/logits/biases', classifier.get_variable_names())
    self.assertNotIn('dnn/logits/weights', classifier.get_variable_names())
    self.assertEquals(1, len(classifier.linear_bias_))
    self.assertEquals(99, len(classifier.linear_weights_))

  def testDNNOnly(self):
    """Tests that DNN-only instantiation works."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        n_classes=3, dnn_feature_columns=cont_features, dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=1000)
    classifier.evaluate(input_fn=test_data.iris_input_multiclass_fn, steps=100)

    self.assertEquals(3, len(classifier.dnn_bias_))
    self.assertEquals(3, len(classifier.dnn_weights_))
    self.assertNotIn('linear/bias_weight', classifier.get_variable_names())
    self.assertNotIn('linear/feature_BUCKETIZED_weights',
                     classifier.get_variable_names())

  def testDNNWeightsBiasesNames(self):
    """Tests the names of DNN weights and biases in the checkpoints."""
    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      labels = tf.constant([[1], [1], [1], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, labels
    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=5)
    # hiddenlayer_0/weights,hiddenlayer_1/weights and dnn_logits/weights.
    self.assertEquals(3, len(classifier.dnn_weights_))
    # hiddenlayer_0/biases, hiddenlayer_1/biases, dnn_logits/biases.
    self.assertEquals(3, len(classifier.dnn_bias_))


class DNNLinearCombinedRegressorTest(tf.test.TestCase):

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, tf.contrib.learn.DNNLinearCombinedRegressor)

  def testRegression_MatrixData(self):
    """Tests regression using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=cont_features,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=test_data.iris_input_logistic_fn, steps=10)
    scores = regressor.evaluate(
        input_fn=test_data.iris_input_logistic_fn, steps=1)
    self.assertIn('loss', scores.keys())

  def testRegression_TensorData(self):
    """Tests regression using tensor data as input."""
    def _input_fn():
      # Create 4 rows of (y = x)
      labels = tf.constant([[100.], [3.], [2.], [2.]])
      features = {'x': tf.constant([[100.], [3.], [2.], [2.]])}
      return features, labels

    classifier = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=10)
    classifier.evaluate(input_fn=_input_fn, steps=1)

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The algorithm should learn (y = 0.25).
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
      }
      return features, labels

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=100)
    scores = regressor.evaluate(input_fn=_input_fn_train, steps=1)
    # Average square loss = (0.75^2 + 3*0.25^2) / 4 = 0.1875
    self.assertAlmostEqual(0.1875, scores['loss'], delta=0.1)

  def testLossWithWeights(self):
    """Tests loss calculation with weights."""

    def _input_fn_train():
      # 4 rows with equal weight, one of them (y = x), three of them (y=Not(x))
      # The algorithm should learn (y = 0.25).
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    def _input_fn_eval():
      # 4 rows, with different weights.
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[7.], [1.], [1.], [1.]])
      }
      return features, labels

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        weight_column_name='w',
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

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
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[100.], [3.], [2.], [2.]])
      }
      return features, labels

    def _input_fn_eval():
      # Create 4 rows (y = x)
      labels = tf.constant([[1.], [1.], [1.], [1.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        weight_column_name='w',
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

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
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      dense_shape=[3, 2])
      }
      return features, tf.constant(labels, dtype=tf.float32)

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column,
            tf.contrib.layers.real_valued_column('age')
        ],
        dnn_feature_columns=[
            tf.contrib.layers.embedding_column(language_column, dimension=1),
            tf.contrib.layers.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=10)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())
    regressor.predict(input_fn=_input_fn, as_iterable=False)

  def testPredict_AsIterable(self):
    """Tests predict method with as_iterable=True."""
    labels = [1., 0., 0.2]
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      dense_shape=[3, 2])
      }
      return features, tf.constant(labels, dtype=tf.float32)

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column,
            tf.contrib.layers.real_valued_column('age')
        ],
        dnn_feature_columns=[
            tf.contrib.layers.embedding_column(language_column, dimension=1),
            tf.contrib.layers.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=10)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    regressor.predict(input_fn=predict_input_fn, as_iterable=True)

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""
    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      features = {'x': tf.train.limit_epochs(
          tf.ones(shape=[4, 1], dtype=tf.float32), num_epochs=num_epochs)}
      return features, labels

    def _my_metric_op(predictions, labels):
      return tf.reduce_sum(tf.multiply(predictions, labels))

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=10)
    scores = regressor.evaluate(
        input_fn=_input_fn,
        steps=1,
        metrics={
            'my_error': tf.contrib.metrics.streaming_mean_squared_error,
            ('my_metric', 'scores'): _my_metric_op
        })
    self.assertIn('loss', set(scores.keys()))
    self.assertIn('my_error', set(scores.keys()))
    self.assertIn('my_metric', set(scores.keys()))
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = np.array(list(regressor.predict(input_fn=predict_input_fn)))
    self.assertAlmostEqual(
        _sklearn.mean_squared_error(np.array([1, 0, 0, 0]), predictions),
        scores['my_error'])

    # Tests the case that the 2nd element of the key is not "scores".
    with self.assertRaises(KeyError):
      regressor.evaluate(
          input_fn=_input_fn,
          steps=1,
          metrics={('my_error', 'predictions'
                   ): tf.contrib.metrics.streaming_mean_squared_error})

    # Tests the case where the tuple of the key doesn't have 2 elements.
    with self.assertRaises(ValueError):
      regressor.evaluate(
          input_fn=_input_fn,
          steps=1,
          metrics={
              ('bad_length_name', 'scores', 'bad_length'):
                  tf.contrib.metrics.streaming_mean_squared_error
          })

  def testCustomMetricsWithMetricSpec(self):
    """Tests custom evaluation metrics."""
    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      features = {'x': tf.train.limit_epochs(
          tf.ones(shape=[4, 1], dtype=tf.float32), num_epochs=num_epochs)}
      return features, labels

    def _my_metric_op(predictions, labels):
      return tf.reduce_sum(tf.multiply(predictions, labels))

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=5)
    scores = regressor.evaluate(
        input_fn=_input_fn,
        steps=1,
        metrics={
            'my_error': MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_mean_squared_error,
                prediction_key='scores'),
            'my_metric': MetricSpec(
                metric_fn=_my_metric_op,
                prediction_key='scores')
        })
    self.assertIn('loss', set(scores.keys()))
    self.assertIn('my_error', set(scores.keys()))
    self.assertIn('my_metric', set(scores.keys()))
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = np.array(list(regressor.predict(input_fn=predict_input_fn)))
    self.assertAlmostEqual(
        _sklearn.mean_squared_error(np.array([1, 0, 0, 0]), predictions),
        scores['my_error'])

    # Tests the case where the prediction_key is not "scores".
    with self.assertRaisesRegexp(KeyError, 'bad_type'):
      regressor.evaluate(
          input_fn=_input_fn,
          steps=1,
          metrics={
              'bad_name': MetricSpec(
                  metric_fn=tf.contrib.metrics.streaming_auc,
                  prediction_key='bad_type')})

  def testExport(self):
    """Tests export model for servo."""
    labels = [1., 0., 0.2]
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      dense_shape=[3, 2])
      }
      return features, tf.constant(labels, dtype=tf.float32)

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column,
            tf.contrib.layers.real_valued_column('age')
        ],
        dnn_feature_columns=[
            tf.contrib.layers.embedding_column(language_column, dimension=1),
        ],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=10)

    export_dir = tempfile.mkdtemp()
    input_feature_key = 'examples'
    def serving_input_fn():
      features, targets = _input_fn()
      features[input_feature_key] = tf.placeholder(tf.string)
      return features, targets
    regressor.export(export_dir, serving_input_fn, input_feature_key,
                     use_deprecated_input_fn=False)

  def testTrainSaveLoad(self):
    """Tests regression with restarting training / evaluate."""
    def _input_fn(num_epochs=None):
      # Create 4 rows of (y = x)
      labels = tf.constant([[100.], [3.], [2.], [2.]])
      features = {'x': tf.train.limit_epochs(
          tf.constant([[100.], [3.], [2.], [2.]]), num_epochs=num_epochs)}
      return features, labels

    model_dir = tempfile.mkdtemp()
    # pylint: disable=g-long-lambda
    new_estimator = lambda: tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    classifier = new_estimator()
    classifier.fit(input_fn=_input_fn, steps=10)
    predictions = list(classifier.predict(input_fn=predict_input_fn))
    del classifier

    classifier = new_estimator()
    predictions2 = list(classifier.predict(input_fn=predict_input_fn))
    self.assertAllClose(predictions, predictions2)

  def testTrainWithPartitionedVariables(self):
    """Tests training with partitioned variables."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      dense_shape=[3, 2])
      }
      return features, tf.constant([1., 0., 0.2], dtype=tf.float32)

    # The given hash_bucket_size results in variables larger than the
    # default min_slice_size attribute, so the variables are partitioned.
    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=2e7)

    tf_config = {
        'cluster': {
            tf.contrib.learn.TaskType.PS: ['fake_ps_0', 'fake_ps_1']
        }
    }
    with tf.test.mock.patch.dict('os.environ',
                                 {'TF_CONFIG': json.dumps(tf_config)}):
      config = tf.contrib.learn.RunConfig(tf_random_seed=1)
      # Because we did not start a distributed cluster, we need to pass an
      # empty ClusterSpec, otherwise the device_setter will look for
      # distributed jobs, such as "/job:ps" which are not present.
      config._cluster_spec = tf.train.ClusterSpec({})

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column,
            tf.contrib.layers.real_valued_column('age')
        ],
        dnn_feature_columns=[
            tf.contrib.layers.embedding_column(language_column, dimension=1),
            tf.contrib.layers.real_valued_column('age')
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
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      dense_shape=[3, 2])
      }
      return features, tf.constant([1., 0., 0.2], dtype=tf.float32)

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column,
            tf.contrib.layers.real_valued_column('age')
        ],
        dnn_feature_columns=[
            tf.contrib.layers.embedding_column(language_column, dimension=1),
            tf.contrib.layers.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        enable_centered_bias=False,
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())

  def testLinearOnly(self):
    """Tests linear-only instantiation and training."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      dense_shape=[3, 2])
      }
      return features, tf.constant([1., 0., 0.2], dtype=tf.float32)

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[
            language_column,
            tf.contrib.layers.real_valued_column('age')
        ],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())

  def testDNNOnly(self):
    """Tests DNN-only instantiation and training."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      dense_shape=[3, 2])
      }
      return features, tf.constant([1., 0., 0.2], dtype=tf.float32)

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)

    regressor = tf.contrib.learn.DNNLinearCombinedRegressor(
        dnn_feature_columns=[
            tf.contrib.layers.embedding_column(language_column, dimension=1),
            tf.contrib.layers.real_valued_column('age')
        ],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores.keys())


class FeatureEngineeringFunctionTest(tf.test.TestCase):
  """Tests feature_engineering_fn."""

  def testNoneFeatureEngineeringFn(self):
    def input_fn():
      # Create 4 rows of (y = x)
      labels = tf.constant([[100.], [3.], [2.], [2.]])
      features = {'x': tf.constant([[100.], [3.], [2.], [2.]])}
      return features, labels

    def feature_engineering_fn(features, labels):
      _, _ = features, labels
      labels = tf.constant([[1000.], [30.], [20.], [20.]])
      features = {'x': tf.constant([[1000.], [30.], [20.], [20.]])}
      return features, labels

    estimator_with_fe_fn = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1),
        feature_engineering_fn=feature_engineering_fn)
    estimator_with_fe_fn.fit(input_fn=input_fn, steps=100)

    estimator_without_fe_fn = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    estimator_without_fe_fn.fit(input_fn=input_fn, steps=100)

    # predictions = y
    prediction_with_fe_fn = next(
        estimator_with_fe_fn.predict(input_fn=input_fn, as_iterable=True))
    self.assertAlmostEqual(1000., prediction_with_fe_fn, delta=10.0)
    prediction_without_fe_fn = next(
        estimator_without_fe_fn.predict(input_fn=input_fn, as_iterable=True))
    self.assertAlmostEqual(100., prediction_without_fe_fn, delta=1.0)


if __name__ == '__main__':
  tf.test.main()
