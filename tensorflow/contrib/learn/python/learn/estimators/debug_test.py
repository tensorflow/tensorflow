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
"""Tests for Debug estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import operator
import tempfile

import numpy as np

from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.learn.python.learn import experiment
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import debug
from tensorflow.contrib.learn.python.learn.estimators import estimator_test_utils
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.estimators import test_data
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_lib

NUM_EXAMPLES = 100
N_CLASSES = 5  #  Cardinality of multiclass labels.
LABEL_DIMENSION = 3  #  Dimensionality of regression labels.


def _train_test_split(features_and_labels):
  features, labels = features_and_labels
  train_set = (features[:int(len(features) / 2)],
               labels[:int(len(features) / 2)])
  test_set = (features[int(len(features) / 2):],
              labels[int(len(features) / 2):])
  return train_set, test_set


def _input_fn_builder(features, labels):

  def input_fn():
    feature_dict = {'features': constant_op.constant(features)}
    my_labels = labels
    if my_labels is not None:
      my_labels = constant_op.constant(my_labels)
    return feature_dict, my_labels

  return input_fn


class DebugClassifierTest(test.TestCase):

  def setUp(self):
    np.random.seed(100)
    self.features = np.random.rand(NUM_EXAMPLES, 5)
    self.labels = np.random.choice(
        range(N_CLASSES), p=[0.1, 0.3, 0.4, 0.1, 0.1], size=NUM_EXAMPLES)
    self.binary_labels = np.random.choice(
        range(2), p=[0.2, 0.8], size=NUM_EXAMPLES)
    self.binary_float_labels = np.random.choice(
        range(2), p=[0.2, 0.8], size=NUM_EXAMPLES)

  def testPredict(self):
    """Tests that DebugClassifier outputs the majority class."""
    (train_features, train_labels), (test_features,
                                     test_labels) = _train_test_split(
                                         [self.features, self.labels])
    majority_class, _ = max(
        collections.Counter(train_labels).items(), key=operator.itemgetter(1))
    expected_prediction = np.vstack(
        [[majority_class] for _ in range(test_labels.shape[0])])

    classifier = debug.DebugClassifier(n_classes=N_CLASSES)
    classifier.fit(
        input_fn=_input_fn_builder(train_features, train_labels), steps=50)

    pred = classifier.predict_classes(
        input_fn=_input_fn_builder(test_features, None))
    self.assertAllEqual(expected_prediction, np.vstack(pred))

  def testPredictBinary(self):
    """Same as above for binary predictions."""
    (train_features, train_labels), (test_features,
                                     test_labels) = _train_test_split(
                                         [self.features, self.binary_labels])

    majority_class, _ = max(
        collections.Counter(train_labels).items(), key=operator.itemgetter(1))
    expected_prediction = np.vstack(
        [[majority_class] for _ in range(test_labels.shape[0])])

    classifier = debug.DebugClassifier(n_classes=2)
    classifier.fit(
        input_fn=_input_fn_builder(train_features, train_labels), steps=50)

    pred = classifier.predict_classes(
        input_fn=_input_fn_builder(test_features, None))
    self.assertAllEqual(expected_prediction, np.vstack(pred))

    (train_features,
     train_labels), (test_features, test_labels) = _train_test_split(
         [self.features, self.binary_float_labels])

    majority_class, _ = max(
        collections.Counter(train_labels).items(), key=operator.itemgetter(1))
    expected_prediction = np.vstack(
        [[majority_class] for _ in range(test_labels.shape[0])])

    classifier = debug.DebugClassifier(n_classes=2)
    classifier.fit(
        input_fn=_input_fn_builder(train_features, train_labels), steps=50)

    pred = classifier.predict_classes(
        input_fn=_input_fn_builder(test_features, None))
    self.assertAllEqual(expected_prediction, np.vstack(pred))

  def testPredictProba(self):
    """Tests that DebugClassifier outputs observed class distribution."""
    (train_features, train_labels), (test_features,
                                     test_labels) = _train_test_split(
                                         [self.features, self.labels])

    class_distribution = np.zeros((1, N_CLASSES))
    for label in train_labels:
      class_distribution[0, label] += 1
    class_distribution /= len(train_labels)

    expected_prediction = np.vstack(
        [class_distribution for _ in range(test_labels.shape[0])])

    classifier = debug.DebugClassifier(n_classes=N_CLASSES)
    classifier.fit(
        input_fn=_input_fn_builder(train_features, train_labels), steps=50)

    pred = classifier.predict_proba(
        input_fn=_input_fn_builder(test_features, None))

    self.assertAllClose(expected_prediction, np.vstack(pred), atol=0.1)

  def testPredictProbaBinary(self):
    """Same as above but for binary classification."""
    (train_features, train_labels), (test_features,
                                     test_labels) = _train_test_split(
                                         [self.features, self.binary_labels])

    class_distribution = np.zeros((1, 2))
    for label in train_labels:
      class_distribution[0, label] += 1
    class_distribution /= len(train_labels)

    expected_prediction = np.vstack(
        [class_distribution for _ in range(test_labels.shape[0])])

    classifier = debug.DebugClassifier(n_classes=2)
    classifier.fit(
        input_fn=_input_fn_builder(train_features, train_labels), steps=50)

    pred = classifier.predict_proba(
        input_fn=_input_fn_builder(test_features, None))

    self.assertAllClose(expected_prediction, np.vstack(pred), atol=0.1)

    (train_features,
     train_labels), (test_features, test_labels) = _train_test_split(
         [self.features, self.binary_float_labels])

    class_distribution = np.zeros((1, 2))
    for label in train_labels:
      class_distribution[0, int(label)] += 1
    class_distribution /= len(train_labels)

    expected_prediction = np.vstack(
        [class_distribution for _ in range(test_labels.shape[0])])

    classifier = debug.DebugClassifier(n_classes=2)
    classifier.fit(
        input_fn=_input_fn_builder(train_features, train_labels), steps=50)

    pred = classifier.predict_proba(
        input_fn=_input_fn_builder(test_features, None))

    self.assertAllClose(expected_prediction, np.vstack(pred), atol=0.1)

  def testExperimentIntegration(self):
    exp = experiment.Experiment(
        estimator=debug.DebugClassifier(n_classes=3),
        train_input_fn=test_data.iris_input_multiclass_fn,
        eval_input_fn=test_data.iris_input_multiclass_fn)
    exp.test()

  def _assertInRange(self, expected_min, expected_max, actual):
    self.assertLessEqual(expected_min, actual)
    self.assertGreaterEqual(expected_max, actual)

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(self, debug.DebugClassifier)

  def testLogisticRegression_MatrixData(self):
    """Tests binary classification using matrix data as input."""
    classifier = debug.DebugClassifier(
        config=run_config.RunConfig(tf_random_seed=1))
    input_fn = test_data.iris_input_logistic_fn
    classifier.fit(input_fn=input_fn, steps=5)
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)

  def testLogisticRegression_MatrixData_Labels1D(self):
    """Same as the last test, but label shape is [100] instead of [100, 1]."""

    def _input_fn():
      iris = test_data.prepare_iris_data_for_logistic_regression()
      return {
          'feature': constant_op.constant(iris.data, dtype=dtypes.float32)
      }, constant_op.constant(
          iris.target, shape=[100], dtype=dtypes.int32)

    classifier = debug.DebugClassifier(
        config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(input_fn=_input_fn, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)

  def testLogisticRegression_NpMatrixData(self):
    """Tests binary classification using numpy matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    train_x = iris.data
    train_y = iris.target
    classifier = debug.DebugClassifier(
        config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(x=train_x, y=train_y, steps=5)
    scores = classifier.evaluate(x=train_x, y=train_y, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])

  def _assertBinaryPredictions(self, expected_len, predictions):
    self.assertEqual(expected_len, len(predictions))
    for prediction in predictions:
      self.assertIn(prediction, (0, 1))

  def _assertProbabilities(self, expected_batch_size, expected_n_classes,
                           probabilities):
    self.assertEqual(expected_batch_size, len(probabilities))
    for b in range(expected_batch_size):
      self.assertEqual(expected_n_classes, len(probabilities[b]))
      for i in range(expected_n_classes):
        self._assertInRange(0.0, 1.0, probabilities[b][i])

  def testLogisticRegression_TensorData(self):
    """Tests binary classification using tensor data as input."""

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
      return features, constant_op.constant([[1], [0], [0]], dtype=dtypes.int32)

    classifier = debug.DebugClassifier(n_classes=2)

    classifier.fit(input_fn=_input_fn, steps=50)

    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(classifier.predict_classes(input_fn=predict_input_fn))
    self._assertBinaryPredictions(3, predictions)

  def testLogisticRegression_FloatLabel(self):
    """Tests binary classification with float labels."""

    def _input_fn_float_label(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[50], [20], [10]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=input_lib.limit_epochs(
                      ['en', 'fr', 'zh'], num_epochs=num_epochs),
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      labels = constant_op.constant([[0.8], [0.], [0.2]], dtype=dtypes.float32)
      return features, labels

    classifier = debug.DebugClassifier(n_classes=2)

    classifier.fit(input_fn=_input_fn_float_label, steps=50)

    predict_input_fn = functools.partial(_input_fn_float_label, num_epochs=1)
    predictions = list(classifier.predict_classes(input_fn=predict_input_fn))
    self._assertBinaryPredictions(3, predictions)
    predictions_proba = list(
        classifier.predict_proba(input_fn=predict_input_fn))
    self._assertProbabilities(3, 2, predictions_proba)

  def testMultiClass_MatrixData(self):
    """Tests multi-class classification using matrix data as input."""
    classifier = debug.DebugClassifier(n_classes=3)

    input_fn = test_data.iris_input_multiclass_fn
    classifier.fit(input_fn=input_fn, steps=200)
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)

  def testMultiClass_MatrixData_Labels1D(self):
    """Same as the last test, but label shape is [150] instead of [150, 1]."""

    def _input_fn():
      iris = base.load_iris()
      return {
          'feature': constant_op.constant(iris.data, dtype=dtypes.float32)
      }, constant_op.constant(
          iris.target, shape=[150], dtype=dtypes.int32)

    classifier = debug.DebugClassifier(n_classes=3)

    classifier.fit(input_fn=_input_fn, steps=200)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])

  def testMultiClass_NpMatrixData(self):
    """Tests multi-class classification using numpy matrix data as input."""
    iris = base.load_iris()
    train_x = iris.data
    train_y = iris.target
    classifier = debug.DebugClassifier(n_classes=3)
    classifier.fit(x=train_x, y=train_y, steps=200)
    scores = classifier.evaluate(x=train_x, y=train_y, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])

  def testMultiClass_StringLabel(self):
    """Tests multi-class classification with string labels."""

    def _input_fn_train():
      labels = constant_op.constant([['foo'], ['bar'], ['baz'], ['bar']])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
      }
      return features, labels

    classifier = debug.DebugClassifier(
        n_classes=3, label_keys=['foo', 'bar', 'baz'])

    classifier.fit(input_fn=_input_fn_train, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn_train, steps=1)
    self.assertIn('loss', scores)

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The logistic prediction should be (y = 0.25).
      labels = constant_op.constant([[1], [0], [0], [0]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
      }
      return features, labels

    classifier = debug.DebugClassifier(n_classes=2)

    classifier.fit(input_fn=_input_fn_train, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn_train, steps=1)
    self.assertIn('loss', scores)

  def testLossWithWeights(self):
    """Tests loss calculation with weights."""

    def _input_fn_train():
      # 4 rows with equal weight, one of them (y = x), three of them (y=Not(x))
      # The logistic prediction should be (y = 0.25).
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    def _input_fn_eval():
      # 4 rows, with different weights.
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[7.], [1.], [1.], [1.]])
      }
      return features, labels

    classifier = debug.DebugClassifier(
        weight_column_name='w',
        n_classes=2,
        config=run_config.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn_train, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    self.assertIn('loss', scores)

  def testTrainWithWeights(self):
    """Tests training with given weight column."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # First row has more weight than others. Model should fit (y=x) better
      # than (y=Not(x)) due to the relative higher weight of the first row.
      labels = constant_op.constant([[1], [0], [0], [0]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[100.], [3.], [2.], [2.]])
      }
      return features, labels

    def _input_fn_eval():
      # Create 4 rows (y = x)
      labels = constant_op.constant([[1], [1], [1], [1]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    classifier = debug.DebugClassifier(weight_column_name='w')

    classifier.fit(input_fn=_input_fn_train, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""

    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = constant_op.constant([[1], [0], [0], [0]])
      features = {
          'x':
              input_lib.limit_epochs(
                  array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
                  num_epochs=num_epochs),
      }
      return features, labels

    def _my_metric_op(predictions, labels):
      # For the case of binary classification, the 2nd column of "predictions"
      # denotes the model predictions.
      labels = math_ops.to_float(labels)
      predictions = array_ops.strided_slice(
          predictions, [0, 1], [-1, 2], end_mask=1)
      labels = math_ops.cast(labels, predictions.dtype)
      return math_ops.reduce_sum(math_ops.multiply(predictions, labels))

    classifier = debug.DebugClassifier(
        config=run_config.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=5)
    scores = classifier.evaluate(
        input_fn=_input_fn,
        steps=5,
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
    predictions = np.array(
        list(classifier.predict_classes(input_fn=predict_input_fn)))
    self.assertEqual(
        _sklearn.accuracy_score([1, 0, 0, 0], predictions),
        scores['my_accuracy'])

    # Test the case where the 2nd element of the key is neither "classes" nor
    # "probabilities".
    with self.assertRaisesRegexp(KeyError, 'bad_type'):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=5,
          metrics={
              'bad_name':
                  MetricSpec(
                      metric_fn=metric_ops.streaming_auc,
                      prediction_key='bad_type')
          })

  def testTrainSaveLoad(self):
    """Tests that insures you can save and reload a trained model."""

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[.8], [.2], [.1]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=input_lib.limit_epochs(
                      ['en', 'fr', 'zh'], num_epochs=num_epochs),
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant([[1], [0], [0]], dtype=dtypes.int32)

    model_dir = tempfile.mkdtemp()
    classifier = debug.DebugClassifier(
        model_dir=model_dir,
        n_classes=3,
        config=run_config.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=5)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions1 = classifier.predict_classes(input_fn=predict_input_fn)
    del classifier

    classifier2 = debug.DebugClassifier(
        model_dir=model_dir,
        n_classes=3,
        config=run_config.RunConfig(tf_random_seed=1))
    predictions2 = classifier2.predict_classes(input_fn=predict_input_fn)
    self.assertEqual(list(predictions1), list(predictions2))

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
    feature_columns = [
        feature_column.real_valued_column('age'),
        feature_column.embedding_column(language, dimension=1)
    ]

    classifier = debug.DebugClassifier(
        config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(input_fn=input_fn, steps=5)

    def default_input_fn(unused_estimator, examples):
      return feature_column_ops.parse_feature_columns_from_examples(
          examples, feature_columns)

    export_dir = tempfile.mkdtemp()
    classifier.export(export_dir, input_fn=default_input_fn)


class DebugRegressorTest(test.TestCase):

  def setUp(self):
    np.random.seed(100)
    self.features = np.random.rand(NUM_EXAMPLES, 5)
    self.targets = np.random.rand(NUM_EXAMPLES, LABEL_DIMENSION)

  def testPredictScores(self):
    """Tests that DebugRegressor outputs the mean target."""
    (train_features, train_labels), (test_features,
                                     test_labels) = _train_test_split(
                                         [self.features, self.targets])
    mean_target = np.mean(train_labels, 0)
    expected_prediction = np.vstack(
        [mean_target for _ in range(test_labels.shape[0])])

    classifier = debug.DebugRegressor(label_dimension=LABEL_DIMENSION)
    classifier.fit(
        input_fn=_input_fn_builder(train_features, train_labels), steps=50)

    pred = classifier.predict_scores(
        input_fn=_input_fn_builder(test_features, None))
    self.assertAllClose(expected_prediction, np.vstack(pred), atol=0.1)

  def testExperimentIntegration(self):
    exp = experiment.Experiment(
        estimator=debug.DebugRegressor(),
        train_input_fn=test_data.iris_input_logistic_fn,
        eval_input_fn=test_data.iris_input_logistic_fn)
    exp.test()

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(self, debug.DebugRegressor)

  def testRegression_MatrixData(self):
    """Tests regression using matrix data as input."""
    regressor = debug.DebugRegressor(
        config=run_config.RunConfig(tf_random_seed=1))
    input_fn = test_data.iris_input_logistic_fn
    regressor.fit(input_fn=input_fn, steps=200)
    scores = regressor.evaluate(input_fn=input_fn, steps=1)
    self.assertIn('loss', scores)

  def testRegression_MatrixData_Labels1D(self):
    """Same as the last test, but label shape is [100] instead of [100, 1]."""

    def _input_fn():
      iris = test_data.prepare_iris_data_for_logistic_regression()
      return {
          'feature': constant_op.constant(iris.data, dtype=dtypes.float32)
      }, constant_op.constant(
          iris.target, shape=[100], dtype=dtypes.int32)

    regressor = debug.DebugRegressor(
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=200)
    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)

  def testRegression_NpMatrixData(self):
    """Tests binary classification using numpy matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    train_x = iris.data
    train_y = iris.target
    regressor = debug.DebugRegressor(
        config=run_config.RunConfig(tf_random_seed=1))
    regressor.fit(x=train_x, y=train_y, steps=200)
    scores = regressor.evaluate(x=train_x, y=train_y, steps=1)
    self.assertIn('loss', scores)

  def testRegression_TensorData(self):
    """Tests regression using tensor data as input."""

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[.8], [.15], [0.]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=input_lib.limit_epochs(
                      ['en', 'fr', 'zh'], num_epochs=num_epochs),
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant([1., 0., 0.2], dtype=dtypes.float32)

    regressor = debug.DebugRegressor(
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=200)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The algorithm should learn (y = 0.25).
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
      }
      return features, labels

    regressor = debug.DebugRegressor(
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=5)
    scores = regressor.evaluate(input_fn=_input_fn_train, steps=1)
    self.assertIn('loss', scores)

  def testLossWithWeights(self):
    """Tests loss calculation with weights."""

    def _input_fn_train():
      # 4 rows with equal weight, one of them (y = x), three of them (y=Not(x))
      # The algorithm should learn (y = 0.25).
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    def _input_fn_eval():
      # 4 rows, with different weights.
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[7.], [1.], [1.], [1.]])
      }
      return features, labels

    regressor = debug.DebugRegressor(
        weight_column_name='w', config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=5)
    scores = regressor.evaluate(input_fn=_input_fn_eval, steps=1)
    self.assertIn('loss', scores)

  def testTrainWithWeights(self):
    """Tests training with given weight column."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # First row has more weight than others. Model should fit (y=x) better
      # than (y=Not(x)) due to the relative higher weight of the first row.
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[100.], [3.], [2.], [2.]])
      }
      return features, labels

    def _input_fn_eval():
      # Create 4 rows (y = x)
      labels = constant_op.constant([[1.], [1.], [1.], [1.]])
      features = {
          'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    regressor = debug.DebugRegressor(
        weight_column_name='w', config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=5)
    scores = regressor.evaluate(input_fn=_input_fn_eval, steps=1)
    self.assertIn('loss', scores)

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""

    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x':
              input_lib.limit_epochs(
                  array_ops.ones(shape=[4, 1], dtype=dtypes.float32),
                  num_epochs=num_epochs),
      }
      return features, labels

    def _my_metric_op(predictions, labels):
      return math_ops.reduce_sum(math_ops.multiply(predictions, labels))

    regressor = debug.DebugRegressor(
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
                MetricSpec(metric_fn=_my_metric_op, prediction_key='scores')
        })
    self.assertIn('loss', set(scores.keys()))
    self.assertIn('my_error', set(scores.keys()))
    self.assertIn('my_metric', set(scores.keys()))
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = np.array(
        list(regressor.predict_scores(input_fn=predict_input_fn)))
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

  def testTrainSaveLoad(self):
    """Tests that insures you can save and reload a trained model."""

    def _input_fn(num_epochs=None):
      features = {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[0.8], [0.15], [0.]]),
                  num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=input_lib.limit_epochs(
                      ['en', 'fr', 'zh'], num_epochs=num_epochs),
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant([1., 0., 0.2], dtype=dtypes.float32)

    model_dir = tempfile.mkdtemp()
    regressor = debug.DebugRegressor(
        model_dir=model_dir, config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=5)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(regressor.predict_scores(input_fn=predict_input_fn))
    del regressor

    regressor2 = debug.DebugRegressor(
        model_dir=model_dir, config=run_config.RunConfig(tf_random_seed=1))
    predictions2 = list(regressor2.predict_scores(input_fn=predict_input_fn))
    self.assertAllClose(predictions, predictions2)


if __name__ == '__main__':
  test.main()
