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

"""Tests for DNNEstimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator_test_utils
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.python.ops import math_ops


def _prepare_iris_data_for_logistic_regression():
  # Converts iris data to a logistic regression problem.
  iris = tf.contrib.learn.datasets.load_iris()
  ids = np.where((iris.target == 0) | (iris.target == 1))
  iris = tf.contrib.learn.datasets.base.Dataset(data=iris.data[ids],
                                                target=iris.target[ids])
  return iris


def _iris_input_logistic_fn():
  iris = _prepare_iris_data_for_logistic_regression()
  return {
      'feature': tf.constant(iris.data, dtype=tf.float32)
  }, tf.constant(iris.target, shape=[100, 1], dtype=tf.int32)


def _iris_input_multiclass_fn():
  iris = tf.contrib.learn.datasets.load_iris()
  return {
      'feature': tf.constant(iris.data, dtype=tf.float32)
  }, tf.constant(iris.target, shape=[150, 1], dtype=tf.int32)


class DNNClassifierTest(tf.test.TestCase):

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, tf.contrib.learn.DNNClassifier)

  def testLogisticRegression_MatrixData(self):
    """Tests binary classification using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=cont_features,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_iris_input_logistic_fn, steps=100)
    scores = classifier.evaluate(input_fn=_iris_input_logistic_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)
    self.assertLess(scores['loss'], 0.3)

  def testLogisticRegression_TensorData(self):
    """Tests binary classification using tensor data as input."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[.8], [.2], [.1]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant([[1], [0], [0]], dtype=tf.int32)

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(language_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=2,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=100)

    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)
    self.assertLess(scores['loss'], 0.3)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(
        classifier.predict(input_fn=predict_input_fn, as_iterable=True))
    self.assertListEqual(predictions, [1, 0, 0])

  def testLogisticRegression_FloatLabel(self):
    """Tests binary classification with float labels."""
    def _input_fn_float_label(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[50], [20], [10]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      target = tf.constant([[0.8], [0.], [0.2]], dtype=tf.float32)
      return features, target

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(language_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=2,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn_float_label, steps=1000)

    predict_input_fn = functools.partial(_input_fn_float_label, num_epochs=1)
    predictions_proba = list(
        classifier.predict_proba(input_fn=predict_input_fn, as_iterable=True))
    # Prediction probabilities mirror the target column, which proves that the
    # classifier learns from float input.
    self.assertAllClose(
        predictions_proba, [[0.2, 0.8], [1., 0.], [0.8, 0.2]], atol=0.05)
    predictions = list(
        classifier.predict(input_fn=predict_input_fn, as_iterable=True))
    self.assertListEqual(predictions, [1, 0, 0])

  def testMultiClass_MatrixData(self):
    """Tests multi-class classification using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=cont_features,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_iris_input_multiclass_fn, steps=200)
    self.assertTrue('centered_bias_weight' in classifier.get_variable_names())
    scores = classifier.evaluate(input_fn=_iris_input_multiclass_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.8)
    self.assertLess(scores['loss'], 0.3)

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The logistic prediction should be (y = 0.25).
      target = tf.constant([[1], [0], [0], [0]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
      }
      return features, target

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=2,
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_train, steps=1)
    # Cross entropy = -0.25*log(0.25)-0.75*log(0.75) = 0.562
    self.assertAlmostEqual(scores['loss'], 0.562, delta=0.1)

  def testLossWithWeights(self):
    """Tests loss calculation with weights."""

    def _input_fn_train():
      # 4 rows with equal weight, one of them (y = x), three of them (y=Not(x))
      # The logistic prediction should be (y = 0.25).
      target = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[1.], [1.], [1.], [1.]])
      }
      return features, target

    def _input_fn_eval():
      # 4 rows, with different weights.
      target = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[7.], [1.], [1.], [1.]])
      }
      return features, target

    classifier = tf.contrib.learn.DNNClassifier(
        weight_column_name='w',
        n_classes=2,
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    # Weighted cross entropy = (-7*log(0.25)-3*log(0.75))/10 = 1.06
    self.assertAlmostEqual(scores['loss'], 1.06, delta=0.1)

  def testTrainWithWeights(self):
    """Tests training with given weight column."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # First row has more weight than others. Model should fit (y=x) better
      # than (y=Not(x)) due to the relative higher weight of the first row.
      target = tf.constant([[1], [0], [0], [0]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[100.], [3.], [2.], [2.]])
      }
      return features, target

    def _input_fn_eval():
      # Create 4 rows (y = x)
      target = tf.constant([[1], [1], [1], [1]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[1.], [1.], [1.], [1.]])
      }
      return features, target

    classifier = tf.contrib.learn.DNNClassifier(
        weight_column_name='w',
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))

    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    # The model should learn (y = x) because of the weights, so the accuracy
    # should be close to 1.
    self.assertGreater(scores['accuracy'], 0.9)

  def testPredict_AsIterableFalse(self):
    """Tests predict and predict_prob methods with as_iterable=False."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[.8], [.2], [.1]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant([[1], [0], [0]], dtype=tf.int32)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))

    classifier.fit(input_fn=_input_fn, steps=100)

    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)
    self.assertLess(scores['loss'], 0.3)
    predictions = classifier.predict(input_fn=_input_fn, as_iterable=False)
    self.assertListEqual(list(predictions), [1, 0, 0])
    predictions = classifier.predict_proba(input_fn=_input_fn,
                                           as_iterable=False)
    self.assertAllClose(
        predictions, [[0., 1., 0.], [1., 0., 0.], [1., 0., 0.]], atol=0.1)

  def testPredict_AsIterable(self):
    """Tests predict and predict_prob methods with as_iterable=True."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[.8], [.2], [.1]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant([[1], [0], [0]], dtype=tf.int32)

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(language_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=200)

    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)
    self.assertLess(scores['loss'], 0.3)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(
        classifier.predict(input_fn=predict_input_fn, as_iterable=True))
    self.assertListEqual(predictions, [1, 0, 0])
    predictions = list(
        classifier.predict_proba(input_fn=predict_input_fn, as_iterable=True))
    self.assertAllClose(
        predictions, [[0., 1., 0.], [1., 0., 0.], [1., 0., 0.]], atol=0.3)

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""
    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      target = tf.constant([[1], [0], [0], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, target

    def _my_metric_op(predictions, targets):
      # For the case of binary classification, the 2nd column of "predictions"
      # denotes the model predictions.
      predictions = tf.slice(predictions, [0, 1], [-1, 1])
      targets = math_ops.cast(targets, predictions.dtype)
      return tf.reduce_sum(tf.mul(predictions, targets))

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(
        input_fn=_input_fn_train,
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
    predictions = classifier.predict(input_fn=_input_fn_train)
    self.assertEqual(_sklearn.accuracy_score([1, 0, 0, 0], predictions),
                     scores['my_accuracy'])

    # Test the case where the 2nd element of the key is neither "classes" nor
    # "probabilities".
    with self.assertRaisesRegexp(KeyError, 'bad_type'):
      classifier.evaluate(
          input_fn=_input_fn_train,
          steps=100,
          metrics={
              'bad_name': MetricSpec(
                  metric_fn=tf.contrib.metrics.streaming_auc,
                  prediction_key='bad_type')})

  def testTrainSaveLoad(self):
    """Tests that insures you can save and reload a trained model."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[.8], [.2], [.1]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant([[1], [0], [0]], dtype=tf.int32)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    model_dir = tempfile.mkdtemp()
    classifier = tf.contrib.learn.DNNClassifier(
        model_dir=model_dir,
        n_classes=3,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=100)
    predictions1 = classifier.predict(input_fn=_input_fn)
    del classifier

    classifier2 = tf.contrib.learn.DNNClassifier(
        model_dir=model_dir,
        n_classes=3,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    predictions2 = classifier2.predict(input_fn=_input_fn)
    self.assertEqual(list(predictions1), list(predictions2))

  def testTrainWithPartitionedVariables(self):
    """Tests training with partitioned variables."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[.8], [.2], [.1]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant([[1], [0], [0]], dtype=tf.int32)

    # The given hash_bucket_size results in variables larger than the
    # default min_slice_size attribute, so the variables are partitioned.
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=2e7)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        # Because we did not start a distributed cluster, we need to pass an
        # empty ClusterSpec, otherwise the device_setter will look for
        # distributed jobs, such as "/job:ps" which are not present.
        config=tf.contrib.learn.RunConfig(
            num_ps_replicas=2, cluster_spec=tf.train.ClusterSpec({}),
            tf_random_seed=5))

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)
    self.assertLess(scores['loss'], 0.3)

  def testExport(self):
    """Tests export model for servo."""

    def input_fn():
      return {
          'age': tf.constant([1]),
          'language': tf.SparseTensor(values=['english'],
                                      indices=[[0, 0]],
                                      shape=[1, 1])
      }, tf.constant([[1]])

    language = tf.contrib.layers.sparse_column_with_hash_bucket('language', 100)
    feature_columns = [
        tf.contrib.layers.real_valued_column('age'),
        tf.contrib.layers.embedding_column(language, dimension=1)
    ]

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[3, 3])
    classifier.fit(input_fn=input_fn, steps=100)

    export_dir = tempfile.mkdtemp()
    classifier.export(export_dir)

  def testDisableCenteredBias(self):
    """Tests that we can disable centered bias."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=cont_features,
        hidden_units=[3, 3],
        enable_centered_bias=False,
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_iris_input_multiclass_fn, steps=200)
    self.assertFalse('centered_bias_weight' in classifier.get_variable_names())
    scores = classifier.evaluate(input_fn=_iris_input_multiclass_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.8)
    self.assertLess(scores['loss'], 0.3)


class DNNRegressorTest(tf.test.TestCase):

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, tf.contrib.learn.DNNRegressor)

  def testRegression_MatrixData(self):
    """Tests regression using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=cont_features,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_iris_input_logistic_fn, steps=200)
    scores = regressor.evaluate(input_fn=_iris_input_logistic_fn, steps=1)
    self.assertLess(scores['loss'], 0.3)

  def testRegression_TensorData(self):
    """Tests regression using tensor data as input."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant([1., 0., 0.2], dtype=tf.float32)

    language_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(language_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=200)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.3)

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The algorithm should learn (y = 0.25).
      target = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
      }
      return features, target

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))

    regressor.fit(input_fn=_input_fn_train, steps=100)
    scores = regressor.evaluate(input_fn=_input_fn_train, steps=1)
    # Average square loss = (0.75^2 + 3*0.25^2) / 4 = 0.1875
    self.assertAlmostEqual(scores['loss'], 0.1875, delta=0.1)

  def testLossWithWeights(self):
    """Tests loss calculation with weights."""

    def _input_fn_train():
      # 4 rows with equal weight, one of them (y = x), three of them (y=Not(x))
      # The algorithm should learn (y = 0.25).
      target = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[1.], [1.], [1.], [1.]])
      }
      return features, target

    def _input_fn_eval():
      # 4 rows, with different weights.
      target = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[7.], [1.], [1.], [1.]])
      }
      return features, target

    regressor = tf.contrib.learn.DNNRegressor(
        weight_column_name='w',
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=100)
    scores = regressor.evaluate(input_fn=_input_fn_eval, steps=1)
    # Weighted average square loss = (7*0.75^2 + 3*0.25^2) / 10 = 0.4125
    self.assertAlmostEqual(scores['loss'], 0.4125, delta=0.1)

  def testTrainWithWeights(self):
    """Tests training with given weight column."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # First row has more weight than others. Model should fit (y=x) better
      # than (y=Not(x)) due to the relative higher weight of the first row.
      target = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[100.], [3.], [2.], [2.]])
      }
      return features, target

    def _input_fn_eval():
      # Create 4 rows (y = x)
      target = tf.constant([[1.], [1.], [1.], [1.]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[1.], [1.], [1.], [1.]])
      }
      return features, target

    regressor = tf.contrib.learn.DNNRegressor(
        weight_column_name='w',
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))

    regressor.fit(input_fn=_input_fn_train, steps=100)
    scores = regressor.evaluate(input_fn=_input_fn_eval, steps=1)
    # The model should learn (y = x) because of the weights, so the loss should
    # be close to zero.
    self.assertLess(scores['loss'], 0.2)

  def testPredict_AsIterableFalse(self):
    """Tests predict method with as_iterable=False."""
    target = [1., 0., 0.2]
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant(target, dtype=tf.float32)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))

    regressor.fit(input_fn=_input_fn, steps=200)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.2)
    predictions = regressor.predict(input_fn=_input_fn, as_iterable=False)
    self.assertAllClose(predictions, target, atol=0.2)

  def testPredict_AsIterable(self):
    """Tests predict method with as_iterable=True."""
    target = [1., 0., 0.2]
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant(target, dtype=tf.float32)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))

    regressor.fit(input_fn=_input_fn, steps=200)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.2)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(
        regressor.predict(input_fn=predict_input_fn, as_iterable=True))
    self.assertAllClose(predictions, target, atol=0.2)

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""
    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      target = tf.constant([[1.], [0.], [0.], [0.]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, target

    def _my_metric_op(predictions, targets):
      return tf.reduce_sum(tf.mul(predictions, targets))

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=100)
    scores = regressor.evaluate(
        input_fn=_input_fn_train,
        steps=1,
        metrics={
            'my_error': tf.contrib.metrics.streaming_mean_squared_error,
            'my_metric': _my_metric_op
        })
    self.assertIn('loss', set(scores.keys()))
    self.assertIn('my_error', set(scores.keys()))
    self.assertIn('my_metric', set(scores.keys()))
    predictions = regressor.predict(input_fn=_input_fn_train)
    self.assertAlmostEqual(
        _sklearn.mean_squared_error(np.array([1, 0, 0, 0]), predictions),
        scores['my_error'])

    # Tests that when the key is a tuple, an error is raised.
    with self.assertRaises(TypeError):
      regressor.evaluate(
          input_fn=_input_fn_train,
          steps=1,
          metrics={('my_error', 'predictions'
                   ): tf.contrib.metrics.streaming_mean_squared_error})

  def testTrainSaveLoad(self):
    """Tests that insures you can save and reload a trained model."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant([1., 0., 0.2], dtype=tf.float32)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    model_dir = tempfile.mkdtemp()
    regressor = tf.contrib.learn.DNNRegressor(
        model_dir=model_dir,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))

    regressor.fit(input_fn=_input_fn, steps=100)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(regressor.predict(input_fn=predict_input_fn))
    del regressor

    regressor2 = tf.contrib.learn.DNNRegressor(
        model_dir=model_dir,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))
    predictions2 = list(regressor2.predict(input_fn=predict_input_fn))
    self.assertAllClose(predictions, predictions2)

  def testTrainWithPartitionedVariables(self):
    """Tests training with partitioned variables."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant([1., 0., 0.2], dtype=tf.float32)

    # The given hash_bucket_size results in variables larger than the
    # default min_slice_size attribute, so the variables are partitioned.
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=2e7)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        # Because we did not start a distributed cluster, we need to pass an
        # empty ClusterSpec, otherwise the device_setter will look for
        # distributed jobs, such as "/job:ps" which are not present.
        config=tf.contrib.learn.RunConfig(
            num_ps_replicas=2, cluster_spec=tf.train.ClusterSpec({}),
            tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.3)

  def testDisableCenteredBias(self):
    """Tests that we can disable centered bias."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[0.8], [0.15], [0.]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      return features, tf.constant([1., 0., 0.2], dtype=tf.float32)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        enable_centered_bias=False,
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))

    regressor.fit(input_fn=_input_fn, steps=200)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.3)


def boston_input_fn():
  boston = tf.contrib.learn.datasets.load_boston()
  features = tf.cast(tf.reshape(tf.constant(boston.data), [-1, 13]), tf.float32)
  target = tf.cast(tf.reshape(tf.constant(boston.target), [-1, 1]), tf.float32)
  return features, target


class FeatureColumnTest(tf.test.TestCase):

  def testTrain(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(
        boston_input_fn)
    est = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns, hidden_units=[3, 3])
    est.fit(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_input_fn, steps=1)


if __name__ == '__main__':
  tf.test.main()
