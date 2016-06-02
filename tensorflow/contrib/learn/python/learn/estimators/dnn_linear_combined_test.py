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

import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import _sklearn


def _get_quantile_based_buckets(feature_values, num_buckets):
  quantiles = np.percentile(
      np.array(feature_values), ([100 * (i + 1.) / (num_buckets + 1.)
                                  for i in range(num_buckets)]))
  return list(quantiles)


def _prepare_iris_data_for_logistic_regression():
  # Converts iris data to a logistic regression problem.
  iris = tf.contrib.learn.datasets.load_iris()
  ids = np.where((iris.target == 0) | (iris.target == 1))
  iris = tf.contrib.learn.datasets.base.Dataset(data=iris.data[ids],
                                                target=iris.target[ids])
  return iris


def _iris_input_fn():
  iris = _prepare_iris_data_for_logistic_regression()
  return {
      'feature': tf.constant(iris.data, dtype=tf.float32)
  }, tf.constant(iris.target, shape=[100, 1], dtype=tf.int32)


class DNNLinearCombinedClassifierTest(tf.test.TestCase):

  def testLogisticRegression_MatrixData(self):
    """Tests binary classification using matrix data as input."""
    iris = _prepare_iris_data_for_logistic_regression()
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]
    bucketized_feature = [tf.contrib.layers.bucketized_column(
        cont_features[0], _get_quantile_based_buckets(iris.data, 10))]

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=bucketized_feature,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_iris_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_iris_input_fn, steps=100)
    self.assertGreater(scores['accuracy'], 0.9)

  def testLogisticRegression_TensorData(self):
    """Tests binary classification using Tensor data as input."""
    def _input_fn():
      iris = _prepare_iris_data_for_logistic_regression()
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
          shape=[len(iris.target), 2])
      target = tf.reshape(tf.constant(iris.target, dtype=tf.int32), [-1, 1])
      return features, target

    iris = _prepare_iris_data_for_logistic_regression()
    cont_features = [tf.contrib.layers.real_valued_column(str(i))
                     for i in range(4)]
    linear_features = [
        tf.contrib.layers.bucketized_column(
            cont_features[i], _get_quantile_based_buckets(iris.data[:, str(i)],
                                                          10)) for i in range(4)
    ]
    linear_features.append(tf.contrib.layers.sparse_column_with_hash_bucket(
        'dummy_sparse_column', hash_bucket_size=100))

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=linear_features,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn, steps=100)
    self.assertGreater(scores['accuracy'], 0.9)

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
            cont_features[0], _get_quantile_based_buckets(iris.data, 10))]

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        n_classes=3,
        linear_feature_columns=bucketized_features,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_iris_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_iris_input_fn, steps=100)
    self.assertGreater(scores['accuracy'], 0.9)

  def testWeightColumn(self):
    """Tests weight column."""

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

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        weight_column_name='w',
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_eval,
                                 steps=100)
    # If there is no weight column, model should learn y=Not(x). All examples in
    # eval data set are y=x. So if weight column is ignored, then accuracy
    # should be zero.
    self.assertGreater(scores['accuracy'], 0.9)

  def testEvaluationShouldUseWeightColumn(self):
    """Tests weight column in evaluation."""

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

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        weight_column_name='w',
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_train, steps=100)
    # If weight column is ignored, then accuracy should be 0.25. If it's not
    # ignored, then it should be greater than 0.6.
    self.assertGreater(scores['accuracy'], 0.6)

  def testCustomOptimizerByObject(self):
    """Tests binary classification using matrix data as input."""
    iris = _prepare_iris_data_for_logistic_regression()
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]
    bucketized_features = [
        tf.contrib.layers.bucketized_column(
            cont_features[0], _get_quantile_based_buckets(iris.data, 10))]

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=bucketized_features,
        linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3],
        dnn_optimizer=tf.train.AdagradOptimizer(learning_rate=0.1))

    classifier.fit(input_fn=_iris_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_iris_input_fn, steps=100)
    self.assertGreater(scores['accuracy'], 0.9)

  def testCustomOptimizerByString(self):
    """Tests binary classification using matrix data as input."""
    iris = _prepare_iris_data_for_logistic_regression()
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]
    bucketized_features = [
        tf.contrib.layers.bucketized_column(
            cont_features[0], _get_quantile_based_buckets(iris.data, 10))]

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=bucketized_features,
        linear_optimizer='Ftrl',
        dnn_feature_columns=cont_features,
        dnn_hidden_units=[3, 3],
        dnn_optimizer='Adagrad')

    classifier.fit(input_fn=_iris_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_iris_input_fn, steps=100)
    self.assertGreater(scores['accuracy'], 0.9)

  def testPredict(self):
    """Tests weight column in evaluation."""
    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      target = tf.constant([[1], [0], [0], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, target

    def _input_fn_predict():
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=100)
    probs = classifier.predict_proba(input_fn=_input_fn_predict)
    self.assertAllClose([[0.75, 0.25]] * 4, probs, 0.01)
    classes = classifier.predict(input_fn=_input_fn_predict)
    self.assertListEqual([0] * 4, list(classes))

  def testCustomMetrics(self):
    """Tests weight column in evaluation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      target = tf.constant([[1], [0], [0], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, target

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(
        input_fn=_input_fn_train,
        steps=100,
        metrics={
            'my_accuracy': tf.contrib.metrics.streaming_accuracy,
            'my_precision': tf.contrib.metrics.streaming_precision
        })
    self.assertTrue(set(['loss', 'my_accuracy', 'my_precision']).issubset(set(
        scores.keys())))
    predictions = classifier.predict(input_fn=_input_fn_train)
    self.assertEqual(_sklearn.accuracy_score([1, 0, 0, 0], predictions),
                     scores['my_accuracy'])

  def testVariableQuery(self):
    """Tests bias is centered or not."""
    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      target = tf.constant([[1], [1], [1], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, target

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=500)
    var_names = classifier.get_variable_names()
    self.assertGreater(len(var_names), 3)
    for name in var_names:
      classifier.get_variable_value(name)

  def testCenteredBias(self):
    """Tests bias is centered or not."""
    def _input_fn_train():
      # Create 4 rows, three (y = x), one (y=Not(x))
      target = tf.constant([[1], [1], [1], [0]])
      features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
      return features, target

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=_input_fn_train, steps=500)
    # logodds(0.75) = 1.09861228867
    self.assertAlmostEqual(
        1.0986,
        float(classifier.get_variable_value('centered_bias_weight')[0]),
        places=2)


class DNNLinearCombinedRegressorTest(tf.test.TestCase):

  def _input_fn_train(self):
    # Create 4 rows of (y = x)
    target = tf.constant([[100.], [3.], [2.], [2.]])
    features = {'x': tf.constant([[100.], [3.], [2.], [2.]])}
    return features, target

  def testRegression(self):
    """Tests a regression problem."""
    classifier = tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3])

    classifier.fit(input_fn=self._input_fn_train, steps=100)
    classifier.evaluate(input_fn=self._input_fn_train, steps=1)

  def testRegressionContinueTraining(self):
    """Tests regression with restarting training / evaluate."""
    output_dir = tempfile.mkdtemp()
    new_estimator = lambda: tf.contrib.learn.DNNLinearCombinedRegressor(
        linear_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_feature_columns=[tf.contrib.layers.real_valued_column('x')],
        dnn_hidden_units=[3, 3], model_dir=output_dir)
    classifier = new_estimator()

    classifier.fit(input_fn=self._input_fn_train, steps=50)
    del classifier

    classifier = new_estimator()
    classifier.fit(input_fn=self._input_fn_train, steps=100)
    del classifier

    classifier = new_estimator()
    classifier.evaluate(input_fn=self._input_fn_train, steps=1)
    del classifier

    classifier = new_estimator()
    classifier.predict(input_fn=self._input_fn_train)


if __name__ == '__main__':
  tf.test.main()
