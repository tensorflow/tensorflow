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

# pylint: disable=g-import-not-at-top
try:
  from sklearn.cross_validation import cross_val_score
  HAS_SKLEARN = True
except ImportError:
  HAS_SKLEARN = False


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


def _input_fn(num_epochs=None):
  features = {
      'age': tf.train.limit_epochs(tf.constant([[50], [20], [10]]),
                                   num_epochs=num_epochs),
      'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                  indices=[[0, 0], [0, 1], [2, 0]],
                                  shape=[3, 2])
  }
  target = tf.constant([[1], [0], [0]], dtype=tf.int32)
  return features, target


class DNNClassifierTest(tf.test.TestCase):

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

  def testWeightColumn(self):
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
    # If there is no weight column, model should learn y=Not(x). All examples in
    # eval data set are y=x. So if weight column is ignored, then accuracy
    # should be zero.
    self.assertGreater(scores['accuracy'], 0.9)

    scores_train_set = classifier.evaluate(input_fn=_input_fn_train, steps=1)
    # If weight column is ignored, then accuracy for the train set should be
    # 0.25. If it's not ignored, then it should be greater than 0.6.
    self.assertGreater(scores_train_set['accuracy'], 0.6)

  def testPredict_AsIterableFalse(self):
    """Tests predict and predict_prob methods with as_iterable=False."""
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
            'my_accuracy': tf.contrib.metrics.streaming_accuracy,
            ('my_precision', 'classes'): tf.contrib.metrics.streaming_precision,
            ('my_metric', 'probabilities'): _my_metric_op
        })
    self.assertTrue(
        set(['loss', 'my_accuracy', 'my_precision', 'my_metric'
            ]).issubset(set(scores.keys())))
    predictions = classifier.predict(input_fn=_input_fn_train)
    self.assertEqual(_sklearn.accuracy_score([1, 0, 0, 0], predictions),
                     scores['my_accuracy'])

    # Test the case where the 2nd element of the key is neither "classes" nor
    # "probabilities".
    with self.assertRaises(ValueError):
      classifier.evaluate(
          input_fn=_input_fn_train,
          steps=100,
          metrics={('bad_name', 'bad_type'): tf.contrib.metrics.streaming_auc})

    # Test the case where the tuple of the key doesn't have 2 elements.
    with self.assertRaises(ValueError):
      classifier.evaluate(
          input_fn=_input_fn_train,
          steps=100,
          metrics={
              ('bad_length_name', 'classes', 'bad_length'):
                  tf.contrib.metrics.streaming_accuracy
          })

  def testTrainSaveLoad(self):
    """Tests that insures you can save and reload a trained model."""
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

  def testSklearnCompatibility(self):
    """Tests compatibility with sklearn"""
    if not HAS_SKLEARN:
      return
    iris = tf.contrib.learn.datasets.load_iris()

    cont_features = [
        tf.contrib.layers.real_valued_column('', dimension=4)]
    kwargs = {
        'n_classes': 3,
        'feature_columns': cont_features,
        'optimizer' : 'Adam',
        'hidden_units' : [3, 4]
    }

    classifier = tf.contrib.learn.DNNClassifier(**kwargs)

    scores = cross_val_score(
      classifier,
      iris.data[1:5],
      iris.target[1:5],
      scoring='accuracy',
      fit_params={'steps': 100}
    )
    self.assertAllClose(scores, [1, 1, 1])


class DNNRegressorTest(tf.test.TestCase):

  def testRegression(self):
    """Tests multi-class classification using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    regressor = tf.contrib.learn.DNNRegressor(feature_columns=cont_features,
                                              hidden_units=[3, 3])

    regressor.fit(input_fn=_iris_input_multiclass_fn, steps=1000)
    regressor.evaluate(input_fn=_iris_input_multiclass_fn, steps=100)


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
