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
import json
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import dnn
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators import estimator_test_utils
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import test_data
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.python.ops import math_ops


class EmbeddingMultiplierTest(tf.test.TestCase):
  """dnn_model_fn tests."""

  def testRaisesNonEmbeddingColumn(self):
    one_hot_language = tf.contrib.layers.one_hot_column(
        tf.contrib.layers.sparse_column_with_hash_bucket('language', 10))

    params = {
        'feature_columns': [one_hot_language],
        'head': head_lib._multi_class_head(2),
        'hidden_units': [1],
        # Set lr mult to 0. to keep embeddings constant.
        'embedding_lr_multipliers': {
            one_hot_language: 0.0
        },
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
      dnn._dnn_model_fn(features, labels,
                        tf.contrib.learn.ModeKeys.TRAIN, params)

  def testMultipliesGradient(self):
    embedding_language = tf.contrib.layers.embedding_column(
        tf.contrib.layers.sparse_column_with_hash_bucket('language', 10),
        dimension=1, initializer=tf.constant_initializer(0.1))
    embedding_wire = tf.contrib.layers.embedding_column(
        tf.contrib.layers.sparse_column_with_hash_bucket('wire', 10),
        dimension=1, initializer=tf.constant_initializer(0.1))

    params = {
        'feature_columns': [embedding_language, embedding_wire],
        'head': head_lib._multi_class_head(2),
        'hidden_units': [1],
        # Set lr mult to 0. to keep embeddings constant.
        'embedding_lr_multipliers': {
            embedding_language: 0.0
        },
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
    model_ops = dnn._dnn_model_fn(features, labels,
                                  tf.contrib.learn.ModeKeys.TRAIN, params)
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


class DNNClassifierTest(tf.test.TestCase):

  def _assertInRange(self, expected_min, expected_max, actual):
    self.assertLessEqual(expected_min, actual)
    self.assertGreaterEqual(expected_max, actual)

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(
        self, tf.contrib.learn.DNNClassifier)

  def testEmbeddingMultiplier(self):
    embedding_language = tf.contrib.layers.embedding_column(
        tf.contrib.layers.sparse_column_with_hash_bucket('language', 10),
        dimension=1, initializer=tf.constant_initializer(0.1))
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=[embedding_language],
        hidden_units=[3, 3],
        embedding_lr_multipliers={embedding_language: 0.8})
    self.assertEqual(
        {embedding_language: 0.8},
        classifier._estimator.params['embedding_lr_multipliers'])

  def testLogisticRegression_MatrixData(self):
    """Tests binary classification using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=cont_features,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

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
          'feature': tf.constant(iris.data, dtype=tf.float32)
      }, tf.constant(iris.target, shape=[100], dtype=tf.int32)

    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=cont_features,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)

  def testLogisticRegression_NpMatrixData(self):
    """Tests binary classification using numpy matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    train_x = iris.data
    train_y = iris.target
    feature_columns = [tf.contrib.layers.real_valued_column('', dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(x=train_x, y=train_y, steps=5)
    scores = classifier.evaluate(x=train_x, y=train_y, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])

  def testLogisticRegression_TensorData(self):
    """Tests binary classification using tensor data as input."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[.8], [0.2], [.1]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
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
        hidden_units=[10, 10],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=50)

    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(
        classifier.predict(input_fn=predict_input_fn, as_iterable=True))
    self.assertListEqual(predictions, [1, 0, 0])

  def testLogisticRegression_FloatLabel(self):
    """Tests binary classification with float labels."""
    def _input_fn_float_label(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[50], [20], [10]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
      }
      labels = tf.constant([[0.8], [0.], [0.2]], dtype=tf.float32)
      return features, labels

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

    classifier.fit(input_fn=_input_fn_float_label, steps=50)

    predict_input_fn = functools.partial(_input_fn_float_label, num_epochs=1)
    predictions_proba = list(
        classifier.predict_proba(input_fn=predict_input_fn, as_iterable=True))
    self.assertEqual(3, len(predictions_proba))
    predictions = list(
        classifier.predict(input_fn=predict_input_fn, as_iterable=True))
    self.assertEqual(3, len(predictions))
    for b in range(3):
      self.assertEqual(2, len(predictions_proba[b]))
      for i in range(2):
        self._assertInRange(0.0, 1.0, predictions_proba[b][i])
      self.assertTrue(predictions[b] in (0, 1))

  def testMultiClass_MatrixData(self):
    """Tests multi-class classification using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=cont_features,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    input_fn = test_data.iris_input_multiclass_fn
    classifier.fit(input_fn=input_fn, steps=200)
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)

  def testMultiClass_MatrixData_Labels1D(self):
    """Same as the last test, but label shape is [150] instead of [150, 1]."""
    def _input_fn():
      iris = tf.contrib.learn.datasets.load_iris()
      return {
          'feature': tf.constant(iris.data, dtype=tf.float32)
      }, tf.constant(iris.target, shape=[150], dtype=tf.int32)

    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=cont_features,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=200)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])

  def testMultiClass_NpMatrixData(self):
    """Tests multi-class classification using numpy matrix data as input."""
    iris = tf.contrib.learn.datasets.load_iris()
    train_x = iris.data
    train_y = iris.target
    feature_columns = [tf.contrib.layers.real_valued_column('', dimension=4)]
    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(x=train_x, y=train_y, steps=200)
    scores = classifier.evaluate(x=train_x, y=train_y, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The logistic prediction should be (y = 0.25).
      labels = tf.constant([[1], [0], [0], [0]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
      }
      return features, labels

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=2,
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn_train, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn_train, steps=1)
    self.assertIn('loss', scores)

  def testLossWithWeights(self):
    """Tests loss calculation with weights."""

    def _input_fn_train():
      # 4 rows with equal weight, one of them (y = x), three of them (y=Not(x))
      # The logistic prediction should be (y = 0.25).
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

    classifier = tf.contrib.learn.DNNClassifier(
        weight_column_name='w',
        n_classes=2,
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn_train, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    self.assertIn('loss', scores)

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
      # Create 4 rows (y = x)
      labels = tf.constant([[1], [1], [1], [1]])
      features = {
          'x': tf.ones(shape=[4, 1], dtype=tf.float32),
          'w': tf.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    classifier = tf.contrib.learn.DNNClassifier(
        weight_column_name='w',
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn_train, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])

  def testPredict_AsIterableFalse(self):
    """Tests predict and predict_prob methods with as_iterable=False."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[.8], [.2], [.1]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
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
        hidden_units=[10, 10],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=100)

    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)
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
          'age': tf.train.limit_epochs(
              tf.constant([[.8], [.2], [.1]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
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
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)
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
    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = tf.constant([[1], [0], [0], [0]])
      features = {
          'x': tf.train.limit_epochs(
              tf.ones(shape=[4, 1], dtype=tf.float32), num_epochs=num_epochs),
      }
      return features, labels

    def _my_metric_op(predictions, labels):
      # For the case of binary classification, the 2nd column of "predictions"
      # denotes the model predictions.
      labels = tf.to_float(labels)
      predictions = tf.slice(predictions, [0, 1], [-1, 1])
      labels = math_ops.cast(labels, predictions.dtype)
      return tf.reduce_sum(tf.mul(predictions, labels))

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    classifier.fit(input_fn=_input_fn, steps=5)
    scores = classifier.evaluate(
        input_fn=_input_fn,
        steps=5,
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
    predictions = np.array(list(classifier.predict(input_fn=predict_input_fn)))
    self.assertEqual(_sklearn.accuracy_score([1, 0, 0, 0], predictions),
                     scores['my_accuracy'])

    # Test the case where the 2nd element of the key is neither "classes" nor
    # "probabilities".
    with self.assertRaisesRegexp(KeyError, 'bad_type'):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=5,
          metrics={
              'bad_name': MetricSpec(
                  metric_fn=tf.contrib.metrics.streaming_auc,
                  prediction_key='bad_type')})

  def testTrainSaveLoad(self):
    """Tests that insures you can save and reload a trained model."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[.8], [.2], [.1]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
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

    classifier.fit(input_fn=_input_fn, steps=5)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions1 = classifier.predict(input_fn=predict_input_fn)
    del classifier

    classifier2 = tf.contrib.learn.DNNClassifier(
        model_dir=model_dir,
        n_classes=3,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    predictions2 = classifier2.predict(input_fn=predict_input_fn)
    self.assertEqual(list(predictions1), list(predictions2))

  def testTrainWithPartitionedVariables(self):
    """Tests training with partitioned variables."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[.8], [.2], [.1]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
      }
      return features, tf.constant([[1], [0], [0]], dtype=tf.int32)

    # The given hash_bucket_size results in variables larger than the
    # default min_slice_size attribute, so the variables are partitioned.
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=2e7)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

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

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=config)

    classifier.fit(input_fn=_input_fn, steps=5)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)

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
    feature_columns = [
        tf.contrib.layers.real_valued_column('age'),
        tf.contrib.layers.embedding_column(language, dimension=1)
    ]

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[3, 3])
    classifier.fit(input_fn=input_fn, steps=5)

    export_dir = tempfile.mkdtemp()
    classifier.export(export_dir)

  def testEnableCenteredBias(self):
    """Tests that we can enable centered bias."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=3,
        feature_columns=cont_features,
        hidden_units=[3, 3],
        enable_centered_bias=True,
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    input_fn = test_data.iris_input_multiclass_fn
    classifier.fit(input_fn=input_fn, steps=5)
    self.assertIn('centered_bias_weight', classifier.get_variable_names())
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)

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

    input_fn = test_data.iris_input_multiclass_fn
    classifier.fit(input_fn=input_fn, steps=5)
    self.assertNotIn('centered_bias_weight', classifier.get_variable_names())
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self._assertInRange(0.0, 1.0, scores['accuracy'])
    self.assertIn('loss', scores)


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

    input_fn = test_data.iris_input_logistic_fn
    regressor.fit(input_fn=input_fn, steps=200)
    scores = regressor.evaluate(input_fn=input_fn, steps=1)
    self.assertIn('loss', scores)

  def testRegression_MatrixData_Labels1D(self):
    """Same as the last test, but label shape is [100] instead of [100, 1]."""
    def _input_fn():
      iris = test_data.prepare_iris_data_for_logistic_regression()
      return {
          'feature': tf.constant(iris.data, dtype=tf.float32)
      }, tf.constant(iris.target, shape=[100], dtype=tf.int32)

    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=cont_features,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=200)
    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)

  def testRegression_NpMatrixData(self):
    """Tests binary classification using numpy matrix data as input."""
    iris = test_data.prepare_iris_data_for_logistic_regression()
    train_x = iris.data
    train_y = iris.target
    feature_columns = [tf.contrib.layers.real_valued_column('', dimension=4)]
    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(x=train_x, y=train_y, steps=200)
    scores = regressor.evaluate(x=train_x, y=train_y, steps=1)
    self.assertIn('loss', scores)

  def testRegression_TensorData(self):
    """Tests regression using tensor data as input."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[.8], [.15], [0.]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
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
    self.assertIn('loss', scores)

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

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=5)
    scores = regressor.evaluate(input_fn=_input_fn_train, steps=1)
    self.assertIn('loss', scores)

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

    regressor = tf.contrib.learn.DNNRegressor(
        weight_column_name='w',
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=5)
    scores = regressor.evaluate(input_fn=_input_fn_eval, steps=1)
    self.assertIn('loss', scores)

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

    regressor = tf.contrib.learn.DNNRegressor(
        weight_column_name='w',
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=5)
    scores = regressor.evaluate(input_fn=_input_fn_eval, steps=1)
    self.assertIn('loss', scores)

  def testPredict_AsIterableFalse(self):
    """Tests predict method with as_iterable=False."""
    labels = [1., 0., 0.2]
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[0.8], [0.15], [0.]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
      }
      return features, tf.constant(labels, dtype=tf.float32)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=200)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)
    predictions = regressor.predict(input_fn=_input_fn, as_iterable=False)
    self.assertAllClose(labels, predictions, atol=0.2)

  def testPredict_AsIterable(self):
    """Tests predict method with as_iterable=True."""
    labels = [1., 0., 0.2]
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[0.8], [0.15], [0.]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
      }
      return features, tf.constant(labels, dtype=tf.float32)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=200)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(
        regressor.predict(input_fn=predict_input_fn, as_iterable=True))
    self.assertAllClose(labels, predictions, atol=0.2)

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""
    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.train.limit_epochs(
              tf.ones(shape=[4, 1], dtype=tf.float32), num_epochs=num_epochs),
      }
      return features, labels

    def _my_metric_op(predictions, labels):
      return tf.reduce_sum(tf.mul(predictions, labels))

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=5)
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
          metrics={('my_error', 'predictions'):
                   tf.contrib.metrics.streaming_mean_squared_error})

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
    """Tests custom evaluation metrics that use MetricSpec."""
    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = tf.constant([[1.], [0.], [0.], [0.]])
      features = {
          'x': tf.train.limit_epochs(
              tf.ones(shape=[4, 1], dtype=tf.float32), num_epochs=num_epochs),
      }
      return features, labels

    def _my_metric_op(predictions, labels):
      return tf.reduce_sum(tf.mul(predictions, labels))

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=[tf.contrib.layers.real_valued_column('x')],
        hidden_units=[3, 3],
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

  def testTrainSaveLoad(self):
    """Tests that insures you can save and reload a trained model."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[0.8], [0.15], [0.]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
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
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=5)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(regressor.predict(input_fn=predict_input_fn))
    del regressor

    regressor2 = tf.contrib.learn.DNNRegressor(
        model_dir=model_dir,
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))
    predictions2 = list(regressor2.predict(input_fn=predict_input_fn))
    self.assertAllClose(predictions, predictions2)

  def testTrainWithPartitionedVariables(self):
    """Tests training with partitioned variables."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[0.8], [0.15], [0.]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
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

    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[3, 3],
        config=config)

    regressor.fit(input_fn=_input_fn, steps=5)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)

  def testEnableCenteredBias(self):
    """Tests that we can enable centered bias."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[0.8], [0.15], [0.]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
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
        enable_centered_bias=True,
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=5)
    self.assertIn('centered_bias_weight', regressor.get_variable_names())

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)

  def testDisableCenteredBias(self):
    """Tests that we can disable centered bias."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(
              tf.constant([[0.8], [0.15], [0.]]), num_epochs=num_epochs),
          'language': tf.SparseTensor(
              values=tf.train.limit_epochs(
                  ['en', 'fr', 'zh'], num_epochs=num_epochs),
              indices=[[0, 0], [0, 1], [2, 0]],
              dense_shape=[3, 2])
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
        config=tf.contrib.learn.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=5)
    self.assertNotIn('centered_bias_weight', regressor.get_variable_names())

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertIn('loss', scores)


def boston_input_fn():
  boston = tf.contrib.learn.datasets.load_boston()
  features = tf.cast(tf.reshape(tf.constant(boston.data), [-1, 13]), tf.float32)
  labels = tf.cast(tf.reshape(tf.constant(boston.target), [-1, 1]), tf.float32)
  return features, labels


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
