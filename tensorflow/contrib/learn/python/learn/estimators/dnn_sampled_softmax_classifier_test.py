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
#
# ==============================================================================
"""Tests for DNNSampledSoftmaxClassifier estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import dnn_sampled_softmax_classifier
from tensorflow.python.ops import math_ops


class DNNSampledSoftmaxClassifierTest(tf.test.TestCase):

  def testMultiClass(self):
    """Tests the following.

    1. Tests fit() and evaluate() calls.
    2. Tests the use of a non default optimizer.
    3. Tests the output of get_variable_names().
    Note that the training output is not verified because it is flaky with the
    Iris dataset.
    """
    def _iris_input_fn():
      iris = tf.contrib.learn.datasets.load_iris()
      return {
          'feature': tf.constant(iris.data, dtype=tf.float32)
      }, tf.constant(iris.target, shape=[150, 1], dtype=tf.int64)

    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        n_classes=3,
        n_samples=1,
        n_labels=1,
        feature_columns=cont_features,
        hidden_units=[3, 3])

    classifier.fit(input_fn=_iris_input_fn, steps=5)
    classifier.evaluate(input_fn=_iris_input_fn, steps=1)
    var_names = classifier.get_variable_names()
    self.assertGreater(len(var_names), 6)

  def testTrainWithPartitionedVariables(self):
    """Tests the following.

    1. Tests training with partitioned variables.
    2. Test that the model actually trains.
    3. Tests the output of evaluate() and predict().
    """
    def _input_fn():
      features = {
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      target = tf.constant([[1], [0], [0]], dtype=tf.int64)
      return features, target

    # The given hash_bucket_size results in variables larger than the
    # default min_slice_size attribute, so the variables are partitioned.
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=2e7)
    embedding_features = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    classifier = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        n_classes=3,
        n_samples=2,
        n_labels=1,
        feature_columns=embedding_features,
        hidden_units=[4, 4],
        # Because we did not start a distributed cluster, we need to pass an
        # empty ClusterSpec, otherwise the device_setter will look for
        # distributed jobs, such as "/job:ps" which are not present.
        config=tf.contrib.learn.RunConfig(
            num_ps_replicas=2, cluster_spec=tf.train.ClusterSpec({}),
            tf_random_seed=5))

    # Test that the model actually trains.
    classifier.fit(input_fn=_input_fn, steps=50)
    evaluate_output = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(evaluate_output['precision_at_1'], 0.9)
    self.assertGreater(evaluate_output['recall_at_1'], 0.9)

    # Test the output of predict()
    predict_output = classifier.predict(input_fn=_input_fn)
    self.assertListEqual([1, 0, 0], list(predict_output))

  def testTrainSaveLoad(self):
    """Tests that ensure that you can save and reload a trained model."""
    def _input_fn():
      features = {
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      target = tf.constant([[1], [0], [0]], dtype=tf.int64)
      return features, target

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=10)
    embedding_features = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    model_dir = tempfile.mkdtemp()
    classifier1 = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        model_dir=model_dir,
        n_classes=3,
        n_samples=2,
        n_labels=1,
        feature_columns=embedding_features,
        hidden_units=[4, 4])

    classifier1.fit(input_fn=_input_fn, steps=1)
    predict_output1 = classifier1.predict(input_fn=_input_fn)
    del classifier1

    classifier2 = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        model_dir=model_dir,
        n_classes=3,
        n_samples=2,
        n_labels=1,
        feature_columns=embedding_features,
        hidden_units=[4, 4])

    predict_output2 = classifier2.predict(input_fn=_input_fn)
    self.assertEqual(list(predict_output1), list(predict_output2))

  def testCustomOptimizerByObject(self):
    """Tests the use of custom optimizer."""
    def _input_fn():
      features = {
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      target = tf.constant([[1], [0], [0]], dtype=tf.int64)
      return features, target

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    embedding_features = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    classifier = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        n_classes=3,
        n_samples=2,
        n_labels=1,
        feature_columns=embedding_features,
        hidden_units=[4, 4],
        optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
        config=tf.contrib.learn.RunConfig(tf_random_seed=5))

    # Test that the model actually trains.
    classifier.fit(input_fn=_input_fn, steps=50)
    evaluate_output = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(evaluate_output['precision_at_1'], 0.9)
    self.assertGreater(evaluate_output['recall_at_1'], 0.9)

    # Test the output of predict()
    predict_output = classifier.predict(input_fn=_input_fn)
    self.assertListEqual([1, 0, 0], list(predict_output))

  def testCustomOptimizerByFunction(self):
    """Tests the use of custom optimizer."""
    def _input_fn():
      features = {
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      target = tf.constant([[1], [0], [0]], dtype=tf.int64)
      return features, target
    def _optimizer_exp_decay():
      global_step = tf.contrib.framework.get_global_step()
      learning_rate = tf.train.exponential_decay(learning_rate=0.01,
                                                 global_step=global_step,
                                                 decay_steps=100,
                                                 decay_rate=0.001)
      return tf.train.AdagradOptimizer(learning_rate=learning_rate)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    embedding_features = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    classifier = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        n_classes=3,
        n_samples=2,
        n_labels=1,
        feature_columns=embedding_features,
        hidden_units=[4, 4],
        optimizer=_optimizer_exp_decay,
        config=tf.contrib.learn.RunConfig(tf_random_seed=5))

    # Test that the model actually trains.
    classifier.fit(input_fn=_input_fn, steps=50)
    evaluate_output = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(evaluate_output['precision_at_1'], 0.6)
    self.assertGreater(evaluate_output['recall_at_1'], 0.6)

  def testExport(self):
    """Tests that export model for servo works."""
    def _input_fn():
      features = {
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      target = tf.constant([[1], [0], [0]], dtype=tf.int64)
      return features, target

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=100)
    embedding_features = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    classifier = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        n_classes=3,
        n_samples=2,
        n_labels=1,
        feature_columns=embedding_features,
        hidden_units=[4, 4])

    export_dir = tempfile.mkdtemp()
    classifier.fit(input_fn=_input_fn, steps=50)
    classifier.export(export_dir)

  def testPredictAsIterable(self):
    """Tests predict() and predict_proba() call with as_iterable set to True."""
    def _input_fn(num_epochs=None):
      features = {
          'age': tf.train.limit_epochs(tf.constant([[.9], [.1], [.1]]),
                                       num_epochs=num_epochs),
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      target = tf.constant([[1], [0], [0]], dtype=tf.int64)
      return features, target

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    feature_columns = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1),
        tf.contrib.layers.real_valued_column('age')
    ]

    classifier = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        n_classes=3,
        n_samples=2,
        n_labels=1,
        feature_columns=feature_columns,
        hidden_units=[4, 4])

    classifier.fit(input_fn=_input_fn, steps=1)

    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    # Test the output of predict() and predict_proba() with as_iterable=True
    predictions = list(
        classifier.predict(input_fn=predict_input_fn, as_iterable=True))
    predictions_proba = list(
        classifier.predict_proba(input_fn=predict_input_fn, as_iterable=True))
    self.assertTrue(np.array_equal(predictions,
                                   np.argmax(predictions_proba, 1)))

  def testCustomMetrics(self):
    """Tests the use of custom metric."""
    def _input_fn():
      features = {
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      target = tf.constant([[1], [0], [0]], dtype=tf.int64)
      return features, target

    def _my_metric_op(predictions, targets):
      """Simply multiplies predictions and targets to return [1, 0 , 0]."""
      prediction_classes = math_ops.argmax(predictions, 1)
      return tf.mul(prediction_classes, tf.reshape(targets, [-1]))

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    embedding_features = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    classifier = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        n_classes=3,
        n_samples=2,
        n_labels=1,
        feature_columns=embedding_features,
        hidden_units=[4, 4],
        optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
        config=tf.contrib.learn.RunConfig(tf_random_seed=5))

    # Test that the model actually trains.
    classifier.fit(input_fn=_input_fn, steps=50)
    metrics = {('my_metric', 'probabilities'): _my_metric_op}
    evaluate_output = classifier.evaluate(input_fn=_input_fn, steps=1,
                                          metrics=metrics)
    self.assertListEqual([1, 0, 0], list(evaluate_output['my_metric']))

  def testMultiLabelTopKWithCustomMetrics(self):
    """Tests the cases where n_labels>1 top_k>1 and custom metrics on top_k."""
    def _input_fn():
      features = {
          'language': tf.SparseTensor(values=['en', 'fr', 'zh'],
                                      indices=[[0, 0], [0, 1], [2, 0]],
                                      shape=[3, 2])
      }
      target = tf.constant([[0, 1], [0, 1], [0, 1]], dtype=tf.int64)
      return features, target

    def _my_metric_op(predictions, targets):
      """Simply adds the predictions and targets."""
      return tf.add(math_ops.to_int64(predictions), targets)

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=20)
    embedding_features = [
        tf.contrib.layers.embedding_column(sparse_column, dimension=1)
    ]

    classifier = dnn_sampled_softmax_classifier._DNNSampledSoftmaxClassifier(
        n_classes=3,
        n_samples=2,
        n_labels=2,
        top_k=2,
        feature_columns=embedding_features,
        hidden_units=[4, 4],
        optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
        config=tf.contrib.learn.RunConfig(tf_random_seed=5))

    classifier.fit(input_fn=_input_fn, steps=50)
    # evaluate() without custom metrics.
    evaluate_output = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(evaluate_output['precision_at_1'], 0.4)
    self.assertGreater(evaluate_output['recall_at_1'], 0.4)
    self.assertGreater(evaluate_output['precision_at_2'], 0.4)
    self.assertGreater(evaluate_output['recall_at_2'], 0.4)

    # evaluate() with custom metrics.
    metrics = {('my_metric', 'top_k'): _my_metric_op}
    evaluate_output = classifier.evaluate(input_fn=_input_fn, steps=1,
                                          metrics=metrics)
    # This test's output is flaky so just testing that 'my_metric' is indeed
    # part of the evaluate_output.
    self.assertTrue('my_metric' in evaluate_output)

    # predict() with top_k.
    predict_output = classifier.predict(input_fn=_input_fn, get_top_k=True)
    self.assertListEqual([3, 2], list(predict_output.shape))
    # TODO(dnivara): Setup this test such that it is not flaky and predict() and
    # evaluate() outputs can be tested.

if __name__ == '__main__':
  tf.test.main()
