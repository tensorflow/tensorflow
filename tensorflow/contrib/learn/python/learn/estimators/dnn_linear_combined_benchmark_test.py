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

"""Regression test for DNNLinearCombinedEstimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import estimator_test_utils
from tensorflow.contrib.learn.python.learn.estimators import test_data


FLAGS = tf.flags.FLAGS


# Desired training steps, reported in benchmark. Actual steps might be slightly
# more than this since supervisor training runs for a non-detrministic number of
# steps.
_ITERS = 100


_METRIC_KEYS = {
    'accuracy',
    'auc',
    'accuracy/threshold_0.500000_mean',
    'loss',
    'precision/positive_threshold_0.500000_mean',
    'recall/positive_threshold_0.500000_mean',
}


class DNNLinearCombinedClassifierBenchmark(tf.test.Benchmark):

  def _assertSingleClassMetrics(self, metrics):
    estimator_test_utils.assert_in_range(0.9, 1.0, 'auc', metrics)
    estimator_test_utils.assert_in_range(
        0.9, 1.0, 'accuracy/threshold_0.500000_mean', metrics)
    estimator_test_utils.assert_in_range(
        0.9, 1.0, 'precision/positive_threshold_0.500000_mean', metrics)
    estimator_test_utils.assert_in_range(
        0.9, 1.0, 'recall/positive_threshold_0.500000_mean', metrics)
    self._assertCommonMetrics(metrics)

  def _assertCommonMetrics(self, metrics):
    estimator_test_utils.assert_in_range(
        _ITERS, _ITERS + 5, 'global_step', metrics)
    estimator_test_utils.assert_in_range(0.9, 1.0, 'accuracy', metrics)
    estimator_test_utils.assert_in_range(0.0, 0.2, 'loss', metrics)
    self.report_benchmark(
        iters=metrics['global_step'],
        extras={k: v for k, v in metrics.items() if k in _METRIC_KEYS})

  def benchmarkMatrixData(self):
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_feature = tf.contrib.layers.real_valued_column('feature', dimension=4)
    bucketized_feature = tf.contrib.layers.bucketized_column(
        cont_feature, test_data.get_quantile_based_buckets(iris.data, 10))

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=tempfile.mkdtemp(),
        linear_feature_columns=(bucketized_feature,),
        dnn_feature_columns=(cont_feature,),
        dnn_hidden_units=(3, 3))

    input_fn = test_data.iris_input_logistic_fn
    metrics = classifier.fit(input_fn=input_fn, steps=_ITERS).evaluate(
        input_fn=input_fn, steps=100)
    self._assertSingleClassMetrics(metrics)

  def benchmarkTensorData(self):
    def _input_fn():
      iris = test_data.prepare_iris_data_for_logistic_regression()
      features = {}
      for i in range(4):
        # The following shows how to provide the Tensor data for
        # RealValuedColumns.
        features.update({
            str(i): tf.reshape(
                tf.constant(iris.data[:, i], dtype=tf.float32), (-1, 1))})
      # The following shows how to provide the SparseTensor data for
      # a SparseColumn.
      features['dummy_sparse_column'] = tf.SparseTensor(
          values=('en', 'fr', 'zh'),
          indices=((0, 0), (0, 1), (60, 0)),
          shape=(len(iris.target), 2))
      labels = tf.reshape(tf.constant(iris.target, dtype=tf.int32), (-1, 1))
      return features, labels

    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_features = [tf.contrib.layers.real_valued_column(str(i))
                     for i in range(4)]
    linear_features = [
        tf.contrib.layers.bucketized_column(
            cont_features[i],
            test_data.get_quantile_based_buckets(iris.data[:, i], 10))
        for i in range(4)
    ]
    linear_features.append(tf.contrib.layers.sparse_column_with_hash_bucket(
        'dummy_sparse_column', hash_bucket_size=100))

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=tempfile.mkdtemp(),
        linear_feature_columns=linear_features,
        dnn_feature_columns=cont_features,
        dnn_hidden_units=(3, 3))

    metrics = classifier.fit(input_fn=_input_fn, steps=_ITERS).evaluate(
        input_fn=_input_fn, steps=100)
    self._assertSingleClassMetrics(metrics)

  def benchmarkPartitionedVariables(self):
    def _input_fn():
      features = {
          'language': tf.SparseTensor(values=('en', 'fr', 'zh'),
                                      indices=((0, 0), (0, 1), (2, 0)),
                                      shape=(3, 2))
      }
      return features, tf.constant(((1), (0), (0)))

    # The given hash_bucket_size results in variables larger than the
    # default min_slice_size attribute, so the variables are partitioned.
    sparse_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=2e7)
    embedding_feature = tf.contrib.layers.embedding_column(
        sparse_feature, dimension=1)

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=tempfile.mkdtemp(),
        linear_feature_columns=(sparse_feature,),
        dnn_feature_columns=(embedding_feature,),
        dnn_hidden_units=(3, 3),
        # Because we did not start a distributed cluster, we need to pass an
        # empty ClusterSpec, otherwise the device_setter will look for
        # distributed jobs, such as "/job:ps" which are not present.
        config=tf.contrib.learn.RunConfig(
            num_ps_replicas=2, cluster_spec=tf.train.ClusterSpec({})))

    metrics = classifier.fit(input_fn=_input_fn, steps=_ITERS).evaluate(
        input_fn=_input_fn, steps=1)
    self._assertSingleClassMetrics(metrics)

  def benchmarkCustomOptimizer(self):
    iris = test_data.prepare_iris_data_for_logistic_regression()
    cont_feature = tf.contrib.layers.real_valued_column('feature', dimension=4)
    bucketized_feature = tf.contrib.layers.bucketized_column(
        cont_feature, test_data.get_quantile_based_buckets(iris.data, 10))

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=tempfile.mkdtemp(),
        linear_feature_columns=(bucketized_feature,),
        linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
        dnn_feature_columns=(cont_feature,),
        dnn_hidden_units=(3, 3),
        dnn_optimizer=tf.train.AdagradOptimizer(learning_rate=0.1))

    input_fn = test_data.iris_input_logistic_fn
    metrics = classifier.fit(input_fn=input_fn, steps=_ITERS).evaluate(
        input_fn=input_fn, steps=100)
    self._assertSingleClassMetrics(metrics)

  def benchmarkMultiClass(self):
    iris = tf.contrib.learn.datasets.load_iris()
    cont_feature = tf.contrib.layers.real_valued_column('feature', dimension=4)
    bucketized_feature = tf.contrib.layers.bucketized_column(
        cont_feature, test_data.get_quantile_based_buckets(iris.data, 10))

    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        n_classes=3,
        linear_feature_columns=(bucketized_feature,),
        dnn_feature_columns=(cont_feature,),
        dnn_hidden_units=(3, 3))

    input_fn = test_data.iris_input_multiclass_fn
    metrics = classifier.fit(input_fn=input_fn, steps=_ITERS).evaluate(
        input_fn=input_fn, steps=100)
    self._assertCommonMetrics(metrics)


if __name__ == '__main__':
  tf.test.main()
