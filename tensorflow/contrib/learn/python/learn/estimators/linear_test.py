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
"""Tests for estimators.linear."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import sys
import tempfile

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.layers.python.layers import feature_column as feature_column_lib
from tensorflow.contrib.learn.python.learn import experiment
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import estimator_test_utils
from tensorflow.contrib.learn.python.learn.estimators import linear
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.estimators import test_data
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer as sdca_optimizer_lib
from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import ftrl
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import server_lib


def _prepare_iris_data_for_logistic_regression():
  # Converts iris data to a logistic regression problem.
  iris = base.load_iris()
  ids = np.where((iris.target == 0) | (iris.target == 1))
  iris = base.Dataset(data=iris.data[ids], target=iris.target[ids])
  return iris


class LinearClassifierTest(test.TestCase):

  def testExperimentIntegration(self):
    cont_features = [
        feature_column_lib.real_valued_column(
            'feature', dimension=4)
    ]

    exp = experiment.Experiment(
        estimator=linear.LinearClassifier(
            n_classes=3, feature_columns=cont_features),
        train_input_fn=test_data.iris_input_multiclass_fn,
        eval_input_fn=test_data.iris_input_multiclass_fn)
    exp.test()

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(self,
                                                   linear.LinearClassifier)

  def testTrain(self):
    """Tests that loss goes down with training."""

    def input_fn():
      return {
          'age':
              constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    age = feature_column_lib.real_valued_column('age')

    classifier = linear.LinearClassifier(feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=200)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)
    self.assertLess(loss2, 0.01)

  def testJointTrain(self):
    """Tests that loss goes down with training with joint weights."""

    def input_fn():
      return {
          'age':
              sparse_tensor.SparseTensor(
                  values=['1'], indices=[[0, 0]], dense_shape=[1, 1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    age = feature_column_lib.sparse_column_with_hash_bucket('age', 2)

    classifier = linear.LinearClassifier(
        _joint_weight=True, feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=200)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss2, loss1)
    self.assertLess(loss2, 0.01)

  def testMultiClass_MatrixData(self):
    """Tests multi-class classification using matrix data as input."""
    feature_column = feature_column_lib.real_valued_column(
        'feature', dimension=4)

    classifier = linear.LinearClassifier(
        n_classes=3, feature_columns=[feature_column])

    classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_multiclass_fn, steps=100)
    self.assertGreater(scores['accuracy'], 0.9)

  def testMultiClass_MatrixData_Labels1D(self):
    """Same as the last test, but labels shape is [150] instead of [150, 1]."""

    def _input_fn():
      iris = base.load_iris()
      return {
          'feature': constant_op.constant(
              iris.data, dtype=dtypes.float32)
      }, constant_op.constant(
          iris.target, shape=[150], dtype=dtypes.int32)

    feature_column = feature_column_lib.real_valued_column(
        'feature', dimension=4)

    classifier = linear.LinearClassifier(
        n_classes=3, feature_columns=[feature_column])

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testMultiClass_NpMatrixData(self):
    """Tests multi-class classification using numpy matrix data as input."""
    iris = base.load_iris()
    train_x = iris.data
    train_y = iris.target
    feature_column = feature_column_lib.real_valued_column('', dimension=4)
    classifier = linear.LinearClassifier(
        n_classes=3, feature_columns=[feature_column])

    classifier.fit(x=train_x, y=train_y, steps=100)
    scores = classifier.evaluate(x=train_x, y=train_y, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testLogisticRegression_MatrixData(self):
    """Tests binary classification using matrix data as input."""

    def _input_fn():
      iris = _prepare_iris_data_for_logistic_regression()
      return {
          'feature': constant_op.constant(
              iris.data, dtype=dtypes.float32)
      }, constant_op.constant(
          iris.target, shape=[100, 1], dtype=dtypes.int32)

    feature_column = feature_column_lib.real_valued_column(
        'feature', dimension=4)

    classifier = linear.LinearClassifier(feature_columns=[feature_column])

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testLogisticRegression_MatrixData_Labels1D(self):
    """Same as the last test, but labels shape is [100] instead of [100, 1]."""

    def _input_fn():
      iris = _prepare_iris_data_for_logistic_regression()
      return {
          'feature': constant_op.constant(
              iris.data, dtype=dtypes.float32)
      }, constant_op.constant(
          iris.target, shape=[100], dtype=dtypes.int32)

    feature_column = feature_column_lib.real_valued_column(
        'feature', dimension=4)

    classifier = linear.LinearClassifier(feature_columns=[feature_column])

    classifier.fit(input_fn=_input_fn, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testLogisticRegression_NpMatrixData(self):
    """Tests binary classification using numpy matrix data as input."""
    iris = _prepare_iris_data_for_logistic_regression()
    train_x = iris.data
    train_y = iris.target
    feature_columns = [feature_column_lib.real_valued_column('', dimension=4)]
    classifier = linear.LinearClassifier(feature_columns=feature_columns)

    classifier.fit(x=train_x, y=train_y, steps=100)
    scores = classifier.evaluate(x=train_x, y=train_y, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testWeightAndBiasNames(self):
    """Tests that weight and bias names haven't changed."""
    feature_column = feature_column_lib.real_valued_column(
        'feature', dimension=4)

    classifier = linear.LinearClassifier(
        n_classes=3, feature_columns=[feature_column])

    classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=100)
    self.assertEqual(4, len(classifier.weights_))
    self.assertEqual(3, len(classifier.bias_))

  def testCustomOptimizerByObject(self):
    """Tests multi-class classification using matrix data as input."""
    feature_column = feature_column_lib.real_valued_column(
        'feature', dimension=4)

    classifier = linear.LinearClassifier(
        n_classes=3,
        optimizer=ftrl.FtrlOptimizer(learning_rate=0.1),
        feature_columns=[feature_column])

    classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_multiclass_fn, steps=100)
    self.assertGreater(scores['accuracy'], 0.9)

  def testCustomOptimizerByString(self):
    """Tests multi-class classification using matrix data as input."""
    feature_column = feature_column_lib.real_valued_column(
        'feature', dimension=4)

    def _optimizer():
      return ftrl.FtrlOptimizer(learning_rate=0.1)

    classifier = linear.LinearClassifier(
        n_classes=3, optimizer=_optimizer, feature_columns=[feature_column])

    classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_multiclass_fn, steps=100)
    self.assertGreater(scores['accuracy'], 0.9)

  def testCustomOptimizerByFunction(self):
    """Tests multi-class classification using matrix data as input."""
    feature_column = feature_column_lib.real_valued_column(
        'feature', dimension=4)

    classifier = linear.LinearClassifier(
        n_classes=3, optimizer='Ftrl', feature_columns=[feature_column])

    classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=100)
    scores = classifier.evaluate(
        input_fn=test_data.iris_input_multiclass_fn, steps=100)
    self.assertGreater(scores['accuracy'], 0.9)

  def testCustomMetrics(self):
    """Tests custom evaluation metrics."""

    def _input_fn(num_epochs=None):
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      labels = constant_op.constant([[1], [0], [0], [0]], dtype=dtypes.float32)
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
      predictions = array_ops.strided_slice(
          predictions, [0, 1], [-1, 2], end_mask=1)
      return math_ops.reduce_sum(math_ops.multiply(predictions, labels))

    classifier = linear.LinearClassifier(
        feature_columns=[feature_column_lib.real_valued_column('x')])

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
    predictions = np.array(list(classifier.predict(input_fn=predict_input_fn)))
    self.assertEqual(
        _sklearn.accuracy_score([1, 0, 0, 0], predictions),
        scores['my_accuracy'])

    # Tests the case where the prediction_key is neither "classes" nor
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

    # Tests the case where the 2nd element of the key is neither "classes" nor
    # "probabilities".
    with self.assertRaises(KeyError):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=100,
          metrics={('bad_name', 'bad_type'): metric_ops.streaming_auc})

    # Tests the case where the tuple of the key doesn't have 2 elements.
    with self.assertRaises(ValueError):
      classifier.evaluate(
          input_fn=_input_fn,
          steps=100,
          metrics={
              ('bad_length_name', 'classes', 'bad_length'):
                  metric_ops.streaming_accuracy
          })

  def testLogisticFractionalLabels(self):
    """Tests logistic training with fractional labels."""

    def input_fn(num_epochs=None):
      return {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([[1], [2]]), num_epochs=num_epochs),
      }, constant_op.constant(
          [[.7], [0]], dtype=dtypes.float32)

    age = feature_column_lib.real_valued_column('age')

    classifier = linear.LinearClassifier(
        feature_columns=[age], config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(input_fn=input_fn, steps=500)

    predict_input_fn = functools.partial(input_fn, num_epochs=1)
    predictions_proba = list(
        classifier.predict_proba(input_fn=predict_input_fn))
    # Prediction probabilities mirror the labels column, which proves that the
    # classifier learns from float input.
    self.assertAllClose([[.3, .7], [1., 0.]], predictions_proba, atol=.1)

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
        feature_column_lib.sparse_column_with_hash_bucket(
            'language', hash_bucket_size=2e7)
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

    classifier = linear.LinearClassifier(
        feature_columns=sparse_features, config=config)
    classifier.fit(input_fn=_input_fn, steps=200)
    loss = classifier.evaluate(input_fn=_input_fn, steps=1)['loss']
    self.assertLess(loss, 0.07)

  def testTrainSaveLoad(self):
    """Tests that insures you can save and reload a trained model."""

    def input_fn(num_epochs=None):
      return {
          'age':
              input_lib.limit_epochs(
                  constant_op.constant([1]), num_epochs=num_epochs),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1]),
      }, constant_op.constant([[1]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    age = feature_column_lib.real_valued_column('age')

    model_dir = tempfile.mkdtemp()
    classifier = linear.LinearClassifier(
        model_dir=model_dir, feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=30)
    predict_input_fn = functools.partial(input_fn, num_epochs=1)
    out1_class = list(
        classifier.predict(
            input_fn=predict_input_fn, as_iterable=True))
    out1_proba = list(
        classifier.predict_proba(
            input_fn=predict_input_fn, as_iterable=True))
    del classifier

    classifier2 = linear.LinearClassifier(
        model_dir=model_dir, feature_columns=[age, language])
    out2_class = list(
        classifier2.predict(
            input_fn=predict_input_fn, as_iterable=True))
    out2_proba = list(
        classifier2.predict_proba(
            input_fn=predict_input_fn, as_iterable=True))
    self.assertTrue(np.array_equal(out1_class, out2_class))
    self.assertTrue(np.array_equal(out1_proba, out2_proba))

  def testWeightColumn(self):
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
      # Create 4 rows (y = x)
      labels = constant_op.constant([[1], [1], [1], [1]])
      features = {
          'x': array_ops.ones(
              shape=[4, 1], dtype=dtypes.float32),
          'w': constant_op.constant([[1.], [1.], [1.], [1.]])
      }
      return features, labels

    classifier = linear.LinearClassifier(
        weight_column_name='w',
        feature_columns=[feature_column_lib.real_valued_column('x')],
        config=run_config.RunConfig(tf_random_seed=3))

    classifier.fit(input_fn=_input_fn_train, steps=100)
    scores = classifier.evaluate(input_fn=_input_fn_eval, steps=1)
    # All examples in eval data set are y=x.
    self.assertGreater(scores['labels/actual_label_mean'], 0.9)
    # If there were no weight column, model would learn y=Not(x). Because of
    # weights, it learns y=x.
    self.assertGreater(scores['labels/prediction_mean'], 0.9)
    # All examples in eval data set are y=x. So if weight column were ignored,
    # then accuracy would be zero. Because of weights, accuracy should be close
    # to 1.0.
    self.assertGreater(scores['accuracy'], 0.9)

    scores_train_set = classifier.evaluate(input_fn=_input_fn_train, steps=1)
    # Considering weights, the mean label should be close to 1.0.
    # If weights were ignored, it would be 0.25.
    self.assertGreater(scores_train_set['labels/actual_label_mean'], 0.9)
    # The classifier has learned y=x.  If weight column were ignored in
    # evaluation, then accuracy for the train set would be 0.25.
    # Because weight is not ignored, accuracy is greater than 0.6.
    self.assertGreater(scores_train_set['accuracy'], 0.6)

  def testWeightColumnLoss(self):
    """Test ensures that you can specify per-example weights for loss."""

    def _input_fn():
      features = {
          'age': constant_op.constant([[20], [20], [20]]),
          'weights': constant_op.constant([[100], [1], [1]]),
      }
      labels = constant_op.constant([[1], [0], [0]])
      return features, labels

    age = feature_column_lib.real_valued_column('age')

    classifier = linear.LinearClassifier(feature_columns=[age])
    classifier.fit(input_fn=_input_fn, steps=100)
    loss_unweighted = classifier.evaluate(input_fn=_input_fn, steps=1)['loss']

    classifier = linear.LinearClassifier(
        feature_columns=[age], weight_column_name='weights')
    classifier.fit(input_fn=_input_fn, steps=100)
    loss_weighted = classifier.evaluate(input_fn=_input_fn, steps=1)['loss']

    self.assertLess(loss_weighted, loss_unweighted)

  def testExport(self):
    """Tests that export model for servo works."""

    def input_fn():
      return {
          'age':
              constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    age = feature_column_lib.real_valued_column('age')

    classifier = linear.LinearClassifier(feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=100)

    export_dir = tempfile.mkdtemp()
    classifier.export(export_dir)

  def testDisableCenteredBias(self):
    """Tests that we can disable centered bias."""

    def input_fn():
      return {
          'age':
              constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    age = feature_column_lib.real_valued_column('age')

    classifier = linear.LinearClassifier(
        feature_columns=[age, language], enable_centered_bias=False)
    classifier.fit(input_fn=input_fn, steps=100)
    self.assertNotIn('centered_bias_weight', classifier.get_variable_names())

  def testEnableCenteredBias(self):
    """Tests that we can disable centered bias."""

    def input_fn():
      return {
          'age':
              constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    age = feature_column_lib.real_valued_column('age')

    classifier = linear.LinearClassifier(
        feature_columns=[age, language], enable_centered_bias=True)
    classifier.fit(input_fn=input_fn, steps=100)
    self.assertIn('centered_bias_weight', classifier.get_variable_names())

  def testTrainOptimizerWithL1Reg(self):
    """Tests l1 regularized model has higher loss."""

    def input_fn():
      return {
          'language':
              sparse_tensor.SparseTensor(
                  values=['hindi'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[1]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    classifier_no_reg = linear.LinearClassifier(feature_columns=[language])
    classifier_with_reg = linear.LinearClassifier(
        feature_columns=[language],
        optimizer=ftrl.FtrlOptimizer(
            learning_rate=1.0, l1_regularization_strength=100.))
    loss_no_reg = classifier_no_reg.fit(input_fn=input_fn, steps=100).evaluate(
        input_fn=input_fn, steps=1)['loss']
    loss_with_reg = classifier_with_reg.fit(input_fn=input_fn,
                                            steps=100).evaluate(
                                                input_fn=input_fn,
                                                steps=1)['loss']
    self.assertLess(loss_no_reg, loss_with_reg)

  def testTrainWithMissingFeature(self):
    """Tests that training works with missing features."""

    def input_fn():
      return {
          'language':
              sparse_tensor.SparseTensor(
                  values=['Swahili', 'turkish'],
                  indices=[[0, 0], [2, 0]],
                  dense_shape=[3, 1])
      }, constant_op.constant([[1], [1], [1]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    classifier = linear.LinearClassifier(feature_columns=[language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.07)

  def testSdcaOptimizerRealValuedFeatures(self):
    """Tests LinearClasssifier with SDCAOptimizer and real valued features."""

    def input_fn():
      return {
          'example_id': constant_op.constant(['1', '2']),
          'maintenance_cost': constant_op.constant([[500.0], [200.0]]),
          'sq_footage': constant_op.constant([[800.0], [600.0]]),
          'weights': constant_op.constant([[1.0], [1.0]])
      }, constant_op.constant([[0], [1]])

    maintenance_cost = feature_column_lib.real_valued_column('maintenance_cost')
    sq_footage = feature_column_lib.real_valued_column('sq_footage')
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    classifier = linear.LinearClassifier(
        feature_columns=[maintenance_cost, sq_footage],
        weight_column_name='weights',
        optimizer=sdca_optimizer)
    classifier.fit(input_fn=input_fn, steps=100)
    loss = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.05)

  def testSdcaOptimizerRealValuedFeatureWithHigherDimension(self):
    """Tests SDCAOptimizer with real valued features of higher dimension."""

    # input_fn is identical to the one in testSdcaOptimizerRealValuedFeatures
    # where 2 1-dimensional dense features have been replaced by 1 2-dimensional
    # feature.
    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2']),
          'dense_feature':
              constant_op.constant([[500.0, 800.0], [200.0, 600.0]])
      }, constant_op.constant([[0], [1]])

    dense_feature = feature_column_lib.real_valued_column(
        'dense_feature', dimension=2)
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    classifier = linear.LinearClassifier(
        feature_columns=[dense_feature], optimizer=sdca_optimizer)
    classifier.fit(input_fn=input_fn, steps=100)
    loss = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.05)

  def testSdcaOptimizerBucketizedFeatures(self):
    """Tests LinearClasssifier with SDCAOptimizer and bucketized features."""

    def input_fn():
      return {
          'example_id': constant_op.constant(['1', '2', '3']),
          'price': constant_op.constant([[600.0], [1000.0], [400.0]]),
          'sq_footage': constant_op.constant([[1000.0], [600.0], [700.0]]),
          'weights': constant_op.constant([[1.0], [1.0], [1.0]])
      }, constant_op.constant([[1], [0], [1]])

    price_bucket = feature_column_lib.bucketized_column(
        feature_column_lib.real_valued_column('price'),
        boundaries=[500.0, 700.0])
    sq_footage_bucket = feature_column_lib.bucketized_column(
        feature_column_lib.real_valued_column('sq_footage'), boundaries=[650.0])
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id', symmetric_l2_regularization=1.0)
    classifier = linear.LinearClassifier(
        feature_columns=[price_bucket, sq_footage_bucket],
        weight_column_name='weights',
        optimizer=sdca_optimizer)
    classifier.fit(input_fn=input_fn, steps=50)
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testSdcaOptimizerSparseFeatures(self):
    """Tests LinearClasssifier with SDCAOptimizer and sparse features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              constant_op.constant([[0.4], [0.6], [0.3]]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 3], [2, 1]],
                  dense_shape=[3, 5]),
          'weights':
              constant_op.constant([[1.0], [1.0], [1.0]])
      }, constant_op.constant([[1], [0], [1]])

    price = feature_column_lib.real_valued_column('price')
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    classifier = linear.LinearClassifier(
        feature_columns=[price, country],
        weight_column_name='weights',
        optimizer=sdca_optimizer)
    classifier.fit(input_fn=input_fn, steps=50)
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testSdcaOptimizerWeightedSparseFeatures(self):
    """LinearClasssifier with SDCAOptimizer and weighted sparse features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              sparse_tensor.SparseTensor(
                  values=[2., 3., 1.],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 5]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 5])
      }, constant_op.constant([[1], [0], [1]])

    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    country_weighted_by_price = feature_column_lib.weighted_sparse_column(
        country, 'price')
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    classifier = linear.LinearClassifier(
        feature_columns=[country_weighted_by_price], optimizer=sdca_optimizer)
    classifier.fit(input_fn=input_fn, steps=50)
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testSdcaOptimizerCrossedFeatures(self):
    """Tests LinearClasssifier with SDCAOptimizer and crossed features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english', 'italian', 'spanish'],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 1]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['US', 'IT', 'MX'],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 1])
      }, constant_op.constant([[0], [0], [1]])

    language = feature_column_lib.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=5)
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    country_language = feature_column_lib.crossed_column(
        [language, country], hash_bucket_size=10)
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    classifier = linear.LinearClassifier(
        feature_columns=[country_language], optimizer=sdca_optimizer)
    classifier.fit(input_fn=input_fn, steps=10)
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testSdcaOptimizerMixedFeatures(self):
    """Tests LinearClasssifier with SDCAOptimizer and a mix of features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              constant_op.constant([[0.6], [0.8], [0.3]]),
          'sq_footage':
              constant_op.constant([[900.0], [700.0], [600.0]]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 3], [2, 1]],
                  dense_shape=[3, 5]),
          'weights':
              constant_op.constant([[3.0], [1.0], [1.0]])
      }, constant_op.constant([[1], [0], [1]])

    price = feature_column_lib.real_valued_column('price')
    sq_footage_bucket = feature_column_lib.bucketized_column(
        feature_column_lib.real_valued_column('sq_footage'),
        boundaries=[650.0, 800.0])
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    sq_footage_country = feature_column_lib.crossed_column(
        [sq_footage_bucket, country], hash_bucket_size=10)
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    classifier = linear.LinearClassifier(
        feature_columns=[price, sq_footage_bucket, country, sq_footage_country],
        weight_column_name='weights',
        optimizer=sdca_optimizer)
    classifier.fit(input_fn=input_fn, steps=50)
    scores = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(scores['accuracy'], 0.9)

  def testEval(self):
    """Tests that eval produces correct metrics.
    """

    def input_fn():
      return {
          'age':
              constant_op.constant([[1], [2]]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['greek', 'chinese'],
                  indices=[[0, 0], [1, 0]],
                  dense_shape=[2, 1]),
      }, constant_op.constant([[1], [0]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    age = feature_column_lib.real_valued_column('age')
    classifier = linear.LinearClassifier(feature_columns=[age, language])

    # Evaluate on trained model
    classifier.fit(input_fn=input_fn, steps=100)
    classifier.evaluate(input_fn=input_fn, steps=1)

    # TODO(ispir): Enable accuracy check after resolving the randomness issue.
    # self.assertLess(evaluated_values['loss/mean'], 0.3)
    # self.assertGreater(evaluated_values['accuracy/mean'], .95)


class LinearRegressorTest(test.TestCase):

  def testExperimentIntegration(self):
    cont_features = [
        feature_column_lib.real_valued_column(
            'feature', dimension=4)
    ]

    exp = experiment.Experiment(
        estimator=linear.LinearRegressor(feature_columns=cont_features),
        train_input_fn=test_data.iris_input_logistic_fn,
        eval_input_fn=test_data.iris_input_logistic_fn)
    exp.test()

  def testEstimatorContract(self):
    estimator_test_utils.assert_estimator_contract(self, linear.LinearRegressor)

  def testRegression(self):
    """Tests that loss goes down with training."""

    def input_fn():
      return {
          'age':
              constant_op.constant([1]),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english'], indices=[[0, 0]], dense_shape=[1, 1])
      }, constant_op.constant([[10.]])

    language = feature_column_lib.sparse_column_with_hash_bucket('language',
                                                                 100)
    age = feature_column_lib.real_valued_column('age')

    classifier = linear.LinearRegressor(feature_columns=[age, language])
    classifier.fit(input_fn=input_fn, steps=100)
    loss1 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    classifier.fit(input_fn=input_fn, steps=200)
    loss2 = classifier.evaluate(input_fn=input_fn, steps=1)['loss']

    self.assertLess(loss2, loss1)
    self.assertLess(loss2, 0.5)

  def testRegression_MatrixData(self):
    """Tests regression using matrix data as input."""
    cont_features = [
        feature_column_lib.real_valued_column(
            'feature', dimension=4)
    ]

    regressor = linear.LinearRegressor(
        feature_columns=cont_features,
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=test_data.iris_input_multiclass_fn, steps=100)
    scores = regressor.evaluate(
        input_fn=test_data.iris_input_multiclass_fn, steps=1)
    self.assertLess(scores['loss'], 0.2)

  def testRegression_TensorData(self):
    """Tests regression using tensor data as input."""

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
      return features, constant_op.constant(
          [1.0, 0., 0.2], dtype=dtypes.float32)

    feature_columns = [
        feature_column_lib.sparse_column_with_hash_bucket(
            'language', hash_bucket_size=20),
        feature_column_lib.real_valued_column('age')
    ]

    regressor = linear.LinearRegressor(
        feature_columns=feature_columns,
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.2)

  def testLoss(self):
    """Tests loss calculation."""

    def _input_fn_train():
      # Create 4 rows, one of them (y = x), three of them (y=Not(x))
      # The algorithm should learn (y = 0.25).
      labels = constant_op.constant([[1.], [0.], [0.], [0.]])
      features = {'x': array_ops.ones(shape=[4, 1], dtype=dtypes.float32),}
      return features, labels

    regressor = linear.LinearRegressor(
        feature_columns=[feature_column_lib.real_valued_column('x')],
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

    regressor = linear.LinearRegressor(
        weight_column_name='w',
        feature_columns=[feature_column_lib.real_valued_column('x')],
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

    regressor = linear.LinearRegressor(
        weight_column_name='w',
        feature_columns=[feature_column_lib.real_valued_column('x')],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn_train, steps=100)
    scores = regressor.evaluate(input_fn=_input_fn_eval, steps=1)
    # The model should learn (y = x) because of the weights, so the loss should
    # be close to zero.
    self.assertLess(scores['loss'], 0.1)

  def testPredict_AsIterableFalse(self):
    """Tests predict method with as_iterable=False."""
    labels = [1.0, 0., 0.2]

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

    feature_columns = [
        feature_column_lib.sparse_column_with_hash_bucket(
            'language', hash_bucket_size=20),
        feature_column_lib.real_valued_column('age')
    ]

    regressor = linear.LinearRegressor(
        feature_columns=feature_columns,
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.1)
    predictions = regressor.predict(input_fn=_input_fn, as_iterable=False)
    self.assertAllClose(labels, predictions, atol=0.1)

  def testPredict_AsIterable(self):
    """Tests predict method with as_iterable=True."""
    labels = [1.0, 0., 0.2]

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

    feature_columns = [
        feature_column_lib.sparse_column_with_hash_bucket(
            'language', hash_bucket_size=20),
        feature_column_lib.real_valued_column('age')
    ]

    regressor = linear.LinearRegressor(
        feature_columns=feature_columns,
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.1)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(
        regressor.predict(
            input_fn=predict_input_fn, as_iterable=True))
    self.assertAllClose(labels, predictions, atol=0.1)

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

    regressor = linear.LinearRegressor(
        feature_columns=[feature_column_lib.real_valued_column('x')],
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)
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
              'bad_name':
                  MetricSpec(
                      metric_fn=metric_ops.streaming_auc,
                      prediction_key='bad_type')
          })

    # Tests the case where the 2nd element of the key is not "scores".
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
                  values=['en', 'fr', 'zh'],
                  indices=[[0, 0], [0, 1], [2, 0]],
                  dense_shape=[3, 2])
      }
      return features, constant_op.constant(
          [1.0, 0., 0.2], dtype=dtypes.float32)

    feature_columns = [
        feature_column_lib.sparse_column_with_hash_bucket(
            'language', hash_bucket_size=20),
        feature_column_lib.real_valued_column('age')
    ]

    model_dir = tempfile.mkdtemp()
    regressor = linear.LinearRegressor(
        model_dir=model_dir,
        feature_columns=feature_columns,
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)
    predict_input_fn = functools.partial(_input_fn, num_epochs=1)
    predictions = list(regressor.predict(input_fn=predict_input_fn))
    del regressor

    regressor2 = linear.LinearRegressor(
        model_dir=model_dir, feature_columns=feature_columns)
    predictions2 = list(regressor2.predict(input_fn=predict_input_fn))
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
      return features, constant_op.constant(
          [1.0, 0., 0.2], dtype=dtypes.float32)

    feature_columns = [
        # The given hash_bucket_size results in variables larger than the
        # default min_slice_size attribute, so the variables are partitioned.
        feature_column_lib.sparse_column_with_hash_bucket(
            'language', hash_bucket_size=2e7),
        feature_column_lib.real_valued_column('age')
    ]

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

    regressor = linear.LinearRegressor(
        feature_columns=feature_columns, config=config)

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.1)

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
      return features, constant_op.constant(
          [1.0, 0., 0.2], dtype=dtypes.float32)

    feature_columns = [
        feature_column_lib.sparse_column_with_hash_bucket(
            'language', hash_bucket_size=20),
        feature_column_lib.real_valued_column('age')
    ]

    regressor = linear.LinearRegressor(
        feature_columns=feature_columns,
        enable_centered_bias=False,
        config=run_config.RunConfig(tf_random_seed=1))

    regressor.fit(input_fn=_input_fn, steps=100)

    scores = regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertLess(scores['loss'], 0.1)

  def testRecoverWeights(self):
    rng = np.random.RandomState(67)
    n = 1000
    n_weights = 10
    bias = 2
    x = rng.uniform(-1, 1, (n, n_weights))
    weights = 10 * rng.randn(n_weights)
    y = np.dot(x, weights)
    y += rng.randn(len(x)) * 0.05 + rng.normal(bias, 0.01)
    feature_columns = estimator.infer_real_valued_columns_from_input(x)
    regressor = linear.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=ftrl.FtrlOptimizer(learning_rate=0.8))
    regressor.fit(x, y, batch_size=64, steps=2000)
    # Have to flatten weights since they come in (x, 1) shape.
    self.assertAllClose(weights, regressor.weights_.flatten(), rtol=1)
    # TODO(ispir): Disable centered_bias.
    # assert abs(bias - regressor.bias_) < 0.1

  def testSdcaOptimizerRealValuedLinearFeatures(self):
    """Tests LinearRegressor with SDCAOptimizer and real valued features."""
    x = [[1.2, 2.0, -1.5], [-2.0, 3.0, -0.5], [1.0, -0.5, 4.0]]
    weights = [[3.0], [-1.2], [0.5]]
    y = np.dot(x, weights)

    def input_fn():
      return {
          'example_id': constant_op.constant(['1', '2', '3']),
          'x': constant_op.constant(x),
          'weights': constant_op.constant([[10.0], [10.0], [10.0]])
      }, constant_op.constant(y)

    x_column = feature_column_lib.real_valued_column('x', dimension=3)
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    regressor = linear.LinearRegressor(
        feature_columns=[x_column],
        weight_column_name='weights',
        optimizer=sdca_optimizer)
    regressor.fit(input_fn=input_fn, steps=20)
    loss = regressor.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.01)
    self.assertAllClose(
        [w[0] for w in weights], regressor.weights_.flatten(), rtol=0.1)

  def testSdcaOptimizerMixedFeaturesArbitraryWeights(self):
    """Tests LinearRegressor with SDCAOptimizer and a mix of features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              constant_op.constant([[0.6], [0.8], [0.3]]),
          'sq_footage':
              constant_op.constant([[900.0], [700.0], [600.0]]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 3], [2, 1]],
                  dense_shape=[3, 5]),
          'weights':
              constant_op.constant([[3.0], [5.0], [7.0]])
      }, constant_op.constant([[1.55], [-1.25], [-3.0]])

    price = feature_column_lib.real_valued_column('price')
    sq_footage_bucket = feature_column_lib.bucketized_column(
        feature_column_lib.real_valued_column('sq_footage'),
        boundaries=[650.0, 800.0])
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    sq_footage_country = feature_column_lib.crossed_column(
        [sq_footage_bucket, country], hash_bucket_size=10)
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id', symmetric_l2_regularization=1.0)
    regressor = linear.LinearRegressor(
        feature_columns=[price, sq_footage_bucket, country, sq_footage_country],
        weight_column_name='weights',
        optimizer=sdca_optimizer)
    regressor.fit(input_fn=input_fn, steps=20)
    loss = regressor.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.05)

  def testSdcaOptimizerSparseFeaturesWithL1Reg(self):
    """Tests LinearClasssifier with SDCAOptimizer and sparse features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              constant_op.constant([[0.4], [0.6], [0.3]]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 3], [2, 1]],
                  dense_shape=[3, 5]),
          'weights':
              constant_op.constant([[10.0], [10.0], [10.0]])
      }, constant_op.constant([[1.4], [-0.8], [2.6]])

    price = feature_column_lib.real_valued_column('price')
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    # Regressor with no L1 regularization.
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    regressor = linear.LinearRegressor(
        feature_columns=[price, country],
        weight_column_name='weights',
        optimizer=sdca_optimizer)
    regressor.fit(input_fn=input_fn, steps=20)
    no_l1_reg_loss = regressor.evaluate(input_fn=input_fn, steps=1)['loss']
    no_l1_reg_weights = regressor.weights_

    # Regressor with L1 regularization.
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id', symmetric_l1_regularization=1.0)
    regressor = linear.LinearRegressor(
        feature_columns=[price, country],
        weight_column_name='weights',
        optimizer=sdca_optimizer)
    regressor.fit(input_fn=input_fn, steps=20)
    l1_reg_loss = regressor.evaluate(input_fn=input_fn, steps=1)['loss']
    l1_reg_weights = regressor.weights_

    # Unregularized loss is lower when there is no L1 regularization.
    self.assertLess(no_l1_reg_loss, l1_reg_loss)
    self.assertLess(no_l1_reg_loss, 0.05)

    # But weights returned by the regressor with L1 regularization have smaller
    # L1 norm.
    l1_reg_weights_norm, no_l1_reg_weights_norm = 0.0, 0.0
    for var_name in sorted(l1_reg_weights):
      l1_reg_weights_norm += sum(
          np.absolute(l1_reg_weights[var_name].flatten()))
      no_l1_reg_weights_norm += sum(
          np.absolute(no_l1_reg_weights[var_name].flatten()))
      print('Var name: %s, value: %s' %
            (var_name, no_l1_reg_weights[var_name].flatten()))
    self.assertLess(l1_reg_weights_norm, no_l1_reg_weights_norm)

  def testSdcaOptimizerBiasOnly(self):
    """Tests LinearClasssifier with SDCAOptimizer and validates bias weight."""

    def input_fn():
      """Testing the bias weight when it's the only feature present.

      All of the instances in this input only have the bias feature, and a
      1/4 of the labels are positive. This means that the expected weight for
      the bias should be close to the average prediction, i.e 0.25.
      Returns:
        Training data for the test.
      """
      num_examples = 40
      return {
          'example_id':
              constant_op.constant([str(x + 1) for x in range(num_examples)]),
          # place_holder is an empty column which is always 0 (absent), because
          # LinearClassifier requires at least one column.
          'place_holder':
              constant_op.constant([[0.0]] * num_examples),
      }, constant_op.constant(
          [[1 if i % 4 is 0 else 0] for i in range(num_examples)])

    place_holder = feature_column_lib.real_valued_column('place_holder')
    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    regressor = linear.LinearRegressor(
        feature_columns=[place_holder], optimizer=sdca_optimizer)
    regressor.fit(input_fn=input_fn, steps=100)

    self.assertNear(
        regressor.get_variable_value('linear/bias_weight')[0], 0.25, err=0.1)

  def testSdcaOptimizerBiasAndOtherColumns(self):
    """Tests LinearClasssifier with SDCAOptimizer and validates bias weight."""

    def input_fn():
      """Testing the bias weight when there are other features present.

      1/2 of the instances in this input have feature 'a', the rest have
      feature 'b', and we expect the bias to be added to each instance as well.
      0.4 of all instances that have feature 'a' are positive, and 0.2 of all
      instances that have feature 'b' are positive. The labels in the dataset
      are ordered to appear shuffled since SDCA expects shuffled data, and
      converges faster with this pseudo-random ordering.
      If the bias was centered we would expect the weights to be:
      bias: 0.3
      a: 0.1
      b: -0.1
      Until b/29339026 is resolved, the bias gets regularized with the same
      global value for the other columns, and so the expected weights get
      shifted and are:
      bias: 0.2
      a: 0.2
      b: 0.0
      Returns:
        The test dataset.
      """
      num_examples = 200
      half = int(num_examples / 2)
      return {
          'example_id':
              constant_op.constant([str(x + 1) for x in range(num_examples)]),
          'a':
              constant_op.constant([[1]] * int(half) + [[0]] * int(half)),
          'b':
              constant_op.constant([[0]] * int(half) + [[1]] * int(half)),
      }, constant_op.constant(
          [[x]
           for x in [1, 0, 0, 1, 1, 0, 0, 0, 1, 0] * int(half / 10) +
           [0, 1, 0, 0, 0, 0, 0, 0, 1, 0] * int(half / 10)])

    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    regressor = linear.LinearRegressor(
        feature_columns=[
            feature_column_lib.real_valued_column('a'),
            feature_column_lib.real_valued_column('b')
        ],
        optimizer=sdca_optimizer)

    regressor.fit(input_fn=input_fn, steps=200)

    # TODO(b/29339026): Change the expected results to expect a centered bias.
    self.assertNear(
        regressor.get_variable_value('linear/bias_weight')[0], 0.2, err=0.05)
    self.assertNear(regressor.weights_['linear/a/weight'][0], 0.2, err=0.05)
    self.assertNear(regressor.weights_['linear/b/weight'][0], 0.0, err=0.05)

  def testSdcaOptimizerBiasAndOtherColumnsFabricatedCentered(self):
    """Tests LinearClasssifier with SDCAOptimizer and validates bias weight."""

    def input_fn():
      """Testing the bias weight when there are other features present.

      1/2 of the instances in this input have feature 'a', the rest have
      feature 'b', and we expect the bias to be added to each instance as well.
      0.1 of all instances that have feature 'a' have a label of 1, and 0.1 of
      all instances that have feature 'b' have a label of -1.
      We can expect the weights to be:
      bias: 0.0
      a: 0.1
      b: -0.1
      Returns:
        The test dataset.
      """
      num_examples = 200
      half = int(num_examples / 2)
      return {
          'example_id':
              constant_op.constant([str(x + 1) for x in range(num_examples)]),
          'a':
              constant_op.constant([[1]] * int(half) + [[0]] * int(half)),
          'b':
              constant_op.constant([[0]] * int(half) + [[1]] * int(half)),
      }, constant_op.constant([[1 if x % 10 == 0 else 0] for x in range(half)] +
                              [[-1 if x % 10 == 0 else 0] for x in range(half)])

    sdca_optimizer = sdca_optimizer_lib.SDCAOptimizer(
        example_id_column='example_id')
    regressor = linear.LinearRegressor(
        feature_columns=[
            feature_column_lib.real_valued_column('a'),
            feature_column_lib.real_valued_column('b')
        ],
        optimizer=sdca_optimizer)

    regressor.fit(input_fn=input_fn, steps=100)

    self.assertNear(
        regressor.get_variable_value('linear/bias_weight')[0], 0.0, err=0.05)
    self.assertNear(regressor.weights_['linear/a/weight'][0], 0.1, err=0.05)
    self.assertNear(regressor.weights_['linear/b/weight'][0], -0.1, err=0.05)


def boston_input_fn():
  boston = base.load_boston()
  features = math_ops.cast(
      array_ops.reshape(constant_op.constant(boston.data), [-1, 13]),
      dtypes.float32)
  labels = math_ops.cast(
      array_ops.reshape(constant_op.constant(boston.target), [-1, 1]),
      dtypes.float32)
  return features, labels


class FeatureColumnTest(test.TestCase):

  def testTrain(self):
    feature_columns = estimator.infer_real_valued_columns_from_input_fn(
        boston_input_fn)
    est = linear.LinearRegressor(feature_columns=feature_columns)
    est.fit(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_input_fn, steps=1)


if __name__ == '__main__':
  test.main()
