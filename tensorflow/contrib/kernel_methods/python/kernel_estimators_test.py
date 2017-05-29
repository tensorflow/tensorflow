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
"""Tests for kernel_estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import layers
from tensorflow.contrib.kernel_methods.python import kernel_estimators
from tensorflow.contrib.kernel_methods.python.mappers.random_fourier_features import RandomFourierFeatureMapper
from tensorflow.contrib.learn.python.learn.estimators import test_data
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.platform import googletest


def _linearly_separable_binary_input_fn():
  """Returns linearly-separable data points (binary classification)."""
  return {
      'feature1': constant_op.constant([[0.0], [1.0], [3.0]]),
      'feature2': constant_op.constant([[1.0], [-1.2], [1.0]]),
  }, constant_op.constant([[1], [0], [1]])


def _linearly_inseparable_binary_input_fn():
  """Returns non-linearly-separable data points (binary classification)."""
  return {
      'multi_dim_feature':
          constant_op.constant([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
                                [-1.0, 1.0]]),
  }, constant_op.constant([[1], [0], [1], [0]])


class KernelLinearClassifierTest(TensorFlowTestCase):

  def testNoFeatureColumnsOrKernelMappers(self):
    """Tests that at least one of feature columns or kernels is provided."""
    with self.assertRaises(ValueError):
      _ = kernel_estimators.KernelLinearClassifier()

  def testInvalidKernelMapper(self):
    """ValueError raised when the kernel mappers provided have invalid type."""

    class DummyKernelMapper(object):

      def __init__(self):
        pass

    feature = layers.real_valued_column('feature')
    kernel_mappers = {feature: [DummyKernelMapper()]}
    with self.assertRaises(ValueError):
      _ = kernel_estimators.KernelLinearClassifier(
          feature_columns=[feature], kernel_mappers=kernel_mappers)

  def testInvalidNumberOfClasses(self):
    """ValueError raised when the kernel mappers provided have invalid type."""

    feature = layers.real_valued_column('feature')
    with self.assertRaises(ValueError):
      _ = kernel_estimators.KernelLinearClassifier(
          feature_columns=[feature], n_classes=1)

  def testLinearlySeparableBinaryDataNoKernels(self):
    """Tests classifier w/o kernels (log. regression) for lin-separable data."""

    feature1 = layers.real_valued_column('feature1')
    feature2 = layers.real_valued_column('feature2')

    logreg_classifier = kernel_estimators.KernelLinearClassifier(
        feature_columns=[feature1, feature2])
    logreg_classifier.fit(
        input_fn=_linearly_separable_binary_input_fn, steps=100)

    metrics = logreg_classifier.evaluate(
        input_fn=_linearly_separable_binary_input_fn, steps=1)
    # Since the data is linearly separable, the classifier should have small
    # loss and perfect accuracy.
    self.assertLess(metrics['loss'], 0.1)
    self.assertEqual(metrics['accuracy'], 1.0)

    # As a result, it should assign higher probability to class 1 for the 1st
    # and 3rd example and higher probability to class 0 for the second example.
    logreg_prob_predictions = list(
        logreg_classifier.predict_proba(input_fn=
                                        _linearly_separable_binary_input_fn))
    self.assertGreater(logreg_prob_predictions[0][1], 0.5)
    self.assertGreater(logreg_prob_predictions[1][0], 0.5)
    self.assertGreater(logreg_prob_predictions[2][1], 0.5)

  def testLinearlyInseparableBinaryDataWithAndWithoutKernels(self):
    """Tests classifier w/ and w/o kernels on non-linearly-separable data."""
    multi_dim_feature = layers.real_valued_column(
        'multi_dim_feature', dimension=2)

    # Data points are non-linearly separable so there will be at least one
    # mis-classified sample (accuracy < 0.8). In fact, the loss is minimized for
    # w1=w2=0.0, in which case each example incurs a loss of ln(2). The overall
    # (average) loss should then be ln(2) and the logits should be approximately
    # 0.0 for each sample.
    logreg_classifier = kernel_estimators.KernelLinearClassifier(
        feature_columns=[multi_dim_feature])
    logreg_classifier.fit(
        input_fn=_linearly_inseparable_binary_input_fn, steps=50)
    logreg_metrics = logreg_classifier.evaluate(
        input_fn=_linearly_inseparable_binary_input_fn, steps=1)
    logreg_loss = logreg_metrics['loss']
    logreg_accuracy = logreg_metrics['accuracy']
    logreg_predictions = logreg_classifier.predict(
        input_fn=_linearly_inseparable_binary_input_fn, as_iterable=False)
    self.assertAlmostEqual(logreg_loss, np.log(2), places=3)
    self.assertLess(logreg_accuracy, 0.8)
    self.assertAllClose(logreg_predictions['logits'], [[0.0], [0.0], [0.0],
                                                       [0.0]])

    # Using kernel mappers allows to discover non-linearities in data. Mapping
    # the data to a higher dimensional feature space using approx RBF kernels,
    # substantially reduces the loss and leads to perfect classification
    # accuracy.
    kernel_mappers = {
        multi_dim_feature: [RandomFourierFeatureMapper(2, 30, 0.6, 1, 'rffm')]
    }
    kernelized_logreg_classifier = kernel_estimators.KernelLinearClassifier(
        feature_columns=[], kernel_mappers=kernel_mappers)
    kernelized_logreg_classifier.fit(
        input_fn=_linearly_inseparable_binary_input_fn, steps=50)
    kernelized_logreg_metrics = kernelized_logreg_classifier.evaluate(
        input_fn=_linearly_inseparable_binary_input_fn, steps=1)
    kernelized_logreg_loss = kernelized_logreg_metrics['loss']
    kernelized_logreg_accuracy = kernelized_logreg_metrics['accuracy']
    self.assertLess(kernelized_logreg_loss, 0.2)
    self.assertEqual(kernelized_logreg_accuracy, 1.0)

  def testVariablesWithAndWithoutKernels(self):
    """Tests variables w/ and w/o kernel."""
    multi_dim_feature = layers.real_valued_column(
        'multi_dim_feature', dimension=2)

    linear_classifier = kernel_estimators.KernelLinearClassifier(
        feature_columns=[multi_dim_feature])
    linear_classifier.fit(
        input_fn=_linearly_inseparable_binary_input_fn, steps=50)
    linear_variables = linear_classifier.get_variable_names()
    self.assertIn('linear/multi_dim_feature/weight', linear_variables)
    self.assertIn('linear/bias_weight', linear_variables)
    linear_weights = linear_classifier.get_variable_value(
        'linear/multi_dim_feature/weight')
    linear_bias = linear_classifier.get_variable_value('linear/bias_weight')

    kernel_mappers = {
        multi_dim_feature: [RandomFourierFeatureMapper(2, 30, 0.6, 1, 'rffm')]
    }
    kernel_linear_classifier = kernel_estimators.KernelLinearClassifier(
        feature_columns=[], kernel_mappers=kernel_mappers)
    kernel_linear_classifier.fit(
        input_fn=_linearly_inseparable_binary_input_fn, steps=50)
    kernel_linear_variables = kernel_linear_classifier.get_variable_names()
    self.assertIn('linear/multi_dim_feature_MAPPED/weight',
                  kernel_linear_variables)
    self.assertIn('linear/bias_weight', kernel_linear_variables)
    kernel_linear_weights = kernel_linear_classifier.get_variable_value(
        'linear/multi_dim_feature_MAPPED/weight')
    kernel_linear_bias = kernel_linear_classifier.get_variable_value(
        'linear/bias_weight')

    # The feature column used for linear classification (no kernels) has
    # dimension 2 so the model will learn a 2-dimension weights vector (and a
    # scalar for the bias). In the kernelized model, the features are mapped to
    # a 30-dimensional feature space and so the weights variable will also have
    # dimension 30.
    self.assertEqual(2, len(linear_weights))
    self.assertEqual(1, len(linear_bias))
    self.assertEqual(30, len(kernel_linear_weights))
    self.assertEqual(1, len(kernel_linear_bias))

  def testClassifierWithAndWithoutKernelsNoRealValuedColumns(self):
    """Tests kernels have no effect for non-real valued columns ."""

    def input_fn():
      return {
          'price':
              constant_op.constant([[0.4], [0.6], [0.3]]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 3], [2, 1]],
                  dense_shape=[3, 5]),
      }, constant_op.constant([[1], [0], [1]])

    price = layers.real_valued_column('price')
    country = layers.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)

    linear_classifier = kernel_estimators.KernelLinearClassifier(
        feature_columns=[price, country])
    linear_classifier.fit(input_fn=input_fn, steps=100)
    linear_metrics = linear_classifier.evaluate(input_fn=input_fn, steps=1)
    linear_loss = linear_metrics['loss']
    linear_accuracy = linear_metrics['accuracy']

    kernel_mappers = {
        country: [RandomFourierFeatureMapper(2, 30, 0.6, 1, 'rffm')]
    }

    kernel_linear_classifier = kernel_estimators.KernelLinearClassifier(
        feature_columns=[price, country], kernel_mappers=kernel_mappers)
    kernel_linear_classifier.fit(input_fn=input_fn, steps=100)
    kernel_linear_metrics = kernel_linear_classifier.evaluate(
        input_fn=input_fn, steps=1)
    kernel_linear_loss = kernel_linear_metrics['loss']
    kernel_linear_accuracy = kernel_linear_metrics['accuracy']

    # The kernel mapping is applied to a non-real-valued feature column and so
    # it should have no effect on the model. The loss and accuracy of the
    # "kernelized" model should match the loss and accuracy of the initial model
    # (without kernels).
    self.assertAlmostEqual(linear_loss, kernel_linear_loss, delta=0.01)
    self.assertAlmostEqual(linear_accuracy, kernel_linear_accuracy, delta=0.01)

  def testMulticlassDataWithAndWithoutKernels(self):
    """Tests classifier w/ and w/o kernels on multiclass data."""
    feature_column = layers.real_valued_column('feature', dimension=4)

    # Metrics for linear classifier (no kernels).
    linear_classifier = kernel_estimators.KernelLinearClassifier(
        feature_columns=[feature_column], n_classes=3)
    linear_classifier.fit(input_fn=test_data.iris_input_multiclass_fn, steps=50)
    linear_metrics = linear_classifier.evaluate(
        input_fn=test_data.iris_input_multiclass_fn, steps=1)
    linear_loss = linear_metrics['loss']
    linear_accuracy = linear_metrics['accuracy']

    # Using kernel mappers allows to discover non-linearities in data (via RBF
    # kernel approximation), reduces loss and increases accuracy.
    kernel_mappers = {
        feature_column: [
            RandomFourierFeatureMapper(
                input_dim=4, output_dim=50, stddev=1.0, name='rffm')
        ]
    }
    kernel_linear_classifier = kernel_estimators.KernelLinearClassifier(
        feature_columns=[], n_classes=3, kernel_mappers=kernel_mappers)
    kernel_linear_classifier.fit(
        input_fn=test_data.iris_input_multiclass_fn, steps=50)
    kernel_linear_metrics = kernel_linear_classifier.evaluate(
        input_fn=test_data.iris_input_multiclass_fn, steps=1)
    kernel_linear_loss = kernel_linear_metrics['loss']
    kernel_linear_accuracy = kernel_linear_metrics['accuracy']
    self.assertLess(kernel_linear_loss, linear_loss)
    self.assertGreater(kernel_linear_accuracy, linear_accuracy)


if __name__ == '__main__':
  googletest.main()
