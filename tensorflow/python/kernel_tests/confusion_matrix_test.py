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
"""Tests for confusion_matrix_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ConfusionMatrixTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testExample(self):
    """This is a test of the example provided in pydoc."""
    with self.cached_session():
      self.assertAllEqual([
          [0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1]
      ], self.evaluate(confusion_matrix.confusion_matrix(
          labels=[1, 2, 4], predictions=[2, 2, 4])))

  def _testConfMatrix(self, labels, predictions, truth, weights=None,
                      num_classes=None):
    with self.cached_session():
      dtype = predictions.dtype
      ans = confusion_matrix.confusion_matrix(
          labels, predictions, dtype=dtype, weights=weights,
          num_classes=num_classes).eval()
      self.assertAllClose(truth, ans, atol=1e-10)
      self.assertEqual(ans.dtype, dtype)

  def _testBasic(self, dtype):
    labels = np.arange(5, dtype=dtype)
    predictions = np.arange(5, dtype=dtype)

    truth = np.asarray(
        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]],
        dtype=dtype)

    self._testConfMatrix(labels=labels, predictions=predictions, truth=truth)

  @test_util.run_deprecated_v1
  def testInt32Basic(self):
    self._testBasic(dtype=np.int32)

  @test_util.run_deprecated_v1
  def testInt64Basic(self):
    self._testBasic(dtype=np.int64)

  def _testConfMatrixOnTensors(self, tf_dtype, np_dtype):
    with self.cached_session() as sess:
      m_neg = array_ops.placeholder(dtype=dtypes.float32)
      m_pos = array_ops.placeholder(dtype=dtypes.float32)
      s = array_ops.placeholder(dtype=dtypes.float32)

      neg = random_ops.random_normal(
          [20], mean=m_neg, stddev=s, dtype=dtypes.float32)
      pos = random_ops.random_normal(
          [20], mean=m_pos, stddev=s, dtype=dtypes.float32)

      data = array_ops.concat([neg, pos], 0)
      data = math_ops.cast(math_ops.round(data), tf_dtype)
      data = math_ops.minimum(math_ops.maximum(data, 0), 1)
      lab = array_ops.concat(
          [
              array_ops.zeros(
                  [20], dtype=tf_dtype), array_ops.ones(
                      [20], dtype=tf_dtype)
          ],
          0)

      cm = confusion_matrix.confusion_matrix(
          lab, data, dtype=tf_dtype, num_classes=2)

      d, l, cm_out = sess.run([data, lab, cm], {m_neg: 0.0, m_pos: 1.0, s: 1.0})

      truth = np.zeros([2, 2], dtype=np_dtype)
      for i in xrange(len(d)):
        truth[l[i], d[i]] += 1

      self.assertEqual(cm_out.dtype, np_dtype)
      self.assertAllClose(cm_out, truth, atol=1e-10)

  @test_util.run_deprecated_v1
  def testOnTensors_int32(self):
    self._testConfMatrixOnTensors(dtypes.int32, np.int32)

  @test_util.run_deprecated_v1
  def testOnTensors_int64(self):
    self._testConfMatrixOnTensors(dtypes.int64, np.int64)

  def _testDifferentLabelsInPredictionAndTarget(self, dtype):
    labels = np.asarray([4, 5, 6], dtype=dtype)
    predictions = np.asarray([1, 2, 3], dtype=dtype)

    truth = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0]],
        dtype=dtype)

    self._testConfMatrix(labels=labels, predictions=predictions, truth=truth)

  @test_util.run_deprecated_v1
  def testInt32DifferentLabels(self, dtype=np.int32):
    self._testDifferentLabelsInPredictionAndTarget(dtype)

  @test_util.run_deprecated_v1
  def testInt64DifferentLabels(self, dtype=np.int64):
    self._testDifferentLabelsInPredictionAndTarget(dtype)

  def _testMultipleLabels(self, dtype):
    labels = np.asarray([1, 1, 2, 3, 5, 1, 3, 6, 3, 1], dtype=dtype)
    predictions = np.asarray([1, 1, 2, 3, 5, 6, 1, 2, 3, 4], dtype=dtype)

    truth = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 2, 0, 0, 1, 0, 1],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 2, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, 0]],
        dtype=dtype)

    self._testConfMatrix(labels=labels, predictions=predictions, truth=truth)

  @test_util.run_deprecated_v1
  def testInt32MultipleLabels(self, dtype=np.int32):
    self._testMultipleLabels(dtype)

  @test_util.run_deprecated_v1
  def testInt64MultipleLabels(self, dtype=np.int64):
    self._testMultipleLabels(dtype)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    labels = np.arange(5, dtype=np.int32)
    predictions = np.arange(5, dtype=np.int32)
    weights = np.arange(5, dtype=np.int32)

    truth = np.asarray(
        [[0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 0, 0, 3, 0],
         [0, 0, 0, 0, 4]],
        dtype=np.int32)

    self._testConfMatrix(
        labels=labels, predictions=predictions, weights=weights, truth=truth)

  @test_util.run_deprecated_v1
  def testLabelsTooLarge(self):
    labels = np.asarray([1, 1, 0, 3, 5], dtype=np.int32)
    predictions = np.asarray([2, 1, 0, 2, 2], dtype=np.int32)
    with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError,
                                             "`labels`.*out of bound"):
      self._testConfMatrix(
          labels=labels, predictions=predictions, num_classes=3, truth=None)

  def testLabelsNegative(self):
    labels = np.asarray([1, 1, 0, -1, -1], dtype=np.int32)
    predictions = np.asarray([2, 1, 0, 2, 2], dtype=np.int32)
    with self.assertRaisesOpError("`labels`.*negative values"):
      self._testConfMatrix(
          labels=labels, predictions=predictions, num_classes=3, truth=None)

  @test_util.run_deprecated_v1
  def testPredictionsTooLarge(self):
    labels = np.asarray([1, 1, 0, 2, 2], dtype=np.int32)
    predictions = np.asarray([2, 1, 0, 3, 5], dtype=np.int32)
    with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError,
                                             "`predictions`.*out of bound"):
      self._testConfMatrix(
          labels=labels, predictions=predictions, num_classes=3, truth=None)

  def testPredictionsNegative(self):
    labels = np.asarray([1, 1, 0, 2, 2], dtype=np.int32)
    predictions = np.asarray([2, 1, 0, -1, -1], dtype=np.int32)
    with self.assertRaisesOpError("`predictions`.*negative values"):
      self._testConfMatrix(
          labels=labels, predictions=predictions, num_classes=3, truth=None)

  @test_util.run_deprecated_v1
  def testInputDifferentSize(self):
    labels = np.asarray([1, 2])
    predictions = np.asarray([1, 2, 3])
    self.assertRaisesRegex(ValueError, "must be equal",
                           confusion_matrix.confusion_matrix, predictions,
                           labels)

  def testOutputIsInt32(self):
    labels = np.arange(2)
    predictions = np.arange(2)
    with self.cached_session():
      cm = confusion_matrix.confusion_matrix(
          labels, predictions, dtype=dtypes.int32)
      tf_cm = self.evaluate(cm)
    self.assertEqual(tf_cm.dtype, np.int32)

  def testOutputIsInt64(self):
    labels = np.arange(2)
    predictions = np.arange(2)
    with self.cached_session():
      cm = confusion_matrix.confusion_matrix(
          labels, predictions, dtype=dtypes.int64)
      tf_cm = self.evaluate(cm)
    self.assertEqual(tf_cm.dtype, np.int64)


class RemoveSqueezableDimensionsTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBothScalarShape(self):
    label_values = 1.0
    prediction_values = 0.0
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder))

    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSameShape(self):
    label_values = np.ones(shape=(2, 3, 1))
    prediction_values = np.zeros_like(label_values)
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder))

    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSameShapeExpectedRankDiff0(self):
    label_values = np.ones(shape=(2, 3, 1))
    prediction_values = np.zeros_like(label_values)
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values, expected_rank_diff=0))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder, expected_rank_diff=0))

    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSqueezableLabels(self):
    label_values = np.ones(shape=(2, 3, 1))
    prediction_values = np.zeros(shape=(2, 3))
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder))

    expected_label_values = np.reshape(label_values, newshape=(2, 3))
    with self.cached_session():
      self.assertAllEqual(expected_label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          expected_label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSqueezableLabelsExpectedRankDiffPlus1(self):
    label_values = np.ones(shape=(2, 3, 1))
    prediction_values = np.zeros(shape=(2, 3, 5))
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values, expected_rank_diff=1))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder, expected_rank_diff=1))

    expected_label_values = np.reshape(label_values, newshape=(2, 3))
    with self.cached_session():
      self.assertAllEqual(expected_label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          expected_label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSqueezablePredictions(self):
    label_values = np.ones(shape=(2, 3))
    prediction_values = np.zeros(shape=(2, 3, 1))
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder))

    expected_prediction_values = np.reshape(prediction_values, newshape=(2, 3))
    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(expected_prediction_values,
                          self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          expected_prediction_values,
          dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSqueezablePredictionsExpectedRankDiffMinus1(self):
    label_values = np.ones(shape=(2, 3, 5))
    prediction_values = np.zeros(shape=(2, 3, 1))
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values, expected_rank_diff=-1))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder, expected_rank_diff=-1))

    expected_prediction_values = np.reshape(prediction_values, newshape=(2, 3))
    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(expected_prediction_values,
                          self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          expected_prediction_values,
          dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testUnsqueezableLabels(self):
    label_values = np.ones(shape=(2, 3, 2))
    prediction_values = np.zeros(shape=(2, 3))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    _, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(labels_placeholder,
                                                      predictions_placeholder))

    with self.cached_session():
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testUnsqueezablePredictions(self):
    label_values = np.ones(shape=(2, 3))
    prediction_values = np.zeros(shape=(2, 3, 2))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, _ = (
        confusion_matrix.remove_squeezable_dimensions(labels_placeholder,
                                                      predictions_placeholder))

    with self.cached_session():
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))


if __name__ == "__main__":
  test.main()
