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
"""Tests for metric_ops_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.metrics.python.ops import metric_ops_util


class RemoveSqueezableDimensionsTest(tf.test.TestCase):

  def testRemoveSqueezableDimensions(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=False,
        labels_have_static_shape=False, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=False,
        labels_have_static_shape=False, labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_staticLabel(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=False,
        labels_have_static_shape=True, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_staticLabel_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=False,
        labels_have_static_shape=True, labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_extraPredictionDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=True,
        labels_have_static_shape=False, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_extraPredictionDim_staticLabel(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=False, predictions_have_extra_dim=True,
        labels_have_static_shape=True, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_staticPrediction(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=False,
        labels_have_static_shape=False, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_staticPrediction_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=False,
        labels_have_static_shape=False, labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_static(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=False,
        labels_have_static_shape=True, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_static_extraLabelDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=False,
        labels_have_static_shape=True, labels_have_extra_dim=True)

  def testRemoveSqueezableDimensions_staticPrediction_extraPredictionDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=True,
        labels_have_static_shape=False, labels_have_extra_dim=False)

  def testRemoveSqueezableDimensions_static_extraPredictionDim(self):
    self._testRemoveSqueezableDimensions(
        predictions_have_static_shape=True, predictions_have_extra_dim=True,
        labels_have_static_shape=True, labels_have_extra_dim=False)

  # TODO(ptucker): Replace this with parameterized test.
  def _testRemoveSqueezableDimensions(
      self,
      predictions_have_static_shape,
      predictions_have_extra_dim,
      labels_have_static_shape,
      labels_have_extra_dim):
    assert not (predictions_have_extra_dim and labels_have_extra_dim)
    predictions_value = (0, 1, 1, 0, 0, 1, 0)
    labels_value = (0, 0, 1, 1, 0, 0, 0)

    input_predictions_value = (
        [[p] for p in predictions_value] if predictions_have_extra_dim else
        predictions_value)
    input_labels_value = (
        [[l] for l in labels_value] if labels_have_extra_dim else labels_value)

    with tf.Graph().as_default() as g:
      feed_dict = {}
      if predictions_have_static_shape:
        predictions = tf.constant(input_predictions_value, dtype=tf.int32)
      else:
        predictions = tf.placeholder(dtype=tf.int32, name='predictions')
        feed_dict[predictions] = input_predictions_value
      if labels_have_static_shape:
        labels = tf.constant(input_labels_value, dtype=tf.int32)
      else:
        labels = tf.placeholder(dtype=tf.int32, name='labels')
        feed_dict[labels] = input_labels_value

      squeezed_predictions, squeezed_labels = (
          metric_ops_util.remove_squeezable_dimensions(predictions, labels))
      with self.test_session(g):
        tf.initialize_local_variables().run()
        self.assertAllClose(
            predictions_value, squeezed_predictions.eval(feed_dict=feed_dict))
        self.assertAllClose(
            labels_value, squeezed_labels.eval(feed_dict=feed_dict))
