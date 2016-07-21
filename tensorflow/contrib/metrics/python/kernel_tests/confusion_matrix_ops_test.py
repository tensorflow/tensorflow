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
import tensorflow as tf

from tensorflow.python.framework import dtypes


class ConfusionMatrixTest(tf.test.TestCase):

  def _testConfMatrix(self, predictions, labels, truth):
    with self.test_session():
      ans = tf.contrib.metrics.confusion_matrix(predictions, labels)
      tf_ans = ans.eval()
      self.assertAllClose(tf_ans, truth, atol=1e-10)

  def _testBasic(self, dtype):
    predictions = np.arange(5, dtype=dtype)
    labels = np.arange(5, dtype=dtype)

    truth = np.asarray(
        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]],
        dtype=dtype)

    self._testConfMatrix(
        predictions=predictions,
        labels=labels,
        truth=truth)

  def testInt32Basic(self):
    self._testBasic(dtype=np.int32)

  def testInt64Basic(self):
    self._testBasic(dtype=np.int64)

  def _testDiffentLabelsInPredictionAndTarget(self, dtype):
    predictions = np.asarray([1, 2, 3], dtype=dtype)
    labels = np.asarray([4, 5, 6], dtype=dtype)

    truth = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        dtype=dtype)

    self._testConfMatrix(
        predictions=predictions,
        labels=labels,
        truth=truth)

  def testInt32DifferentLabels(self, dtype=np.int32):
    self._testDiffentLabelsInPredictionAndTarget(dtype)

  def testInt64DifferentLabels(self, dtype=np.int64):
    self._testDiffentLabelsInPredictionAndTarget(dtype)

  def _testMultipleLabels(self, dtype):
    predictions = np.asarray([1, 1, 2, 3, 5, 6, 1, 2, 3, 4], dtype=dtype)
    labels = np.asarray([1, 1, 2, 3, 5, 1, 3, 6, 3, 1], dtype=dtype)

    truth = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 2, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 2, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 0, 0]],
        dtype=dtype)

    self._testConfMatrix(
        predictions=predictions,
        labels=labels,
        truth=truth)

  def testInt32MultipleLabels(self, dtype=np.int32):
    self._testMultipleLabels(dtype)

  def testInt64MultipleLabels(self, dtype=np.int64):
    self._testMultipleLabels(dtype)

  def testInvalidRank(self):
    predictions = np.asarray([[1, 2, 3]])
    labels = np.asarray([1, 2, 3])
    self.assertRaisesRegexp(
        ValueError, "an not squeeze dim",
        tf.contrib.metrics.confusion_matrix, predictions, labels)

    predictions = np.asarray([1, 2, 3])
    labels = np.asarray([[1, 2, 3]])
    self.assertRaisesRegexp(
        ValueError, "an not squeeze dim",
        tf.contrib.metrics.confusion_matrix, predictions, labels)

  def testInputDifferentSize(self):
    predictions = np.asarray([1, 2, 3])
    labels = np.asarray([1, 2])
    self.assertRaisesRegexp(
        ValueError, "are not compatible",
        tf.contrib.metrics.confusion_matrix, predictions, labels)

  def testOutputIsInt32(self):
    predictions = np.arange(2)
    labels = np.arange(2)
    with self.test_session():
      cm = tf.contrib.metrics.confusion_matrix(
          predictions, labels, dtype=dtypes.int32)
      tf_cm = cm.eval()
    self.assertEqual(tf_cm.dtype, np.int32)

  def testOutputIsInt64(self):
    predictions = np.arange(2)
    labels = np.arange(2)
    with self.test_session():
      cm = tf.contrib.metrics.confusion_matrix(
          predictions, labels, dtype=dtypes.int64)
      tf_cm = cm.eval()
    self.assertEqual(tf_cm.dtype, np.int64)


if __name__ == "__main__":
  tf.test.main()
