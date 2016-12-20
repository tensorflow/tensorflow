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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ConfusionMatrixTest(test.TestCase):

  def _testConfMatrix(self, predictions, labels, truth, weights=None):
    with self.test_session():
      dtype = predictions.dtype
      ans = confusion_matrix.confusion_matrix(
          labels, predictions, dtype=dtype, weights=weights)
      tf_ans = ans.eval()
      self.assertAllClose(tf_ans, truth, atol=1e-10)
      self.assertEqual(tf_ans.dtype, dtype)

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

    self._testConfMatrix(predictions=predictions, labels=labels, truth=truth)

  def testInt32Basic(self):
    self._testBasic(dtype=np.int32)

  def testInt64Basic(self):
    self._testBasic(dtype=np.int64)

  def _testConfMatrixOnTensors(self, tf_dtype, np_dtype):
    with self.test_session() as sess:
      m_neg = array_ops.placeholder(dtype=dtypes.float32)
      m_pos = array_ops.placeholder(dtype=dtypes.float32)
      s = array_ops.placeholder(dtype=dtypes.float32)

      neg = random_ops.random_normal(
          [20], mean=m_neg, stddev=s, dtype=dtypes.float32)
      pos = random_ops.random_normal(
          [20], mean=m_pos, stddev=s, dtype=dtypes.float32)

      data = array_ops.concat_v2([neg, pos], 0)
      data = math_ops.cast(math_ops.round(data), tf_dtype)
      data = math_ops.minimum(math_ops.maximum(data, 0), 1)
      lab = array_ops.concat_v2(
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
      try:
        range_builder = xrange
      except NameError:  # In Python 3.
        range_builder = range
      for i in range_builder(len(d)):
        truth[d[i], l[i]] += 1

      self.assertEqual(cm_out.dtype, np_dtype)
      self.assertAllClose(cm_out, truth, atol=1e-10)

  def _testOnTensors_int32(self):
    self._testConfMatrixOnTensors(dtypes.int32, np.int32)

  def testOnTensors_int64(self):
    self._testConfMatrixOnTensors(dtypes.int64, np.int64)

  def _testDifferentLabelsInPredictionAndTarget(self, dtype):
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

    self._testConfMatrix(predictions=predictions, labels=labels, truth=truth)

  def testInt32DifferentLabels(self, dtype=np.int32):
    self._testDifferentLabelsInPredictionAndTarget(dtype)

  def testInt64DifferentLabels(self, dtype=np.int64):
    self._testDifferentLabelsInPredictionAndTarget(dtype)

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

    self._testConfMatrix(predictions=predictions, labels=labels, truth=truth)

  def testInt32MultipleLabels(self, dtype=np.int32):
    self._testMultipleLabels(dtype)

  def testInt64MultipleLabels(self, dtype=np.int64):
    self._testMultipleLabels(dtype)

  def testWeighted(self):
    predictions = np.arange(5, dtype=np.int32)
    labels = np.arange(5, dtype=np.int32)
    weights = constant_op.constant(np.arange(5, dtype=np.int32))

    truth = np.asarray(
        [[0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 0, 0, 3, 0],
         [0, 0, 0, 0, 4]],
        dtype=np.int32)

    self._testConfMatrix(
        predictions=predictions, labels=labels, weights=weights, truth=truth)

  def testInvalidRank(self):
    predictions = np.asarray([[1, 2, 3]])
    labels = np.asarray([1, 2, 3])
    self.assertRaisesRegexp(ValueError, "an not squeeze dim",
                            confusion_matrix.confusion_matrix, predictions,
                            labels)

    predictions = np.asarray([1, 2, 3])
    labels = np.asarray([[1, 2, 3]])
    self.assertRaisesRegexp(ValueError, "an not squeeze dim",
                            confusion_matrix.confusion_matrix, predictions,
                            labels)

  def testInputDifferentSize(self):
    predictions = np.asarray([1, 2, 3])
    labels = np.asarray([1, 2])
    self.assertRaisesRegexp(ValueError, "must be equal",
                            confusion_matrix.confusion_matrix, predictions,
                            labels)

  def testOutputIsInt32(self):
    predictions = np.arange(2)
    labels = np.arange(2)
    with self.test_session():
      cm = confusion_matrix.confusion_matrix(
          labels, predictions, dtype=dtypes.int32)
      tf_cm = cm.eval()
    self.assertEqual(tf_cm.dtype, np.int32)

  def testOutputIsInt64(self):
    predictions = np.arange(2)
    labels = np.arange(2)
    with self.test_session():
      cm = confusion_matrix.confusion_matrix(
          labels, predictions, dtype=dtypes.int64)
      tf_cm = cm.eval()
    self.assertEqual(tf_cm.dtype, np.int64)


if __name__ == "__main__":
  test.main()
