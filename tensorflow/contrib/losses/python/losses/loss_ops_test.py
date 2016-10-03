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

"""Tests for contrib.losses.python.losses.loss_ops."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import numpy as np
import tensorflow as tf


class AbsoluteDifferenceLossTest(tf.test.TestCase):

  def setUp(self):
    self._predictions = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    self._targets = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.absolute_difference(
            self._predictions, self._predictions, weight=None)

  def testAllCorrectNoLossWeight(self):
    loss = tf.contrib.losses.absolute_difference(
        self._predictions, self._predictions)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testNonZeroLoss(self):
    loss = tf.contrib.losses.absolute_difference(
        self._predictions, self._targets)
    with self.test_session():
      self.assertAlmostEqual(5.5, loss.eval(), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.absolute_difference(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(5.5 * weight, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.absolute_difference(
        self._predictions, self._targets, tf.constant(weight))
    with self.test_session():
      self.assertAlmostEqual(5.5 * weight, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weight = tf.constant([1.2, 0.0], shape=[2,])
    loss = tf.contrib.losses.absolute_difference(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(5.6, loss.eval(), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeights(self):
    weight = tf.constant([1.2, 0.0], shape=[2, 1])
    loss = tf.contrib.losses.absolute_difference(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(5.6, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeights(self):
    weight = tf.constant([3, 6, 5, 0, 4, 2], shape=[2, 3])
    loss = tf.contrib.losses.absolute_difference(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(16.6, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weight = tf.constant([0, 0, 0, 0, 0, 2], shape=[2, 3])
    loss = tf.contrib.losses.absolute_difference(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(6.0, loss.eval(), 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    weight = tf.zeros((2, 3))
    loss = tf.contrib.losses.absolute_difference(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class SoftmaxCrossEntropyLossTest(tf.test.TestCase):

  def testNoneWeightRaisesValueError(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.softmax_cross_entropy(logits, labels, weight=None)

  def testAllCorrect(self):
    with self.test_session():
      logits = tf.constant([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
      labels = tf.constant([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
      loss = tf.contrib.losses.softmax_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllWrong(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]])

    with self.test_session():
      loss = tf.contrib.losses.softmax_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]])
    weight = 2.3
    with self.test_session():
      loss = tf.contrib.losses.softmax_cross_entropy(logits, labels, weight)
      self.assertAlmostEqual(loss.eval(), weight * 10.0, 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]])
    weight = 2.3
    with self.test_session():
      loss = tf.contrib.losses.softmax_cross_entropy(
          logits, labels, tf.constant(weight))
      self.assertAlmostEqual(loss.eval(), weight * 10.0, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]])
    weight = tf.constant([1.2, 3.4, 5.6], shape=[3])
    with self.test_session():
      loss = tf.contrib.losses.softmax_cross_entropy(logits, labels, weight)
      self.assertAlmostEqual(loss.eval(), (1.2 + 3.4 + 5.6) * 10.0 / 3.0, 3)

  def testAllWrongAllWeightsMissing(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]])
    weight = tf.constant([0, 0, 0], shape=[3])
    with self.test_session():
      loss = tf.contrib.losses.softmax_cross_entropy(logits, labels, weight)
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testSomeWeightsMissing(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]])
    weight = tf.constant([1.2, 0, 0], shape=[3])
    with self.test_session():
      loss = tf.contrib.losses.softmax_cross_entropy(logits, labels, weight)
      self.assertAlmostEqual(loss.eval(), 12.0, 3)

  def testSoftmaxWithMeasurementSpecificWeightsRaisesException(self):
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0],
                            [-100.0, 100.0, -100.0],
                            [-100.0, -100.0, 100.0]])
      labels = tf.constant([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
      weight = tf.constant([[3, 4, 5],
                            [2, 6, 0],
                            [8, 0, 1]])

      with self.assertRaises(ValueError):
        tf.contrib.losses.softmax_cross_entropy(
            logits, labels, weight=weight).eval()

  def testSoftmaxLabelSmoothing(self):
    with self.test_session():
      # Softmax Cross Entropy Loss is:
      #   -\sum_i p_i \log q_i
      # where for a softmax activation
      # \log q_i = x_i - \log \sum_j \exp x_j
      #          = x_i - x_max - \log \sum_j \exp (x_j - x_max)
      # For our activations, [100, -100, -100] the log partion function becomes
      # \log ( exp(0) + exp(-200) + exp(-200) ) = 0
      # so our log softmaxes become: [0, -200, -200]
      # so our cross entropy loss is:
      # -(1 - L + L/n) * 0 + 400 * L/n = 400 L/n
      logits = tf.constant([[100.0, -100.0, -100.0]])
      labels = tf.constant([[1, 0, 0]])
      label_smoothing = 0.1
      loss = tf.contrib.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=label_smoothing)
      self.assertEquals(loss.op.name, 'softmax_cross_entropy_loss/value')
      expected_value = 400.0 * label_smoothing / 3.0
      self.assertAlmostEqual(loss.eval(), expected_value, 3)


class SparseSoftmaxCrossEntropyLossTest(tf.test.TestCase):

  def testNoneWeightRaisesValueError(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[0], [1], [2]])
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.sparse_softmax_cross_entropy(
            logits, labels, weight=None)

  def testAllCorrectInt32Labels(self):
    with self.test_session():
      logits = tf.constant([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
      labels = tf.constant([[0], [1], [2]], dtype=tf.int32)
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllCorrectInt64Labels(self):
    with self.test_session():
      logits = tf.constant([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
      labels = tf.constant([[0], [1], [2]], dtype=tf.int64)
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllCorrectNonColumnLabels(self):
    with self.test_session():
      logits = tf.constant([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
      labels = tf.constant([0, 1, 2])
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllWrongInt32Labels(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[2], [0], [1]], dtype=tf.int32)

    with self.test_session():
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testAllWrongInt64Labels(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[2], [0], [1]], dtype=tf.int64)

    with self.test_session():
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testAllWrongNonColumnLabels(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([2, 0, 1])

    with self.test_session():
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[2], [0], [1]])
    weight = 2.3
    with self.test_session():
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(
          logits, labels, weight)
      self.assertAlmostEqual(loss.eval(), weight * 10.0, 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[2], [0], [1]])
    weight = 2.3
    with self.test_session():
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(
          logits, labels, tf.constant(weight))
      self.assertAlmostEqual(loss.eval(), weight * 10.0, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[2], [0], [1]])
    weight = tf.constant([1.2, 3.4, 5.6], shape=[3])
    with self.test_session():
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(
          logits, labels, weight)
      self.assertAlmostEqual(loss.eval(), (1.2 + 3.4 + 5.6) * 10.0 / 3.0, 3)

  def testNonZeroLossWithColumnWeights(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[2], [0], [1]])
    weight = tf.constant([[1.2], [3.4], [5.6]])
    with self.test_session():
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(
          logits, labels, weight)
      self.assertAlmostEqual(loss.eval(), (1.2 + 3.4 + 5.6) * 10.0 / 3.0, 3)

  def testAllWrongAllWeightsMissing(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[2], [0], [1]])
    weight = tf.constant([0, 0, 0], shape=[3])
    with self.test_session():
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(
          logits, labels, weight)
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testSomeWeightsMissing(self):
    logits = tf.constant([[10.0, 0.0, 0.0],
                          [0.0, 10.0, 0.0],
                          [0.0, 0.0, 10.0]])
    labels = tf.constant([[2], [0], [1]])
    weight = tf.constant([1.2, 0, 0], shape=[3])
    with self.test_session():
      loss = tf.contrib.losses.sparse_softmax_cross_entropy(
          logits, labels, weight)
      self.assertAlmostEqual(loss.eval(), 12.0, 3)

  def testMeasurementSpecificWeightsRaisesException(self):
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0],
                            [-100.0, 100.0, -100.0],
                            [-100.0, -100.0, 100.0]])
      labels = tf.constant([[0], [1], [2]])
      weight = tf.constant([[3, 4, 5],
                            [2, 6, 0],
                            [8, 0, 1]])

      with self.assertRaises(ValueError):
        tf.contrib.losses.sparse_softmax_cross_entropy(
            logits, labels, weight=weight).eval()

  def testInconsistentWeightSizeRaisesException(self):
    """The weight tensor has incorrect number of elements."""
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0],
                            [-100.0, 100.0, -100.0],
                            [-100.0, -100.0, 100.0]])
      labels = tf.constant([[0], [1], [2]])
      weight = tf.constant([1.2, 3.4, 5.6, 7.8])

      with self.assertRaises(ValueError):
        tf.contrib.losses.sparse_softmax_cross_entropy(
            logits, labels, weight=weight).eval()

  def testInconsistentLabelSizeRaisesException(self):
    """The label tensor has incorrect number of elements."""
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0],
                            [-100.0, 100.0, -100.0],
                            [-100.0, -100.0, 100.0]])
      labels = tf.constant([[0], [1], [2], [3]])
      weight = tf.constant([1.2, 3.4, 5.6])

      with self.assertRaises(ValueError):
        tf.contrib.losses.sparse_softmax_cross_entropy(
            logits, labels, weight=weight).eval()

  def testInconsistentWeightShapeRaisesException(self):
    """The weight tensor has incorrect shape."""
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0, -100.0],
                            [-100.0, 100.0, -100.0, -100.0],
                            [-100.0, -100.0, 100.0, -100.0],
                            [-100.0, -100.0, -100.0, 100.0]])
      labels = tf.constant([[0], [1], [2], [3]])
      weight = tf.constant([[1.2, 3.4], [5.6, 7.8]])

      with self.assertRaises(ValueError):
        tf.contrib.losses.sparse_softmax_cross_entropy(
            logits, labels, weight=weight).eval()

  def testInconsistentLabelShapeRaisesException(self):
    """The label tensor has incorrect shape."""
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0, -100.0],
                            [-100.0, 100.0, -100.0, -100.0],
                            [-100.0, -100.0, 100.0, -100.0],
                            [-100.0, -100.0, -100.0, 100.0]])
      labels = tf.constant([[0, 1], [2, 3]])
      weight = tf.constant([1.2, 3.4, 5.6, 7.8])

      with self.assertRaises(tf.errors.InvalidArgumentError):
        tf.contrib.losses.sparse_softmax_cross_entropy(
            logits, labels, weight=weight).eval()


class SigmoidCrossEntropyLossTest(tf.test.TestCase):

  def testAllCorrectSigmoid(self):
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0],
                            [-100.0, 100.0, -100.0],
                            [-100.0, -100.0, 100.0]])
      labels = tf.constant([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
      loss = tf.contrib.losses.sigmoid_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testLossWithSingleDimPlaceholderForLogitsAndWeights1(self):
    logits = tf.placeholder(tf.float32, shape=(None, 1))
    labels = tf.placeholder(tf.float32, shape=(None, 1))
    weight = tf.ones_like(logits, dtype=tf.float32)

    loss = tf.contrib.losses.sigmoid_cross_entropy(logits, labels, weight)

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={
          logits: np.ones((32, 1)),
          labels: np.ones((32, 1)),
      })
      self.assertAlmostEqual(loss, 0.313, 3)

  def testLossWithSingleDimPlaceholderForLogitsAndWeights2(self):
    logits = tf.placeholder(tf.float32, shape=(None, 2))
    labels = tf.placeholder(tf.float32, shape=(None, 2))
    weight = tf.ones_like(logits, dtype=tf.float32)

    loss = tf.contrib.losses.sigmoid_cross_entropy(logits, labels, weight)

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={
          logits: np.ones((32, 2)),
          labels: np.ones((32, 2)),
      })
      self.assertAlmostEqual(loss, 0.313, 3)

  def testAllWrongSigmoid(self):
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0],
                            [-100.0, 100.0, -100.0],
                            [-100.0, -100.0, 100.0]])
      labels = tf.constant([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])
      loss = tf.contrib.losses.sigmoid_cross_entropy(logits, labels)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 600.0 / 9.0, 3)

  def testAllWrongSigmoidWithMeasurementSpecificWeights(self):
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0],
                            [-100.0, 100.0, -100.0],
                            [-100.0, -100.0, 100.0]])
      labels = tf.constant([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])
      weight = tf.constant([[3, 4, 5],
                            [2, 6, 0],
                            [8, 0, 1]])
      loss = tf.contrib.losses.sigmoid_cross_entropy(
          logits, labels, weight=weight)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 1700.0 / 7.0, 3)

  def testMultiCorrectSigmoid(self):
    logits = tf.constant([[100.0, -100.0, 100.0],
                          [100.0, 100.0, -100.0],
                          [-100.0, 100.0, 100.0]])
    labels = tf.constant([[1, 0, 1],
                          [1, 1, 0],
                          [0, 1, 1]])
    loss = tf.contrib.losses.sigmoid_cross_entropy(logits, labels)
    self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')

    with self.test_session():
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testSigmoidLabelSmoothingCorrect(self):
    with self.test_session():
      logits = tf.constant([[100.0, -100.0, -100.0]])
      labels = tf.constant([[1, 0, 1]])
      # Sigmoid cross entropy loss is:
      #   max(x,0) - x*z + log(1 + exp(-abs(x)))
      # The new labels are:
      #    z' = z * (1 - L) + 0.5 L
      #    1 -> 1 - 0.5 L
      #    0 -> 0.5 L
      # here we expect:
      # 1/3 * (100 - 100 * (1 - 0.5 L)  + 0
      #       + 0  + 100 * (0.5 L)      + 0
      #       + 0  + 100 * (1 - 0.5 L)  + 0)
      # = 1/3 * (100 + 50 L)
      label_smoothing = 0.1
      loss = tf.contrib.losses.sigmoid_cross_entropy(
          logits, labels, label_smoothing=label_smoothing)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
      self.assertAlmostEqual(loss.eval(), expected_value, 3)

  def testSigmoidLabelSmoothingEqualsSoftmaxTwoLabel(self):
    with self.test_session():
      label_smoothing = 0.1
      sigmoid_logits = tf.constant([[100.0, -100.0, -100.0]])
      sigmoid_labels = tf.constant([[1, 0, 1]])
      sigmoid_loss = tf.contrib.losses.sigmoid_cross_entropy(
          sigmoid_logits, sigmoid_labels, label_smoothing=label_smoothing)

      softmax_logits = tf.constant([[0.0, 100.0], [100.0, 0.0], [100.0, 0.0]])
      softmax_labels = tf.constant([[0, 1], [1, 0], [0, 1]])
      softmax_loss = tf.contrib.losses.softmax_cross_entropy(
          softmax_logits, softmax_labels, label_smoothing=label_smoothing)
      self.assertAlmostEqual(sigmoid_loss.eval(), softmax_loss.eval(), 3)


class LogLossTest(tf.test.TestCase):

  def setUp(self):
    predictions = np.asarray([.9, .2, .2, .8, .4, .6]).reshape((2, 3))
    targets = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]).reshape((2, 3))

    self._np_predictions = predictions
    self._np_targets = targets

    epsilon = 1e-7
    self._expected_losses = np.multiply(
        targets, np.log(predictions + epsilon)) + np.multiply(
            1 - targets, np.log(1 - predictions + epsilon))

    self._predictions = tf.constant(predictions)
    self._targets = tf.constant(targets)

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.log_loss(self._targets, self._targets, weight=None)

  def testAllCorrectNoLossWeight(self):
    loss = tf.contrib.losses.log_loss(self._targets, self._targets)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testAllCorrectNoLossWeightWithPlaceholder(self):
    tf_predictions = tf.placeholder(tf.float32, shape=self._np_targets.shape)
    loss = tf.contrib.losses.log_loss(tf_predictions, self._targets)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(feed_dict={
          tf_predictions: self._np_targets}), 3)

  def testNonZeroLoss(self):
    loss = tf.contrib.losses.log_loss(self._predictions, self._targets)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(self._expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.log_loss(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(weight * -np.sum(self._expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.log_loss(
        self._predictions, self._targets, tf.constant(weight))
    with self.test_session():
      self.assertAlmostEqual(weight * -np.sum(self._expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholder(self):
    tf_predictions = tf.placeholder(tf.float32,
                                    shape=self._np_predictions.shape)
    weight = 2.3
    loss = tf.contrib.losses.log_loss(
        tf_predictions, self._targets, tf.constant(weight))
    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weight * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholderWithRankOnly(self):
    tf_predictions = tf.placeholder(tf.float32, shape=[None, None])
    weight = 2.3
    loss = tf.contrib.losses.log_loss(
        tf_predictions, self._targets, tf.constant(weight))
    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weight * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weight = tf.constant([1.2, 3.4], shape=[2])
    expected_losses = np.multiply(
        self._expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    loss = tf.contrib.losses.log_loss(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeightsSomeZero(self):
    weight = tf.constant([1.2, 0], shape=[2])
    expected_losses = np.multiply(
        self._expected_losses,
        np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape((2, 3)))
    loss = tf.contrib.losses.log_loss(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 3.0,
                             loss.eval(), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeightsSomeZero(self):
    weight = tf.constant([1.2, 0], shape=[2, 1])
    expected_losses = np.multiply(
        self._expected_losses,
        np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape((2, 3)))
    loss = tf.contrib.losses.log_loss(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 3.0,
                             loss.eval(), 3)

  def testWeightsWithSameNumDimsButWrongShapeThrowsException(self):
    weight = tf.constant(np.random.normal(size=(2, 4)), shape=[2, 4])
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.log_loss(self._predictions, self._targets, weight)

  def testNonZeroLossWithMeasurementSpecificWeights(self):
    weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weight)

    loss = tf.contrib.losses.log_loss(
        self._predictions,
        self._targets,
        weight=tf.constant(weight, shape=(2, 3)))
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 5.0, loss.eval(), 3)

  def testNonZeroLossWithMeasurementSpecificWeightsWithPlaceholder(self):
    weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weight)

    tf_predictions = tf.placeholder(tf.float32, shape=[2, 3])
    loss = tf.contrib.losses.log_loss(
        tf_predictions,
        self._targets,
        weight=tf.constant(weight, shape=(2, 3)))

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expected_losses) / 5.0, loss, 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weight = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weight)

    loss = tf.contrib.losses.log_loss(
        self._predictions,
        self._targets,
        weight=tf.constant(weight, shape=(2, 3)))
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses), loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZeroWithPlaceholder(self):
    weight = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weight)

    tf_predictions = tf.placeholder(tf.float32, shape=[2, 3])
    tf_weight = tf.constant(weight, shape=(2, 3))
    loss = tf.contrib.losses.log_loss(tf_predictions, self._targets, tf_weight)

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expected_losses), loss, 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    tf_weight = tf.zeros(shape=(2, 3))
    loss = tf.contrib.losses.log_loss(
        self._predictions, self._targets, tf_weight)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class HingeLossTest(tf.test.TestCase):

  def testIncompatibleShapes(self):
    with self.test_session():
      logits = tf.constant([[-1.0], [2.1]])
      target = tf.constant([0.0, 1.0])
      with self.assertRaises(ValueError):
        _ = tf.contrib.losses.hinge_loss(logits, target).eval()

  def testAllOutsideMargin(self):
    with self.test_session():
      logits = tf.constant([1.2, -1.4, -1.0, 2.1])
      target = tf.constant([1.0, 0.0, 0.0, 1.0])
      loss = tf.contrib.losses.hinge_loss(logits, target)
      self.assertAllClose(loss.eval(), [0.0, 0.0, 0.0, 0.0], atol=1e-3)

  def testSomeInsideMargin(self):
    with self.test_session():
      logits = tf.constant([[-0.7], [-1.4], [1.4], [0.6]])
      target = tf.constant([[0.0], [0.0], [1.0], [1.0]])
      loss = tf.contrib.losses.hinge_loss(logits, target)
      # Examples 1 and 4 are on the correct side of the hyperplane but within
      # the margin so they incur some (small) loss.
      self.assertAllClose(loss.eval(), [[0.3], [0.0], [0.0], [0.4]], atol=1e-3)

  def testSomeMisclassified(self):
    with self.test_session():
      logits = tf.constant([[[1.2], [0.4], [-1.0], [-1.1]]])
      target = tf.constant([[[1.0], [0.0], [0.0], [1.0]]])
      loss = tf.contrib.losses.hinge_loss(logits, target)
      # Examples 2 and 4 are on the wrong side of the hyperplane so they incur
      # some (fairly large) loss.
      self.assertAllClose(
          loss.eval(), [[[0.0], [1.4], [0.0], [2.1]]], atol=1e-3)


class MeanSquaredErrorTest(tf.test.TestCase):

  def setUp(self):
    self._predictions = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    self._targets = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))

  def testDeprecatedName(self):
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._predictions)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.mean_squared_error(
            self._predictions, self._predictions, weight=None)

  def testAllCorrectNoLossWeight(self):
    loss = tf.contrib.losses.mean_squared_error(
        self._predictions, self._predictions)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testNonZeroLoss(self):
    loss = tf.contrib.losses.mean_squared_error(
        self._predictions, self._targets)
    with self.test_session():
      self.assertAlmostEqual(49.5, loss.eval(), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.mean_squared_error(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(49.5 * weight, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.mean_squared_error(
        self._predictions, self._targets, tf.constant(weight))
    with self.test_session():
      self.assertAlmostEqual(49.5 * weight, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weight = tf.constant([1.2, 3.4], shape=[2,])
    loss = tf.contrib.losses.mean_squared_error(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(767.8 / 6.0, loss.eval(), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeights(self):
    weight = tf.constant([1.2, 3.4], shape=[2, 1])
    loss = tf.contrib.losses.mean_squared_error(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(767.8 / 6.0, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeights(self):
    weight = tf.constant([3, 6, 5, 0, 4, 2], shape=[2, 3])
    loss = tf.contrib.losses.mean_squared_error(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(587 / 5.0, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weight = tf.constant([0, 0, 0, 0, 0, 2], shape=[2, 3])
    loss = tf.contrib.losses.mean_squared_error(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(18.0, loss.eval(), 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    weight = tf.zeros((2, 3))
    loss = tf.contrib.losses.mean_squared_error(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class MeanPairwiseSquaresErrorTest(tf.test.TestCase):

  def setUp(self):
    self._predictions = np.array([[4, 8, 12],
                                  [8, 1, 3]])
    self._targets = np.array([[1, 9, 2],
                              [-5, -5, 7]])

    batch_size, dims = self._targets.shape

    # Compute the expected loss 'manually'.
    total = np.zeros((batch_size, 1))
    for b in range(batch_size):
      for i in range(dims):
        for j in range(dims):
          x = self._predictions[b, i].item() - self._predictions[b, j].item()
          y = self._targets[b, i].item() - self._targets[b, j].item()
          tmp = (x-y) * (x-y)
          total[b] += tmp

    self._expected_losses = np.divide(total, 9.0)

  def testDeprecatedName(self):
    loss = tf.contrib.losses.sum_of_pairwise_squares(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets))
    with self.test_session():
      self.assertAlmostEqual(np.sum(self._expected_losses), loss.eval(), 3)

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.mean_pairwise_squared_error(
            predictions=tf.constant(self._targets),
            targets=tf.constant(self._targets),
            weight=None)

  def testAllCorrectNoLossWeight(self):
    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf.constant(self._targets),
        targets=tf.constant(self._targets))
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testNonZeroLoss(self):
    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets))
    with self.test_session():
      self.assertAlmostEqual(np.sum(self._expected_losses), loss.eval(), 3)

  def testGradientWithZeroWeight(self):
    with tf.Graph().as_default():
      tf.set_random_seed(0)

      inputs = tf.ones((2, 3))
      weights = tf.get_variable('weights',
                                shape=[3, 4],
                                initializer=tf.truncated_normal_initializer())
      predictions = tf.matmul(inputs, weights)

      optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
      loss = tf.contrib.losses.mean_pairwise_squared_error(
          predictions,
          predictions,
          0)

      gradients_to_variables = optimizer.compute_gradients(loss)

      init_op = tf.initialize_all_variables()

      with self.test_session() as sess:
        sess.run(init_op)
        for grad, _ in gradients_to_variables:
          np_grad = sess.run(grad)
          self.assertFalse(np.isnan(np_grad).any())

  def testNonZeroLossWithPythonScalarWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        weight=weight)
    with self.test_session():
      self.assertAlmostEqual(weight * np.sum(self._expected_losses),
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        weight=tf.constant(weight))
    with self.test_session():
      self.assertAlmostEqual(weight * np.sum(self._expected_losses),
                             loss.eval(), 3)

  def testNonZeroLossWithScalarZeroWeight(self):
    weight = 0
    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        weight=tf.constant(weight))
    with self.test_session():
      self.assertAlmostEqual(0, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeightWithPlaceholder(self):
    weight = 2.3
    tf_predictions = tf.placeholder(tf.float32, shape=self._predictions.shape)
    tf_targets = tf.placeholder(tf.float32, shape=self._targets.shape)
    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf_predictions,
        targets=tf_targets,
        weight=tf.constant(weight))
    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={
          tf_predictions: self._predictions,
          tf_targets: self._targets,
      })
      self.assertAlmostEqual(weight * np.sum(self._expected_losses), loss, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weight = np.asarray([2.0, 1.0]).reshape((2, 1))
    expected_losses = np.multiply(weight, self._expected_losses)

    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        weight=tf.constant(weight, shape=[2]))
    with self.test_session():
      self.assertAlmostEqual(np.sum(expected_losses), loss.eval(), 3)

  def testZeroLossWithOneDimBatchZeroWeights(self):
    weight = np.asarray([0.0, 0.0]).reshape((2, 1))
    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        weight=tf.constant(weight, shape=[2]))
    with self.test_session():
      self.assertAlmostEqual(0, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeightsAndPlaceholders(self):
    weight = np.asarray([1.2, 3.4]).reshape((2, 1))
    expected_losses = np.multiply(weight, self._expected_losses)

    tf_predictions = tf.placeholder(tf.float32, shape=self._predictions.shape)
    tf_targets = tf.placeholder(tf.int32, shape=self._targets.shape)
    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf_predictions,
        targets=tf_targets,
        weight=tf.constant(weight, shape=[2]))

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={
          tf_predictions: self._predictions,
          tf_targets: self._targets,
      })
      self.assertAlmostEqual(np.sum(expected_losses), loss, 3)

  def testLossWithAllZeroBatchSpecificWeights(self):
    weight = np.zeros((2, 1))
    loss = tf.contrib.losses.mean_pairwise_squared_error(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        weight=tf.constant(weight, shape=[2]))
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class CosineDistanceLossTest(tf.test.TestCase):

  def setUp(self):
    self._predictions = np.asarray([[1, 0, 0],  # Batch 1
                                    [0, 0, -1],
                                    [1, 0, 0],  # Batch 2
                                    [1, 0, 0],
                                    [0, 0, -1],  # Batch 3
                                    [1, 0, 0]]).reshape((3, 2, 3))

    self._targets = np.asarray([[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]]).reshape((3, 2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.cosine_distance(
            predictions=tf.constant(self._targets),
            targets=tf.constant(self._targets),
            dim=2,
            weight=None)

  def testAllCorrectNoWeights(self):
    loss = tf.contrib.losses.cosine_distance(
        predictions=tf.constant(self._targets),
        targets=tf.constant(self._targets),
        dim=2)
    with self.test_session():
      self.assertAlmostEqual(0, loss.eval(), 5)

  def testPartiallyCorrectWithIntegerValues(self):
    loss = tf.contrib.losses.cosine_distance(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        dim=2)
    with self.test_session():
      self.assertAlmostEqual(1, loss.eval(), 5)

  def testPartiallyCorrectFloatingPointValues(self):
    predictions = np.matrix((
        '0.819031913261206 0.567041924552012 0.087465312324590;'
        '-0.665139432070255 -0.739487441769973 -0.103671883216994;'
        '0.707106781186548 -0.707106781186548 0'))
    targets = np.matrix((
        '0.819031913261206 0.567041924552012 0.087465312324590;'
        '0.665139432070255 0.739487441769973 0.103671883216994;'
        '0.707106781186548 0.707106781186548 0'))

    tf_preds = tf.constant(predictions, shape=(3, 1, 3), dtype=tf.float32)
    tf_targets = tf.constant(targets, shape=(3, 1, 3), dtype=tf.float32)
    loss = tf.contrib.losses.cosine_distance(tf_preds, tf_targets, dim=2)

    with self.test_session():
      self.assertAlmostEqual(1.0, loss.eval(), 5)

  def testSampleSpecificWeights(self):
    loss = tf.contrib.losses.cosine_distance(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        dim=2,
        weight=tf.constant([1, 0, 0]))
    with self.test_session():
      self.assertEqual(1.0, loss.eval())

  def testMeasurementSpecificWeights(self):
    loss = tf.contrib.losses.cosine_distance(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        dim=2,
        weight=tf.constant([1, 0, 0, 1, 1, 1], shape=(3, 2)))
    with self.test_session():
      self.assertEqual(3.0 / 4.0, loss.eval())

  def testValueErrorThrownWithShapelessPlaceholder(self):
    tf_predictions = tf.placeholder(tf.float32)
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.cosine_distance(
            predictions=tf_predictions,
            targets=tf.constant(self._targets),
            dim=2,
            weight=tf.constant([1, 0, 0, 1, 1, 1], shape=(3, 2)))

  def testMeasurementSpecificWeightsWithPlaceholderWithShape(self):
    tf_predictions = tf.placeholder(tf.float32, shape=self._targets.shape)
    loss = tf.contrib.losses.cosine_distance(
        predictions=tf_predictions,
        targets=tf.constant(self._targets),
        dim=2,
        weight=tf.constant([1, 0, 0, 1, 1, 1], shape=(3, 2)))
    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._predictions})
      self.assertEqual(3.0 / 4.0, loss)

  def testZeroLossWhenAllSampleSpecificWeightsAreZero(self):
    loss = tf.contrib.losses.cosine_distance(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        dim=2,
        weight=tf.zeros((3,)))
    with self.test_session():
      self.assertEqual(0, loss.eval())

  def testZeroLossWhenAllMeasurementSpecificWeightsAreZero(self):
    loss = tf.contrib.losses.cosine_distance(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        dim=2,
        weight=tf.zeros((3, 2)))
    with self.test_session():
      self.assertEqual(0, loss.eval())


class ComputeWeightedLossTest(tf.test.TestCase):

  def testHingeLoss(self):
    logits = tf.constant([1.2, 0.4, -1.0, -1.1])
    target = tf.constant([1.0, 0.0, 0.0, 1.0])
    losses = tf.contrib.losses.hinge_loss(logits, target)
    self.assertFalse(tf.contrib.losses.get_losses())
    loss = tf.contrib.losses.compute_weighted_loss(losses)
    self.assertTrue(tf.contrib.losses.get_losses())
    with self.test_session():
      self.assertAllClose(losses.eval(), [0.0, 1.4, 0.0, 2.1], atol=1e-3)
      self.assertAllClose(loss.eval(), 3.5/4.0, atol=1e-3)


class AddLossTest(tf.test.TestCase):

  def testAddExternalLoss(self):
    logits = tf.constant([1.2, 0.4, -1.0, -1.1])
    target = tf.constant([1.0, 0.0, 0.0, 1.0])
    losses = tf.contrib.losses.hinge_loss(logits, target)
    self.assertFalse(tf.contrib.losses.get_losses())
    tf.contrib.losses.add_loss(tf.reduce_mean(losses))
    self.assertTrue(tf.contrib.losses.get_losses())
    total_loss = tf.contrib.losses.get_total_loss()
    with self.test_session():
      self.assertAllClose(losses.eval(), [0.0, 1.4, 0.0, 2.1], atol=1e-3)
      self.assertAllClose(total_loss.eval(), 3.5/4.0, atol=1e-3)

  def testNoneLossCollection(self):
    logits = tf.constant([1.2, 0.4, -1.0, -1.1])
    target = tf.constant([1.0, 0.0, 0.0, 1.0])
    losses = tf.contrib.losses.hinge_loss(logits, target)
    self.assertFalse(tf.contrib.losses.get_losses())
    tf.contrib.losses.add_loss(tf.reduce_mean(losses), loss_collection=None)
    self.assertFalse(tf.contrib.losses.get_losses())
    with self.test_session():
      self.assertAllClose(losses.eval(), [0.0, 1.4, 0.0, 2.1], atol=1e-3)

  def testNoCollectLosses(self):
    logits = tf.constant([1.2, 0.4, -1.0, -1.1])
    target = tf.constant([1.0, 0.0, 0.0, 1.0])
    self.assertFalse(tf.contrib.losses.get_losses())
    with tf.contrib.framework.arg_scope([tf.contrib.losses.add_loss],
                                        loss_collection=None):
      tf.contrib.losses.absolute_difference(logits, target)
      tf.contrib.losses.log_loss(logits, target)
      tf.contrib.losses.mean_squared_error(logits, target)
      tf.contrib.losses.sigmoid_cross_entropy(logits, target)
      tf.contrib.losses.softmax_cross_entropy(logits, target)
    self.assertFalse(tf.contrib.losses.get_losses())

if __name__ == '__main__':
  tf.test.main()
