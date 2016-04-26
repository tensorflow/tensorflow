# Copyright 2016 Google Inc. All Rights Reserved.
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

  def testAllWrongAllMissing(self):
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

  def testSomeMissing(self):
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
      logits = tf.constant([[100.0, -100.0, -100.0]])
      labels = tf.constant([[1, 0, 0]])
      label_smoothing = 0.1
      loss = tf.contrib.losses.sigmoid_cross_entropy(
          logits, labels, label_smoothing=label_smoothing)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      expected_value = 400.0 * label_smoothing / 9.0
      self.assertAlmostEqual(loss.eval(), expected_value, 3)


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
      labels = tf.constant([[1, 0, 0]])
      label_smoothing = 0.1
      loss = tf.contrib.losses.sigmoid_cross_entropy(
          logits, labels, label_smoothing=label_smoothing)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      expected_value = 400.0 * label_smoothing / 9.0
      self.assertAlmostEqual(loss.eval(), expected_value, 3)


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
        tf.contrib.losses.log(self._targets, self._targets, weight=None)

  def testAllCorrectNoLossWeight(self):
    loss = tf.contrib.losses.log(self._targets, self._targets)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testAllCorrectNoLossWeightWithPlaceholder(self):
    tf_predictions = tf.placeholder(tf.float32, shape=self._np_targets.shape)
    loss = tf.contrib.losses.log(tf_predictions, self._targets)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(feed_dict={
          tf_predictions: self._np_targets}), 3)

  def testNonZeroLoss(self):
    loss = tf.contrib.losses.log(self._predictions, self._targets)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(self._expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.log(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(weight * -np.sum(self._expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.log(
        self._predictions, self._targets, tf.constant(weight))
    with self.test_session():
      self.assertAlmostEqual(weight * -np.sum(self._expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholder(self):
    tf_predictions = tf.placeholder(tf.float32,
                                    shape=self._np_predictions.shape)
    weight = 2.3
    loss = tf.contrib.losses.log(
        tf_predictions, self._targets, tf.constant(weight))
    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weight * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholderWithRankOnly(self):
    tf_predictions = tf.placeholder(tf.float32, shape=[None, None])
    weight = 2.3
    loss = tf.contrib.losses.log(
        tf_predictions, self._targets, tf.constant(weight))
    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weight * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weight = tf.constant([1.2, 3.4], shape=[2])
    expectedes = np.multiply(
        self._expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    loss = tf.contrib.losses.log(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expectedes) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeightsSomeZero(self):
    weight = tf.constant([1.2, 0], shape=[2])
    expectedes = np.multiply(
        self._expected_losses,
        np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape((2, 3)))
    loss = tf.contrib.losses.log(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expectedes) / 3.0,
                             loss.eval(), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeightsSomeZero(self):
    weight = tf.constant([1.2, 0], shape=[2, 1])
    expectedes = np.multiply(
        self._expected_losses,
        np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape((2, 3)))
    loss = tf.contrib.losses.log(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expectedes) / 3.0,
                             loss.eval(), 3)

  def testWeightsWithSameNumDimsButWrongShapeThrowsException(self):
    weight = tf.constant(np.random.normal(size=(2, 4)), shape=[2, 4])
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.log(self._predictions, self._targets, weight)

  def testNonZeroLossWithMeasurementSpecificWeights(self):
    weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expectedes = np.multiply(self._expected_losses, weight)

    loss = tf.contrib.losses.log(
        self._predictions,
        self._targets,
        weight=tf.constant(weight, shape=(2, 3)))
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expectedes) / 5.0, loss.eval(), 3)

  def testNonZeroLossWithMeasurementSpecificWeightsWithPlaceholder(self):
    weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expectedes = np.multiply(self._expected_losses, weight)

    tf_predictions = tf.placeholder(tf.float32, shape=[2, 3])
    loss = tf.contrib.losses.log(
        tf_predictions,
        self._targets,
        weight=tf.constant(weight, shape=(2, 3)))

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expectedes) / 5.0, loss, 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weight = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expectedes = np.multiply(self._expected_losses, weight)

    loss = tf.contrib.losses.log(
        self._predictions,
        self._targets,
        weight=tf.constant(weight, shape=(2, 3)))
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expectedes), loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZeroWithPlaceholder(self):
    weight = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expectedes = np.multiply(self._expected_losses, weight)

    tf_predictions = tf.placeholder(tf.float32, shape=[2, 3])
    tf_weight = tf.constant(weight, shape=(2, 3))
    loss = tf.contrib.losses.log(tf_predictions, self._targets, tf_weight)

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expectedes), loss, 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    tf_weight = tf.zeros(shape=(2, 3))
    loss = tf.contrib.losses.log(
        self._predictions, self._targets, tf_weight)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class SumOfSquaresLossTest(tf.test.TestCase):

  def setUp(self):
    self._predictions = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    self._targets = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.sum_of_squares(
            self._predictions, self._predictions, weight=None)

  def testAllCorrectNoLossWeight(self):
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._predictions)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testNonZeroLoss(self):
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._targets)
    with self.test_session():
      self.assertAlmostEqual(49.5, loss.eval(), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(49.5 * weight, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._targets, tf.constant(weight))
    with self.test_session():
      self.assertAlmostEqual(49.5 * weight, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weight = tf.constant([1.2, 3.4], shape=[2,])
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(767.8 / 6.0, loss.eval(), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeights(self):
    weight = tf.constant([1.2, 3.4], shape=[2, 1])
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(767.8 / 6.0, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeights(self):
    weight = tf.constant([3, 6, 5, 0, 4, 2], shape=[2, 3])
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(587 / 5.0, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weight = tf.constant([0, 0, 0, 0, 0, 2], shape=[2, 3])
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(18.0, loss.eval(), 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    weight = tf.zeros((2, 3))
    loss = tf.contrib.losses.sum_of_squares(
        self._predictions, self._targets, weight)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class SumOfPairwiseSquaresLossTest(tf.test.TestCase):

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

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.losses.sum_of_pairwise_squares(
            predictions=tf.constant(self._targets),
            targets=tf.constant(self._targets),
            weight=None)

  def testAllCorrectNoLossWeight(self):
    loss = tf.contrib.losses.sum_of_pairwise_squares(
        predictions=tf.constant(self._targets),
        targets=tf.constant(self._targets))
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testNonZeroLoss(self):
    loss = tf.contrib.losses.sum_of_pairwise_squares(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets))
    with self.test_session():
      self.assertAlmostEqual(np.sum(self._expected_losses), loss.eval(), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.sum_of_pairwise_squares(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        weight=weight)
    with self.test_session():
      self.assertAlmostEqual(weight * np.sum(self._expected_losses),
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weight = 2.3
    loss = tf.contrib.losses.sum_of_pairwise_squares(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        weight=tf.constant(weight))
    with self.test_session():
      self.assertAlmostEqual(weight * np.sum(self._expected_losses),
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeightWithPlaceholder(self):
    weight = 2.3
    tf_predictions = tf.placeholder(tf.float32, shape=self._predictions.shape)
    tf_targets = tf.placeholder(tf.float32, shape=self._targets.shape)
    loss = tf.contrib.losses.sum_of_pairwise_squares(
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
    expectedes = np.multiply(weight, self._expected_losses)

    loss = tf.contrib.losses.sum_of_pairwise_squares(
        predictions=tf.constant(self._predictions),
        targets=tf.constant(self._targets),
        weight=tf.constant(weight, shape=[2]))
    with self.test_session():
      self.assertAlmostEqual(np.sum(expectedes), loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeightsAndPlaceholders(self):
    weight = np.asarray([1.2, 3.4]).reshape((2, 1))
    expectedes = np.multiply(weight, self._expected_losses)

    tf_predictions = tf.placeholder(tf.float32, shape=self._predictions.shape)
    tf_targets = tf.placeholder(tf.int32, shape=self._targets.shape)
    loss = tf.contrib.losses.sum_of_pairwise_squares(
        predictions=tf_predictions,
        targets=tf_targets,
        weight=tf.constant(weight, shape=[2]))

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={
          tf_predictions: self._predictions,
          tf_targets: self._targets,
      })
      self.assertAlmostEqual(np.sum(expectedes), loss, 3)

  def testLossWithAllZeroBatchSpecificWeights(self):
    weight = np.zeros((2, 1))
    loss = tf.contrib.losses.sum_of_pairwise_squares(
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


if __name__ == '__main__':
  tf.test.main()
