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
"""Tests for losses."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.platform import test
from tensorflow.python.training import momentum as momentum_lib


@test_util.run_deprecated_v1
class AbsoluteDifferenceLossTest(test.TestCase):

  def setUp(self):
    super(AbsoluteDifferenceLossTest, self).setUp()
    self._predictions = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    self._labels = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.absolute_difference(
            self._predictions, self._predictions, weights=None)

  def testAllCorrectNoLossWeight(self):
    loss = losses.absolute_difference(self._predictions, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testNonZeroLoss(self):
    loss = losses.absolute_difference(self._labels, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(5.5, self.evaluate(loss), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weights = 2.3
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(5.5 * weights, self.evaluate(loss), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weights = 2.3
    loss = losses.absolute_difference(self._labels, self._predictions,
                                      constant_op.constant(weights))
    with self.cached_session():
      self.assertAlmostEqual(5.5 * weights, self.evaluate(loss), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weights = constant_op.constant((1.2, 0.0), shape=(2, 1))
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(5.6, self.evaluate(loss), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeights(self):
    weights = constant_op.constant([1.2, 0.0], shape=[2, 1])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(5.6, self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeights(self):
    weights = constant_op.constant([3, 6, 5, 0, 4, 2], shape=[2, 3])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(16.6, self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weights = constant_op.constant([0, 0, 0, 0, 0, 2], shape=[2, 3])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(6.0, self.evaluate(loss), 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    weights = array_ops.zeros((2, 3))
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testEagerNoMemoryLeaked(self):
    # This is a somewhat convoluted way of testing that nothing gets added to
    # a global collection.
    predictions = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    labels = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    losses.absolute_difference(labels, predictions)


class SoftmaxCrossEntropyLossTest(test.TestCase):

  def testNoneWeightRaisesValueError(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.softmax_cross_entropy(labels, logits, weights=None)

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    with self.cached_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      loss = losses.softmax_cross_entropy(labels, logits)
      self.assertEqual('softmax_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrong(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 10.0, 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPythonScalarWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = 2.3
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithScalarTensorWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = 2.3
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits,
                                          constant_op.constant(weights))
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant((1.2, 3.4, 5.6))
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0,
                             self.evaluate(loss), 3)

  def testAllWrongAllWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant([0, 0, 0], shape=[3])
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testSomeWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant([1.2, 0, 0], shape=[3])
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(12.0, self.evaluate(loss), 3)

  def testSoftmaxWithMeasurementSpecificWeightsRaisesException(self):
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])

      with self.assertRaises(ValueError):
        losses.softmax_cross_entropy(labels, logits, weights=weights).eval()

  @test_util.run_deprecated_v1
  def testSoftmaxLabelSmoothing(self):
    with self.cached_session():
      # Softmax Cross Entropy Loss is:
      #   -\sum_i p_i \log q_i
      # where for a softmax activation
      # \log q_i = x_i - \log \sum_j \exp x_j
      #          = x_i - x_max - \log \sum_j \exp (x_j - x_max)
      # For our activations, [100, -100, -100] the log partition function
      # becomes \log ( exp(0) + exp(-200) + exp(-200) ) = 0
      # so our log softmaxes become: [0, -200, -200]
      # so our cross entropy loss is:
      # -(1 - L + L/n) * 0 + 400 * L/n = 400 L/n
      logits = constant_op.constant([[100.0, -100.0, -100.0]])
      labels = constant_op.constant([[1, 0, 0]])
      label_smoothing = 0.1
      loss = losses.softmax_cross_entropy(
          labels, logits, label_smoothing=label_smoothing)
      self.assertEqual(loss.op.name, 'softmax_cross_entropy_loss/value')
      expected_value = 400.0 * label_smoothing / 3.0
      self.assertAlmostEqual(self.evaluate(loss), expected_value, 3)


class SparseSoftmaxCrossEntropyLossTest(test.TestCase):

  def testNoneWeightRaisesValueError(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0], [1], [2]])
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(labels, logits, weights=None)

  @test_util.run_deprecated_v1
  def testAllCorrectInt32Labels(self):
    with self.cached_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[0], [1], [2]], dtype=dtypes.int32)
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testEagerNoMemoryLeaked(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0], [1], [2]], dtype=dtypes.int32)
    losses.sparse_softmax_cross_entropy(labels, logits)

  @test_util.run_deprecated_v1
  def testAllCorrectInt64Labels(self):
    with self.cached_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[0], [1], [2]], dtype=dtypes.int64)
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  @test_util.run_deprecated_v1
  def testAllCorrectNonColumnLabels(self):
    with self.cached_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([0, 1, 2])
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrongInt32Labels(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]], dtype=dtypes.int32)

    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 10.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrongInt64Labels(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]], dtype=dtypes.int64)

    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 10.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrongNonColumnLabels(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([2, 0, 1])

    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 10.0, 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPythonScalarWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = 2.3
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithScalarTensorWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = 2.3
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits,
                                                 constant_op.constant(weights))
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  def testNonZeroLossWith1DTensorWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = 2.3
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(
          labels, logits, constant_op.constant((weights,)))
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPlaceholderForWeights(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0],
                                   [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = array_ops.placeholder(dtypes.float32)
    with self.cached_session() as sess:
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      loss_val = sess.run(loss,
                          feed_dict={weights: ((1.2,), (3.4,), (5.6,))})
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0, loss_val, 3)

  @test_util.run_deprecated_v1
  def testUnknownShapePlaceholderForLogitsLabelsButScalarWeights(self):
    logits = array_ops.placeholder(dtypes.float32)
    labels = array_ops.placeholder(dtypes.int32)
    weights = 1.0
    with self.cached_session() as sess:
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      loss_val = sess.run(loss,
                          feed_dict={
                              logits: [[10.0, 0.0, 0.0],
                                       [0.0, 10.0, 0.0],
                                       [0.0, 0.0, 10.0]],
                              labels: [[2], [0], [1]],
                          })
      self.assertAlmostEqual((1.0 + 1.0 + 1.0) * 10.0 / 3.0, loss_val, 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPlaceholderForLogitsLabelsAndWeights(self):
    logits = array_ops.placeholder(dtypes.float32, shape=(None, 3))
    labels = array_ops.placeholder(dtypes.int32, shape=(None, 1))
    weights = array_ops.placeholder(dtypes.float32)
    with self.cached_session() as sess:
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      loss_val = sess.run(loss,
                          feed_dict={
                              logits: [[10.0, 0.0, 0.0],
                                       [0.0, 10.0, 0.0],
                                       [0.0, 0.0, 10.0]],
                              labels: [[2], [0], [1]],
                              weights: ((1.2,), (3.4,), (5.6,)),
                          })
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0, loss_val, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([1.2, 3.4, 5.6], shape=(3, 1))
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithColumnWeights(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([[1.2], [3.4], [5.6]])
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0,
                             self.evaluate(loss), 3)

  def testAllWrongAllWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([0, 0, 0], shape=(3, 1))
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testSomeWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([1.2, 0, 0], shape=(3, 1))
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(12.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testMeasurementSpecificWeightsRaisesException(self):
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  def testInconsistentWeightSizeRaisesException(self):
    """The weight tensor has incorrect number of elements."""
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2]])
      weights = constant_op.constant([1.2, 3.4, 5.6, 7.8])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  def testInconsistentLabelSizeRaisesException(self):
    """The label tensor has incorrect number of elements."""
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2], [3]])
      weights = constant_op.constant([1.2, 3.4, 5.6])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  @test_util.run_deprecated_v1
  def testInconsistentWeightShapeRaisesException(self):
    """The weight tensor has incorrect shape."""
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0, -100.0],
                                     [-100.0, -100.0, 100.0, -100.0],
                                     [-100.0, -100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2], [3]])
      weights = constant_op.constant([[1.2, 3.4], [5.6, 7.8]])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  @test_util.run_deprecated_v1
  def testInconsistentLabelShapeRaisesException(self):
    """The label tensor has incorrect shape."""
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0, -100.0],
                                     [-100.0, -100.0, 100.0, -100.0],
                                     [-100.0, -100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0, 1], [2, 3]])
      weights = constant_op.constant(1.2)

      with self.assertRaisesRegex(
          ValueError,
          '`labels.shape.rank` must equal `logits.shape.rank - 1`'):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()


class SigmoidCrossEntropyLossTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testAllCorrectSigmoid(self):
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      loss = losses.sigmoid_cross_entropy(labels, logits)
      self.assertEqual(logits.dtype, loss.dtype)
      self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testLossWithSingleDimPlaceholderForLogitsAndWeights1(self):
    logits = array_ops.placeholder(dtypes.float32, shape=(None, 1))
    labels = array_ops.placeholder(dtypes.float32, shape=(None, 1))
    weights = array_ops.ones_like(logits, dtype=dtypes.float32)

    loss = losses.sigmoid_cross_entropy(labels, logits, weights)
    self.assertEqual(logits.dtype, loss.dtype)

    with self.cached_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          logits: np.ones((32, 1)),
                          labels: np.ones((32, 1)),
                      })
      self.assertAlmostEqual(0.313, loss, 3)

  @test_util.run_deprecated_v1
  def testLossWithSingleDimPlaceholderForLogitsAndWeights2(self):
    logits = array_ops.placeholder(dtypes.float32, shape=(None, 2))
    labels = array_ops.placeholder(dtypes.float32, shape=(None, 2))
    weights = array_ops.ones_like(logits, dtype=dtypes.float32)

    loss = losses.sigmoid_cross_entropy(labels, logits, weights)
    self.assertEqual(logits.dtype, loss.dtype)

    with self.cached_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          logits: np.ones((32, 2)),
                          labels: np.ones((32, 2)),
                      })
      self.assertAlmostEqual(0.313, loss, 3)

  @test_util.run_deprecated_v1
  def testAllWrongSigmoid(self):
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
      loss = losses.sigmoid_cross_entropy(labels, logits)
      self.assertEqual(logits.dtype, loss.dtype)
      self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(self.evaluate(loss), 600.0 / 9.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrongSigmoidWithMeasurementSpecificWeights(self):
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])
      loss = losses.sigmoid_cross_entropy(labels, logits, weights)
      self.assertEqual(logits.dtype, loss.dtype)
      self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(1700.0 / 7.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testMultiCorrectSigmoid(self):
    logits = constant_op.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0],
                                   [-100.0, 100.0, 100.0]])
    labels = constant_op.constant([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    loss = losses.sigmoid_cross_entropy(labels, logits)
    self.assertEqual(logits.dtype, loss.dtype)
    self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)

    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testSigmoidFloat64(self):
    logits = constant_op.constant((
        (100.0, -100.0, 100.0),
        (100.0, -100.0, 100.0),
        (100.0, 100.0, -100.0)
    ), dtype=dtypes.float64)
    labels = constant_op.constant((
        (1, 0, 1), (1, 1, 0), (0, 1, 1)
    ), dtype=dtypes.int64)
    loss = losses.sigmoid_cross_entropy(labels, logits)
    self.assertEqual(logits.dtype, loss.dtype)

    with self.cached_session():
      self.assertAlmostEqual(44.444, self.evaluate(loss), 3)

  def testSigmoidNoReduction(self):
    logits = constant_op.constant((
        (100.0, -100.0, 100.0),
        (100.0, -100.0, 100.0),
        (100.0, 100.0, -100.0)))
    labels = constant_op.constant(((1, 0, 1), (1, 1, 0), (0, 1, 1)))
    loss = losses.sigmoid_cross_entropy(
        labels, logits, reduction=losses.Reduction.NONE)
    self.assertEqual(logits.dtype, loss.dtype)

    with self.cached_session():
      self.assertAllClose(((0., 0., 0.), (0., 100., 100.), (100., 0., 100.)),
                          self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testSigmoidLabelSmoothingCorrect(self):
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0]])
      labels = constant_op.constant([[1, 0, 1]])
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
      loss = losses.sigmoid_cross_entropy(
          labels, logits, label_smoothing=label_smoothing)
      self.assertEqual(logits.dtype, loss.dtype)
      self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)
      expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
      self.assertAlmostEqual(self.evaluate(loss), expected_value, 3)

  @test_util.run_deprecated_v1
  def testSigmoidLabelSmoothingEqualsSoftmaxTwoLabel(self):
    with self.cached_session():
      label_smoothing = 0.1
      sigmoid_logits = constant_op.constant([[100.0, -100.0, -100.0]])
      sigmoid_labels = constant_op.constant([[1, 0, 1]])
      sigmoid_loss = losses.sigmoid_cross_entropy(
          sigmoid_labels, sigmoid_logits, label_smoothing=label_smoothing)
      self.assertEqual(sigmoid_logits.dtype, sigmoid_loss.dtype)

      softmax_logits = constant_op.constant(
          [[0.0, 100.0], [100.0, 0.0], [100.0, 0.0]])
      softmax_labels = constant_op.constant([[0, 1], [1, 0], [0, 1]])
      softmax_loss = losses.softmax_cross_entropy(
          softmax_labels, softmax_logits, label_smoothing=label_smoothing)
      self.assertAlmostEqual(
          self.evaluate(sigmoid_loss), self.evaluate(softmax_loss), 3)


@test_util.run_deprecated_v1
class LogLossTest(test.TestCase):

  def setUp(self):
    super(LogLossTest, self).setUp()
    predictions = np.asarray([.9, .2, .2, .8, .4, .6]).reshape((2, 3))
    labels = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]).reshape((2, 3))

    self._np_predictions = predictions
    self._np_labels = labels

    epsilon = 1e-7
    self._expected_losses = np.multiply(
        labels, np.log(predictions + epsilon)) + np.multiply(
            1 - labels, np.log(1 - predictions + epsilon))

    self._predictions = constant_op.constant(predictions)
    self._labels = constant_op.constant(labels)

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.log_loss(self._labels, self._labels, weights=None)

  def testAllCorrectNoLossWeight(self):
    loss = losses.log_loss(self._labels, self._labels)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testAllCorrectNoLossWeightWithPlaceholder(self):
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._np_labels.shape)
    loss = losses.log_loss(self._labels, tf_predictions)
    with self.cached_session():
      self.assertAlmostEqual(
          0.0, loss.eval(feed_dict={tf_predictions: self._np_labels}), 3)

  def testNonZeroLoss(self):
    loss = losses.log_loss(self._labels, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(self._expected_losses) / 6.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weights = 2.3
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weights = 2.3
    loss = losses.log_loss(self._labels, self._predictions,
                           constant_op.constant(weights))
    with self.cached_session():
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholder(self):
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._np_predictions.shape)
    weights = 2.3
    loss = losses.log_loss(self._labels, tf_predictions,
                           constant_op.constant(weights))
    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholderWithRankOnly(self):
    tf_predictions = array_ops.placeholder(dtypes.float32, shape=[None, None])
    weights = 2.3
    loss = losses.log_loss(self._labels, tf_predictions,
                           constant_op.constant(weights))
    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weights = constant_op.constant((1.2, 3.4), shape=(2, 1))
    expected_losses = np.multiply(
        self._expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 6.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeightsSomeZero(self):
    weights = constant_op.constant((1.2, 0), shape=(2, 1))
    expected_losses = np.multiply(self._expected_losses,
                                  np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape(
                                      (2, 3)))
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 3.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeightsSomeZero(self):
    weights = constant_op.constant([1.2, 0], shape=[2, 1])
    expected_losses = np.multiply(self._expected_losses,
                                  np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape(
                                      (2, 3)))
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 3.0,
                             self.evaluate(loss), 3)

  def testWeightsWithSameNumDimsButWrongShapeThrowsException(self):
    weights = constant_op.constant(np.random.normal(size=(2, 4)), shape=[2, 4])
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.log_loss(self._labels, self._predictions, weights)

  def testNonZeroLossWithMeasurementSpecificWeights(self):
    weights = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    loss = losses.log_loss(
        self._labels,
        self._predictions,
        constant_op.constant(
            weights, shape=(2, 3)))
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 5.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithMeasurementSpecificWeightsWithPlaceholder(self):
    weights = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    tf_predictions = array_ops.placeholder(dtypes.float32, shape=[2, 3])
    loss = losses.log_loss(
        self._labels,
        tf_predictions,
        constant_op.constant(
            weights, shape=(2, 3)))

    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expected_losses) / 5.0, loss, 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weights = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    loss = losses.log_loss(
        self._labels,
        self._predictions,
        constant_op.constant(
            weights, shape=(2, 3)))
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses), self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZeroWithPlaceholder(self):
    weights = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    tf_predictions = array_ops.placeholder(dtypes.float32, shape=[2, 3])
    tf_weights = constant_op.constant(weights, shape=(2, 3))
    loss = losses.log_loss(self._labels, tf_predictions, tf_weights)

    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expected_losses), loss, 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    tf_weights = array_ops.zeros(shape=(2, 3))
    loss = losses.log_loss(self._labels, self._predictions, tf_weights)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)


class HingeLossTest(test.TestCase):

  def testIncompatibleShapes(self):
    with self.cached_session():
      logits = constant_op.constant([[-1.0], [2.1]])
      labels = constant_op.constant([0.0, 1.0])
      with self.assertRaises(ValueError):
        _ = losses.hinge_loss(labels, logits).eval()

  @test_util.run_deprecated_v1
  def testAllOutsideMargin(self):
    with self.cached_session():
      logits = constant_op.constant([1.2, -1.4, -1.0, 2.1])
      labels = constant_op.constant([1.0, 0.0, 0.0, 1.0])
      loss = losses.hinge_loss(labels, logits)
      self.assertAllClose(loss, 0.0, atol=1e-3)

  @test_util.run_deprecated_v1
  def testSomeInsideMargin(self):
    with self.cached_session():
      logits = constant_op.constant([[-0.7], [-1.4], [1.4], [0.6]])
      labels = constant_op.constant([[0.0], [0.0], [1.0], [1.0]])
      loss = losses.hinge_loss(labels, logits)
      # Examples 1 and 4 are on the correct side of the hyperplane but within
      # the margin so they incur some (small) loss.
      self.assertAllClose(loss, 0.175, atol=1e-3)

  @test_util.run_deprecated_v1
  def testSomeMisclassified(self):
    with self.cached_session():
      logits = constant_op.constant([[[1.2], [0.4], [-1.0], [-1.1]]])
      labels = constant_op.constant([[[1.0], [0.0], [0.0], [1.0]]])
      loss = losses.hinge_loss(labels, logits)
      # Examples 2 and 4 are on the wrong side of the hyperplane so they incur
      # some (fairly large) loss.
      self.assertAllClose(loss, 0.875, atol=1e-3)


class HuberLossTest(test.TestCase):

  def testIncompatibleShapes(self):
    with self.cached_session():
      predictions = constant_op.constant([[-1.0], [2.1]])
      labels = constant_op.constant([0.0, 1.0])
      with self.assertRaises(ValueError):
        _ = losses.huber_loss(labels, predictions).eval()

  @test_util.run_deprecated_v1
  def testAllQuadratic(self):
    with self.cached_session():
      predictions = constant_op.constant([1.5, -1.4, -1.0, 0.0])
      labels = constant_op.constant([1.0, -1.0, 0.0, 0.5])
      loss = losses.huber_loss(labels, predictions)
      self.assertAllClose(
          loss, 0.5 * (0.25 + 0.16 + 1.0 + 0.25) / 4., atol=1e-5)

  @test_util.run_deprecated_v1
  def testAllLinear(self):
    with self.cached_session():
      predictions = constant_op.constant([1.5, -1.4, -1.0, 0.0])
      labels = constant_op.constant([0.0, 1.0, 0.0, 1.5])
      loss = losses.huber_loss(labels, predictions)
      self.assertAllClose(loss, (1.5 + 2.4 + 1.0 + 1.5) / 4. - 0.5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testMixedQuadraticLinear(self):
    with self.cached_session():
      predictions = constant_op.constant([[1.5, -1.4, -1.0, 0.0],
                                          [1.5, -1.4, -1.0, 0.0]])
      labels = constant_op.constant([[1.0, -1.0, 0.0, 0.5],
                                     [0.0, 1.0, 0.0, 1.5]])
      loss = losses.huber_loss(labels, predictions)
      quadratic = 0.5 * (0.25 + 0.16 + 1.0 + 0.25) / 4.
      linear = (1.5 + 2.4 + 1.0 + 1.5) / 4. - 0.5
      expected_loss = (quadratic + linear) / 2.
      self.assertAllClose(loss, expected_loss, atol=1e-5)

  def testAllQuadraticDelta(self):
    with self.cached_session():
      delta = 0.5
      predictions = constant_op.constant([1.5, -1.4, -0.5, 0.0])
      labels = constant_op.constant([1.0, -1.0, 0.0, 0.5])
      expected = 0.5 * np.array([0.5**2, 0.4**2, 0.5**2, 0.5**2]).mean()
      loss = losses.huber_loss(labels, predictions, delta=delta)
      self.assertAllClose(expected, self.evaluate(loss), atol=1e-5)

  def testAllLinearDelta(self):
    delta = 0.5
    predictions = constant_op.constant([1.5, -1.4, -1.0, 0.0])
    labels = constant_op.constant([0.0, 1.0, 0.0, 1.5])
    expected = delta * np.array([1.5, 2.4, 1.0, 1.5]).mean()
    expected -= 0.5 * delta**2
    loss = losses.huber_loss(labels, predictions, delta=delta)
    with self.cached_session():
      self.assertAllClose(expected, self.evaluate(loss), atol=1e-5)


@test_util.run_deprecated_v1
class MeanSquaredErrorTest(test.TestCase):

  def setUp(self):
    super(MeanSquaredErrorTest, self).setUp()
    self._predictions = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    self._labels = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.mean_squared_error(
            self._predictions, self._predictions, weights=None)

  @test_util.run_deprecated_v1
  def testScalar(self):
    with self.cached_session():
      self.assertEqual(
          0.0,
          losses.mean_squared_error(predictions=constant_op.constant(0),
                                    labels=constant_op.constant(0)).eval())

  @test_util.run_deprecated_v1
  def testAllCorrectNoLossWeight(self):
    loss = losses.mean_squared_error(self._predictions, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLoss(self):
    loss = losses.mean_squared_error(self._labels, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(49.5, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPythonScalarWeight(self):
    weights = 2.3
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(49.5 * weights, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithScalarTensorWeight(self):
    weights = 2.3
    loss = losses.mean_squared_error(self._labels, self._predictions,
                                     constant_op.constant(weights))
    with self.cached_session():
      self.assertAlmostEqual(49.5 * weights, self.evaluate(loss), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weights = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(767.8 / 6.0, self.evaluate(loss), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeights(self):
    weights = constant_op.constant([1.2, 3.4], shape=[2, 1])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(767.8 / 6.0, self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeights(self):
    weights = constant_op.constant([3, 6, 5, 0, 4, 2], shape=[2, 3])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(587 / 5.0, self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weights = constant_op.constant([0, 0, 0, 0, 0, 2], shape=[2, 3])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(18.0, self.evaluate(loss), 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    weights = array_ops.zeros((2, 3))
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)


@test_util.run_deprecated_v1
class MeanPairwiseSquaredErrorTest(test.TestCase):

  def setUp(self):
    super(MeanPairwiseSquaredErrorTest, self).setUp()
    self._predictions = np.array([[4, 8, 12], [8, 1, 3]])
    self._labels = np.array([[1, 9, 2], [-5, -5, 7]])

    batch_size, dims = self._labels.shape  # pylint: disable=unpacking-non-sequence

    # Compute the expected loss 'manually'.
    total = np.zeros((batch_size,))
    for b in range(batch_size):
      for i in range(dims - 1):
        for j in range(i + 1, dims):
          x = self._predictions[b, i].item() - self._predictions[b, j].item()
          y = self._labels[b, i].item() - self._labels[b, j].item()
          diff = (x - y)
          total[b] += (diff * diff)

    self._expected_losses = np.divide(total, 3.0)

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.mean_pairwise_squared_error(
            predictions=constant_op.constant(self._labels),
            labels=constant_op.constant(self._labels),
            weights=None)

  def _test_valid_weights(
      self, labels, predictions, expected_loss, weights=1.0):
    with self.cached_session():
      static_inputs_op = losses.mean_pairwise_squared_error(
          predictions=predictions, labels=labels, weights=weights)
      self.assertAlmostEqual(
          expected_loss, self.evaluate(static_inputs_op), places=3)

      predictions_placeholder = array_ops.placeholder(
          dtypes.float32, shape=np.asarray(predictions.shape))
      labels_placeholder = array_ops.placeholder(
          dtypes.int32, shape=np.asarray(labels.shape))
      weights_placeholder = array_ops.placeholder(
          dtypes.float32, shape=np.asarray(weights).shape)
      dynamic_inputs_op = losses.mean_pairwise_squared_error(
          predictions=predictions_placeholder,
          labels=labels_placeholder,
          weights=weights_placeholder)
      feed_dict = {
          predictions_placeholder: predictions,
          labels_placeholder: labels,
          weights_placeholder: weights,
      }
      self.assertAlmostEqual(
          expected_loss, dynamic_inputs_op.eval(feed_dict=feed_dict), places=3)

  def testAllCorrectNoLossWeight(self):
    self._test_valid_weights(
        self._labels, self._labels, expected_loss=0.0)

  def testNonZeroLoss(self):
    self._test_valid_weights(
        self._labels, self._predictions,
        expected_loss=np.sum(self._expected_losses))

  def testGradientWithZeroWeight(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)

      inputs = array_ops.ones((2, 3))
      weights = variable_scope.get_variable(
          'weights',
          shape=[3, 4],
          initializer=init_ops.truncated_normal_initializer())
      predictions = math_ops.matmul(inputs, weights)

      optimizer = momentum_lib.MomentumOptimizer(
          learning_rate=0.001, momentum=0.9)
      loss = losses.mean_pairwise_squared_error(predictions, predictions, 0)

      gradients_to_variables = optimizer.compute_gradients(loss)

      init_op = variables.global_variables_initializer()

      with self.cached_session() as sess:
        self.evaluate(init_op)
        for grad, _ in gradients_to_variables:
          np_grad = self.evaluate(grad)
          self.assertFalse(np.isnan(np_grad).any())

  def testNonZeroLossWithPythonScalarWeight(self):
    weight = 2.3
    self._test_valid_weights(
        self._labels, self._predictions,
        expected_loss=weight * np.sum(self._expected_losses),
        weights=weight)

  def testNonZeroLossWithScalarTensorWeight(self):
    weights = 2.3
    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        weights=constant_op.constant(weights))
    with self.cached_session():
      self.assertAlmostEqual(weights * np.sum(self._expected_losses),
                             self.evaluate(loss), 3)

  def testNonZeroLossWithScalarZeroWeight(self):
    self._test_valid_weights(
        self._labels, self._predictions, expected_loss=0.0, weights=0.0)

  def test3d(self):
    labels = np.array([
        [[1, 9, 2], [12, 11, 10], [9, 8, 7]],
        [[-5, -5, 7], [6, 5, 4], [3, 2, 1]],
    ])
    predictions = np.array([
        [[4, 8, 12], [1, 2, 3], [4, 5, 6]],
        [[8, 1, 3], [7, 8, 9], [10, 11, 12]],
    ])
    self._test_valid_weights(labels, predictions, expected_loss=137.5)

  def test3dWeightedScalar(self):
    labels = np.array([
        [[1, 9, 2], [12, 11, 10], [9, 8, 7]],
        [[-5, -5, 7], [6, 5, 4], [3, 2, 1]],
    ])
    predictions = np.array([
        [[4, 8, 12], [1, 2, 3], [4, 5, 6]],
        [[8, 1, 3], [7, 8, 9], [10, 11, 12]],
    ])
    weight = 3.0
    self._test_valid_weights(
        labels, predictions, expected_loss=weight * 137.5, weights=weight)

  def _test_invalid_weights(
      self, labels, predictions, weights=1.0):
    expected_error_msg = 'weights can not be broadcast to values'

    # Static check.
    with self.assertRaisesRegex(ValueError, expected_error_msg):
      losses.mean_pairwise_squared_error(
          predictions=predictions, labels=labels, weights=weights)

    # Dynamic check.
    predictions_placeholder = array_ops.placeholder(dtypes.float32)
    labels_placeholder = array_ops.placeholder(dtypes.int32)
    weights_placeholder = array_ops.placeholder(dtypes.float32)
    dynamic_inputs_op = losses.mean_pairwise_squared_error(
        predictions=predictions_placeholder,
        labels=labels_placeholder,
        weights=weights_placeholder)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.OpError, expected_error_msg):
        dynamic_inputs_op.eval(feed_dict={
            predictions_placeholder: predictions,
            labels_placeholder: labels,
            weights_placeholder: weights,
        })

  def testInvalid3dWeighted2x0(self):
    labels = np.array([
        [[1, 9, 2], [12, 11, 10], [9, 8, 7]],
        [[-5, -5, 7], [6, 5, 4], [3, 2, 1]],
    ])
    predictions = np.array([
        [[4, 8, 12], [1, 2, 3], [4, 5, 6]],
        [[8, 1, 3], [7, 8, 9], [10, 11, 12]],
    ])
    self._test_invalid_weights(
        labels, predictions, weights=np.asarray((1.2, 3.4)))

  def test3dWeighted2x3x3(self):
    labels = np.array([
        [[1, 9, 2], [12, 11, 10], [9, 8, 7]],
        [[-5, -5, 7], [6, 5, 4], [3, 2, 1]],
    ])
    predictions = np.array([
        [[4, 8, 12], [1, 2, 3], [4, 5, 6]],
        [[8, 1, 3], [7, 8, 9], [10, 11, 12]],
    ])
    self._test_valid_weights(
        # TODO(ptucker): This doesn't look right.
        labels,
        predictions,
        expected_loss=9 * 137.5,
        weights=np.ones((2, 3, 3)))

  def testLossWithAllZeroBatchSpecificWeights(self):
    self._test_valid_weights(
        self._labels, self._predictions, expected_loss=0.0,
        weights=np.zeros((2, 1)))

  def testLossIsAssociativeAcrossBatchElements(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)

      height = 3
      width = 4
      shape = (1, height, width, 1)

      labels0 = random_ops.random_uniform(
          shape, minval=0, maxval=1, dtype=dtypes.float32)
      predictions0 = random_ops.random_uniform(
          shape, minval=0, maxval=1, dtype=dtypes.float32)

      labels1 = random_ops.random_uniform(
          shape, minval=0, maxval=1, dtype=dtypes.float32)
      predictions1 = random_ops.random_uniform(
          shape, minval=0, maxval=1, dtype=dtypes.float32)

      loss0 = losses.mean_pairwise_squared_error(
          labels=labels0,
          predictions=predictions0)
      loss1 = losses.mean_pairwise_squared_error(
          labels=labels1,
          predictions=predictions1)
      loss0_1 = losses.mean_pairwise_squared_error(
          labels=array_ops.concat([labels0, labels1], 0),
          predictions=array_ops.concat([predictions0, predictions1], 0))

      with self.cached_session() as session:
        loss0, loss1, loss0_1 = session.run([loss0, loss1, loss0_1])

        self.assertTrue(loss0 > 0)
        self.assertTrue(loss1 > 0)
        self.assertAlmostEqual(loss0 + loss1, loss0_1, 5)


@test_util.run_deprecated_v1
class CosineDistanceLossTest(test.TestCase):

  def setUp(self):
    super(CosineDistanceLossTest, self).setUp()
    self._predictions = np.asarray([
        [1, 0, 0],  # Batch 1
        [0, 0, -1],
        [1, 0, 0],  # Batch 2
        [1, 0, 0],
        [0, 0, -1],  # Batch 3
        [1, 0, 0]
    ]).reshape((3, 2, 3))

    self._labels = np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                               [0, 0, 1], [0, 1, 0]]).reshape((3, 2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.cosine_distance(
            predictions=constant_op.constant(self._labels),
            labels=constant_op.constant(self._labels),
            dim=2,
            weights=None)

  def testAllCorrectNoWeights(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._labels),
        labels=constant_op.constant(self._labels),
        dim=2)
    with self.cached_session():
      self.assertAlmostEqual(0, self.evaluate(loss), 5)

  def testPartiallyCorrectWithIntegerValues(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2)
    with self.cached_session():
      self.assertAlmostEqual(1, self.evaluate(loss), 5)

  def testPartiallyCorrectFloatingPointValues(self):
    predictions = np.matrix(
        ('0.819031913261206 0.567041924552012 0.087465312324590;'
         '-0.665139432070255 -0.739487441769973 -0.103671883216994;'
         '0.707106781186548 -0.707106781186548 0'))
    labels = np.matrix(('0.819031913261206 0.567041924552012 0.087465312324590;'
                        '0.665139432070255 0.739487441769973 0.103671883216994;'
                        '0.707106781186548 0.707106781186548 0'))

    tf_preds = constant_op.constant(
        predictions, shape=(3, 1, 3), dtype=dtypes.float32)
    tf_labels = constant_op.constant(
        labels, shape=(3, 1, 3), dtype=dtypes.float32)
    loss = losses.cosine_distance(tf_labels, tf_preds, dim=2)

    with self.cached_session():
      self.assertAlmostEqual(1.0, self.evaluate(loss), 5)

  def testSampleSpecificWeights(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=np.asarray((1, 0, 0)).reshape((3, 1, 1)))
    with self.cached_session():
      self.assertEqual(1.0, self.evaluate(loss))

  def testMeasurementSpecificWeights(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=constant_op.constant(
            [1, 0, 0, 1, 1, 1], shape=(3, 2, 1)))
    with self.cached_session():
      self.assertEqual(3.0 / 4.0, self.evaluate(loss))

  def testMeasurementSpecificWeightsWithPlaceholderWithShape(self):
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._labels.shape)
    loss = losses.cosine_distance(
        predictions=tf_predictions,
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=constant_op.constant(
            [1, 0, 0, 1, 1, 1], shape=(3, 2, 1)))
    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._predictions})
      self.assertEqual(3.0 / 4.0, loss)

  def testZeroLossWhenAllSampleSpecificWeightsAreZero(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=array_ops.zeros((3, 1, 1)))
    with self.cached_session():
      self.assertEqual(0, self.evaluate(loss))

  def testZeroLossWhenAllMeasurementSpecificWeightsAreZero(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=array_ops.zeros((3, 2, 1)))
    with self.cached_session():
      self.assertEqual(0, self.evaluate(loss))


class AddLossTest(test.TestCase):

  def testNoCollectLossesBatch2(self):
    logits = constant_op.constant([[1.2, 0.4, -1.0, -1.1]] * 2)
    labels = constant_op.constant([[1.0, 0.0, 0.0, 1.0]] * 2)
    self.assertFalse(util.get_losses())
    losses.absolute_difference(logits, labels, loss_collection=None)
    losses.log_loss(logits, labels, loss_collection=None)
    losses.mean_squared_error(logits, labels, loss_collection=None)
    losses.sigmoid_cross_entropy(logits, labels, loss_collection=None)
    losses.softmax_cross_entropy(logits, labels, loss_collection=None)
    self.assertFalse(util.get_losses())


class ComputeWeightedLossTest(test.TestCase):

  def setUp(self):
    super(ComputeWeightedLossTest, self).setUp()
    self._shape = (3, 2, 4)
    raw_losses = np.zeros(self._shape)
    next_loss = 0.0
    for i in range(self._shape[0]):
      for j in range(self._shape[1]):
        for k in range(self._shape[2]):
          raw_losses[i][j][k] = next_loss
          next_loss += 1.0
    raw_losses.setflags(write=False)
    self._raw_losses = raw_losses

  def testUnweighted(self):
    for reduction in losses.Reduction.all():
      with ops.Graph().as_default() as g:
        self.assertEqual(0, len(util.get_losses()))
        raw_losses = self._raw_losses
        unweighted_losses = (
            losses.compute_weighted_loss(raw_losses, reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 1, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 1, 4)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 2, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 2, 4)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((3, 1, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((3, 1, 4)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((3, 2, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones(self._shape), reduction=reduction)
        )
        self.assertEqual(9, len(util.get_losses()))
        with self.session(g):
          for unweighted_loss in unweighted_losses:
            if reduction == losses.Reduction.NONE:
              self.assertAllClose(self._raw_losses,
                                  self.evaluate(unweighted_loss))
            elif reduction == losses.Reduction.SUM:
              self.assertAllClose(
                  np.sum(self._raw_losses), self.evaluate(unweighted_loss))
            else:
              # reduction one of MEAN, SUM_OVER_NONZERO_WEIGHTS,
              # SUM_BY_NONZERO_WEIGHTS or SUM_OVER_BATCH_SIZE.
              self.assertAllClose(
                  np.mean(self._raw_losses), self.evaluate(unweighted_loss))

  def testUnweightedFromPlaceholder(self):
    for reduction in losses.Reduction.all():
      with ops.Graph().as_default() as g:
        self.assertEqual(0, len(util.get_losses()))
        raw_losses = array_ops.placeholder(dtype=dtypes.float32)
        feed_dict = {raw_losses: self._raw_losses}
        unweighted_losses = (
            losses.compute_weighted_loss(raw_losses, reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 1, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 1, 4)), reduction=reduction),
        )
        self.assertEqual(3, len(util.get_losses()))
        with self.session(g):
          for unweighted_loss in unweighted_losses:
            if reduction == losses.Reduction.NONE:
              self.assertAllClose(
                  self._raw_losses, unweighted_loss.eval(feed_dict))
            elif reduction == losses.Reduction.SUM:
              self.assertAllClose(
                  np.sum(self._raw_losses), unweighted_loss.eval(feed_dict))
            else:
              # reduction one of MEAN, SUM_OVER_NONZERO_WEIGHTS,
              # SUM_BY_NONZERO_WEIGHTS or SUM_OVER_BATCH_SIZE.
              self.assertAllClose(
                  np.mean(self._raw_losses), unweighted_loss.eval(feed_dict))

  def testScalarWeight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(util.get_losses()))
      weight = 17.0
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weight)
      self.assertEqual(1, len(util.get_losses()))
      with self.cached_session():
        self.assertAllClose(
            np.mean(weight * self._raw_losses), self.evaluate(weighted_loss))

  def _test_invalid_weights(self, weights):
    with ops.Graph().as_default():
      self.assertEqual(0, len(util.get_losses()))
      expected_error_msg = 'weights can not be broadcast to values'

      # Static check.
      with self.assertRaisesRegex(ValueError, expected_error_msg):
        losses.compute_weighted_loss(self._raw_losses, weights=weights)

      # Dynamic check.
      weights_placeholder = array_ops.placeholder(dtypes.float32)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights_placeholder)
      self.assertEqual(1, len(util.get_losses()))
      with self.cached_session():
        with self.assertRaisesRegex(errors_impl.OpError, expected_error_msg):
          weighted_loss.eval(feed_dict={weights_placeholder: weights})

  def testInvalidWeightTooManyDims(self):
    self._test_invalid_weights(np.zeros(shape=(2, 2, 2, 2)))

  def testInvalidWeightMismatchedDim(self):
    with ops.Graph().as_default():
      raw_losses = array_ops.reshape(self._raw_losses, shape=(3, 2, 4, 1))
      weights = np.ones(shape=(3, 2, 4, 2))
      expected_error_msg = 'weights can not be broadcast to values'
      self.assertEqual(0, len(util.get_losses()))

      # Static check.
      with self.assertRaisesRegex(ValueError, expected_error_msg):
        losses.compute_weighted_loss(raw_losses, weights=weights)

      # Dynamic check.
      weights_placeholder = array_ops.placeholder(dtypes.float32)
      weighted_loss = losses.compute_weighted_loss(
          raw_losses, weights=weights_placeholder)
      self.assertEqual(1, len(util.get_losses()))
      with self.cached_session():
        with self.assertRaisesRegex(errors_impl.OpError, expected_error_msg):
          weighted_loss.eval(feed_dict={weights_placeholder: weights})

  def testInvalid3Weight(self):
    self._test_invalid_weights((17.0, 5.0, 2.0))

  def testInvalid3x1Weight(self):
    self._test_invalid_weights(((17.0,), (5.0,), (2.0,),))

  def testInvalid3x2Weight(self):
    self._test_invalid_weights((
        (17.0, 3.0),
        (5.0, 31.0),
        (2.0, 7.0),))

  def testInvalid1x2Weight(self):
    self._test_invalid_weights((17.0, 3.0,),)

  def testInvalidScalar1DWeight(self):
    self._test_invalid_weights((17.0,),)

  def _test_valid_weights(self, weights):
    for reduction in losses.Reduction.all():
      with ops.Graph().as_default() as g:
        self.assertEqual(0, len(util.get_losses()))
        weighted_loss = losses.compute_weighted_loss(
            self._raw_losses, weights=weights, reduction=reduction)
        self.assertEqual(1, len(util.get_losses()))
        with self.session(g):
          weighted_losses = weights * self._raw_losses
          weighted_sum = np.sum(weighted_losses)
          if reduction == losses.Reduction.NONE:
            self.assertAllClose(weighted_losses, self.evaluate(weighted_loss))
          elif reduction == losses.Reduction.SUM:
            self.assertAllClose(weighted_sum, self.evaluate(weighted_loss))
          else:
            broadcast_weights = weights * np.ones_like(self._raw_losses)
            if reduction == losses.Reduction.MEAN:
              self.assertAllClose(weighted_sum / np.sum(broadcast_weights),
                                  self.evaluate(weighted_loss))
            elif (reduction == losses.Reduction.SUM_OVER_NONZERO_WEIGHTS or
                  reduction == losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
              self.assertAllClose(
                  weighted_sum / np.count_nonzero(broadcast_weights),
                  self.evaluate(weighted_loss))
            elif reduction == losses.Reduction.SUM_OVER_BATCH_SIZE:
              self.assertAllClose(weighted_sum / self._raw_losses.size,
                                  self.evaluate(weighted_loss))

  def test1x1x1Weight(self):
    self._test_valid_weights((((17.0,),),))

  def test1x2x1Weight(self):
    self._test_valid_weights((((17.0,), (3.0,),),))

  def test1x1x4Weight(self):
    self._test_valid_weights((((17.0, 0.0, 2.0, 5.0),),))

  def test3x1x1Weight(self):
    self._test_valid_weights((((17.0,),), ((5.0,),), ((2.0,),),))

  def test3x2x1Weight(self):
    self._test_valid_weights((
        ((17.0,), (3.0,)),
        ((5.0,), (31.0,)),
        ((2.0,), (7.0,)),
    ))

  def test3x1x4Weight(self):
    self._test_valid_weights((
        ((17.0, 0.0, 2.0, 5.0),),
        ((5.0, 31.0, 17.0, 5.0),),
        ((7.0, 3.0, 11.0, 5.0),),
    ))

  def test1x2x4Weight(self):
    self._test_valid_weights(((
        (17.0, 0.0, 2.0, 5.0),
        (3.0, 13.0, 11.0, 2.0),
    ),))

  def test3x2x4Weight(self):
    self._test_valid_weights((
        ((17.0, 0.0, 2.0, 5.0), (3.0, 13.0, 11.0, 2.0),),
        ((5.0, 31.0, 17.0, 5.0), (13.0, 3.0, 0.0, 11.0),),
        ((0.0, 3.0, 11.0, 5.0), (13.0, 11.0, 1.0, 7.0),),
    ))


if __name__ == '__main__':
  test.main()
