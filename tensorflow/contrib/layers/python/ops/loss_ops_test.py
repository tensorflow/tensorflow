# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for contrib.layers.python.ops.loss_ops."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ReduceBatchSumTest(tf.test.TestCase):

  def testDimensionNone(self):
    with self.test_session():
      input_array = np.array([
          [1.0, 2.0],
          [-1.0, -2.0]
      ], dtype=np.float32)
      placeholder_vec = tf.placeholder(tf.float32, name="placeholder_vec")
      expected_result = np.array([3.0, -3.0])
      actual_result = tf.contrib.layers.reduce_batch_sum(placeholder_vec)
      self.assertEqual(actual_result.get_shape().as_list(), [None])
      self.assertAllClose(expected_result, actual_result.eval(feed_dict={
          placeholder_vec: input_array
      }))

  def testDimension0(self):
    with self.test_session():
      input_vec = tf.constant(2.0)
      with self.assertRaises(ValueError):
        tf.contrib.layers.reduce_batch_sum(input_vec)

  def testDimension1(self):
    with self.test_session():
      input_vec = tf.constant([1.0, 2.0])
      expected_result = np.array([1.0, 2.0])
      actual_result = tf.contrib.layers.reduce_batch_sum(input_vec)
      self.assertAllClose(expected_result, actual_result.eval())

  def testDimension2(self):
    with self.test_session():
      input_vec = tf.constant([
          [1.0, 2.0],
          [-1.0, -2.0]
      ])
      expected_result = np.array([3.0, -3.0])
      actual_result = tf.contrib.layers.reduce_batch_sum(input_vec)
      self.assertAllClose(expected_result, actual_result.eval())

  def testReturnShape(self):
    with self.test_session():
      input_vec = tf.constant([
          [1.0, 2.0],
          [-1.0, -2.0]
      ])
      expected_result = np.array([3.0, -3.0])
      actual_result = tf.contrib.layers.reduce_batch_sum(input_vec)
      self.assertShapeEqual(expected_result, actual_result)

  def testDimensionN(self):
    with self.test_session():
      input_vec = tf.constant([
          [
              [1.0, 2.0],
              [3.0, 4.0]
          ],
          [
              [5.0, 6.0],
              [7.0, 8.0]
          ]
      ])
      expected_result = np.array([10.0, 26.0])
      actual_result = tf.contrib.layers.reduce_batch_sum(input_vec)
      self.assertAllClose(expected_result, actual_result.eval())


class ReduceBatchMeanTest(tf.test.TestCase):

  def testDimensionNone(self):
    with self.test_session():
      input_array = np.array([
          [1.0, 2.0],
          [-1.0, -2.0]
      ], dtype=np.float32)
      placeholder_vec = tf.placeholder(tf.float32, name="placeholder_vec")
      expected_result = np.array([1.5, -1.5])
      actual_result = tf.contrib.layers.reduce_batch_mean(placeholder_vec)
      self.assertEqual(actual_result.get_shape().as_list(), [None])
      self.assertAllClose(expected_result, actual_result.eval(feed_dict={
          placeholder_vec: input_array
      }))

  def testDimension0(self):
    with self.test_session():
      input_vec = tf.constant(2.0)
      with self.assertRaises(ValueError):
        tf.contrib.layers.reduce_batch_mean(input_vec)

  def testDimension1(self):
    with self.test_session():
      input_vec = tf.constant([1.0, 2.0])
      expected_result = np.array([1.0, 2.0])
      actual_result = tf.contrib.layers.reduce_batch_mean(input_vec)
      self.assertAllClose(expected_result, actual_result.eval())

  def testDimension2(self):
    with self.test_session():
      input_vec = tf.constant([
          [1.0, 2.0],
          [-1.0, -2.0]
      ])
      expected_result = np.array([1.5, -1.5])
      actual_result = tf.contrib.layers.reduce_batch_mean(input_vec)
      self.assertAllClose(expected_result, actual_result.eval())

  def testReturnShape(self):
    with self.test_session():
      input_vec = tf.constant([
          [1.0, 2.0],
          [-1.0, -2.0]
      ])
      expected_result = np.array([3.0, -3.0])
      actual_result = tf.contrib.layers.reduce_batch_mean(input_vec)
      self.assertShapeEqual(expected_result, actual_result)

  def testDimensionN(self):
    with self.test_session():
      input_vec = tf.constant([
          [
              [1.0, 2.0],
              [3.0, 4.0]
          ],
          [
              [5.0, 6.0],
              [7.0, 8.0]
          ]
      ])
      expected_result = np.array([2.5, 6.5])
      actual_result = tf.contrib.layers.reduce_batch_mean(input_vec)
      self.assertAllClose(expected_result, actual_result.eval())


class AbsoluteLossTest(tf.test.TestCase):

  def _getTestVectors(self):
    target = tf.constant([1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="target")
    predicted = tf.constant([1.1, -0.2, 3.3, 1.6], shape=[2, 2],
                            name="predicted")
    expected_loss = np.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2)
    return target, predicted, expected_loss

  def testAbsoluteLoss(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.absolute_loss(predicted, target)
      self.assertAllClose(expected_loss, result.eval())

  def testAbsoluteLossReturnShape(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.absolute_loss(predicted, target)
      self.assertShapeEqual(expected_loss, result)

  def testInvalidShapesValueError(self):
    with self.test_session():
      target = tf.constant([1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="target")
      incompatible_shape = tf.constant([0.0, 1.1], shape=[2],
                                       name="incompatible_shape")
      with self.assertRaises(ValueError):
        tf.contrib.layers.absolute_loss(incompatible_shape, target)


class SquaredLossTest(tf.test.TestCase):

  def _getTestVectors(self):
    target = tf.constant([1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="target")
    predicted = tf.constant([1.1, -0.2, 3.3, 1.6], shape=[2, 2],
                            name="predicted")
    expected_loss = np.array([0.01, 0.04, 0.09, 0.16]).reshape(2, 2)
    return target, predicted, expected_loss

  def testSquaredLoss(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.squared_loss(predicted, target)
      self.assertAllClose(expected_loss, result.eval())

  def testSquaredLossReturnShape(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.squared_loss(predicted, target)
      self.assertShapeEqual(expected_loss, result)

  def testInvalidShapesValueError(self):
    with self.test_session():
      target = tf.constant([1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="target")
      incompatible_shape = tf.constant([0.0, 1.1], shape=[2],
                                       name="incompatible_shape")
      with self.assertRaises(ValueError):
        tf.contrib.layers.squared_loss(incompatible_shape, target)


class SumSquaredLossTest(tf.test.TestCase):

  def _getTestVectors(self):
    target = tf.constant([[0.0, 1.0],
                          [3.0, 2.0]],
                         shape=[2, 2],
                         name="target")
    predicted = tf.constant([[3.0, -2.0],
                             [1.0, 2.0]],
                            shape=[2, 2],
                            name="predicted")
    expected_loss = np.array([9.0, 2.0])
    return target, predicted, expected_loss

  def testSumSquaredLoss(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.sum_squared_loss(predicted, target)
      self.assertAllClose(expected_loss, result.eval())

  def testSumSquaredLossReturnShape(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.sum_squared_loss(predicted, target)
      self.assertShapeEqual(expected_loss, result)

  def testInvalidShapesValueError(self):
    with self.test_session():
      target = tf.constant([1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="target")
      incompatible_shape = tf.constant([0.0, 1.1], shape=[2],
                                       name="incompatible_shape")
      with self.assertRaises(ValueError):
        tf.contrib.layers.sum_squared_loss(incompatible_shape, target)


class MeanAbsoluteLossTest(tf.test.TestCase):

  def _getTestVectors(self):
    target = tf.constant([[0.0, 1.0, 2.0],
                          [3.0, 2.0, 4.0]],
                         shape=[2, 3],
                         name="target")
    predicted = tf.constant([[3.0, -3.0, 0.0],
                             [1.0, 2.0, 0.0]],
                            shape=[2, 3],
                            name="predicted")
    expected_loss = np.array([3.0, 2.0])
    return target, predicted, expected_loss

  def testMeanAbsoluteLoss(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.mean_absolute_loss(predicted, target)
      self.assertAllClose(expected_loss, result.eval())

  def testMeanAbsoluteLossReturnShape(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.mean_absolute_loss(predicted, target)
      self.assertShapeEqual(expected_loss, result)

  def testInvalidShapesValueError(self):
    with self.test_session():
      target = tf.constant([1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="target")
      incompatible_shape = tf.constant([0.0, 1.1], shape=[2],
                                       name="incompatible_shape")
      with self.assertRaises(ValueError):
        tf.contrib.layers.mean_absolute_loss(incompatible_shape, target)


class MeanSquaredLossTest(tf.test.TestCase):

  def _getTestVectors(self):
    target = tf.constant([[0.0, 1.0, 2.0],
                          [3.0, 2.0, 4.0]],
                         shape=[2, 3],
                         name="target")
    predicted = tf.constant([[3.0, -3.0, 0.0],
                             [1.0, 2.0, 0.0]],
                            shape=[2, 3],
                            name="predicted")
    expected_loss = np.array([9.666667, 6.666667])
    return target, predicted, expected_loss

  def testMeanSquaredLoss(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.mean_squared_loss(predicted, target)
      self.assertAllClose(expected_loss, result.eval())

  def testMeanSquaredLossReturnShape(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.mean_squared_loss(predicted, target)
      self.assertShapeEqual(expected_loss, result)

  def testInvalidShapesValueError(self):
    with self.test_session():
      target = tf.constant([1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="target")
      incompatible_shape = tf.constant([0.0, 1.1], shape=[2],
                                       name="incompatible_shape")
      with self.assertRaises(ValueError):
        tf.contrib.layers.mean_squared_loss(incompatible_shape, target)


class RootMeanSquaredLossTest(tf.test.TestCase):

  def _getTestVectors(self):
    target = tf.constant([[0.0, 1.0, 2.0],
                          [3.0, 2.0, 4.0]],
                         shape=[2, 3],
                         name="target")
    predicted = tf.constant([[3.0, -3.0, 0.0],
                             [1.0, 2.0, 0.0]],
                            shape=[2, 3],
                            name="predicted")
    expected_loss = np.array([3.109126, 2.5819889])
    return target, predicted, expected_loss

  def testRootMeanSquaredLoss(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.root_mean_squared_loss(predicted, target)
      self.assertAllClose(expected_loss, result.eval())

  def testRootMeanSquaredLossReturnShape(self):
    with self.test_session():
      target, predicted, expected_loss = self._getTestVectors()
      result = tf.contrib.layers.root_mean_squared_loss(predicted, target)
      self.assertShapeEqual(expected_loss, result)

  def testInvalidShapesValueError(self):
    with self.test_session():
      target = tf.constant([1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="target")
      incompatible_shape = tf.constant([0.0, 1.1], shape=[2],
                                       name="incompatible_shape")
      with self.assertRaises(ValueError):
        tf.contrib.layers.root_mean_squared_loss(incompatible_shape, target)


if __name__ == "__main__":
  tf.test.main()
