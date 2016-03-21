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
from tensorflow.contrib.layers.python.framework import tensor_util

pi = 3.14
indiana_pi = 3.2  # https://en.wikipedia.org/wiki/Indiana_Pi_Bill


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
    expected_loss = np.array([0.005, 0.02, 0.045, 0.08]).reshape(2, 2)
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


class ScalarAbsoluteLossTest(tf.test.TestCase):

  def testScalarAbsoluteLoss(self):
    with self.test_session():
      actual = tf.constant([pi], name="pi")
      actual_placeholder = tf.placeholder(tf.float32)
      label = tf.constant([indiana_pi], name="lbl")
      label_placeholder = tf.placeholder(tf.float32, name="lbl_ph")
      expected_loss = abs(indiana_pi - pi)

      # Both shapes are set.
      both_shapes_loss = tf.contrib.layers.scalar_absolute_loss(actual, label)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          both_shapes_loss.eval(), expected_loss, decimal=6)

      # No shape for 'actual' - check that the loss layer can be created.
      no_actual_shape_loss = tf.contrib.layers.scalar_absolute_loss(
          actual_placeholder, label)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_actual_shape_loss.eval({actual_placeholder: [pi]}),
          expected_loss, decimal=6)

      # No shape for 'label' - check that the loss layer can be created.
      no_label_shape_loss = tf.contrib.layers.scalar_absolute_loss(
          actual, label_placeholder)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_label_shape_loss.eval({label_placeholder: [indiana_pi]}),
          expected_loss, decimal=6)

      # No shapes.
      no_shape_loss = tf.contrib.layers.scalar_absolute_loss(
          actual_placeholder, label_placeholder)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_shape_loss.eval({label_placeholder: [indiana_pi],
                              actual_placeholder: [pi]}),
          expected_loss, decimal=6)

      # Evaluate the previous one again, but this time with different
      # (matching) shapes.  This should still work.
      np.testing.assert_almost_equal(
          no_shape_loss.eval({label_placeholder: [indiana_pi, indiana_pi],
                              actual_placeholder: [pi, pi]}),
          expected_loss, decimal=6)


class ScalarSquaredLossTest(tf.test.TestCase):

  def testScalarSquaredLoss(self):
    with self.test_session():
      actual = tf.constant([pi], name="pi")
      actual_placeholder = tf.placeholder(tf.float32)
      label = tf.constant([indiana_pi], name="lbl")
      label_placeholder = tf.placeholder(tf.float32, name="lbl_ph")
      expected_loss = (indiana_pi - pi) * (indiana_pi - pi) / 2

      # Both shapes are set.
      both_shapes_loss = tf.contrib.layers.scalar_squared_loss(actual, label)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          both_shapes_loss.eval(), expected_loss, decimal=6)

      # No shape for 'actual' - check that the loss layer can be created.
      no_actual_shape_loss = tf.contrib.layers.scalar_squared_loss(
          actual_placeholder, label)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_actual_shape_loss.eval({actual_placeholder: [pi]}),
          expected_loss, decimal=6)

      # No shape for 'label' - check that the loss layer can be created.
      no_label_shape_loss = tf.contrib.layers.scalar_squared_loss(
          actual, label_placeholder)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_label_shape_loss.eval({label_placeholder: [indiana_pi]}),
          expected_loss,
          decimal=6)

      # No shapes.
      no_shape_loss = tf.contrib.layers.scalar_squared_loss(
          actual_placeholder, label_placeholder)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_shape_loss.eval({label_placeholder: [indiana_pi],
                              actual_placeholder: [pi]}),
          expected_loss, decimal=6)

      # Evaluate the previous one again, but this time with different
      # (matching) shapes. This should still work.
      np.testing.assert_almost_equal(
          no_shape_loss.eval({label_placeholder: [indiana_pi, indiana_pi],
                              actual_placeholder: [pi, pi]}),
          expected_loss, decimal=6)


class ScalarLogisticLossTest(tf.test.TestCase):

  def _expected_loss(self, logit, target):
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    logistic_loss = (target * -np.log(sigmoid)) - (
        (1.0 - target) * np.log(1.0 - sigmoid))
    batch_losses = np.sum(logistic_loss, 1)

    return np.sum(batch_losses) / len(batch_losses)

  def test_scalar_logistic_loss(self):
    logit = np.array([[9.45, -42], [4.2, 1], [-0.6, 20]])
    target = np.array([[0.8, 0.9], [0.45, 0.99999], [0.1, 0.0006]])
    with self.test_session():
      result = tf.contrib.layers.scalar_logistic_loss(
          tf.constant(logit), tf.constant(target))
      self.assertAllClose(self._expected_loss(logit, target), result.eval())


if __name__ == "__main__":
  tf.test.main()
