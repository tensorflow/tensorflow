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

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import tensor_util

pi = 3.14
indiana_pi = 3.2  # https://en.wikipedia.org/wiki/Indiana_Pi_Bill


class AbsoluteLossTest(tf.test.TestCase):

  def testAbsoluteLoss(self):
    with self.test_session():
      actual = tf.constant([pi], name="pi")
      actual_placeholder = tf.placeholder(tf.float32)
      label = tf.constant([indiana_pi], name="lbl")
      label_placeholder = tf.placeholder(tf.float32, name="lbl_ph")
      expected_loss = abs(indiana_pi - pi)

      # Both shapes are set.
      both_shapes_loss = tf.contrib.losses.absolute(actual, label)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          both_shapes_loss.eval(), expected_loss, decimal=6)

      # No shape for 'actual' - check that the loss layer can be created.
      no_actual_shape_loss = tf.contrib.losses.absolute(
          actual_placeholder, label)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_actual_shape_loss.eval({actual_placeholder: [pi]}),
          expected_loss, decimal=6)

      # No shape for 'label' - check that the loss layer can be created.
      no_label_shape_loss = tf.contrib.losses.absolute(
          actual, label_placeholder)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_label_shape_loss.eval({label_placeholder: [indiana_pi]}),
          expected_loss, decimal=6)

      # No shapes.
      no_shape_loss = tf.contrib.losses.absolute(
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


class SquaredLossTest(tf.test.TestCase):

  def testSquaredLoss(self):
    with self.test_session():
      actual = tf.constant([pi], name="pi")
      actual_placeholder = tf.placeholder(tf.float32)
      label = tf.constant([indiana_pi], name="lbl")
      label_placeholder = tf.placeholder(tf.float32, name="lbl_ph")
      expected_loss = (indiana_pi - pi) * (indiana_pi - pi) / 2

      # Both shapes are set.
      both_shapes_loss = tf.contrib.losses.squared(actual, label)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          both_shapes_loss.eval(), expected_loss, decimal=6)

      # No shape for 'actual' - check that the loss layer can be created.
      no_actual_shape_loss = tf.contrib.losses.squared(
          actual_placeholder, label)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_actual_shape_loss.eval({actual_placeholder: [pi]}),
          expected_loss, decimal=6)

      # No shape for 'label' - check that the loss layer can be created.
      no_label_shape_loss = tf.contrib.losses.squared(
          actual, label_placeholder)
      tf.initialize_all_variables().run()
      np.testing.assert_almost_equal(
          no_label_shape_loss.eval({label_placeholder: [indiana_pi]}),
          expected_loss,
          decimal=6)

      # No shapes.
      no_shape_loss = tf.contrib.losses.squared(
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


class LogisticTest(tf.test.TestCase):

  def _expected_loss(self, logit, target):
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    logistic_loss = (target * -np.log(sigmoid)) - (
        (1.0 - target) * np.log(1.0 - sigmoid))
    batch_losses = np.sum(logistic_loss, 1)

    return np.sum(batch_losses) / len(batch_losses)

  def testSimple(self):
    logit = np.array([[9.45, -42], [4.2, 1], [-0.6, 20]])
    target = np.array([[0.8, 0.9], [0.45, 0.99999], [0.1, 0.0006]])
    with self.test_session():
      loss = tf.contrib.losses.logistic(tf.constant(logit), tf.constant(target))
      self.assertAllClose(self._expected_loss(logit, target), loss.eval())

  def testComplex(self):
    with self.test_session():
      # [batch] and [batch,1] work the same.
      loss3x0 = tf.contrib.losses.logistic(
          tf.constant([-1.0, 3.0, -3.0]),
          tf.constant([0.3, 0.1, 0.4]))
      tf.initialize_all_variables().run()
      self.assertAllClose(1.536812, loss3x0.eval())

      expected_loss = 1.536812
      actual3x1 = [[-1.0], [3.0], [-3.0]]
      label3x1 = [[0.3], [0.1], [0.4]]
      loss3x1 = tf.contrib.losses.logistic(
          tf.constant(actual3x1), tf.constant(label3x1))
      tf.initialize_all_variables().run()
      self.assertAllClose(expected_loss, loss3x1.eval())

      # Batch average stays the same with repeats of the same examples.
      loss9x1 = tf.contrib.losses.logistic(
          tf.constant(actual3x1 * 3), tf.constant(label3x1 * 3))
      tf.initialize_all_variables().run()
      self.assertAllClose(expected_loss, loss9x1.eval())

      # Loss stays the same when adding another class with 0 loss.
      loss3x2 = tf.contrib.losses.logistic(
          tf.constant([[-1.0, 100.0], [3.0, -100.0], [-3.0, -100.0]]),
          tf.constant([[0.3, 1.0], [0.1, 0.0], [0.4, 0.0]]))
      tf.initialize_all_variables().run()
      self.assertAllClose(expected_loss, loss3x2.eval())

      # Loss stays the same with additional x1 dimension.
      loss3x1x2 = tf.contrib.losses.logistic(
          tf.constant([[[-1.0, 100.0]], [[3.0, -100.0]], [[-3.0, -100.0]]]),
          tf.constant([[[0.3, 1.0]], [[0.1, 0.0]], [[0.4, 0.0]]]))
      tf.initialize_all_variables().run()
      self.assertAllClose(expected_loss, loss3x1x2.eval())

      # We have set one label value to be out of range (the -0.4) and
      # expect the absence of a crash since we did not set validate=True
      loss = tf.contrib.losses.logistic(
          tf.constant([[[-1.0, 100.0]], [[3.0, -100.0]], [[-3.0, -100.0]]]),
          tf.constant([[[0.3, 1.0]], [[0.1, 0.0]], [[-0.4, 0.0]]]))
      tf.initialize_all_variables().run()
      loss.eval()

  def testLogisticVsSoftmax(self):
    with self.test_session():
      # Each logit = L and target = T used for logistic_loss corresponds to
      # logits [a, b] where a - b = L and targets [T, 1 - T] for
      # softmax_loss.

      expected_loss = (0.69314718 + 1.01326168 + 2.10692811) / 3.0

      logistic_loss = tf.contrib.losses.logistic(
          tf.constant([0.0, 1.0, 2.0]),
          tf.constant([0.5, 0.3, 0.01]))
      tf.initialize_all_variables().run()
      self.assertAllClose(expected_loss, logistic_loss.eval())

      softmax_loss = tf.contrib.losses.softmax(
          tf.constant([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
          tf.constant([[0.5, 0.5], [0.3, 0.7], [0.01, 0.99]]))
      tf.initialize_all_variables().run()
      self.assertAllClose(expected_loss, softmax_loss.eval())


class SoftmaxTest(tf.test.TestCase):

  def testAllCorrect(self):
    with self.test_session():
      logits = tf.constant([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
      labels = tf.constant([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
      loss = tf.contrib.losses.softmax(logits, labels)
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllWrong(self):
    with self.test_session():
      logits = tf.constant([[10.0, 0.0, 0.0],
                            [0.0, 10.0, 0.0],
                            [0.0, 0.0, 10.0]])
      labels = tf.constant([[0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
      loss = tf.contrib.losses.softmax(logits, labels)
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testSoftmax(self):
    with self.test_session():
      # [batch] and [batch,1] fail, softmax_loss is only for multiclass.
      self.assertRaisesRegexp(
          ValueError, "must have rank 2", tf.contrib.losses.softmax,
          tf.constant([-100.0, 10.0, 0.0]),
          tf.constant([1.0, 1.0, 1.0]))

      self.assertRaisesRegexp(
          ValueError, "only 1 class", tf.contrib.losses.softmax,
          tf.constant([[-100.0], [10.0], [0.0]]),
          tf.constant([[1.0], [1.0], [1.0]]))

      expected_loss = 3.173363
      loss3x2 = tf.contrib.losses.softmax(
          tf.constant([[-1.0, 1.0], [0.0, 0.0], [10.0, -1.0]]),
          tf.constant([[0.5, 0.5], [0.3, 0.7], [0.3, 0.7]]))
      tf.initialize_all_variables().run()
      self.assertAllClose(expected_loss, loss3x2.eval())

      # Loss stays the same when adding another negative class.
      loss3x3 = tf.contrib.losses.softmax(
          tf.constant(
              [[-1.0, 1.0, -100.0], [0.0, 0.0, -100.0], [10.0, -1.0, -100.0]]),
          tf.constant([[0.5, 0.5, 0.0], [0.3, 0.7, 0.0], [0.3, 0.7, 0.0]]))
      tf.initialize_all_variables().run()
      self.assertAllClose(expected_loss, loss3x3.eval())

      # Fails for rank > 2.
      self.assertRaisesRegexp(
          ValueError, "must have rank 2", tf.contrib.losses.softmax,
          tf.constant([[[-1.0, 1.0]], [[0.0, 0.0]], [[10.0, -1.0]]]),
          tf.constant([[[0.5, 0.5]], [[0.3, 0.7]], [[0.3, 0.7]]]))


if __name__ == "__main__":
  tf.test.main()
