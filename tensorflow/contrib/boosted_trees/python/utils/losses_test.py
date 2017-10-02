# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for trainer hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.contrib.boosted_trees.python.utils import losses
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class LossesTest(test_util.TensorFlowTestCase):

  def test_per_example_exp_loss(self):

    def _logit(p):
      return np.log(p) - np.log(1 - p)

    labels_positive = array_ops.ones([10, 1], dtypes.float32)
    weights = array_ops.ones([10, 1], dtypes.float32)
    labels_negative = array_ops.zeros([10, 1], dtypes.float32)
    predictions_probs = np.array(
        [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [0.99]],
        dtype=np.float32)
    prediction_logits = _logit(predictions_probs)

    eps = 0.2

    with self.test_session():
      predictions_tensor = constant_op.constant(
          prediction_logits, dtype=dtypes.float32)
      loss_for_positives, _ = losses.per_example_exp_loss(
          labels_positive, weights, predictions_tensor, eps=eps)

      loss_for_negatives, _ = losses.per_example_exp_loss(
          labels_negative, weights, predictions_tensor, eps=eps)

      pos_loss = loss_for_positives.eval()
      neg_loss = loss_for_negatives.eval()
      # For positive labels, points <= 0.3 get max loss of e.
      # For negative labels, these points have minimum loss of 1/e.
      for i in range(2):
        self.assertAlmostEqual(math.exp(1), pos_loss[i], places=4)
        self.assertAlmostEqual(math.exp(-1), neg_loss[i], places=4)

      # For positive lables, p oints with predictions 0.7 and larger get minimum
      # loss value of 1/e. For negative labels, these points are wrongly
      # classified and get loss e.
      for i in range(6, 10):
        self.assertAlmostEqual(math.exp(-1), pos_loss[i], places=4)
        self.assertAlmostEqual(math.exp(1), neg_loss[i], places=4)

      # Points in between 0.5-eps, 0..5+eps get loss exp(-label_m*y), where
      # y = 1/eps *x -1/(2eps), where x is the probability and label_m is either
      # 1 or -1 (for label of 0).
      for i in range(2, 6):
        self.assertAlmostEqual(
            math.exp(-1.0 * (predictions_probs[i] * 1.0 / eps - 0.5 / eps)),
            pos_loss[i],
            places=4)
        self.assertAlmostEqual(
            math.exp(1.0 * (predictions_probs[i] * 1.0 / eps - 0.5 / eps)),
            neg_loss[i],
            places=4)

  def test_per_example_squared_loss(self):

    def _squared_loss(p, y):
      return np.mean(1.0 * (p - y) * (p - y))

    labels = np.array([[0.123], [224.2], [-3], [2], [.3]], dtype=np.float32)
    weights = array_ops.ones([5, 1], dtypes.float32)
    predictions = np.array(
        [[0.123], [23.2], [233], [52], [3]], dtype=np.float32)

    with self.test_session():
      loss_tensor, _ = losses.per_example_squared_loss(labels, weights,
                                                       predictions)

      loss = loss_tensor.eval()
      for i in range(5):
        self.assertAlmostEqual(
            _squared_loss(labels[i], predictions[i]), loss[i], places=4)


if __name__ == "__main__":
  googletest.main()
