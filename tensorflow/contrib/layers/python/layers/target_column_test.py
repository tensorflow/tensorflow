# pylint: disable=g-bad-file-header
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
"""Tests for TargetColumn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class TargetColumnTest(tf.test.TestCase):

  def testRegression(self):
    target_column = tf.contrib.layers.regression_target()
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1.], [1.], [3.]])
      targets = tf.constant([[0.], [1.], [1.]])
      self.assertAlmostEqual(5. / 3,
                             sess.run(target_column.loss(logits, targets, {})))

  def testRegressionWithWeights(self):
    target_column = tf.contrib.layers.regression_target(
        weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([[1.], [0.], [0.]])}
      logits = tf.constant([[1.], [1.], [3.]])
      targets = tf.constant([[0.], [1.], [1.]])
      self.assertAlmostEqual(
          1.,
          sess.run(target_column.loss(logits, targets, features)))

  # TODO(zakaria): test multlabel regresssion.

  def testSoftmax(self):
    target_column = tf.contrib.layers.multi_class_target(
        n_classes=3)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1., 0., 0.]])
      targets = tf.constant([2])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(1.5514446,
                             sess.run(target_column.loss(logits, targets, {})))

  def testSoftmaxWithInvalidNClass(self):
    try:
      tf.contrib.layers.multi_class_target(n_classes=1)
      self.fail("Softmax with no n_classes did not raise error.")
    except ValueError:
      # Expected
      pass


if __name__ == "__main__":
  tf.test.main()
