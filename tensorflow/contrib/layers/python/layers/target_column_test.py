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


class RegressionTargetColumnTest(tf.test.TestCase):

  # TODO(zakaria): test multilabel regresssion.
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


class MulltiClassTargetColumnTest(tf.test.TestCase):

  def testBinaryClassification(self):
    target_column = tf.contrib.layers.multi_class_target(n_classes=2)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1.], [1.]])
      targets = tf.constant([[1.], [0.]])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(.81326163,
                             sess.run(target_column.loss(logits, targets, {})))

  def testBinaryClassificationWithWeights(self):
    target_column = tf.contrib.layers.multi_class_target(
        n_classes=2,
        weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([[1.], [0.]])}
      logits = tf.constant([[1.], [1.]])
      targets = tf.constant([[1.], [0.]])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(
          .31326166, sess.run(target_column.loss(logits, targets, features)))

  def testBinaryEvalMetrics(self):
    target_column = tf.contrib.layers.multi_class_target(n_classes=2)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1.], [1.], [-1.]])
      targets = tf.constant([[1.], [0.], [1.]])
      eval_dict = target_column.get_eval_ops({}, logits, targets)
      # TODO(zakaria): test all metrics
      accuracy_op, update_op = eval_dict["accuracy/threshold_0.500000_mean"]
      sess.run(tf.initialize_all_variables())
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      self.assertAlmostEqual(1.0/3, sess.run(accuracy_op))

  def testMultiClass(self):
    target_column = tf.contrib.layers.multi_class_target(n_classes=3)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1., 0., 0.]])
      targets = tf.constant([2])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(1.5514446,
                             sess.run(target_column.loss(logits, targets, {})))

  def testMultiClassWithWeight(self):
    target_column = tf.contrib.layers.multi_class_target(
        n_classes=3,
        weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([0.1])}
      logits = tf.constant([[1., 0., 0.]])
      targets = tf.constant([2])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(1.5514446,
                             sess.run(target_column.loss(
                                 logits, targets, features)))

  def testMultiClassWithInvalidNClass(self):
    try:
      tf.contrib.layers.multi_class_target(n_classes=1)
      self.fail("Softmax with no n_classes did not raise error.")
    except ValueError:
      # Expected
      pass

  def testMultiClassEvalMetrics(self):
    target_column = tf.contrib.layers.multi_class_target(n_classes=3)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1., 0., 0.]])
      targets = tf.constant([2])
      eval_dict = target_column.get_eval_ops({}, logits, targets)
      loss_op, update_op = eval_dict["loss"]
      sess.run(tf.initialize_all_variables())
      sess.run(tf.initialize_local_variables())
      sess.run(update_op)
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(1.5514446, sess.run(loss_op))


if __name__ == "__main__":
  tf.test.main()
