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
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.constant([[0.], [1.], [1.]])
      self.assertAlmostEqual(
          5. / 3, sess.run(target_column.loss(prediction, labels, {})))

  def testRegressionWithWeights(self):
    target_column = tf.contrib.layers.regression_target(
        weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([[2.], [5.], [0.]])}
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.constant([[0.], [1.], [1.]])
      self.assertAlmostEqual(
          2. / 7,
          sess.run(target_column.loss(prediction, labels, features)),
          places=3)
      self.assertAlmostEqual(
          2. / 3,
          sess.run(target_column.training_loss(prediction, labels, features)),
          places=3)


class MultiClassTargetColumnTest(tf.test.TestCase):

  def testBinaryClassification(self):
    target_column = tf.contrib.layers.multi_class_target(n_classes=2)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1.], [1.]])
      labels = tf.constant([[1.], [0.]])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(0.81326175,
                             sess.run(target_column.loss(logits, labels, {})),
                             delta=1e-6)

  def testBinaryClassificationWithWeights(self):
    target_column = tf.contrib.layers.multi_class_target(
        n_classes=2, weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([[1.], [0.]])}
      logits = tf.constant([[1.], [1.]])
      labels = tf.constant([[1.], [0.]])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(
          .31326166,
          sess.run(target_column.loss(logits, labels, features)),
          delta=1e-6)

  def testBinaryEvalMetrics(self):
    target_column = tf.contrib.layers.multi_class_target(n_classes=2)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1.], [1.], [-1.]])
      labels = tf.constant([[1.], [0.], [1.]])
      eval_dict = target_column.get_eval_ops({}, logits, labels)
      # TODO(zakaria): test all metrics
      accuracy_op, update_op = eval_dict["accuracy/threshold_0.500000_mean"]
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      self.assertAlmostEqual(1.0 / 3, sess.run(accuracy_op))

  def testMultiClass(self):
    target_column = tf.contrib.layers.multi_class_target(n_classes=3)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([2])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(1.5514446,
                             sess.run(target_column.loss(logits, labels, {})))

  def testMultiClassWithWeight(self):
    target_column = tf.contrib.layers.multi_class_target(
        n_classes=3, weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([0.1])}
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([2])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(
          1.5514446, sess.run(target_column.loss(logits, labels, features)))

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
      labels = tf.constant([2])
      eval_dict = target_column.get_eval_ops({}, logits, labels)
      loss_op, update_op = eval_dict["loss"]
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      self.assertAlmostEqual(1.5514446, sess.run(loss_op))

  def testBinarySVMDefaultWeights(self):
    target_column = tf.contrib.layers.binary_svm_target()
    predictions = tf.constant([[-0.5], [1.2]])
    labels = tf.constant([0, 1])
    loss = target_column.loss(predictions, labels, {})
    # Prediction for first example is in the right side of the hyperplane (i.e.,
    # < 0) but it is within the [-1,1] margin. There is a 0.5 loss incurred by
    # this example. The 2nd prediction is outside the margin so it incurs no
    # loss at all. The overall (normalized) loss is therefore 0.5/(1+1) = 0.25.
    with tf.Session() as sess:
      self.assertAlmostEqual(0.25, sess.run(loss))

  def testBinarySVMWithWeights(self):
    target_column = tf.contrib.layers.binary_svm_target(
        weight_column_name="weights")
    predictions = tf.constant([[-0.7], [0.2]])
    labels = tf.constant([0, 1])
    features = {"weights": tf.constant([2.0, 10.0])}
    loss = target_column.loss(predictions, labels, features)
    training_loss = target_column.training_loss(predictions, labels, features)
    # Prediction for both examples are in the right side of the hyperplane but
    # within the margin. The (weighted) loss incurred is 2*0.3=0.6 and 10*0.8=8
    # respectively. The overall (normalized) loss is therefore 8.6/12.
    with tf.Session() as sess:
      self.assertAlmostEqual(8.6 / 12, sess.run(loss), places=3)
      self.assertAlmostEqual(8.6 / 2, sess.run(training_loss), places=3)


if __name__ == "__main__":
  tf.test.main()
