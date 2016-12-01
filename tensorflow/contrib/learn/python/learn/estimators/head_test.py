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
"""Tests for head.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import head as head_lib


def _assert_variables(
    test_case, expected_global=None, expected_model=None,
    expected_trainable=None):
  test_case.assertItemsEqual(
      [] if expected_global is None else expected_global,
      [k.name for k in tf.global_variables()])
  test_case.assertItemsEqual(
      [] if expected_model is None else expected_model,
      [k.name for k in tf.model_variables()])
  test_case.assertItemsEqual(
      [] if expected_trainable is None else expected_trainable,
      [k.name for k in tf.trainable_variables()])


def _assert_no_variables(test_case):
  _assert_variables(test_case, set([]), set([]), set([]))


class RegressionModelHeadTest(tf.test.TestCase):

  def _assert_metrics(self, model_fn_ops):
    self.assertItemsEqual((
        "loss",
    ), six.iterkeys(model_fn_ops.eval_metric_ops))

  # TODO(zakaria): test multilabel regresssion.
  def testRegression(self):
    head = head_lib._regression_head()
    with tf.Graph().as_default(), tf.Session() as sess:
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.constant([[0.], [1.], [1.]])
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=prediction)
      self._assert_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(5. / 3, sess.run(model_fn_ops.loss))

      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.EVAL,
                                   _noop_train_op, logits=prediction)
      self.assertIsNone(model_fn_ops.train_op)

  def testRegressionWithWeights(self):
    head = head_lib._regression_head(
        weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([[2.], [5.], [0.]])}
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.constant([[0.], [1.], [1.]])
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=prediction)
      self._assert_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(2. / 3, sess.run(model_fn_ops.loss), places=3)

  def testRegressionWithCenteredBias(self):
    head = head_lib._regression_head(
        weight_column_name="label_weight", enable_centered_bias=True)
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([[2.], [5.], [0.]])}
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.constant([[0.], [1.], [1.]])
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=prediction)
      self._assert_metrics(model_fn_ops)
      _assert_variables(self, expected_global=(
          "centered_bias_weight:0",
          "train_op/centered_bias_step/centered_bias_weight/Adagrad:0",
      ), expected_trainable=(
          "centered_bias_weight:0",
      ))
      tf.global_variables_initializer().run()
      self.assertAlmostEqual(2. / 3, sess.run(model_fn_ops.loss), places=3)

  def testErrorInSparseTensorLabels(self):
    head = head_lib._regression_head()
    with tf.Graph().as_default():
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.SparseTensor(
          indices=tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
          values=tf.constant([0., 1., 1.]),
          shape=[3, 1])
      with self.assertRaisesRegexp(
          ValueError, "SparseTensor is not supported as labels."):
        head.head_ops({}, labels, tf.contrib.learn.ModeKeys.TRAIN,
                      _noop_train_op, logits=prediction)


class MultiLabelModelHeadTest(tf.test.TestCase):

  def _assert_metrics(self, model_fn_ops):
    self.assertItemsEqual((
        "accuracy",
        "loss",
    ), six.iterkeys(model_fn_ops.eval_metric_ops))

  def testMultiLabel(self):
    head = head_lib._multi_label_head(n_classes=3)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([[0, 0, 1]])
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      self._assert_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(0.89985204, sess.run(model_fn_ops.loss))

      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.EVAL,
                                   _noop_train_op, logits=logits)
      self.assertIsNone(model_fn_ops.train_op)

  def testMultiLabelWithWeight(self):
    head = head_lib._multi_label_head(
        n_classes=3, weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([0.1])}
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([[0, 0, 1]])
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      self._assert_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(0.089985214, sess.run(model_fn_ops.loss))

  def testMultiLabelWithCenteredBias(self):
    head = head_lib._multi_label_head(n_classes=3, enable_centered_bias=True)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([[0, 0, 1]])
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      self._assert_metrics(model_fn_ops)
      _assert_variables(self, expected_global=(
          "centered_bias_weight:0",
          "train_op/centered_bias_step/centered_bias_weight/Adagrad:0",
      ), expected_trainable=(
          "centered_bias_weight:0",
      ))
      tf.global_variables_initializer().run()
      self.assertAlmostEqual(0.89985204, sess.run(model_fn_ops.loss))


class MultiClassModelHeadTest(tf.test.TestCase):

  def _assert_binary_metrics(self, model_fn_ops):
    self.assertItemsEqual((
        "accuracy",
        "accuracy/baseline_label_mean",
        "accuracy/threshold_0.500000_mean",
        "auc",
        "labels/actual_label_mean",
        "labels/prediction_mean",
        "loss",
        "precision/positive_threshold_0.500000_mean",
        "recall/positive_threshold_0.500000_mean",
    ), six.iterkeys(model_fn_ops.eval_metric_ops))

  def testBinaryClassification(self):
    head = head_lib._multi_class_head(n_classes=2)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1.], [1.]])
      labels = tf.constant([[1.], [0.]])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      self._assert_binary_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(0.81326175, sess.run(model_fn_ops.loss),
                             delta=1e-6)
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.EVAL,
                                   _noop_train_op, logits=logits)
      self.assertIsNone(model_fn_ops.train_op)

  def testErrorInSparseTensorLabels(self):
    head = head_lib._multi_class_head(n_classes=2)
    with tf.Graph().as_default():
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.SparseTensor(
          indices=tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
          values=tf.constant([0, 1, 1]),
          shape=[3, 1])
      with self.assertRaisesRegexp(
          ValueError, "SparseTensor is not supported as labels."):
        head.head_ops({}, labels, tf.contrib.learn.ModeKeys.TRAIN,
                      _noop_train_op, logits=prediction)

  def testBinaryClassificationWithWeights(self):
    head = head_lib._multi_class_head(
        n_classes=2, weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([[1.], [0.]])}
      logits = tf.constant([[1.], [1.]])
      labels = tf.constant([[1.], [0.]])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      self._assert_binary_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(.31326166 / 2, sess.run(model_fn_ops.loss),
                             delta=1e-6)

  def testBinaryClassificationWithCenteredBias(self):
    head = head_lib._multi_class_head(n_classes=2, enable_centered_bias=True)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1.], [1.]])
      labels = tf.constant([[1.], [0.]])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      self._assert_binary_metrics(model_fn_ops)
      _assert_variables(self, expected_global=(
          "centered_bias_weight:0",
          "train_op/centered_bias_step/centered_bias_weight/Adagrad:0",
      ), expected_trainable=(
          "centered_bias_weight:0",
      ))
      tf.global_variables_initializer().run()
      self.assertAlmostEqual(0.81326175, sess.run(model_fn_ops.loss),
                             delta=1e-6)

  def _assert_multi_class_metrics(self, model_fn_ops):
    self.assertItemsEqual((
        "accuracy",
        "loss",
    ), six.iterkeys(model_fn_ops.eval_metric_ops))

  def testMultiClass(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes=n_classes)
    with tf.Graph().as_default(), tf.Session() as sess:
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([2])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      self._assert_multi_class_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(1.5514446, sess.run(model_fn_ops.loss))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.EVAL,
                                   _noop_train_op, logits=logits)
      self.assertIsNone(model_fn_ops.train_op)

  def testMultiClassWithWeight(self):
    n_classes = 3
    head = head_lib._multi_class_head(
        n_classes=n_classes, weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session() as sess:
      features = {"label_weight": tf.constant([0.1])}
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([2])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      self._assert_multi_class_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(.15514446, sess.run(model_fn_ops.loss))

  def testInvalidNClasses(self):
    for n_classes in (None, -1, 0, 1):
      with self.assertRaisesRegexp(ValueError, "n_classes must be > 1"):
        head_lib._multi_class_head(n_classes=n_classes)


class BinarySvmModelHeadTest(tf.test.TestCase):

  def setUp(self):
    # Prediction for first example is in the right side of the hyperplane
    # (i.e., < 0) but it is within the [-1,1] margin. There is a 0.5 loss
    # incurred by this example. The 2nd prediction is outside the margin so it
    # incurs no loss at all.
    self._predictions = ((-0.5,), (1.2,))
    self._labels = (0, 1)
    self._expected_losses = (0.5, 0.0)

  def _assert_metrics(self, model_fn_ops):
    self.assertItemsEqual((
        "accuracy",
        "loss",
    ), six.iterkeys(model_fn_ops.eval_metric_ops))

  def testBinarySVMDefaultWeights(self):
    head = head_lib._binary_svm_head()
    with tf.Graph().as_default(), tf.Session():
      predictions = tf.constant(self._predictions)
      labels = tf.constant(self._labels)
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=predictions)
      self._assert_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(
          np.average(self._expected_losses), model_fn_ops.loss.eval())

    model_fn_ops = head.head_ops({}, labels,
                                 tf.contrib.learn.ModeKeys.EVAL,
                                 _noop_train_op, logits=predictions)
    self.assertIsNone(model_fn_ops.train_op)

  def testBinarySVMWithWeights(self):
    head = head_lib._binary_svm_head(weight_column_name="weights")
    with tf.Graph().as_default(), tf.Session():
      predictions = tf.constant(self._predictions)
      labels = tf.constant(self._labels)
      weights = (7.0, 11.0)
      features = {"weights": tf.constant(weights)}
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=predictions)
      self._assert_metrics(model_fn_ops)
      _assert_no_variables(self)
      self.assertAlmostEqual(
          np.sum(np.multiply(weights, self._expected_losses)) / 2.0,
          model_fn_ops.loss.eval())

  def testBinarySVMWithCenteredBias(self):
    head = head_lib._binary_svm_head(enable_centered_bias=True)
    with tf.Graph().as_default(), tf.Session():
      predictions = tf.constant(self._predictions)
      labels = tf.constant(self._labels)
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=predictions)
      self._assert_metrics(model_fn_ops)
      _assert_variables(self, expected_global=(
          "centered_bias_weight:0",
          "train_op/centered_bias_step/centered_bias_weight/Adagrad:0",
      ), expected_trainable=(
          "centered_bias_weight:0",
      ))
      tf.global_variables_initializer().run()
      self.assertAlmostEqual(
          np.average(self._expected_losses), model_fn_ops.loss.eval())


def _noop_train_op(unused_loss):
  return tf.no_op()

if __name__ == "__main__":
  tf.test.main()
