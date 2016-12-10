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

import math
import numpy as np
import six
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.core.framework import summary_pb2


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
  _assert_variables(test_case)


# This must be called from within a tf.Session.
def _assert_metrics(
    test_case, expected_loss, expected_eval_metrics, model_fn_ops):
  test_case.assertAlmostEqual(expected_loss, model_fn_ops.loss.eval(), places=4)
  for k in six.iterkeys(expected_eval_metrics):
    test_case.assertIn(k, six.iterkeys(model_fn_ops.eval_metric_ops))
  tf.initialize_local_variables().run()
  for key, expected_value in six.iteritems(expected_eval_metrics):
    value_tensor, update_tensor = model_fn_ops.eval_metric_ops[key]
    update = update_tensor.eval()
    test_case.assertAlmostEqual(
        expected_value, update, places=4,
        msg="%s: update, expected %s, got %s." % (key, expected_value, update))
    value = value_tensor.eval()
    test_case.assertAlmostEqual(
        expected_value, value, places=4,
        msg="%s: value, expected %s, got %s." % (key, expected_value, value))


# This must be called from within a tf.Session.
def _assert_summary_tags(test_case, expected_tags=None):
  actual_tags = []
  for summary_op in tf.get_collection(tf.GraphKeys.SUMMARIES):
    summ = summary_pb2.Summary()
    summ.ParseFromString(summary_op.eval())
    actual_tags.append(summ.value[0].tag)
  test_case.assertItemsEqual(expected_tags or [], actual_tags)


def _sigmoid(x):
  return 1. / (1. + math.exp(-1 * x))


class RegressionModelHeadTest(tf.test.TestCase):

  # TODO(zakaria): test multilabel regression.
  def testRegression(self):
    head = head_lib._regression_head()
    with tf.Graph().as_default(), tf.Session():
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.constant([[0.], [1.], [1.]])
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=prediction)
      _assert_summary_tags(self, ["loss"])
      _assert_no_variables(self)
      _assert_metrics(self, 5. / 3, {"loss": 5. / 3}, model_fn_ops)

  def testRegressionEvalMode(self):
    head = head_lib._regression_head()
    with tf.Graph().as_default(), tf.Session():
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.constant([[0.], [1.], [1.]])
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.EVAL,
                                   _noop_train_op, logits=prediction)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(self, 5. / 3, {"loss": 5. / 3}, model_fn_ops)

  def testRegressionWithLabelName(self):
    label_name = "my_label"
    head = head_lib._regression_head(label_name=label_name)
    with tf.Graph().as_default(), tf.Session():
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = {label_name: tf.constant([[0.], [1.], [1.]])}
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=prediction)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(self, 5. / 3, {"loss": 5. / 3}, model_fn_ops)

  def testRegressionWithWeights(self):
    head = head_lib._regression_head(
        weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session():
      weights = ((2.,), (5.,), (0.,))
      features = {"label_weight": tf.constant(weights)}
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.constant([[0.], [1.], [1.]])
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=prediction)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(self, 2. / len(weights), {
          "loss": 2. / np.sum(weights)
      }, model_fn_ops)

  def testRegressionWithCenteredBias(self):
    head = head_lib._regression_head(enable_centered_bias=True)
    with tf.Graph().as_default(), tf.Session():
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.constant([[0.], [1.], [1.]])
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=prediction)
      _assert_variables(self, expected_global=(
          "centered_bias_weight:0",
          "centered_bias_weight/Adagrad:0",
      ), expected_trainable=(
          "centered_bias_weight:0",
      ))
      tf.global_variables_initializer().run()
      _assert_summary_tags(self, ["loss", "centered_bias/bias_0"])
      _assert_metrics(self, 5. / 3, {"loss": 5. / 3}, model_fn_ops)

  def testErrorInSparseTensorLabels(self):
    head = head_lib._regression_head()
    with tf.Graph().as_default():
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.SparseTensor(
          indices=tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
          values=tf.constant([0., 1., 1.]),
          dense_shape=[3, 1])
      with self.assertRaisesRegexp(
          ValueError, "SparseTensor is not supported as labels."):
        head.head_ops({}, labels, tf.contrib.learn.ModeKeys.TRAIN,
                      _noop_train_op, logits=prediction)


class MultiLabelModelHeadTest(tf.test.TestCase):

  def setUp(self):
    self._logits = ((1., 0., 0.),)
    self._labels = ((0, 0, 1),)

  def _expected_eval_metrics(self, expected_loss):
    return {
        "accuracy": 1. / 3,
        "auc": 1. / 4,
        "loss": expected_loss,
        "auc/class0": 1.,
        "auc/class1": 1.,
        "auc/class2": 0.,
        "labels/actual_label_mean/class0": self._labels[0][0],
        "labels/actual_label_mean/class1": self._labels[0][1],
        "labels/actual_label_mean/class2": self._labels[0][2],
        "labels/logits_mean/class0": self._logits[0][0],
        "labels/logits_mean/class1": self._logits[0][1],
        "labels/logits_mean/class2": self._logits[0][2],
        "labels/prediction_mean/class0": self._logits[0][0],
        "labels/prediction_mean/class1": self._logits[0][1],
        "labels/prediction_mean/class2": self._logits[0][2],
        "labels/probability_mean/class0": _sigmoid(self._logits[0][0]),
        "labels/probability_mean/class1": _sigmoid(self._logits[0][1]),
        "labels/probability_mean/class2": _sigmoid(self._logits[0][2]),
    }

  def testMultiLabel(self):
    n_classes = 3
    head = head_lib._multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant(self._logits)
      labels = tf.constant(self._labels)
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .89985204
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)

  def testMultiLabelEvalMode(self):
    n_classes = 3
    head = head_lib._multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([[0, 0, 1]])
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.EVAL,
                                   _noop_train_op, logits=logits)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .89985204
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)

  def testMultiLabelWithLabelName(self):
    n_classes = 3
    label_name = "my_label"
    head = head_lib._multi_label_head(
        n_classes=n_classes, label_name=label_name,
        metric_class_ids=range(n_classes))
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant([[1., 0., 0.]])
      labels = {label_name: tf.constant([[0, 0, 1]])}
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .89985204
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)

  def testMultiLabelWithWeight(self):
    n_classes = 3
    head = head_lib._multi_label_head(
        n_classes=n_classes, weight_column_name="label_weight",
        metric_class_ids=range(n_classes))
    with tf.Graph().as_default(), tf.Session():
      features = {"label_weight": tf.constant([.1])}
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([[0, 0, 1]])
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(
          self, .089985214, self._expected_eval_metrics(2.69956),
          model_fn_ops)

  def testMultiLabelWithCenteredBias(self):
    n_classes = 3
    head = head_lib._multi_label_head(
        n_classes=n_classes, enable_centered_bias=True,
        metric_class_ids=range(n_classes))
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant([[1., 0., 0.]])
      labels = tf.constant([[0, 0, 1]])
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_variables(self, expected_global=(
          "centered_bias_weight:0",
          "centered_bias_weight/Adagrad:0",
      ), expected_trainable=(
          "centered_bias_weight:0",
      ))
      tf.global_variables_initializer().run()
      _assert_summary_tags(self, ["loss",
                                  "centered_bias/bias_0",
                                  "centered_bias/bias_1",
                                  "centered_bias/bias_2"])
      expected_loss = .89985204
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)


class BinaryClassificationModelHeadTest(tf.test.TestCase):

  def setUp(self):
    self._logits = ((1.,), (1.,))
    self._labels = ((1.,), (0.,))

  def _expected_eval_metrics(self, expected_loss):
    return {
        "accuracy": 1. / 2,
        "accuracy/baseline_label_mean": np.mean(self._labels),
        "accuracy/threshold_0.500000_mean": 1. / 2,
        "auc": 1. / 2,
        "labels/actual_label_mean": np.mean(self._labels),
        "labels/prediction_mean": .731059,  # softmax
        "loss": expected_loss,
        "precision/positive_threshold_0.500000_mean": 1. / 2,
        "recall/positive_threshold_0.500000_mean": 1. / 1,
    }

  def testBinaryClassification(self):
    n_classes = 2
    head = head_lib._multi_class_head(n_classes=n_classes)
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant(self._logits)
      labels = tf.constant(self._labels)
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .81326175
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)

  def testBinaryClassificationEvalMode(self):
    n_classes = 2
    head = head_lib._multi_class_head(n_classes=n_classes)
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant(self._logits)
      labels = tf.constant(self._labels)
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.EVAL,
                                   _noop_train_op, logits=logits)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .81326175
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)

  def testErrorInSparseTensorLabels(self):
    n_classes = 2
    head = head_lib._multi_class_head(n_classes=n_classes)
    with tf.Graph().as_default():
      prediction = tf.constant([[1.], [1.], [3.]])
      labels = tf.SparseTensor(
          indices=tf.constant([[0, 0], [1, 0], [2, 0]], dtype=tf.int64),
          values=tf.constant([0, 1, 1]),
          dense_shape=[3, 1])
      with self.assertRaisesRegexp(
          ValueError, "SparseTensor is not supported as labels."):
        head.head_ops({}, labels, tf.contrib.learn.ModeKeys.TRAIN,
                      _noop_train_op, logits=prediction)

  def testBinaryClassificationWithLabelName(self):
    label_name = "my_label"
    head = head_lib._multi_class_head(n_classes=2, label_name=label_name)
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant(self._logits)
      labels = {label_name: tf.constant(self._labels)}
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .81326175
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)

  def testBinaryClassificationWithWeights(self):
    n_classes = 2
    head = head_lib._multi_class_head(
        n_classes=n_classes, weight_column_name="label_weight")
    with tf.Graph().as_default(), tf.Session():
      weights = ((1.,), (0.,))
      features = {"label_weight": tf.constant(weights)}
      logits = tf.constant(self._logits)
      labels = tf.constant(self._labels)
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_total_loss = .31326166
      _assert_metrics(
          self, expected_total_loss / len(weights), {
              "accuracy": 1. / 1,
              "accuracy/baseline_label_mean": 1. / 1,
              "accuracy/threshold_0.500000_mean": 1. / 1,
              "auc": 0. / 1,
              "labels/actual_label_mean": 1. / 1,
              "labels/prediction_mean": .731059,  # softmax
              # TODO(ptucker): Is this the correct eval loss, sum not average?
              "loss": expected_total_loss,
              "precision/positive_threshold_0.500000_mean": 1. / 1,
              "recall/positive_threshold_0.500000_mean": 1. / 1,
          }, model_fn_ops)

  def testBinaryClassificationWithCenteredBias(self):
    head = head_lib._multi_class_head(n_classes=2, enable_centered_bias=True)
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant(self._logits)
      labels = tf.constant(self._labels)
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_variables(self, expected_global=(
          "centered_bias_weight:0",
          "centered_bias_weight/Adagrad:0",
      ), expected_trainable=(
          "centered_bias_weight:0",
      ))
      tf.global_variables_initializer().run()
      _assert_summary_tags(self, ["loss", "centered_bias/bias_0"])
      expected_loss = .81326175
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)


class MultiClassModelHeadTest(tf.test.TestCase):

  def setUp(self):
    self._logits = ((1., 0., 0.),)
    self._labels = (2,)

  def _expected_eval_metrics(self, expected_loss):
    return {
        "accuracy": 0.,
        "auc": 1. / 4,
        "loss": expected_loss,
        "auc/class0": 1.,
        "auc/class1": 1.,
        "auc/class2": 0.,
        "labels/actual_label_mean/class0": 0. / 1,
        "labels/actual_label_mean/class1": 0. / 1,
        "labels/actual_label_mean/class2": 1. / 1,
        "labels/logits_mean/class0": self._logits[0][0],
        "labels/logits_mean/class1": self._logits[0][1],
        "labels/logits_mean/class2": self._logits[0][2],
        "labels/prediction_mean/class0": self._logits[0][0],
        "labels/prediction_mean/class1": self._logits[0][1],
        "labels/prediction_mean/class2": self._logits[0][2],
        "labels/probability_mean/class0": 0.576117,  # softmax
        "labels/probability_mean/class1": 0.211942,  # softmax
        "labels/probability_mean/class2": 0.211942,  # softmax
    }

  def testMultiClass(self):
    n_classes = 3
    head = head_lib._multi_class_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant(self._logits)
      labels = tf.constant(self._labels)
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.5514446
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)

  def testMultiClassEvalMode(self):
    n_classes = 3
    head = head_lib._multi_class_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with tf.Graph().as_default(), tf.Session():
      logits = tf.constant(self._logits)
      labels = tf.constant(self._labels)
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.EVAL,
                                   _noop_train_op, logits=logits)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.5514446
      _assert_metrics(
          self, expected_loss, self._expected_eval_metrics(expected_loss),
          model_fn_ops)

  def testMultiClassWithWeight(self):
    n_classes = 3
    head = head_lib._multi_class_head(
        n_classes=n_classes, weight_column_name="label_weight",
        metric_class_ids=range(n_classes))
    with tf.Graph().as_default(), tf.Session():
      weight = .1
      features = {"label_weight": tf.constant([weight])}
      logits = tf.constant(self._logits)
      labels = tf.constant(self._labels)
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=logits)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.5514446
      _assert_metrics(
          self, expected_loss * weight,
          self._expected_eval_metrics(expected_loss), model_fn_ops)

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
    self._predictions = ((-.5,), (1.2,))
    self._labels = (0, 1)
    self._expected_losses = (.5, 0.)

  def testBinarySVMDefaultWeights(self):
    head = head_lib._binary_svm_head()
    with tf.Graph().as_default(), tf.Session():
      predictions = tf.constant(self._predictions)
      labels = tf.constant(self._labels)
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=predictions)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = np.average(self._expected_losses)
      _assert_metrics(self, expected_loss, {
          "accuracy": 1.,
          "loss": expected_loss,
      }, model_fn_ops)

  def testBinarySVMEvalMode(self):
    head = head_lib._binary_svm_head()
    with tf.Graph().as_default(), tf.Session():
      predictions = tf.constant(self._predictions)
      labels = tf.constant(self._labels)
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.EVAL,
                                   _noop_train_op, logits=predictions)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = np.average(self._expected_losses)
      _assert_metrics(self, expected_loss, {
          "accuracy": 1.,
          "loss": expected_loss,
      }, model_fn_ops)

  def testBinarySVMWithLabelName(self):
    label_name = "my_label"
    head = head_lib._binary_svm_head(label_name=label_name)
    with tf.Graph().as_default(), tf.Session():
      predictions = tf.constant(self._predictions)
      labels = {label_name: tf.constant(self._labels)}
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=predictions)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = np.average(self._expected_losses)
      _assert_metrics(self, expected_loss, {
          "accuracy": 1.,
          "loss": expected_loss,
      }, model_fn_ops)

  def testBinarySVMWithWeights(self):
    head = head_lib._binary_svm_head(weight_column_name="weights")
    with tf.Graph().as_default(), tf.Session():
      predictions = tf.constant(self._predictions)
      labels = tf.constant(self._labels)
      weights = (7., 11.)
      features = {"weights": tf.constant(weights)}
      model_fn_ops = head.head_ops(features, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=predictions)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_weighted_sum = np.sum(np.multiply(
          weights, self._expected_losses))
      _assert_metrics(self, expected_weighted_sum / len(weights), {
          "accuracy": 1.,
          "loss": expected_weighted_sum / np.sum(weights),
      }, model_fn_ops)

  def testBinarySVMWithCenteredBias(self):
    head = head_lib._binary_svm_head(enable_centered_bias=True)
    with tf.Graph().as_default(), tf.Session():
      predictions = tf.constant(self._predictions)
      labels = tf.constant(self._labels)
      model_fn_ops = head.head_ops({}, labels,
                                   tf.contrib.learn.ModeKeys.TRAIN,
                                   _noop_train_op, logits=predictions)
      _assert_variables(self, expected_global=(
          "centered_bias_weight:0",
          "centered_bias_weight/Adagrad:0",
      ), expected_trainable=(
          "centered_bias_weight:0",
      ))
      tf.global_variables_initializer().run()
      _assert_summary_tags(self, ["loss", "centered_bias/bias_0"])
      expected_loss = np.average(self._expected_losses)
      _assert_metrics(self, expected_loss, {
          "accuracy": 1.,
          "loss": expected_loss,
      }, model_fn_ops)


class MultiHeadTest(tf.test.TestCase):

  def testTrain_withNoHeadWeights(self):
    head1 = head_lib._multi_class_head(n_classes=3, label_name="label1",
                                       head_name="head1")
    head2 = head_lib._multi_class_head(n_classes=4, label_name="label2",
                                       head_name="head2")
    head = head_lib._multi_head([head1, head2])
    logits = tf.constant([[-0.7, 0.2, .1, .1, .1, .1, .1]])
    labels = {
        "label1": tf.constant([1]),
        "label2": tf.constant([1])

    }
    features = {"weights": tf.constant([2.0, 10.0])}
    model_fn_ops = head.head_ops(features, labels,
                                 tf.contrib.learn.ModeKeys.TRAIN,
                                 _noop_train_op, logits=logits)

    self.assertEquals(None, model_fn_ops.predictions)
    self.assertTrue(model_fn_ops.loss is not None)
    self.assertTrue(model_fn_ops.train_op is not None)
    self.assertFalse(model_fn_ops.eval_metric_ops)
    self.assertEquals(None, model_fn_ops.signature_fn)
    self.assertEquals(None, model_fn_ops.output_alternatives)

    with tf.Session() as sess:
      self.assertAlmostEqual(2.224, sess.run(model_fn_ops.loss), places=3)

  def testTrain_withHeadWeights(self):
    head1 = head_lib._multi_class_head(n_classes=3, label_name="label1",
                                       head_name="head1")
    head2 = head_lib._multi_class_head(n_classes=4, label_name="label2",
                                       head_name="head2")
    head = head_lib._multi_head([head1, head2], [1, .5])
    logits = tf.constant([[-0.7, 0.2, .1, .1, .1, .1, .1]])
    labels = {
        "label1": tf.constant([1]),
        "label2": tf.constant([1])
    }
    features = {"weights": tf.constant([2.0, 10.0])}
    model_fn_ops = head.head_ops(features, labels,
                                 tf.contrib.learn.ModeKeys.TRAIN,
                                 _noop_train_op, logits=logits)
    self.assertEquals(None, model_fn_ops.predictions)
    self.assertTrue(model_fn_ops.loss is not None)
    self.assertTrue(model_fn_ops.train_op is not None)
    self.assertFalse(model_fn_ops.eval_metric_ops)
    self.assertEquals(None, model_fn_ops.signature_fn)
    self.assertEquals(None, model_fn_ops.output_alternatives)

    with tf.Session() as sess:
      self.assertAlmostEqual(1.531, sess.run(model_fn_ops.loss), places=3)

  def testInfer(self):
    head1 = head_lib._multi_class_head(n_classes=3, label_name="label1",
                                       head_name="head1")
    head2 = head_lib._multi_class_head(n_classes=4, label_name="label2",
                                       head_name="head2")
    head = head_lib._multi_head([head1, head2], [1, .5])
    logits = tf.constant([[-0.7, 0.2, .1, .1, .1, .1, .1]])
    labels = {
        "label1": tf.constant([1]),
        "label2": tf.constant([1])

    }
    features = {"weights": tf.constant([2.0, 10.0])}
    model_fn_ops = head.head_ops(features, labels,
                                 tf.contrib.learn.ModeKeys.INFER,
                                 _noop_train_op, logits=logits)

    self.assertTrue(model_fn_ops.predictions)
    self.assertEquals(None, model_fn_ops.loss)
    self.assertEquals(None, model_fn_ops.train_op)
    self.assertFalse(model_fn_ops.eval_metric_ops)
    self.assertEquals(None, model_fn_ops.signature_fn)
    self.assertTrue(len(model_fn_ops.output_alternatives) == 2)

    # Tests predictions keys
    pred_keys = model_fn_ops.predictions.keys()
    self.assertTrue(("head1", prediction_key.PredictionKey.PROBABILITIES) in
                    pred_keys)
    self.assertTrue(("head1", prediction_key.PredictionKey.CLASSES) in
                    pred_keys)
    self.assertTrue(("head2", prediction_key.PredictionKey.PROBABILITIES) in
                    pred_keys)
    self.assertTrue(("head2", prediction_key.PredictionKey.CLASSES) in
                    pred_keys)

    # Tests output alternative
    out_alts = model_fn_ops.output_alternatives
    self.assertEquals(constants.ProblemType.CLASSIFICATION,
                      out_alts["head1"][0])
    self.assertTrue(prediction_key.PredictionKey.PROBABILITIES in
                    out_alts["head1"][1].keys())
    self.assertTrue(prediction_key.PredictionKey.CLASSES in
                    out_alts["head1"][1].keys())

    self.assertEquals(constants.ProblemType.CLASSIFICATION,
                      out_alts["head2"][0])
    self.assertTrue(prediction_key.PredictionKey.PROBABILITIES in
                    out_alts["head2"][1].keys())
    self.assertTrue(prediction_key.PredictionKey.CLASSES in
                    out_alts["head2"][1].keys())

  def testEval(self):
    head1 = head_lib._multi_class_head(n_classes=3, label_name="label1",
                                       head_name="head1")
    head2 = head_lib._multi_class_head(n_classes=4, label_name="label2",
                                       head_name="head2")
    head = head_lib._multi_head([head1, head2], [1, .5])
    logits = tf.constant([[-0.7, 0.2, .1, .1, .1, .1, .1]])
    labels = {
        "label1": tf.constant([1]),
        "label2": tf.constant([1])

    }
    features = {"weights": tf.constant([2.0, 10.0])}
    model_fn_ops = head.head_ops(features, labels,
                                 tf.contrib.learn.ModeKeys.EVAL,
                                 _noop_train_op, logits=logits)

    self.assertTrue(model_fn_ops.predictions)
    self.assertTrue(model_fn_ops.loss is not None)
    self.assertEquals(None, model_fn_ops.train_op)
    self.assertTrue(model_fn_ops.eval_metric_ops)
    self.assertEquals(None, model_fn_ops.signature_fn)
    self.assertEquals(None, model_fn_ops.output_alternatives)

    metric_ops = model_fn_ops.eval_metric_ops

    # Tests eval keys
    self.assertTrue("accuracy/head1" in metric_ops.keys())
    self.assertTrue("accuracy/head2" in metric_ops.keys())


def _noop_train_op(unused_loss):
  return tf.no_op()

if __name__ == "__main__":
  tf.test.main()
