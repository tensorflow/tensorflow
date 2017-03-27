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

# pylint: disable=g-bad-todo,g-import-not-at-top
import numpy as np
import six

from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.core.framework import summary_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses as losses_lib
from tensorflow.python.platform import test
# pylint: enable=g-bad-todo,g-import-not-at-top


def _assert_variables(test_case,
                      expected_global=None,
                      expected_model=None,
                      expected_trainable=None):
  test_case.assertItemsEqual(
      tuple([] if expected_global is None else expected_global),
      tuple([k.name for k in variables.global_variables()]))
  test_case.assertItemsEqual(
      tuple([] if expected_model is None else expected_model),
      tuple([k.name for k in variables.model_variables()]))
  test_case.assertItemsEqual(
      tuple([] if expected_trainable is None else expected_trainable),
      tuple([k.name for k in variables.trainable_variables()]))


def _assert_no_variables(test_case):
  _assert_variables(test_case)


# This must be called from within a tf.Session.
def _assert_metrics(test_case, expected_loss, expected_eval_metrics,
                    model_fn_ops):
  test_case.assertAlmostEqual(expected_loss, model_fn_ops.loss.eval(), places=4)
  for k in six.iterkeys(expected_eval_metrics):
    test_case.assertIn(k, six.iterkeys(model_fn_ops.eval_metric_ops))
  variables.initialize_local_variables().run()
  for key, expected_value in six.iteritems(expected_eval_metrics):
    value_tensor, update_tensor = model_fn_ops.eval_metric_ops[key]
    update = update_tensor.eval()
    test_case.assertAlmostEqual(
        expected_value,
        update,
        places=4,
        msg="%s: update, expected %s, got %s." % (key, expected_value, update))
    value = value_tensor.eval()
    test_case.assertAlmostEqual(
        expected_value,
        value,
        places=4,
        msg="%s: value, expected %s, got %s." % (key, expected_value, value))


# This must be called from within a tf.Session.
def _assert_summary_tags(test_case, expected_tags=None):
  actual_tags = []
  for summary_op in ops.get_collection(ops.GraphKeys.SUMMARIES):
    summ = summary_pb2.Summary()
    summ.ParseFromString(summary_op.eval())
    actual_tags.append(summ.value[0].tag)
  test_case.assertItemsEqual(expected_tags or [], actual_tags)


def _sigmoid(x):
  return 1. / (1. + math.exp(-1 * x))


class PoissonHeadTest(test.TestCase):

  def _assert_output_alternatives(self, model_fn_ops):
    self.assertEquals({
        None: constants.ProblemType.LINEAR_REGRESSION
    }, {
        k: v[0] for k, v in six.iteritems(model_fn_ops.output_alternatives)
    })

  def _log_poisson_loss(self, logits, labels):
    x = np.array([f[0] for f in logits])
    z = np.array([f[0] for f in labels])
    lpl = np.exp(x) - z * x
    stirling_approx = z * np.log(z) - z + 0.5 * np.log(2. * np.pi * z)
    lpl += np.ma.masked_array(stirling_approx, mask=(z <= 1)).filled(0.)
    return sum(lpl)/len(lpl)

  def testPoissonWithLogits(self):
    head = head_lib.poisson_regression_head()
    labels = ((0.,), (1.,), (1.,))
    logits = ((0.,), (-1.,), (3.,))
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          labels=labels,
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_summary_tags(self, ["loss"])
      _assert_no_variables(self)
      loss = self._log_poisson_loss(logits, labels)
      _assert_metrics(self, loss, {"loss": loss}, model_fn_ops)


class RegressionHeadTest(test.TestCase):

  def _assert_output_alternatives(self, model_fn_ops):
    self.assertEquals({
        None: constants.ProblemType.LINEAR_REGRESSION
    }, {
        k: v[0] for k, v in six.iteritems(model_fn_ops.output_alternatives)
    })

  # TODO(zakaria): test multilabel regression.
  def testRegressionWithLogits(self):
    head = head_lib.regression_head()
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          labels=((0.,), (1.,), (1.,)),
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=((1.,), (1.,), (3.,)))
      self._assert_output_alternatives(model_fn_ops)
      _assert_summary_tags(self, ["loss"])
      _assert_no_variables(self)
      _assert_metrics(self, 5. / 3, {"loss": 5. / 3}, model_fn_ops)

  def testRegressionWithInvalidLogits(self):
    head = head_lib.regression_head()
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(ValueError, "Dimensions.*not compatible"):
        head.create_model_fn_ops(
            {},
            labels=((0.,), (1.,), (1.,)),
            mode=model_fn.ModeKeys.TRAIN,
            train_op_fn=head_lib.no_op_train_fn,
            logits=((1., 1.), (1., 1.), (3., 1.)))

  def testRegressionWithLogitsInput(self):
    head = head_lib.regression_head()
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          labels=((0.,), (1.,), (1.,)),
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits_input=((0., 0.), (0., 0.), (0., 0.)))
      self._assert_output_alternatives(model_fn_ops)
      w = ("regression_head/logits/weights:0",
           "regression_head/logits/biases:0")
      _assert_variables(
          self, expected_global=w, expected_model=w, expected_trainable=w)
      variables.global_variables_initializer().run()
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(self, 2. / 3, {"loss": 2. / 3}, model_fn_ops)

  def testRegressionWithLogitsAndLogitsInput(self):
    head = head_lib.regression_head()
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(
          ValueError, "Both logits and logits_input supplied"):
        head.create_model_fn_ops(
            {},
            labels=((0.,), (1.,), (1.,)),
            mode=model_fn.ModeKeys.TRAIN,
            train_op_fn=head_lib.no_op_train_fn,
            logits_input=((0., 0.), (0., 0.), (0., 0.)),
            logits=((1.,), (1.,), (3.,)))

  def testRegressionEvalMode(self):
    head = head_lib.regression_head()
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          labels=((1.,), (1.,), (3.,)),
          mode=model_fn.ModeKeys.EVAL,
          train_op_fn=head_lib.no_op_train_fn,
          logits=((0.,), (1.,), (1.,)))
      self._assert_output_alternatives(model_fn_ops)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(self, 5. / 3, {"loss": 5. / 3}, model_fn_ops)

  def testRegressionWithLabelName(self):
    label_name = "my_label"
    head = head_lib.regression_head(label_name=label_name)
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          labels={label_name: ((0.,), (1.,), (1.,))},
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=((1.,), (1.,), (3.,)))
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(self, 5. / 3, {"loss": 5. / 3}, model_fn_ops)

  def testRegressionWithWeights(self):
    head = head_lib.regression_head(weight_column_name="label_weight")
    with ops.Graph().as_default(), session.Session():
      weights = ((2.,), (5.,), (0.,))
      model_fn_ops = head.create_model_fn_ops(
          features={"label_weight": weights},
          labels=((0.,), (1.,), (1.,)),
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=((1.,), (1.,), (3.,)))
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(self, 2. / len(weights), {"loss": 2. / np.sum(weights)},
                      model_fn_ops)

  def testRegressionWithCenteredBias(self):
    head = head_lib.regression_head(enable_centered_bias=True)
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          labels=((0.,), (1.,), (1.,)),
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=((1.,), (1.,), (3.,)))
      self._assert_output_alternatives(model_fn_ops)
      _assert_variables(
          self,
          expected_global=(
              "regression_head/centered_bias_weight:0",
              "regression_head/regression_head/centered_bias_weight/Adagrad:0",
          ),
          expected_trainable=("regression_head/centered_bias_weight:0",))
      variables.global_variables_initializer().run()
      _assert_summary_tags(
          self, ["loss", "regression_head/centered_bias/bias_0"])
      _assert_metrics(self, 5. / 3, {"loss": 5. / 3}, model_fn_ops)

  def testRegressionErrorInSparseTensorLabels(self):
    head = head_lib.regression_head()
    with ops.Graph().as_default():
      labels = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (2, 0)),
          values=(0., 1., 1.),
          dense_shape=(3, 1))
      with self.assertRaisesRegexp(ValueError,
                                   "SparseTensor is not supported"):
        head.create_model_fn_ops(
            {},
            labels=labels,
            mode=model_fn.ModeKeys.TRAIN,
            train_op_fn=head_lib.no_op_train_fn,
            logits=((1.,), (1.,), (3.,)))


class MultiLabelHeadTest(test.TestCase):

  def _assert_output_alternatives(self, model_fn_ops):
    self.assertEquals({
        None: constants.ProblemType.CLASSIFICATION
    }, {
        k: v[0] for k, v in six.iteritems(model_fn_ops.output_alternatives)
    })

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

  def testMultiLabelWithLogits(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .89985204
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testMultiLabelTwoClasses(self):
    n_classes = 2
    labels = ((0, 1),)
    logits = ((1., 0.),)
    head = head_lib.multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, labels=labels,
          train_op_fn=head_lib.no_op_train_fn, logits=logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.00320443
      _assert_metrics(self, expected_loss, {
          "accuracy": 0.,
          "auc": 0.,
          "loss": expected_loss,
          "auc/class0": 1.,
          "auc/class1": 0.,
          "labels/actual_label_mean/class0": labels[0][0],
          "labels/actual_label_mean/class1": labels[0][1],
          "labels/logits_mean/class0": logits[0][0],
          "labels/logits_mean/class1": logits[0][1],
          "labels/prediction_mean/class0": logits[0][0],
          "labels/prediction_mean/class1": logits[0][1],
          "labels/probability_mean/class0": _sigmoid(logits[0][0]),
          "labels/probability_mean/class1": _sigmoid(logits[0][1]),
      }, model_fn_ops)

  def testMultiLabelWithInvalidLogits(self):
    head = head_lib.multi_label_head(n_classes=len(self._labels[0]) + 1)
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(ValueError, "Dimensions.*not compatible"):
        head.create_model_fn_ops(
            {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
            logits=self._logits)

  def testMultiLabelWithLogitsInput(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
          logits_input=((0., 0.),))
      self._assert_output_alternatives(model_fn_ops)
      w = ("multi_label_head/logits/weights:0",
           "multi_label_head/logits/biases:0")
      _assert_variables(
          self, expected_global=w, expected_model=w, expected_trainable=w)
      variables.global_variables_initializer().run()
      _assert_summary_tags(self, ["loss"])
      expected_loss = .69314718
      _assert_metrics(self, expected_loss, {
          "accuracy": 2. / 3,
          "auc": 2. / 4,
          "loss": expected_loss,
          "auc/class0": 1.,
          "auc/class1": 1.,
          "auc/class2": 0.,
          "labels/actual_label_mean/class0": self._labels[0][0],
          "labels/actual_label_mean/class1": self._labels[0][1],
          "labels/actual_label_mean/class2": self._labels[0][2],
          "labels/logits_mean/class0": 0.,
          "labels/logits_mean/class1": 0.,
          "labels/logits_mean/class2": 0.,
          "labels/prediction_mean/class0": 0.,
          "labels/prediction_mean/class1": 0.,
          "labels/prediction_mean/class2": 0.,
          "labels/probability_mean/class0": .5,
          "labels/probability_mean/class1": .5,
          "labels/probability_mean/class2": .5,
      }, model_fn_ops)

  def testMultiLabelWithLogitsAndLogitsInput(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(
          ValueError, "Both logits and logits_input supplied"):
        head.create_model_fn_ops(
            {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
            logits_input=((0., 0.),), logits=self._logits)

  def testMultiLabelEvalMode(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.EVAL, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .89985204
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testMultiClassEvalModeWithLargeLogits(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    logits = ((2., 0., -1),)
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.EVAL, self._labels, head_lib.no_op_train_fn,
          logits=logits)
      self._assert_output_alternatives(model_fn_ops)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.377779
      expected_eval_metrics = {
          "accuracy": 1. / 3,
          "auc": 9.99999e-07,
          "loss": expected_loss,
          "auc/class0": 1.,
          "auc/class1": 1.,
          "auc/class2": 0.,
          "labels/actual_label_mean/class0": 0. / 1,
          "labels/actual_label_mean/class1": 0. / 1,
          "labels/actual_label_mean/class2": 1. / 1,
          "labels/logits_mean/class0": logits[0][0],
          "labels/logits_mean/class1": logits[0][1],
          "labels/logits_mean/class2": logits[0][2],
          "labels/prediction_mean/class0": 1,
          "labels/prediction_mean/class1": 0,
          "labels/prediction_mean/class2": 0,
          "labels/probability_mean/class0": _sigmoid(logits[0][0]),
          "labels/probability_mean/class1": _sigmoid(logits[0][1]),
          "labels/probability_mean/class2": _sigmoid(logits[0][2]),
      }
      _assert_metrics(self, expected_loss,
                      expected_eval_metrics, model_fn_ops)

  def testMultiLabelWithLabelName(self):
    n_classes = 3
    label_name = "my_label"
    head = head_lib.multi_label_head(
        n_classes=n_classes,
        label_name=label_name,
        metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, {label_name: self._labels},
          head_lib.no_op_train_fn, logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .89985204
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testMultiLabelWithWeight(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes,
        weight_column_name="label_weight",
        metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          features={"label_weight": .1},
          labels=self._labels,
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(self, .089985214,
                      self._expected_eval_metrics(2.69956), model_fn_ops)

  def testMultiLabelWithCustomLoss(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes,
        weight_column_name="label_weight",
        metric_class_ids=range(n_classes),
        loss_fn=_sigmoid_cross_entropy)
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          features={"label_weight": .1},
          labels=self._labels,
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      _assert_metrics(self, 0.089985214,
                      self._expected_eval_metrics(0.089985214), model_fn_ops)

  def testMultiLabelWithCenteredBias(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes,
        enable_centered_bias=True,
        metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_variables(
          self,
          expected_global=(
              "multi_label_head/centered_bias_weight:0",
              ("multi_label_head/multi_label_head/centered_bias_weight/"
               "Adagrad:0"),),
          expected_trainable=("multi_label_head/centered_bias_weight:0",))
      variables.global_variables_initializer().run()
      _assert_summary_tags(self, (
          "loss",
          "multi_label_head/centered_bias/bias_0",
          "multi_label_head/centered_bias/bias_1",
          "multi_label_head/centered_bias/bias_2"
      ))
      expected_loss = .89985204
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testMultiLabelSparseTensorLabels(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      labels = sparse_tensor.SparseTensorValue(
          indices=((0, 0),),
          values=(2,),
          dense_shape=(1, 1))
      model_fn_ops = head.create_model_fn_ops(
          features={},
          mode=model_fn.ModeKeys.TRAIN,
          labels=labels,
          train_op_fn=head_lib.no_op_train_fn,
          logits=self._logits)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .89985204
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testMultiLabelSparseTensorLabelsTooFewClasses(self):
    n_classes = 3
    head = head_lib.multi_label_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    # Set _logits_dimension (n_classes) to a lower value; if it's set to 1
    # upfront, the class throws an error during initialization.
    head._logits_dimension = 1
    with ops.Graph().as_default(), session.Session():
      labels = sparse_tensor.SparseTensorValue(
          indices=((0, 0),),
          values=(2,),
          dense_shape=(1, 1))
      with self.assertRaisesRegexp(ValueError,
                                   "Must set num_classes >= 2 when passing"):
        head.create_model_fn_ops(
            features={},
            labels=labels,
            mode=model_fn.ModeKeys.TRAIN,
            train_op_fn=head_lib.no_op_train_fn,
            logits=[0.])


class BinaryClassificationHeadTest(test.TestCase):

  def _assert_output_alternatives(self, model_fn_ops):
    self.assertEquals({
        None: constants.ProblemType.LOGISTIC_REGRESSION
    }, {
        k: v[0] for k, v in six.iteritems(model_fn_ops.output_alternatives)
    })

  def setUp(self):
    self._logits = ((1.,), (1.,))
    self._labels = ((1.,), (0.,))

  def _expected_eval_metrics(self, expected_loss):
    label_mean = np.mean(self._labels)
    return {
        "accuracy": 1. / 2,
        "accuracy/baseline_label_mean": label_mean,
        "accuracy/threshold_0.500000_mean": 1. / 2,
        "auc": 1. / 2,
        "labels/actual_label_mean": label_mean,
        "labels/prediction_mean": .731059,  # softmax
        "loss": expected_loss,
        "precision/positive_threshold_0.500000_mean": 1. / 2,
        "recall/positive_threshold_0.500000_mean": 1. / 1,
    }

  def testBinaryClassificationWithLogits(self):
    n_classes = 2
    head = head_lib.multi_class_head(n_classes=n_classes)
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .81326175
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testBinaryClassificationWithInvalidLogits(self):
    head = head_lib.multi_class_head(n_classes=len(self._labels) + 1)
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(ValueError, "Dimensions.*not compatible"):
        head.create_model_fn_ops(
            {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
            logits=self._logits)

  def testBinaryClassificationWithLogitsInput(self):
    n_classes = 2
    head = head_lib.multi_class_head(n_classes=n_classes)
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
          logits_input=((0., 0.), (0., 0.)))
      self._assert_output_alternatives(model_fn_ops)
      w = ("binary_logistic_head/logits/weights:0",
           "binary_logistic_head/logits/biases:0")
      _assert_variables(
          self, expected_global=w, expected_model=w, expected_trainable=w)
      variables.global_variables_initializer().run()
      _assert_summary_tags(self, ["loss"])
      expected_loss = .69314718
      label_mean = np.mean(self._labels)
      _assert_metrics(self, expected_loss, {
          "accuracy": 1. / 2,
          "accuracy/baseline_label_mean": label_mean,
          "accuracy/threshold_0.500000_mean": 1. / 2,
          "auc": 1. / 2,
          "labels/actual_label_mean": label_mean,
          "labels/prediction_mean": .5,  # softmax
          "loss": expected_loss,
          "precision/positive_threshold_0.500000_mean": 0. / 2,
          "recall/positive_threshold_0.500000_mean": 0. / 1,
      }, model_fn_ops)

  def testBinaryClassificationWithLogitsAndLogitsInput(self):
    head = head_lib.multi_class_head(n_classes=2)
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(
          ValueError, "Both logits and logits_input supplied"):
        head.create_model_fn_ops(
            {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
            logits_input=((0., 0.), (0., 0.)), logits=self._logits)

  def testBinaryClassificationEvalMode(self):
    n_classes = 2
    head = head_lib.multi_class_head(n_classes=n_classes)
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.EVAL, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .81326175
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testBinaryClassificationInferMode(self):
    n_classes = 2
    head = head_lib.multi_class_head(n_classes=n_classes)
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.INFER, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)

  def testBinaryClassificationInferMode_withWightColumn(self):
    n_classes = 2
    head = head_lib.multi_class_head(n_classes=n_classes,
                                     weight_column_name="label_weight")
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          # This is what is being tested, features should not have weight for
          # inference.
          {}, model_fn.ModeKeys.INFER, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)

  def testErrorInSparseTensorLabels(self):
    n_classes = 2
    head = head_lib.multi_class_head(n_classes=n_classes)
    with ops.Graph().as_default():
      labels = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (2, 0)),
          values=(0, 1, 1),
          dense_shape=(3, 1))
      with self.assertRaisesRegexp(ValueError,
                                   "SparseTensor is not supported"):
        head.create_model_fn_ops(
            {},
            model_fn.ModeKeys.TRAIN,
            labels,
            head_lib.no_op_train_fn,
            logits=((1.,), (1.,), (3.,)))

  def testBinaryClassificationWithLabelName(self):
    label_name = "my_label"
    head = head_lib.multi_class_head(n_classes=2, label_name=label_name)
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {},
          labels={label_name: self._labels},
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = .81326175
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testBinaryClassificationWithWeights(self):
    n_classes = 2
    head = head_lib.multi_class_head(
        n_classes=n_classes, weight_column_name="label_weight")
    with ops.Graph().as_default(), session.Session():
      weights = ((1.,), (0.,))
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          features={"label_weight": weights},
          labels=self._labels,
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_total_loss = .31326166
      _assert_metrics(
          self,
          expected_total_loss / len(weights),
          {
              "accuracy": 1. / 1,
              "accuracy/baseline_label_mean": 1. / 1,
              "accuracy/threshold_0.500000_mean": 1. / 1,
              "auc": 0. / 1,
              "labels/actual_label_mean": 1. / 1,
              "labels/prediction_mean": .731059,  # softmax
              # eval loss is weighted loss divided by sum of weights.
              "loss": expected_total_loss,
              "precision/positive_threshold_0.500000_mean": 1. / 1,
              "recall/positive_threshold_0.500000_mean": 1. / 1,
          },
          model_fn_ops)

  def testBinaryClassificationWithCustomLoss(self):
    head = head_lib.multi_class_head(
        n_classes=2, weight_column_name="label_weight",
        loss_fn=_sigmoid_cross_entropy)
    with ops.Graph().as_default(), session.Session():
      weights = ((.2,), (0.,))
      model_fn_ops = head.create_model_fn_ops(
          features={"label_weight": weights},
          labels=self._labels,
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      # expected_loss is (total_weighted_loss)/1 since htere is 1 nonzero
      # weight.
      expected_loss = 0.062652342
      _assert_metrics(
          self,
          expected_loss,
          {
              "accuracy": 1. / 1,
              "accuracy/baseline_label_mean": 1. / 1,
              "accuracy/threshold_0.500000_mean": 1. / 1,
              "auc": 0. / 1,
              "labels/actual_label_mean": 1. / 1,
              "labels/prediction_mean": .731059,  # softmax
              "loss": expected_loss,
              "precision/positive_threshold_0.500000_mean": 1. / 1,
              "recall/positive_threshold_0.500000_mean": 1. / 1,
          },
          model_fn_ops)

  def testBinaryClassificationWithCenteredBias(self):
    head = head_lib.multi_class_head(n_classes=2, enable_centered_bias=True)
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_variables(
          self,
          expected_global=(
              "binary_logistic_head/centered_bias_weight:0",
              ("binary_logistic_head/binary_logistic_head/centered_bias_weight/"
               "Adagrad:0"),),
          expected_trainable=("binary_logistic_head/centered_bias_weight:0",))
      variables.global_variables_initializer().run()
      _assert_summary_tags(
          self, ["loss", "binary_logistic_head/centered_bias/bias_0"])
      expected_loss = .81326175
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)


class MultiClassHeadTest(test.TestCase):

  def _assert_output_alternatives(self, model_fn_ops):
    self.assertEquals({
        None: constants.ProblemType.CLASSIFICATION
    }, {
        k: v[0] for k, v in six.iteritems(model_fn_ops.output_alternatives)
    })

  def setUp(self):
    self._logits = ((1., 0., 0.),)
    self._labels = (2,)

  def _expected_eval_metrics(self, expected_loss):
    return {
        "accuracy": 0.,
        "loss": expected_loss,
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

  def testMultiClassWithLogits(self):
    n_classes = 3
    head = head_lib.multi_class_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.5514447
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testMultiClassWithInvalidLogits(self):
    head = head_lib.multi_class_head(n_classes=len(self._logits[0]) + 1)
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(ValueError, "Dimensions.*not compatible"):
        head.create_model_fn_ops(
            {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
            logits=self._logits)

  def testMultiClassWithNoneTrainOpFnInTrain(self):
    head = head_lib.multi_class_head(n_classes=3)
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(
          ValueError, "train_op_fn can not be None in TRAIN mode"):
        head.create_model_fn_ops(
            {}, model_fn.ModeKeys.TRAIN, self._labels,
            train_op_fn=None,
            logits=self._logits)

  def testMultiClassWithLogitsInput(self):
    n_classes = 3
    head = head_lib.multi_class_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
          logits_input=((0., 0.),))
      self._assert_output_alternatives(model_fn_ops)
      w = ("multi_class_head/logits/weights:0",
           "multi_class_head/logits/biases:0")
      _assert_variables(
          self, expected_global=w, expected_model=w, expected_trainable=w)
      variables.global_variables_initializer().run()
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.0986123
      _assert_metrics(self, expected_loss, {
          "accuracy": 0.,
          "loss": expected_loss,
          "labels/actual_label_mean/class0": 0. / 1,
          "labels/actual_label_mean/class1": 0. / 1,
          "labels/actual_label_mean/class2": 1. / 1,
          "labels/logits_mean/class0": 0.,
          "labels/logits_mean/class1": 0.,
          "labels/logits_mean/class2": 0.,
          "labels/prediction_mean/class0": 1.,
          "labels/prediction_mean/class1": 0.,
          "labels/prediction_mean/class2": 0.,
          "labels/probability_mean/class0": 0.333333,  # softmax
          "labels/probability_mean/class1": 0.333333,  # softmax
          "labels/probability_mean/class2": 0.333333,  # softmax
      }, model_fn_ops)

  def testMultiClassWithLogitsAndLogitsInput(self):
    n_classes = 3
    head = head_lib.multi_class_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(
          ValueError, "Both logits and logits_input supplied"):
        head.create_model_fn_ops(
            {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
            logits_input=((0., 0.),), logits=self._logits)

  def testMultiClassEnableCenteredBias(self):
    n_classes = 3
    head = head_lib.multi_class_head(
        n_classes=n_classes, enable_centered_bias=True)
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_variables(
          self,
          expected_global=(
              "multi_class_head/centered_bias_weight:0",
              ("multi_class_head/multi_class_head/centered_bias_weight/"
               "Adagrad:0"),
          ),
          expected_trainable=("multi_class_head/centered_bias_weight:0",))
      variables.global_variables_initializer().run()
      _assert_summary_tags(self,
                           ["loss",
                            "multi_class_head/centered_bias/bias_0",
                            "multi_class_head/centered_bias/bias_1",
                            "multi_class_head/centered_bias/bias_2"])

  def testMultiClassEvalMode(self):
    n_classes = 3
    head = head_lib.multi_class_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.EVAL, self._labels, head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.5514447
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testMultiClassEvalModeWithLargeLogits(self):
    n_classes = 3
    head = head_lib.multi_class_head(
        n_classes=n_classes, metric_class_ids=range(n_classes))
    logits = ((2., 0., -1),)
    with ops.Graph().as_default(), session.Session():
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          {}, model_fn.ModeKeys.EVAL, self._labels, head_lib.no_op_train_fn,
          logits=logits)
      self._assert_output_alternatives(model_fn_ops)
      self.assertIsNone(model_fn_ops.train_op)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 3.1698461
      expected_eval_metrics = {
          "accuracy": 0.,
          "loss": expected_loss,
          "labels/actual_label_mean/class0": 0. / 1,
          "labels/actual_label_mean/class1": 0. / 1,
          "labels/actual_label_mean/class2": 1. / 1,
          "labels/logits_mean/class0": logits[0][0],
          "labels/logits_mean/class1": logits[0][1],
          "labels/logits_mean/class2": logits[0][2],
          "labels/prediction_mean/class0": 1,
          "labels/prediction_mean/class1": 0,
          "labels/prediction_mean/class2": 0,
          "labels/probability_mean/class0": 0.843795,  # softmax
          "labels/probability_mean/class1": 0.114195,  # softmax
          "labels/probability_mean/class2": 0.0420101,  # softmax
      }
      _assert_metrics(self, expected_loss,
                      expected_eval_metrics, model_fn_ops)

  def testMultiClassWithWeight(self):
    n_classes = 3
    head = head_lib.multi_class_head(
        n_classes=n_classes,
        weight_column_name="label_weight",
        metric_class_ids=range(n_classes))
    with ops.Graph().as_default(), session.Session():
      weight = .1
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          features={"label_weight": weight},
          labels=self._labels,
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.5514447
      _assert_metrics(self, expected_loss * weight,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testMultiClassWithCustomLoss(self):
    n_classes = 3
    head = head_lib.multi_class_head(
        n_classes=n_classes,
        weight_column_name="label_weight",
        metric_class_ids=range(n_classes),
        loss_fn=losses_lib.sparse_softmax_cross_entropy)
    with ops.Graph().as_default(), session.Session():
      weight = .1
      # logloss: z:label, x:logit
      # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      model_fn_ops = head.create_model_fn_ops(
          features={"label_weight": weight},
          labels=self._labels,
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=self._logits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.5514447 * weight
      _assert_metrics(self, expected_loss,
                      self._expected_eval_metrics(expected_loss), model_fn_ops)

  def testMultiClassInfer(self):
    n_classes = 3
    head = head_lib._multi_class_head(
        n_classes=n_classes,
        head_name="head_name")
    with ops.Graph().as_default():
      model_fn_ops = head.create_model_fn_ops(
          features={},
          mode=model_fn.ModeKeys.INFER,
          train_op_fn=head_lib.no_op_train_fn,
          logits=((1., 0., 0.), (0., 0., 1.),))
      with session.Session():
        data_flow_ops.tables_initializer().run()
        self.assertAllEqual(
            [0, 2],
            model_fn_ops.predictions["classes"].eval())
        self.assertItemsEqual(
            ["head_name"], six.iterkeys(model_fn_ops.output_alternatives))
        self.assertEqual(
            constants.ProblemType.CLASSIFICATION,
            model_fn_ops.output_alternatives["head_name"][0])
        predictions_for_serving = (
            model_fn_ops.output_alternatives["head_name"][1])
        self.assertIn("classes", six.iterkeys(predictions_for_serving))
        self.assertAllEqual(
            [[0, 1, 2], [0, 1, 2]],
            predictions_for_serving["classes"].eval())
        self.assertIn("probabilities", six.iterkeys(predictions_for_serving))
        self.assertAllClose(
            [[0.576117, 0.2119416, 0.2119416],
             [0.2119416, 0.2119416, 0.576117]],
            predictions_for_serving["probabilities"].eval())

  def testInvalidNClasses(self):
    for n_classes in (None, -1, 0, 1):
      with self.assertRaisesRegexp(ValueError, "n_classes must be > 1"):
        head_lib.multi_class_head(n_classes=n_classes)

  def testMultiClassWithLabelKeysInvalidShape(self):
    with self.assertRaisesRegexp(
        ValueError, "Length of label_keys must equal n_classes"):
      head_lib._multi_class_head(
          n_classes=3, label_keys=("key0", "key1"))

  def testMultiClassWithLabelKeysTwoClasses(self):
    with self.assertRaisesRegexp(
        ValueError, "label_keys is not supported for n_classes=2"):
      head_lib._multi_class_head(
          n_classes=2, label_keys=("key0", "key1"))

  def testMultiClassWithLabelKeysInfer(self):
    n_classes = 3
    label_keys = ("key0", "key1", "key2")
    head = head_lib._multi_class_head(
        n_classes=n_classes, label_keys=label_keys,
        metric_class_ids=range(n_classes),
        head_name="head_name")
    with ops.Graph().as_default():
      model_fn_ops = head.create_model_fn_ops(
          features={},
          mode=model_fn.ModeKeys.INFER,
          train_op_fn=head_lib.no_op_train_fn,
          logits=((1., 0., 0.), (0., 0., 1.),))
      with session.Session():
        data_flow_ops.tables_initializer().run()
        self.assertAllEqual(
            [b"key0", b"key2"],
            model_fn_ops.predictions["classes"].eval())
        self.assertItemsEqual(
            ["head_name"], six.iterkeys(model_fn_ops.output_alternatives))
        self.assertEqual(
            constants.ProblemType.CLASSIFICATION,
            model_fn_ops.output_alternatives["head_name"][0])
        predictions_for_serving = (
            model_fn_ops.output_alternatives["head_name"][1])
        self.assertIn("classes", six.iterkeys(predictions_for_serving))
        self.assertAllEqual(
            [[b"key0", b"key1", b"key2"], [b"key0", b"key1", b"key2"]],
            predictions_for_serving["classes"].eval())
        self.assertIn("probabilities", six.iterkeys(predictions_for_serving))
        self.assertAllClose(
            [[0.576117, 0.2119416, 0.2119416],
             [0.2119416, 0.2119416, 0.576117]],
            predictions_for_serving["probabilities"].eval())

  def testMultiClassWithLabelKeysEvalAccuracy0(self):
    n_classes = 3
    label_keys = ("key0", "key1", "key2")
    head = head_lib._multi_class_head(
        n_classes=n_classes,
        label_keys=label_keys)
    with ops.Graph().as_default():
      model_fn_ops = head.create_model_fn_ops(
          features={},
          mode=model_fn.ModeKeys.EVAL,
          labels=("key2",),
          train_op_fn=head_lib.no_op_train_fn,
          logits=((1., 0., 0.),))
      with session.Session():
        data_flow_ops.tables_initializer().run()
        self.assertIsNone(model_fn_ops.train_op)
        _assert_no_variables(self)
        _assert_summary_tags(self, ["loss"])
        expected_loss = 1.5514447
        expected_eval_metrics = {
            "accuracy": 0.,
            "loss": expected_loss,
        }
        _assert_metrics(self, expected_loss,
                        expected_eval_metrics, model_fn_ops)

  def testMultiClassWithLabelKeysEvalAccuracy1(self):
    n_classes = 3
    label_keys = ("key0", "key1", "key2")
    head = head_lib._multi_class_head(
        n_classes=n_classes,
        label_keys=label_keys)
    with ops.Graph().as_default():
      model_fn_ops = head.create_model_fn_ops(
          features={},
          mode=model_fn.ModeKeys.EVAL,
          labels=("key2",),
          train_op_fn=head_lib.no_op_train_fn,
          logits=((0., 0., 1.),))
      with session.Session():
        data_flow_ops.tables_initializer().run()
        self.assertIsNone(model_fn_ops.train_op)
        _assert_no_variables(self)
        _assert_summary_tags(self, ["loss"])
        expected_loss = 0.5514447
        expected_eval_metrics = {
            "accuracy": 1.,
            "loss": expected_loss,
        }
        _assert_metrics(self, expected_loss,
                        expected_eval_metrics, model_fn_ops)


class BinarySvmHeadTest(test.TestCase):

  def _assert_output_alternatives(self, model_fn_ops):
    self.assertEquals({
        None: constants.ProblemType.LOGISTIC_REGRESSION
    }, {
        k: v[0] for k, v in six.iteritems(model_fn_ops.output_alternatives)
    })

  def setUp(self):
    # Prediction for first example is in the right side of the hyperplane
    # (i.e., < 0) but it is within the [-1,1] margin. There is a 0.5 loss
    # incurred by this example. The 2nd prediction is outside the margin so it
    # incurs no loss at all.
    self._predictions = ((-.5,), (1.2,))
    self._labels = (0, 1)
    self._expected_losses = (.5, 0.)

  def testBinarySVMWithLogits(self):
    head = head_lib.binary_svm_head()
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          model_fn.ModeKeys.TRAIN,
          self._labels,
          head_lib.no_op_train_fn,
          logits=self._predictions)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = np.average(self._expected_losses)
      _assert_metrics(self, expected_loss, {
          "accuracy": 1.,
          "loss": expected_loss,
      }, model_fn_ops)

  def testBinarySVMWithInvalidLogits(self):
    head = head_lib.binary_svm_head()
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(ValueError, "Dimensions.*not compatible"):
        head.create_model_fn_ops(
            {}, model_fn.ModeKeys.TRAIN, self._labels, head_lib.no_op_train_fn,
            logits=np.ones((2, 2)))

  def testBinarySVMWithLogitsInput(self):
    head = head_lib.binary_svm_head()
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          model_fn.ModeKeys.TRAIN,
          self._labels,
          head_lib.no_op_train_fn,
          logits_input=((0., 0.), (0., 0.)))
      self._assert_output_alternatives(model_fn_ops)
      w = ("binary_svm_head/logits/weights:0",
           "binary_svm_head/logits/biases:0")
      _assert_variables(
          self, expected_global=w, expected_model=w, expected_trainable=w)
      variables.global_variables_initializer().run()
      _assert_summary_tags(self, ["loss"])
      expected_loss = 1.
      _assert_metrics(self, expected_loss, {
          "accuracy": .5,
          "loss": expected_loss,
      }, model_fn_ops)

  def testBinarySVMWithLogitsAndLogitsInput(self):
    head = head_lib.binary_svm_head()
    with ops.Graph().as_default(), session.Session():
      with self.assertRaisesRegexp(
          ValueError, "Both logits and logits_input supplied"):
        head.create_model_fn_ops(
            {},
            model_fn.ModeKeys.TRAIN,
            self._labels,
            head_lib.no_op_train_fn,
            logits_input=((0., 0.), (0., 0.)),
            logits=self._predictions)

  def testBinarySVMEvalMode(self):
    head = head_lib.binary_svm_head()
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          model_fn.ModeKeys.EVAL,
          self._labels,
          head_lib.no_op_train_fn,
          logits=self._predictions)
      self._assert_output_alternatives(model_fn_ops)
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
    head = head_lib.binary_svm_head(label_name=label_name)
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          model_fn.ModeKeys.TRAIN,
          {label_name: self._labels},
          head_lib.no_op_train_fn,
          logits=self._predictions)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_loss = np.average(self._expected_losses)
      _assert_metrics(self, expected_loss, {
          "accuracy": 1.,
          "loss": expected_loss,
      }, model_fn_ops)

  def testBinarySVMWithWeights(self):
    head = head_lib.binary_svm_head(weight_column_name="weights")
    with ops.Graph().as_default(), session.Session():
      weights = (7., 11.)
      model_fn_ops = head.create_model_fn_ops(
          features={"weights": weights},
          mode=model_fn.ModeKeys.TRAIN,
          labels=self._labels,
          train_op_fn=head_lib.no_op_train_fn,
          logits=self._predictions)
      self._assert_output_alternatives(model_fn_ops)
      _assert_no_variables(self)
      _assert_summary_tags(self, ["loss"])
      expected_weighted_sum = np.sum(
          np.multiply(weights, self._expected_losses))
      _assert_metrics(self, expected_weighted_sum / len(weights), {
          "accuracy": 1.,
          "loss": expected_weighted_sum / np.sum(weights),
      }, model_fn_ops)

  def testBinarySVMWithCenteredBias(self):
    head = head_lib.binary_svm_head(enable_centered_bias=True)
    with ops.Graph().as_default(), session.Session():
      model_fn_ops = head.create_model_fn_ops(
          {},
          model_fn.ModeKeys.TRAIN,
          self._labels,
          head_lib.no_op_train_fn,
          logits=self._predictions)
      self._assert_output_alternatives(model_fn_ops)
      _assert_variables(
          self,
          expected_global=(
              "binary_svm_head/centered_bias_weight:0",
              ("binary_svm_head/binary_svm_head/centered_bias_weight/"
               "Adagrad:0"),
          ),
          expected_trainable=("binary_svm_head/centered_bias_weight:0",))
      variables.global_variables_initializer().run()
      _assert_summary_tags(
          self, ["loss", "binary_svm_head/centered_bias/bias_0"])
      expected_loss = np.average(self._expected_losses)
      _assert_metrics(self, expected_loss, {
          "accuracy": 1.,
          "loss": expected_loss,
      }, model_fn_ops)


class MultiHeadTest(test.TestCase):

  def testInvalidHeads(self):
    named_head = head_lib.multi_class_head(
        n_classes=3, label_name="label", head_name="head1")
    unnamed_head = head_lib.multi_class_head(
        n_classes=4, label_name="label")
    with self.assertRaisesRegexp(ValueError, "must have names"):
      head_lib.multi_head((named_head, unnamed_head))

  def testTrainWithNoneTrainOpFn(self):
    head1 = head_lib.multi_class_head(
        n_classes=3, label_name="label1", head_name="head1")
    head2 = head_lib.multi_class_head(
        n_classes=4, label_name="label2", head_name="head2")
    head = head_lib.multi_head((head1, head2))
    labels = {
        "label1": (1,),
        "label2": (1,)
    }
    with self.assertRaisesRegexp(
        ValueError, "train_op_fn can not be None in TRAIN mode"):
      head.create_model_fn_ops(
          features={"weights": (2.0, 10.0)},
          labels=labels,
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=None,
          logits=((-0.7, 0.2, .1, .1, .1, .1, .1),))

  def testTrain_withNoHeadWeights(self):
    head1 = head_lib.multi_class_head(
        n_classes=3, label_name="label1", head_name="head1")
    head2 = head_lib.multi_class_head(
        n_classes=4, label_name="label2", head_name="head2")
    head = head_lib.multi_head((head1, head2))
    labels = {
        "label1": (1,),
        "label2": (1,)
    }
    model_fn_ops = head.create_model_fn_ops(
        features={"weights": (2.0, 10.0)},
        labels=labels,
        mode=model_fn.ModeKeys.TRAIN,
        train_op_fn=head_lib.no_op_train_fn,
        logits=((-0.7, 0.2, .1, .1, .1, .1, .1),))

    self.assertIsNone(model_fn_ops.predictions)
    self.assertIsNotNone(model_fn_ops.loss)
    self.assertIsNotNone(model_fn_ops.train_op)
    self.assertFalse(model_fn_ops.eval_metric_ops)
    self.assertIsNone(model_fn_ops.output_alternatives)

    with session.Session() as sess:
      self.assertAlmostEqual(2.224, sess.run(model_fn_ops.loss), places=3)

  def testTrain_withHeadWeights(self):
    head1 = head_lib.multi_class_head(
        n_classes=3, label_name="label1", head_name="head1")
    head2 = head_lib.multi_class_head(
        n_classes=4, label_name="label2", head_name="head2")
    head = head_lib.multi_head((head1, head2), (1, .5))
    labels = {
        "label1": (1,),
        "label2": (1,)
    }
    model_fn_ops = head.create_model_fn_ops(
        features={"weights": (2.0, 10.0)},
        labels=labels,
        mode=model_fn.ModeKeys.TRAIN,
        train_op_fn=head_lib.no_op_train_fn,
        logits=((-0.7, 0.2, .1, .1, .1, .1, .1),))
    self.assertIsNone(model_fn_ops.predictions)
    self.assertIsNotNone(model_fn_ops.loss)
    self.assertIsNotNone(model_fn_ops.train_op)
    self.assertFalse(model_fn_ops.eval_metric_ops)
    self.assertIsNone(model_fn_ops.output_alternatives)

    with session.Session() as sess:
      self.assertAlmostEqual(1.531, sess.run(model_fn_ops.loss), places=3)

  def testTrain_withDictLogits(self):
    head1 = head_lib.multi_class_head(
        n_classes=3, label_name="label1", head_name="head1")
    head2 = head_lib.multi_class_head(
        n_classes=4, label_name="label2", head_name="head2")
    head = head_lib.multi_head((head1, head2))
    labels = {
        "label1": (1,),
        "label2": (1,)
    }
    model_fn_ops = head.create_model_fn_ops(
        features={"weights": (2.0, 10.0)},
        labels=labels,
        mode=model_fn.ModeKeys.TRAIN,
        train_op_fn=head_lib.no_op_train_fn,
        logits={head1.head_name: ((-0.7, 0.2, .1),),
                head2.head_name: ((.1, .1, .1, .1),)})

    self.assertIsNone(model_fn_ops.predictions)
    self.assertIsNotNone(model_fn_ops.loss)
    self.assertIsNotNone(model_fn_ops.train_op)
    self.assertFalse(model_fn_ops.eval_metric_ops)
    self.assertIsNone(model_fn_ops.output_alternatives)

    with session.Session() as sess:
      self.assertAlmostEqual(2.224, sess.run(model_fn_ops.loss), places=3)

  def testInfer(self):
    head1 = head_lib.multi_class_head(
        n_classes=3, label_name="label1", head_name="head1")
    head2 = head_lib.multi_class_head(
        n_classes=4, label_name="label2", head_name="head2")
    head = head_lib.multi_head((head1, head2), (1, .5))
    labels = {
        "label1": (1,),
        "label2": (1,)
    }
    model_fn_ops = head.create_model_fn_ops(
        features={"weights": (2.0, 10.0)},
        labels=labels,
        mode=model_fn.ModeKeys.INFER,
        train_op_fn=head_lib.no_op_train_fn,
        logits=((-0.7, 0.2, .1, .1, .1, .1, .1),))

    self.assertIsNotNone(model_fn_ops.predictions)
    self.assertIsNone(model_fn_ops.loss)
    self.assertIsNone(model_fn_ops.train_op)
    self.assertFalse(model_fn_ops.eval_metric_ops)

    # Tests predictions keys.
    self.assertItemsEqual((
        ("head1", prediction_key.PredictionKey.LOGITS),
        ("head1", prediction_key.PredictionKey.PROBABILITIES),
        ("head1", prediction_key.PredictionKey.CLASSES),
        ("head2", prediction_key.PredictionKey.LOGITS),
        ("head2", prediction_key.PredictionKey.PROBABILITIES),
        ("head2", prediction_key.PredictionKey.CLASSES),
    ), model_fn_ops.predictions.keys())

    # Tests output alternative.
    self.assertEquals({
        "head1": constants.ProblemType.CLASSIFICATION,
        "head2": constants.ProblemType.CLASSIFICATION,
    }, {
        k: v[0] for k, v in six.iteritems(model_fn_ops.output_alternatives)
    })
    self.assertItemsEqual((
        prediction_key.PredictionKey.PROBABILITIES,
        prediction_key.PredictionKey.CLASSES,
    ), model_fn_ops.output_alternatives["head1"][1].keys())
    self.assertItemsEqual((
        prediction_key.PredictionKey.PROBABILITIES,
        prediction_key.PredictionKey.CLASSES,
    ), model_fn_ops.output_alternatives["head2"][1].keys())

  def testEval(self):
    head1 = head_lib.multi_class_head(
        n_classes=3, label_name="label1", head_name="head1")
    head2 = head_lib.multi_class_head(
        n_classes=4, label_name="label2", head_name="head2")
    head = head_lib.multi_head((head1, head2), (1, .5))
    labels = {
        "label1": (1,),
        "label2": (1,)
    }
    model_fn_ops = head.create_model_fn_ops(
        features={"weights": (2.0, 10.0)},
        labels=labels,
        mode=model_fn.ModeKeys.EVAL,
        train_op_fn=head_lib.no_op_train_fn,
        logits=((-0.7, 0.2, .1, .1, .1, .1, .1),))

    self.assertIsNotNone(model_fn_ops.predictions)
    self.assertIsNotNone(model_fn_ops.loss)
    self.assertIsNone(model_fn_ops.train_op)
    self.assertIsNotNone(model_fn_ops.eval_metric_ops)
    self.assertIsNone(model_fn_ops.output_alternatives)

    metric_ops = model_fn_ops.eval_metric_ops

    # Tests eval keys.
    self.assertIn("accuracy/head1", metric_ops.keys())
    self.assertIn("accuracy/head2", metric_ops.keys())


def _sigmoid_cross_entropy(labels, logits, weights):
  return losses_lib.sigmoid_cross_entropy(labels, logits, weights)


if __name__ == "__main__":
  test.main()
