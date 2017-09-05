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
"""ModelFnOps tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.client import session
from tensorflow.python.estimator.export import export_output as core_export_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session


class ModelFnopsTest(test.TestCase):
  """Multi-output tests."""

  def create_predictions(self):
    probabilities = constant_op.constant([1., 1., 1.])
    scores = constant_op.constant([1., 2., 3.])
    classes = constant_op.constant([b"0", b"1", b"2"])
    return {
        "probabilities": probabilities,
        "scores": scores,
        "classes": classes}

  def create_model_fn_ops(self, predictions, output_alternatives,
                          mode=model_fn.ModeKeys.INFER):

    return model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions=predictions,
        loss=constant_op.constant([1]),
        train_op=control_flow_ops.no_op(),
        eval_metric_ops={
            "metric_key": (constant_op.constant(1.), control_flow_ops.no_op()),
            "loss": (constant_op.constant(1.), control_flow_ops.no_op()),
        },
        training_chief_hooks=[basic_session_run_hooks.StepCounterHook()],
        training_hooks=[basic_session_run_hooks.StepCounterHook()],
        output_alternatives=output_alternatives,
        scaffold=monitored_session.Scaffold())

  def assertEquals_except_export_and_eval_loss(
      self, model_fn_ops, estimator_spec):
    expected_eval_metric_ops = {}
    for key, value in six.iteritems(model_fn_ops.eval_metric_ops):
      if key != "loss":
        expected_eval_metric_ops[key] = value
    self.assertEqual(model_fn_ops.predictions, estimator_spec.predictions)
    self.assertEqual(model_fn_ops.loss, estimator_spec.loss)
    self.assertEqual(model_fn_ops.train_op, estimator_spec.train_op)
    self.assertEqual(expected_eval_metric_ops,
                     estimator_spec.eval_metric_ops)
    self.assertAllEqual(model_fn_ops.training_chief_hooks,
                        estimator_spec.training_chief_hooks)
    self.assertAllEqual(model_fn_ops.training_hooks,
                        estimator_spec.training_hooks)
    self.assertEqual(model_fn_ops.scaffold, estimator_spec.scaffold)

  def testEstimatorSpec_except_export(self):
    predictions = self.create_predictions()
    model_fn_ops = self.create_model_fn_ops(
        predictions, None, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

  def testEstimatorSpec_export_regression_with_scores(self):
    predictions = self.create_predictions()
    output_alternatives = {"regression_head": (
        constants.ProblemType.LINEAR_REGRESSION, predictions)}
    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      regression_output = estimator_spec.export_outputs["regression_head"]
      self.assertTrue(isinstance(
          regression_output, core_export_lib.RegressionOutput))
      self.assertAllEqual(predictions["scores"].eval(),
                          regression_output.value.eval())

  def testEstimatorSpec_export_regression_with_probabilities(self):
    predictions = self.create_predictions()
    output_alternatives_predictions = predictions.copy()
    del output_alternatives_predictions["scores"]
    output_alternatives = {"regression_head": (
        constants.ProblemType.LINEAR_REGRESSION,
        output_alternatives_predictions)}
    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      regression_output = estimator_spec.export_outputs["regression_head"]
      self.assertTrue(isinstance(
          regression_output, core_export_lib.RegressionOutput))
      self.assertAllEqual(predictions["probabilities"].eval(),
                          regression_output.value.eval())

  def testEstimatorSpec_export_classification(self):
    predictions = self.create_predictions()
    output_alternatives = {"classification_head": (
        constants.ProblemType.CLASSIFICATION, predictions)}
    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      classification_output = estimator_spec.export_outputs[
          "classification_head"]
      self.assertTrue(isinstance(classification_output,
                                 core_export_lib.ClassificationOutput))
      self.assertAllEqual(predictions["scores"].eval(),
                          classification_output.scores.eval())
      self.assertAllEqual(predictions["classes"].eval(),
                          classification_output.classes.eval())

  def testEstimatorSpec_export_classification_with_missing_scores(self):
    predictions = self.create_predictions()
    output_alternatives_predictions = predictions.copy()
    del output_alternatives_predictions["scores"]
    output_alternatives = {"classification_head": (
        constants.ProblemType.CLASSIFICATION, output_alternatives_predictions)}
    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      classification_output = estimator_spec.export_outputs[
          "classification_head"]
      self.assertTrue(isinstance(classification_output,
                                 core_export_lib.ClassificationOutput))
      self.assertAllEqual(predictions["probabilities"].eval(),
                          classification_output.scores.eval())
      self.assertAllEqual(predictions["classes"].eval(),
                          classification_output.classes.eval())

  def testEstimatorSpec_export_classification_with_missing_scores_proba(self):
    predictions = self.create_predictions()
    output_alternatives_predictions = predictions.copy()
    del output_alternatives_predictions["scores"]
    del output_alternatives_predictions["probabilities"]
    output_alternatives = {"classification_head": (
        constants.ProblemType.CLASSIFICATION, output_alternatives_predictions)}
    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      classification_output = estimator_spec.export_outputs[
          "classification_head"]
      self.assertTrue(isinstance(classification_output,
                                 core_export_lib.ClassificationOutput))
      self.assertIsNone(classification_output.scores)
      self.assertAllEqual(predictions["classes"].eval(),
                          classification_output.classes.eval())

  def testEstimatorSpec_export_classification_with_missing_classes(self):
    predictions = self.create_predictions()
    output_alternatives_predictions = predictions.copy()
    del output_alternatives_predictions["classes"]
    output_alternatives = {"classification_head": (
        constants.ProblemType.CLASSIFICATION, output_alternatives_predictions)}
    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      classification_output = estimator_spec.export_outputs[
          "classification_head"]
      self.assertTrue(isinstance(classification_output,
                                 core_export_lib.ClassificationOutput))
      self.assertAllEqual(predictions["scores"].eval(),
                          classification_output.scores.eval())
      self.assertIsNone(classification_output.classes)

  def testEstimatorSpec_export_classification_with_nonstring_classes(self):
    predictions = self.create_predictions()
    output_alternatives_predictions = predictions.copy()
    output_alternatives_predictions["classes"] = constant_op.constant(
        [1, 2, 3])
    output_alternatives = {"classification_head": (
        constants.ProblemType.CLASSIFICATION, output_alternatives_predictions)}
    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      classification_output = estimator_spec.export_outputs[
          "classification_head"]
      self.assertTrue(isinstance(classification_output,
                                 core_export_lib.ClassificationOutput))
      self.assertAllEqual(predictions["scores"].eval(),
                          classification_output.scores.eval())
      self.assertIsNone(classification_output.classes)

  def testEstimatorSpec_export_logistic(self):
    predictions = self.create_predictions()
    output_alternatives = {"logistic_head": (
        constants.ProblemType.LOGISTIC_REGRESSION, predictions)}
    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      logistic_output = estimator_spec.export_outputs["logistic_head"]
      self.assertTrue(isinstance(logistic_output,
                                 core_export_lib.ClassificationOutput))
      self.assertAllEqual(predictions["scores"].eval(),
                          logistic_output.scores.eval())
      self.assertAllEqual(predictions["classes"].eval(),
                          logistic_output.classes.eval())

  def testEstimatorSpec_export_unspecified(self):
    predictions = self.create_predictions()
    output_alternatives = {"unspecified_head": (
        constants.ProblemType.UNSPECIFIED, predictions)}

    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec()
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      unspecified_output = estimator_spec.export_outputs["unspecified_head"]
      self.assertTrue(isinstance(unspecified_output,
                                 core_export_lib.PredictOutput))
      self.assertEqual(predictions, unspecified_output.outputs)

  def testEstimatorSpec_export_multihead(self):
    predictions = self.create_predictions()
    output_alternatives = {
        "regression_head": (
            constants.ProblemType.LINEAR_REGRESSION, predictions),
        "classification_head": (
            constants.ProblemType.CLASSIFICATION, predictions)}
    model_fn_ops = self.create_model_fn_ops(
        predictions, output_alternatives, mode=model_fn.ModeKeys.INFER)

    estimator_spec = model_fn_ops.estimator_spec("regression_head")
    self.assertEquals_except_export_and_eval_loss(model_fn_ops, estimator_spec)

    with session.Session():
      regression_output = estimator_spec.export_outputs["regression_head"]
      self.assertTrue(isinstance(
          regression_output, core_export_lib.RegressionOutput))
      self.assertAllEqual(predictions["scores"].eval(),
                          regression_output.value.eval())

      default_output = estimator_spec.export_outputs[
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
      self.assertTrue(isinstance(default_output,
                                 core_export_lib.RegressionOutput))
      self.assertAllEqual(predictions["scores"].eval(),
                          default_output.value.eval())

if __name__ == "__main__":
  test.main()
