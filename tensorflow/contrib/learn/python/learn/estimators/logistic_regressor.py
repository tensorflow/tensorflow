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
"""Logistic regression (aka binary classifier) class (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.

This defines some useful basic metrics for using logistic regression to classify
a binary event (0 vs 1).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import metric_key
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops


def _get_model_fn_with_logistic_metrics(model_fn):
  """Returns a model_fn with additional logistic metrics.

  Args:
    model_fn: Model function with the signature:
      `(features, labels, mode) -> (predictions, loss, train_op)`.
      Expects the returned predictions to be probabilities in [0.0, 1.0].

  Returns:
    model_fn that can be used with Estimator.
  """

  def _model_fn(features, labels, mode, params):
    """Model function that appends logistic evaluation metrics."""
    thresholds = params.get('thresholds') or [.5]

    predictions, loss, train_op = model_fn(features, labels, mode)
    if mode == model_fn_lib.ModeKeys.EVAL:
      eval_metric_ops = _make_logistic_eval_metric_ops(
          labels=labels,
          predictions=predictions,
          thresholds=thresholds)
    else:
      eval_metric_ops = None
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        output_alternatives={
            'head': (constants.ProblemType.LOGISTIC_REGRESSION, {
                'predictions': predictions
            })
        })

  return _model_fn


# TODO(roumposg): Deprecate and delete after converting users to use head.
def LogisticRegressor(  # pylint: disable=invalid-name
    model_fn, thresholds=None, model_dir=None, config=None,
    feature_engineering_fn=None):
  """Builds a logistic regression Estimator for binary classification.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  This method provides a basic Estimator with some additional metrics for custom
  binary classification models, including AUC, precision/recall and accuracy.

  Example:

  ```python
    # See tf.contrib.learn.Estimator(...) for details on model_fn structure
    def my_model_fn(...):
      pass

    estimator = LogisticRegressor(model_fn=my_model_fn)

    # Input builders
    def input_fn_train:
      pass

    estimator.fit(input_fn=input_fn_train)
    estimator.predict(x=x)
  ```

  Args:
    model_fn: Model function with the signature:
      `(features, labels, mode) -> (predictions, loss, train_op)`.
      Expects the returned predictions to be probabilities in [0.0, 1.0].
    thresholds: List of floating point thresholds to use for accuracy,
      precision, and recall metrics. If `None`, defaults to `[0.5]`.
    model_dir: Directory to save model parameters, graphs, etc. This can also
      be used to load checkpoints from the directory into a estimator to
      continue training a previously saved model.
    config: A RunConfig configuration object.
    feature_engineering_fn: Feature engineering function. Takes features and
                      labels which are the output of `input_fn` and
                      returns features and labels which will be fed
                      into the model.

  Returns:
    An `Estimator` instance.
  """
  return estimator.Estimator(
      model_fn=_get_model_fn_with_logistic_metrics(model_fn),
      model_dir=model_dir,
      config=config,
      params={'thresholds': thresholds},
      feature_engineering_fn=feature_engineering_fn)


def _make_logistic_eval_metric_ops(labels, predictions, thresholds):
  """Returns a dictionary of evaluation metric ops for logistic regression.

  Args:
    labels: The labels `Tensor`, or a dict with only one `Tensor` keyed by name.
    predictions: The predictions `Tensor`.
    thresholds: List of floating point thresholds to use for accuracy,
      precision, and recall metrics.

  Returns:
    A dict of metric results keyed by name.
  """
  # If labels is a dict with a single key, unpack into a single tensor.
  labels_tensor = labels
  if isinstance(labels, dict) and len(labels) == 1:
    labels_tensor = labels.values()[0]

  metrics = {}
  metrics[metric_key.MetricKey.PREDICTION_MEAN] = metrics_lib.streaming_mean(
      predictions)
  metrics[metric_key.MetricKey.LABEL_MEAN] = metrics_lib.streaming_mean(
      labels_tensor)
  # Also include the streaming mean of the label as an accuracy baseline, as
  # a reminder to users.
  metrics[metric_key.MetricKey.ACCURACY_BASELINE] = metrics_lib.streaming_mean(
      labels_tensor)

  metrics[metric_key.MetricKey.AUC] = metrics_lib.streaming_auc(
      labels=labels_tensor, predictions=predictions)

  for threshold in thresholds:
    predictions_at_threshold = math_ops.cast(
        math_ops.greater_equal(predictions, threshold),
        dtypes.float32,
        name='predictions_at_threshold_%f' % threshold)
    metrics[metric_key.MetricKey.ACCURACY_MEAN % threshold] = (
        metrics_lib.streaming_accuracy(labels=labels_tensor,
                                       predictions=predictions_at_threshold))
    # Precision for positive examples.
    metrics[metric_key.MetricKey.PRECISION_MEAN % threshold] = (
        metrics_lib.streaming_precision(labels=labels_tensor,
                                        predictions=predictions_at_threshold))
    # Recall for positive examples.
    metrics[metric_key.MetricKey.RECALL_MEAN % threshold] = (
        metrics_lib.streaming_recall(labels=labels_tensor,
                                     predictions=predictions_at_threshold))

  return metrics
