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
"""Classification metrics library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import distribute as distribute_lib

# TODO(nsilberman): move into metrics/python/ops/


def accuracy(predictions, labels, weights=None, name=None):
  """Computes the percentage of times that predictions matches labels.

  Args:
    predictions: the predicted values, a `Tensor` whose dtype and shape
                 matches 'labels'.
    labels: the ground truth values, a `Tensor` of any shape and
            bool, integer, or string dtype.
    weights: None or `Tensor` of float values to reweight the accuracy.
    name: A name for the operation (optional).

  Returns:
    Accuracy `Tensor`.

  Raises:
    ValueError: if dtypes don't match or
                if dtype is not bool, integer, or string.
  """
  if not (labels.dtype.is_integer or
          labels.dtype in (dtypes.bool, dtypes.string)):
    raise ValueError(
        'Labels should have bool, integer, or string dtype, not %r' %
        labels.dtype)
  if not labels.dtype.is_compatible_with(predictions.dtype):
    raise ValueError('Dtypes of predictions and labels should match. '
                     'Given: predictions (%r) and labels (%r)' %
                     (predictions.dtype, labels.dtype))
  with ops.name_scope(name, 'accuracy', values=[predictions, labels]):
    is_correct = math_ops.cast(
        math_ops.equal(predictions, labels), dtypes.float32)
    if weights is not None:
      is_correct = math_ops.multiply(is_correct, weights)
      num_values = math_ops.multiply(weights, array_ops.ones_like(is_correct))
      return math_ops.div(math_ops.reduce_sum(is_correct),
                          math_ops.reduce_sum(num_values))
    return math_ops.reduce_mean(is_correct)


def f1_score(labels, predictions, weights=None, num_thresholds=200,
             metrics_collections=None, updates_collections=None, name=None):
  """Computes the approximately best F1-score across different thresholds.

  The f1_score function applies a range of thresholds to the predictions to
  convert them from [0, 1] to bool. Precision and recall are computed by
  comparing them to the labels. The F1-Score is then defined as
  2 * precision * recall / (precision + recall). The best one across the
  thresholds is returned.

  Disclaimer: In practice it may be desirable to choose the best threshold on
  the validation set and evaluate the F1 score with this threshold on a
  separate test set. Or it may be desirable to use a fixed threshold (e.g. 0.5).

  This function internally creates four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives` that are used to
  compute the pairs of recall and precision values for a linearly spaced set of
  thresholds from which the best f1-score is derived.

  This value is ultimately returned as `f1-score`, an idempotent operation that
  computes the F1-score (computed using the aforementioned variables). The
  `num_thresholds` variable controls the degree of discretization with larger
  numbers of thresholds more closely approximating the true best F1-score.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the F1-score.

  Example usage with a custom estimator:
  def model_fn(features, labels, mode):
    predictions = make_predictions(features)
    loss = make_loss(predictions, labels)
    train_op = tf.contrib.training.create_train_op(
          total_loss=loss,
          optimizer='Adam')
    eval_metric_ops = {'f1': f1_score(labels, predictions)}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs)
  estimator = tf.estimator.Estimator(model_fn=model_fn)

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use when discretizing the roc
      curve.
    metrics_collections: An optional list of collections that `f1_score` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    f1_score: A scalar `Tensor` representing the current best f1-score across
      different thresholds.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches the `f1_score`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  with variable_scope.variable_scope(
      name, 'f1', (labels, predictions, weights)):
    predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
        predictions=predictions, labels=labels, weights=weights)
    # To account for floating point imprecisions / avoid division by zero.
    epsilon = 1e-7
    thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                  for i in range(num_thresholds - 2)]
    thresholds = [0.0 - epsilon] + thresholds + [1.0 + epsilon]

    # Confusion matrix.
    values, update_ops = metrics_impl._confusion_matrix_at_thresholds(  # pylint: disable=protected-access
        labels, predictions, thresholds, weights, includes=('tp', 'fp', 'fn'))

    # Compute precision and recall at various thresholds.
    def compute_best_f1_score(tp, fp, fn, name):
      precision_at_t = math_ops.div(tp, epsilon + tp + fp,
                                    name='precision_' + name)
      recall_at_t = math_ops.div(tp, epsilon + tp + fn, name='recall_' + name)
      # Compute F1 score.
      f1_at_thresholds = (
          2.0 * precision_at_t * recall_at_t /
          (precision_at_t + recall_at_t + epsilon))
      return math_ops.reduce_max(f1_at_thresholds)

    def f1_across_towers(_, values):
      best_f1 = compute_best_f1_score(tp=values['tp'], fp=values['fp'],
                                      fn=values['fn'], name='value')
      if metrics_collections:
        ops.add_to_collections(metrics_collections, best_f1)
      return best_f1

    best_f1 = distribute_lib.get_tower_context().merge_call(
        f1_across_towers, values)

    update_op = compute_best_f1_score(tp=update_ops['tp'], fp=update_ops['fp'],
                                      fn=update_ops['fn'], name='update')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return best_f1, update_op
