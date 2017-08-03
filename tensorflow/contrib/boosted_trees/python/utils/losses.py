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
"""Losses for Gtflow Estimator and Batch Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def per_example_logistic_loss(labels, weights, predictions):
  """Logistic loss given labels, example weights and predictions.

  Args:
    labels: Rank 2 (N, 1) tensor of per-example labels.
    weights: Rank 2 (N, 1) tensor of per-example weights.
    predictions: Rank 2 (N, 1) tensor of per-example predictions.

  Returns:
    loss: A Rank 2 (N, 1) tensor of per-example logistic loss.
    update_op: An update operation to update the loss's internal state.
  """
  labels = math_ops.to_float(labels)
  unweighted_loss = nn.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=predictions)
  return unweighted_loss * weights, control_flow_ops.no_op()


# This is classical form of Maximum entropy loss, that is twice differentiable
# (sparse_softmax_cross_entropy which is what we go for is not twice
# differentiable).
def per_example_maxent_loss(labels, weights, logits, num_classes, eps=1e-15):
  """Maximum entropy loss for multiclass problems.

  Maximum entropy is a generalization of logistic loss for the case when more
  than 2 classes are present.

  Args:
    labels: Rank 2 (N, 1) or Rank 1 (N) tensor of per-example labels.
    weights: Rank 2 (N, 1) tensor of per-example weights.
    logits: Rank 2 (N, K) tensor of per-example predictions, K - num of
    classes.
    num_classes: number of classes in classification task. Used to expand label
    indices into one-hot encodings.
    eps: tolerance, used as a minimum possible value.

  Returns:
    loss: A Rank 2 (N, 1) tensor of per-example maxent loss
    update_op: An update operation to update the loss's internal state.
  """
  labels = math_ops.to_int64(labels)
  # If labels are of rank 1, make them rank 2.
  labels_shape = labels.get_shape()
  if len(labels_shape) != 2:
    labels = array_ops.expand_dims(labels, 1)
  # Labels are indices of classes, convert them to one hot encodings.
  target_one_hot = array_ops.one_hot(indices=labels, depth=num_classes)
  labels = math_ops.reduce_sum(
      input_tensor=target_one_hot, reduction_indices=[1])
  labels = math_ops.to_float(labels)

  # Calculate softmax probabilities for each class.
  unnormalized_probs = math_ops.exp(logits)
  normalizers = math_ops.reduce_sum(unnormalized_probs, 1, keep_dims=True)
  softmax_predictions = math_ops.divide(unnormalized_probs,
                                        math_ops.add(normalizers, eps))

  # Pull out the probabilities for real label.
  probs_for_real_class = math_ops.reduce_sum(labels * softmax_predictions, 1)

  # Add handling for values near 0 and 1.
  zeros = array_ops.zeros_like(probs_for_real_class, dtype=logits.dtype) + eps
  one_minus_eps = array_ops.ones_like(
      probs_for_real_class, dtype=logits.dtype) - eps

  # Take maximum(eps, pred)
  cond = (probs_for_real_class >= eps)
  probs_for_real_class = array_ops.where(cond, probs_for_real_class, zeros)

  # Take minimum(1-eps, pred)
  cond = (probs_for_real_class <= 1 - eps)
  probs_for_real_class = array_ops.where(cond, probs_for_real_class,
                                         one_minus_eps)

  unweighted_loss = array_ops.expand_dims(-math_ops.log(probs_for_real_class),
                                          1)
  return unweighted_loss * weights, control_flow_ops.no_op()


def per_example_squared_loss(labels, weights, predictions):
  """Squared loss given labels, example weights and predictions.

  Args:
    labels: Rank 2 (N, D) tensor of per-example labels.
    weights: Rank 2 (N, 1) tensor of per-example weights.
    predictions: Rank 2 (N, D) tensor of per-example predictions.

  Returns:
    loss: A Rank 2 (N, 1) tensor of per-example squared loss.
    update_op: An update operation to update the loss's internal state.
  """
  unweighted_loss = math_ops.reduce_sum(
      math_ops.square(predictions - labels), 1, keep_dims=True)

  return unweighted_loss * weights, control_flow_ops.no_op()


def per_example_exp_loss(labels, weights, predictions, name=None, eps=0.1):
  """Exponential loss given labels, example weights and predictions.

  Note that this is only for binary classification.
  If logistic loss tries to make sure that the classifier is certain of its
  predictions, exp loss says: "as long as it got it correct, even barely, i
  don't care". Can be used on noisy data, or when you don't care about getting
  the actual probabilities from the model, just the correct label.

  The loss returns is exp(-targets*modified_predictions), where
  modified_predictions are 1 if sigmoid is >= 0.5+eps (eg we predict positive
  class), -1 if sigmoid < 0.5-eps (e.g. we predict negative class) and ax+b in
  the interval 0.5-eps, 0.5+eps, where a = 1/eps, b=1/(2eps).

  Args:
    labels: Rank 2 (N, D) tensor of per-example labels.
    weights: Rank 2 (N, 1) tensor of per-example weights.
    predictions: Rank 2 (N, D) tensor of per-example predictions.
    name: A name for the operation (optional).
    eps: For the range (0.5-eps, 0.5+eps) we set the predictions to be ax+b.

  Returns:
    loss: A Rank 2 (N, 1) tensor of per-example exp loss
    update_op: An update operation to update the loss's internal state.
  """

  def exp_with_logits(name, eps, labels=None, logits=None):
    """Computes exponential loss given `logits`.

    The loss returns is exp(-targets*modified_predictions), where
    modified_predictions are 1 if sigmoid is >= 0.5+eps (eg we predict positive
    class), -1 if sigmoid < 0.5-eps (e.g. we predict negative class) and ax+b in
    the interval 0.5-eps, 0.5+eps, where a = 1/eps, b=1/(2eps).

    Args:
      name: A name for the operation (optional).
      eps: For the range (0.5-eps, 0.5+eps) we set the predictions to be ax+b.
      labels: A `Tensor` of the same type and shape as `logits`.
      logits: A `Tensor` of type `float32` or `float64`.

    Returns:
      A `Tensor` of the same shape as `logits` with the componentwise
      exponential losses.

    Raises:
      ValueError: If `logits` and `labels` do not have the same shape.
    """
    with ops.name_scope(name, "exp_loss", [logits, labels]) as name:
      logits = ops.convert_to_tensor(logits, name="logits")
      labels = ops.convert_to_tensor(labels, name="labels")
      try:
        labels.get_shape().merge_with(logits.get_shape())
      except ValueError:
        raise ValueError("logits and labels must have the same shape (%s vs %s)"
                         % (logits.get_shape(), labels.get_shape()))

    # Default threshold to switch between classes
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    ones = array_ops.ones_like(logits, dtype=logits.dtype)
    neg_ones = -array_ops.ones_like(logits, dtype=logits.dtype)

    # Convert labels to 1 and -1
    cond_labels = (labels > zeros)
    labels_converted = array_ops.where(cond_labels, ones, neg_ones)

    # Convert predictions to 1 and -1
    # The loss we build is min(1, max(-1,ax+b))
    # where a=1/eps, b=-1/2eps.

    a = 1.0 / eps
    b = -1.0 / 2 / eps
    probs = math_ops.sigmoid(logits)
    y = a * probs + b
    # Build max(-1, ax+b)
    cond = (y < -1)
    max_res = array_ops.where(cond, neg_ones, y)
    # Build min part
    cond = (max_res > 1)
    min_res = array_ops.where(cond, ones, max_res)
    preds_converted = min_res
    return math_ops.exp(-preds_converted * labels_converted)

  labels = math_ops.to_float(labels)
  unweighted_loss = exp_with_logits(
      name=name, eps=eps, labels=labels, logits=predictions)
  return unweighted_loss * weights, control_flow_ops.no_op()
