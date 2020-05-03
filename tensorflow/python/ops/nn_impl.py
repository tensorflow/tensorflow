# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Implementation of Neural Net (NN) functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export


@tf_export("nn.log_poisson_loss")
def log_poisson_loss(targets, log_input, compute_full_loss=False, name=None):
  """Computes log Poisson loss given `log_input`.

  Gives the log-likelihood loss between the prediction and the target under the
  assumption that the target has a Poisson distribution.
  Caveat: By default, this is not the exact loss, but the loss minus a
    constant term [log(z!)]. That has no effect for optimization, but
    does not play well with relative loss comparisons. To compute an
    approximation of the log factorial term, specify
    compute_full_loss=True to enable Stirling's Approximation.

  For brevity, let `c = log(x) = log_input`, `z = targets`.  The log Poisson
  loss is

        -log(exp(-x) * (x^z) / z!)
      = -log(exp(-x) * (x^z)) + log(z!)
      ~ -log(exp(-x)) - log(x^z) [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
          [ Note the second term is the Stirling's Approximation for log(z!).
            It is invariant to x and does not affect optimization, though
            important for correct relative loss comparisons. It is only
            computed when compute_full_loss == True. ]
      = x - z * log(x) [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
      = exp(c) - z * c [+ z * log(z) - z + 0.5 * log(2 * pi * z)]

  Args:
    targets: A `Tensor` of the same type and shape as `log_input`.
    log_input: A `Tensor` of type `float32` or `float64`.
    compute_full_loss: whether to compute the full loss. If false, a constant
      term is dropped in favor of more efficient optimization.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `log_input` with the componentwise
    logistic losses.

  Raises:
    ValueError: If `log_input` and `targets` do not have the same shape.
  """
  with ops.name_scope(name, "log_poisson_loss", [log_input, targets]) as name:
    log_input = ops.convert_to_tensor(log_input, name="log_input")
    targets = ops.convert_to_tensor(targets, name="targets")
    try:
      targets.get_shape().merge_with(log_input.get_shape())
    except ValueError:
      raise ValueError(
          "log_input and targets must have the same shape (%s vs %s)" %
          (log_input.get_shape(), targets.get_shape()))

    result = math_ops.exp(log_input) - log_input * targets
    if compute_full_loss:
      # need to create constant tensors here so that their dtypes can be matched
      # to that of the targets.
      point_five = constant_op.constant(0.5, dtype=targets.dtype)
      two_pi = constant_op.constant(2 * math.pi, dtype=targets.dtype)

      stirling_approx = (targets * math_ops.log(targets)) - targets + (
          point_five * math_ops.log(two_pi * targets))
      zeros = array_ops.zeros_like(targets, dtype=targets.dtype)
      ones = array_ops.ones_like(targets, dtype=targets.dtype)
      cond = math_ops.logical_and(targets >= zeros, targets <= ones)
      result += array_ops.where(cond, zeros, stirling_approx)
    return result


@tf_export(v1=["nn.sigmoid_cross_entropy_with_logits"])
def sigmoid_cross_entropy_with_logits(  # pylint: disable=invalid-name
    _sentinel=None,
    labels=None,
    logits=None,
    name=None):
  """Computes sigmoid cross entropy given `logits`.

  Measures the probability error in discrete classification tasks in which each
  class is independent and not mutually exclusive.  For instance, one could
  perform multilabel classification where a picture can contain both an elephant
  and a dog at the same time.

  For brevity, let `x = logits`, `z = labels`.  The logistic loss is

        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  For x < 0, to avoid overflow in exp(-x), we reformulate the above

        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))

  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation

      max(x, 0) - x * z + log(1 + exp(-abs(x)))

  `logits` and `labels` must have the same type and shape.

  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.

  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  # pylint: disable=protected-access
  nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
                           labels, logits)
  # pylint: enable=protected-access

  with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    try:
      labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))

    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = array_ops.where(cond, logits, zeros)
    neg_abs_logits = array_ops.where(cond, -logits, logits)
    return math_ops.add(
        relu_logits - logits * labels,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=name)


# Note: intentionally calling this v2 to not allow existing code with indirect
# imports to ignore the sentinel behavior.
@tf_export("nn.sigmoid_cross_entropy_with_logits", v1=[])
def sigmoid_cross_entropy_with_logits_v2(  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    name=None):
  """Computes sigmoid cross entropy given `logits`.

  Measures the probability error in discrete classification tasks in which each
  class is independent and not mutually exclusive.  For instance, one could
  perform multilabel classification where a picture can contain both an elephant
  and a dog at the same time.

  For brevity, let `x = logits`, `z = labels`.  The logistic loss is

        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  For x < 0, to avoid overflow in exp(-x), we reformulate the above

        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))

  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation

      max(x, 0) - x * z + log(1 + exp(-abs(x)))

  `logits` and `labels` must have the same type and shape.

  Args:
    labels: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.

  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  return sigmoid_cross_entropy_with_logits(
      logits=logits, labels=labels, name=name)


@tf_export("nn.weighted_cross_entropy_with_logits", v1=[])
def weighted_cross_entropy_with_logits_v2(labels, logits, pos_weight,
                                          name=None):
  """Computes a weighted cross entropy.

  This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`,
  allows one to trade off recall and precision by up- or down-weighting the
  cost of a positive error relative to a negative error.

  The usual cross-entropy cost is defined as:

      labels * -log(sigmoid(logits)) +
          (1 - labels) * -log(1 - sigmoid(logits))

  A value `pos_weight > 1` decreases the false negative count, hence increasing
  the recall.
  Conversely setting `pos_weight < 1` decreases the false positive count and
  increases the precision.
  This can be seen from the fact that `pos_weight` is introduced as a
  multiplicative coefficient for the positive labels term
  in the loss expression:

      labels * -log(sigmoid(logits)) * pos_weight +
          (1 - labels) * -log(1 - sigmoid(logits))

  For brevity, let `x = logits`, `z = labels`, `q = pos_weight`.
  The loss is:

        qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
      = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

  Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
  the implementation uses

      (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

  `logits` and `labels` must have the same type and shape.

  Args:
    labels: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    pos_weight: A coefficient to use on the positive examples.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    weighted logistic losses.

  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    try:
      labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))

    # The logistic loss formula from above is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
    # To avoid branching, we use the combined version
    #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
    log_weight = 1 + (pos_weight - 1) * labels
    return math_ops.add(
        (1 - labels) * logits,
        log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                      nn_ops.relu(-logits)),
        name=name)


@tf_export(v1=["nn.weighted_cross_entropy_with_logits"])
@deprecated_args(None, "targets is deprecated, use labels instead", "targets")
def weighted_cross_entropy_with_logits(labels=None,
                                       logits=None,
                                       pos_weight=None,
                                       name=None,
                                       targets=None):
  """Computes a weighted cross entropy.

  This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`,
  allows one to trade off recall and precision by up- or down-weighting the
  cost of a positive error relative to a negative error.

  The usual cross-entropy cost is defined as:

      labels * -log(sigmoid(logits)) +
          (1 - labels) * -log(1 - sigmoid(logits))

  A value `pos_weight > 1` decreases the false negative count, hence increasing
  the recall.
  Conversely setting `pos_weight < 1` decreases the false positive count and
  increases the precision.
  This can be seen from the fact that `pos_weight` is introduced as a
  multiplicative coefficient for the positive labels term
  in the loss expression:

      labels * -log(sigmoid(logits)) * pos_weight +
          (1 - labels) * -log(1 - sigmoid(logits))

  For brevity, let `x = logits`, `z = labels`, `q = pos_weight`.
  The loss is:

        qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
      = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

  Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
  the implementation uses

      (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

  `logits` and `labels` must have the same type and shape.

  Args:
    labels: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    pos_weight: A coefficient to use on the positive examples.
    name: A name for the operation (optional).
    targets: Deprecated alias for labels.

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    weighted logistic losses.

  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  labels = deprecated_argument_lookup("labels", labels, "targets", targets)
  return weighted_cross_entropy_with_logits_v2(labels, logits, pos_weight, name)


@tf_export("nn.compute_average_loss")
def compute_average_loss(per_example_loss,
                         sample_weight=None,
                         global_batch_size=None):
  """Scales per-example losses with sample_weights and computes their average.

  Usage with distribution strategy and custom training loop:

  ```python
  with strategy.scope():
    def compute_loss(labels, predictions, sample_weight=None):

      # If you are using a `Loss` class instead, set reduction to `NONE` so that
      # we can do the reduction afterwards and divide by global batch size.
      per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)

      # Compute loss that is scaled by sample_weight and by global batch size.
      return tf.nn.compute_average_loss(
          per_example_loss,
          sample_weight=sample_weight,
          global_batch_size=GLOBAL_BATCH_SIZE)
  ```

  Args:
    per_example_loss: Per-example loss.
    sample_weight: Optional weighting for each example.
    global_batch_size: Optional global batch size value. Defaults to (size of
      first dimension of `losses`) * (number of replicas).

  Returns:
    Scalar loss value.
  """  # pylint: disable=g-doc-exception
  per_example_loss = ops.convert_to_tensor(per_example_loss)
  input_dtype = per_example_loss.dtype

  with losses_util.check_per_example_loss_rank(per_example_loss):
    if sample_weight is not None:
      per_example_loss = losses_util.scale_losses_by_sample_weight(
          per_example_loss, sample_weight)
    per_example_loss = math_ops.cast(per_example_loss, input_dtype)

    if global_batch_size is None:
      if ds.has_strategy() and ds.in_cross_replica_context():
        raise RuntimeError(
            "You are calling `compute_average_loss` in cross replica context, "
            "while it was expected to be called in replica context.")

      num_replicas = ds.get_strategy().num_replicas_in_sync
      per_replica_batch_size = array_ops.shape_v2(per_example_loss)[0]
      global_batch_size = per_replica_batch_size * num_replicas
      global_batch_size = math_ops.cast(global_batch_size, input_dtype)

    return math_ops.reduce_sum(per_example_loss) / global_batch_size


@tf_export("nn.scale_regularization_loss")
def scale_regularization_loss(regularization_loss):
  """Scales the sum of the given regularization losses by number of replicas.

  Usage with distribution strategy and custom training loop:

  ```python
  with strategy.scope():
    def compute_loss(self, label, predictions):
      per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)

      # Compute loss that is scaled by sample_weight and by global batch size.
      loss = tf.nn.compute_average_loss(
          per_example_loss,
          sample_weight=sample_weight,
          global_batch_size=GLOBAL_BATCH_SIZE)

      # Add scaled regularization losses.
      loss += tf.nn.scale_regularization_loss(tf.nn.l2_loss(weights))
      return loss
  ```

  Args:
    regularization_loss: Regularization loss.

  Returns:
    Scalar loss value.
  """  # pylint: disable=g-doc-exception
  if ds.has_strategy() and ds.in_cross_replica_context():
    raise RuntimeError(
        "You are calling `scale_regularization_loss` in cross replica context, "
        "while it was expected to be called in replica context.")

  num_replicas = ds.get_strategy().num_replicas_in_sync
  return math_ops.reduce_sum(regularization_loss) / num_replicas


@tf_export(v1=["nn.relu_layer"])
def relu_layer(x, weights, biases, name=None):
  """Computes Relu(x * weight + biases).

  Args:
    x: a 2D tensor.  Dimensions typically: batch, in_units
    weights: a 2D tensor.  Dimensions typically: in_units, out_units
    biases: a 1D tensor.  Dimensions: out_units
    name: A name for the operation (optional).  If not specified
      "nn_relu_layer" is used.

  Returns:
    A 2-D Tensor computing relu(matmul(x, weights) + biases).
    Dimensions typically: batch, out_units.
  """
  with ops.name_scope(name, "relu_layer", [x, weights, biases]) as name:
    x = ops.convert_to_tensor(x, name="x")
    weights = ops.convert_to_tensor(weights, name="weights")
    biases = ops.convert_to_tensor(biases, name="biases")
    xw_plus_b = nn_ops.bias_add(math_ops.matmul(x, weights), biases)
    return nn_ops.relu(xw_plus_b, name=name)


@tf_export("nn.swish")
@custom_gradient.custom_gradient
def swish(features):
  # pylint: disable=g-doc-args
  """Computes the Swish activation function: `x * sigmoid(x)`.

  Source: "Searching for Activation Functions" (Ramachandran et al. 2017)
  https://arxiv.org/abs/1710.05941

  Args:
    features: A `Tensor` representing preactivation values.
    name: A name for the operation (optional).

  Returns:
    The activation value.
  """
  # pylint: enable=g-doc-args
  features = ops.convert_to_tensor(features, name="features")

  def grad(dy):
    """Gradient for the Swish activation function"""
    # Naively, x * tf.nn.sigmoid(x) requires keeping both x and sigmoid(x)
    # around for backprop, effectively doubling the tensor's memory consumption.
    # We use a control dependency here so that sigmoid(features) is re-computed
    # during backprop (the control dep prevents it being de-duped with the
    # forward pass) and we can free the sigmoid(features) expression immediately
    # after use during the forward pass.
    with ops.control_dependencies([dy]):
      sigmoid_features = math_ops.sigmoid(features)
    activation_grad = (
        sigmoid_features * (1.0 + features * (1.0 - sigmoid_features)))
    return dy * activation_grad

  return features * math_ops.sigmoid(features), grad


# pylint: disable=redefined-builtin
@tf_export("linalg.normalize")
def normalize(tensor, ord="euclidean", axis=None, name=None):
  """Normalizes `tensor` along dimension `axis` using specified norm.

  This uses `tf.linalg.norm` to compute the norm along `axis`.

  This function can compute several different vector norms (the 1-norm, the
  Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0) and
  matrix norms (Frobenius, 1-norm, 2-norm and inf-norm).

  Args:
    tensor: `Tensor` of types `float32`, `float64`, `complex64`, `complex128`
    ord: Order of the norm. Supported values are `'fro'`, `'euclidean'`, `1`,
      `2`, `np.inf` and any positive real number yielding the corresponding
      p-norm. Default is `'euclidean'` which is equivalent to Frobenius norm if
      `tensor` is a matrix and equivalent to 2-norm for vectors.
      Some restrictions apply: a) The Frobenius norm `'fro'` is not defined for
        vectors, b) If axis is a 2-tuple (matrix norm), only `'euclidean'`,
        '`fro'`, `1`, `2`, `np.inf` are supported. See the description of `axis`
        on how to compute norms for a batch of vectors or matrices stored in a
        tensor.
    axis: If `axis` is `None` (the default), the input is considered a vector
      and a single vector norm is computed over the entire set of values in the
      tensor, i.e. `norm(tensor, ord=ord)` is equivalent to
      `norm(reshape(tensor, [-1]), ord=ord)`. If `axis` is a Python integer, the
      input is considered a batch of vectors, and `axis` determines the axis in
      `tensor` over which to compute vector norms. If `axis` is a 2-tuple of
      Python integers it is considered a batch of matrices and `axis` determines
      the axes in `tensor` over which to compute a matrix norm.
      Negative indices are supported. Example: If you are passing a tensor that
        can be either a matrix or a batch of matrices at runtime, pass
        `axis=[-2,-1]` instead of `axis=None` to make sure that matrix norms are
        computed.
    name: The name of the op.

  Returns:
    normalized: A normalized `Tensor` with the same shape as `tensor`.
    norm: The computed norms with the same shape and dtype `tensor` but the
      final axis is 1 instead. Same as running
      `tf.cast(tf.linalg.norm(tensor, ord, axis keepdims=True), tensor.dtype)`.

  Raises:
    ValueError: If `ord` or `axis` is invalid.
  """
  with ops.name_scope(name, "normalize", [tensor]) as name:
    tensor = ops.convert_to_tensor(tensor)
    norm = linalg_ops.norm(tensor, ord, axis, keepdims=True)
    norm = math_ops.cast(norm, tensor.dtype)
    normalized = tensor / norm
    return normalized, norm


@tf_export(v1=["math.l2_normalize", "linalg.l2_normalize", "nn.l2_normalize"])
@deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def l2_normalize(x, axis=None, epsilon=1e-12, name=None, dim=None):
  """Normalizes along dimension `axis` using an L2 norm.

  For a 1-D tensor with `axis = 0`, computes

      output = x / sqrt(max(sum(x**2), epsilon))

  For `x` with more dimensions, independently normalizes each 1-D slice along
  dimension `axis`.

  Args:
    x: A `Tensor`.
    axis: Dimension along which to normalize.  A scalar or a vector of
      integers.
    epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
      divisor if `norm < sqrt(epsilon)`.
    name: A name for this operation (optional).
    dim: Deprecated alias for axis.

  Returns:
    A `Tensor` with the same shape as `x`.
  """
  axis = deprecated_argument_lookup("axis", axis, "dim", dim)
  return l2_normalize_v2(x, axis, epsilon, name)


@tf_export("math.l2_normalize", "linalg.l2_normalize", "nn.l2_normalize", v1=[])
def l2_normalize_v2(x, axis=None, epsilon=1e-12, name=None):
  """Normalizes along dimension `axis` using an L2 norm.

  For a 1-D tensor with `axis = 0`, computes

      output = x / sqrt(max(sum(x**2), epsilon))

  For `x` with more dimensions, independently normalizes each 1-D slice along
  dimension `axis`.

  Args:
    x: A `Tensor`.
    axis: Dimension along which to normalize.  A scalar or a vector of
      integers.
    epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
      divisor if `norm < sqrt(epsilon)`.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` with the same shape as `x`.
  """
  with ops.name_scope(name, "l2_normalize", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    square_sum = math_ops.reduce_sum(math_ops.square(x), axis, keepdims=True)
    x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
    return math_ops.multiply(x, x_inv_norm, name=name)


def _count_nonzero(input_tensor, dtype=dtypes.int64):
  """Same as math_ops.count_nonzero.

  The reduction is done in dtype, which can be faster for 32-bit dtypes.

  Args:
      input_tensor: numeric tensor
      dtype: reduction dtype

  Returns:
      number of nonzero values with type dtype
  """
  with ops.name_scope("count_nonzero", values=[input_tensor]):
    zero = array_ops.zeros([], dtype=input_tensor.dtype)
    nonzero_count = math_ops.reduce_sum(
        math_ops.cast(
            math_ops.not_equal(input_tensor, zero),
            dtype=dtype), name="nonzero_count")
    return nonzero_count


@tf_export("math.zero_fraction", "nn.zero_fraction")
def zero_fraction(value, name=None):
  """Returns the fraction of zeros in `value`.

  If `value` is empty, the result is `nan`.

  This is useful in summaries to measure and report sparsity.  For example,

  ```python
      z = tf.nn.relu(...)
      summ = tf.compat.v1.summary.scalar('sparsity', tf.nn.zero_fraction(z))
  ```

  Args:
    value: A tensor of numeric type.
    name: A name for the operation (optional).

  Returns:
    The fraction of zeros in `value`, with type `float32`.
  """
  with ops.name_scope(name, "zero_fraction", [value]):
    value = ops.convert_to_tensor(value, name="value")
    size = array_ops.size(value, out_type=dtypes.int64)
    # If the count is small, we can save memory/CPU with an int32 reduction.
    num_nonzero = control_flow_ops.cond(
        size <= dtypes.int32.max,
        # pylint: disable=g-long-lambda
        true_fn=lambda: math_ops.cast(
            _count_nonzero(value, dtype=dtypes.int32),
            dtype=dtypes.int64),
        false_fn=lambda: _count_nonzero(value, dtype=dtypes.int64))

    with ops.name_scope("counts_to_fraction"):
      num_zero = size - num_nonzero
      num_zero_float32 = math_ops.cast(num_zero, dtype=dtypes.float32)
      size_float32 = math_ops.cast(size, dtype=dtypes.float32)
      zero_fraction_float32 = num_zero_float32 / size_float32

    return array_ops.identity(zero_fraction_float32, "fraction")


# pylint: disable=redefined-builtin
@tf_export(v1=["nn.depthwise_conv2d"])
def depthwise_conv2d(input,
                     filter,
                     strides,
                     padding,
                     rate=None,
                     name=None,
                     data_format=None,
                     dilations=None):
  """Depthwise 2-D convolution.

  Given a 4D input tensor ('NHWC' or 'NCHW' data formats)
  and a filter tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`
  containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
  applies a different filter to each input channel (expanding from 1 channel
  to `channel_multiplier` channels for each), then concatenates the results
  together.  The output has `in_channels * channel_multiplier` channels.

  In detail, with the default NHWC format,

      output[b, i, j, k * channel_multiplier + q] = sum_{di, dj}
           filter[di, dj, k, q] * input[b, strides[1] * i + rate[0] * di,
                                           strides[2] * j + rate[1] * dj, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the
  same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
  If any value in `rate` is greater than 1, we perform atrous depthwise
  convolution, in which case all values in the `strides` tensor must be equal
  to 1.

  Usage Example:
  >>> x = np.array([
  ...     [1., 2.],
  ...     [3., 4.],
  ...     [5., 6.]
  ... ], dtype=np.float32).reshape((1, 3, 2, 1))
  >>> kernel = np.array([
  ...     [1., 2.],
  ...     [3., 4]
  ... ], dtype=np.float32).reshape((2, 1, 1, 2))
  >>> tf.compat.v1.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
  ...                                  padding='VALID').numpy()
    array([[[[10., 14.],
             [14., 20.]],
            [[18., 26.],
             [22., 32.]]]], dtype=float32)

  >>> tf.compat.v1.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
  ...                                  padding=[[0, 0], [1, 0], [1, 0], [0, 0]]
  ...                                 ).numpy()
    array([[[[ 0.,  0.],
             [ 3.,  4.],
             [ 6.,  8.]],
            [[ 0.,  0.],
             [10., 14.],
             [14., 20.]],
            [[ 0.,  0.],
             [18., 26.],
             [22., 32.]]]], dtype=float32)

  Args:
    input: 4-D with shape according to `data_format`.
    filter: 4-D with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
    strides: 1-D of size 4.  The stride of the sliding window for each
      dimension of `input`.
    padding: Controls how to pad the image before applying the convolution. Can
      be the string `"SAME"` or `"VALID"` indicating the type of padding
      algorithm to use, or a list indicating the explicit paddings at the start
      and end of each dimension. When explicit padding is used and data_format
      is `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    rate: 1-D of size 2. The dilation rate in which we sample input values
      across the `height` and `width` dimensions in atrous convolution. If it is
      greater than 1, then all values of strides must be 1.
    name: A name for this operation (optional).
    data_format: The data format for input. Either "NHWC" (default) or "NCHW".
    dilations: Alias of rate.

  Returns:
    A 4-D `Tensor` with shape according to `data_format`.  E.g., for
    "NHWC" format, shape is
    `[batch, out_height, out_width, in_channels * channel_multiplier].`
  """
  rate = deprecated_argument_lookup("dilations", dilations, "rate", rate)
  with ops.name_scope(name, "depthwise", [input, filter]) as name:
    input = ops.convert_to_tensor(input, name="tensor_in")
    filter = ops.convert_to_tensor(filter, name="filter_in")
    if rate is None:
      rate = [1, 1]

    # Use depthwise_conv2d_native if executing on TPU.
    if device_context.enclosing_tpu_context() is not None:
      if data_format == "NCHW":
        dilations = [1, 1, rate[0], rate[1]]
      else:
        dilations = [1, rate[0], rate[1], 1]
      return nn_ops.depthwise_conv2d_native(
          input=input,
          filter=filter,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations,
          name=name)

    def op(input_converted, _, padding):
      return nn_ops.depthwise_conv2d_native(
          input=input_converted,
          filter=filter,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name=name)

    return nn_ops.with_space_to_batch(
        input=input,
        filter_shape=array_ops.shape(filter),
        dilation_rate=rate,
        padding=padding,
        data_format=data_format,
        op=op)


@tf_export("nn.depthwise_conv2d", v1=[])
def depthwise_conv2d_v2(input,
                        filter,
                        strides,
                        padding,
                        data_format=None,
                        dilations=None,
                        name=None):
  """Depthwise 2-D convolution.

  Given a 4D input tensor ('NHWC' or 'NCHW' data formats)
  and a filter tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`
  containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
  applies a different filter to each input channel (expanding from 1 channel
  to `channel_multiplier` channels for each), then concatenates the results
  together.  The output has `in_channels * channel_multiplier` channels.

  In detail, with the default NHWC format,

      output[b, i, j, k * channel_multiplier + q] = sum_{di, dj}
           filter[di, dj, k, q] * input[b, strides[1] * i + rate[0] * di,
                                           strides[2] * j + rate[1] * dj, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the
  same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
  If any value in `rate` is greater than 1, we perform atrous depthwise
  convolution, in which case all values in the `strides` tensor must be equal
  to 1.

  Usage Example:
  >>> x = np.array([
  ...     [1., 2.],
  ...     [3., 4.],
  ...     [5., 6.]
  ... ], dtype=np.float32).reshape((1, 3, 2, 1))
  >>> kernel = np.array([
  ...     [1., 2.],
  ...     [3., 4]
  ... ], dtype=np.float32).reshape((2, 1, 1, 2))
  >>> tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
  ...                        padding='VALID').numpy()
    array([[[[10., 14.],
             [14., 20.]],
            [[18., 26.],
             [22., 32.]]]], dtype=float32)

  >>> tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
  ...                        padding=[[0, 0], [1, 0], [1, 0], [0, 0]]).numpy()
    array([[[[ 0.,  0.],
             [ 3.,  4.],
             [ 6.,  8.]],
            [[ 0.,  0.],
             [10., 14.],
             [14., 20.]],
            [[ 0.,  0.],
             [18., 26.],
             [22., 32.]]]], dtype=float32)

  Args:
    input: 4-D with shape according to `data_format`.
    filter: 4-D with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
    strides: 1-D of size 4.  The stride of the sliding window for each
      dimension of `input`.
    padding: Controls how to pad the image before applying the convolution. Can
      be the string `"SAME"` or `"VALID"` indicating the type of padding
      algorithm to use, or a list indicating the explicit paddings at the start
      and end of each dimension. When explicit padding is used and data_format
      is `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: The data format for input. Either "NHWC" (default) or "NCHW".
    dilations: 1-D of size 2. The dilation rate in which we sample input values
      across the `height` and `width` dimensions in atrous convolution. If it is
      greater than 1, then all values of strides must be 1.
    name: A name for this operation (optional).

  Returns:
    A 4-D `Tensor` with shape according to `data_format`.  E.g., for
    "NHWC" format, shape is
    `[batch, out_height, out_width, in_channels * channel_multiplier].`
  """
  return depthwise_conv2d(input=input,
                          filter=filter,
                          strides=strides,
                          padding=padding,
                          rate=dilations,
                          name=name,
                          data_format=data_format)

# pylint: enable=redefined-builtin


# pylint: disable=redefined-builtin,line-too-long
@tf_export(v1=["nn.separable_conv2d"])
def separable_conv2d(input,
                     depthwise_filter,
                     pointwise_filter,
                     strides,
                     padding,
                     rate=None,
                     name=None,
                     data_format=None,
                     dilations=None):
  """2-D convolution with separable filters.

  Performs a depthwise convolution that acts separately on channels followed by
  a pointwise convolution that mixes channels.  Note that this is separability
  between dimensions `[1, 2]` and `3`, not spatial separability between
  dimensions `1` and `2`.

  In detail, with the default NHWC format,

      output[b, i, j, k] = sum_{di, dj, q, r}
          input[b, strides[1] * i + di, strides[2] * j + dj, q] *
          depthwise_filter[di, dj, q, r] *
          pointwise_filter[0, 0, q * channel_multiplier + r, k]

  `strides` controls the strides for the depthwise convolution only, since
  the pointwise convolution has implicit strides of `[1, 1, 1, 1]`.  Must have
  `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
  If any value in `rate` is greater than 1, we perform atrous depthwise
  convolution, in which case all values in the `strides` tensor must be equal
  to 1.

  Args:
    input: 4-D `Tensor` with shape according to `data_format`.
    depthwise_filter: 4-D `Tensor` with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
      Contains `in_channels` convolutional filters of depth 1.
    pointwise_filter: 4-D `Tensor` with shape
      `[1, 1, channel_multiplier * in_channels, out_channels]`.  Pointwise
      filter to mix channels after `depthwise_filter` has convolved spatially.
    strides: 1-D of size 4.  The strides for the depthwise convolution for
      each dimension of `input`.
    padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    rate: 1-D of size 2. The dilation rate in which we sample input values
      across the `height` and `width` dimensions in atrous convolution. If it is
      greater than 1, then all values of strides must be 1.
    name: A name for this operation (optional).
    data_format: The data format for input. Either "NHWC" (default) or "NCHW".
    dilations: Alias of rate.

  Returns:
    A 4-D `Tensor` with shape according to 'data_format'. For
      example, with data_format="NHWC", shape is [batch, out_height,
      out_width, out_channels].
  """
  rate = deprecated_argument_lookup("dilations", dilations, "rate", rate)
  with ops.name_scope(name, "separable_conv2d",
                      [input, depthwise_filter, pointwise_filter]) as name:
    input = ops.convert_to_tensor(input, name="tensor_in")
    depthwise_filter = ops.convert_to_tensor(
        depthwise_filter, name="depthwise_filter")
    pointwise_filter = ops.convert_to_tensor(
        pointwise_filter, name="pointwise_filter")

    pointwise_filter_shape = pointwise_filter.get_shape().with_rank(4)
    pointwise_filter_shape.dims[0].assert_is_compatible_with(1)
    pointwise_filter_shape.dims[1].assert_is_compatible_with(1)

    if rate is None:
      rate = [1, 1]

    # The layout of the ops in the graph are expected to be as follows:
    # depthwise_conv2d  // Conv2D op corresponding to native depthwise conv.
    # separable_conv2d  // Conv2D op corresponding to the pointwise conv.

    def op(input_converted, _, padding):
      return nn_ops.depthwise_conv2d_native(
          input=input_converted,
          filter=depthwise_filter,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name="depthwise")

    depthwise = nn_ops.with_space_to_batch(
        input=input,
        filter_shape=array_ops.shape(depthwise_filter),
        dilation_rate=rate,
        padding=padding,
        data_format=data_format,
        op=op)

    return nn_ops.conv2d(
        depthwise,
        pointwise_filter, [1, 1, 1, 1],
        padding="VALID",
        data_format=data_format,
        name=name)


@tf_export("nn.separable_conv2d", v1=[])
def separable_conv2d_v2(
    input,
    depthwise_filter,
    pointwise_filter,
    strides,
    padding,
    data_format=None,
    dilations=None,
    name=None,
):
  """2-D convolution with separable filters.

  Performs a depthwise convolution that acts separately on channels followed by
  a pointwise convolution that mixes channels.  Note that this is separability
  between dimensions `[1, 2]` and `3`, not spatial separability between
  dimensions `1` and `2`.

  In detail, with the default NHWC format,

      output[b, i, j, k] = sum_{di, dj, q, r}
          input[b, strides[1] * i + di, strides[2] * j + dj, q] *
          depthwise_filter[di, dj, q, r] *
          pointwise_filter[0, 0, q * channel_multiplier + r, k]

  `strides` controls the strides for the depthwise convolution only, since
  the pointwise convolution has implicit strides of `[1, 1, 1, 1]`.  Must have
  `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
  If any value in `rate` is greater than 1, we perform atrous depthwise
  convolution, in which case all values in the `strides` tensor must be equal
  to 1.

  Args:
    input: 4-D `Tensor` with shape according to `data_format`.
    depthwise_filter: 4-D `Tensor` with shape `[filter_height, filter_width,
      in_channels, channel_multiplier]`. Contains `in_channels` convolutional
      filters of depth 1.
    pointwise_filter: 4-D `Tensor` with shape `[1, 1, channel_multiplier *
      in_channels, out_channels]`.  Pointwise filter to mix channels after
      `depthwise_filter` has convolved spatially.
    strides: 1-D of size 4.  The strides for the depthwise convolution for each
      dimension of `input`.
    padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm. See
      the "returns" section of `tf.nn.convolution` for details.
    data_format: The data format for input. Either "NHWC" (default) or "NCHW".
    dilations: 1-D of size 2. The dilation rate in which we sample input values
      across the `height` and `width` dimensions in atrous convolution. If it is
      greater than 1, then all values of strides must be 1.
    name: A name for this operation (optional).

  Returns:
    A 4-D `Tensor` with shape according to 'data_format'. For
      example, with data_format="NHWC", shape is [batch, out_height,
      out_width, out_channels].
  """
  return separable_conv2d(
      input,
      depthwise_filter,
      pointwise_filter,
      strides,
      padding,
      rate=dilations,
      name=name,
      data_format=data_format)

# pylint: enable=redefined-builtin,line-too-long


@tf_export(v1=["nn.sufficient_statistics"])
def sufficient_statistics(x, axes, shift=None, keep_dims=None, name=None,
                          keepdims=None):
  """Calculate the sufficient statistics for the mean and variance of `x`.

  These sufficient statistics are computed using the one pass algorithm on
  an input that's optionally shifted. See:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data

  Args:
    x: A `Tensor`.
    axes: Array of ints. Axes along which to compute mean and variance.
    shift: A `Tensor` containing the value by which to shift the data for
      numerical stability, or `None` if no shift is to be performed. A shift
      close to the true mean provides the most numerically stable results.
    keep_dims: produce statistics with the same dimensionality as the input.
    name: Name used to scope the operations that compute the sufficient stats.
    keepdims: Alias for keep_dims.

  Returns:
    Four `Tensor` objects of the same type as `x`:

    * the count (number of elements to average over).
    * the (possibly shifted) sum of the elements in the array.
    * the (possibly shifted) sum of squares of the elements in the array.
    * the shift by which the mean must be corrected or None if `shift` is None.
  """
  axes = list(set(axes))
  keep_dims = deprecated_argument_lookup(
      "keepdims", keepdims, "keep_dims", keep_dims)
  if keep_dims is None:
    keep_dims = False
  with ops.name_scope(name, "sufficient_statistics", [x, shift]):
    x = ops.convert_to_tensor(x, name="x")
    x_shape = x.get_shape()
    if x_shape.rank is not None and all(
        x_shape.dims[d].value is not None for d in axes):
      counts = 1
      for d in axes:
        counts *= x_shape.dims[d].value
      counts = constant_op.constant(counts, dtype=x.dtype)
    else:  # shape needs to be inferred at runtime.
      x_dims = array_ops.gather(
          math_ops.cast(array_ops.shape(x), x.dtype), axes)
      counts = math_ops.reduce_prod(x_dims, name="count")
    if shift is not None:
      shift = ops.convert_to_tensor(shift, name="shift")
      m_ss = math_ops.subtract(x, shift)
      v_ss = math_ops.squared_difference(x, shift)
    else:  # no shift.
      m_ss = x
      v_ss = math_ops.square(x)
    m_ss = math_ops.reduce_sum(m_ss, axes, keepdims=keep_dims, name="mean_ss")
    v_ss = math_ops.reduce_sum(v_ss, axes, keepdims=keep_dims, name="var_ss")
  return counts, m_ss, v_ss, shift


@tf_export("nn.sufficient_statistics", v1=[])
def sufficient_statistics_v2(x, axes, shift=None, keepdims=False, name=None):
  """Calculate the sufficient statistics for the mean and variance of `x`.

  These sufficient statistics are computed using the one pass algorithm on
  an input that's optionally shifted. See:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data

  Args:
    x: A `Tensor`.
    axes: Array of ints. Axes along which to compute mean and variance.
    shift: A `Tensor` containing the value by which to shift the data for
      numerical stability, or `None` if no shift is to be performed. A shift
      close to the true mean provides the most numerically stable results.
    keepdims: produce statistics with the same dimensionality as the input.
    name: Name used to scope the operations that compute the sufficient stats.

  Returns:
    Four `Tensor` objects of the same type as `x`:

    * the count (number of elements to average over).
    * the (possibly shifted) sum of the elements in the array.
    * the (possibly shifted) sum of squares of the elements in the array.
    * the shift by which the mean must be corrected or None if `shift` is None.
  """
  return sufficient_statistics(
      x=x, axes=axes, shift=shift, keep_dims=keepdims, name=name)


@tf_export("nn.normalize_moments")
def normalize_moments(counts, mean_ss, variance_ss, shift, name=None):
  """Calculate the mean and variance of based on the sufficient statistics.

  Args:
    counts: A `Tensor` containing the total count of the data (one value).
    mean_ss: A `Tensor` containing the mean sufficient statistics: the (possibly
      shifted) sum of the elements to average over.
    variance_ss: A `Tensor` containing the variance sufficient statistics: the
      (possibly shifted) squared sum of the data to compute the variance over.
    shift: A `Tensor` containing the value by which the data is shifted for
      numerical stability, or `None` if no shift was performed.
    name: Name used to scope the operations that compute the moments.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  with ops.name_scope(name, "normalize", [counts, mean_ss, variance_ss, shift]):
    divisor = math_ops.reciprocal(counts, name="divisor")
    if shift is not None:
      shifted_mean = math_ops.multiply(mean_ss, divisor, name="shifted_mean")
      mean = math_ops.add(shifted_mean, shift, name="mean")
    else:  # no shift.
      shifted_mean = math_ops.multiply(mean_ss, divisor, name="mean")
      mean = shifted_mean
    variance = math_ops.subtract(
        math_ops.multiply(variance_ss, divisor),
        math_ops.square(shifted_mean),
        name="variance")
  return (mean, variance)


@tf_export(v1=["nn.moments"])
def moments(
    x,
    axes,
    shift=None,  # pylint: disable=unused-argument
    name=None,
    keep_dims=None,
    keepdims=None):
  """Calculate the mean and variance of `x`.

  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.

  Note: shift is currently not used; the true mean is computed and used.

  When using these moments for batch normalization (see
  `tf.nn.batch_normalization`):

   * for so-called "global normalization", used with convolutional filters with
     shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
   * for simple batch normalization pass `axes=[0]` (batch only).

  Args:
    x: A `Tensor`.
    axes: Array of ints.  Axes along which to compute mean and
      variance.
    shift: Not used in the current implementation
    name: Name used to scope the operations that compute the moments.
    keep_dims: produce moments with the same dimensionality as the input.
    keepdims: Alias to keep_dims.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  keep_dims = deprecated_argument_lookup(
      "keepdims", keepdims, "keep_dims", keep_dims)
  if keep_dims is None:
    keep_dims = False
  with ops.name_scope(name, "moments", [x, axes]):
    # The dynamic range of fp16 is too limited to support the collection of
    # sufficient statistics. As a workaround we simply perform the operations
    # on 32-bit floats before converting the mean and variance back to fp16
    y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
    # Compute true mean while keeping the dims for proper broadcasting.
    mean = math_ops.reduce_mean(y, axes, keepdims=True, name="mean")
    # sample variance, not unbiased variance
    # Note: stop_gradient does not change the gradient that gets
    #       backpropagated to the mean from the variance calculation,
    #       because that gradient is zero
    variance = math_ops.reduce_mean(
        math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
        axes,
        keepdims=True,
        name="variance")
    if not keep_dims:
      mean = array_ops.squeeze(mean, axes)
      variance = array_ops.squeeze(variance, axes)
    if x.dtype == dtypes.float16:
      return (math_ops.cast(mean, dtypes.float16),
              math_ops.cast(variance, dtypes.float16))
    else:
      return (mean, variance)


@tf_export("nn.moments", v1=[])
def moments_v2(
    x,
    axes,
    shift=None,
    keepdims=False,
    name=None):
  """Calculates the mean and variance of `x`.

  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.

  Note: shift is currently not used; the true mean is computed and used.

  When using these moments for batch normalization (see
  `tf.nn.batch_normalization`):

   * for so-called "global normalization", used with convolutional filters with
     shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
   * for simple batch normalization pass `axes=[0]` (batch only).

  Args:
    x: A `Tensor`.
    axes: Array of ints.  Axes along which to compute mean and
      variance.
    shift: Not used in the current implementation.
    keepdims: produce moments with the same dimensionality as the input.
    name: Name used to scope the operations that compute the moments.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  return moments(x=x, axes=axes, shift=shift, name=name, keep_dims=keepdims)


@tf_export(v1=["nn.weighted_moments"])
def weighted_moments(x, axes, frequency_weights, name=None, keep_dims=None,
                     keepdims=None):
  """Returns the frequency-weighted mean and variance of `x`.

  Args:
    x: A tensor.
    axes: 1-d tensor of int32 values; these are the axes along which
      to compute mean and variance.
    frequency_weights: A tensor of positive weights which can be
      broadcast with x.
    name: Name used to scope the operation.
    keep_dims: Produce moments with the same dimensionality as the input.
    keepdims: Alias of keep_dims.

  Returns:
    Two tensors: `weighted_mean` and `weighted_variance`.
  """
  keep_dims = deprecated_argument_lookup(
      "keepdims", keepdims, "keep_dims", keep_dims)
  if keep_dims is None:
    keep_dims = False
  with ops.name_scope(name, "weighted_moments", [x, frequency_weights, axes]):
    x = ops.convert_to_tensor(x, name="x")
    frequency_weights = ops.convert_to_tensor(
        frequency_weights, name="frequency_weights")

    # Unlike moments(), this just uses a simpler two-pass method.

    # See comment in moments() WRT precision; it applies here too.
    needs_cast = x.dtype == dtypes.float16
    if needs_cast:
      x = math_ops.cast(x, dtypes.float32)

    if frequency_weights.dtype != x.dtype:
      frequency_weights = math_ops.cast(frequency_weights, x.dtype)

    # Note that we use keep_dims=True for our reductions regardless of the arg;
    # this is so that the results remain broadcast-compatible with the inputs.
    weighted_input_sum = math_ops.reduce_sum(
        frequency_weights * x, axes, name="weighted_input_sum", keepdims=True)

    # The shape of the weights isn't necessarily the same as x's
    # shape, just broadcast-compatible with it -- so this expression
    # performs broadcasting to give a per-item weight, with the same
    # shape as (frequency_weights * x). This avoids having to reason
    # through all the broadcast logic to compute a correct
    # sum_of_weights.
    broadcasted_weights = frequency_weights + array_ops.zeros_like(x)

    sum_of_weights = math_ops.reduce_sum(
        broadcasted_weights, axes, name="sum_of_weights", keepdims=True)

    divisor = math_ops.reciprocal(sum_of_weights, name="inv_weight_sum")

    weighted_mean = math_ops.multiply(weighted_input_sum, divisor)

    # Have the weighted mean; now on to variance:
    weighted_distsq = math_ops.reduce_sum(
        frequency_weights * math_ops.squared_difference(x, weighted_mean),
        axes,
        name="weighted_distsq",
        keepdims=True)

    weighted_variance = math_ops.multiply(weighted_distsq, divisor)

    if not keep_dims:
      weighted_mean = array_ops.squeeze(weighted_mean, axis=axes)
      weighted_variance = array_ops.squeeze(
          weighted_variance, axis=axes)

    if needs_cast:
      weighted_mean = math_ops.cast(weighted_mean, dtypes.float16)
      weighted_variance = math_ops.cast(weighted_variance, dtypes.float16)

    return weighted_mean, weighted_variance


@tf_export("nn.weighted_moments", v1=[])
def weighted_moments_v2(x, axes, frequency_weights, keepdims=False, name=None):
  """Returns the frequency-weighted mean and variance of `x`.

  Args:
    x: A tensor.
    axes: 1-d tensor of int32 values; these are the axes along which
      to compute mean and variance.
    frequency_weights: A tensor of positive weights which can be
      broadcast with x.
    keepdims: Produce moments with the same dimensionality as the input.
    name: Name used to scope the operation.

  Returns:
    Two tensors: `weighted_mean` and `weighted_variance`.
  """
  return weighted_moments(
      x=x,
      axes=axes,
      frequency_weights=frequency_weights,
      name=name,
      keep_dims=keepdims)


@tf_export("nn.batch_normalization")
def batch_normalization(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        name=None):
  r"""Batch normalization.

  Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
  `scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):

  \\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)

  `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
  shapes:

    * In all generality, they can have the same number of dimensions as the
      input `x`, with identical sizes as `x` for the dimensions that are not
      normalized over (the 'depth' dimension(s)), and dimension 1 for the
      others which are being normalized over.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keepdims=True)` during training, or running averages
      thereof during inference.
    * In the common case where the 'depth' dimension is the last dimension in
      the input tensor `x`, they may be one dimensional tensors of the same
      size as the 'depth' dimension.
      This is the case for example for the common `[batch, depth]` layout of
      fully-connected layers, and `[batch, height, width, depth]` for
      convolutions.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keepdims=False)` during training, or running averages
      thereof during inference.

  See equation 11 in Algorithm 2 of source: 
  [Batch Normalization: Accelerating Deep Network Training by
  Reducing Internal Covariate Shift; S. Ioffe, C. Szegedy]
  (http://arxiv.org/abs/1502.03167).

  Args:
    x: Input `Tensor` of arbitrary dimensionality.
    mean: A mean `Tensor`.
    variance: A variance `Tensor`.
    offset: An offset `Tensor`, often denoted \\(\beta\\) in equations, or
      None. If present, will be added to the normalized tensor.
    scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
      `None`. If present, the scale is applied to the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    name: A name for this operation (optional).

  Returns:
    the normalized, scaled, offset tensor.

  References:
    Batch Normalization - Accelerating Deep Network Training by Reducing
    Internal Covariate Shift:
      [Ioffe et al., 2015](http://arxiv.org/abs/1502.03167)
      ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
  """
  with ops.name_scope(name, "batchnorm", [x, mean, variance, scale, offset]):
    inv = math_ops.rsqrt(variance + variance_epsilon)
    if scale is not None:
      inv *= scale
    # Note: tensorflow/contrib/quantize/python/fold_batch_norms.py depends on
    # the precise order of ops that are generated by the expression below.
    return x * math_ops.cast(inv, x.dtype) + math_ops.cast(
        offset - mean * inv if offset is not None else -mean * inv, x.dtype)


@tf_export(v1=["nn.fused_batch_norm"])
def fused_batch_norm(
    x,
    scale,
    offset,  # pylint: disable=invalid-name
    mean=None,
    variance=None,
    epsilon=0.001,
    data_format="NHWC",
    is_training=True,
    name=None,
    exponential_avg_factor=1.0):
  r"""Batch normalization.


  See Source: [Batch Normalization: Accelerating Deep Network Training by
  Reducing Internal Covariate Shift; S. Ioffe, C. Szegedy]
  (http://arxiv.org/abs/1502.03167).

  Args:
    x: Input `Tensor` of 4 dimensions.
    scale: A `Tensor` of 1 dimension for scaling.
    offset: A `Tensor` of 1 dimension for bias.
    mean: A `Tensor` of 1 dimension for population mean. The shape and meaning
          of this argument depends on the value of is_training and
          exponential_avg_factor as follows:
          is_training==False (inference):
            Mean must be a `Tensor` of the same shape as scale containing the
            estimated population mean computed during training.
          is_training==True and exponential_avg_factor == 1.0:
            Mean must be None.
          is_training==True and exponential_avg_factor != 1.0:
            Mean must be a `Tensor` of the same shape as scale containing the
            exponential running mean.
    variance: A `Tensor` of 1 dimension for population variance. The shape and
          meaning of this argument depends on the value of is_training and
          exponential_avg_factor as follows:
          is_training==False (inference):
            Variance must be a `Tensor` of the same shape as scale containing
            the estimated population variance computed during training.
          is_training==True and exponential_avg_factor == 1.0:
            Variance must be None.
          is_training==True and exponential_avg_factor != 1.0:
            Variance must be a `Tensor` of the same shape as scale containing
            the exponential running variance.
    epsilon: A small float number added to the variance of x.
    data_format: The data format for x. Either "NHWC" (default) or "NCHW".
    is_training: A bool value to specify if the operation is used for
                 training or inference.
    name: A name for this operation (optional).
    exponential_avg_factor: A float number (usually between 0 and 1) used
                            for controlling the decay of the running
                            population average of mean and variance.
                            If set to 1.0, the current batch average is
                            returned.

  Returns:
    y: A 4D Tensor for the normalized, scaled, offsetted x.
    running_mean: A 1D Tensor for the exponential running mean of x.
                  The output value is (1 - exponential_avg_factor) * mean +
                  exponential_avg_factor * batch_mean), where batch_mean
                  is the mean of the current batch in x.
    running_var: A 1D Tensor for the exponential running variance
                 The output value is (1 - exponential_avg_factor) * variance +
                 exponential_avg_factor * batch_variance), where batch_variance
                 is the variance of the current batch in x.

  References:
    Batch Normalization - Accelerating Deep Network Training by Reducing
    Internal Covariate Shift:
      [Ioffe et al., 2015](http://proceedings.mlr.press/v37/ioffe15.html)
      ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
  """
  if is_training and exponential_avg_factor == 1.0:
    if (mean is not None) or (variance is not None):
      raise ValueError("Both 'mean' and 'variance' must be None when "
                       "is_training is True and "
                       "exponential_avg_factor == 1.0.")
  else:
    if (mean is None) or (variance is None):
      raise ValueError("Both 'mean' and 'variance' must be a 1D tensor when "
                       "is_training is False or "
                       "exponential_avg_factor != 1.0.")
  x = ops.convert_to_tensor(x, name="input")
  scale = ops.convert_to_tensor(scale, name="scale")
  offset = ops.convert_to_tensor(offset, name="offset")
  if mean is None:
    mean = constant_op.constant([])
  if variance is None:
    variance = constant_op.constant([])

  # Set a minimum epsilon to 1.001e-5, which is a requirement by CUDNN to
  # prevent exception (see cudnn.h).
  min_epsilon = 1.001e-5
  epsilon = epsilon if epsilon > min_epsilon else min_epsilon

  if compat.forward_compatible(2020, 3, 6):
    y, running_mean, running_var, _, _, _ = gen_nn_ops.fused_batch_norm_v3(
        x,
        scale,
        offset,
        mean,
        variance,
        epsilon=epsilon,
        exponential_avg_factor=exponential_avg_factor,
        data_format=data_format,
        is_training=is_training,
        name=name)
    return y, running_mean, running_var
  else:
    y, running_mean, running_var, _, _, _ = gen_nn_ops.fused_batch_norm_v3(
        x,
        scale,
        offset,
        mean,
        variance,
        epsilon=epsilon,
        data_format=data_format,
        is_training=is_training,
        name=name)
    return y, running_mean, running_var


@tf_export(v1=["nn.batch_norm_with_global_normalization"])
def batch_norm_with_global_normalization(t=None,
                                         m=None,
                                         v=None,
                                         beta=None,
                                         gamma=None,
                                         variance_epsilon=None,
                                         scale_after_normalization=None,
                                         name=None,
                                         input=None,  # pylint: disable=redefined-builtin
                                         mean=None,
                                         variance=None):
  """Batch normalization.

  This op is deprecated. See `tf.nn.batch_normalization`.

  Args:
    t: A 4D input Tensor.
    m: A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    scale_after_normalization: A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for this operation (optional).
    input: Alias for t.
    mean: Alias for m.
    variance: Alias for v.

  Returns:
     A batch-normalized `t`.

  References:
    Batch Normalization - Accelerating Deep Network Training by Reducing
    Internal Covariate Shift:
      [Ioffe et al., 2015](http://proceedings.mlr.press/v37/ioffe15.html)
      ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
  """
  t = deprecated_argument_lookup("input", input, "t", t)
  m = deprecated_argument_lookup("mean", mean, "m", m)
  v = deprecated_argument_lookup("variance", variance, "v", v)
  return batch_normalization(t, m, v, beta, gamma if scale_after_normalization
                             else None, variance_epsilon, name)


# pylint: disable=redefined-builtin,line-too-long
@tf_export("nn.batch_norm_with_global_normalization", v1=[])
def batch_norm_with_global_normalization_v2(input,
                                            mean,
                                            variance,
                                            beta,
                                            gamma,
                                            variance_epsilon,
                                            scale_after_normalization,
                                            name=None):
  """Batch normalization.

  This op is deprecated. See `tf.nn.batch_normalization`.

  Args:
    input: A 4D input Tensor.
    mean: A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    variance: A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    scale_after_normalization: A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for this operation (optional).

  Returns:
     A batch-normalized `t`.

  References:
    Batch Normalization - Accelerating Deep Network Training by Reducing Internal Covariate Shift:
      [Ioffe et al., 2015](http://proceedings.mlr.press/v37/ioffe15.html)
      ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
  """
  return batch_norm_with_global_normalization(t=input,
                                              m=mean,
                                              v=variance,
                                              beta=beta,
                                              gamma=gamma,
                                              variance_epsilon=variance_epsilon,
                                              scale_after_normalization=scale_after_normalization,
                                              name=name)

# pylint: enable=redefined-builtin,line-too-long


def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  cols = array_ops.shape(x)[1]
  ones_shape = array_ops.stack([cols, 1])
  ones = array_ops.ones(ones_shape, x.dtype)
  return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None,
                            seed=None):
  """Helper function for nce_loss and sampled_softmax_loss functions.

  Computes sampled output training logits and labels suitable for implementing
  e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
  sampled_softmax_loss).

  Note: In the case where num_true > 1, we assign to each target class
  the target probability 1 / num_true so that the target probabilities
  sum to 1 per-example.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
        class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    subtract_log_q: A `bool`.  whether to subtract the log expected count of
        the labels in the sample to get the logits of the true labels.
        Default is True.  Turn off for Negative Sampling.
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).
    seed: random seed for candidate sampling. Default to None, which doesn't set
        the op-level random seed for candidate sampling.
  Returns:
    out_logits: `Tensor` object with shape
        `[batch_size, num_true + num_sampled]`, for passing to either
        `nn.sigmoid_cross_entropy_with_logits` (NCE) or
        `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    out_labels: A Tensor object with the same shape as `out_logits`.
  """

  if isinstance(weights, variables.PartitionedVariable):
    weights = list(weights)
  if not isinstance(weights, list):
    weights = [weights]

  with ops.name_scope(name, "compute_sampled_logits",
                      weights + [biases, inputs, labels]):
    if labels.dtype != dtypes.int64:
      labels = math_ops.cast(labels, dtypes.int64)
    labels_flat = array_ops.reshape(labels, [-1])

    # Sample the negative labels.
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor
    if sampled_values is None:
      sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes,
          seed=seed)
    # NOTE: pylint cannot tell that 'sampled_values' is a sequence
    # pylint: disable=unpacking-non-sequence
    sampled, true_expected_count, sampled_expected_count = (
        array_ops.stop_gradient(s) for s in sampled_values)
    # pylint: enable=unpacking-non-sequence
    sampled = math_ops.cast(sampled, dtypes.int64)

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    all_ids = array_ops.concat([labels_flat, sampled], 0)

    # Retrieve the true weights and the logits of the sampled weights.

    # weights shape is [num_classes, dim]
    all_w = embedding_ops.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)
    if all_w.dtype != inputs.dtype:
      all_w = math_ops.cast(all_w, inputs.dtype)

    # true_w shape is [batch_size * num_true, dim]
    true_w = array_ops.slice(all_w, [0, 0],
                             array_ops.stack(
                                 [array_ops.shape(labels_flat)[0], -1]))

    sampled_w = array_ops.slice(
        all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # Apply X*W', which yields [batch_size, num_sampled]
    sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

    # Retrieve the true and sampled biases, compute the true logits, and
    # add the biases to the true and sampled logits.
    all_b = embedding_ops.embedding_lookup(
        biases, all_ids, partition_strategy=partition_strategy)
    if all_b.dtype != inputs.dtype:
      all_b = math_ops.cast(all_b, inputs.dtype)
    # true_b is a [batch_size * num_true] tensor
    # sampled_b is a [num_sampled] float tensor
    true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
    sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
    row_wise_dots = math_ops.multiply(
        array_ops.expand_dims(inputs, 1),
        array_ops.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = array_ops.reshape(row_wise_dots,
                                       array_ops.concat([[-1], dim], 0))
    true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    true_b = array_ops.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_logits += sampled_b

    if remove_accidental_hits:
      acc_hits = candidate_sampling_ops.compute_accidental_hits(
          labels, sampled, num_true=num_true)
      acc_indices, acc_ids, acc_weights = acc_hits

      # This is how SparseToDense expects the indices.
      acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
      acc_ids_2d_int32 = array_ops.reshape(
          math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
      sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                        "sparse_indices")
      # Create sampled_logits_shape = [batch_size, num_sampled]
      sampled_logits_shape = array_ops.concat(
          [array_ops.shape(labels)[:1],
           array_ops.expand_dims(num_sampled, 0)], 0)
      if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
      sampled_logits += gen_sparse_ops.sparse_to_dense(
          sparse_indices,
          sampled_logits_shape,
          acc_weights,
          default_value=0.0,
          validate_indices=False)

    if subtract_log_q:
      # Subtract log of Q(l), prior probability that l appears in sampled.
      true_logits -= math_ops.log(true_expected_count)
      sampled_logits -= math_ops.log(sampled_expected_count)

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = array_ops.concat([true_logits, sampled_logits], 1)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    out_labels = array_ops.concat([
        array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)
    ], 1)

    return out_logits, out_labels


@tf_export("nn.nce_loss", v1=[])
def nce_loss_v2(weights,
                biases,
                labels,
                inputs,
                num_sampled,
                num_classes,
                num_true=1,
                sampled_values=None,
                remove_accidental_hits=False,
                name="nce_loss"):
  """Computes and returns the noise-contrastive estimation training loss.

  See [Noise-contrastive estimation: A new estimation principle for
  unnormalized statistical
  models](http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
  Also see our [Candidate Sampling Algorithms
  Reference](https://www.tensorflow.org/extras/candidate_sampling.pdf)

  A common use case is to use this method for training, and calculate the full
  sigmoid loss for evaluation or inference as in the following example:

  ```python
  if mode == "train":
    loss = tf.nn.nce_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        ...)
  elif mode == "eval":
    logits = tf.matmul(inputs, tf.transpose(weights))
    logits = tf.nn.bias_add(logits, biases)
    labels_one_hot = tf.one_hot(labels, n_classes)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits)
    loss = tf.reduce_sum(loss, axis=1)
  ```

  Note: when doing embedding lookup on `weights` and `bias`, "div" partition
  strategy will be used. Support for other partition strategy will be added
  later.

  Note: By default this uses a log-uniform (Zipfian) distribution for sampling,
  so your labels must be sorted in order of decreasing frequency to achieve
  good results.  For more details, see
  `tf.random.log_uniform_candidate_sampler`.

  Note: In the case where `num_true` > 1, we assign to each target class
  the target probability 1 / `num_true` so that the target probabilities
  sum to 1 per-example.

  Note: It would be useful to allow a variable number of target classes per
  example.  We hope to provide this functionality in a future release.
  For now, if you have a variable number of target classes, you can pad them
  out to a constant number by either repeating them or by padding
  with an otherwise unused class.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
      objects whose concatenation along dimension 0 has shape [num_classes,
      dim].  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size, num_true]`. The
      target classes.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of
      the input network.
    num_sampled: An `int`.  The number of negative classes to randomly sample
      per batch. This single sample of negative classes is evaluated for each
      element in the batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
      `sampled_expected_count`) returned by a `*_candidate_sampler` function.
      (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  Whether to remove "accidental hits"
      where a sampled class equals one of the target classes.  If set to `True`,
      this is a "Sampled Logistic" loss instead of NCE, and we are learning to
      generate log-odds instead of log probabilities.  See our [Candidate
      Sampling Algorithms Reference]
        (https://www.tensorflow.org/extras/candidate_sampling.pdf). Default is
          False.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example NCE losses.
  """
  # TODO(yuefengz): get partition_strategy from either variables or distribution
  # strategies.
  return nce_loss(
      weights,
      biases,
      labels,
      inputs,
      num_sampled,
      num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy="div",
      name=name)


@tf_export(v1=["nn.nce_loss"])
def nce_loss(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss"):
  """Computes and returns the noise-contrastive estimation training loss.

  A common use case is to use this method for training, and calculate the full
  sigmoid loss for evaluation or inference. In this case, you must set
  `partition_strategy="div"` for the two losses to be consistent, as in the
  following example:

  ```python
  if mode == "train":
    loss = tf.nn.nce_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        ...,
        partition_strategy="div")
  elif mode == "eval":
    logits = tf.matmul(inputs, tf.transpose(weights))
    logits = tf.nn.bias_add(logits, biases)
    labels_one_hot = tf.one_hot(labels, n_classes)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits)
    loss = tf.reduce_sum(loss, axis=1)
  ```

  Note: By default this uses a log-uniform (Zipfian) distribution for sampling,
  so your labels must be sorted in order of decreasing frequency to achieve
  good results.  For more details, see
  `tf.random.log_uniform_candidate_sampler`.

  Note: In the case where `num_true` > 1, we assign to each target class
  the target probability 1 / `num_true` so that the target probabilities
  sum to 1 per-example.

  Note: It would be useful to allow a variable number of target classes per
  example.  We hope to provide this functionality in a future release.
  For now, if you have a variable number of target classes, you can pad them
  out to a constant number by either repeating them or by padding
  with an otherwise unused class.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of negative classes to randomly sample
        per batch. This single sample of negative classes is evaluated for each
        element in the batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  Whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  If set to
        `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
        learning to generate log-odds instead of log probabilities. See
        our Candidate Sampling Algorithms Reference
        ([pdf](https://www.tensorflow.org/extras/candidate_sampling.pdf)).
        Default is False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example NCE losses.

  References:
    Noise-contrastive estimation - A new estimation principle for unnormalized
    statistical models:
      [Gutmann et al., 2010](http://proceedings.mlr.press/v9/gutmann10a)
      ([pdf](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf))
  """
  logits, labels = _compute_sampled_logits(
      weights=weights,
      biases=biases,
      labels=labels,
      inputs=inputs,
      num_sampled=num_sampled,
      num_classes=num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      subtract_log_q=True,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy=partition_strategy,
      name=name)
  sampled_losses = sigmoid_cross_entropy_with_logits(
      labels=labels, logits=logits, name="sampled_losses")
  # sampled_losses is batch_size x {true_loss, sampled_losses...}
  # We sum out true and sampled losses.
  return _sum_rows(sampled_losses)


@tf_export("nn.sampled_softmax_loss", v1=[])
def sampled_softmax_loss_v2(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            remove_accidental_hits=True,
                            seed=None,
                            name="sampled_softmax_loss"):
  """Computes and returns the sampled softmax training loss.

  This is a faster way to train a softmax classifier over a huge number of
  classes.

  This operation is for training only.  It is generally an underestimate of
  the full softmax loss.

  A common use case is to use this method for training, and calculate the full
  sigmoid loss for evaluation or inference as in the following example:

  ```python
  if mode == "train":
    loss = tf.nn.sampled_softmax_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        ...)
  elif mode == "eval":
    logits = tf.matmul(inputs, tf.transpose(weights))
    logits = tf.nn.bias_add(logits, biases)
    labels_one_hot = tf.one_hot(labels, n_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits)
  ```

  See our [Candidate Sampling Algorithms Reference]
  (https://www.tensorflow.org/extras/candidate_sampling.pdf)

  Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
  ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

  Note: when doing embedding lookup on `weights` and `bias`, "div" partition
  strategy will be used. Support for other partition strategy will be added
  later.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
      objects whose concatenation along dimension 0 has shape [num_classes,
      dim].  The (possibly-sharded) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size, num_true]`. The
      target classes.  Note that this format differs from the `labels` argument
      of `nn.softmax_cross_entropy_with_logits`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of
      the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
      `sampled_expected_count`) returned by a `*_candidate_sampler` function.
      (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
      where a sampled class equals one of the target classes.  Default is True.
    seed: random seed for candidate sampling. Default to None, which doesn't set
      the op-level random seed for candidate sampling.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example sampled softmax losses.

  """
  return sampled_softmax_loss(
      weights,
      biases,
      labels,
      inputs,
      num_sampled,
      num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy="div",
      name=name,
      seed=seed)


@tf_export(v1=["nn.sampled_softmax_loss"])
def sampled_softmax_loss(weights,
                         biases,
                         labels,
                         inputs,
                         num_sampled,
                         num_classes,
                         num_true=1,
                         sampled_values=None,
                         remove_accidental_hits=True,
                         partition_strategy="mod",
                         name="sampled_softmax_loss",
                         seed=None):
  """Computes and returns the sampled softmax training loss.

  This is a faster way to train a softmax classifier over a huge number of
  classes.

  This operation is for training only.  It is generally an underestimate of
  the full softmax loss.

  A common use case is to use this method for training, and calculate the full
  softmax loss for evaluation or inference. In this case, you must set
  `partition_strategy="div"` for the two losses to be consistent, as in the
  following example:

  ```python
  if mode == "train":
    loss = tf.nn.sampled_softmax_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        ...,
        partition_strategy="div")
  elif mode == "eval":
    logits = tf.matmul(inputs, tf.transpose(weights))
    logits = tf.nn.bias_add(logits, biases)
    labels_one_hot = tf.one_hot(labels, n_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits)
  ```

  See our Candidate Sampling Algorithms Reference
  ([pdf](https://www.tensorflow.org/extras/candidate_sampling.pdf)).
  Also see Section 3 of (Jean et al., 2014) for the math.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-sharded) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        True.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).
    seed: random seed for candidate sampling. Default to None, which doesn't set
        the op-level random seed for candidate sampling.

  Returns:
    A `batch_size` 1-D tensor of per-example sampled softmax losses.

  References:
    On Using Very Large Target Vocabulary for Neural Machine Translation:
      [Jean et al., 2014]
      (https://aclanthology.coli.uni-saarland.de/papers/P15-1001/p15-1001)
      ([pdf](http://aclweb.org/anthology/P15-1001))
  """
  logits, labels = _compute_sampled_logits(
      weights=weights,
      biases=biases,
      labels=labels,
      inputs=inputs,
      num_sampled=num_sampled,
      num_classes=num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      subtract_log_q=True,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy=partition_strategy,
      name=name,
      seed=seed)
  labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")
  sampled_losses = nn_ops.softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=logits)
  # sampled_losses is a [batch_size] tensor.
  return sampled_losses
