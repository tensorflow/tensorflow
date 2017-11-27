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
"""Implementation of kernel-methods-related loss operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.losses import losses


def sparse_multiclass_hinge_loss(
    labels,
    logits,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds Ops for computing the multiclass hinge loss.

  The implementation is based on the following paper:
  On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
  by Crammer and Singer.
  link: http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf

  This is a generalization of standard (binary) hinge loss. For a given instance
  with correct label c*, the loss is given by:
    loss = max_{c != c*} logits_c - logits_{c*} + 1.
  or equivalently
    loss = max_c { logits_c - logits_{c*} + I_{c != c*} }
  where I_{c != c*} = 1 if c != c* and 0 otherwise.

  Args:
    labels: `Tensor` of shape [batch_size] or [batch_size, 1]. Corresponds to
      the ground truth. Each entry must be an index in `[0, num_classes)`.
    logits: `Tensor` of shape [batch_size, num_classes] corresponding to the
      unscaled logits. Its dtype should be either `float32` or `float64`.
    weights: Optional (python) scalar or `Tensor`. If a non-scalar `Tensor`, its
      rank should be either 1 ([batch_size]) or 2 ([batch_size, 1]).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is a scalar.

  Raises:
    ValueError: If `logits`, `labels` or `weights` have invalid or inconsistent
      shapes.
    ValueError: If `labels` tensor has invalid dtype.
  """

  with ops.name_scope(scope, 'sparse_multiclass_hinge_loss', (logits,
                                                              labels)) as scope:

    # Check logits Tensor has valid rank.
    logits_shape = logits.get_shape()
    logits_rank = logits_shape.ndims
    if logits_rank != 2:
      raise ValueError(
          'logits should have rank 2 ([batch_size, num_classes]). Given rank is'
          ' {}'.format(logits_rank))
    batch_size, num_classes = logits_shape[0].value, logits_shape[1].value
    logits = math_ops.to_float(logits)

    # Check labels have valid type.
    if labels.dtype != dtypes.int32 and labels.dtype != dtypes.int64:
      raise ValueError(
          'Invalid dtype for labels: {}. Acceptable dtypes: int32 and int64'.
          format(labels.dtype))

    # Check labels and weights have valid ranks and are consistent.
    labels_rank = labels.get_shape().ndims
    if labels_rank not in [1, 2]:
      raise ValueError(
          'labels should have rank 1 ([batch_size]) or 2 ([batch_size, 1]). '
          'Given rank is {}'.format(labels_rank))
    with ops.control_dependencies([
        check_ops.assert_less(labels, math_ops.cast(num_classes, labels.dtype))
    ]):
      labels = array_ops.reshape(labels, shape=[-1])

    weights = ops.convert_to_tensor(weights)
    weights_rank = weights.get_shape().ndims
    if weights_rank not in [0, 1, 2]:
      raise ValueError(
          'non-scalar weights should have rank 1 ([batch_size]) or 2 '
          '([batch_size, 1]). Given rank is {}'.format(labels_rank))

    if weights_rank > 0:
      weights = array_ops.reshape(weights, shape=[-1])
      # Check weights and labels have the same number of elements.
      weights.get_shape().assert_is_compatible_with(labels.get_shape())

    # Compute the logits tensor corresponding to the correct class per instance.
    example_indices = array_ops.reshape(
        math_ops.range(batch_size), shape=[batch_size, 1])
    indices = array_ops.concat(
        [
            example_indices,
            array_ops.reshape(
                math_ops.cast(labels, example_indices.dtype),
                shape=[batch_size, 1])
        ],
        axis=1)
    label_logits = array_ops.reshape(
        array_ops.gather_nd(params=logits, indices=indices),
        shape=[batch_size, 1])

    one_cold_labels = array_ops.one_hot(
        indices=labels, depth=num_classes, on_value=0.0, off_value=1.0)
    margin = logits - label_logits + one_cold_labels
    margin = nn_ops.relu(margin)
    loss = math_ops.reduce_max(margin, axis=1)
    return losses.compute_weighted_loss(
        loss, weights, scope, loss_collection, reduction=reduction)
