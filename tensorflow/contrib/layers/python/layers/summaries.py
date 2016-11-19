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
# ==============================================================================
"""Utility functions for summary creation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re

from tensorflow.python import summary
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

__all__ = [
    'summarize_tensor',
    'summarize_activation',
    'summarize_tensors',
    'summarize_collection',
    'summarize_variables',
    'summarize_weights',
    'summarize_biases',
    'summarize_activations',
]

# TODO(wicke): add more unit tests for summarization functions.


def _add_scalar_summary(tensor, tag=None):
  """Add a scalar summary operation for the tensor.

  Args:
    tensor: The tensor to summarize.
    tag: The tag to use, if None then use tensor's op's name.

  Returns:
    The created histogram summary.

  Raises:
    ValueError: If the tag is already in use or the rank is not 0.
  """
  tensor.get_shape().assert_has_rank(0)
  tag = tag or '%s_summary' % tensor.op.name
  return summary.scalar(tag, tensor)


def _add_histogram_summary(tensor, tag=None):
  """Add a summary operation for the histogram of a tensor.

  Args:
    tensor: The tensor to summarize.
    tag: The tag to use, if None then use tensor's op's name.

  Returns:
    The created histogram summary.

  Raises:
    ValueError: If the tag is already in use.
  """
  tag = tag or '%s_summary' % tensor.op.name
  return summary.histogram(tag, tensor)


def summarize_activation(op):
  """Summarize an activation.

  This applies the given activation and adds useful summaries specific to the
  activation.

  Args:
    op: The tensor to summarize (assumed to be a layer activation).
  Returns:
    The summary op created to summarize `op`.
  """
  if op.op.type in ('Relu', 'Softplus', 'Relu6'):
    # Using inputs to avoid floating point equality and/or epsilons.
    _add_scalar_summary(
        standard_ops.reduce_mean(standard_ops.to_float(standard_ops.less(
            op.op.inputs[0], standard_ops.cast(0.0, op.op.inputs[0].dtype)))),
        '%s/zeros' % op.op.name)
  if op.op.type == 'Relu6':
    _add_scalar_summary(
        standard_ops.reduce_mean(standard_ops.to_float(standard_ops.greater(
            op.op.inputs[0], standard_ops.cast(6.0, op.op.inputs[0].dtype)))),
        '%s/sixes' % op.op.name)
  return _add_histogram_summary(op, '%s/activation' % op.op.name)


def summarize_tensor(tensor, tag=None):
  """Summarize a tensor using a suitable summary type.

  This function adds a summary op for `tensor`. The type of summary depends on
  the shape of `tensor`. For scalars, a `scalar_summary` is created, for all
  other tensors, `histogram_summary` is used.

  Args:
    tensor: The tensor to summarize
    tag: The tag to use, if None then use tensor's op's name.

  Returns:
    The summary op created or None for string tensors.
  """
  # Skips string tensors and boolean tensors (not handled by the summaries).
  if (tensor.dtype.is_compatible_with(dtypes.string) or
      tensor.dtype.base_dtype == dtypes.bool):
    return None

  if tensor.get_shape().ndims == 0:
    # For scalars, use a scalar summary.
    return _add_scalar_summary(tensor, tag)
  else:
    # We may land in here if the rank is still unknown. The histogram won't
    # hurt if this ends up being a scalar.
    return _add_histogram_summary(tensor, tag)


def summarize_tensors(tensors, summarizer=summarize_tensor):
  """Summarize a set of tensors."""
  return [summarizer(tensor) for tensor in tensors]


def summarize_collection(collection, name_filter=None,
                         summarizer=summarize_tensor):
  """Summarize a graph collection of tensors, possibly filtered by name."""
  tensors = []
  for op in ops.get_collection(collection):
    if name_filter is None or re.match(name_filter, op.op.name):
      tensors.append(op)
  return summarize_tensors(tensors, summarizer)


# Utility functions for commonly used collections
summarize_variables = functools.partial(summarize_collection,
                                        ops.GraphKeys.GLOBAL_VARIABLES)


summarize_weights = functools.partial(summarize_collection,
                                      ops.GraphKeys.WEIGHTS)


summarize_biases = functools.partial(summarize_collection,
                                     ops.GraphKeys.BIASES)


def summarize_activations(name_filter=None, summarizer=summarize_activation):
  """Summarize activations, using `summarize_activation` to summarize."""
  return summarize_collection(ops.GraphKeys.ACTIVATIONS, name_filter,
                              summarizer)
