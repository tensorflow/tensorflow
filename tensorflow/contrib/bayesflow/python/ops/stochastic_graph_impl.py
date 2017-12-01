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
"""Classes and helper functions for Stochastic Computation Graphs.

## Stochastic Computation Graph Helper Functions

@@surrogate_loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.bayesflow.python.ops import stochastic_tensor_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging


def _upstream_stochastic_nodes(tensors):
  """Map tensors to the stochastic tensors upstream of them.

  Args:
    tensors: a list of Tensors.

  Returns:
    A dict that maps the tensors passed in to the `StochasticTensor` objects
    upstream of them.
  """
  reverse_map = _stochastic_dependencies_map(tensors)
  upstream = collections.defaultdict(set)
  for st, ts in reverse_map.items():
    for t in ts:
      upstream[t].add(st)
  return upstream


def _stochastic_dependencies_map(fixed_losses, stochastic_tensors=None):
  """Map stochastic tensors to the fixed losses that depend on them.

  Args:
    fixed_losses: a list of `Tensor`s.
    stochastic_tensors: a list of `StochasticTensor`s to map to fixed losses.
      If `None`, all `StochasticTensor`s in the graph will be used.

  Returns:
    A dict `dependencies` that maps `StochasticTensor` objects to subsets of
    `fixed_losses`.

    If `loss in dependencies[st]`, for some `loss` in `fixed_losses` then there
    is a direct path from `st.value()` to `loss` in the graph.
  """
  stoch_value_collection = stochastic_tensors or ops.get_collection(
      stochastic_tensor_impl.STOCHASTIC_TENSOR_COLLECTION)

  if not stoch_value_collection:
    return {}

  stoch_value_map = dict(
      (node.value(), node) for node in stoch_value_collection)

  # Step backwards through the graph to see which surrogate losses correspond
  # to which fixed_losses.
  #
  # TODO(ebrevdo): Ensure that fixed_losses and stochastic values are in the
  # same frame.
  stoch_dependencies_map = collections.defaultdict(set)
  for loss in fixed_losses:
    boundary = set([loss])
    while boundary:
      edge = boundary.pop()
      edge_stoch_node = stoch_value_map.get(edge, None)
      if edge_stoch_node:
        stoch_dependencies_map[edge_stoch_node].add(loss)
      boundary.update(edge.op.inputs)

  return stoch_dependencies_map


def surrogate_loss(sample_losses,
                   stochastic_tensors=None,
                   name="SurrogateLoss"):
  """Surrogate loss for stochastic graphs.

  This function will call `loss_fn` on each `StochasticTensor`
  upstream of `sample_losses`, passing the losses that it influenced.

  Note that currently `surrogate_loss` does not work with `StochasticTensor`s
  instantiated in `while_loop`s or other control structures.

  Args:
    sample_losses: a list or tuple of final losses. Each loss should be per
      example in the batch (and possibly per sample); that is, it should have
      dimensionality of 1 or greater. All losses should have the same shape.
    stochastic_tensors: a list of `StochasticTensor`s to add loss terms for.
      If None, defaults to all `StochasticTensor`s in the graph upstream of
      the `Tensor`s in `sample_losses`.
    name: the name with which to prepend created ops.

  Returns:
    `Tensor` loss, which is the sum of `sample_losses` and the
    `loss_fn`s returned by the `StochasticTensor`s.

  Raises:
    TypeError: if `sample_losses` is not a list or tuple, or if its elements
      are not `Tensor`s.
    ValueError: if any loss in `sample_losses` does not have dimensionality 1
      or greater.
  """
  with ops.name_scope(name, values=sample_losses):
    if not isinstance(sample_losses, (list, tuple)):
      raise TypeError("sample_losses must be a list or tuple")
    for loss in sample_losses:
      if not isinstance(loss, ops.Tensor):
        raise TypeError("loss is not a Tensor: %s" % loss)
      ndims = loss.get_shape().ndims
      if not (ndims is not None and ndims >= 1):
        raise ValueError("loss must have dimensionality 1 or greater: %s" %
                         loss)

    stoch_dependencies_map = _stochastic_dependencies_map(
        sample_losses, stochastic_tensors=stochastic_tensors)
    if not stoch_dependencies_map:
      logging.warn(
          "No collection of Stochastic Tensors found for current graph.")
      return math_ops.add_n(sample_losses)

    # Iterate through all of the stochastic dependencies, adding
    # surrogate terms where necessary.
    sample_losses = [ops.convert_to_tensor(loss) for loss in sample_losses]
    loss_terms = sample_losses
    for (stoch_node, dependent_losses) in stoch_dependencies_map.items():
      dependent_losses = list(dependent_losses)

      logging.info("Losses influenced by StochasticTensor %s: [%s]",
                   stoch_node.name, ", ".join(
                       [loss.name for loss in dependent_losses]))

      # Sum up the downstream losses for this ST
      influenced_loss = _add_n_or_sum(dependent_losses)

      # Compute surrogate loss term
      loss_term = stoch_node.loss(array_ops.stop_gradient(influenced_loss))
      if loss_term is not None:
        loss_terms.append(loss_term)

    return _add_n_or_sum(loss_terms)


def _add_n_or_sum(terms):
  # add_n works for Tensors of the same dtype and shape
  shape = terms[0].get_shape()
  dtype = terms[0].dtype

  if all(term.get_shape().is_fully_defined() and
         term.get_shape().is_compatible_with(shape) and term.dtype == dtype
         for term in terms):
    return math_ops.add_n(terms)
  else:
    return sum(terms)
