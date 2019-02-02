# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=unidiomatic-typecheck
"""Utility to lift subgraphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def _graph_inputs(op):
  return [x.op for x in op.inputs] + list(op.control_inputs)


def _as_operation(op_or_tensor):
  if isinstance(op_or_tensor, ops.Tensor):
    return op_or_tensor.op
  return op_or_tensor


class UnliftableError(Exception):
  """Raised if a Tensor cannot be lifted from the graph."""
  pass


def lift_to_graph(init_tensor, graph, sources=None):
  """Copies the tensor and all its inputs recursively to the outer graph."""
  # Check that the initializer does not depend on any placeholders.
  if sources is None:
    sources = set([])
  visited_ops = set([x.op for x in sources])
  ops_to_visit = [_as_operation(init_tensor)]
  op_outputs = collections.defaultdict(set)
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in visited_ops:
      continue
    visited_ops.add(op)
    # TODO(apassos) distinguish arg placeholders, capture placeholders,
    # and placeholders the user might directly use to initialize
    # variables.
    if op.type == "Placeholder":
      raise UnliftableError(
          "Unable to lift tensor", init_tensor,
          "because it depends transitively on placeholder ", op)
    for inp in _graph_inputs(op):
      op_outputs[inp].add(op)
      if inp not in visited_ops and inp not in sources:
        ops_to_visit.append(inp)
  # Topologically sort the nodes we've extracted. Now we know how many of their
  # outputs are part of this subgraph.
  ops_to_copy = []
  marked_ops = set([])
  ops_to_visit = [_as_operation(init_tensor)]
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in marked_ops:
      continue
    marked_ops.add(op)
    ops_to_copy.append(op)
    for inp in _graph_inputs(op):
      if all(x in marked_ops for x in op_outputs[inp]) and inp not in sources:
        ops_to_visit.append(inp)
  # ops_to_copy now holds a reverse topologically sorted list of ops which
  # ends in the initializer. We copy those to the outermost graph and
  # build the initialization op there.
  with graph.as_default():
    op_map = {}
    source_ops = set()
    for s in sources:
      source_ops.add(s.op)
      op_map[s] = array_ops.placeholder(dtype=s.dtype, shape=s.shape)
    for op in reversed(ops_to_copy):
      if op in source_ops:
        continue
      copied_inputs = [op_map[x] for x in op.inputs]
      copied_control_inputs = [op_map[x] for x in op.control_inputs]
      with ops.control_dependencies(copied_control_inputs):
        copied_op = graph.create_op(
            op.type, copied_inputs, [x.dtype for x in op.outputs],
            attrs=op.node_def.attr)
      op_map[op] = copied_op
      for i, o in enumerate(op.outputs):
        op_map[o] = copied_op.outputs[i]
    return op_map
