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

from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity


UnliftableError = op_selector.UnliftableError


def _as_operation(op_or_tensor):
  if isinstance(op_or_tensor, ops.Tensor):
    return op_or_tensor.op
  return op_or_tensor


def _constant_inputs(op_or_tensor):
  return all(_as_operation(i).type == u"Const"
             and not _as_operation(i).control_inputs
             for i in op_selector.graph_inputs(_as_operation(op_or_tensor)))


# Represents an input to `copied_op` which must be updated once
# `old_graph_tensor` has been copied.
_InputMutation = collections.namedtuple(
    "_InputMutation",
    ["copied_op", "input_index", "old_graph_tensor"])


# Represents a control input to `copied_op` which must be added once
# `old_graph_op` has been copied.
_ControlMutation = collections.namedtuple(
    "_ControlMutation",
    ["copied_op", "old_graph_op"])


def _copy_non_source(op, graph, op_map, base_graph):
  """Copy an op directly to a given graph.

  Generally `op`'s inputs should already have been copied. If this is not the
  case, for example with v1 while_loops, then `_copy_non_source` inserts
  placeholders for the unavailable Tensors and returns a list of required
  mutations.

  Args:
    op: The op to be copied.
    graph: The destination graph.
    op_map: A dict mapping ops and tensors in the old graph to the new one.
    base_graph: The graph we're copying from, for any necessary functions.
  Returns:
    A tuple of (required_inputs, required_control_inputs):
      required_inputs:
        A list of `_InputMutation` tuples containing inputs to `copied_op` which
        must be updated once `old_graph_tensor` has been copied.
      required_control_inputs:
        A list of `_ControlMutation` tuples containing control inputs to
        `copied_op` which must be added once `old_graph_op` has been copied.
  """
  input_mutations = []
  control_mutations = []
  copied_inputs = []
  for input_index, original_input in enumerate(op.inputs):
    copied_input = op_map.get(original_input, None)
    if copied_input is None:
      # An input for this op is missing due to a loop in the graph. We'll insert
      # a placeholder for now and return information about the required post-hoc
      # mutation.
      copied_input = array_ops.placeholder(
          name="unused_control_flow_input",
          shape=original_input.shape,
          dtype=original_input.dtype)
      input_mutations.append(
          # `copied_op` is filled in below, after we've created it.
          _InputMutation(copied_op=None,
                         input_index=input_index,
                         old_graph_tensor=original_input))
    copied_inputs.append(copied_input)

  copied_control_inputs = []
  for original_control_input in op.control_inputs:
    copied_control_input = op_map.get(original_control_input, None)
    if copied_control_input is None:
      control_mutations.append(
          _ControlMutation(copied_op=None,
                           old_graph_op=original_control_input))
    else:
      copied_control_inputs.append(copied_control_input)

  # Don't copy over nodes with _tpu_replicate attribute. This attributed is used
  # to signal that the op was built inside a tpu_replicate context; if we're
  # lifting it to another graph we're similarly lifting it into another context.
  with ops.control_dependencies(copied_control_inputs), ops.device(op.device):
    # pylint: disable=protected-access
    f = base_graph._functions.get(op.type, None)
    if f is not None and compat.as_str(f.name) not in graph._functions:
      f.add_to_graph(graph)
    # pylint: enable=protected-access

    # Create a new op in the destination graph if it doesn't exist before.
    copied_op = graph.create_op(
        op_type=op.type,
        inputs=copied_inputs,
        dtypes=[x.dtype for x in op.outputs],
        attrs={
            key: value for key, value in op.node_def.attr.items()
            if not key.startswith("_class") and
            not key.startswith("_tpu_replicate")
        },  # b/128981532.
        name=op.name)
  op_map[op] = copied_op
  for i, o in enumerate(op.outputs):
    op_map[o] = copied_op.outputs[i]

  return ([mutation._replace(copied_op=copied_op)
           for mutation in input_mutations],
          [mutation._replace(copied_op=copied_op)
           for mutation in control_mutations])


def _copy_source(s, graph, op_map, handle_captures, inverse_captures,
                 base_graph):
  """Create a source in a graph based on a Tensor from a different graph.

  This function creates a placeholder analog of `s` in a graph with the
  following behavior:

  1) If s is a captured Tensor or Variable and handle_captures is set to True,
     simply capture it in the new graph as well.

  2) If s is a PlaceholderWithDefault whose default is a constant, preserve
     said default in the new graph.

  3) When applicable, copy resource variable metadata from `s` to the newly
     created placeholder.

  Args:
    s: The source of interest.
    graph: The destination graph.
    op_map: A dict mapping ops and tensors in the old graph to the new one.
    handle_captures: A boolean indicating whether to re-capture s in the new
      graph or simply create a vanilla placeholder.
    inverse_captures: A dict mapping s back to the Tensor or Variable that it
      captures.
    base_graph: The graph being copied from.
  """
  if handle_captures and s in inverse_captures:
    copied_placeholder = graph.capture(inverse_captures[s], name=s.op.name)
  elif s.op.type == "PlaceholderWithDefault" and _constant_inputs(s):
    # Copy the default value to the graph.
    default_value = s.op.inputs[0]
    unavailable_inputs, unavailable_control_inputs = _copy_non_source(
        op=default_value.op, graph=graph, op_map=op_map,
        base_graph=base_graph)
    if unavailable_inputs or unavailable_control_inputs:
      raise AssertionError(
          "Could not copy source node {} because it has inputs."
          .format(default_value))

    with ops.device(s.op.device):
      copied_placeholder = array_ops.placeholder_with_default(
          input=op_map[default_value], shape=s.shape, name=s.op.name)
  else:
    with ops.device(s.op.device):
      copied_placeholder = array_ops.placeholder(
          dtype=s.dtype, shape=s.shape, name=s.op.name)

  base_handle = resource_variable_ops.get_resource_handle_data(s)
  if base_handle.shape_and_type:
    resource_variable_ops._set_handle_shapes_and_types(  # pylint: disable=protected-access
        copied_placeholder,
        base_handle,
        graph_mode=True)

  op_map[s] = copied_placeholder
  # Add an entry for the op of the source tensor so that if there are any nodes
  # depending on that op via control dependencies it can work correctly.
  op_map[s.op] = copied_placeholder.op


def lift_to_graph(tensors,
                  graph,
                  sources=None,
                  disallowed_placeholders=None,
                  add_sources=False,
                  handle_captures=False,
                  base_graph=None,
                  op_map=None):
  """Copies the tensor and all its inputs recursively to the outer graph.

  Args:
    tensors: The Tensors to lift.
    graph: The graph to lift to.
    sources: Optional sequence of nodes to start from. If omitted the whole
      subgraph which feeds into `init_tensor` is lifted.
    disallowed_placeholders: An optional set of ops which may not appear in the
      lifted graph. Defaults to all placeholders.
    add_sources: A boolean indicating whether placeholders which are not in
      sources should be allowed.
    handle_captures: A boolean indicating whether to re-capture s in the new
      graph or simply create a vanilla placeholder.
    base_graph: The graph from which to lift ops. This will be inferred if not
      specified.
    op_map: A map contains all the existing nodes that have been lifted to the
      destination graph, so they won't be lifted and copied again.

  Returns:
    A mapping from ops in the current default graph to ops in `graph`.

  Raises:
    UnliftableError: If a placeholder blocks lifting.
  """
  variable_init_tensors = []
  init_tensors = []
  for tensor in tensors:
    if isinstance(tensor, resource_variable_ops.ResourceVariable):
      variable_init_tensors.append(tensor)
    else:
      init_tensors.append(tensor)
  base_graph = base_graph or init_tensors[0].graph
  op_map = op_map or object_identity.ObjectIdentityDictionary()

  # Check that the initializer does not depend on any placeholders.
  sources = object_identity.ObjectIdentitySet(sources or [])
  visited_ops = set(x.op for x in sources)
  op_outputs = collections.defaultdict(set)

  # First we extract the subgraph between init_tensors and sources.
  for init_tensor in init_tensors:
    sources.update(op_selector.map_subgraph(
        init_tensor=init_tensor,
        sources=sources,
        disallowed_placeholders=disallowed_placeholders,
        visited_ops=visited_ops,
        op_outputs=op_outputs,
        add_sources=add_sources))

  # Try to topologically sort the nodes we've extracted. Now we know how many of
  # their outputs are part of this subgraph.
  ops_to_copy = []
  marked_ops = set([])
  ops_to_visit = [_as_operation(t) for t in init_tensors
                  if not op_outputs[_as_operation(t)]]
  unvisited_ops = set(ops_to_visit)
  while unvisited_ops:
    while ops_to_visit:
      op = ops_to_visit.pop()
      if op in marked_ops:
        continue
      marked_ops.add(op)
      ops_to_copy.append(op)
      for inp in op_selector.graph_inputs(op):
        # Don't lift the TPUReplicateMetadata nodes out of the function, because
        # it has no registered kernels.
        if inp.type == "TPUReplicateMetadata":
          continue
        unvisited_ops.add(inp)
        if (all(x in marked_ops for x in op_outputs[inp]) and
            inp not in sources):
          ops_to_visit.append(inp)
    unvisited_ops.difference_update(marked_ops)
    if unvisited_ops:
      # `unvisited_ops` should only have elements if the graph has a loop. In
      # this case we want to keep copying and there's no topological ordering;
      # we'll do ugly post-hoc mutations instead.
      ops_to_visit.append(next(iter(unvisited_ops)))

  # When lifting from one FuncGraph to another, we will need to capture the
  # relevant tensors as well.
  captures = []
  inverse_captures = object_identity.ObjectIdentityDictionary()
  internal_captures = []
  if (isinstance(base_graph, func_graph.FuncGraph) and
      isinstance(graph, func_graph.FuncGraph)):
    captures = base_graph.captures
    for external_capture, internal_capture in captures:
      inverse_captures[internal_capture] = external_capture
    internal_captures = base_graph.internal_captures

  # ops_to_copy now holds a reverse topologically sorted list of ops which
  # ends in the initializer. We copy those to the outermost graph and
  # build the initialization op there.
  with graph.as_default():
    for i in variable_init_tensors:
      op_map[i] = i
    source_ops = set()
    # Add the sources in the same order as the original graph.
    for s in internal_captures:
      if s in sources:
        sources.remove(s)
        source_ops.add(s.op)
        _copy_source(
            s=s,
            graph=graph,
            op_map=op_map,
            handle_captures=handle_captures,
            inverse_captures=inverse_captures,
            base_graph=base_graph)
    for s in sources:
      source_ops.add(s.op)
      _copy_source(
          s=s,
          graph=graph,
          op_map=op_map,
          handle_captures=handle_captures,
          inverse_captures=inverse_captures,
          base_graph=base_graph)

    input_mutations = []
    control_mutations = []
    for op in reversed(ops_to_copy):
      if op in source_ops or op in op_map:
        continue
      new_input_mutations, new_control_mutations = _copy_non_source(
          op=op, graph=graph, op_map=op_map, base_graph=base_graph)
      input_mutations.extend(new_input_mutations)
      control_mutations.extend(new_control_mutations)

    # Mutate the new graph to insert any loops which existed in the source
    # graph due to v1 while_loops.
    #
    # pylint: disable=protected-access
    with graph._mutation_lock():
      for mutation in input_mutations:
        mutation.copied_op._update_input(
            mutation.input_index, op_map[mutation.old_graph_tensor])
      for mutation in control_mutations:
        # Don't lift the TPUReplicateMetadata nodes out of the function, because
        # it has no registered kernels.
        if mutation.old_graph_op.type == "TPUReplicateMetadata":
          continue
        mutation.copied_op._add_control_input(op_map[mutation.old_graph_op])
    # pylint: enable=protected-access

    return op_map
