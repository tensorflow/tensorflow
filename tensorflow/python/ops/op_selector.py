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
"""Tools for selecting ops in a graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.util import object_identity


def is_differentiable(op):
  try:
    return ops._gradient_registry.lookup(op.op_def.name) is not None  # pylint: disable=protected-access
  except LookupError:
    return False


def is_iterable(obj):
  """Return true if the object is iterable."""
  if isinstance(obj, ops.Tensor):
    return False
  try:
    _ = iter(obj)
  except Exception:  # pylint: disable=broad-except
    return False
  return True


def concatenate_unique(la, lb):
  """Add all the elements of `lb` to `la` if they are not there already.

  The elements added to `la` maintain ordering with respect to `lb`.

  Args:
    la: List of Python objects.
    lb: List of Python objects.
  Returns:
    `la`: The list `la` with missing elements from `lb`.
  """
  la_set = set(la)
  for l in lb:
    if l not in la_set:
      la.append(l)
      la_set.add(l)
  return la


def get_tensors(graph):
  """get all the tensors which are input or output of an op in the graph.

  Args:
    graph: a `tf.Graph`.
  Returns:
    A list of `tf.Tensor`.
  Raises:
    TypeError: if graph is not a `tf.Graph`.
  """
  if not isinstance(graph, ops.Graph):
    raise TypeError("Expected a graph, got: {}".format(type(graph)))
  ts = []
  for op in graph.get_operations():
    ts += op.outputs
  return ts


def get_unique_graph(tops, check_types=None, none_if_empty=False):
  """Return the unique graph used by the all the elements in tops.

  Args:
    tops: list of elements to check (usually a list of tf.Operation and/or
      tf.Tensor). Or a tf.Graph.
    check_types: check that the element in tops are of given type(s). If None,
      the types (tf.Operation, tf.Tensor) are used.
    none_if_empty: don't raise an error if tops is an empty list, just return
      None.
  Returns:
    The unique graph used by all the tops.
  Raises:
    TypeError: if tops is not a iterable of tf.Operation.
    ValueError: if the graph is not unique.
  """
  if isinstance(tops, ops.Graph):
    return tops
  if not is_iterable(tops):
    raise TypeError("{} is not iterable".format(type(tops)))
  if check_types is None:
    check_types = (ops.Operation, ops.Tensor)
  elif not is_iterable(check_types):
    check_types = (check_types,)
  g = None
  for op in tops:
    if not isinstance(op, check_types):
      raise TypeError("Expected a type in ({}), got: {}".format(", ".join([str(
          t) for t in check_types]), type(op)))
    if g is None:
      g = op.graph
    elif g._graph_key != op.graph._graph_key:  # pylint: disable=protected-access
      raise ValueError("Operation {} does not belong to given graph".format(op))
  if g is None and not none_if_empty:
    raise ValueError("Can't find the unique graph of an empty list")
  return g


def check_graphs(*args):
  """Check that all the element in args belong to the same graph.

  Args:
    *args: a list of object with a obj.graph property.
  Raises:
    ValueError: if all the elements do not belong to the same graph.
  """
  graph = None
  for i, sgv in enumerate(args):
    if graph is None and sgv.graph is not None:
      graph = sgv.graph
    elif sgv.graph is not None and sgv.graph is not graph:
      raise ValueError(f"args[{i}] does not belong to the same graph as "
                       "other arguments.")


def make_list_of_t(ts, check_graph=True, allow_graph=True, ignore_ops=False):
  """Convert ts to a list of `tf.Tensor`.

  Args:
    ts: can be an iterable of `tf.Tensor`, a `tf.Graph` or a single tensor.
    check_graph: if `True` check if all the tensors belong to the same graph.
    allow_graph: if `False` a `tf.Graph` cannot be converted.
    ignore_ops: if `True`, silently ignore `tf.Operation`.
  Returns:
    A newly created list of `tf.Tensor`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `tf.Tensor` or,
     if `check_graph` is `True`, if all the ops do not belong to the same graph.
  """
  if isinstance(ts, ops.Graph):
    if allow_graph:
      return get_tensors(ts)
    else:
      raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  else:
    if not is_iterable(ts):
      ts = [ts]
    if not ts:
      return []
    if check_graph:
      check_types = None if ignore_ops else ops.Tensor
      get_unique_graph(ts, check_types=check_types)
    return [t for t in ts if isinstance(t, ops.Tensor)]


def get_generating_ops(ts):
  """Return all the generating ops of the tensors in `ts`.

  Args:
    ts: a list of `tf.Tensor`
  Returns:
    A list of all the generating `tf.Operation` of the tensors in `ts`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `tf.Tensor`.
  """
  ts = make_list_of_t(ts, allow_graph=False)
  return [t.op for t in ts]


def get_consuming_ops(ts):
  """Return all the consuming ops of the tensors in ts.

  Args:
    ts: a list of `tf.Tensor`
  Returns:
    A list of all the consuming `tf.Operation` of the tensors in `ts`.
  Raises:
    TypeError: if ts cannot be converted to a list of `tf.Tensor`.
  """
  ts = make_list_of_t(ts, allow_graph=False)
  tops = []
  for t in ts:
    for op in t.consumers():
      if op not in tops:
        tops.append(op)
  return tops


def make_list_of_op(tops, check_graph=True, allow_graph=True, ignore_ts=False):
  """Convert ops to a list of `tf.Operation`.

  Args:
    tops: can be an iterable of `tf.Operation`, a `tf.Graph` or a single
      operation.
    check_graph: if `True` check if all the operations belong to the same graph.
    allow_graph: if `False` a `tf.Graph` cannot be converted.
    ignore_ts: if True, silently ignore `tf.Tensor`.
  Returns:
    A newly created list of `tf.Operation`.
  Raises:
    TypeError: if tops cannot be converted to a list of `tf.Operation` or,
     if `check_graph` is `True`, if all the ops do not belong to the
     same graph.
  """
  if isinstance(tops, ops.Graph):
    if allow_graph:
      return tops.get_operations()
    else:
      raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  else:
    if not is_iterable(tops):
      tops = [tops]
    if not tops:
      return []
    if check_graph:
      check_types = None if ignore_ts else ops.Operation
      get_unique_graph(tops, check_types=check_types)
    return [op for op in tops if isinstance(op, ops.Operation)]


def _get_inputs(op, only_differentiable):
  op_inputs = op.inputs
  if only_differentiable:
    return op_inputs if is_differentiable(op) else []
  else:
    return op_inputs


def get_backward_walk_ops(seed_ops,
                          inclusive=True,
                          within_ops=None,
                          within_ops_fn=None,
                          stop_at_ts=(),
                          control_inputs=False,
                          only_differentiable=False):
  """Do a backward graph walk and return all the visited ops.

  Args:
    seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    inclusive: if True the given seed_ops are also part of the resulting set.
    within_ops: an iterable of `tf.Operation` within which the search is
      restricted. If `within_ops` is `None`, the search is performed within
      the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    stop_at_ts: an iterable of tensors at which the graph walk stops.
    control_inputs: if True, control inputs will be used while moving backward.
    only_differentiable: if True, only traverse ops which are differentiable.
      This includes natively differentiable ops, or ops with custom gradients.
  Returns:
    A Python set of all the `tf.Operation` behind `seed_ops`.
  Raises:
    TypeError: if `seed_ops` or `within_ops` cannot be converted to a list of
      `tf.Operation`.
  """
  control_inputs = control_inputs and (not only_differentiable)

  if not is_iterable(seed_ops):
    seed_ops = [seed_ops]
  if not seed_ops:
    return []
  if isinstance(seed_ops[0], ops.Tensor):
    ts = make_list_of_t(seed_ops, allow_graph=False)
    seed_ops = get_generating_ops(ts)
  else:
    seed_ops = make_list_of_op(seed_ops, allow_graph=False)

  stop_at_ts = object_identity.ObjectIdentitySet(make_list_of_t(stop_at_ts))
  seed_ops = object_identity.ObjectIdentitySet(make_list_of_op(seed_ops))
  if within_ops:
    within_ops = make_list_of_op(within_ops, allow_graph=False)
    within_ops = object_identity.ObjectIdentitySet(within_ops)
    seed_ops &= within_ops

  def is_within(op):
    return (within_ops is None or op in within_ops) and (
        within_ops_fn is None or within_ops_fn(op))

  result = list(seed_ops)
  wave = set(seed_ops)
  while wave:
    new_wave = set()
    for op in wave:
      for new_t in _get_inputs(op, only_differentiable=only_differentiable):
        if new_t in stop_at_ts:
          continue
        if new_t.op not in result and is_within(new_t.op):
          new_wave.add(new_t.op)
      if control_inputs:
        for new_op in op.control_inputs:
          if new_op not in result and is_within(new_op):
            new_wave.add(new_op)
    concatenate_unique(result, new_wave)
    wave = new_wave
  if not inclusive:
    result = [op for op in result if op not in seed_ops]
  return result


class UnliftableError(Exception):
  """Raised if a Tensor cannot be lifted from the graph."""

  # Prevent autograph from rewriting this error.
  ag_pass_through = True


def _as_operation(op_or_tensor):
  if isinstance(op_or_tensor, ops.Tensor):
    return op_or_tensor.op
  return op_or_tensor


def graph_inputs(op):
  return [x.op for x in op.inputs] + list(op.control_inputs)


def _path_from(from_op, tensor, sources):
  """Find one path from `from_op` to `tensor`, ignoring `sources`.

  Args:
    from_op: A `tf.Operation`.
    tensor: A `tf.Operation` or `tf.Tensor`.
    sources: A list of `tf.Tensor`.

  Returns:
    A python string containing the path, or "??" if none is found.
  """
  if isinstance(from_op, ops.Tensor):
    from_op = from_op.op

  visited_ops = set(x.op for x in sources)
  ops_to_visit = [_as_operation(tensor)]
  some_op_output = {}
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in visited_ops:
      continue
    visited_ops.add(op)
    if op == from_op:
      path_op = op
      path = [path_op]
      final_op = _as_operation(tensor)
      while path_op != final_op:
        path_op = some_op_output[path_op]
        path.append(path_op)
      return " <- ".join("%s (%s)" % (x.name, x.type) for x in reversed(path))
    else:
      for inp in graph_inputs(op):
        if inp not in visited_ops and inp not in sources:
          some_op_output[inp] = op
          ops_to_visit.append(inp)
  return "??"


# TODO(jmenick) - there is considerable duplication of functionality between
# this function and get_backward_walk_ops(). Need to deduplicate.
def map_subgraph(init_tensor, sources, disallowed_placeholders, visited_ops,
                 op_outputs, add_sources):
  """Walk a Graph and capture the subgraph between init_tensor and sources.

  Note: This function mutates visited_ops and op_outputs.

  Args:
    init_tensor:  A Tensor or Operation where the subgraph terminates.
    sources:  A set of Tensors where subgraph extraction should stop.
    disallowed_placeholders: An optional set of ops which may not appear in the
      lifted graph. Defaults to all placeholders.
    visited_ops: A set of operations which were visited in a prior pass.
    op_outputs: A defaultdict containing the outputs of an op which are to be
      copied into the new subgraph.
    add_sources: A boolean indicating whether placeholders which are not in
      sources should be allowed.

  Returns:
    The set of placeholders upon which init_tensor depends and are not in
    sources.

  Raises:
    UnliftableError: if init_tensor depends on a placeholder which is not in
      sources and add_sources is False.
  """
  ops_to_visit = [_as_operation(init_tensor)]
  extra_sources = object_identity.ObjectIdentitySet()
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in visited_ops:
      continue
    visited_ops.add(op)

    should_raise = False
    if disallowed_placeholders is not None and op in disallowed_placeholders:
      should_raise = True
    elif op.type == "Placeholder":
      if disallowed_placeholders is None and not add_sources:
        should_raise = True
      extra_sources.update(op.outputs)

    if should_raise:
      raise UnliftableError(
          "Unable to lift tensor %s because it depends transitively on "
          "placeholder %s via at least one path, e.g.: %s"
          % (repr(init_tensor), repr(op), _path_from(op, init_tensor, sources)))
    for inp in graph_inputs(op):
      op_outputs[inp].add(op)
      if inp not in visited_ops and inp not in (sources or extra_sources):
        ops_to_visit.append(inp)

  return extra_sources
