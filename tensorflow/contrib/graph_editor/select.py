# pylint: disable=g-bad-file-header
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
"""Various ways of selecting operations and tensors in a graph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.contrib.graph_editor import util
from tensorflow.python.framework import ops as tf_ops


def _can_be_regex(obj):
  """Return True if obj can be turned into a regular expression."""
  return isinstance(obj, (str, type(re.compile(""))))


def _make_regex(obj):
  """Return a compiled regular expression.

  Args:
    obj: a string or a regular expression.
  Returns:
    A compiled regular expression.
  Raises:
    ValueError: if obj could not be converted to a regular expression.
  """
  if not _can_be_regex(obj):
    raise ValueError("Exepected a string or a regex, got: {}".format(type(obj)))

  if isinstance(obj, str):
    return re.compile(obj)
  else:
    return obj


def get_input_ts(ops):
  """Compute the set of input tensors of all the op in ops.

  Args:
    ops: an object convertible to a list of tf.Operation.
  Returns:
    The set of input tensors of all the op in ops.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  ops = util.make_list_of_op(ops)
  ts = set()
  for op in ops:
    ts.update(op.inputs)
  return ts


def get_output_ts(ops):
  """Compute the set of output tensors of all the op in ops.

  Args:
    ops: an object convertible to a list of tf.Operation.
  Returns:
    The set of output tensors of all the op in ops.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  ops = util.make_list_of_op(ops)
  ts = set()
  for op in ops:
    ts.update(op.outputs)
  return ts


def filter_ts(ops, positive_filter=None):
  """Get all the tensors which are input or output of an op in ops.

  Args:
    ops: an object convertible to a list of tf.Operation.
    positive_filter: a function deciding whether to keep a tensor or not.
  Returns:
    A list of tf.Tensor.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  ops = util.make_list_of_op(ops)
  tensors = set()
  tensors.update(get_input_ts(ops))
  tensors.update(get_output_ts(ops))
  if positive_filter is not None:
    tensors = [t for t in tensors if positive_filter(t)]
  return tensors


def filter_ts_from_regex(ops, regex):
  r"""Get all the tensors linked to ops that match the given regex.

  Args:
    ops: an object convertible to a list of tf.Operation.
    regex: a regular expression matching the tensors' name.
      For example, "^foo(/.*)?:\d+$" will match all the tensors in the "foo"
      scope.
  Returns:
    A list of tf.Tensor.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  ops = util.make_list_of_op(ops)
  regex_obj = _make_regex(regex)
  return filter_ts(ops,
                   positive_filter=lambda op: regex_obj.search(op.name))


def filter_ops(ops, positive_filter=None):
  """Get the ops passing the given filter.

  Args:
    ops: an object convertible to a list of tf.Operation.
    positive_filter: a function deciding where to keep an operation or not.
  Returns:
    A list of selected tf.Operation.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  ops = util.make_list_of_op(ops)
  if positive_filter is not None:
    ops = [op for op in ops if positive_filter(op)]
  return ops


def filter_ops_from_regex(ops, regex):
  """Get all the operations that match the given regex.

  Args:
    ops: an object convertible to a list of tf.Operation.
    regex: a regular expression matching the operation's name.
      For example, "^foo(/.*)?$" will match all the operations in the "foo"
      scope.
  Returns:
    A list of tf.Operation.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  ops = util.make_list_of_op(ops)
  regex_obj = _make_regex(regex)
  return filter_ops(ops, lambda op: regex_obj.search(op.name))


def get_name_scope_ops(ops, scope):
  """Get all the operations under the given scope path.

  Args:
    ops: an object convertible to a list of tf.Operation.
    scope: a scope path.
  Returns:
    A list of tf.Operation.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  if scope and scope[-1] == "/":
    scope = scope[:-1]
  return filter_ops_from_regex(ops, "^{}(/.*)?$".format(scope))


def get_ops_ios(ops, control_outputs=True):
  """Return all the tf.Operation which are connected to an op in ops.

  Args:
    ops: an object convertible to a list of tf.Operation.
    control_outputs: an object convertible to a control output dictionary
      (or None). If the dictionary can be created, it will be used to determine
      the surrounding ops (in addition to the regular inputs and outputs).
  Returns:
    All the tf.Operation surrounding the given ops.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  control_outputs = util.convert_to_control_outputs(ops, control_outputs)
  ops = util.make_list_of_op(ops)
  res = set()
  for op in ops:
    res.update([t.op for t in op.inputs])
    for t in op.outputs:
      res.update(t.consumers())
    if control_outputs is not None and op in control_outputs:
      res.update(control_outputs[op])
  return res


def compute_boundary_ts(ops, keep_order=False, ambiguous_are_outputs=True):
  """Compute the tensors at the boundary of a set of ops.

  This function looks at all the tensors connected to the given ops (in/out)
  and classify them into three categories:
  1) input tensors: tensors whose generating operation is not in ops.
  2) output tensors: tensors whose consumer operations are not in ops
  3) inside tensors: tensors which are neither input nor output tensors.

  Args:
    ops: an object convertible to a list of tf.Operation.
    keep_order: if True use ops to determine the order of the resulting input
      and output tensors.
    ambiguous_are_outputs: a tensor can have consumers both inside and outside
      ops. Such tensors are treated as outside tensor if inside_output_as_output
      is True, otherwise they are treated as inside tensor.
  Returns:
    A Python set (list if keep_order is True) of input tensors.
    A Python set (list if keep_order is True) of output tensors.
    A Python set of inside tensors.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  ops = util.make_list_of_op(ops)
  input_tensors = get_input_ts(ops)
  output_tensors = get_output_ts(ops)
  inside_tensors = input_tensors & output_tensors
  # deal with ambiguous tensors
  if ambiguous_are_outputs:
    inside_and_output_tensors = set()
    for t in inside_tensors:
      for op in t.consumers():
        if op not in ops:
          inside_and_output_tensors.add(t)
          break
    output_tensors |= inside_and_output_tensors
    inside_tensors -= inside_and_output_tensors
  outside_input_tensors = input_tensors - inside_tensors
  outside_output_tensors = output_tensors - inside_tensors
  if keep_order:
    outside_input_tensors = [t for t in input_tensors
                             if t in outside_input_tensors]
    outside_output_tensors = [t for t in output_tensors
                              if t in outside_output_tensors]
  return outside_input_tensors, outside_output_tensors, inside_tensors


def get_within_boundary_ops(ops,
                            seed_ops,
                            boundary_ops,
                            inclusive=True,
                            control_outputs=True):
  """Return all the tf.Operation within the given boundary.

  Args:
    ops: an object convertible to a list of tf.Operation. those ops define the
      set in which to perform the operation (if a tf.Graph is given, it
      will be converted to the list of all its operations).
    seed_ops: the operations from which to start expanding.
    boundary_ops: the ops forming the boundary.
    inclusive: if True, the result will also include the boundary ops.
    control_outputs: an object convertible to a control output dictionary
      (or None). If the dictionary can be created, it will be used while
      expanding.
  Returns:
    All the tf.Operation surrounding the given ops.
  Raises:
    TypeError: if ops or seed_ops cannot be converted to a list of tf.Operation.
    ValueError: if the boundary is intersecting with the seeds.
  """
  ops = util.make_list_of_op(ops)
  control_outputs = util.convert_to_control_outputs(ops, control_outputs)
  seed_ops = util.make_list_of_op(seed_ops, allow_graph=False)
  boundary_ops = set(util.make_list_of_op(boundary_ops))
  res = set(seed_ops)
  if boundary_ops & res:
    raise ValueError("Boundary is intersecting with the seeds.")
  wave = set(seed_ops)
  while wave:
    new_wave = set()
    ops_io = get_ops_ios(wave, control_outputs)
    for op in ops_io:
      if op in res:
        continue
      if op in boundary_ops:
        if inclusive:
          res.add(op)
      else:
        new_wave.add(op)
    res.update(new_wave)
    wave = new_wave
  return res


def get_generating_ops(ts):
  """Return all the generating ops of the tensors in ts.

  Args:
    ts: a list of tf.Tensor
  Returns:
    A list of all the generating tf.Operation of the tensors in ts.
  Raises:
    TypeError: if ts cannot be converted to a list of tf.Tensor.
  """
  ts = util.make_list_of_t(ts, allow_graph=False)
  return [t.op for t in ts]


def get_consuming_ops(ts):
  """Return all the consuming ops of the tensors in ts.

  Args:
    ts: a list of tf.Tensor
  Returns:
    A list of all the consuming tf.Operation of the tensors in ts.
  Raises:
    TypeError: if ts cannot be converted to a list of tf.Tensor.
  """
  ts = util.make_list_of_t(ts, allow_graph=False)
  ops = []
  for t in ts:
    for op in t.consumers():
      if op not in ops:
        ops.append(op)
  return ops


def get_forward_walk_ops(seed_ops, inclusive=True, within_ops=None,
                         control_outputs=True):
  """Do a forward graph walk and return all the visited ops.

  Args:
    seed_ops: an iterable of operations from which the forward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the consumers of those tensors.
    inclusive: if True the given seed_ops are also part of the resulting set.
    within_ops: an iterable of tf.Operation whithin which the search is
      restricted. If within_ops is None, the search is performed within
      the whole graph.
    control_outputs: an object convertible to a control output dictionary
      (see function util.convert_to_control_outputs for more details).
      If the dictionary can be created, it will be used while walking the graph
      forward.
  Returns:
    A Python set of all the tf.Operation ahead of seed_ops.
  Raises:
    TypeError: if seed_ops or within_ops cannot be converted to a list of
      tf.Operation.
  """
  if not util.is_iterable(seed_ops): seed_ops = [seed_ops]
  if not seed_ops: return set()
  if isinstance(seed_ops[0], tf_ops.Tensor):
    ts = util.make_list_of_t(seed_ops, allow_graph=False)
    seed_ops = get_consuming_ops(ts)
  else:
    seed_ops = util.make_list_of_op(seed_ops, allow_graph=False)

  control_outputs = util.convert_to_control_outputs(seed_ops, control_outputs)

  seed_ops = frozenset(seed_ops)
  if within_ops:
    within_ops = util.make_list_of_op(within_ops, allow_graph=False)
    within_ops = frozenset(within_ops)
    seed_ops &= within_ops
  def is_within(op):
    return within_ops is None or op in within_ops
  result = set(seed_ops)
  wave = set(seed_ops)
  while wave:
    new_wave = set()
    for op in wave:
      for new_t in op.outputs:
        for new_op in new_t.consumers():
          if new_op not in result and is_within(new_op):
            new_wave.add(new_op)
      if control_outputs is not None and op in control_outputs:
        for new_op in control_outputs[op]:
          if new_op not in result and is_within(new_op):
            new_wave.add(new_op)
    result.update(new_wave)
    wave = new_wave
  if not inclusive:
    result.difference_update(seed_ops)
  return result


def get_backward_walk_ops(seed_ops, inclusive=True, within_ops=None,
                          control_inputs=True):
  """Do a backward graph walk and return all the visited ops.

  Args:
    seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    inclusive: if True the given seed_ops are also part of the resulting set.
    within_ops: an iterable of tf.Operation whithin which the search is
      restricted. If within_ops is None, the search is performed within
      the whole graph.
    control_inputs: if it evaluates to True, control inputs will be used while
      moving backward.
  Returns:
    A Python set of all the tf.Operation behind seed_ops.
  Raises:
    TypeError: if seed_ops or within_ops cannot be converted to a list of
      tf.Operation.
  """
  if not util.is_iterable(seed_ops): seed_ops = [seed_ops]
  if not seed_ops: return set()
  if isinstance(seed_ops[0], tf_ops.Tensor):
    ts = util.make_list_of_t(seed_ops, allow_graph=False)
    seed_ops = get_generating_ops(ts)
  else:
    seed_ops = util.make_list_of_op(seed_ops, allow_graph=False)

  seed_ops = frozenset(util.make_list_of_op(seed_ops))
  if within_ops:
    within_ops = util.make_list_of_op(within_ops, allow_graph=False)
    within_ops = frozenset(within_ops)
    seed_ops &= within_ops
  def is_within(op):
    return within_ops is None or op in within_ops
  result = set(seed_ops)
  wave = set(seed_ops)
  while wave:
    new_wave = set()
    for op in wave:
      for new_t in op.inputs:
        if new_t.op not in result and is_within(new_t.op):
          new_wave.add(new_t.op)
      if control_inputs:
        for new_op in op.control_inputs:
          if new_op not in result and is_within(new_op):
            new_wave.add(new_op)
    result.update(new_wave)
    wave = new_wave
  if not inclusive:
    result.difference_update(seed_ops)
  return result


def get_forward_backward_walk_intersection_ops(forward_seed_ops,
                                               backward_seed_ops,
                                               forward_inclusive=True,
                                               backward_inclusive=True,
                                               within_ops=None,
                                               control_inputs=True):
  """Return the intersection of a foward and a backward walk.

  Args:
    forward_seed_ops: an iterable of operations from which the forward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the consumers of those tensors.
    backward_seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    forward_inclusive: if True the given forward_seed_ops are also part of the
      resulting set.
    backward_inclusive: if True the given backward_seed_ops are also part of the
      resulting set.
    within_ops: an iterable of tf.Operation whithin which the search is
      restricted. If within_ops is None, the search is performed within
      the whole graph.
    control_inputs: an object convertible to a control output dictionary
      (see function util.convert_to_control_outputs for more details).
      If the dictionary can be created, it will be used while walking the graph
      forward.
  Returns:
    A Python set of all the tf.Operation in the intersection of a foward and a
      backward walk.
  Raises:
    TypeError: if forward_seed_ops or backward_seed_ops or within_ops cannot be
      converted to a list of tf.Operation.
  """
  forward_ops = get_forward_walk_ops(forward_seed_ops,
                                     inclusive=forward_inclusive,
                                     within_ops=within_ops,
                                     control_outputs=control_inputs)
  backward_ops = get_backward_walk_ops(backward_seed_ops,
                                       inclusive=backward_inclusive,
                                       within_ops=within_ops,
                                       control_inputs=control_inputs)
  return forward_ops & backward_ops


def get_forward_backward_walk_union_ops(forward_seed_ops,
                                        backward_seed_ops,
                                        forward_inclusive=True,
                                        backward_inclusive=True,
                                        within_ops=None,
                                        control_inputs=True):
  """Return the union of a foward and a backward walk.

  Args:
    forward_seed_ops: an iterable of operations from which the forward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the consumers of those tensors.
    backward_seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    forward_inclusive: if True the given forward_seed_ops are also part of the
      resulting set.
    backward_inclusive: if True the given backward_seed_ops are also part of the
      resulting set.
    within_ops: restrict the search within those operations. If within_ops is
      None, the search is done within the whole graph.
    control_inputs: an object convertible to a control output dictionary
      (see function util.convert_to_control_outputs for more details).
      If the dictionary can be created, it will be used while walking the graph
      forward.
  Returns:
    A Python set of all the tf.Operation in the union of a foward and a
      backward walk.
  Raises:
    TypeError: if forward_seed_ops or backward_seed_ops or within_ops cannot be
      converted to a list of tf.Operation.
  """
  forward_ops = get_forward_walk_ops(forward_seed_ops,
                                     inclusive=forward_inclusive,
                                     within_ops=within_ops,
                                     control_outputs=control_inputs)
  backward_ops = get_backward_walk_ops(backward_seed_ops,
                                       inclusive=backward_inclusive,
                                       within_ops=within_ops,
                                       control_inputs=control_inputs)
  return forward_ops | backward_ops


def select_ops(*args, **kwargs):
  """Helper to select operations.

  Args:
    *args: list of 1) regular expressions (compiled or not) or  2) (array of)
      tf.Operation. tf.Tensor instances are silently ignored.
    **kwargs: 'graph': tf.Graph in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if positive_filter(elem) is
        True. This is optional.
      'restrict_regex': a regular expression is ignored if it doesn't start
        with the substring "(?#ops)".
  Returns:
    list of tf.Operation
  Raises:
    TypeError: if the optional keyword argument graph is not a tf.Graph
      or if an argument in args is not an (array of) tf.Operation
      or an (array of) tf.Tensor (silently ignored) or a string
      or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  """
  # get keywords arguments
  graph = None
  positive_filter = None
  restrict_regex = False
  for k, v in kwargs.iteritems():
    if k == "graph":
      graph = v
      if graph is not None and not isinstance(graph, tf_ops.Graph):
        raise TypeError("Expected a tf.Graph, got: {}".format(type(graph)))
    elif k == "positive_filter":
      positive_filter = v
    elif k == "restrict_regex":
      restrict_regex = v
    else:
      raise ValueError("Wrong keywords argument: {}.".format(k))

  ops = []

  for arg in args:
    if _can_be_regex(arg):
      if graph is None:
        raise ValueError("Use the keyword argument 'graph' to use regex.")
      regex = _make_regex(arg)
      if regex.pattern.startswith("(?#ts)"):
        continue
      if restrict_regex and not regex.pattern.startswith("(?#ops)"):
        continue
      ops_ = filter_ops_from_regex(graph, regex)
      for op_ in ops_:
        if op_ not in ops:
          if positive_filter is None or positive_filter(op_):
            ops.append(op_)
    else:
      ops_aux = util.make_list_of_op(arg, ignore_ts=True)
      if positive_filter is not None:
        ops_aux = [op for op in ops_aux if positive_filter(op)]
      ops_aux = [op for op in ops_aux if op not in ops]
      ops += ops_aux

  return ops


def select_ts(*args, **kwargs):
  """Helper to select tensors.

  Args:
    *args: list of 1) regular expressions (compiled or not) or  2) (array of)
      tf.Tensor. tf.Operation instances are silently ignored.
    **kwargs: 'graph': tf.Graph in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if positive_filter(elem) is
        True. This is optional.
      'restrict_regex': a regular expression is ignored if it doesn't start
        with the substring "(?#ts)".
  Returns:
    list of tf.Tensor
  Raises:
    TypeError: if the optional keyword argument graph is not a tf.Graph
      or if an argument in args is not an (array of) tf.Tensor
      or an (array of) tf.Operation (silently ignored) or a string
      or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  """
  # get keywords arguments
  graph = None
  positive_filter = None
  restrict_regex = False
  for k, v in kwargs.iteritems():
    if k == "graph":
      graph = v
      if graph is not None and not isinstance(graph, tf_ops.Graph):
        raise TypeError("Expected a tf.Graph, got {}".format(type(graph)))
    elif k == "positive_filter":
      positive_filter = v
    elif k == "restrict_regex":
      restrict_regex = v
    else:
      raise ValueError("Wrong keywords argument: {}.".format(k))

  ts = []

  for arg in args:
    if _can_be_regex(arg):
      if graph is None:
        raise ValueError("Use the keyword argument 'graph' to use regex.")
      regex = _make_regex(arg)
      if regex.pattern.startswith("(?#ops)"):
        continue
      if restrict_regex and not regex.pattern.startswith("(?#ts)"):
        continue
      ts_ = filter_ts_from_regex(graph, regex)
      for t_ in ts_:
        if t_ not in ts:
          if positive_filter is None or positive_filter(t_):
            ts.append(t_)
    else:
      ts_aux = util.make_list_of_t(arg, ignore_ops=True)
      if positive_filter is not None:
        ts_aux = [t for t in ts_aux if positive_filter(t)]
      ts_aux = [t for t in ts_aux if t not in ts]
      ts += ts_aux

  return ts


def select_ops_and_ts(*args, **kwargs):
  """Helper to select operations and tensors.

  Args:
    *args: list of 1) regular expressions (compiled or not) or  2) (array of)
      tf.Operation 3) (array of) tf.Tensor.
    **kwargs: 'graph': tf.Graph in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if positive_filter(elem) is
        True. This is optional.
  Returns:
    list of tf.Operation
    list of tf.Tensor
  Raises:
    TypeError: if the optional keyword argument graph is not a tf.Graph
      or if an argument in args is not an (array of) tf.Tensor
      or an (array of) tf.Operation or a string or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  """
  ops = select_ops(*args, restrict_regex=False, **kwargs)
  ts = select_ts(*args, restrict_regex=True, **kwargs)
  return ops, ts


def select_tops(*args, **kwargs):
  """Helper to select operations and tensors combined into a single list.

  Args:
    *args: list of 1) regular expressions (compiled or not) or  2) (array of)
      tf.Operation 3) (array of) tf.Tensor.
    **kwargs: 'graph': tf.Graph in which to perform the regex query.This is
      required when using regex.
      'positive_filter': an elem if selected only if positive_filter(elem) is
        True. This is optional.
  Returns:
    list of tf.Operation or tf.Tensor (combined)
  Raises:
    TypeError: if the optional keyword argument graph is not a tf.Graph
      or if an argument in args is not an (array of) tf.Tensor
      or an (array of) tf.Operation or a string or a regular expression.
    ValueError: if one of the keyword arguments is unexpected or if a regular
      expression is used without passing a graph as a keyword argument.
  """
  ops = select_ops(*args, restrict_regex=False, **kwargs)
  ts = select_ts(*args, restrict_regex=True, **kwargs)
  return ops + ts
