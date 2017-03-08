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
"""Utility funtions for the graph_editor.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import array_ops as tf_array_ops


# TODO(fkp): very generic code, it should be moved in a more generic place.
class ListView(object):
  """Immutable list wrapper.

  This class is strongly inspired by the one in tf.Operation.
  """

  def __init__(self, list_):
    if not isinstance(list_, list):
      raise TypeError("Expected a list, got: {}.".format(type(list_)))
    self._list = list_

  def __iter__(self):
    return iter(self._list)

  def __len__(self):
    return len(self._list)

  def __bool__(self):
    return bool(self._list)

  # Python 3 wants __bool__, Python 2.7 wants __nonzero__
  __nonzero__ = __bool__

  def __getitem__(self, i):
    return self._list[i]


# TODO(fkp): very generic code, it should be moved in a more generic place.
def is_iterable(obj):
  """Return true if the object is iterable."""
  try:
    _ = iter(obj)
  except Exception:  # pylint: disable=broad-except
    return False
  return True


def get_unique_graph(tops, check_types=None):
  """Return the unique graph used by the all the elements in tops.

  Args:
    tops: list of elements to check (usually a list of tf.Operation and/or
      tf.Tensor).
    check_types: check that the element in tops are of given type(s). If None,
      the types (tf.Operation, tf.Tensor) are used.
  Returns:
    The unique graph used by all the tops.
  Raises:
    TypeError: if tops is not a iterable of tf.Operation.
    ValueError: if the graph is not unique.
  """
  if not is_iterable(tops):
    raise TypeError("{} is not iterable".format(type(tops)))
  if check_types is None:
    check_types = (tf_ops.Operation, tf_ops.Tensor)
  g = None
  for op in tops:
    if not isinstance(op, check_types):
      raise TypeError("Expected a tf.Operation, got: {}".format(type(op)))
    if g is None:
      g = op.graph
    elif g != op.graph:
      raise ValueError("Operation {} does not belong to given graph".format(op))
  if g is None:
    raise ValueError("Can't find the unique graph of an empty list")
  return g


def make_list_of_op(ops, check_graph=True, allow_graph=True, ignore_ts=False):
  """Convert ops to a list of tf.Operation.

  Args:
    ops: can be an iterable of tf.Operation, a tf.Graph or a single operation.
    check_graph: if True check if all the operations belong to the same graph.
    allow_graph: if False a tf.Graph cannot be converted.
    ignore_ts: if True, silently ignore tf.Tensor.
  Returns:
    a newly created list of tf.Operation.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation or,
     if check_graph is True, if all the ops do not belong to the same graph.
  """
  if isinstance(ops, tf_ops.Graph):
    if allow_graph:
      return ops.get_operations()
    else:
      raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  else:
    if not is_iterable(ops): ops = [ops]
    if not ops: return []
    if check_graph:
      check_types = None if ignore_ts else tf_ops.Operation
      get_unique_graph(ops, check_types=check_types)
    return [op for op in ops if isinstance(op, tf_ops.Operation)]


# TODO(fkp): move this function in tf.Graph?
def get_tensors(graph):
  """get all the tensors which are input or output of an op in the graph.

  Args:
    graph: a tf.Graph.
  Returns:
    A list of tf.Tensor.
  Raises:
    TypeError: if graph is not a tf.Graph.
  """
  if not isinstance(graph, tf_ops.Graph):
    raise TypeError("Expected a graph, got: {}".format(type(graph)))
  ts = set()
  for op in graph.get_operations():
    ts.update(op.inputs)
    ts.update(op.outputs)
  return ts


def make_list_of_t(ts, check_graph=True, allow_graph=True, ignore_ops=False):
  """Convert ts to a list of tf.Tensor.

  Args:
    ts: can be an iterable of tf.Tensor, a tf.Graph or a single tensor.
    check_graph: if True check if all the tensors belong to the same graph.
    allow_graph: if False a tf.Graph cannot be converted.
    ignore_ops: if True, silently ignore tf.Operation.
  Returns:
    a newly created list of tf.Tensor.
  Raises:
    TypeError: if ts cannot be converted to a list of tf.Tensor or,
     if check_graph is True, if all the ops do not belong to the same graph.
  """
  if isinstance(ts, tf_ops.Graph):
    if allow_graph:
      return get_tensors(ts)
    else:
      raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  else:
    if not is_iterable(ts): ts = [ts]
    if not ts: return []
    if check_graph:
      check_types = None if ignore_ops else tf_ops.Tensor
      get_unique_graph(ts, check_types=check_types)
    return [t for t in ts if isinstance(t, tf_ops.Tensor)]


def create_control_outputs(ops):
  """Create a dictionary of control-output dependencies.

  Args:
    ops: an object convertible to a list of tf.Operation.
  Returns:
    A dictionary where a key is a tf.Operation instance and the corresponding
    value is a list of all the ops which have the key as one of their
    control-input dependencies.
  Raises:
    TypeError: if ops cannot be converted to a list of tf.Operation.
  """
  ops = make_list_of_op(ops)
  res = {}
  for op in ops:
    for control_input in op.control_inputs:
      if control_input not in ops:
        continue
      if control_input not in res:
        res[control_input] = set()
      res[control_input].add(op)
  return res


def convert_to_control_outputs(ops, control_outputs):
  """Create a control output dictionary if needed (and if not already created).

  Args:
    ops: an object convertible to a list of tf.Operation.
    control_outputs: can be None, False, True or the dictionary returned by
      the function create_control_outputs.
  Returns:
    None if control_outputs is None or False and a control outputs dictionary
      if control_outputs if True or an existing dictionary.
  Raises:
    TypeError: if control_outputs cannot be converted to a control outputs
      dictionary or None.
  """
  if control_outputs is None:
    return None
  elif isinstance(control_outputs, bool):
    if control_outputs:
      return create_control_outputs(ops)
    else:
      return None
  else:
    if not isinstance(control_outputs, dict):
      raise TypeError("Expected a dict, got: {}".format(type(control_outputs)))
    return control_outputs


def scope_finalize(scope):
  if scope and scope[-1] != "/":
    scope += "/"
  return scope


def scope_dirname(scope):
  slash = scope.rfind("/")
  if slash == -1: return ""
  return scope[:slash+1]


def scope_basename(scope):
  slash = scope.rfind("/")
  if slash == -1: return scope
  return scope[slash+1:]


def placeholder_name(t=None, scope=None):
  """Create placeholder name for tjhe graph editor.

  Args:
    t: optional tensor on which the placeholder operation's name will be based
      on
    scope: absolute scope with which to predix the placeholder's name. None
      means that the scope of t is preserved. "" means the root scope.
  Returns:
    A new placeholder name prefixed by "geph". Note that "geph" stands for
      Graph Editor PlaceHolder. This convention allows to quickly identify the
      placeholder generated by the Graph Editor.
  Raises:
    TypeError: if t is not None or a tf.Tensor.
  """
  if scope is not None:
    scope = scope_finalize(scope)
  if t is not None:
    if not isinstance(t, tf_ops.Tensor):
      raise TypeError("Expected a tf.Tenfor, got: {}".format(type(t)))
    op_dirname = scope_dirname(t.op.name)
    op_basename = scope_basename(t.op.name)
    if scope is None:
      scope = op_dirname

    if op_basename.startswith("geph__"):
      ph_name = op_basename
    else:
      ph_name = "geph__{}_{}".format(op_basename, t.value_index)

    return scope + ph_name
  else:
    if scope is None:
      scope = ""
    return scope + "geph"


def make_placeholder_from_tensor(t, scope=None):
  """Create a tf.placeholder for the Graph Editor.

  Note that the correct graph scope must be set by the calling function.

  Args:
    t: a tf.Tensor whose name will be used to create the placeholder
      (see function placeholder_name).
    scope: absolute scope within which to create the placeholder. None
      means that the scope of t is preserved. "" means the root scope.
  Returns:
    A newly created tf.placeholder.
  Raises:
    TypeError: if t is not None or a tf.Tensor.
  """
  return tf_array_ops.placeholder(dtype=t.dtype,
                                  shape=t.shape,
                                  name=placeholder_name(t, scope=scope))


def make_placeholder_from_dtype_and_shape(dtype, shape=None, scope=None):
  """Create a tf.placeholder for the Graph Editor.

  Note that the correct graph scope must be set by the calling function.
  The placeholder is named using the function placeholder_name (with no
  tensor argument).

  Args:
    dtype: the tensor type.
    shape: the tensor shape (optional).
    scope: absolute scope within which to create the placeholder. None
      means that the scope of t is preserved. "" means the root scope.
  Returns:
    A newly created tf.placeholder.
  """
  return tf_array_ops.placeholder(dtype=dtype,
                                  shape=shape,
                                  name=placeholder_name(scope=scope))


def check_ts_compatibility(ts0, ts1):
  """Make sure the shape and dtype of the two tensor's lists are compatible.

  Args:
    ts0: an object convertible to a list of tf.Tensor.
    ts1: an object convertible to a list of tf.Tensor.
  Raises:
    ValueError: if any pair of tensors (same index in ts0 and ts1) have
      a dtype or a shape which is not compatible.
  """
  ts0 = make_list_of_t(ts0)
  ts1 = make_list_of_t(ts1)
  if len(ts0) != len(ts1):
    raise ValueError("ts0 and ts1 have different sizes: {} != {}".format(
        len(ts0), len(ts1)))
  for t0, t1 in zip(ts0, ts1):
    # check dtype
    dtype0, dtype1 = t0.dtype, t1.dtype
    if not dtype0.is_compatible_with(dtype1):
      raise ValueError("Dtypes {} and {} are not compatible.".format(dtype0,
                                                                     dtype1))
    # check shape
    shape0, shape1 = t0.get_shape(), t1.get_shape()
    if not shape0.is_compatible_with(shape1):
      raise ValueError("Shapes {} and {} are not compatible.".format(shape0,
                                                                     shape1))


class RerouteMode(object):
  """Enums for reroute's mode.

  swap: the end of tensors a and b are swapped.
  a2b:  the end of the tensor a are also rerouted to the end of the tensor b
    (the end of b is left dangling).
  b2a:  the end of the tensor b are also rerouted to the end of the tensor a
    (the end of a is left dangling).
  """
  swap, a2b, b2a = range(3)

  @classmethod
  def check(cls, mode):
    """Check swap mode.

    Args:
      mode: an integer representing one of the modes.
    Returns:
      True if a is rerouted to b (mode is swap or a2b).
      True if b is rerouted to a (mode is swap or b2a).
    Raises:
      ValueError: if mode is outside the enum range.
    """
    if mode == cls.swap:
      return True, True
    elif mode == cls.b2a:
      return False, True
    elif mode == cls.a2b:
      return True, False
    else:
      raise ValueError("Unknown RerouteMode: {}".format(mode))


def reroute_ts(ts0, ts1, mode, can_modify=None, cannot_modify=None):
  """Reroute the end of the tensors in each pair (t0,t1) in ts0 x ts1.

  This function is the back-bone of the Graph-Editor. It is essentially a thin
  wrapper on top of the tf.Operation._update_input.

  Given a pair of tensor t0, t1 in ts0 x ts1, this function re-route the end
  of t0 and t1 in three possible ways:
  1) The reroute mode is "a<->b" or "b<->a": the tensors' end are swapped. After
  this operation, the previous consumers of t0 are now consumers of t1 and
  vice-versa.
  2) The reroute mode is "a->b": the tensors' end of t0 are re-routed to the
  tensors's end of t1 (which are left dangling). After this operation, the
  previous consumers of t0 are still consuming t0 but the previous consumers of
  t1 are not also consuming t0. The tensor t1 has no consumer.
  3) The reroute mode is "b->a": this mode is the symmetric of the "a->b" mode.

  Note that this function is re-routing the end of two tensors, not the start.
  Re-routing the start of two tensors is not supported by this library. The
  reason for that is the following: TensorFlow, by design, create a strong bond
  between an op and its output tensor. This Graph editor follow this design and
  treat a operation A and its generating tensors {t_i} as an entity which cannot
  be broken. In other words, an op cannot be detached from any of its output
  tensor, ever. What's possible however is to detach an op from its input
  tensors, which is what this function concerns itself about.

  Args:
    ts0: an object convertible to a list of tf.Tensor.
    ts1: an object convertible to a list of tf.Tensor.
    mode: what to do with those tensors: "a->b" or "b<->a" for swaping and
      "a->b" or "b->a" for one direction re-routing.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot unmodified. Any operation
      within cannot_modify will be left untouched by this function.
  Returns:
    the number of individual modifications made by the function.
  Raises:
    TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
    TypeError: if can_modify or cannot_modify is not None and cannot be
      converted to a list of tf.Operation.
  """
  a2b, b2a = RerouteMode.check(mode)
  ts0 = make_list_of_t(ts0)
  ts1 = make_list_of_t(ts1)
  check_ts_compatibility(ts0, ts1)
  if cannot_modify is not None:
    cannot_modify = frozenset(make_list_of_op(cannot_modify))
  if can_modify is not None:
    can_modify = frozenset(make_list_of_op(can_modify))
  nb_update_inputs = 0
  for t0, t1 in zip(ts0, ts1):
    if t0 == t1:
      continue  # silently ignore identical tensors
    consumers0 = set(t0.consumers())
    consumers1 = set(t1.consumers())
    if can_modify is not None:
      consumers0 &= can_modify
      consumers1 &= can_modify
    if cannot_modify is not None:
      consumers0 -= cannot_modify
      consumers1 -= cannot_modify
    consumers0_indices = {}
    consumers1_indices = {}
    for consumer0 in consumers0:
      consumers0_indices[consumer0] = [i for i, t in enumerate(consumer0.inputs)
                                       if t == t0]
    for consumer1 in consumers1:
      consumers1_indices[consumer1] = [i for i, t in enumerate(consumer1.inputs)
                                       if t == t1]
    if a2b:
      for consumer1 in consumers1:
        for i in consumers1_indices[consumer1]:
          consumer1._update_input(i, t0)  # pylint: disable=protected-access
          nb_update_inputs += 1
    if b2a:
      for consumer0 in consumers0:
        for i in consumers0_indices[consumer0]:
          consumer0._update_input(i, t1)  # pylint: disable=protected-access
          nb_update_inputs += 1
  return nb_update_inputs


def swap_ts(ts0, ts1, can_modify=None, cannot_modify=None):
  """For each tensor's pair, swap the end of (t0,t1).

  B0 B1     B0 B1
  |  |    =>  X
  A0 A1     A0 A1

  Args:
    ts0: an object convertible to a list of tf.Tensor.
    ts1: an object convertible to a list of tf.Tensor.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot unmodified. Any operation
      within cannot_modify will be left untouched by this function.
  Returns:
    the number of individual modifications made by the function.
  Raises:
    TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
    TypeError: if can_modify or cannot_modify is not None and cannot be
      converted to a list of tf.Operation.
  """
  return reroute_ts(ts0, ts1, RerouteMode.swap, can_modify, cannot_modify)


def reroute_a2b_ts(ts0, ts1, can_modify=None, cannot_modify=None):
  """For each tensor's pair, replace the end of t1 by the end of t0.

  B0 B1     B0 B1
  |  |    => |/
  A0 A1     A0 A1

  The end of the tensors in ts1 are left dangling.

  Args:
    ts0: an object convertible to a list of tf.Tensor.
    ts1: an object convertible to a list of tf.Tensor.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot unmodified. Any operation
      within cannot_modify will be left untouched by this function.
  Returns:
    the number of individual modifications made by the function.
  Raises:
    TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
    TypeError: if can_modify or cannot_modify is not None and cannot be
      converted to a list of tf.Operation.
  """
  return reroute_ts(ts0, ts1, RerouteMode.a2b, can_modify, cannot_modify)


def reroute_b2a_ts(ts0, ts1, can_modify=None, cannot_modify=None):
  r"""For each tensor's pair, replace the end of t0 by the end of t1.

  B0 B1     B0 B1
  |  |    =>  \|
  A0 A1     A0 A1

  The end of the tensors in ts0 are left dangling.

  Args:
    ts0: an object convertible to a list of tf.Tensor.
    ts1: an object convertible to a list of tf.Tensor.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot unmodified. Any operation
      within cannot_modify will be left untouched by this function.
  Returns:
    the number of individual modifications made by the function.
  Raises:
    TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
    TypeError: if can_modify or cannot_modify is not None and cannot be
      converted to a list of tf.Operation.
  """
  return reroute_ts(ts0, ts1, RerouteMode.b2a, can_modify, cannot_modify)

