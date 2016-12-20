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
"""Various function for graph rerouting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.graph_editor import subgraph
from tensorflow.contrib.graph_editor import util
from tensorflow.python.framework import ops as tf_ops

__all__ = [
    "swap_ts",
    "reroute_a2b_ts",
    "reroute_b2a_ts",
    "swap_inputs",
    "reroute_a2b_inputs",
    "reroute_b2a_inputs",
    "swap_outputs",
    "reroute_a2b_outputs",
    "reroute_b2a_outputs",
    "swap",
    "reroute_a2b",
    "reroute_b2a",
    "remove_control_inputs",
    "add_control_inputs",
]


def _check_ts_compatibility(ts0, ts1):
  """Make sure the shape and dtype of the two tensor's lists are compatible.

  Args:
    ts0: an object convertible to a list of `tf.Tensor`.
    ts1: an object convertible to a list of `tf.Tensor`.
  Raises:
    ValueError: if any pair of tensors (same index in ts0 and ts1) have
      a dtype or a shape which is not compatible.
  """
  ts0 = util.make_list_of_t(ts0)
  ts1 = util.make_list_of_t(ts1)
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


class _RerouteMode(object):
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
      A tuple `(a2b, b2a)` boolean indicating what rerouting needs doing.
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
      raise ValueError("Unknown _RerouteMode: {}".format(mode))


def _reroute_t(t0, t1, consumers1, can_modify=None, cannot_modify=None):
  """Reroute the end of the tensors (t0,t1).

  Warning: this function is directly manipulating the internals of the
  `tf.Graph`.

  Args:
    t0: a tf.Tensor.
    t1: a tf.Tensor.
    consumers1: The consumers of t1 which needs to be rerouted.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot be modified.
      Any operation within cannot_modify will be left untouched by this
      function.
  Returns:
    The number of individual modifications made by the function.
  """
  nb_update_inputs = 0
  if can_modify is not None:
    consumers1 &= can_modify
  if cannot_modify is not None:
    consumers1 -= cannot_modify
  consumers1_indices = {}
  for consumer1 in consumers1:
    consumers1_indices[consumer1] = [i for i, t in enumerate(consumer1.inputs)
                                     if t is t1]
  for consumer1 in consumers1:
    for i in consumers1_indices[consumer1]:
      consumer1._update_input(i, t0)  # pylint: disable=protected-access
      nb_update_inputs += 1
  return nb_update_inputs


def _reroute_ts(ts0, ts1, mode, can_modify=None, cannot_modify=None):
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
  reason for that is the following: TensorFlow, by design, creates a strong bond
  between an op and its output tensor. This Graph editor follows this design and
  treats an operation A and its generating tensors {t_i} as an entity which
  cannot be broken. In other words, an op cannot be detached from any of its
  output tensors, ever. But it is possible to detach an op from its input
  tensors, which is what this function concerns itself with.

  Warning: this function is directly manipulating the internals of the tf.Graph.

  Args:
    ts0: an object convertible to a list of `tf.Tensor`.
    ts1: an object convertible to a list of `tf.Tensor`.
    mode: what to do with those tensors: "a->b" or "b<->a" for swaping and
      "a->b" or "b->a" for one direction re-routing.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot be modified.
      Any operation within cannot_modify will be left untouched by this
      function.
  Returns:
    The number of individual modifications made by the function.
  Raises:
    TypeError: if `ts0` or `ts1` cannot be converted to a list of `tf.Tensor`.
    TypeError: if `can_modify` or `cannot_modify` is not `None` and cannot be
      converted to a list of `tf.Operation`.
  """
  a2b, b2a = _RerouteMode.check(mode)
  ts0 = util.make_list_of_t(ts0)
  ts1 = util.make_list_of_t(ts1)
  _check_ts_compatibility(ts0, ts1)
  if cannot_modify is not None:
    cannot_modify = frozenset(util.make_list_of_op(cannot_modify))
  if can_modify is not None:
    can_modify = frozenset(util.make_list_of_op(can_modify))
  nb_update_inputs = 0
  precomputed_consumers = []
  # precompute consumers to avoid issue with repeated tensors:
  for t0, t1 in zip(ts0, ts1):
    consumers0 = set(t0.consumers())
    consumers1 = set(t1.consumers())
    precomputed_consumers.append((consumers0, consumers1))
  for t0, t1, consumers in zip(ts0, ts1, precomputed_consumers):
    if t0 is t1:
      continue  # Silently ignore identical tensors.
    consumers0, consumers1 = consumers
    if a2b:
      nb_update_inputs += _reroute_t(t0, t1, consumers1, can_modify,
                                     cannot_modify)
    if b2a:
      nb_update_inputs += _reroute_t(t1, t0, consumers0, can_modify,
                                     cannot_modify)
  return nb_update_inputs


def swap_ts(ts0, ts1, can_modify=None, cannot_modify=None):
  """For each tensor's pair, swap the end of (t0,t1).

  B0 B1     B0 B1
  |  |    =>  X
  A0 A1     A0 A1

  Args:
    ts0: an object convertible to a list of `tf.Tensor`.
    ts1: an object convertible to a list of `tf.Tensor`.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot be modified.
      Any operation within cannot_modify will be left untouched by this
      function.
  Returns:
    The number of individual modifications made by the function.
  Raises:
    TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
    TypeError: if can_modify or cannot_modify is not None and cannot be
      converted to a list of tf.Operation.
  """
  return _reroute_ts(ts0, ts1, _RerouteMode.swap, can_modify, cannot_modify)


def reroute_a2b_ts(ts0, ts1, can_modify=None, cannot_modify=None):
  """For each tensor's pair, replace the end of t1 by the end of t0.

  B0 B1     B0 B1
  |  |    => |/
  A0 A1     A0 A1

  The end of the tensors in ts1 are left dangling.

  Args:
    ts0: an object convertible to a list of `tf.Tensor`.
    ts1: an object convertible to a list of `tf.Tensor`.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot be modified. Any
      operation within cannot_modify will be left untouched by this function.
  Returns:
    The number of individual modifications made by the function.
  Raises:
    TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
    TypeError: if can_modify or cannot_modify is not None and cannot be
      converted to a list of tf.Operation.
  """
  return _reroute_ts(ts0, ts1, _RerouteMode.a2b, can_modify, cannot_modify)


def reroute_b2a_ts(ts0, ts1, can_modify=None, cannot_modify=None):
  r"""For each tensor's pair, replace the end of t0 by the end of t1.

  B0 B1     B0 B1
  |  |    =>  \|
  A0 A1     A0 A1

  The end of the tensors in ts0 are left dangling.

  Args:
    ts0: an object convertible to a list of `tf.Tensor`.
    ts1: an object convertible to a list of `tf.Tensor`.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.
    cannot_modify: iterable of operations which cannot be modified.
      Any operation within cannot_modify will be left untouched by this
      function.
  Returns:
    The number of individual modifications made by the function.
  Raises:
    TypeError: if ts0 or ts1 cannot be converted to a list of tf.Tensor.
    TypeError: if can_modify or cannot_modify is not None and cannot be
      converted to a list of tf.Operation.
  """
  return _reroute_ts(ts0, ts1, _RerouteMode.b2a, can_modify, cannot_modify)


def _reroute_sgv_remap(sgv0, sgv1, mode):
  """Remap in place the inputs of two subgraph views to mimic the reroute.

  This function is meant to used by reroute_inputs only.

  Args:
    sgv0: the first subgraph to have its inputs remapped.
    sgv1: the second subgraph to have its inputs remapped.
    mode: reroute mode, see _reroute_ts(...).
  Raises:
    TypeError: if svg0 or svg1 are not SubGraphView.
    ValueError: if sgv0 and sgv1 do not belong to the same graph.
  """
  a2b, b2a = _RerouteMode.check(mode)
  if not isinstance(sgv0, subgraph.SubGraphView):
    raise TypeError("Expected a SubGraphView, got {}".format(type(sgv0)))
  if not isinstance(sgv1, subgraph.SubGraphView):
    raise TypeError("Expected a SubGraphView, got {}".format(type(sgv1)))
  util.check_graphs(sgv0, sgv1)
  sgv0_ = sgv0.copy()
  sgv1_ = sgv1.copy()
  # pylint: disable=protected-access
  if a2b and b2a:
    (sgv0_._input_ts, sgv1_._input_ts) = (sgv1_._input_ts, sgv0_._input_ts)
    (sgv0_._passthrough_ts, sgv1_._passthrough_ts) = (sgv1_._passthrough_ts,
                                                      sgv0_._passthrough_ts)
  elif a2b:
    sgv1_._input_ts = sgv0_._input_ts[:]
    sgv1_._passthrough_ts = sgv0_._passthrough_ts[:]
  elif b2a:
    sgv0_._input_ts = sgv1_._input_ts[:]
    sgv0_._passthrough_ts = sgv1_._passthrough_ts[:]
  # pylint: enable=protected-access

  # Update the passthrough outputs as well.
  def update_passthrough_outputs(a, b):
    # pylint: disable=protected-access
    for i, t in enumerate(b._output_ts):
      if t in a._passthrough_ts:
        ii = a._input_ts.index(t)
        b._output_ts[i] = b._input_ts[ii]
    # pylint: enable=protected-access

  if a2b:
    update_passthrough_outputs(sgv0_, sgv1_)
  if b2a:
    update_passthrough_outputs(sgv1_, sgv0_)

  # in-place
  # pylint: disable=protected-access
  sgv0._assign_from(sgv0_)
  sgv1._assign_from(sgv1_)
  # pylint: enable=protected-access


def _reroute_sgv_inputs(sgv0, sgv1, mode):
  """Re-route all the inputs of two subgraphs.

  Args:
    sgv0: the first subgraph to have its inputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    sgv1: the second subgraph to have its inputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    mode: reroute mode, see _reroute_ts(...).
  Returns:
    A tuple `(sgv0, sgv1)` of subgraph views with their inputs swapped.
      Note that the function argument sgv0 and sgv1 are also modified in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv0 = subgraph.make_view(sgv0)
  sgv1 = subgraph.make_view(sgv1)
  util.check_graphs(sgv0, sgv1)
  can_modify = sgv0.ops + sgv1.ops
  # also allow consumers of passthrough to be modified:
  can_modify += util.get_consuming_ops(sgv0.passthroughs)
  can_modify += util.get_consuming_ops(sgv1.passthroughs)
  _reroute_ts(sgv0.inputs, sgv1.inputs, mode, can_modify=can_modify)
  _reroute_sgv_remap(sgv0, sgv1, mode)
  return sgv0, sgv1


def _reroute_sgv_outputs(sgv0, sgv1, mode):
  """Re-route all the outputs of two operations.

  Args:
    sgv0: the first subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    sgv1: the second subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    mode: reroute mode, see _reroute_ts(...).
  Returns:
    A tuple `(sgv0, sgv1)` of subgraph views with their outputs swapped.
      Note that the function argument sgv0 and sgv1 are also modified in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv0 = subgraph.make_view(sgv0)
  sgv1 = subgraph.make_view(sgv1)
  util.check_graphs(sgv0, sgv1)
  cannot_modify = sgv0.ops + sgv1.ops
  _reroute_ts(sgv0.outputs, sgv1.outputs, mode, cannot_modify=cannot_modify)
  return sgv0, sgv1


def _reroute_sgv(sgv0, sgv1, mode):
  """Re-route both the inputs and the outputs of the two subgraph views.

  This involves swapping all the inputs/ouputs of the two subgraph views.

  Args:
    sgv0: the first subgraph to be swapped. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
    sgv1: the second subgraph to be swapped. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
    mode: reroute mode, see _reroute_ts(...).
  Returns:
    A tuple `(sgv0, sgv1)` of subgraph views with their outputs and inputs
      swapped.
      Note that the function argument sgv0 and sgv1 are also modified in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  _reroute_sgv_outputs(sgv0, sgv1, mode)
  _reroute_sgv_inputs(sgv0, sgv1, mode)
  return sgv0, sgv1


def swap_inputs(sgv0, sgv1):
  """Swap all the inputs of sgv0 and sgv1 (see reroute_inputs)."""
  return _reroute_sgv_inputs(sgv0, sgv1, _RerouteMode.swap)


def reroute_a2b_inputs(sgv0, sgv1):
  """Re-route all the inputs of sgv0 to sgv1 (see reroute_inputs)."""
  return _reroute_sgv_inputs(sgv0, sgv1, _RerouteMode.a2b)


def reroute_b2a_inputs(sgv0, sgv1):
  """Re-route all the inputs of sgv1 to sgv0 (see reroute_inputs)."""
  return _reroute_sgv_inputs(sgv0, sgv1, _RerouteMode.b2a)


def swap_outputs(sgv0, sgv1):
  """Swap all the outputs of sgv0 and sgv1 (see _reroute_outputs)."""
  return _reroute_sgv_outputs(sgv0, sgv1, _RerouteMode.swap)


def reroute_a2b_outputs(sgv0, sgv1):
  """Re-route all the outputs of sgv0 to sgv1 (see _reroute_outputs)."""
  return _reroute_sgv_outputs(sgv0, sgv1, _RerouteMode.a2b)


def reroute_b2a_outputs(sgv0, sgv1):
  """Re-route all the outputs of sgv1 to sgv0 (see _reroute_outputs)."""
  return _reroute_sgv_outputs(sgv0, sgv1, _RerouteMode.b2a)


def swap(sgv0, sgv1):
  """Swap the inputs and outputs of sgv1 to sgv0 (see _reroute)."""
  return _reroute_sgv(sgv0, sgv1, _RerouteMode.swap)


def reroute_a2b(sgv0, sgv1):
  """Re-route the inputs and outputs of sgv0 to sgv1 (see _reroute)."""
  return _reroute_sgv(sgv0, sgv1, _RerouteMode.a2b)


def reroute_b2a(sgv0, sgv1):
  """Re-route the inputs and outputs of sgv1 to sgv0 (see _reroute)."""
  return _reroute_sgv(sgv0, sgv1, _RerouteMode.b2a)


def remove_control_inputs(op, cops):
  """Remove the control inputs cops from co.

  Warning: this function is directly manipulating the internals of the
  `tf.Graph`.

  Args:
    op: a `tf.Operation` from which to remove the control inputs.
    cops: an object convertible to a list of `tf.Operation`.
  Raises:
    TypeError: if op is not a `tf.Operation`.
    ValueError: if any cop in cops is not a control input of op.
  """
  if not isinstance(op, tf_ops.Operation):
    raise TypeError("Expected a tf.Operation, got: {}", type(op))
  cops = util.make_list_of_op(cops, allow_graph=False)
  for cop in cops:
    if cop not in op.control_inputs:
      raise ValueError("{} is not a control_input of {}".format(op.name,
                                                                cop.name))
  # pylint: disable=protected-access
  op._control_inputs = [cop for cop in op._control_inputs if cop not in cops]
  op._recompute_node_def()
  # pylint: enable=protected-access


def add_control_inputs(op, cops):
  """Add the control inputs cops to co.

  Warning: this function is directly manipulating the internals of the tf.Graph.

  Args:
    op: a tf.Operation to which the control inputs are added.
    cops: an object convertible to a list of `tf.Operation`.
  Raises:
    TypeError: if op is not a tf.Operation
    ValueError: if any cop in cops is already a control input of op.
  """
  if not isinstance(op, tf_ops.Operation):
    raise TypeError("Expected a tf.Operation, got: {}", type(op))
  cops = util.make_list_of_op(cops, allow_graph=False)
  for cop in cops:
    if cop in op.control_inputs:
      raise ValueError("{} is already a control_input of {}".format(op.name,
                                                                    cop.name))
  # pylint: disable=protected-access
  op._control_inputs += cops
  op._recompute_node_def()
  # pylint: enable=protected-access
