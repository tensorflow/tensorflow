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
"""Various function for graph editing."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.graph_editor import select
from tensorflow.contrib.graph_editor import subgraph
from tensorflow.contrib.graph_editor import util
from tensorflow.python.ops import array_ops as tf_array_ops


def _check_graphs(*args):
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
    elif sgv.graph is not None and sgv.graph != graph:
      raise ValueError("Argument[{}]: Wrong graph!".format(i))


def _reroute_sgv_remap(sgv0, sgv1, mode):
  """Remap in place the inputs of two subgraph views to mimic the reroute.

  This function is meant to used by reroute_inputs only.

  Args:
    sgv0: the first subgraph to have its inputs remapped.
    sgv1: the second subgraph to have its inputs remapped.
    mode: reroute mode, see util.reroute_ts(...).
  Raises:
    TypeError: if svg0 or svg1 are not SubGraphView.
    ValueError: if sgv0 and sgv1 do not belong to the same graph.
  """
  a2b, b2a = util.RerouteMode.check(mode)
  if not isinstance(sgv0, subgraph.SubGraphView):
    raise TypeError("Expected a SubGraphView, got {}".format(type(sgv0)))
  if not isinstance(sgv1, subgraph.SubGraphView):
    raise TypeError("Expected a SubGraphView, got {}".format(type(sgv1)))
  _check_graphs(sgv0, sgv1)
  sgv0_ = sgv0.copy()
  sgv1_ = sgv1.copy()
  # pylint: disable=protected-access
  if a2b and b2a:
    (sgv0_._input_ts, sgv1_._input_ts) = (
        sgv1_._input_ts, sgv0_._input_ts)
    (sgv0_._passthrough_ts, sgv1_._passthrough_ts) = (
        sgv1_._passthrough_ts, sgv0_._passthrough_ts)
  elif a2b:
    sgv1_._input_ts = sgv0_._input_ts[:]
    sgv1_._passthrough_ts = sgv0_._passthrough_ts[:]
  elif b2a:
    sgv0_._input_ts = sgv1_._input_ts[:]
    sgv0_._passthrough_ts = sgv1_._passthrough_ts[:]

  # Update the passthrough outputs as well.
  def update_passthrough_outputs(a, b):
    for i, t in enumerate(b._output_ts):
      if t in a._passthrough_ts:
        ii = a._input_ts.index(t)
        b._output_ts[i] = b._input_ts[ii]
  if a2b: update_passthrough_outputs(sgv0_, sgv1_)
  if b2a: update_passthrough_outputs(sgv1_, sgv0_)

  # in-place
  sgv0._assign_from(sgv0_)
  sgv1._assign_from(sgv1_)


def reroute_inputs(sgv0, sgv1, mode):
  """Re-route all the inputs of two subgraphs.

  Args:
    sgv0: the first subgraph to have its inputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    sgv1: the second subgraph to have its inputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    mode: reroute mode, see util.reroute_ts(...).
  Returns:
    Two new subgraph views with their inputs swapped.
      Note that sgv0 and sgv1 are also modified in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv0 = subgraph.make_view(sgv0)
  sgv1 = subgraph.make_view(sgv1)
  _check_graphs(sgv0, sgv1)
  can_modify = sgv0.ops + sgv1.ops
  # also allow consumers of passthrough to be modified:
  can_modify += select.get_consuming_ops(sgv0.passthroughs)
  can_modify += select.get_consuming_ops(sgv1.passthroughs)
  util.reroute_ts(sgv0.inputs, sgv1.inputs, mode, can_modify=can_modify)
  _reroute_sgv_remap(sgv0, sgv1, mode)
  return sgv0, sgv1


def swap_inputs(sgv0, sgv1):
  """Swap all the inputs of sgv0 and sgv1 (see reroute_inputs)."""
  return reroute_inputs(sgv0, sgv1, util.RerouteMode.swap)


def reroute_a2b_inputs(sgv0, sgv1):
  """Re-route all the inputs of sgv0 to sgv1 (see reroute_inputs)."""
  return reroute_inputs(sgv0, sgv1, util.RerouteMode.a2b)


def reroute_b2a_inputs(sgv0, sgv1):
  """Re-route all the inputs of sgv1 to sgv0 (see reroute_inputs)."""
  return reroute_inputs(sgv0, sgv1, util.RerouteMode.b2a)


def reroute_outputs(sgv0, sgv1, mode):
  """Re-route all the outputs of two operations.

  Args:
    sgv0: the first subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    sgv1: the second subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    mode: reroute mode, see util.reroute_ts(...).
  Returns:
    Two new subgraph views with their outputs swapped.
      Note that sgv0 and sgv1 are also modified in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv0 = subgraph.make_view(sgv0)
  sgv1 = subgraph.make_view(sgv1)
  _check_graphs(sgv0, sgv1)
  cannot_modify = sgv0.ops + sgv1.ops
  util.reroute_ts(sgv0.outputs, sgv1.outputs, mode, cannot_modify=cannot_modify)
  return sgv0, sgv1


def swap_outputs(sgv0, sgv1):
  """Swap all the outputs of sgv0 and sgv1 (see reroute_outputs)."""
  return reroute_outputs(sgv0, sgv1, util.RerouteMode.swap)


def reroute_a2b_outputs(sgv0, sgv1):
  """Re-route all the outputs of sgv0 to sgv1 (see reroute_outputs)."""
  return reroute_outputs(sgv0, sgv1, util.RerouteMode.a2b)


def reroute_b2a_outputs(sgv0, sgv1):
  """Re-route all the outputs of sgv1 to sgv0 (see reroute_outputs)."""
  return reroute_outputs(sgv0, sgv1, util.RerouteMode.b2a)


def reroute(sgv0, sgv1, mode):
  """Re-route both the inputs and the outputs of the two subgraph views.

  This involves swapping all the inputs/ouputs of the two subgraph views.

  Args:
    sgv0: the first subgraph to be swapped. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
    sgv1: the second subgraph to be swapped. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
    mode: reroute mode, see util.reroute_ts(...).
  Returns:
    Two new subgraph views with their outputs and inputs swapped.
      Note that sgv0 and sgv1 are also modified in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  reroute_outputs(sgv0, sgv1, mode)
  reroute_inputs(sgv0, sgv1, mode)
  return sgv0, sgv1


def swap(sgv0, sgv1):
  """Swap the inputs and outputs of sgv1 to sgv0 (see reroute)."""
  return reroute(sgv0, sgv1, util.RerouteMode.swap)


def reroute_a2b(sgv0, sgv1):
  """Re-route the inputs and outputs of sgv0 to sgv1 (see reroute_outputs)."""
  return reroute(sgv0, sgv1, util.RerouteMode.a2b)


def reroute_b2a(sgv0, sgv1):
  """Re-route the inputs and outputs of sgv1 to sgv0 (see reroute_outputs)."""
  return reroute(sgv0, sgv1, util.RerouteMode.b2a)


def detach_inputs(sgv):
  """Detach the inputs of a subgraph view.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
  Returns:
    A new subgraph view of the detached subgraph.
      Note that sgv is also modified in place.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv = subgraph.make_view(sgv)

  with sgv.graph.as_default():
    input_placeholders = [
        tf_array_ops.placeholder(dtype=input_t.dtype,
                                 name=util.placeholder_name(input_t))
        for input_t in sgv.inputs
    ]

  return swap_inputs(sgv, input_placeholders)


def detach_outputs(sgv):
  """Detach the outputa of a subgraph view.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
  Returns:
    A new subgraph view of the detached subgraph.
      Note that sgv is also modified in place.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv = subgraph.make_view(sgv)
  # only select outputs with consumers
  sgv_ = sgv.remap_outputs([output_id
                            for output_id, output_t in enumerate(sgv.outputs)
                            if output_t.consumers()])
  # create consumer subgraph and remap
  consumers_sgv = subgraph.SubGraphView(sgv_.consumers())
  consumers_sgv = consumers_sgv.remap_inputs(
      [input_id for input_id, input_t in enumerate(consumers_sgv.inputs)
       if input_t in sgv_.outputs])

  with sgv_.graph.as_default():
    output_placeholders = [
        util.make_placeholder_from_tensor(input_t)
        for input_t in consumers_sgv.inputs
    ]

  return swap_outputs(sgv_, output_placeholders)


def detach(sgv):
  """Detach both the inputs and the outputs of a subgraph view.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
  Returns:
    A new subgraph view of the detached subgraph.
      Note that sgv is also modified in place.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  _, detached_inputs = detach_inputs(sgv)
  _, detached_outputs = detach_outputs(sgv)
  return sgv, detached_inputs, detached_outputs


def connect(sgv0, sgv1, disconnect_first=False):
  """Connect the outputs of sgv0 to the inputs of sgv1.

  Args:
    sgv0: the first subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    sgv1: the second subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules than the function
      subgraph.make_view.
    disconnect_first: if True thecurrent outputs of sgv0 are disconnected.
  Returns:
    Two new subgraph views (now connected). sgv0 and svg1 are also modified
      in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv0 = subgraph.make_view(sgv0)
  sgv1 = subgraph.make_view(sgv1)
  _check_graphs(sgv0, sgv1)
  if disconnect_first:
    detach_outputs(sgv0)
  sgv0_outputs = subgraph.SubGraphView(passthrough_ts=sgv0.outputs)
  reroute_a2b_inputs(sgv0_outputs, sgv1)
  return sgv0, sgv1


def remove(sgv, reconnect_after=False):
  """Remove sgv and optionally reconnect its inputs and outputs.

  Args:
    sgv: the subgraph view to be removed. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
    reconnect_after: if False, the inputs and outputs of sgv are not
      reconnected after the removal.
  Returns:
    A new subgraph view of the removed subgraph.
      Note that sgv is also modified in place.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv = subgraph.make_view(sgv)
  util.check_ts_compatibility(sgv.inputs, sgv.outputs)
  sgv, detached_inputs, detached_outputs = detach(sgv)
  if reconnect_after:
    connect(detached_inputs, detached_outputs)
  return sgv

