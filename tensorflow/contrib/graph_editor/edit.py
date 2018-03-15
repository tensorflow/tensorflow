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

from tensorflow.contrib.graph_editor import reroute
from tensorflow.contrib.graph_editor import select
from tensorflow.contrib.graph_editor import subgraph
from tensorflow.contrib.graph_editor import util
from tensorflow.python.ops import array_ops as tf_array_ops

__all__ = [
    "detach_control_inputs",
    "detach_control_outputs",
    "detach_inputs",
    "detach_outputs",
    "detach",
    "connect",
    "bypass",
]


def detach_control_inputs(sgv):
  """Detach all the external control inputs of the subgraph sgv.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
  """
  sgv = subgraph.make_view(sgv)
  for op in sgv.ops:
    cops = [cop for cop in op.control_inputs if cop not in sgv.ops]
    reroute.remove_control_inputs(op, cops)


def detach_control_outputs(sgv, control_outputs):
  """Detach all the external control outputs of the subgraph sgv.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
    control_outputs: a util.ControlOutputs instance.
  """
  if not isinstance(control_outputs, util.ControlOutputs):
    raise TypeError("Expected a util.ControlOutputs, got: {}",
                    type(control_outputs))
  control_outputs.update()
  sgv = subgraph.make_view(sgv)
  for op in sgv.ops:
    for cop in control_outputs.get(op):
      if cop not in sgv.ops:
        reroute.remove_control_inputs(cop, op)


def detach_inputs(sgv, control_inputs=False):
  """Detach the inputs of a subgraph view.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
      Note that sgv is modified in place.
    control_inputs: if True control_inputs are also detached.
  Returns:
    A tuple `(sgv, input_placeholders)` where
      `sgv` is a new subgraph view of the detached subgraph;
      `input_placeholders` is a list of the created input placeholders.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv = subgraph.make_view(sgv)

  with sgv.graph.as_default():
    input_placeholders = [
        tf_array_ops.placeholder(
            dtype=input_t.dtype, name=util.placeholder_name(input_t))
        for input_t in sgv.inputs
    ]

  reroute.swap_inputs(sgv, input_placeholders)
  if control_inputs:
    detach_control_inputs(sgv)
  return sgv, input_placeholders


def detach_outputs(sgv, control_outputs=None):
  """Detach the output of a subgraph view.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
      Note that sgv is modified in place.
    control_outputs: a util.ControlOutputs instance or None. If not None the
      control outputs are also detached.
  Returns:
    A tuple `(sgv, output_placeholders)` where
      `sgv` is a new subgraph view of the detached subgraph;
      `output_placeholders` is a list of the created output placeholders.
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

  reroute.swap_outputs(sgv_, output_placeholders)
  if control_outputs is not None:
    detach_control_outputs(sgv_, control_outputs)
  return sgv_, output_placeholders


def detach(sgv, control_inputs=False, control_outputs=None, control_ios=None):
  """Detach both the inputs and the outputs of a subgraph view.

  Args:
    sgv: the subgraph view to be detached. This argument is converted to a
      subgraph using the same rules as the function subgraph.make_view.
      Note that sgv is modified in place.
    control_inputs: A boolean indicating whether control inputs are enabled.
    control_outputs: An instance of util.ControlOutputs or None. If not None,
      control outputs are enabled.
    control_ios:  An instance of util.ControlOutputs or None. If not None, both
      control inputs and control outputs are enabled. This is equivalent to set
      control_inputs to True and control_outputs to the util.ControlOutputs
      instance.
  Returns:
    A tuple `(sgv, detached_inputs, detached_outputs)` where:
    `sgv` is a new subgraph view of the detached subgraph;
    `detach_inputs` is a list of the created input placeholders;
    `detach_outputs` is a list of the created output placeholders.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  control_inputs, control_outputs = select.check_cios(control_inputs,
                                                      control_outputs,
                                                      control_ios)
  _, detached_inputs = detach_inputs(sgv, control_inputs)
  _, detached_outputs = detach_outputs(sgv, control_outputs)
  return sgv, detached_inputs, detached_outputs


def connect(sgv0, sgv1, disconnect_first=False):
  """Connect the outputs of sgv0 to the inputs of sgv1.

  Args:
    sgv0: the first subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules as the function
      subgraph.make_view.
      Note that sgv0 is modified in place.
    sgv1: the second subgraph to have its outputs swapped. This argument is
      converted to a subgraph using the same rules as the function
      subgraph.make_view.
      Note that sgv1 is modified in place.
    disconnect_first: if True the current outputs of sgv0 are disconnected.
  Returns:
    A tuple `(sgv0, sgv1)` of the now connected subgraphs.
  Raises:
    StandardError: if sgv0 or sgv1 cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  sgv0 = subgraph.make_view(sgv0)
  sgv1 = subgraph.make_view(sgv1)
  util.check_graphs(sgv0, sgv1)
  if disconnect_first:
    detach_outputs(sgv0)
  sgv0_outputs = subgraph.SubGraphView(passthrough_ts=sgv0.outputs)
  reroute.reroute_inputs(sgv0_outputs, sgv1)
  return sgv0, sgv1


def bypass(sgv):
  """Bypass the given subgraph by connecting its inputs to its outputs.

  Args:
    sgv: the subgraph view to be bypassed. This argument is converted to a
      subgraph using the same rules than the function subgraph.make_view.
      Note that sgv is modified in place.
  Returns:
    A tuple `(sgv, detached_inputs)` where:
      `sgv` is a new subgraph view of the bypassed subgraph;
      `detached_inputs` is a list of the created input placeholders.
  Raises:
    StandardError: if sgv cannot be converted to a SubGraphView using
      the same rules than the function subgraph.make_view.
  """
  # TODO(fkp): allows to plug sgv.inputs to individual sgv.outputs consumers
  sgv = subgraph.make_view(sgv)
  sgv_inputs = list(sgv.inputs)
  sgv, detached_inputs = detach_inputs(sgv)
  reroute.reroute_ts(sgv_inputs, sgv.outputs)
  return sgv, detached_inputs
