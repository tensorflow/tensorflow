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
"""AutomaticControlDependencies and related functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator

# Op types that should not run in program order, e.g. because they need to run
# asynchronously to avoid deadlock.
ASYNC_STATEFUL_OPS = [
    "CollectiveReduce",
    "CollectiveBcastSend",
    "CollectiveBcastRecv",
    "NcclAllReduce",
]


class AutomaticControlDependencies(object):
  """Context manager to automatically add control dependencies.

  Code under this context manager will act as if a sensible set of control
  dependencies were present. More specifically:
    1. All stateful ops in the scope will execute (with the exception of ops in
       ASYNC_STATEFUL_OPS)
    2. Stateful ops which modify the same resource will execute in program order

  Note: creating variables in an automatic control dependencies context is not
  supported (the value of the variables will never change as they will keep
  getting reinitialized).

  NOT THREAD SAFE
  """

  def __init__(self):
    self._returned_tensors = set()

  def mark_as_return(self, tensor):
    """Acts like identity but marks the `Tensor` as a return value.

    This will possibly return a copy of the `Tensor`. Usage:

    ```
      with AutomaticControlDependencies() as a:
       ...
       t = a.mark_as_return(t)
      _ = ...(t...)  # i.e. it's safe to use t here
    ```

    Args:
      tensor: the `Tensor` to be marked

    Returns:
      a copy of the `Tensor`.
    """
    if isinstance(tensor, ops.IndexedSlices):
      values = array_ops.identity(tensor.values)
      indices = array_ops.identity(tensor.indices)
      self._returned_tensors.add(indices)
      self._returned_tensors.add(values)
      return ops.IndexedSlices(values, indices, dense_shape=tensor.dense_shape)
    elif isinstance(tensor, sparse_tensor.SparseTensor):
      values = array_ops.identity(tensor.values)
      indices = array_ops.identity(tensor.indices)
      self._returned_tensors.add(indices)
      self._returned_tensors.add(values)
      return sparse_tensor.SparseTensor(
          indices, values, dense_shape=tensor.dense_shape)
    elif isinstance(tensor, tensor_array_ops.TensorArray):
      flow = array_ops.identity(tensor.flow)
      self._returned_tensors.add(flow)
      return tensor_array_ops.build_ta_with_new_flow(tensor, flow)
    # We want to make the return values depend on the stateful operations, but
    # we don't want to introduce a cycle, so we make the return value the result
    # of a new identity operation that the stateful operations definitely don't
    # depend on.
    tensor = array_ops.identity(tensor)
    self._returned_tensors.add(tensor)
    return tensor

  def __enter__(self):
    if context.executing_eagerly():
      return self
    # This code assumes no other thread is adding ops to the graph while
    # we're adding ops to the graph.
    # TODO(apassos): Fix this by locking the graph or using a temporary
    # graph (but that would mess up devices and collections at least,
    # probably other things as well).
    self._graph = ops.get_default_graph()
    self._graph._add_control_dependencies = True  # pylint: disable=protected-access
    self._n_operations = len(self._graph.get_operations())
    return self

  def _process_switch(self, switch_op, ops_which_must_run,
                      last_op_using_resource_tensor, merge_for_resource):
    """Processes a switch node for a resource input.

    When tensorflow creates a cond, it creates a control flow context for each
    branch of the cond. Each external tensor accessed by that branch is routed
    through a switch op, which gets created in the graph _after_ the op which
    uses that tensor get created.

    If the resource comes from another switch op we process that one first.

    _process_switch creates a corresponding merge node for the switch node. This
    merge node is added to the outer control flow context of the switch
    node. We also ensure that:

      1. The switch node executes after the previous op which used the resource
         tensor

      2. Any op which uses a resource output of the switch node executes before
         the merge for the switch node.

      3. The next op which uses the input resource to the switch node (which
         might be another switch node for the other branch of the conditional)
         will execute after the merge node is done.

      4. The merge node is marked as must_run so it will run even if no
         subsequent operation uses the resource.

    Args:
      switch_op: the switch op to be processed
      ops_which_must_run: the set of ops which must run
      last_op_using_resource_tensor: map from resource tensor to last op using
        it
      merge_for_resource: map from resource tensor to merge which must follow
        all usages of it.
    """
    inp = switch_op.inputs[0]
    if inp.dtype == dtypes_module.resource and inp.op.type == "Switch":
      self._process_switch(inp.op, ops_which_must_run,
                           last_op_using_resource_tensor, merge_for_resource)
    if switch_op.outputs[0] in merge_for_resource:
      return
    new_merge = control_flow_ops.merge(switch_op.outputs,
                                       name="artificial_merge")
    new_merge[0].op._control_flow_context = (  # pylint: disable=protected-access
        switch_op._control_flow_context.outer_context)  # pylint: disable=protected-access
    # Ensures the merge always runs
    ops_which_must_run.add(new_merge[0].op)
    if inp in last_op_using_resource_tensor:
      # Ensures the switch executes after the previous op using the resource.
      switch_op._add_control_input(last_op_using_resource_tensor[inp])  # pylint: disable=protected-access
    # Ensure the next op outside the cond happens after the merge.
    last_op_using_resource_tensor[inp] = new_merge[0].op
    if inp in merge_for_resource:
      merge_for_resource[inp]._add_control_input(new_merge[0].op)  # pylint: disable=protected-access
    for o in switch_op.outputs:
      # Ensures the merge will execute after all ops inside the cond
      merge_for_resource[o] = new_merge[0].op

  def __exit__(self, unused_type, unused_value, unused_traceback):
    if context.executing_eagerly():
      return

    if self._graph is not ops.get_default_graph():
      raise RuntimeError(
          "Graph changed while trying to add control dependencies.")

    # pylint: disable=protected-access
    if hasattr(self._graph, "outer_graph"):
      outer_val = self._graph.outer_graph._add_control_dependencies
      self._graph._add_control_dependencies = outer_val
    else:
      self._graph._add_control_dependencies = False
    # pylint: enable=protected-access

    # map from resource tensor to the last op which used it
    last_op_using_resource_tensor = {}
    # set of conditional and loop exits
    ops_which_must_run = set()
    # merge which must depend on ops which use this resource
    merge_for_resource = {}

    new_operations = self._graph.get_operations()[self._n_operations:]

    # Ensures that uses of resource tensors get serialized properly and all
    # execute. This is done by keeping a map from resource tensor to the last op
    # in graph-construction order which used it (last_op_using_resource_tensor).
    #
    # Conditionals are written in TensorFlow such that every external tensor
    # accessed in the conditional goes through a switch op and every return
    # tensor (it's guaranteed that there will be at least one) goes through a
    # merge op.
    #
    # To handle conditionals, switches are handled in a special way (see
    # comments for _process_switch). Merge nodes created by TF's conditional
    # logic (as opposed to by _process_switch) are forced to run and also get a
    # control dependency added to them to ensure all stateful ops inside their
    # control flow context run.
    #
    # We also ensure that if an op is using a resource output by a switch node
    # (that is, a resource tensor for which there's a value in
    # merge_for_resource) this op will run before the merge for that resource.
    #
    # We try to add control inputs to nodes respecting their control flow
    # contexts to avoid dead nodes propagating everywhere and leading to
    # "retval[0] doesn't have value" errors. If a node gets a control dependency
    # on a dead node (i.e. a note from an untaken control flow branch) that node
    # will be marked as dead unless it's a merge node.
    #
    # TODO(apassos): serialize non-resource-taking stateful ops as well, and
    # test that it works. Support while loops. Support init_scope escaping from
    # this.
    for op in new_operations:
      # TODO(apassos) make this code safely support while loops.
      if control_flow_util.IsInWhileLoop(op):
        continue
      control_inputs = set()
      # Ensure stateful ops run
      if (op.type not in self._graph._registered_ops  # pylint: disable=protected-access
          or (self._graph._registered_ops[op.type].is_stateful   # pylint: disable=protected-access
              and op.type not in ASYNC_STATEFUL_OPS)):
        ops_which_must_run.add(op)
      # Ignore switches (they're handled separately)
      if op.type == "Switch" and op.inputs[0].dtype == dtypes_module.resource:
        continue
      # Make merges trigger all other computation which must run
      if op.type == "Merge":
        for o in ops_which_must_run:
          op._add_control_input(o)  # pylint: disable=protected-access
          for inp in o.inputs:
            if inp in last_op_using_resource_tensor:
              last_op_using_resource_tensor[inp] = op
        ops_which_must_run = set([op])
        continue
      found_resource = False
      # Check for any resource inputs. If we find any, we update control_inputs
      # and last_op_using_resource_tensor. Note that we dedup op.inputs in case
      # op receives the same resource tensor twice as input, which would result
      # in op getting a control dependency on itself.
      for inp in set(op.inputs):
        if inp.dtype != dtypes_module.resource:
          continue
        found_resource = True
        # Deal with switches, finally.
        if inp.op.type == "Switch":
          self._process_switch(inp.op, ops_which_must_run,
                               last_op_using_resource_tensor,
                               merge_for_resource)
        # Ensure uses of resources are serialized
        if inp in last_op_using_resource_tensor:
          if (last_op_using_resource_tensor[inp]._control_flow_context  # pylint: disable=protected-access
              is op._control_flow_context):  # pylint: disable=protected-access
            control_inputs.add(last_op_using_resource_tensor[inp])
        # Ensure merges happen after the closing of a cond block
        if inp in merge_for_resource:
          merge_for_resource[inp]._add_control_input(op)  # pylint: disable=protected-access
        last_op_using_resource_tensor[inp] = op
      if (op.op_def.is_stateful and op.type not in ASYNC_STATEFUL_OPS
          and not found_resource and op._control_flow_context is None):  # pylint: disable=protected-access
        if None in last_op_using_resource_tensor:
          op._add_control_input(last_op_using_resource_tensor[None])  # pylint: disable=protected-access
        last_op_using_resource_tensor[None] = op
      control_inputs = [c for c in control_inputs
                        if c._control_flow_context is op._control_flow_context]  # pylint: disable=protected-access
      op._add_control_inputs(control_inputs)  # pylint: disable=protected-access

    # Ensure all ops which must run do run
    for r in self._returned_tensors:
      if ops_which_must_run:
        r.op._add_control_inputs(  # pylint: disable=protected-access
            [o for o in ops_which_must_run
             if o._control_flow_context is r.op._control_flow_context])  # pylint: disable=protected-access


def automatic_control_dependencies(f):
  """Wraps f to automatically insert control dependencies.

  The inserted dependencies ensure that:
    1. All stateful ops in f run when the result of f runs
    2. Updates to the same resources happen in order.

  Args:
    f: the function to be wrapped.

  Returns:
    The wrapped function.
  """

  def wrapper(*args, **kwargs):
    with AutomaticControlDependencies() as a:
      result = f(*args, **kwargs)
      result_flat = [a.mark_as_return(t) for t in nest.flatten(result)]
      return nest.pack_sequence_as(result, result_flat)

  return tf_decorator.make_decorator(f, wrapper)
