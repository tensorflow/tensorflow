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

import collections
import enum

from tensorflow.python.eager import context
from tensorflow.python.framework import auto_control_deps_utils as utils
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import registry
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator

# LINT.IfChange
# Op types that should not run in program order, e.g. because they need to run
# asynchronously to avoid deadlock.
ASYNC_STATEFUL_OPS = [
    "CollectiveGather",
    "CollectiveGatherV2",
    "CollectiveReduce",
    "CollectiveReduceV2",
    "CollectiveBcastSend",
    "CollectiveBcastRecv",
    "NcclAllReduce",
    # We do not add "Send" here since we want it to be added as a control output
    # in order to avoid being pruned.
    "Recv",
]

LEGACY_RANDOM_OPS = [
    # These may be used in variable initializers -- thus their execution should
    # not be dependent on other stateful operations.  This is because although
    # according to program order, tf.Variables may be created in sequence,
    # their initialization happens outside of the program order (specifically,
    # in graph mode their initialization happens by calling a grouped
    # initializer operation or in eager mode, where initialization is lifted
    # out of the tf.function and executed the first time the function is
    # executed).
    #
    # Unless there is a specific dependency between the initializers
    # themselves (e.g. one initializer depends on a Variable whose value depends
    # on another initializer), the initialization can happen in any order so
    # long as it's before the associated Variable read operations.
    #
    # Note that in general the randomness of legacy random operations is only
    # guaranteed by providing a graph-level and op-level seed (and ordering of
    # the same op across multiple iterations of a while_loop is specifically not
    # guaranteed; see the discussion below).
    #
    # There is a possible race condition inside while_loop where the same
    # random OpKernel instantiation is reused across multiple steps
    # of the loop.  Since legacy Random OpKernels have an internal rng state,
    # automatic dependency tracking across loop steps would likely
    # fix this race; and for that case this denylist is problematic.
    # However, since automatic dependency tracking inside while loops is not
    # currently supported, and there are no other examples of OpKernel reuse
    # (each OpKernel is associated with a unique op in graph mode),
    # this denylist has no effect on the aforementioned behavior.
    #
    # TODO(ebrevdo,skyewm): Modify the check against this denylist to
    # only occur when the op is inside a "variable initialization scope"; and
    # add proper autodeps inside while_loops that respects this updated check.
    "RandomUniform",
    "RandomUniformInt",
    "RandomStandardNormal",
    "ParameterizedTruncatedNormal",
    "TruncatedNormal",
    "RandomShuffle",
    "Multinomial",
    "RandomGamma",
    "RandomGammaGrad",
    "RandomPoisson",
    "RandomPoissonV2",
]

_ORDER_INSENSITIVE_STATEFUL_OPS = [
    "CudnnRNN", "CudnnRNNBackprop", "CudnnRNNV2", "CudnnRNNV3",
    "CudnnRNNBackpropV2", "CudnnRNNBackpropV3",
    "EnqueueTPUEmbeddingSparseBatch", "EnqueueTPUEmbeddingIntegerBatch",
    "EnqueueTPUEmbeddingSparseTensorBatch",
    "EnqueueTPUEmbeddingRaggedTensorBatch", "RestoreV2", "SaveV2"
]
# LINT.ThenChange(//tensorflow/core/grappler/optimizers/function_optimizer.cc)

_ALL_DENYLISTED_OPS = (
    set(ASYNC_STATEFUL_OPS) | set(LEGACY_RANDOM_OPS)
    | set(_ORDER_INSENSITIVE_STATEFUL_OPS))

# Op types that are marked as stateless, but should be allowlisted to add auto
# control dependencies.
_ALLOWLIST_STATELESS_OPS = [
    # As TPU collective ops are blocking, if there are more than one collective
    # op in the function, we need to make sure different collectives ops are
    # scheduled in certain orders. Otherwise if at the same time all the
    # replicas are launching different collective ops/programs, it may cause
    # deadlock.
    "AllToAll",
    "CrossReplicaSum",
    "CollectivePermute",
]


def op_is_stateful(op):
  # pylint: disable=protected-access
  return (op._is_stateful and op.type not in _ALL_DENYLISTED_OPS) or (
      op.type in _ALLOWLIST_STATELESS_OPS)


class ResourceType(enum.Enum):
  READ_ONLY = "read-only"
  READ_WRITE = "read-write"


def collective_manager_ids_from_op(op):
  """Returns CollectiveManager ID from the op if one exists, else None.

  CollectiveManager adds collective and no_op operations tagged with an ID,
  unique to the manager object. This function extracts that ID, or None, if the
  node was not generated by a CollectiveManager.

  Args:
    op: `Operation` to get the collective manager ID from.

  Returns:
    List of CollectiveManager IDs used by the op.
  """
  if op.type == "CollectiveReduce":
    try:
      return [op.get_attr("_collective_manager_id")]
    except ValueError:
      pass
  elif op.type == "StatefulPartitionedCall":
    try:
      return op.get_attr(utils.COLLECTIVE_MANAGER_IDS)
    except ValueError:
      pass
  return []


class AutomaticControlDependencies(object):
  """Context manager to automatically add control dependencies.

  Code under this context manager will act as if a sensible set of control
  dependencies were present. More specifically:
    1. All stateful ops in the scope will execute (with the exception of ops in
       ASYNC_STATEFUL_OPS and LEGACY_RANDOM_OPS)
    2. Stateful ops which modify the same resource will execute in program order

  Note: creating variables in an automatic control dependencies context is not
  supported (the value of the variables will never change as they will keep
  getting reinitialized).

  NOT THREAD SAFE
  """

  __slots__ = [
      "_returned_tensors", "ops_which_must_run", "_graph", "_n_operations",
      "collective_manager_ids_used"
  ]

  def __init__(self):
    self._returned_tensors = object_identity.ObjectIdentitySet()
    self.ops_which_must_run = set()

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
                      last_write_to_resource, merge_for_resource):
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
      last_write_to_resource: map from resource tensor to last op updating
        it
      merge_for_resource: map from resource tensor to merge which must follow
        all usages of it.
    """
    # pylint: disable=protected-access
    inp = switch_op.inputs[0]
    input_id = ops.tensor_id(inp)
    if inp.dtype == dtypes_module.resource and inp.op.type == "Switch":
      self._process_switch(inp.op, ops_which_must_run, last_write_to_resource,
                           merge_for_resource)
    output = switch_op.outputs[0]
    output_id = ops.tensor_id(output)
    if output_id in merge_for_resource:
      return
    new_merge = control_flow_ops.merge(
        switch_op.outputs, name="artificial_merge")
    new_merge[0].op._control_flow_context = (
        switch_op._control_flow_context.outer_context)
    # Ensures the merge always runs
    ops_which_must_run.add(new_merge[0].op)
    if input_id in last_write_to_resource:
      # Ensures the switch executes after the previous op using the resource.
      switch_op._add_control_input(last_write_to_resource[input_id])
    # Ensure the next op outside the cond happens after the merge.
    last_write_to_resource[input_id] = new_merge[0].op
    if input_id in merge_for_resource:
      merge_for_resource[input_id]._add_control_input(new_merge[0].op)
    for o in switch_op.outputs:
      # Ensures the merge will execute after all ops inside the cond
      merge_for_resource[ops.tensor_id(o)] = new_merge[0].op

  def __exit__(self, unused_type, unused_value, unused_traceback):
    # pylint: disable=protected-access
    if context.executing_eagerly():
      return

    if self._graph is not ops.get_default_graph():
      raise RuntimeError(
          "Graph changed while trying to add control dependencies.")

    if hasattr(self._graph, "outer_graph"):
      outer_val = self._graph.outer_graph._add_control_dependencies
      self._graph._add_control_dependencies = outer_val
    else:
      self._graph._add_control_dependencies = False

    # map from resource tensor to the last op which wrote to it
    last_write_to_resource = {}
    # map from resource tensor to the list of reads from it since the last
    # write or since the beginning of the function.
    reads_since_last_write_to_resource = collections.defaultdict(list)
    # CollectiveManager manager_ids within a particular function call should not
    # be needed outside of that function call. So we keep them separate (though
    # the general idea of the maps is the same, in the future, we'll need to
    # correctly thread the control output outside).
    # Map from collective manager scope to the last op which used it
    collective_manager_scopes_opened = {}
    collective_manager_scopes_used = {}
    # set of conditional and loop exits
    ops_which_must_run = set()
    # merge which must depend on ops which use this resource
    merge_for_resource = {}

    new_operations = self._graph.get_operations()[self._n_operations:]

    # Ensures that uses of resource tensors get serialized properly and all
    # execute. This is done by keeping a map from resource tensor to the last op
    # in graph-construction order which used it (last_write_to_resource).
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
      # Ensure stateful ops run.
      # Read-only ops are added to control outputs if the read value is
      # consumed. This covers the case when the read value is returned from
      # the function since that goes through a tf.identity in mark_as_return.
      if (op_def_registry.get(op.type) is None or
          (op_is_stateful(op) and
           (op.type not in utils.RESOURCE_READ_OPS or
            any(output.consumers() for output in op.outputs)))):
        ops_which_must_run.add(op)
      # Make a note of all opened manager_ids.
      if op.type == "NoOp":
        try:
          collective_manager_scopes_opened[op.get_attr(
              "_collective_manager_id")] = op
        except ValueError:
          pass
      # Ignore switches (they're handled separately)
      if op.type == "Switch" and op.inputs[0].dtype == dtypes_module.resource:
        continue
      # Make merges trigger all other computation which must run
      if op.type == "Merge":
        for o in ops_which_must_run:
          op._add_control_input(o)
          for inp in o.inputs:
            input_id = ops.tensor_id(inp)
            if input_id in last_write_to_resource:
              last_write_to_resource[input_id] = op
        ops_which_must_run = set([op])
        continue

      resource_inputs = set()
      # Check for any resource inputs. If we find any, we update control_inputs
      # and last_write_to_resource.
      for inp, resource_type in _get_resource_inputs(op):
        is_read = resource_type == ResourceType.READ_ONLY
        input_id = ops.tensor_id(inp)

        # If the op receives the same resource tensor twice as an input, we skip
        # to avoid the op getting a control dependency on itself.
        if input_id in resource_inputs:
          continue

        resource_inputs.add(input_id)
        # Deal with switches, finally.
        if inp.op.type == "Switch":
          self._process_switch(inp.op, ops_which_must_run,
                               last_write_to_resource, merge_for_resource)
        is_building_function = op.graph.building_function
        # Ensure uses of resources are serialized
        if input_id in last_write_to_resource:
          if is_building_function or (
              last_write_to_resource[input_id]._control_flow_context
              is op._control_flow_context):
            control_inputs.add(last_write_to_resource[input_id])
        # Ensure merges happen after the closing of a cond block
        if input_id in merge_for_resource:
          merge_for_resource[input_id]._add_control_input(op)
        if is_read:
          reads_since_last_write_to_resource[input_id].append(op)
        else:
          control_inputs.update(reads_since_last_write_to_resource[input_id])
          reads_since_last_write_to_resource[input_id] = []
          last_write_to_resource[input_id] = op

      if (op_is_stateful(op) and not resource_inputs
          and op._control_flow_context is None):
        if None in last_write_to_resource:
          op._add_control_input(last_write_to_resource[None])
        last_write_to_resource[None] = op

      # Ensure ordering of collective ops
      manager_ids = collective_manager_ids_from_op(op)
      for manager_id in manager_ids:
        if manager_id in collective_manager_scopes_opened:
          # Chain this function call if the scope was opened.
          op._add_control_input(collective_manager_scopes_opened[manager_id])
          collective_manager_scopes_opened[manager_id] = op
        else:
          # If this op is in a scope not created here, create a chain starting
          # at this op.
          if manager_id in collective_manager_scopes_used:
            op._add_control_input(collective_manager_scopes_used[manager_id])
          collective_manager_scopes_used[manager_id] = op

      if control_inputs and not is_building_function:
        control_inputs = [
            c for c in control_inputs
            if c._control_flow_context is op._control_flow_context
        ]

      op._add_control_inputs(control_inputs)

    # Ensure all ops which must run do run
    self.ops_which_must_run.update(ops_which_must_run)
    for r in nest.flatten(list(self._returned_tensors), expand_composites=True):
      if self.ops_which_must_run:
        updated_ops_which_must_run = []
        if r.graph.building_function:
          updated_ops_which_must_run = self.ops_which_must_run
        else:
          updated_ops_which_must_run = [
              o for o in self.ops_which_must_run
              if o._control_flow_context is r.op._control_flow_context
          ]
        r.op._add_control_inputs(updated_ops_which_must_run)

    self.collective_manager_ids_used = collective_manager_scopes_used


_acd_resource_resolvers_registry = registry.Registry("acd_resource_resolvers")


def register_acd_resource_resolver(f):
  """Register a function for resolving resources touched by an op.

  `f` is called for every Operation added in the ACD context with the op's
  original resource reads and writes. `f` is expected to update the sets of
  resource reads and writes in-place and return True if it updated either of the
  sets, False otherwise.

  Example:
  @register_acd_resource_resolver
  def ResolveIdentity(op, resource_reads, resource_writes):
    # op: The `Operation` being processed by ACD currently.
    # resource_reads: An `ObjectIdentitySet` of read-only resources.
    # resource_writes: An `ObjectIdentitySet` of read-write resources.
    if not resource_reads or resource_writes:
      return False
    def update(resource_inputs):
      to_add = []
      to_remove = []
      for t in resource_inputs:
        if t.op.type == "Identity":
          to_remove.append(t)
          to_add.append(t.op.inputs[0])
      if not to_add and not to_remove:
        return False
      for t in to_remove:
        resource_inputs.discard(t)
      resource_inputs.update(to_add)
      return True
    return update(resource_reads) or update(resource_writes)

  Args:
    f: Python function with signature
    (Operation, ObjectIdentitySet, ObjectIdentitySet) -> bool

  Returns:
    The function `f` after adding it to the registry.
  """
  _acd_resource_resolvers_registry.register(f)
  return f


def _get_resource_inputs(op):
  """Returns an iterable of resources touched by this `op`."""
  reads, writes = utils.get_read_write_resource_inputs(op)
  saturated = False
  while not saturated:
    saturated = True
    for key in _acd_resource_resolvers_registry.list():
      # Resolvers should return true if they are updating the list of
      # resource_inputs.
      # TODO(srbs): An alternate would be to just compare the old and new set
      # but that may not be as fast.
      updated = _acd_resource_resolvers_registry.lookup(key)(op, reads, writes)
      if updated:
        # Conservatively remove any resources from `reads` that are also writes.
        reads = reads.difference(writes)
      saturated = saturated and not updated

  # Note: A resource handle that is not written to is treated as read-only. We
  # don't have a special way of denoting an unused resource.
  for t in reads:
    yield (t, ResourceType.READ_ONLY)
  for t in writes:
    yield (t, ResourceType.READ_WRITE)


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
