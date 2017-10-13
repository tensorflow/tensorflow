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
# ======================================

"""Library of TPU helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope


def initialize_system(embedding_config=None, job=None):
  """Initializes a distributed TPU system for use with TensorFlow.

  Args:
    embedding_config: If not None, an EmbeddingLayerConfiguration proto
      describing the desired configuration of the hardware embedding lookup
      tables. If embedding_config is None, no hardware embeddings can be used.
    job: The job (the XXX in TensorFlow device specification /job:XXX)
      that contains the TPU devices that will be initialized. If job=None
      it is assumed there is only one job in the TensorFlow flock, and an
      error will be returned if this assumption does not hold.
  Returns:
    Op which, when executed, will initialize the system.
  """
  if job is None:
    device_name = "/device:TPU_SYSTEM:0"
  else:
    device_name = "/job:%s/device:TPU_SYSTEM:0" % job
  config_string = ("" if embedding_config is None else
                   embedding_config.SerializeToString())
  with ops.device(device_name):
    init_distributed_tpu = tpu_ops.configure_distributed_tpu(
        embedding_config=config_string)
  return init_distributed_tpu


def shutdown_system(job=None):
  """Shuts down a running a distributed TPU system."""
  if job is None:
    device_name = "/device:TPU_SYSTEM:0"
  else:
    device_name = "/job:%s/device:TPU_SYSTEM:0" % job
  with ops.device(device_name):
    shutdown_distributed_tpu = tpu_ops.shutdown_distributed_tpu()
  return shutdown_distributed_tpu


def core(num):
  """Returns the device name for a core in a replicated TPU computation.

  Args:
    num: the virtual core number within each replica to which operators should
    be assigned.
  Returns:
    A device name, suitable for passing to tf.device().
  """
  return "device:TPU_REPLICATED_CORE:{}".format(num)


# Experimental API to 'break out' of a tpu.rewrite() (or shard(), etc.) context.
# In
#
# XXX
# with tpu.rewrite(...):
#   YYY
#   with tpu.outside_all_rewrites():
#     ZZZ
#
# the Ops in ZZZ are added outside the scope of the rewrite().
# TODO(phawkins): currently outside_all_rewrites() pops out of all nested
# control flow scopes, for example loops. It would make more sense if it only
# popped out of a single scope.
@contextlib.contextmanager
def outside_all_rewrites():
  """Experimental API to 'break out' of a tpu.rewrite() (or shard(), etc.)."""
  with ops.control_dependencies(None):
    yield


class TPUReplicateContext(control_flow_ops.ControlFlowContext):
  """A ControlFlowContext for nodes inside a TPU computation.

  The primary role of TPUReplicateContext is to mark operators inside a
  tpu.replicate() computation with the attribute "_tpu_replicate=XYZ", where XYZ
  is a unique name.

  We use a ControlFlowContext to perform the annotation since it
  integrates with Tensorflow constructs like ResourceVariables. For example,
  if a ResourceVariable is constructed inside a tpu.replicate() block, the
  ResourceVariable implementation can use "with ops.control_dependencies(None)"
  to build the variable's definition outside the replicated computation.
  """

  def __init__(self, name):
    control_flow_ops.ControlFlowContext.__init__(self)
    self._name = name

  def AddOp(self, op):
    self._AddOpInternal(op)

  def _AddOpInternal(self, op):
    # pylint: disable=protected-access
    if any(x.dtype._is_ref_dtype for x in op.inputs):
      raise NotImplementedError(
          "Non-resource Variables are not supported inside TPU computations "
          "(operator name: %s)" % op.name)
    # pylint: enable=protected-access
    if "_tpu_replicate" in op.node_def.attr:
      raise ValueError("TPU computations cannot be nested")
    op.node_def.attr["_tpu_replicate"].s = self._name
    op.graph.prevent_feeding(op)
    op.graph.prevent_fetching(op)

  def AddValue(self, val):
    result = val
    if self._outer_context:
      result = self._outer_context.AddValue(val)
    return result

  def AddInnerOp(self, op):
    self._AddOpInternal(op)
    if self._outer_context:
      self._outer_context.AddInnerOp(op)


def replicate(computation,
              inputs=None,
              infeed_queue=None,
              global_tpu_id=None,
              name=None):
  """Builds a graph operator that runs a replicated TPU computation.

  Args:
    computation: a Python function that builds the computation to replicate.
    inputs: a list of lists of input tensors or None (equivalent to
      [[]]), indexed by [replica_num][input_num]. All replicas must
      have the same number of inputs.
    infeed_queue: if not None, the InfeedQueue from which to append a tuple
      of arguments as inputs to computation.
    global_tpu_id: if not None, a Numpy 2D array indicating the global
      id of each TPU device in the system. The outer dimension of the
      array is host task id, and the inner dimension is device ordinal,
      so e.g., global_tpu_id[x][y] indicates the global id of device
      /task:x/device:TPU_NODE:y.
    name: name of the operator.
  Returns:
    A list of lists of output tensors, indexed by [replica_num][output_num].
  Raises:
    ValueError: if all replicas do not have equal numbers of input tensors.
    ValueError: if the number of inputs per replica does not match
      the number of formal parameters to `computation`.
  """
  if name is None:
    name = "TPUReplicate"
  inputs = [[]] if inputs is None else inputs

  if global_tpu_id is not None:
    # Turn the Numpy array into a flattened list.
    global_tpu_id = global_tpu_id.flatten().tolist()

  if ((not isinstance(inputs, list)) or
      any(not isinstance(inp, (list, tuple)) for inp in inputs)):
    raise TypeError("tpu.replicate() inputs must be a list of lists/tuples")

  num_replicas = len(inputs)

  # No replicas? Nothing to do.
  if num_replicas == 0:
    return []

  # Converts inputs to Tensors.
  inputs = [[ops.convert_to_tensor(x) for x in inp] for inp in inputs]

  # Verifies that all replicas have matching numbers and types of inputs
  input_types = [x.dtype for x in inputs[0]]
  input_arity = len(input_types)
  for i in range(num_replicas):
    if len(inputs[i]) != input_arity:
      raise ValueError("Replicas must have the same number of inputs. "
                       "Replica 0 had {} inputs, replica {} had {} "
                       "inputs.".format(input_arity, i, len(inputs[i])))

    types = [x.dtype for x in inputs[i]]
    if types != input_types:
      raise ValueError(
          "Replicas must have matching input types. Replica 0 had "
          "input types {}, replica {} had input types {}".format(
              input_types, i, types))

  arg_error = tpu_function.check_function_argument_count(
      computation, input_arity, infeed_queue)
  if arg_error is not None:
    if infeed_queue is None:
      raise TypeError(
          "Supplied computation cannot be called with the specified inputs. "
          "You specified %d inputs: %s, but the computation needs %s" % (
              input_arity, str([i.name for i in inputs[0]]), arg_error))
    else:
      raise TypeError(
          "Supplied computation cannot be called with the specified inputs. "
          "You specified %d inputs: %s and %d additional inputs from infeed,"
          " but the computation needs %s" % (input_arity, str(
              [i.name
               for i in inputs[0]]), infeed_queue.number_of_tuple_elements,
                                             arg_error))

  graph = ops.get_default_graph()

  with ops.name_scope(name, "replicate"):
    # Fan-in: Builds a TPUReplicatedInput node for each input.
    computation_inputs = []
    for i in range(0, input_arity):
      replicas = [inputs[replica][i] for replica in xrange(num_replicas)]
      computation_inputs.append(
          tpu_ops.tpu_replicated_input(replicas, name="input{}".format(i)))

    context = TPUReplicateContext(name=graph.unique_name("cluster"))
    try:
      context.Enter()

      metadata = tpu_ops.tpu_replicate_metadata(
          num_replicas=num_replicas, global_tpu_id=global_tpu_id)

      with tpu_function.tpu_shard_context(
          num_replicas), ops.control_dependencies([metadata]):

        # The EncapsulateTPUComputations rewrite needs to identify the
        # replicated arguments inside each computation. Adds identity operators
        # tagged with an attribute _tpu_replicated_input to identify the
        # replicated inputs.
        # pylint: disable=protected-access
        with graph._attr_scope({"_tpu_replicated_input":
                                attr_value_pb2.AttrValue(b=True)}):
          computation_inputs = [
              array_ops.identity(x, name="replicated_input_{}".format(i))
              for i, x in enumerate(computation_inputs)]
        # pylint: enable=protected-access

        # If there is an infeed queue, adds the dequeued values to the
        # computation's inputs.
        if infeed_queue is not None:
          infeed_queue.set_number_of_shards(num_replicas)
          for t in infeed_queue.generate_dequeue_op():
            computation_inputs.append(t)

        # Only resource variables work inside a TPU computation, so turn on
        # resource variables for the computation.
        # TODO(phawkins): consider removing this code. It will
        # be less confusing to clients if they knowingly choose to use resource
        # variables.
        vscope = variable_scope.get_variable_scope()
        saved_use_resource = vscope.use_resource
        vscope.set_use_resource(True)

        outputs = computation(*computation_inputs)

        vscope.set_use_resource(saved_use_resource)

      # If the computation only returned one value, makes it a tuple.
      if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

      try:
        with ops.device(core(0)):
          outputs = [
              o if isinstance(o, ops.Operation) else ops.convert_to_tensor(o)
              for o in outputs
          ]
      except Exception as e:
        raise ValueError(
            "TPU function return values must all either be Operations or "
            "convertible to Tensors. Got '%s'" % str(e))

      # Separates the returned Operations and Tensors.
      output_operations = [o for o in outputs if isinstance(o, ops.Operation)]
      output_tensors = [o for o in outputs
                        if not isinstance(o, ops.Operation)]

      if outputs != output_tensors + output_operations:
        raise ValueError(
            "TPU functions must return zero-or more Tensor values followed by "
            "zero or more Operations.")
      output_arity = len(output_tensors)

      # Wraps outputs in Identity ops. Otherwise a replicated input copied
      # straight to an output would bypass the replicate(). This would be bad
      # because the TPUReplicatedInput/TPUReplicatedOutput operator would not
      # be rewritten away, leading to a runtime error.
      # TODO(phawkins): extend the rewrite to elide these nodes instead.
      with ops.device(core(0)):
        output_tensors = [array_ops.identity(x) for x in output_tensors]
    finally:
      context.Exit()

    # Fan-out: Builds a TPUReplicatedOutput node for each output.
    outputs = [tpu_ops.tpu_replicated_output(output_tensors[i], num_replicas,
                                             name="output{}".format(i))
               for i in xrange(output_arity)]

    with ops.control_dependencies(output_operations):
      if output_arity == 0:
        # Returns a list of NoOps dependent on the replication Op, indexed by
        # [replica_num].
        return [
            control_flow_ops.no_op(name="%s_shard_%d" % (name, i))
            for i in range(num_replicas)
        ]
      else:
        # Wraps the outputs in identity operators so the names of any possible
        # `fetch` nodes are preserved by the replication rewrite.
        return [
            [array_ops.identity(outputs[out][replica],
                                name="output_%d_shard_%d" % (out, replica))
             for out in xrange(output_arity)]
            for replica in xrange(num_replicas)
        ]


def shard(computation,
          inputs=None,
          num_shards=1,
          input_shard_axes=None,
          outputs_from_all_shards=True,
          output_shard_axes=None,
          infeed_queue=None,
          global_tpu_id=None,
          name=None):
  """Shards `computation` for parallel execution.

  `inputs` must be a list of Tensors or None (equivalent to an empty
  list), each of which has a corresponding split axis (from
  `input_shard_axes`). Each input is split into `num_shards` pieces
  along the corresponding axis, and computation is applied to each
  shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  TODO(phawkins): consider adding support for broadcasting Tensors passed
  as inputs.

  If `outputs_from_all_shards` is true, the outputs from all shards of
  `computation` are concatenated back together along their `output_shards_axes`.
  Otherwise, each output is taken from an arbitrary shard.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: a Python function that builds a computation to apply to each
      shard of the input.
    inputs: a list of input tensors or None (equivalent to an empty
      list). Each input tensor has a corresponding shard axes, given
      by `input_shard_axes`, which must have size divisible by
      `num_shards`.
    num_shards: the number of shards.
    input_shard_axes: a list of dimensions along which to shard `inputs`, or
      `None`. `None` means "shard all inputs along dimension 0". If not `None`,
      there must be one dimension per input.
    outputs_from_all_shards: boolean or list of boolean. For each output, if
      `True`, outputs from all shards are concatenated along the corresponding
      `output_shard_axes` entry. Otherwise, each output is taken
      from an arbitrary shard. If the argument is a boolean, the argument's
      value is used for each output.
    output_shard_axes: a list of dimensions along which to concatenate the
      outputs of `computation`, or `None`. `None` means "concatenate all outputs
      along dimension 0". If not `None`, there must be one dimension per output.
      Ignored if `outputs_from_all_shards` is False.
    infeed_queue: if not None, the InfeedQueue to use to augment the inputs of
      `computation`.
    global_tpu_id: if not None, a Numpy 2D array indicating the global
      id of each TPU device in the system. The outer dimension of the
      array is host task id, and the inner dimension is device ordinal,
      so e.g., global_tpu_id[x][y] indicates the global id of device
      /task:x/device:TPU_NODE:y.
    name: name of the operator.
  Returns:
    A list of output tensors.
  Raises:
    ValueError: if num_shards <= 0
    ValueError: if len(input_shard_axes) != len(inputs)
    ValueError: if len(output_shard_axes) != len(outputs from `computation`)
  """

  if num_shards <= 0:
    raise ValueError("num_shards must be a positive integer.")

  # Converts inputs to Tensors.
  inputs = [] if inputs is None else [ops.convert_to_tensor(x) for x in inputs]

  if input_shard_axes is None:
    input_shard_axes = [0] * len(inputs)
  if len(inputs) != len(input_shard_axes):
    raise ValueError("Length of input_shard_axes must be equal to the number "
                     "of inputs.")

  if inputs:
    # Splits the `inputs` along the corresponding `input_shard_axes`, giving
    # lists with layout [input][shard]
    split_inputs = [
        array_ops.split(x, num_shards, axis=axis)
        for (axis, x) in zip(input_shard_axes, inputs)]

    # Transposes the input lists to have layout [shard][input]
    transposed_inputs = [list(i) for i in zip(*split_inputs)]
  else:
    transposed_inputs = [[]] * num_shards

  outputs = replicate(
      computation,
      transposed_inputs,
      infeed_queue=infeed_queue,
      global_tpu_id=global_tpu_id,
      name=name)

  # There must be at least one shard since num_shards > 0.
  # TODO(b/36647078) remove disable when pylint bug is fixed.
  # pylint: disable=indexing-exception
  if isinstance(outputs[0], ops.Operation):
    # pylint: enable=indexing-exception
    # There were no outputs from the computation and replicate returned a list
    # of NoOps with control dependencies on the computation. Return the first
    # one so it can be used as a control dependency or fetch node.
    # TODO(b/36647078) remove disable when pylint bug is fixed.
    # pylint: disable=indexing-exception
    return [outputs[0]]
    # pylint: enable=indexing-exception

  # TODO(b/36647078) remove disable when pylint bug is fixed.
  # pylint: disable=indexing-exception
  num_outputs = len(outputs[0])
  # pylint: enable=indexing-exception

  if output_shard_axes is None:
    output_shard_axes = [0] * num_outputs
  if num_outputs != len(output_shard_axes):
    raise ValueError("Length of output_shard_axes must be equal to the number "
                     "of outputs.")

  if isinstance(outputs_from_all_shards, bool):
    outputs_from_all_shards = [outputs_from_all_shards] * num_outputs

  if num_outputs != len(outputs_from_all_shards):
    raise ValueError("Length of outputs_from_all_shards must be equal to the "
                     "number of outputs.")

  results = []
  for (axis, all_shards, x) in zip(output_shard_axes, outputs_from_all_shards,
                                   zip(*outputs)):
    if all_shards:
      # Concatenate all of the outputs together (use stack for scalars).
      shape = x[0].shape
      is_scalar = shape is not None and (shape.ndims == 0)
      results.append((array_ops.stack(list(x)) if is_scalar
                      else array_ops.concat(list(x), axis=axis)))
    else:
      # TODO(phawkins): use a smarter policy, e.g., round-robin across shards.
      results.append(x[0])

  return results


def batch_parallel(computation,
                   inputs=None,
                   num_shards=1,
                   infeed_queue=None,
                   global_tpu_id=None,
                   name=None):
  """Shards `computation` along the batch dimension for parallel execution.

  Convenience wrapper around shard().

  `inputs` must be a list of Tensors or None (equivalent to an empty
  list). Each input is split into `num_shards` pieces along the 0-th
  dimension, and computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  The outputs from all shards are concatenated back together along their 0-th
  dimension.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: a Python function that builds a computation to apply to each
      shard of the input.
    inputs: a list of input tensors or None (equivalent to an empty
      list). The 0-th dimension of each Tensor must have size
      divisible by `num_shards`.
    num_shards: the number of shards.
    infeed_queue: if not None, the InfeedQueue from which to append a tuple
      of arguments as inputs to `computation`.
    global_tpu_id: if not None, a Numpy 2D array indicating the global
      id of each TPU device in the system. The outer dimension of the
      array is host task id, and the inner dimension is device ordinal,
      so e.g., global_tpu_id[x][y] indicates the global id of device
      /task:x/device:TPU_NODE:y.
    name: name of the operator.
  Returns:
    A list of output tensors.
  Raises:
    ValueError: if num_shards <= 0
  """
  return shard(
      computation,
      inputs,
      num_shards=num_shards,
      infeed_queue=infeed_queue,
      global_tpu_id=global_tpu_id,
      name=name)


def rewrite(computation,
            inputs=None,
            infeed_queue=None,
            global_tpu_id=None,
            name=None):
  """Rewrites `computation` for execution on a TPU system.

  Args:
    computation: a Python function that builds a computation to apply
      to the input. If the function takes n inputs, 'inputs' should be
      a list of n tensors. If the function returns m outputs, rewrite
      will return a list of m tensors.
    inputs: a list of input tensors or None (equivalent to an empty list).
    infeed_queue: if not None, the InfeedQueue from which to append a tuple
      of arguments as inputs to `computation`.
    global_tpu_id: if not None, a Numpy 2D array indicating the global
      id of each TPU device in the system. The outer dimension of the
      array is host task id, and the inner dimension is device ordinal,
      so e.g., global_tpu_id[x][y] indicates the global id of device
      /task:x/device:TPU_NODE:y.
    name: name of the operator.
  Returns:
    A list of output tensors.
  """
  if inputs is not None and not isinstance(inputs, (list, tuple)):
    raise TypeError("tpu.rewrite() inputs must be a list or tuple")

  # TODO(b/36647078) remove disable when pylint bug is fixed.
  # pylint: disable=indexing-exception
  return replicate(
      computation,
      None if inputs is None else [inputs],
      infeed_queue=infeed_queue,
      global_tpu_id=global_tpu_id,
      name=name)[0]
  # pylint: enable=indexing-exception
