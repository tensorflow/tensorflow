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
"""TPU Distribution Strategy.

This is experimental.  It's not ready for general use.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
from tensorflow.contrib.distribute.python import one_device_strategy
from tensorflow.contrib.distribute.python import values
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.contrib.tpu.python.tpu import training_loop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training import device_util
from tensorflow.python.util import nest


def get_tpu_system_metadata(tpu_cluster_resolver):
  """Retrieves TPU system metadata given a TPUClusterResolver."""
  master = tpu_cluster_resolver.master()

  # pylint: disable=protected-access
  cluster_spec = tpu_cluster_resolver.cluster_spec()
  cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
  tpu_system_metadata = (
      tpu_system_metadata_lib._query_tpu_system_metadata(
          master,
          cluster_def=cluster_def,
          query_topology=False))

  return tpu_system_metadata


class TPUStrategy(one_device_strategy.OneDeviceStrategy):
  """Experimental TPU distribution strategy implementation."""

  def __init__(self, tpu_cluster_resolver, steps_per_run, num_cores=None):
    """Initializes the TPUStrategy object.

    Args:
      tpu_cluster_resolver: A tf.contrib.cluster_resolver.TPUClusterResolver,
          which provides information about the TPU cluster.
      steps_per_run: Number of steps to run on device before returning to the
          host. Note that this can have side-effects on performance, hooks,
          metrics, summaries etc.
          This parameter is only used when Distribution Strategy is used with
          estimator or keras.
      num_cores: Number of cores to use on the TPU. If None specified, then
          auto-detect the cores and topology of the TPU system.
    """
    # TODO(sourabhbajaj): OneDeviceStrategy should be initialized with the
    # master node fetched from the cluster resolver.
    super(TPUStrategy, self).__init__('/device:CPU:0')

    self._tpu_cluster_resolver = tpu_cluster_resolver
    self._tpu_metadata = get_tpu_system_metadata(self._tpu_cluster_resolver)
    # TODO(sourabhbajaj): Change this from num_cores to metadata_override
    self._num_cores_override = num_cores

    # TODO(sourabhbajaj): Remove this once performance of running one step
    # at a time is comparable to multiple steps.
    self.steps_per_run = steps_per_run

  def _get_enqueue_op_per_host(self, host_id, iterator, input_shapes,
                               iterations):
    """Create an enqueue op for a single host identified using host_id.

    The while_loop op returned will run `iterations` times and in each run
    enqueue batches for each shard.

    Args:
      host_id: integer, id of the host to run the enqueue ops on.
      iterator: `tf.data` iterator to read the input data.
      input_shapes: shape of inputs to be enqueue on the queue. This is same as
        the value of `nest.flatten(iterator.output_shapes)`.
      iterations: integer, number of iterations to be run; determines the
        number of batches to be enqueued.

    Returns:
      while_loop_op running `iterations` times; in each run we enqueue a batch
      on the infeed queue from the host with id `host_id` for each device shard.
    """
    host = self.get_host_cpu_device(host_id)

    def _infeed_enqueue_ops_fn():
      """Enqueue ops for one iteration."""
      control_deps = []
      sharded_inputs = []
      enqueue_ops = []

      with ops.device(host):
        for _ in range(self.num_towers_per_host):
          # Use control dependencies to ensure a deterministic ordering.
          with ops.control_dependencies(control_deps):
            inputs = nest.flatten(iterator.get_next())
            control_deps.extend(inputs)
            sharded_inputs.append(inputs)

      for core_id, shard_input in enumerate(sharded_inputs):
        enqueue_ops.append(
            tpu_ops.infeed_enqueue_tuple(
                inputs=shard_input,
                shapes=input_shapes,
                device_ordinal=core_id))
      return enqueue_ops

    def enqueue_ops_loop_body(i):
      """Callable for the loop body of the while_loop instantiated below."""
      with ops.control_dependencies(_infeed_enqueue_ops_fn()):
        return i + 1

    with ops.device(host):
      enqueue_op_per_host = control_flow_ops.while_loop(
          lambda i: i < iterations,
          enqueue_ops_loop_body,
          [constant_op.constant(0)],
          parallel_iterations=1)

    return enqueue_op_per_host

  def distribute_dataset(self, dataset_fn):
    # TODO(priyag): Perhaps distribute across cores here.
    return self._call_dataset_fn(dataset_fn)

  # TODO(priyag): Deal with OutOfRange errors once b/111349762 is fixed.
  # TODO(sourabhbajaj): Remove the initial_loop_values parameter when we have
  # a mechanism to infer the outputs of `fn`. Pending b/110550782.
  def _run_steps_on_dataset(self, fn, iterator, iterations,
                            initial_loop_values=None):

    shapes = nest.flatten(iterator.output_shapes)
    if any([not s.is_fully_defined() for s in shapes]):
      raise ValueError(
          'TPU currently requires fully defined shapes. Either use '
          'set_shape() on the input tensors or use '
          'dataset.apply(map_and_batch(..., drop_remainder=True)).')
    types = nest.flatten(iterator.output_types)

    enqueue_ops = [
        self._get_enqueue_op_per_host(host_id, iterator, shapes, iterations)
        for host_id in range(self.num_hosts)]

    def dequeue_fn():
      dequeued = tpu_ops.infeed_dequeue_tuple(dtypes=types, shapes=shapes)
      return nest.pack_sequence_as(iterator.output_shapes, dequeued)

    # Wrap `fn` for repeat.
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)
    ctx = values.MultiStepContext()
    def run_fn(*args, **kwargs):
      """Single step on the TPU device."""
      del args, kwargs
      fn_inputs = dequeue_fn()
      if not isinstance(fn_inputs, tuple):
        fn_inputs = (fn_inputs,)
      fn_result = fn(ctx, *fn_inputs)
      flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
      if flat_last_step_outputs:
        with ops.control_dependencies([fn_result]):
          return [array_ops.identity(f) for f in flat_last_step_outputs]
      else:
        return fn_result

    # TODO(sourabhbajaj): The input to while loop should be based on the output
    # type of the step_fn
    def iterate_on_tpu():
      return training_loop.repeat(iterations, run_fn, initial_loop_values)

    # We capture the control_flow_context at this point, before we run `fn`
    # inside a while_loop and TPU replicate context. This is useful in cases
    # where we might need to exit these contexts and get back to the outer
    # context to do some things, for e.g. create an op which should be
    # evaluated only once at the end of the loop on the host. One such usage
    # is in creating metrics' value op.
    self._outer_control_flow_context = (
        ops.get_default_graph()._get_control_flow_context())  # pylint: disable=protected-access

    replicate_inputs = [[]] * self.num_towers
    replicate_outputs = tpu.replicate(iterate_on_tpu, replicate_inputs)
    del self._outer_control_flow_context
    ctx.run_op = control_flow_ops.group(replicate_outputs, enqueue_ops)

    # Filter out any ops from the outputs, typically this would be the case
    # when there were no tensor outputs.
    last_step_tensor_outputs = [x for x in replicate_outputs
                                if not isinstance(x, ops.Operation)]

    # Outputs are currently of the structure (grouped by device)
    # [[output0_device0, output1_device0, output2_device0],
    #  [output0_device1, output1_device1, output2_device1]]
    # Convert this to the following structure instead: (grouped by output)
    # [[output0_device0, output0_device1],
    #  [output1_device0, output1_device1],
    #  [output2_device0, output2_device1]]
    last_step_tensor_outputs = [list(x) for x in zip(*last_step_tensor_outputs)]

    # Convert replicate_outputs to the original dict structure of
    # last_step_outputs.
    last_step_tensor_outputs_dict = nest.pack_sequence_as(
        ctx.last_step_outputs, last_step_tensor_outputs)

    for (name, aggregation) in ctx._last_step_outputs_aggregations.items():  # pylint: disable=protected-access
      output = last_step_tensor_outputs_dict[name]
      # For outputs that have already been aggregated, take the first value
      # from the list as each value should be the same. Else return the full
      # list of values.
      if aggregation is not variables_lib.VariableAggregation.NONE:
        # TODO(priyag): Should this return the element or a list with 1 element
        last_step_tensor_outputs_dict[name] = output[0]
    ctx._set_last_step_outputs(last_step_tensor_outputs_dict)  # pylint: disable=protected-access

    return ctx

  def _call_for_each_tower(self, fn, *args, **kwargs):
    kwargs.pop('run_concurrently', None)
    with one_device_strategy._OneDeviceTowerContext(self):  # pylint: disable=protected-access
      return fn(*args, **kwargs)

  def initialize(self):
    if context.executing_eagerly():
      # TODO(priyag): Add appopriate call here when eager is supported for TPUs.
      raise NotImplementedError('Eager mode not supported in TPUStrategy.')
    else:
      return [tpu.initialize_system()]

  def finalize(self):
    if context.executing_eagerly():
      # TODO(priyag): Add appopriate call here when eager is supported for TPUs.
      raise NotImplementedError('Eager mode not supported in TPUStrategy.')
    else:
      return [tpu.shutdown_system()]

  def _reduce(self, aggregation, value, destinations):
    graph = ops.get_default_graph()
    cf_context = graph._get_control_flow_context()  # pylint: disable=protected-access
    # If we're inside the ReplicateContext, reduction should be done using
    # CrossReplicaSum while outside we can directly use an add_n op.
    while cf_context:
      if isinstance(cf_context, tpu.TPUReplicateContext):
        if aggregation == vs.VariableAggregation.MEAN:
          # TODO(jhseu):  Revisit once we support model-parallelism.
          value *= (1. / self.num_towers)
        elif aggregation != vs.VariableAggregation.SUM:
          raise NotImplementedError(
              'Currently only support sum & mean in TPUStrategy.')
        return tpu_ops.cross_replica_sum(value)
      cf_context = cf_context.outer_context

    # Validate that the destination is same as the host device
    # Note we don't do this when in replicate context as the reduction is
    # performed on the TPU device itself.
    devices = cross_tower_ops_lib.get_devices_from(destinations)
    if len(devices) == 1:
      assert device_util.canonicalize(devices[0]) == device_util.canonicalize(
          self.get_host_cpu_device(0))
    else:
      raise ValueError('Multiple devices are not supported for TPUStrategy')

    if aggregation == vs.VariableAggregation.ONLY_FIRST_TOWER:
      return value[0]
    output = math_ops.add_n(value)
    if aggregation == vs.VariableAggregation.MEAN:
      return output * (1. / len(value))
    return output

  def _unwrap(self, value):
    if isinstance(value, list):
      return value
    return [value]

  @property
  def num_towers(self):
    return self._num_cores_override or self._tpu_metadata.num_cores

  @property
  def num_hosts(self):
    return self._tpu_metadata.num_hosts

  @property
  def num_towers_per_host(self):
    return self._tpu_metadata.num_of_cores_per_host

  def get_host_cpu_device(self, host_id):
    if self._tpu_cluster_resolver.get_master() in ('', 'local'):
      return '/replica:0/task:0/device:CPU:0'
    job_name = self._tpu_cluster_resolver.get_job_name() or 'tpu_worker'
    return '/job:%s/task:%d/device:CPU:0' % (job_name, host_id)

  def configure(self,
                session_config=None,
                cluster_spec=None,
                task_type=None,
                task_id=None):
    del cluster_spec, task_type, task_id
    if session_config:
      session_config.isolate_session_state = True
      cluster_spec = self._tpu_cluster_resolver.cluster_spec()
      if cluster_spec:
        session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

