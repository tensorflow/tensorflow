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
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import device_util
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest


def get_tpu_system_metadata(tpu_cluster_resolver):
  """Retrieves TPU system metadata given a TPUClusterResolver."""
  master = tpu_cluster_resolver.master()

  # pylint: disable=protected-access
  cluster_def = (tpu_cluster_resolver.cluster_spec()
                 or server_lib.ClusterSpec({})).as_cluster_def()
  tpu_system_metadata = (
      tpu_system_metadata_lib._query_tpu_system_metadata(
          master,
          cluster_def=cluster_def,
          query_topology=True))

  return tpu_system_metadata


class TPUStrategy(one_device_strategy.OneDeviceStrategy):
  """Experimental TPU distribution strategy implementation."""

  def __init__(self, tpu_cluster_resolver):
    """Initializes the TPUStrategy object.

    Args:
      tpu_cluster_resolver: A tf.contrib.cluster_resolver.TPUClusterResolver,
          which provides information about the TPU cluster.
    """
    # TODO(isaprykin): Generalize the defaults.  They are currently tailored for
    # the unit test.
    super(TPUStrategy, self).__init__('/device:CPU:0')

    self._tpu_cluster_resolver = tpu_cluster_resolver
    self._tpu_metadata = get_tpu_system_metadata(self._tpu_cluster_resolver)

    # TODO(priyag): This should not be hardcoded here.
    self._host = '/device:CPU:0'

  def distribute_dataset(self, dataset_fn):
    # TODO(priyag): Perhaps distribute across cores here.
    return self._call_dataset_fn(dataset_fn)

  # TODO(priyag): Deal with OutOfRange errors.
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

    def enqueue_ops_fn():
      """Enqueue ops for one iteration."""
      control_deps = []
      sharded_inputs = []
      with ops.device(self._host):
        for _ in range(self.num_towers):
          # Use control dependencies to ensure a deterministic ordering.
          with ops.control_dependencies(control_deps):
            inputs = nest.flatten(iterator.get_next())
            control_deps.extend(inputs)
            sharded_inputs.append(inputs)

      enqueue_ops = []
      for core_id, shard_input in enumerate(sharded_inputs):
        enqueue_ops.append(
            tpu_ops.infeed_enqueue_tuple(
                inputs=shard_input, shapes=shapes, device_ordinal=core_id))
      return enqueue_ops

    def enqueue_ops_loop_body(i):
      with ops.control_dependencies(enqueue_ops_fn()):
        return i + 1

    with ops.device(self._host):
      enqueue_ops = control_flow_ops.while_loop(
          lambda i: i < iterations,
          enqueue_ops_loop_body,
          [constant_op.constant(0)],
          parallel_iterations=1)

    def dequeue_fn():
      dequeued = tpu_ops.infeed_dequeue_tuple(dtypes=types, shapes=shapes)
      return nest.pack_sequence_as(iterator.output_shapes, dequeued)

    # Wrap `fn` for repeat.
    if initial_loop_values is None:
      initial_loop_values = []
    ctx = values.MultiStepContext(initial_loop_values)
    def run_fn(*args, **kwargs):
      del args, kwargs
      fn_result = fn(ctx, dequeue_fn())
      if ctx.last_step_outputs is None:
        ctx.last_step_outputs = []
      with ops.control_dependencies([fn_result]):
        return array_ops.identity(ctx.last_step_outputs)

    # TODO(sourabhbajaj): The input to while loop should be based on the output
    # type of the step_fn
    def iterate_on_tpu():
      return training_loop.repeat(iterations, run_fn, [initial_loop_values])

    replicate_inputs = [[]] * self.num_towers
    outputs = tpu.replicate(iterate_on_tpu, replicate_inputs)
    last_step_tensor_outputs = [list(x) for x in zip(*outputs)]

    # Take index [0] of last_step_tensor_outputs as we wrapped
    # initial_loop_values in a list in the `repeat` call.
    return (control_flow_ops.group(last_step_tensor_outputs, enqueue_ops),
            last_step_tensor_outputs[0], ctx)

  def _call_for_each_tower(self, fn, *args, **kwargs):
    kwargs.pop('run_concurrently', None)
    with one_device_strategy._OneDeviceTowerContext(self):  # pylint: disable=protected-access
      return fn(*args, **kwargs)

  def get_initialization_ops(self):
    return [tpu.initialize_system()]

  def get_finalize_ops(self):
    return [tpu.shutdown_system()]

  def _reduce(self, aggregation, value, destinations):
    graph = ops.get_default_graph()
    context = graph._get_control_flow_context()  # pylint: disable=protected-access
    # If we're inside the ReplicateContext, reduction should be done using
    # CrossReplicaSum while outside we can directly use an add_n op.
    while context:
      if isinstance(context, tpu.TPUReplicateContext):
        if aggregation == vs.VariableAggregation.MEAN:
          # TODO(jhseu):  Revisit once we support model-parallelism.
          value *= (1. / self._num_cores_per_host)
        return tpu_ops.cross_replica_sum(value)
      context = context.outer_context

    # Validate that the destination is same as the host device
    # Note we don't do this when in replicate context as the reduction is
    # performed on the TPU device itself.
    devices = cross_tower_ops_lib.get_devices_from(destinations)
    if len(devices) == 1:
      assert device_util.canonicalize(devices[0]) == device_util.canonicalize(
          self._host)
    else:
      raise ValueError('Multiple devices are not supported for TPUStrategy')

    output = math_ops.add_n(value)
    if aggregation == vs.VariableAggregation.MEAN:
      return output * (1. / len(value))
    return output

  @property
  def num_towers(self):
    return self._tpu_metadata.num_of_cores_per_host
