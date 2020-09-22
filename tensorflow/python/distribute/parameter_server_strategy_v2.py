# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Parameter server strategy V2 class.

This is currently under development and the API is subject to change.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import logging
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.training import server_lib
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import tf_inspect


# pylint: disable=protected-access
class ParameterServerStrategyV2(distribute_lib.Strategy):
  """An asynchronous multi-worker parameter server tf.distribute strategy.

  Currently, `ParameterServerStrategyV2` is not supported to be used as a
  standalone tf.distribute strategy. It should be used in conjunction with
  `Client`. Please see `Client` for more information.

  This is currently under development, and the API as well as implementation
  is subject to changes.
  """

  def __init__(self, cluster_resolver, variable_partitioner=None):
    """Initializes the V2 parameter server strategy.

    This also connects to the remote server cluster.

    Args:
      cluster_resolver: a `tf.distribute.cluster_resolver.ClusterResolver`
        object.
      variable_partitioner: a callable with the signature `num_partitions =
        fn(shape, dtype)`, where `num_partitions` is a list/tuple representing
        the number of partitions on each axis, and `shape` and `dtype` are of
        types `tf.TensorShape` and `tf.dtypes.Dtype`. If None, variables will
        not be partitioned. * `variable_partitioner` will be called for all
        variables created under strategy `scope` to instruct how the variables
        should be partitioned. Variables will be partitioned if there are more
        than one partitions along the partitioning axis, otherwise it falls back
        to normal `tf.Variable`. * Only the first / outermost axis partitioning
        is supported, namely, elements in `num_partitions` must be 1 other than
        the first element. * Partitioner like `min_max_variable_partitioner`,
        `variable_axis_size_partitioner` and `fixed_size_partitioner` are also
        supported since they conform to the required signature. * Div partition
        strategy is used to partition variables. Assuming we assign consecutive
        integer ids along the first axis of a variable, then ids are assigned to
        shards in a contiguous manner, while attempting to keep each shard size
        identical. If the ids do not evenly divide the number of shards, each of
        the first several shards will be assigned one more id. For instance, a
        variable whose first dimension is
        13 has 13 ids, and they are split across 5 shards as: `[[0, 1, 2], [3,
          4, 5], [6, 7, 8], [9, 10], [11, 12]]`. * Variables created under
          `strategy.extended.colocate_vars_with` will not be partitioned, e.g,
          optimizer's slot variables.
    """
    self._cluster_resolver = cluster_resolver
    self._extended = ParameterServerStrategyV2Extended(self, cluster_resolver,
                                                       variable_partitioner)
    self._verify_args_and_config(cluster_resolver)
    logging.info(
        "ParameterServerStrategyV2 is initialized with cluster_spec: "
        "%s", cluster_resolver.cluster_spec())

    # TODO(b/167894802): Make chief, worker, and ps names customizable.
    self._connect_to_cluster(client_name="chief")
    super(ParameterServerStrategyV2, self).__init__(self._extended)

  def _connect_to_cluster(self, client_name):
    if client_name in ["worker", "ps"]:
      raise ValueError("Client name should not be 'worker' or 'ps'.")
    cluster_spec = self._cluster_resolver.cluster_spec()
    self._num_workers = len(cluster_spec.as_dict().get("worker", ()))
    self._num_ps = len(cluster_spec.as_dict().get("ps", ()))

    device_filters = server_lib.ClusterDeviceFilters()
    # For any worker, only the devices on PS and chief nodes are visible
    for i in range(self._num_workers):
      device_filters.set_device_filters(
          "worker", i, ["/job:ps", "/job:%s" % client_name])
    # Similarly for any ps, only the devices on workers and chief are visible
    for i in range(self._num_ps):
      device_filters.set_device_filters(
          "ps", i, ["/job:worker", "/job:%s" % client_name])

    # Allow at most one outstanding RPC for each worker at a certain time. This
    # is to simplify worker failure handling in the runtime
    os.environ["TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE"] = "False"

    logging.info("%s is now connecting to cluster with cluster_spec: %r",
                 self.__class__.__name__, cluster_spec)
    remote.connect_to_cluster(
        cluster_spec,
        job_name=client_name,
        protocol=self._cluster_resolver.rpc_layer,
        cluster_device_filters=device_filters)

  def _verify_args_and_config(self, cluster_resolver):
    if not cluster_resolver.cluster_spec():
      raise ValueError("Cluster spec must be non-empty in `cluster_resolver`.")
    if self.extended._num_gpus_per_worker > 1:
      raise NotImplementedError("Multi-gpu is not supported yet.")


class ParameterServerStrategyV2Extended(
    parameter_server_strategy.ParameterServerStrategyExtended):
  """Extended class for ParameterServerStrategyV2.

  Please see `tf.distribute.StrategyExtended` doc for more information.
  """

  def __init__(self, container_strategy, cluster_resolver,
               variable_partitioner):
    """Initialization of ParameterServerStrategyV2Extended."""
    super(ParameterServerStrategyV2Extended, self).__init__(container_strategy)
    self._num_ps = len(cluster_resolver.cluster_spec().as_dict().get("ps", []))
    self._variable_count = 0
    self._variable_partitioner = variable_partitioner

  def _create_variable(self, next_creator, **kwargs):
    """Implements StrategyExtendedV2._create_variable.

    Creates a `Variable` or a `ShardedVariable`. A `ShardedVariable` will be
    created if satisfying all the following criteria:
      1. `self._variable_partitioner` results in more than one partition on the
         first axis.
      2. variable's rank is greater than 0.
      3. variable is not colocated with another variable.
    Otherwise a `Variable` will be created.

    Args:
      next_creator: See `variable_scope.variable_creator_scope`; the next
        creator in the chain.
      **kwargs: Passed through to the next creator.

    Returns:
      A `Variable` or `ShardedVariable`.
    """

    if "colocate_with" in kwargs:  # Never partition colocated_with variables.
      colocate_with = kwargs["colocate_with"]
      # Clear the variable scope to avoid possible conflicts between device
      # scope and colocation scope.
      with ops.device(None):
        with ops.colocate_with(colocate_with):
          var = next_creator(**kwargs)
          logging.debug(
              "Creating variable (name:%s, shape:%r) that colocates with %s",
              var.name, var.shape, kwargs["colocate_with"].name)
          return var

    if self._variable_partitioner is None:
      return self._create_variable_round_robin(next_creator, **kwargs)

    name = kwargs.get("name", None)
    initial_value = kwargs.get("initial_value", None)
    if initial_value is None:
      raise ValueError("initial_value must be specified.")

    # Two cases where initial_value can be a callable:
    #   1. initial_value is passed as a callable, e.g, an `initializer` class.
    #   2. restoring from checkpoint, initial_value is a
    #     "CheckpointInitialValueCallable".
    init_from_fn = callable(initial_value)

    dtype = kwargs.get("dtype", None)
    shape = kwargs.get("shape", None)
    if init_from_fn and (shape is None or dtype is None):
      init_from_fn = False
      initial_value = initial_value()
    if not init_from_fn:
      # The initial_value is created on client, it will need to be sent to
      # PS for variable initialization, which can be inefficient and can
      # potentially hit the 2GB limit on protobuf serialization.
      initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
      dtype = initial_value.dtype
      shape = initial_value.shape
    else:
      shape = tensor_shape.as_shape(shape)

    if shape.rank == 0:  # Skip partitioning rank-0 variable.
      return self._create_variable_round_robin(next_creator, **kwargs)

    num_partitions = self._variable_partitioner(shape=shape, dtype=dtype)
    if not num_partitions or num_partitions[0] == 0 or any(
        v != 1 for v in num_partitions[1:]):
      raise ValueError(
          "variable_partitioner must return a list/tuple whose elements are 1"
          " besides the first element (non-zero), got: %r" % num_partitions)

    if num_partitions[0] == 1:  # no partition
      return self._create_variable_round_robin(next_creator, **kwargs)

    # Use "div" partition strategy to partition the variable.
    num_partitions = min(num_partitions[0], shape[0])
    base = shape[0] // num_partitions
    extra = shape[0] % num_partitions
    # An example: num_partitions=4, shape[0]=10, partitions: [3, 3, 2, 2]
    # offsets: [0, 3, 6, 8, 10]
    offsets = []
    for i in range(num_partitions):
      if i == 0:
        offsets.append(0)
      else:
        prev_shard_size = base + (1 if i - 1 < extra else 0)
        offsets.append(offsets[i - 1] + prev_shard_size)
    offsets.append(shape[0])

    def init_shard_fn(shard_index):
      if not init_from_fn:
        logging.log_if(
            logging.WARNING, _INEFFICIENT_INIT_WARNING % name,
            shard_index == 0 and
            shape.num_elements() > _LARGE_VARIABLE_NUM_ELEMENTS)
        return initial_value[offsets[shard_index]:offsets[shard_index + 1]]
      arg_spec = tf_inspect.getfullargspec(initial_value)
      if ("shard_info" not in arg_spec.args and
          "shard_info" not in arg_spec.kwonlyargs):
        # `initial_value` is a callable that doesn't accept `shard_info`.
        logging.log_if(
            logging.WARNING, _INEFFICIENT_INIT_WARNING % name,
            shard_index == 0 and
            shape.num_elements() > _LARGE_VARIABLE_NUM_ELEMENTS)
        full_value = initial_value()
        return full_value[offsets[shard_index]:offsets[shard_index + 1]]
      else:
        # Memory-efficient way of initializing sharded variable. It requires
        # the `init_fn` to accept a namedtuple `shard_info`.
        component_shape = (offsets[shard_index + 1] -
                           offsets[shard_index],) + shape[1:]
        offsets_all_axes = (offsets[shard_index],) + (0,) * len(shape[1:])
        return initial_value(
            shard_info=trackable.ShardInfo(
                shape=tensor_shape.as_shape(component_shape),
                offset=offsets_all_axes))

    var_list = []
    for i in range(num_partitions):
      kwargs["shape"] = (offsets[i + 1] - offsets[i],) + shape[1:]
      kwargs["initial_value"] = lambda: init_shard_fn(i)
      if name is not None:
        kwargs["name"] = "{}/part_{}".format(name, i)
      var_list.append(self._create_variable_round_robin(next_creator, **kwargs))

    result = sharded_variable.ShardedVariable(var_list)
    return result

  def _create_variable_round_robin(self, next_creator, **kwargs):
    # Clear the colocation scope to avoid possible conflicts between device
    # scope and colocation scope.
    with ops.colocate_with(None, ignore_existing=True):
      with ops.device("/job:ps/task:%d" %
                      (self._variable_count % self._num_ps)):
        var = next_creator(**kwargs)
        logging.debug(
            "Creating variable (name:%s, shape:%r) on /job:ps/task:%d",
            var.name, var.shape, (self._variable_count % self._num_ps))
        self._variable_count += 1
        return var

  def _call_for_each_replica(self, fn, args, kwargs):
    with distribute_lib.ReplicaContext(
        self._container_strategy(),
        replica_id_in_sync_group=constant_op.constant(0, dtypes.int32)):
      # TODO(rchao): Support multi-replica per worker or sync-group.
      return distribute_utils.regroup((fn(*args, **kwargs),))


# The warning that will be logged if the way we initialize sharded variables
# is memory-inefficient.
_INEFFICIENT_INIT_WARNING = (
    "Large variable %s is partitioned but not initialized in a memory-efficient"
    " way. The full value is first being created and then sliced into smaller "
    "values. To reduce the memory footprint, explicitly specify `dtype` and "
    "`shape` when creating variables, and pass a callable to Variable's "
    "`initial_value`. The callable should take only one argument which is a "
    "namedtuple (shape: `tf.TensorShape`, offsets: list/tuple) where shape is "
    "the shape of the component variable, and offsets is the offsets of the "
    "smaller variable on each axis.")

_LARGE_VARIABLE_NUM_ELEMENTS = 1e9
