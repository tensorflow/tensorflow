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
import threading

from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export

ALLOWED_TASK_TYPES = ("chief", "worker", "ps")

cluster_coordinator = LazyLoader(
    "cluster_coordinator", globals(),
    "tensorflow.python.distribute.coordinator.cluster_coordinator"
)

load_context = LazyLoader(
    "load_context", globals(),
    "tensorflow.python.keras.saving.saved_model.load_context"
)


@tf_export("distribute.experimental.ParameterServerStrategy", v1=[])
class ParameterServerStrategyV2(distribute_lib.Strategy):
  """An multi-worker tf.distribute strategy with parameter servers.

  Parameter server training is a common data-parallel method to scale up a
  machine learning model on multiple machines. A parameter server training
  cluster consists of workers and parameter servers. Variables are created on
  parameter servers and they are read and updated by workers in each step.
  By default, workers read and update these variables independently without
  synchronizing with each other. Under this configuration, it is known as
  asynchronous training.

  In TensorFlow 2, we recommend an architecture based on central coordination
  for parameter server training. Each worker and parameter server runs a
  `tf.distribute.Server`, and on top of that, a coordinator task is responsible
  for creating resources on workers and parameter servers, dispatching
  functions, and coordinating the training. The coordinator uses a
  `tf.distribute.experimental.coordinator.ClusterCoordinator` to coordinate the
  cluster, and a `tf.distribute.experimental.ParameterServerStrategy` to define
  variables on parameter servers and computation on workers.

  For the training to work, the coordinator dispatches `tf.function`s to be
  executed on remote workers. Upon receiving requests from the coordinator, a
  worker executes the `tf.function` by reading the variables from parameter
  servers, executing the ops, and updating the variables on the parameter
  servers. Each of the worker only processes the requests from the coordinator,
  and communicates with parameter servers, without direct interactions with
  other workers in the cluster.

  As a result, failures of some workers do not prevent the cluster from
  continuing the work, and this allows the cluster to train with instances that
  can be occasionally unavailable (e.g. preemptible or spot instances). The
  coordinator and parameter servers though, must be available at all times for
  the cluster to make progress.

  Note that the coordinator is not one of the training workers. Instead, it
  creates resources such as variables and datasets, dispatchs `tf.function`s,
  saves checkpoints and so on. In addition to workers, parameter servers and
  the coordinator, an optional evaluator can be run on the side that
  periodically reads the checkpoints saved by the coordinator and runs
  evaluations against each checkpoint.

  `ParameterServerStrategy` is supported with two training APIs: [Custom
  Training Loop (CTL)]
  (https://www.tensorflow.org/tutorials/distribute/custom_training)
  and [Keras Training API, also known as `Model.fit`]
  (https://www.tensorflow.org/tutorials/distribute/keras). CTL is recommended
  when users prefer to define the details of their training loop, and
  `Model.fit` is recommended when users prefer a high-level abstraction and
  handling of training.

  When using a CTL, `ParameterServerStrategy` has to work in conjunction with a
  `tf.distribute.experimental.coordinator.ClusterCoordinator` object.

  When using `Model.fit`, currently only the
  `tf.keras.utils.experimental.DatasetCreator` input type is supported.

  __Example code for coordinator__

  This section provides code snippets that are intended to be run on (the only)
  one task that is designated as the coordinator. Note that `cluster_resolver`,
  `variable_partitioner`, and `dataset_fn` arguments are explained in the
  following "Cluster setup", "Variable partitioning", and "Dataset preparation"
  sections.

  With a CTL,

  ```python
  # Prepare a strategy to use with the cluster and variable partitioning info.
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver=...,
      variable_partitioner=...)
  coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
      strategy=strategy)

  # Prepare a distribute dataset that will place datasets on the workers.
  distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn=...)

  with strategy.scope():
    model = ...
    optimizer, metrics = ...  # Keras optimizer/metrics are great choices
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=2)
    # `load_checkpoint` infers initial epoch from `optimizer.iterations`.
    initial_epoch = load_checkpoint(checkpoint_manager) or 0

  @tf.function
  def worker_fn(iterator):

    def replica_fn(inputs):
      batch_data, labels = inputs
      # calculate gradient, applying gradient, metrics update etc.

    strategy.run(replica_fn, args=(next(iterator),))

  for epoch in range(initial_epoch, num_epoch):
    distributed_iterator = iter(distributed_dataset)  # Reset iterator state.
    for step in range(steps_per_epoch):

      # Asynchronously schedule the `worker_fn` to be executed on an arbitrary
      # worker. This call returns immediately.
      coordinator.schedule(worker_fn, args=(distributed_iterator,))

    # `join` blocks until all scheduled `worker_fn`s finish execution. Once it
    # returns, we can read the metrics and save checkpoints as needed.
    coordinator.join()
    logging.info('Metric result: %r', metrics.result())
    train_accuracy.reset_states()
    checkpoint_manager.save()
  ```

  With `Model.fit`,

  ```python
  # Prepare a strategy to use with the cluster and variable partitioning info.
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver=...,
      variable_partitioner=...)

  # A dataset function takes a `input_context` and returns a `Dataset`
  def dataset_fn(input_context):
    dataset = tf.data.Dataset.from_tensors(...)
    return dataset.repeat().shard(...).batch(...).prefetch(...)

  # With `Model.fit`, a `DatasetCreator` needs to be used.
  input = tf.keras.utils.experimental.DatasetCreator(dataset_fn=...)

  with strategy.scope():
    model = ...  # Make sure the `Model` is created within scope.
  model.compile(optimizer="rmsprop", loss="mse", steps_per_execution=..., ...)

  # Optional callbacks to checkpoint the model, back up the progress, etc.
  callbacks = [tf.keras.callbacks.ModelCheckpoint(...), ...]

  # `steps_per_epoch` is required with `ParameterServerStrategy`.
  model.fit(input, epochs=..., steps_per_epoch=..., callbacks=callbacks)
  ```

  __Example code for worker and parameter servers__

  In addition to the coordinator, there should be tasks designated as
  "worker" or "ps". They should run the following code to start a TensorFlow
  server, waiting for coordinator's requests:

  ```python
  # Provide a `tf.distribute.cluster_resolver.ClusterResolver` that serves
  # the cluster information. See below "Cluster setup" section.
  cluster_resolver = ...

  server = tf.distribute.Server(
      cluster_resolver.cluster_spec(),
      job_name=cluster_resolver.task_type,
      task_index=cluster_resolver.task_id,
      protocol="grpc")

  # Blocking the process that starts a server from exiting.
  server.join()
  ```

  __Cluster setup__

  In order for the tasks in the cluster to know other tasks' addresses,
  a `tf.distribute.cluster_resolver.ClusterResolver` is required to be used
  in coordinator, worker, and ps. The
  `tf.distribute.cluster_resolver.ClusterResolver` is responsible for providing
  the cluster information, as well as the task type and id of the current task.
  See `tf.distribute.cluster_resolver.ClusterResolver` for more information.

  If `TF_CONFIG` environment variable is set, a
  `tf.distribute.cluster_resolver.TFConfigClusterResolver` should be used as
  well.

  Since there are assumptions in
  `tf.distribute.experimental.ParameterServerStrategy` around the naming of the
  task types, "chief", "ps", and "worker" should be used in the
  `tf.distribute.cluster_resolver.ClusterResolver` to refer to the coordinator,
  parameter servers, and workers, respectively.

  The following example demonstrates setting `TF_CONFIG` for the task designated
  as a parameter server (task type "ps") and index 1 (the second task), in a
  cluster with 1 chief, 2 parameter servers, and 3 workers. Note that it needs
  to be set before the use of
  `tf.distribute.cluster_resolver.TFConfigClusterResolver`.

  Example code for cluster setup:
  ```python
  os.environ['TF_CONFIG'] = '''
  {
    "cluster": {
      "chief": ["chief.example.com:2222"],
      "ps": ["ps0.example.com:2222", "ps1.example.com:2222"],
      "worker": ["worker0.example.com:2222", "worker1.example.com:2222",
                 "worker2.example.com:2222"]
    },
    "task": {
      "type": "ps",
      "index": 1
    }
  }
  '''
  ```

  If you prefer to run the same binary for all tasks, you will need to let the
  binary branch into different roles at the beginning of the program:
  ```python
  # If coordinator, create a strategy and start the training program.
  if cluster_resolver.task_type == 'chief':
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    ...

  # If worker/ps, create a server
  elif cluster_resolver.task_type in ("worker", "ps"):
    server = tf.distribute.Server(...)
    ...
  ```
  Alternatively, you can also start a bunch of TensorFlow servers in advance and
  connect to them later. The coordinator can be in the same cluster or on any
  machine that has connectivity to workers and parameter servers. This is
  covered in our guide and tutorial.

  __Variable creation with `strategy.scope()`__

  `tf.distribute.experimental.ParameterServerStrategy` follows the
  `tf.distribute` API contract where variable creation is expected to be inside
  the context manager returned by `strategy.scope()`, in order to be correctly
  placed on parameter servers in a round-robin manner:

  ```python
  # In this example, we're assuming having 3 ps.
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
      strategy=strategy)

  # Variables should be created inside scope to be placed on parameter servers.
  # If created outside scope such as `v1` here, it would be placed on the
  # coordinator.
  v1 = tf.Variable(initial_value=0.0)

  with strategy.scope():
    v2 = tf.Variable(initial_value=1.0)
    v3 = tf.Variable(initial_value=2.0)
    v4 = tf.Variable(initial_value=3.0)
    v5 = tf.Variable(initial_value=4.0)

  # v2 through v5 are created in scope and are distributed on parameter servers.
  # Default placement is round-robin but the order should not be relied on.
  assert v2.device == "/job:ps/replica:0/task:0/device:CPU:0"
  assert v3.device == "/job:ps/replica:0/task:1/device:CPU:0"
  assert v4.device == "/job:ps/replica:0/task:2/device:CPU:0"
  assert v5.device == "/job:ps/replica:0/task:0/device:CPU:0"
  ```

  See `distribute.Strategy.scope` for more information.

  __Variable partitioning__

  Having dedicated servers to store variables means being able to divide up, or
  "shard" the variables across the ps. Partitioning large variable among ps is a
  commonly used technique to boost training throughput and mitigate memory
  constraints. It enables parallel computations and updates on different shards
  of a variable, and often yields better load balancing across parameter
  servers. Without sharding, models with large variables (e.g, embeddings) that
  can't fit into one machine's memory would otherwise be unable to train.

  With `tf.distribute.experimental.ParameterServerStrategy`, if a
  `variable_partitioner` is provided to `__init__` and certain conditions are
  satisfied, the resulting variables created in scope are sharded across the
  parameter servers, in a round-robin fashion. The variable reference returned
  from `tf.Variable` becomes a type that serves as the container of the sharded
  variables. One can access `variables` attribute of this container for the
  actual variable components. If building model with `tf.Module` or Keras,
  the variable components are collected in the `variables` alike attributes.

  It is recommended to use size-based partitioners like
  `tf.distribute.experimental.partitioners.MinSizePartitioner` to avoid
  partitioning small variables, which could have negative impact on model
  training speed.

  ```python
  # Partition the embedding layer into 2 shards.
  variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
      min_shard_bytes=(256 << 10),
      max_shards = 2))
  strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver=...,
    variable_partitioner = variable_partitioner)
  with strategy.scope():
    embedding = tf.keras.layers.Embedding(input_dim=1024, output_dim=1024)
  assert len(embedding.variables) == 2
  assert isinstance(embedding.variables[0], tf.Variable)
  assert isinstance(embedding.variables[1], tf.Variable)
  assert embedding.variables[0].shape == (512, 1024)
  assert embedding.variables[1].shape == (512, 1024)
  ```

  The sharded variable container can be converted to a `Tensor` via
  `tf.convert_to_tensor`. This means the container can be directly used in most
  Python Ops where such `Tensor` conversion automatically happens. For example,
  in the above code snippet, `x * self.w` would implicitly apply the said tensor
  conversion. Note that such conversion can be expensive, as the variable
  components need to be transferred from multiple parameter servers to where
  the value is used.

  `tf.nn.embedding_lookup` on the other hand doesn't apply the tensor
  conversion, and performs parallel lookups on the variable components instead.
  This is crucial to scale up embedding lookups when the embedding table
  variable is large.

  When a partitioned variable is saved to a `SavedModel`, it will be saved as if
  it is one single variable. This improves serving efficiency by eliminating
  a number of Ops that handle the partiton aspects.

  Known limitations of variable partitioning:

  * Number of partitions must not change across Checkpoint saving/loading.

  * After saving partitioned variables to a SavedModel, the SavedModel can't be
    loaded via `tf.saved_model.load`.

  * Partition variable doesn't directly work with `tf.GradientTape`, please use
    the `variables` attributes to get the actual variable components and use
    them in gradient APIs instead.

  __Dataset preparation__

  With `tf.distribute.experimental.ParameterServerStrategy`, a dataset is
  created in each of the workers to be used for training. This is done by
  creating a `dataset_fn` that takes no argument and returns a
  `tf.data.Dataset`, and passing the `dataset_fn` into
  `tf.distribute.experimental.coordinator.
  ClusterCoordinator.create_per_worker_dataset`. We recommend the dataset to be
  shuffled and repeated to have the examples run through the training as evenly
  as possible.

  ```python
  def dataset_fn():
    filenames = ...
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Dataset is recommended to be shuffled, and repeated.
    return dataset.shuffle(buffer_size=...).repeat().batch(batch_size=...)

  coordinator =
      tf.distribute.experimental.coordinator.ClusterCoordinator(strategy=...)
  distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)
  ```

  __Limitations__

  * `tf.distribute.experimental.ParameterServerStrategy` in TF2 is experimental,
  and the API is subject to further changes.

  * When using `Model.fit`, `tf.distribute.experimental.ParameterServerStrategy`
  must be used with a `tf.keras.utils.experimental.DatasetCreator`, and
  `steps_per_epoch` must be specified.
  """

  # pyformat: disable
  def __init__(self, cluster_resolver, variable_partitioner=None):
    """Initializes the TF2 parameter server strategy.

    This initializes the `tf.distribute.experimental.ParameterServerStrategy`
    object to be ready for use with
    `tf.distribute.experimental.coordinator.ClusterCoordinator`.

    Args:
      cluster_resolver: a `tf.distribute.cluster_resolver.ClusterResolver`
        object.
      variable_partitioner:
        a `distribute.experimental.partitioners.Partitioner` that specifies
        how to partition variables. If `None`, variables will not be
        partitioned.

        * Predefined partitioners in `tf.distribute.experimental.partitioners`
        can be used for this argument. A commonly used partitioner is
        `MinSizePartitioner(min_shard_bytes = 256 << 10, max_shards = num_ps)`,
        which allocates at least 256K per shard, and each ps gets at most one
        shard.

        * `variable_partitioner` will be called for each variable created under
        strategy `scope` to instruct how the variable should be partitioned.
        Variables that have only one partition along the partitioning axis
        (i.e., no need for partition) will be created as a normal `tf.Variable`.

        * Only the first / outermost axis partitioning is supported.

        * Div partition strategy is used to partition variables. Assuming we
        assign consecutive integer ids along the first axis of a variable, then
        ids are assigned to shards in a contiguous manner, while attempting to
        keep each shard size identical. If the ids do not evenly divide the
        number of shards, each of the first several shards will be assigned one
        more id. For instance, a variable whose first dimension is 13 has 13
        ids, and they are split across 5 shards as:
        `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

        * Variables created under `strategy.extended.colocate_vars_with` will
        not be partitioned.
    """
    # pyformat: enable
    self._cluster_resolver = cluster_resolver

    self._verify_args_and_config(cluster_resolver)
    self._cluster_coordinator = None
    logging.info(
        "`tf.distribute.experimental.ParameterServerStrategy` is initialized "
        "with cluster_spec: %s", cluster_resolver.cluster_spec())

    # TODO(b/167894802): Make coordinator, worker, and ps names customizable.
    self._connect_to_cluster(coordinator_name="chief")
    self._extended = ParameterServerStrategyV2Extended(self, cluster_resolver,
                                                       variable_partitioner)
    super(ParameterServerStrategyV2, self).__init__(self._extended)
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "ParameterServerStrategy")
    self._should_use_with_coordinator = True
    # Used while constructing distributed iterators.
    self._canonicalize_devices = False

  def _connect_to_cluster(self, coordinator_name):
    if coordinator_name in ["worker", "ps"]:
      raise ValueError("coordinator name should not be 'worker' or 'ps'.")
    cluster_spec = self._cluster_resolver.cluster_spec()
    self._num_workers = len(cluster_spec.as_dict().get("worker", ()))
    self._num_ps = len(cluster_spec.as_dict().get("ps", ()))

    device_filters = server_lib.ClusterDeviceFilters()
    # For any worker, only the devices on ps and coordinator nodes are visible
    for i in range(self._num_workers):
      device_filters.set_device_filters(
          "worker", i, ["/job:ps", "/job:%s" % coordinator_name])
    # Similarly for any ps, only the devices on workers and coordinator are
    # visible
    for i in range(self._num_ps):
      device_filters.set_device_filters(
          "ps", i, ["/job:worker", "/job:%s" % coordinator_name])

    # Allow at most one outstanding RPC for each worker at a certain time. This
    # is to simplify worker failure handling in the runtime
    os.environ["TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE"] = "False"

    # Disable async executors to make context.async_wait a no-op. This avoids
    # sending RPCs to remote workers since the executors used by PSStrategy
    # are known to be always synchronous.
    os.environ["TF_PS_DISABLE_ASYNC_EXECUTOR_GLOBALLY"] = "True"

    logging.info("%s is now connecting to cluster with cluster_spec: %r",
                 self.__class__.__name__, cluster_spec)
    remote.connect_to_cluster(
        cluster_spec,
        job_name=coordinator_name,
        protocol=self._cluster_resolver.rpc_layer,
        cluster_device_filters=device_filters)

    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "ps_strategy_num_workers").set(self._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "ps_strategy_num_ps").set(self._num_ps)

  def _verify_args_and_config(self, cluster_resolver):
    if not cluster_resolver.cluster_spec():
      raise ValueError("Cluster spec must be non-empty in "
                       "`tf.distribute.cluster_resolver.ClusterResolver`.")
    cluster_spec = cluster_resolver.cluster_spec()

    # The following checks if the task types are allowed (chief, ps, worker).
    multi_worker_util._validate_cluster_spec(  # pylint: disable=protected-access
        cluster_spec,
        cluster_resolver.task_type,
        cluster_resolver.task_id)

    if multi_worker_util.task_count(cluster_spec, "ps") < 1:
      raise ValueError("There must be at least one ps.")

    if multi_worker_util.task_count(cluster_spec, "worker") < 1:
      raise ValueError("There must be at least one worker.")


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
    self._num_workers = len(cluster_resolver.cluster_spec().as_dict().get(
        "worker", []))
    self._variable_count = 0

    self._variable_partitioner = variable_partitioner
    # The following two attrs are to verify that `ParameterServerStrategy`
    # methods are properly used with a `ClusterCoordinator`.
    self._used_with_coordinator = False
    self._being_scheduled = False
    self._set_num_gpus()
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_gpus_per_worker").set(self._num_gpus_per_worker)

    # Don't canonicalize the devices here since this code is executed on Chief,
    # but we want the reduce evaluation to be done on each worker. Placer will
    # automatically choose the right device based on current context.
    # TODO(ishark): Use select_cross_device_ops instead.
    self._cross_device_ops = cross_device_ops_lib.ReductionToOneDevice(
        reduce_to_device="/device:CPU:0")
    self._cross_device_ops._canonicalize_devices = False  # pylint: disable=protected-access
    self._allow_run_without_coordinator = False
    self._coordinator_creation_lock = threading.Lock()

  def _set_num_gpus(self):
    devices = config.list_logical_devices("GPU")
    per_worker_gpus = {}
    for d in devices:
      d_spec = tf_device.DeviceSpec.from_string(d.name)
      if d_spec.device_type == "GPU" and d_spec.job == "worker":
        # TODO(b/167894802): update if worker name is customizable
        job_spec = d_spec.replace(device_type=None, device_index=None)
        per_worker_gpus[job_spec] = per_worker_gpus.get(job_spec, 0) + 1

    num_gpus = 0
    for _, count in per_worker_gpus.items():
      if num_gpus > 0 and count != num_gpus:
        raise ValueError("Mismatched number of GPUs per worker")
      num_gpus = count

    self._num_gpus_per_worker = num_gpus
    logging.info(f"Number of GPUs on workers: {self._num_gpus_per_worker}")

  @property
  def _num_replicas_in_sync(self):
    return self._num_gpus_per_worker or 1

  def _create_var_creator(self, next_creator, **kwargs):
    aggregation = kwargs.pop("aggregation", vs.VariableAggregation.NONE)

    def var_creator(**kwargs):
      """Create an AggregatingVariable."""
      # Create and wrap the variable.
      v = next_creator(**kwargs)
      wrapped_v = ps_values.CachingVariable(v)
      wrapped = ps_values.AggregatingVariable(self._container_strategy(),
                                              wrapped_v, aggregation)
      return wrapped

    if self._num_replicas_in_sync > 1:
      if aggregation not in (
          vs.VariableAggregation.NONE,
          vs.VariableAggregation.SUM,
          vs.VariableAggregation.MEAN,
          vs.VariableAggregation.ONLY_FIRST_REPLICA
      ):
        raise ValueError("Invalid variable aggregation mode: " + aggregation +
                         " for variable: " + kwargs["name"])
      return var_creator
    else:
      def variable_creator_single_replica(**kwargs):
        v = next_creator(**kwargs)
        return ps_values.CachingVariable(v)
      return variable_creator_single_replica

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

    var_creator = self._create_var_creator(next_creator, **kwargs)
    if "colocate_with" in kwargs:  # Never partition colocated_with variables.
      colocate_with = kwargs["colocate_with"]
      # Clear the variable scope to avoid possible conflicts between device
      # scope and colocation scope.
      with ops.device(None):
        with ops.colocate_with(colocate_with):
          var = var_creator(**kwargs)
          logging.debug(
              "Creating variable (name:%s, shape:%r) that colocates with %s",
              var.name, var.shape, kwargs["colocate_with"].name)
          return var

    if self._variable_partitioner is None:
      return self._create_variable_round_robin(var_creator, **kwargs)

    name = kwargs.get("name", None)
    initial_value = kwargs.get("initial_value", None)
    if initial_value is None:
      raise ValueError(
          "It looks like you are using `ParameterServerStrategy` with a "
          "`variable_partitioner`, and trying to create a variable without "
          "specifying `initial_value`. This is not allowed. Please specify the "
          "`initial_value`. This can also happen if you are trying to load a "
          "saved_model within a `ParameterServerStrategy` scope. Loading a "
          "saved_model with `variable_partitioner` is not supported.")

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
      # The initial_value is created on coordinator, it will need to be sent to
      # ps for variable initialization, which can be inefficient and can
      # potentially hit the 2GB limit on protobuf serialization.
      initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
      dtype = initial_value.dtype
      shape = initial_value.shape
    else:
      shape = tensor_shape.as_shape(shape)

    if shape.rank == 0:  # Skip partitioning rank-0 variable.
      return self._create_variable_round_robin(var_creator, **kwargs)

    num_partitions = self._variable_partitioner(shape=shape, dtype=dtype)
    if not num_partitions or num_partitions[0] == 0 or any(
        v != 1 for v in num_partitions[1:]):
      raise ValueError(
          "variable_partitioner must return a list/tuple whose elements are 1"
          " besides the first element (non-zero), got: %r" % num_partitions)

    if num_partitions[0] == 1:  # no partition
      return self._create_variable_round_robin(var_creator, **kwargs)

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
            logging.WARN, _INEFFICIENT_INIT_WARNING % name, shard_index == 0 and
            shape.num_elements() > _LARGE_VARIABLE_NUM_ELEMENTS)
        return initial_value[offsets[shard_index]:offsets[shard_index + 1]]
      partition_shape = (offsets[shard_index + 1] -
                         offsets[shard_index],) + shape[1:]
      partition_offset = (offsets[shard_index],) + (0,) * len(shape[1:])
      arg_spec = tf_inspect.getfullargspec(initial_value)
      if ("shard_info" not in arg_spec.args and
          "shard_info" not in arg_spec.kwonlyargs):
        try:
          value = initial_value(
              partition_shape=partition_shape,
              partition_offset=partition_offset)
        except (TypeError, ValueError):
          # TypeError: Initializer doesn't accept kwargs
          # ValueError: Initializer doesn't accept partition kwargs
          # In both cases we go ahead creating the full value and then slice.
          value = initial_value()

        if value.shape == partition_shape:
          # Initializer supports partition: value is the partition value.
          return value
        else:
          # Initializer doesn't support partition: value is the full value
          # and needs to be sliced to get the partition value.
          logging.log_if(
              logging.WARN, _INEFFICIENT_INIT_WARNING % name,
              shard_index == 0 and
              shape.num_elements() > _LARGE_VARIABLE_NUM_ELEMENTS)
          return value[offsets[shard_index]:offsets[shard_index + 1]]
      else:
        # For compatibility with `CheckpointInitialValueCallable`.
        return initial_value(
            shard_info=trackable.ShardInfo(
                shape=tensor_shape.as_shape(partition_shape),
                offset=partition_offset))

    var_list = []
    for i in range(num_partitions):
      kwargs["shape"] = (offsets[i + 1] - offsets[i],) + shape[1:]
      kwargs["initial_value"] = lambda: init_shard_fn(i)
      if name is not None:
        kwargs["name"] = "{}/part_{}".format(name, i)
      var_list.append(self._create_variable_round_robin(var_creator, **kwargs))

    result = sharded_variable.ShardedVariable(var_list)
    return result

  def _create_variable_round_robin(self, next_creator, **kwargs):
    # Clear the colocation scope to avoid possible conflicts between device
    # scope and colocation scope.
    with ops.colocate_with(None, ignore_existing=True):
      # Explicitly set CPU:0 device for PS in case create variable is called
      # inside replica_fn and worker has with GPU:0 scope.
      with ops.device("/job:ps/task:%d/device:CPU:0" %
                      (self._variable_count % self._num_ps)):
        var = next_creator(**kwargs)
        logging.debug(
            "Creating variable (name:%s, shape:%r) on "
            "/job:ps/task:%d/device:CPU:0",
            var.name, var.shape, (self._variable_count % self._num_ps))
        self._variable_count += 1
        return var

  def _resource_creator_scope(self):

    with self._coordinator_creation_lock:
      if not self._container_strategy()._cluster_coordinator:  # pylint: disable=protected-access
        cluster_coordinator.ClusterCoordinator(
            strategy=self._container_strategy())

    # TODO(wxinyi): We should warn the user of the inefficiency of creating
    # `StaticHashTable` inside a `@tf.function`-wrapped `dataset_fn` to be
    # distributed with `distribute_datasets_from_function` and
    # `create_per_worker_dataset`. This is because the `dataset_fn` does not
    # use the same `default_graph` as `scope` to which the
    # `resource_creator_stack` belongs. Thus, `StaticHashTable` creation inside
    # `dataset_fn` is not intercepted. And since its resource creation under a
    # `tf.function` is lifted out, all workers will share the same resource on
    # the coordinator which incurs worker-coordinator communication overhead.

    def lookup_creator(next_creator, *args, **kwargs):
      if load_context.in_load_context:
        return (ps_values.RestoredDistributedTable(
            self._container_strategy(), lambda: next_creator(*args, **kwargs)))  # pylint: disable=protected-access
      else:
        return ps_values.DistributedTable(self._container_strategy(),
                                          lambda: next_creator(*args, **kwargs))  # pylint: disable=protected-access

    def restored_lookup_creator(next_creator, *args, **kwargs):
      return (ps_values.RestoredDistributedTable(
          self._container_strategy(), lambda: next_creator(*args, **kwargs)))  # pylint: disable=protected-access

    return [ops.resource_creator_scope("StaticHashTable", lookup_creator),
            ops.resource_creator_scope("RestoredStaticHashTable",
                                       restored_lookup_creator)]

  def _assert_used_with_cluster_coordinator(self):
    if (not self._used_with_coordinator and
        not self._allow_run_without_coordinator):
      raise NotImplementedError(
          "`tf.distribute.experimental.ParameterServerStrategy` must be used "
          "with `tf.distribute.experimental.coordinator.ClusterCoordinator` in "
          "a custom training loop. If you are using `Model.fit`, please supply "
          "a dataset function directly to a "
          "`tf.keras.utils.experimental.DatasetCreator` instead.")

  def _assert_being_scheduled_by_cluster_coordinator(self):
    if not self._being_scheduled and not self._allow_run_without_coordinator:
      logging.warning(
          "It is detected that a function used with "
          "`tf.distribute.experimental.ParameterServerStrategy` "
          "is executed locally on the coordinator. This is inefficient but may "
          "be valid for one-off tasks such as inferring output signature. "
          "To properly distribute functions to run on workers, `run` or "
          "`reduce` should be used within a function passed to `"
          "tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`."
      )

  # options is not used right now. But we may want to support options while
  # creating InputWorkers in future, similar to MirroredStrategy.
  def _input_workers_with_options(self, options=None):
    input_workers_devices = (
        ("/device:CPU:0", self.worker_devices),)
    return input_lib.InputWorkers(
        input_workers_devices, canonicalize_devices=False)

  def _experimental_distribute_dataset(self, dataset, options):
    input_workers_devices = self._input_workers_with_options()

    # If this DistributedDataset is created outside ClusterCoordinator, i,e,
    # outside a tf.function, we don't build its underlying datasets immediately
    # until it is passed to ClusterCoordinator.create_per_worker_dataset.
    return input_lib.get_distributed_dataset(
        dataset,
        input_workers_devices,
        self._container_strategy(),
        num_replicas_in_sync=self._num_replicas_in_sync,
        options=options,
        build=ops.inside_function())  # will be built by ClusterCoordinator

  def _distribute_datasets_from_function(self, dataset_fn, options):
    # There is no synchronization beyond a worker and thus, the number of
    # input pipelines in sync is only 1 per worker.
    input_pipeline_id_in_sync = 0
    num_input_pipelines_in_sync = 1

    input_context = distribute_lib.InputContext(
        num_input_pipelines=num_input_pipelines_in_sync,
        input_pipeline_id=input_pipeline_id_in_sync,
        num_replicas_in_sync=self._num_replicas_in_sync)

    # If this DistributedDatasetFromFunction is created outside
    # ClusterCoordinator, i,e, outside a tf.function, we don't build its
    # underlying datasets immediately until it is passed to
    # ClusterCoordinator.create_per_worker_dataset.
    return input_lib.get_distributed_datasets_from_function(
        dataset_fn,
        self._input_workers_with_options(options),
        [input_context],
        self._container_strategy(),
        options=options,
        build=ops.inside_function())  # will be built by ClusterCoordinator

  @property
  def worker_devices(self):
    num_gpus = self._num_gpus_per_worker
    if num_gpus > 0:
      compute_devices = tuple("/device:GPU:%d" % (i,) for i in range(num_gpus))
    else:
      compute_devices = ("/device:CPU:0",)
    return compute_devices

  def _call_for_each_replica(self, fn, args, kwargs):
    self._assert_being_scheduled_by_cluster_coordinator()

    return mirrored_run.call_for_each_replica(self._container_strategy(), fn,
                                              args, kwargs)

  def _reduce(self, reduce_op, value):
    self._assert_being_scheduled_by_cluster_coordinator()
    dst = device_util.current() or self._default_device or "/device:CPU:0"
    destinations = device_util.canonicalize_without_job_and_task(dst)
    result = self._local_results(
        self.reduce_to(reduce_op, value, destinations))[0]
    return result

  def _reduce_to(self, reduce_op, value, destinations, options):
    self._assert_being_scheduled_by_cluster_coordinator()

    def get_values(x):
      if isinstance(x, values.DistributedValues):
        return self._cross_device_ops.reduce(
            reduce_op, x, destinations=destinations)  # pylint: disable=protected-access
      return x

    return nest.map_structure(get_values, value)


# The warning that will be logged if the way we initialize sharded variables
# is memory-inefficient.
_INEFFICIENT_INIT_WARNING = (
    "Large variable %s is partitioned but not initialized in a "
    "memory-efficient way. On each shard, the full value is first being "
    "created and then sliced into smaller values. To reduce the memory "
    "footprint, explicitly specify `dtype` and `shape` when creating "
    "variables, and use `tf.initializers` to initialize the variable. "
    "Note that some initializers (e.g., orthogonal) don't support "
    "memory-efficient initialization and there is not much you can do here.")

_LARGE_VARIABLE_NUM_ELEMENTS = 1e9
