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
"""Strategy combinations for combinations.combine()."""

import sys
import unittest
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import cluster_resolver
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import one_device_strategy as one_device_lib
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util as framework_test_util
from tensorflow.python.platform import flags
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export

_did_connect_to_cluster = False
_topology = None
CollectiveAllReduceExtended = (
    collective_all_reduce_strategy.CollectiveAllReduceExtended
)


def _version_chooser(tf1_cls, tf2_cls):
  def creator(*args, **kwargs):
    if tf2.enabled():
      return tf2_cls(*args, **kwargs)
    return tf1_cls(*args, **kwargs)

  return creator


MirroredStrategy = _version_chooser(
    mirrored_lib.MirroredStrategyV1, mirrored_lib.MirroredStrategy
)
CentralStorageStrategy = _version_chooser(
    central_storage_strategy.CentralStorageStrategyV1,
    central_storage_strategy.CentralStorageStrategy,
)
OneDeviceStrategy = _version_chooser(
    one_device_lib.OneDeviceStrategyV1, one_device_lib.OneDeviceStrategy
)
# Only V2 CollectiveAllReduceStrategy combinations are supported.
CollectiveAllReduceStrategy = (
    collective_all_reduce_strategy.CollectiveAllReduceStrategy
)


# pylint: disable=missing-docstring
def _get_tpu_strategy_creator(
    steps_per_run,
    use_single_core=False,
    enable_packed_variable=False,
    enable_spmd_xla_paritioning=False,
    **kwargs
):
  def _create_tpu_strategy():
    FLAGS = flags.FLAGS  # pylint: disable=invalid-name
    global _did_connect_to_cluster
    global _topology

    try:
      # Attempt to locally discover the TPU. This will fail for Cloud TPU, in
      # which case we fall back to the values passed as flags.
      resolver = tpu_cluster_resolver.TPUClusterResolver()
      did_automatically_resolve = True
    except ValueError:
      did_automatically_resolve = False

      # These flags will be defined by tpu_test_wrapper.py.
      resolver = tpu_cluster_resolver.TPUClusterResolver(
          tpu=hasattr(FLAGS, "tpu") and FLAGS.tpu or "",
          zone=hasattr(FLAGS, "zone") and FLAGS.zone or None,
          project=hasattr(FLAGS, "project") and FLAGS.project or None,
      )

    # Only connect once per process, rather than per test method.
    if not _did_connect_to_cluster:
      if getattr(FLAGS, "tpu", "") or did_automatically_resolve:
        remote.connect_to_cluster(resolver)
        _did_connect_to_cluster = True
      _topology = tpu_cluster_resolver.initialize_tpu_system(resolver)

    device_assignment = None
    if use_single_core:
      device_assignment = device_assignment_lib.DeviceAssignment(
          _topology,
          core_assignment=device_assignment_lib.SINGLE_CORE_ASSIGNMENT,
      )

    # Steps per run is only supported in TF 1.x
    if tf2.enabled():
      strategy = tpu_lib.TPUStrategyV2(
          resolver,
          device_assignment,
          experimental_spmd_xla_partitioning=enable_spmd_xla_paritioning,
          **kwargs
      )
    else:
      strategy = tpu_lib.TPUStrategyV1(
          resolver, steps_per_run, device_assignment, **kwargs
      )
    if enable_packed_variable and enable_spmd_xla_paritioning:
      raise ValueError("Packed Variable is not compatible with SPMD mode")
    strategy._enable_packed_variable_in_eager_mode = (  # pylint: disable=protected-access
        enable_packed_variable
    )
    return strategy

  return _create_tpu_strategy


def _mirrored_strategy_with_collective_key_base(devices):
  required_cpus_nums = sum(
      1
      for d in devices
      if tf_device.DeviceSpec.from_string(d).device_type == "CPU"
  )

  # If required virtual CPUs are not setup yet, config the logical devices.
  if required_cpus_nums > len(context.context().list_logical_devices("CPU")):
    context._reset_context()  # pylint: disable=protected-access
    test_util.set_logical_devices_to_at_least("CPU", required_cpus_nums)

  # Increase collective base key to avoid key collision across subtests.
  mirrored_lib.MirroredStrategyV1._collective_key_base += 100000
  mirrored_lib.MirroredStrategy._collective_key_base += 100000
  return MirroredStrategy(devices)


def _mirrored_strategy_with_no_merge_call(devices):
  mirrored_lib.MirroredStrategyV1._collective_key_base += 100000
  mirrored_lib.MirroredStrategy._collective_key_base += 100000
  out = MirroredStrategy(devices)
  # Stub out merge call usage.
  out.extended._use_merge_call = lambda: False  # pylint: disable=protected-access
  return out


def _get_multi_worker_mirrored_creator(required_gpus, use_merge_call=True):
  def _create_multi_worker_mirrored():
    tf_config = cluster_resolver.TFConfigClusterResolver()
    master = tf_config.master()
    if tf_config.rpc_layer:
      # Strip off the rpc_layer suffix.
      master = master[len("%s://" % tf_config.rpc_layer) :]
    resolver = cluster_resolver.SimpleClusterResolver(
        cluster_spec=tf_config.cluster_spec(),
        task_type=tf_config.task_type,
        task_id=tf_config.task_id,
        master=master,
        environment=tf_config.environment,
        num_accelerators={"GPU": required_gpus},
        rpc_layer=tf_config.rpc_layer or "grpc",
    )
    # Disable health check and coordination service. We don't have a reliable
    # way to shutdown the strategy (and thus the strategy health check or
    # coordination service heartbeat) at the end of a test. Turning on the
    # strategy health check or coordination service heartbeat causes some
    # flakiness since we re-create part of the server when creating a strategy,
    # and our tests are capable of handling failures.
    CollectiveAllReduceExtended._enable_check_health = (  # pylint: disable=protected-access
        False
    )
    context.context().configure_coordination_service(service_type="")
    # Always create the strategy in eager mode so that it starts the server and
    # configures the eager context. The eager context can no longer be
    # configured after initialization.
    with context.eager_mode():
      strategy = CollectiveAllReduceStrategy(cluster_resolver=resolver)

    if not use_merge_call:
      strategy.extended._use_merge_call = lambda: False  # pylint: disable=protected-access
    # TODO(b/152320929): Wait for the cluster before proceeding, otherwise
    # collectives may hang if any worker launches collectives before the chief
    # creates the strategy.
    try:
      multi_process_runner.get_barrier().wait()
    except ValueError:
      # If the creator is called in the main process,
      # multi_process_runner.get_barrier() raises ValueError, which is safe to
      # ignore.
      pass
    return strategy

  def skip_if_cannot_start_grpc_server():
    try:
      return _create_multi_worker_mirrored()
    except errors.UnknownError as e:
      if "Could not start gRPC server" in e.message and (
          len(sys.argv) >= 1 and "bazel" in sys.argv[0]
      ):
        raise unittest.SkipTest("Cannot start std servers.")
      else:
        raise

  return skip_if_cannot_start_grpc_server


# Due to b/195615322, FixedShardsPartitioner will wrongly partition
# RNG state, so we use MinSizePartitioner as the default. Maximum RNG
# state size is int64[3] which is 8 * 3 bytes, so we set
# min_shard_bytes to 8 * 3 + 1.
DEFAULT_PARTITIONER = sharded_variable.MinSizePartitioner(
    min_shard_bytes=8 * 3 + 1, max_shards=2
)


def _get_ps_strategy_creator(
    num_workers,
    num_ps,
    required_gpus=0,
    variable_partitioner=DEFAULT_PARTITIONER,
):
  def _create_ps_strategy(resolver, variable_partitioner):
    return parameter_server_strategy_v2.ParameterServerStrategyV2(
        resolver, variable_partitioner=variable_partitioner
    )

  def _create_parameter_server():
    if framework_test_util.is_xla_enabled():
      # To address test failures resulting in XLA with MultiProcessRunner,
      # continue to use in-process cluster for XLA tests.
      cluster_def = multi_worker_test_base.create_in_process_cluster(
          num_workers=num_workers, num_ps=num_ps, rpc_layer="grpc"
      )
      resolver = cluster_resolver.SimpleClusterResolver(
          server_lib.ClusterSpec(cluster_def),
          num_accelerators={"GPU": required_gpus},
          rpc_layer="grpc",
      )
      return _create_ps_strategy(resolver, variable_partitioner)
    else:
      tf_config = cluster_resolver.TFConfigClusterResolver()
      cluster_def = tf_config.cluster_spec().as_dict()
      if not cluster_def:
        # When MultiProcessRunner cluster is used, the cluster is not created
        # initially when the decorator is called. When the test runs, initially
        # this method is invoked via decorator before setting up the
        # MultiProcessRunner with worker and ps in the combinations.py. After
        # setup is done, the subprocess invokes this method again to get
        # strategy object. We return None strategy when the main thread invokes
        # this method before setting up cluster.
        # Returning None is fine here, since this thread will proceed to create
        # MultiProcessRunner and invoke tests with decorator inside
        # subprocesses.
        return None
      # MultiProcessRunner is already setup and this method is invoked from a
      # subprocess running the actual test.
      resolver = cluster_resolver.SimpleClusterResolver(
          server_lib.ClusterSpec(cluster_def),
          num_accelerators={"GPU": required_gpus},
          task_type=tf_config.task_type,
          task_id=tf_config.task_id,
          environment=tf_config.environment,
          rpc_layer=tf_config.rpc_layer or "grpc",
      )
      if tf_config.task_type in ("worker", "ps"):
        worker_config = config_pb2.ConfigProto()
        worker_config.inter_op_parallelism_threads = 4  # max num_workers + 1

        try:
          server = server_lib.Server(
              cluster_def,
              job_name=tf_config.task_type,
              task_index=tf_config.task_id,
              protocol="grpc",
              config=worker_config,
          )
        except errors.UnknownError as e:
          if "Could not start gRPC server" in e.message:
            raise unittest.SkipTest("Cannot start std servers.")
          else:
            raise

        # Blocking the process that starts a server from exiting.
        server.join()

      return _create_ps_strategy(resolver, variable_partitioner)

  return _create_parameter_server


def _deferred_pool_runner(
    has_chief, num_workers, initializer=None, share_gpu=True
):
  """Returns a callable that returns the pool runner.

  It creates the pool runner only upon first invocation. This avoids creating it
  when this file is imported.

  Args:
    has_chief: whether there should be a chief.
    num_workers: the number of workers excluding the chief.
    initializer: initializer of each process.
    share_gpu: whether to share GPU between the workers.

  Returns:
    A callable that returns the runner.
  """

  container = []

  def get_or_create():
    if not container:
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief, num_workers=num_workers, num_ps=0, has_eval=False
      )
      runner = multi_process_runner.MultiProcessPoolRunner(
          cluster_spec, initializer=initializer, share_gpu=share_gpu
      )
      container.append(runner)
    return container[0]

  return get_or_create


# We need to create the strategy in the initializer to start the server before
# any test runs.
_two_worker_pool = _deferred_pool_runner(
    has_chief=True,
    num_workers=1,
    initializer=_get_multi_worker_mirrored_creator(required_gpus=0),
)

# Two-worker pool where each worker gets it's own GPU. Useful for testing MWMS
# on a single host.
_two_worker_pool_noshare = _deferred_pool_runner(
    has_chief=True,
    num_workers=1,
    initializer=_get_multi_worker_mirrored_creator(required_gpus=0),
    share_gpu=False,
)
_four_worker_pool = _deferred_pool_runner(
    has_chief=True,
    num_workers=3,
    initializer=_get_multi_worker_mirrored_creator(required_gpus=0),
)

# pylint: disable=g-long-lambda
default_strategy = combinations.NamedDistribution(
    "Default",
    distribute_lib._get_default_strategy,  # pylint: disable=protected-access
    required_gpus=None,
)
one_device_strategy = combinations.NamedDistribution(
    "OneDeviceCPU", lambda: OneDeviceStrategy("/cpu:0"), required_gpus=None
)
one_device_strategy_gpu = combinations.NamedDistribution(
    "OneDeviceGPU", lambda: OneDeviceStrategy("/gpu:0"), required_gpus=1
)
one_device_strategy_on_worker_1 = combinations.NamedDistribution(
    "OneDeviceOnWorker1CPU",
    lambda: OneDeviceStrategy("/job:worker/replica:0/task:1/cpu:0"),
    required_gpus=None,
)
one_device_strategy_gpu_on_worker_1 = combinations.NamedDistribution(
    "OneDeviceOnWorker1GPU",
    lambda: OneDeviceStrategy("/job:worker/replica:0/task:1/gpu:0"),
    required_gpus=1,
)
tpu_strategy = combinations.NamedDistribution(
    "TPU", _get_tpu_strategy_creator(steps_per_run=2), required_tpu=True
)
tpu_strategy_packed_var = combinations.NamedDistribution(
    "TPUPackedVar",
    _get_tpu_strategy_creator(steps_per_run=2, enable_packed_variable=True),
    required_tpu=True,
)
tpu_strategy_spmd = combinations.NamedDistribution(
    "TPUUseSPMD",
    _get_tpu_strategy_creator(
        steps_per_run=2, enable_spmd_xla_paritioning=True
    ),
    required_tpu=True,
)
tpu_strategy_one_step = combinations.NamedDistribution(
    "TPUOneStep", _get_tpu_strategy_creator(steps_per_run=1), required_tpu=True
)
tpu_strategy_one_core = combinations.NamedDistribution(
    "TPUOneCore",
    _get_tpu_strategy_creator(steps_per_run=2, use_single_core=True),
    required_tpu=True,
)
tpu_strategy_one_step_one_core = combinations.NamedDistribution(
    "TPUOneStepOneCore",
    _get_tpu_strategy_creator(steps_per_run=1, use_single_core=True),
    required_tpu=True,
)
cloud_tpu_strategy = combinations.NamedDistribution(
    "CloudTPU",
    _get_tpu_strategy_creator(steps_per_run=2),
    required_tpu=True,
    use_cloud_tpu=True,
)
mirrored_strategy_with_one_cpu = combinations.NamedDistribution(
    "Mirrored1CPU",
    lambda: _mirrored_strategy_with_collective_key_base(["/cpu:0"]),
)
mirrored_strategy_with_one_gpu = combinations.NamedDistribution(
    "Mirrored1GPU",
    lambda: _mirrored_strategy_with_collective_key_base(["/gpu:0"]),
    required_gpus=1,
)
mirrored_strategy_with_gpu_and_cpu = combinations.NamedDistribution(
    "MirroredCPUAndGPU",
    lambda: _mirrored_strategy_with_collective_key_base(["/gpu:0", "/cpu:0"]),
    required_gpus=1,
)
mirrored_strategy_with_two_cpus = combinations.NamedDistribution(
    "Mirrored2CPUs",
    lambda: _mirrored_strategy_with_collective_key_base(["/cpu:0", "/cpu:1"]),
    required_gpus=0,
)
mirrored_strategy_with_two_gpus = combinations.NamedDistribution(
    "Mirrored2GPUs",
    lambda: _mirrored_strategy_with_collective_key_base(["/gpu:0", "/gpu:1"]),
    required_gpus=2,
)
mirrored_strategy_with_two_gpus_no_merge_call = combinations.NamedDistribution(
    "Mirrored2GPUsNoMergeCall",
    lambda: _mirrored_strategy_with_no_merge_call(["/gpu:0", "/gpu:1"]),
    required_physical_gpus=2,
)
# Should call set_virtual_cpus_to_at_least(3) in your test's setUp methods.
# Deprecated, use mirrored_strategy_with_two_cpus instead.
mirrored_strategy_with_cpu_1_and_2 = combinations.NamedDistribution(
    "Mirrored2CPU",
    lambda: _mirrored_strategy_with_collective_key_base(["/cpu:1", "/cpu:2"]),
)
mirrored_strategy_with_cpu_1_and_2.__doc__ = (
    """Mirrored strategy with 2 virtual CPUs.

    Should set up logical devices before use
    """
)
central_storage_strategy_with_two_gpus = combinations.NamedDistribution(
    "CentralStorage2GPUs",
    lambda: CentralStorageStrategy(["/gpu:0", "/gpu:1"]),
    required_gpus=2,
)
central_storage_strategy_with_gpu_and_cpu = combinations.NamedDistribution(
    "CentralStorageCPUAndGPU",
    lambda: CentralStorageStrategy(["/gpu:0", "/cpu:0"]),
    required_gpus=1,
)
# chief + 1 worker, with CPU.
multi_worker_mirrored_2x1_cpu = combinations.NamedDistribution(
    "MultiWorkerMirrored2x1CPU",
    _get_multi_worker_mirrored_creator(required_gpus=0),
    has_chief=True,
    num_workers=1,
    pool_runner_fn=_two_worker_pool,
    no_xla=True,
)
# chief + 1 worker, with 1 GPU each.
multi_worker_mirrored_2x1_gpu = combinations.NamedDistribution(
    "MultiWorkerMirrored2x1GPU",
    _get_multi_worker_mirrored_creator(required_gpus=1),
    has_chief=True,
    num_workers=1,
    required_gpus=1,
    pool_runner_fn=_two_worker_pool,
    share_gpu=False,
)

# Same as above, but not sharing the GPU between the workers.
multi_worker_mirrored_2x1_gpu_noshare = combinations.NamedDistribution(
    "MultiWorkerMirrored2x1GPUNoShare",
    _get_multi_worker_mirrored_creator(required_gpus=1),
    has_chief=True,
    num_workers=1,
    required_gpus=1,
    pool_runner_fn=_two_worker_pool_noshare,
    share_gpu=False,
)
# chief + 1 worker, with 2 GPU each.
multi_worker_mirrored_2x2_gpu = combinations.NamedDistribution(
    "MultiWorkerMirrored2x2GPU",
    _get_multi_worker_mirrored_creator(required_gpus=2),
    has_chief=True,
    num_workers=1,
    required_gpus=2,
    pool_runner_fn=_two_worker_pool,
    no_xla=True,
)
multi_worker_mirrored_2x2_gpu_no_merge_call = combinations.NamedDistribution(
    "MultiWorkerMirrored2x2GPUNoMergeCall",
    _get_multi_worker_mirrored_creator(required_gpus=2, use_merge_call=False),
    has_chief=True,
    num_workers=1,
    required_physical_gpus=2,
    pool_runner_fn=_two_worker_pool,
    no_xla=True,
)
# chief + 3 workers, with CPU.
multi_worker_mirrored_4x1_cpu = combinations.NamedDistribution(
    "MultiWorkerMirrored4x1CPU",
    _get_multi_worker_mirrored_creator(required_gpus=0),
    has_chief=True,
    num_workers=3,
    pool_runner_fn=_four_worker_pool,
    no_xla=True,
)


def parameter_server_strategy_fn(
    name,
    num_workers,
    num_ps,
    required_gpus=0,
    variable_partitioner=DEFAULT_PARTITIONER,
):
  return combinations.NamedDistribution(
      name,
      _get_ps_strategy_creator(
          num_workers=num_workers,
          num_ps=num_ps,
          required_gpus=required_gpus,
          variable_partitioner=variable_partitioner,
      ),
      required_gpus=required_gpus,
      num_workers=num_workers,
      has_chief=True,
      num_ps=num_ps,
  )


parameter_server_strategy_3worker_2ps_cpu = parameter_server_strategy_fn(
    "ParameterServer3Worker2PSCPU", num_workers=3, num_ps=2
)
parameter_server_strategy_1worker_2ps_cpu = parameter_server_strategy_fn(
    "ParameterServer1Worker2PSCPU", num_workers=1, num_ps=2
)
parameter_server_strategy_3worker_2ps_1gpu = parameter_server_strategy_fn(
    "ParameterServer3Worker2PS1GPU", num_workers=3, num_ps=2, required_gpus=1
)
parameter_server_strategy_1worker_2ps_1gpu = parameter_server_strategy_fn(
    "ParameterServer1Worker2PS1GPU", num_workers=1, num_ps=2, required_gpus=1
)

graph_and_eager_modes = ["graph", "eager"]


# TODO(crccw): remove after tf-nightly picks up the new API.
def set_virtual_cpus_to_at_least(num_virtual_cpus):
  test_util.set_logical_devices_to_at_least("CPU", num_virtual_cpus)


strategies_minus_tpu = [
    default_strategy,
    one_device_strategy,
    one_device_strategy_gpu,
    mirrored_strategy_with_gpu_and_cpu,
    mirrored_strategy_with_two_gpus,
    central_storage_strategy_with_gpu_and_cpu,
]

strategies_minus_default_and_tpu = [
    one_device_strategy,
    one_device_strategy_gpu,
    mirrored_strategy_with_gpu_and_cpu,
    mirrored_strategy_with_two_gpus,
]

tpu_strategies = [
    tpu_strategy,  # steps_per_run=2
    tpu_strategy_one_step,
    tpu_strategy_packed_var,
    cloud_tpu_strategy,
]

all_strategies_minus_default = strategies_minus_default_and_tpu + tpu_strategies

all_strategies = strategies_minus_tpu + tpu_strategies

two_replica_strategies = [
    mirrored_strategy_with_gpu_and_cpu,
    mirrored_strategy_with_two_gpus,
    multi_worker_mirrored_2x1_cpu,
    multi_worker_mirrored_2x1_gpu,
    tpu_strategy,  # steps_per_run=2
    tpu_strategy_one_step,
    central_storage_strategy_with_gpu_and_cpu,
]

four_replica_strategies = [
    multi_worker_mirrored_2x2_gpu,
    multi_worker_mirrored_4x1_cpu,
]

# TODO(b/159831907): replace with two_replica_strategies after the tests using
# it work with MWMS.
multidevice_strategies = [
    mirrored_strategy_with_gpu_and_cpu,
    mirrored_strategy_with_two_gpus,
    tpu_strategy,  # steps_per_run=2
    tpu_strategy_one_step,
]

multiworker_strategies = [
    multi_worker_mirrored_2x1_cpu,
    multi_worker_mirrored_2x1_gpu,
    multi_worker_mirrored_2x2_gpu,
]


def strategy_minus_tpu_combinations():
  return combinations.combine(
      distribution=strategies_minus_tpu, mode=["graph", "eager"]
  )


def tpu_strategy_combinations():
  return combinations.combine(distribution=tpu_strategies, mode=["graph"])


def all_strategy_combinations():
  return strategy_minus_tpu_combinations() + tpu_strategy_combinations()


def all_strategy_minus_default_and_tpu_combinations():
  return combinations.combine(
      distribution=[
          one_device_strategy,
          one_device_strategy_gpu,
          mirrored_strategy_with_gpu_and_cpu,
          mirrored_strategy_with_two_gpus,
      ],
      mode=["graph", "eager"],
  )


def all_strategy_combinations_minus_default():
  return (
      all_strategy_minus_default_and_tpu_combinations()
      + tpu_strategy_combinations()
  )


tf_export(
    "__internal__.distribute.combinations.central_storage_strategy_with_gpu_and_cpu",
    v1=[],
).export_constant(__name__, "central_storage_strategy_with_gpu_and_cpu")
tf_export(
    "__internal__.distribute.combinations.central_storage_strategy_with_two_gpus",
    v1=[],
).export_constant(__name__, "central_storage_strategy_with_two_gpus")
tf_export(
    "__internal__.distribute.combinations.cloud_tpu_strategy", v1=[]
).export_constant(__name__, "cloud_tpu_strategy")
tf_export(
    "__internal__.distribute.combinations.default_strategy", v1=[]
).export_constant(__name__, "default_strategy")
tf_export(
    "__internal__.distribute.combinations.mirrored_strategy_with_cpu_1_and_2",
    v1=[],
).export_constant(__name__, "mirrored_strategy_with_cpu_1_and_2")
tf_export(
    "__internal__.distribute.combinations.mirrored_strategy_with_two_cpus",
    v1=[],
).export_constant(__name__, "mirrored_strategy_with_two_cpus")
tf_export(
    "__internal__.distribute.combinations.mirrored_strategy_with_gpu_and_cpu",
    v1=[],
).export_constant(__name__, "mirrored_strategy_with_gpu_and_cpu")
tf_export(
    "__internal__.distribute.combinations.mirrored_strategy_with_one_cpu", v1=[]
).export_constant(__name__, "mirrored_strategy_with_one_cpu")
tf_export(
    "__internal__.distribute.combinations.mirrored_strategy_with_one_gpu", v1=[]
).export_constant(__name__, "mirrored_strategy_with_one_gpu")
tf_export(
    "__internal__.distribute.combinations.mirrored_strategy_with_two_gpus",
    v1=[],
).export_constant(__name__, "mirrored_strategy_with_two_gpus")
tf_export(
    "__internal__.distribute.combinations.mirrored_strategy_with_two_gpus_no_merge_call",
    v1=[],
).export_constant(__name__, "mirrored_strategy_with_two_gpus_no_merge_call")
tf_export(
    "__internal__.distribute.combinations.multi_worker_mirrored_2x1_cpu", v1=[]
).export_constant(__name__, "multi_worker_mirrored_2x1_cpu")
tf_export(
    "__internal__.distribute.combinations.multi_worker_mirrored_2x1_gpu", v1=[]
).export_constant(__name__, "multi_worker_mirrored_2x1_gpu")
tf_export(
    "__internal__.distribute.combinations.multi_worker_mirrored_2x1_gpu_noshare",
    v1=[],
).export_constant(__name__, "multi_worker_mirrored_2x1_gpu_noshare")
tf_export(
    "__internal__.distribute.combinations.multi_worker_mirrored_2x2_gpu", v1=[]
).export_constant(__name__, "multi_worker_mirrored_2x2_gpu")
tf_export(
    "__internal__.distribute.combinations.multi_worker_mirrored_2x2_gpu_no_merge_call",
    v1=[],
).export_constant(__name__, "multi_worker_mirrored_2x2_gpu_no_merge_call")
tf_export(
    "__internal__.distribute.combinations.one_device_strategy", v1=[]
).export_constant(__name__, "one_device_strategy")
tf_export(
    "__internal__.distribute.combinations.one_device_strategy_gpu", v1=[]
).export_constant(__name__, "one_device_strategy_gpu")
tf_export(
    "__internal__.distribute.combinations.tpu_strategy", v1=[]
).export_constant(__name__, "tpu_strategy")
tf_export(
    "__internal__.distribute.combinations.parameter_server_strategy_3worker_2ps_cpu",
    v1=[],
).export_constant(__name__, "parameter_server_strategy_3worker_2ps_cpu")
tf_export(
    "__internal__.distribute.combinations.parameter_server_strategy_1worker_2ps_cpu",
    v1=[],
).export_constant(__name__, "parameter_server_strategy_1worker_2ps_cpu")
tf_export(
    "__internal__.distribute.combinations.parameter_server_strategy_3worker_2ps_1gpu",
    v1=[],
).export_constant(__name__, "parameter_server_strategy_3worker_2ps_1gpu")
tf_export(
    "__internal__.distribute.combinations.parameter_server_strategy_1worker_2ps_1gpu",
    v1=[],
).export_constant(__name__, "parameter_server_strategy_1worker_2ps_1gpu")
tf_export(
    "__internal__.distribute.combinations.tpu_strategy_one_core", v1=[]
).export_constant(__name__, "tpu_strategy_one_core")
tf_export(
    "__internal__.distribute.combinations.tpu_strategy_packed_var", v1=[]
).export_constant(__name__, "tpu_strategy_packed_var")
