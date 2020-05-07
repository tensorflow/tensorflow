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
"""Strategy and optimizer combinations for combinations.combine()."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import tf2
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import one_device_strategy as one_device_lib
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_keras_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_keras_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_keras_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_keras_v2
from tensorflow.python.keras.optimizer_v2 import ftrl as ftrl_keras_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_keras_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_keras_v2
from tensorflow.python.platform import flags
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.training import adagrad
from tensorflow.python.training import adam
from tensorflow.python.training import ftrl
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import rmsprop


FLAGS = flags.FLAGS

_did_connect_to_cluster = False


# pylint: disable=missing-docstring
def _get_tpu_strategy_creator(steps_per_run, use_single_core=False, **kwargs):
  def _create_tpu_strategy():
    global _did_connect_to_cluster

    # These flags will be defined by tpu_test_wrapper.py.
    resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu=hasattr(FLAGS, "tpu") and FLAGS.tpu or "",
        zone=hasattr(FLAGS, "zone") and FLAGS.zone or None,
        project=hasattr(FLAGS, "project") and FLAGS.project or None,
    )
    # Only connect once per process, rather than per test method.
    if hasattr(FLAGS, "tpu") and FLAGS.tpu and not _did_connect_to_cluster:
      remote.connect_to_cluster(resolver)
      _did_connect_to_cluster = True

    topology = tpu_strategy_util.initialize_tpu_system(resolver)
    device_assignment = None
    if use_single_core:
      device_assignment = device_assignment_lib.DeviceAssignment(
          topology, core_assignment=device_assignment_lib.
          SINGLE_CORE_ASSIGNMENT)

    # Steps per run is only supported in TF 1.x
    if tf2.enabled():
      return tpu_lib.TPUStrategy(resolver, device_assignment, **kwargs)
    else:
      return tpu_lib.TPUStrategyV1(resolver, steps_per_run,
                                   device_assignment, **kwargs)
  return _create_tpu_strategy


# pylint: disable=g-long-lambda
default_strategy = combinations.NamedDistribution(
    "Default",
    distribution_strategy_context._get_default_strategy,  # pylint: disable=protected-access
    required_gpus=None)
one_device_strategy = combinations.NamedDistribution(
    "OneDeviceCPU",
    lambda: one_device_lib.OneDeviceStrategy("/cpu:0"),
    required_gpus=None)
one_device_strategy_gpu = combinations.NamedDistribution(
    "OneDeviceGPU",
    lambda: one_device_lib.OneDeviceStrategy("/gpu:0"),
    required_gpus=1)
one_device_strategy_on_worker_1 = combinations.NamedDistribution(
    "OneDeviceOnWorker1CPU",
    lambda: one_device_lib.OneDeviceStrategy("/job:worker/replica:0/task:1/cpu:0"),  # pylint: disable=line-too-long
    required_gpus=None)
one_device_strategy_gpu_on_worker_1 = combinations.NamedDistribution(
    "OneDeviceOnWorker1GPU",
    lambda: one_device_lib.OneDeviceStrategy("/job:worker/replica:0/task:1/gpu:0"),  # pylint: disable=line-too-long
    required_gpus=1)
tpu_strategy = combinations.NamedDistribution(
    "TPU", _get_tpu_strategy_creator(steps_per_run=2), required_tpu=True)
tpu_strategy_one_step = combinations.NamedDistribution(
    "TPUOneStep", _get_tpu_strategy_creator(steps_per_run=1), required_tpu=True)
tpu_strategy_one_core = combinations.NamedDistribution(
    "TPUOneCore",
    _get_tpu_strategy_creator(steps_per_run=2, use_single_core=True),
    required_tpu=True)
tpu_strategy_one_step_one_core = combinations.NamedDistribution(
    "TPUOneStepOneCore",
    _get_tpu_strategy_creator(steps_per_run=1, use_single_core=True),
    required_tpu=True)
cloud_tpu_strategy = combinations.NamedDistribution(
    "CloudTPU",
    _get_tpu_strategy_creator(steps_per_run=2),
    required_tpu=True,
    use_cloud_tpu=True)
mirrored_strategy_with_one_cpu = combinations.NamedDistribution(
    "Mirrored1CPU", lambda: mirrored_lib.MirroredStrategy(["/cpu:0"]))
mirrored_strategy_with_one_gpu = combinations.NamedDistribution(
    "Mirrored1GPU",
    lambda: mirrored_lib.MirroredStrategy(["/gpu:0"]),
    required_gpus=1)
mirrored_strategy_with_gpu_and_cpu = combinations.NamedDistribution(
    "MirroredCPUAndGPU",
    lambda: mirrored_lib.MirroredStrategy(["/gpu:0", "/cpu:0"]),
    required_gpus=1)
mirrored_strategy_with_two_gpus = combinations.NamedDistribution(
    "Mirrored2GPUs",
    lambda: mirrored_lib.MirroredStrategy(["/gpu:0", "/gpu:1"]),
    required_gpus=2)
# Should call set_virtual_cpus_to_at_least(3) in your test's setUp methods.
mirrored_strategy_with_cpu_1_and_2 = combinations.NamedDistribution(
    "Mirrored2CPU", lambda: mirrored_lib.MirroredStrategy(["/cpu:1", "/cpu:2"]))
central_storage_strategy_with_two_gpus = combinations.NamedDistribution(
    "CentralStorage2GPUs",
    lambda: central_storage_strategy.CentralStorageStrategy._from_num_gpus(2),  # pylint: disable=protected-access
    required_gpus=2)
central_storage_strategy_with_gpu_and_cpu = combinations.NamedDistribution(
    "CentralStorageCPUAndGPU",
    lambda: central_storage_strategy.CentralStorageStrategy(
        ["/gpu:0", "/cpu:0"]),
    required_gpus=1)
multi_worker_mirrored_two_workers = combinations.NamedDistribution(
    "MultiWorkerMirrroedTwoWorkers",
    collective_all_reduce_strategy.CollectiveAllReduceStrategy,
    has_chief=False,
    num_workers=2,
)
multi_worker_mirrored_one_chief_one_worker = combinations.NamedDistribution(
    "MultiWorkerMirrroedOneChiefOneWorker",
    collective_all_reduce_strategy.CollectiveAllReduceStrategy,
    has_chief=True,
    num_workers=1,
)

gradient_descent_optimizer_v1_fn = combinations.NamedObject(
    "GradientDescentV1",
    lambda: gradient_descent.GradientDescentOptimizer(0.001))
adagrad_optimizer_v1_fn = combinations.NamedObject(
    "AdagradV1", lambda: adagrad.AdagradOptimizer(0.001))
adam_optimizer_v1_fn = combinations.NamedObject(
    "AdamV1", lambda: adam.AdamOptimizer(0.001, epsilon=1))
ftrl_optimizer_v1_fn = combinations.NamedObject(
    "FtrlV1", lambda: ftrl.FtrlOptimizer(0.001))
rmsprop_optimizer_v1_fn = combinations.NamedObject(
    "RmsPropV1", lambda: rmsprop.RMSPropOptimizer(0.001))

# TODO(shiningsun): consider adding the other v1 optimizers
optimizers_v1 = [
    gradient_descent_optimizer_v1_fn, adagrad_optimizer_v1_fn,
    ftrl_optimizer_v1_fn, rmsprop_optimizer_v1_fn
]

adadelta_optimizer_keras_v2_fn = combinations.NamedObject(
    "AdadeltaKerasV2", lambda: adadelta_keras_v2.Adadelta(0.001))
adagrad_optimizer_keras_v2_fn = combinations.NamedObject(
    "AdagradKerasV2", lambda: adagrad_keras_v2.Adagrad(0.001))
adam_optimizer_keras_v2_fn = combinations.NamedObject(
    "AdamKerasV2", lambda: adam_keras_v2.Adam(0.001, epsilon=1.0))
adamax_optimizer_keras_v2_fn = combinations.NamedObject(
    "AdamaxKerasV2", lambda: adamax_keras_v2.Adamax(0.001, epsilon=1.0))
nadam_optimizer_keras_v2_fn = combinations.NamedObject(
    "NadamKerasV2", lambda: nadam_keras_v2.Nadam(0.001, epsilon=1.0))
ftrl_optimizer_keras_v2_fn = combinations.NamedObject(
    "FtrlKerasV2", lambda: ftrl_keras_v2.Ftrl(0.001))
gradient_descent_optimizer_keras_v2_fn = combinations.NamedObject(
    "GradientDescentKerasV2", lambda: gradient_descent_keras_v2.SGD(0.001))
rmsprop_optimizer_keras_v2_fn = combinations.NamedObject(
    "RmsPropKerasV2", lambda: rmsprop_keras_v2.RMSprop(0.001))

# TODO(shiningsun): consider adding the other v2 optimizers
optimizers_v2 = [
    gradient_descent_optimizer_keras_v2_fn, adagrad_optimizer_keras_v2_fn
]

optimizers_v1_and_v2 = optimizers_v1 + optimizers_v2

graph_and_eager_modes = ["graph", "eager"]


# This function should be called in a test's `setUp` method with the
# maximum value needed in any test.
def set_virtual_cpus_to_at_least(num_virtual_cpus):
  """Create virtual CPU devices if they haven't yet been created."""
  if num_virtual_cpus < 1:
    raise ValueError("`num_virtual_cpus` must be at least 1 not %r" %
                     (num_virtual_cpus,))
  physical_devices = config.list_physical_devices("CPU")
  if not physical_devices:
    raise RuntimeError("No CPUs found")
  configs = config.get_logical_device_configuration(physical_devices[0])
  if configs is None:
    logical_devices = [
        context.LogicalDeviceConfiguration() for _ in range(num_virtual_cpus)
    ]
    config.set_logical_device_configuration(physical_devices[0],
                                            logical_devices)
  else:
    if len(configs) < num_virtual_cpus:
      raise RuntimeError("Already configured with %d < %d virtual CPUs" %
                         (len(configs), num_virtual_cpus))


def distributions_and_v1_optimizers():
  """A common set of combination with DistributionStrategies and Optimizers."""
  return combinations.combine(
      distribution=[
          one_device_strategy,
          mirrored_strategy_with_gpu_and_cpu,
          mirrored_strategy_with_two_gpus,
      ],
      optimizer_fn=optimizers_v1)


def distributions_and_v2_optimizers():
  """A common set of combination with DistributionStrategies and Optimizers."""
  return combinations.combine(
      distribution=[
          one_device_strategy,
          mirrored_strategy_with_gpu_and_cpu,
          mirrored_strategy_with_two_gpus,
      ],
      optimizer_fn=optimizers_v2)


def distributions_and_v1_and_v2_optimizers():
  """A common set of combination with DistributionStrategies and Optimizers."""
  return combinations.combine(
      distribution=[
          one_device_strategy,
          mirrored_strategy_with_gpu_and_cpu,
          mirrored_strategy_with_two_gpus,
      ],
      optimizer_fn=optimizers_v1_and_v2)


strategies_minus_tpu = [
    default_strategy, one_device_strategy, one_device_strategy_gpu,
    mirrored_strategy_with_gpu_and_cpu, mirrored_strategy_with_two_gpus,
    central_storage_strategy_with_gpu_and_cpu
]

strategies_minus_default_and_tpu = [
    one_device_strategy, one_device_strategy_gpu,
    mirrored_strategy_with_gpu_and_cpu, mirrored_strategy_with_two_gpus
]

tpu_strategies = [
    tpu_strategy,  # steps_per_run=2
    tpu_strategy_one_step,
    cloud_tpu_strategy,
]

all_strategies_minus_default = strategies_minus_default_and_tpu + tpu_strategies

all_strategies = strategies_minus_tpu + tpu_strategies

multidevice_strategies = [
    mirrored_strategy_with_gpu_and_cpu,
    mirrored_strategy_with_two_gpus,
    tpu_strategy,  # steps_per_run=2
    tpu_strategy_one_step
]


def strategy_minus_tpu_combinations():
  return combinations.combine(
      distribution=strategies_minus_tpu, mode=["graph", "eager"])


def tpu_strategy_combinations():
  return combinations.combine(distribution=tpu_strategies, mode=["graph"])


def all_strategy_combinations():
  return strategy_minus_tpu_combinations() + tpu_strategy_combinations()


def all_strategy_minus_default_and_tpu_combinations():
  return combinations.combine(
      distribution=[
          one_device_strategy, one_device_strategy_gpu,
          mirrored_strategy_with_gpu_and_cpu, mirrored_strategy_with_two_gpus
      ],
      mode=["graph", "eager"])


def all_strategy_combinations_minus_default():
  return (all_strategy_minus_default_and_tpu_combinations() +
          tpu_strategy_combinations())
