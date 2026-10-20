# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Zero-Code Distributed Training Optimizer."""

import json
import os

from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.util.tf_export import tf_export


@tf_export("distribute.experimental.auto_strategy", v1=[])
def auto_strategy() -> distribute_lib.StrategyBase:
  """Automatically detects hardware and returns the optimal strategy.

  `auto_strategy` is a factory that detects the available hardware
  configuration (TPUs, multiple GPUs, multi-worker clusters) and instantiates
  the most appropriate `tf.distribute.Strategy`. This eliminates the need for
  manual device profiling and strategy selection.

  Returns:
    An instance of a `tf.distribute.Strategy`.

  Example:
  ```python
  strategy = tf.distribute.experimental.auto_strategy()
  with strategy.scope():
    model = ...
  ```
  """
  tpu_visible = config.get_visible_devices("TPU")
  tpu_name = os.environ.get("TPU_NAME", "")
  has_tpu = bool(
      tpu_visible or (tpu_name and tpu_name.lower() != "local")
  )

  # Check for Multi-worker setup in TF_CONFIG first when no TPU is
  # present. This must happen before TPU initialization to avoid
  # initializing the eager runtime context, which would cause
  # CollectiveAllReduceStrategy() to raise RuntimeError("Collective ops must
  # be configured at program startup") in eager mode.
  tf_config_str = os.environ.get("TF_CONFIG", "")
  if tf_config_str and not has_tpu:
    try:
      tf_config = json.loads(tf_config_str)
    except (ValueError, TypeError):
      tf_config = {}

    if isinstance(tf_config, dict):
      cluster = tf_config.get("cluster", {})
      if isinstance(cluster, dict):
        task = tf_config.get("task") or {}
        # Exclude parameter-server clusters: a "chief" or "worker" in a cluster
        # that contains a "ps" key must not be routed to
        # CollectiveAllReduceStrategy.
        is_ps_cluster = bool(cluster.get("ps"))
        if (
            isinstance(task, dict)
            and task.get("type") in ("worker", "chief")
            and not is_ps_cluster
        ):
          workers = cluster.get("worker") or []
          chiefs = cluster.get("chief") or []
          if (
              isinstance(workers, (list, tuple))
              and isinstance(chiefs, (list, tuple))
              and len(workers) + len(chiefs) > 1
          ):
            try:
              return (
                  collective_all_reduce_strategy.CollectiveAllReduceStrategy()
              )
            except (ValueError, TypeError, KeyError):
              pass

  # Check for TPUs
  if has_tpu or bool(tpu_strategy_util.get_initialized_tpu_systems()):
    try:
      is_local_tpu = bool(
          tpu_visible and (not tpu_name or tpu_name.lower() == "local")
      )
      resolver = (
          tpu_cluster_resolver.TPUClusterResolver(tpu="local")
          if is_local_tpu
          else tpu_cluster_resolver.TPUClusterResolver()
      )
      if not tpu_strategy_util.get_initialized_tpu_systems():
        remote.connect_to_cluster(resolver)
        tpu_cluster_resolver.initialize_tpu_system(resolver)
      return tpu_strategy.TPUStrategy(resolver)
    except (ValueError, RuntimeError, ImportError, errors.OpError) as e:
      tf_logging.warning(
          "Failed to initialize TPU, falling back to next strategy: %s", e
      )

  # Check for GPUs
  visible_gpus = config.get_visible_devices("GPU")
  if visible_gpus:
    logical_gpus = config.list_logical_devices("GPU")
    if len(logical_gpus) > 1:
      return mirrored_strategy.MirroredStrategy()
    elif len(logical_gpus) == 1:
      return one_device_strategy.OneDeviceStrategy(logical_gpus[0].name)

  # Fallback to CPU
  cpu_devices = config.list_logical_devices("CPU")
  if cpu_devices:
    return one_device_strategy.OneDeviceStrategy(cpu_devices[0].name)
  return one_device_strategy.OneDeviceStrategy("/device:CPU:0")
