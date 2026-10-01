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

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.distribute.experimental import (
    multi_worker_mirrored_strategy)
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.util.tf_export import tf_export


@tf_export("distribute.AutoStrategy", v1=[])
def AutoStrategy() -> distribute_lib.StrategyBase:
  """Automatically detects hardware and returns the optimal strategy.

  `AutoStrategy` is a factory that detects the available hardware configuration
  (TPUs, multiple GPUs, multi-worker clusters) and instantiates the most
  appropriate `tf.distribute.Strategy`. This eliminates the need for manual
  device profiling and strategy selection.

  Returns:
    An instance of a `tf.distribute.Strategy`.

  Example:
  ```python
  strategy = tf.distribute.AutoStrategy()
  with strategy.scope():
    model = ...
  ```
  """
  # Check for Multi-worker setup in TF_CONFIG
  tf_config_str = os.environ.get("TF_CONFIG", "")
  if tf_config_str:
    try:
      tf_config = json.loads(tf_config_str)
    except (ValueError, TypeError):
      tf_config = {}

    if isinstance(tf_config, dict):
      cluster = tf_config.get("cluster", {})
      if isinstance(cluster, dict):
        if (len(cluster.get("worker", [])) > 1
            or len(cluster.get("chief", [])) > 0):
          return multi_worker_mirrored_strategy.MultiWorkerMirroredStrategy()

  # Check for TPUs
  if config.list_logical_devices("TPU") or os.environ.get("TPU_NAME"):
    try:
      resolver = tpu_cluster_resolver.TPUClusterResolver()
      remote.connect_to_cluster(resolver)
      if not tpu_strategy_util.get_initialized_tpu_systems():
        tpu_cluster_resolver.initialize_tpu_system(resolver)
      return tpu_strategy.TPUStrategy(resolver)
    except (ValueError, RuntimeError, ImportError):
      pass

  # Check for GPUs
  visible_gpus = config.get_visible_devices("GPU")
  if visible_gpus:
    logical_gpus = config.list_logical_devices("GPU")
    if len(logical_gpus) > 1:
      return mirrored_strategy.MirroredStrategy()
    elif len(logical_gpus) == 1:
      return one_device_strategy.OneDeviceStrategy("/GPU:0")

  # Fallback to CPU
  return one_device_strategy.OneDeviceStrategy("/CPU:0")
