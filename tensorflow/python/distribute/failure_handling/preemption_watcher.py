# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Provides a utility class for preemption detection and recovery."""

import threading

from absl import logging

from tensorflow.python.distribute.failure_handling.failure_handling_util import detect_platform
from tensorflow.python.distribute.failure_handling.failure_handling_util import PlatformDevice
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework.errors import AbortedError
from tensorflow.python.framework.errors import CancelledError
from tensorflow.python.framework.errors import InternalError
from tensorflow.python.framework.errors import UnavailableError
from tensorflow.python.util.tf_export import tf_export


_preemption_watcher_initialization_counter = monitoring.Counter(
    "/tensorflow/api/distribution_strategy/preemption_watcher_initialized",
    "Counter for usages of PreemptionWatcher",
)
_preemption_handling_counter = monitoring.Counter(
    "/tensorflow/api/distribution_strategy/preemption_watcher_handled",
    "Counter for number of preempions caught and handled by PreemptionWatcher",
)

_PREEMPTION_KEY = "TF_DEFAULT_PREEMPTION_NOTICE_KEY"


@tf_export("distribute.experimental.PreemptionWatcher", v1=[])
class PreemptionWatcher:
  """Watch preemption signal and store it.

  Notice: Currently only support Borg TPU environment with TPUClusterResolver.

  This class provides a way to monitor the preemption signal during training on
  TPU. It will start a background thread to watch the training process, trying
  to fetch preemption message from the coordination service. When preemption
  happens, the preempted worker will write the preemption message to the
  coordination service. Thus getting a non-empty preemption message means there
  is a preemption happened.

  User can use the preemption message as a reliable preemption indicator, and
  then set the coordinator to reconnect to the TPU worker instead of a fully
  restart triggered by Borg. For example, a training process with
  preemption recovery will be like:

  ```python
  keep_running = True
  preemption_watcher = None
  while keep_running:
    try:
      # Initialize TPU cluster and stratygy.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.TPUStrategy(resolver)

      # PreemptionWatcher must be created after connected to cluster.
      preemption_watcher = tf.distribute.experimental.PreemptionWatcher()
      train_model(strategy)
      keep_running = False
    except Exception as e:
      if preemption_watcher and preemption_watcher.preemption_message:
        preemption_watcher.block_until_worker_exit()
        keep_running = True
      else:
        raise e
  ```

  Attributes:
    preemption_message: A variable to store the preemption message fetched from
      the coordination service. If it is not None, then there is a preemption
      happened.
    platform: A PlatformDevice to indicate the current job's platform. Refer to
      failure_handling_util.py for the definition of enum class PlatformDevice.
  """

  def __init__(self):
    # TODO(b/254321514): Integrate with GPU and cloud enviornmenmt.
    self._preemption_message = None
    self._platform = detect_platform()
    if self._platform != PlatformDevice.INTERNAL_TPU:
      logging.warning(
          "Preemption watcher does not support environment: %s", self._platform
      )
    else:
      _preemption_watcher_initialization_counter.get_cell().increase_by(1)
      threading.Thread(target=self._watch_preemption_key, daemon=True).start()

  @property
  def preemption_message(self):
    """Returns the preemption message."""
    return self._preemption_message

  def _watch_preemption_key(self):
    logging.info("Watching preemption signal.")
    message = context.context().get_config_key_value(_PREEMPTION_KEY)
    _preemption_handling_counter.get_cell().increase_by(1)
    logging.info("Preemption signal received.")
    self._preemption_message = message

  def block_until_worker_exit(self):
    """Block coordinator until workers exit.

    In some rare cases, another error could be raised during the
    preemption grace period. This will cause the coordinator to reconnect to the
    same TPU workers, which will be killed later. It prevents the coordinator to
    reconnect to new TPU workers, and falls back to a hard restart. To avoid
    this situation, this method will block the coordinator to reconnect until
    workers exit. This method will be a no-op for non-TPU platform.
    """
    if self._platform != PlatformDevice.INTERNAL_TPU:
      return
    try:
      context.context().get_config_key_value("BLOCK_TILL_EXIT")
    except InternalError as e:
      # Ensure that internal error is related to coordination service.
      if "Coordination service is not enabled." not in e.message:
        raise
      logging.info("Workers exited.")
    except (AbortedError, CancelledError, UnavailableError):
      logging.info("Workers exited.")
