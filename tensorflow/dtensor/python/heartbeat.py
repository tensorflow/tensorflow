# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""A heartbeat service periodically pinging all workers.

In normal cases, all workers will exchange the same randomly generated number
until normal program termination. If any worker stops or restarts, other workers
will detect that and crash themselves.

In this module, logging.fatal is used to guarantee a worker crash no matter how
the functions below are called, in a thread or not.
"""

import atexit
import threading
import time

import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops.collective_ops import all_reduce
from tensorflow.python.platform import tf_logging as logging

# More than these many consecutive failures will cause a crash.
_CONSECUTIVE_FAILURES_LIMIT = 3
_failure_count = 0
_heartbeat_timer = None


def _heartbeat(
    period: int,  # in seconds
    timer: threading.Event,
    token: int,
    num_tasks: int,
    task_id: int,
    device: tf_device.DeviceSpec,
):
  """Periodically sends and receives a heartbeat signal."""
  logging.info('Starting a heartbeat thread')
  global _failure_count
  while True:
    # `timer.wait` blocks until one of two things happens.
    # It returns True if the timer is explicitly set at process exit, and we
    # should gracefully end this heartbeat thread.
    # Otherwise, it returns False when `period` has elapsed, meaning it's time
    # for the next heartbeat exchange.
    # See https://docs.python.org/3/library/threading.html#threading.Event.wait.
    if timer.wait(period):
      logging.info('Exiting the heartbeat thread normally')
      return

    # Every worker fills in one element of the signal with `token`.
    signal = np.zeros([num_tasks], dtype=np.int32)
    signal[task_id] = token

    logging.vlog(2, 'Sending heartbeat signal %s', signal)
    try:
      with ops.device(device):
        # Always use 0 for group and instance keys to reduce unnecessary
        # collective hangs and simplify failure analysis. This also avoid
        # collision with normal collectives.
        signal = all_reduce(
            constant_op.constant(signal),
            group_size=num_tasks,
            group_key=0,
            instance_key=0,
            timeout=max(period - 10, 2)).numpy()
    except Exception as e:  # pylint: disable=broad-except
      _failure_count += 1
      if _failure_count < _CONSECUTIVE_FAILURES_LIMIT:
        logging.warning('Heartbeat failure %d, %d more until limit: %s',
                        _failure_count,
                        _CONSECUTIVE_FAILURES_LIMIT - _failure_count, e)
      else:
        logging.fatal('Heartbeat failure %d, limit of %d reached: %s',
                      _failure_count, _CONSECUTIVE_FAILURES_LIMIT, e)
    logging.vlog(2, 'Received heartbeat signal %s', signal)

    # Out of sync workers will cause this, crash immediately.
    if not np.all(signal == token):
      logging.fatal('Unexpected heartbeat signal received: %s', signal)

    # Any success resets the failure counter.
    _failure_count = 0


def start(period: int) -> threading.Event:
  """Starts a persistent thread exchanging heartbeats between workers.

  Args:
    period: Heartbeat interval in seconds. Heartbeat timeout is set to the
      larger of `period` - 10 and 2s.

  Returns:
    A threading.Event object. Users can choose to call its set() method to shut
    down the heartbeat service gracefully. This isn't necessary in most cases,
    because the heartbeat service automatically shuts down at successful program
    exit through atexit handlers. But in situations when atexit handlers are not
    invoked, such as when multiprocessing processes exit in tests, users can
    manually request a shutdown.
  """
  global _heartbeat_timer
  if _heartbeat_timer is not None:
    logging.warning('A heartbeat thread is already running, skipping this one.')
    return _heartbeat_timer

  task_id = api.client_id()
  num_tasks = api.num_clients()

  # Worker 0 generates a random token. All other workers receive that token.
  if task_id == 0:
    token = np.random.randint(0, pow(2, 16) - 1)  # reserve the other 16 bits
    signal = np.full([num_tasks], token, dtype=np.int32)
  else:
    signal = np.zeros([num_tasks], dtype=np.int32)
  logging.info('Initial heartbeat signal: %s', signal)

  device = tf_device.DeviceSpec(
      job=api.job_name(),
      replica=0,
      task=task_id,
      device_type='CPU',
      device_index=0)
  # Always use 0 for group and instance keys to reduce unnecessary
  # collective hangs and simplify failure analysis. This also avoid
  # collision with normal collectives.
  with ops.device(device):
    signal = all_reduce(
        constant_op.constant(signal),
        group_size=num_tasks,
        group_key=0,
        instance_key=0,
        timeout=max(period - 10, 2)).numpy()
  logging.info('Merged heartbeat signal %s', signal)

  # The merged signal should have equal elements. If not, some worker(s) may be
  # out of sync, and we should terminate all workers.
  if task_id == 0:
    if not np.all(signal == token):
      logging.fatal('Merged heartbeat signal has value != %d', token)
  else:
    if len(set(signal)) != 1:
      logging.fatal('Merged heartbeat signal has unequal elements')
    token = signal[0]

  # On normal main process exit, set the timer to stop the heartbeat thread.
  _heartbeat_timer = threading.Event()

  def stop_heartbeat():
    logging.info('Stopping the heartbeat thread')
    _heartbeat_timer.set()
    # Give the threads some time to clean up.
    time.sleep(max(period // 10, 2))

  atexit.register(stop_heartbeat)

  # Start the persistent heartbeat thread.
  thread = threading.Thread(
      target=_heartbeat,
      args=[period, _heartbeat_timer, token, num_tasks, task_id, device],
      daemon=True)
  thread.start()

  return _heartbeat_timer
