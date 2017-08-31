# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Contains a helper context for running queue runners.

@@NestedQueueRunnerError
@@QueueRunners
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import threading

from tensorflow.python.framework import ops
from tensorflow.python.training import coordinator

__all__ = [
    'NestedQueueRunnerError',
    'QueueRunners',
]

_queue_runner_lock = threading.Lock()


class NestedQueueRunnerError(Exception):
  pass


@contextmanager
def QueueRunners(session):
  """Creates a context manager that handles starting and stopping queue runners.

  Args:
    session: the currently running session.

  Yields:
    a context in which queues are run.

  Raises:
    NestedQueueRunnerError: if a QueueRunners context is nested within another.
  """
  if not _queue_runner_lock.acquire(False):
    raise NestedQueueRunnerError('QueueRunners cannot be nested')

  coord = coordinator.Coordinator()
  threads = []
  for qr in ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS):
    threads.extend(
        qr.create_threads(
            session, coord=coord, daemon=True, start=True))
  try:
    yield
  finally:
    coord.request_stop()
    try:
      coord.join(threads, stop_grace_period_secs=120)
    except RuntimeError:
      session.close()

    _queue_runner_lock.release()
