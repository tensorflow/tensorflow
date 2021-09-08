# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""The execution context for ClusterCoordinator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import threading

from tensorflow.python.util.lazy_loader import LazyLoader

# There is a circular dependency between this and the `cluster_coordinator`
# module. So we load it lazily to work around this.
cluster_coordinator = LazyLoader(
    "cluster_coordinator", globals(),
    "tensorflow.python.distribute.coordinator.cluster_coordinator"
)

_dispatch_context = threading.local()


def get_current_dispatch_context():
  try:
    return _dispatch_context.current
  except AttributeError:
    return None


@contextlib.contextmanager
def with_dispatch_context(worker_obj):
  previous_context = getattr(_dispatch_context, "current", None)
  _dispatch_context.current = DispatchContext(worker_obj)
  yield
  _dispatch_context.current = previous_context


class DispatchContext(object):
  """Context entered when executing a closure on a given worker."""

  def __init__(self, worker_obj):
    self._worker = worker_obj
    self._worker_index = worker_obj.worker_index

  @property
  def worker(self):
    return self._worker

  @property
  def worker_index(self):
    return self._worker_index

  def maybe_rebuild_remote_values(self, remote_value):
    e = (
        cluster_coordinator._maybe_rebuild_remote_values(  # pylint: disable=protected-access
            self._worker, remote_value))
    if e:
      if not isinstance(e, cluster_coordinator.InputError):
        e = cluster_coordinator.InputError(e)
      raise e

  def maybe_get_remote_value(self, ret):
    return cluster_coordinator._maybe_get_remote_value(ret)  # pylint: disable=protected-access
