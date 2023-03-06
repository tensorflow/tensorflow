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

import contextlib
import threading

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

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

  def maybe_get_remote_value(self, ret):
    return maybe_get_remote_value(ret)


def maybe_get_remote_value(val):
  """Gets the value of `val` if it is a `RemoteValue`."""
  if isinstance(val, remote_value.RemoteValue):
    error = val._get_error()  # pylint: disable=protected-access
    if error:
      raise AssertionError(
          "RemoteValue doesn't have a value because it has error %r:%s" %
          (error, error))
    elif val._status is not remote_value.RemoteValueStatus.READY:  # pylint: disable=protected-access
      raise AssertionError("The input RemoteValue has not been executed.")
    else:
      return val._get_values()  # pylint: disable=protected-access
  else:
    return val


@tf_export("distribute.experimental.coordinator.get_current_worker_index",
           v1=[])
def get_current_worker_index():
  """Returns the current worker index, when called within a worker closure.

  Some parameter server training workloads may require the worker to know its
  index, for example for data sharding for reduced-variance training.

  This method may be used within a `tf.function` that is executed on a worker.
  That is, either a `dataset_fn` that runs via
  `ClusterCoordinator.create_per_worker_dataset`, or any other function
  scheduled via `ClusterCoordinator.schedule`.

  Example (sharding data by worker):

  ```python
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = (
      tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))

  def dataset_fn(context):
    dataset = tf.data.Dataset.range(10)
    worker_index = (
        tf.distribute.experimental.coordinator.get_current_worker_index()
    )
    dataset = dataset.shard(
        num_shards=num_workers,
        index=worker_index,
    )
    return dataset

  @tf.function
  def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)

  per_worker_dataset = coordinator.create_per_worker_dataset(
      per_worker_dataset_fn)
  ```

  Raises:
    RuntimeError: if called from outside a `tf.function` or outside of a remote
      closure execution context (that is, on a non-worker machine).
  """

  msg = ("Cannot retrieve the worker index. `get_worker_idx_and_num_workers` "
         "should be called from within a tf.function being executed on a "
         "worker. This method should only be called from either a dataset_fn "
         "that is passed into `ClusterCoordinator.create_per_worker_dataset`, "
         "or a tf.function that is passed into `ClusterCoordinator.schedule`.")
  if not ops.inside_function():
    raise RuntimeError(msg)

  def call_time_worker_index():
    dispatch_context = get_current_dispatch_context()
    if not dispatch_context:
      raise RuntimeError(msg)
    return dispatch_context.worker_index

  worker_index = ops.get_default_graph().capture_call_time_value(
      call_time_worker_index, tensor.TensorSpec([], dtype=dtypes.int64))
  worker_index.op._set_attr(  # pylint: disable=protected-access
      "_user_specified_name",
      attr_value_pb2.AttrValue(s=compat.as_bytes("worker_index")))
  return worker_index
