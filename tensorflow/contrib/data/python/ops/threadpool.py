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
"""Experimental API for controlling threading in `tf.data` pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.contrib.data.python.ops import contrib_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.data.python.ops import gen_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import resource_variable_ops

_uid_counter = 0
_uid_lock = threading.Lock()


def _generate_shared_name(prefix):
  with _uid_lock:
    global _uid_counter
    uid = _uid_counter
    _uid_counter += 1
  return "{}{}".format(prefix, uid)


class PrivateThreadPool(object):
  """A stateful resource that represents a private thread pool."""

  def __init__(self, num_threads, display_name=None):
    """Creates a `PrivateThreadPool` with the given number of threads."""
    if context.executing_eagerly():
      shared_name = _generate_shared_name("privatethreadpool")
      self._resource = gen_dataset_ops.thread_pool_handle(
          num_threads=num_threads,
          display_name=display_name,
          shared_name=shared_name)
      self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._resource, handle_device=context.context().device_name)
    else:
      self._resource = gen_dataset_ops.thread_pool_handle(
          num_threads=num_threads, display_name=display_name)


class _ThreadPoolDataset(dataset_ops.Dataset):
  """A `Dataset` that acts as an identity, and sets a custom threadpool."""

  def __init__(self, input_dataset, thread_pool):
    super(_ThreadPoolDataset, self).__init__()
    self._input_dataset = input_dataset
    self._thread_pool = thread_pool

  def _as_variant_tensor(self):
    return gen_dataset_ops.thread_pool_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._thread_pool._resource,  # pylint: disable=protected-access
        **dataset_ops.flat_structure(self))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_classes(self):
    return self._input_dataset.output_classes


def override_threadpool(dataset, thread_pool):
  """Returns a new dataset that uses the given thread pool for its operations.

  Args:
    dataset: A `tf.data.Dataset` object.
    thread_pool: A `PrivateThreadPool` object.

  Returns:
    A dataset containing the same values as `dataset`, but which uses
    `thread_pool` to compute any of its parallel operations (such as
    @{tf.data.Dataset.map}).
  """
  return _ThreadPoolDataset(dataset, thread_pool)
