# coding=utf-8
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for collectives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import enum

from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# TODO(b/170340570): print deprecation warning for CollectiveCommunication.
@tf_export("distribute.experimental.CommunicationImplementation",
           "distribute.experimental.CollectiveCommunication")
class CommunicationImplementation(enum.Enum):
  """Cross device communication implementation.

  Warning: The alias `tf.distribute.experimental.CollectiveCommunication` is
  deprecated and will be removed in a future version. Use
  `tf.distribute.experimental.CommunicationImplementation` instead.

  * `AUTO`: Automatically chosen by Tensorflow.
  * `RING`: TensorFlow's ring algorithms for all-reduce and
    all-gather.
  * `NCCL`: NVIDIA®'s NCCL library. This is now only used for all-reduce on
    GPUs; all-reduce on CPU, all-gather and broadcast fallbacks to RING.
  """
  AUTO = "AUTO"
  RING = "RING"
  NCCL = "NCCL"
  # TODO(ayushd): add ncclAllGather implementation.


CollectiveCommunication = CommunicationImplementation


@tf_export("distribute.experimental.CommunicationOptions")
class _OptionsExported(object):
  """Options for cross device communications like All-reduce.

  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.

  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.

  Examples:

  ```python
  options = tf.distribute.experimental.CommunicationOptions(
      bytes_per_pack=50 * 1024 * 1024,
      timeout_seconds=120,
      implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
  )
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, options=options)
  optimizer.apply_gradients(zip(grads, vars),
      experimental_aggregate_gradients=False)
  ```

  """

  def __new__(cls, *args, **kwargs):
    # We expose a dummy class so that we can separate internal and public APIs.
    # Note that __init__ won't be called on the returned object if it's a
    # different class [1].
    # [1] https://docs.python.org/3/reference/datamodel.html#object.__new__
    return Options(*args, **kwargs)

  def __init__(self,
               bytes_per_pack=0,
               timeout_seconds=None,
               implementation=CommunicationImplementation.AUTO):
    """Creates a CollectiveHints.

    Args:
      bytes_per_pack: a non-negative integer. Breaks collective operations into
        packs of certain size. If it's zero, the value is determined
        automatically. This only applies to all-reduce with
        `MultiWorkerMirroredStrategy` currently.
      timeout_seconds: a float or None, timeout in seconds. If not None, the
        collective raises `tf.errors.DeadlineExceededError` if it takes longer
        than this timeout. Zero disables timeout. This can be useful when
        debugging hanging issues.  This should only be used for debugging since
        it creates a new thread for each collective, i.e. an overhead of
        `timeout_seconds * num_collectives_per_second` more threads. This only
        works for `tf.distribute.experimental.MultiWorkerMirroredStrategy`.
      implementation: a
        `tf.distribute.experimental.CommunicationImplementation`. This is a hint
        on the preferred communication implementation. Possible values include
        `AUTO`, `RING`, and `NCCL`. NCCL is generally more performant for GPU,
        but doesn't work for CPU. This only works for
        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

    Raises:
      ValueError: When arguments have invalid value.
    """
    pass


class Options(object):
  """Implementation of OptionsInterface."""

  def __init__(self,
               bytes_per_pack=0,
               timeout_seconds=None,
               implementation=CommunicationImplementation.AUTO):
    if bytes_per_pack < 0:
      raise ValueError("bytes_per_pack must be non-negative")
    if isinstance(implementation, str):
      implementation = CommunicationImplementation(implementation.upper())
    if not isinstance(implementation, CommunicationImplementation):
      raise ValueError("implementation should be a "
                       "tf.distribute.experimental.CommunicationImplementation")
    self.bytes_per_pack = bytes_per_pack
    self.timeout_seconds = timeout_seconds
    self.implementation = implementation

  __init__.__doc__ = _OptionsExported.__init__.__doc__

  def merge(self, options):
    """Merges with another options and returns a new one.

    Values specified in the `options` takes precedence if they're not the
    default.

    Args:
      options: a `tf.distribute.experimental.CollectiveCommunication`.

    Returns:
      A new `tf.distribute.experimental.CollectiveCommunication`.
    """
    merged = copy.deepcopy(self)
    if options is None:
      return merged
    if options.bytes_per_pack != 0:
      merged.bytes_per_pack = options.bytes_per_pack
    if options.timeout_seconds is not None:
      merged.timeout_seconds = options.timeout_seconds
    if options.implementation != CommunicationImplementation.AUTO:
      merged.implementation = options.implementation
    return merged


@tf_export("distribute.experimental.CollectiveHints")
class Hints(object):
  """Hints for collective operations like AllReduce.

  This can be passed to methods like
  `tf.distribute.get_replica_context().all_reduce()` to optimize collective
  operation performance. Note that these are only hints, which may or may not
  change the actual behavior. Some options only apply to certain strategy and
  are ignored by others.

  One common optimization is to break gradients all-reduce into multiple packs
  so that weight updates can overlap with gradient all-reduce.

  Examples:

  - bytes_per_pack

  ```python
  hints = tf.distribute.experimental.CollectiveHints(
      bytes_per_pack=50 * 1024 * 1024)
  grads = tf.distribute.get_replica_context().all_reduce(
      'sum', grads, experimental_hints=hints)
  optimizer.apply_gradients(zip(grads, vars),
      experimental_aggregate_gradients=False)
  ```

  - timeout_seconds

  ```python
  strategy = tf.distribute.MirroredStrategy()
  hints = tf.distribute.experimental.CollectiveHints(
      timeout_seconds=120)
  try:
    strategy.reduce("sum", v, axis=None, experimental_hints=hints)
  except tf.errors.DeadlineExceededError:
    do_something()
  ```

  """

  @deprecation.deprecated(
      None, "use distribute.experimental.CommunicationOptions instead")
  def __new__(cls, bytes_per_pack=0, timeout_seconds=None):
    return Options(
        bytes_per_pack=bytes_per_pack, timeout_seconds=timeout_seconds)

  def __init__(self, bytes_per_pack=0, timeout_seconds=None):
    """Creates a CollectiveHints.

    Args:
      bytes_per_pack: a non-negative integer. Breaks collective operations into
        packs of certain size. If it's zero, the value is determined
        automatically. This only applies to all-reduce with
        `MultiWorkerMirroredStrategy` currently.
      timeout_seconds: a float or None, timeout in seconds. If not None, the
        collective raises `tf.errors.DeadlineExceededError` if it takes longer
        than this timeout. This can be useful when debugging hanging issues.
        This should only be used for debugging since it creates a new thread for
        each collective, i.e. an overhead of `timeout_seconds *
        num_collectives_per_second` more threads.  This only works for
        `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

    Raises:
      ValueError: When arguments have invalid value.
    """
    pass
