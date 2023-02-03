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
"""Utilities for strategies that are backed by DTensor."""

from tensorflow.dtensor.python import api as d_api
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values
from tensorflow.python.ops import summary_ops_v2


class DTensorDistributedValue(values.DistributedValues):
  """DistributedValue backed by a DTensor instance.

  This class is useful to align the interface between DTensor and tf.distribute.
  Most of the tf.distribute API will accept/return DistributedValue, whereas
  DTensor low level API will only accept DTensor instance. In order to avoid
  the conversion back and forth between DistributedValue and DTensor, we
  introduce this class so that it can work with both side.
  """

  def __init__(self, dtensor):
    if not d_api.is_dtensor(dtensor):
      raise ValueError("The DTensorDistributedValue can only be built with "
                       f"DTensor instance, got {type(dtensor)}")
    super().__init__(d_api.unpack(dtensor))
    self._dtensor = dtensor

  def get_dtensor(self):
    return self._dtensor

  @property
  def values(self):
    # Note that this method exists so that it match the interface for PerReplica
    # The public API in `tf.types.experimental.distributed.PerReplica` doesn't
    # define any methods.
    return self._values


class DTensorReplicaContext(distribute_lib.ReplicaContext):
  """ReplicaContext for strategy that is backed by DTensor.

  Since the DTensor is operated in the global context, most of the methods from
  existing strategy ReplicaContext is not applicable since they need to access
  local values. For now most of the methods in this class will raise explicit
  error to user, and we will add more support for local values in future.
  """
  _UNSUPPORTED_ERROR_MSG = (
      "Strategy that is backed by DTensor is run with a global context, and "
      "doesn't support operations for local context, like any call to merge/"
      "gather/reduce or local replica ID. Please use any strategy that is not "
      "backed by DTensor")

  def __init__(self, strategy):
    # Since DTensor strategy only runs in a global context, and we can't have
    # a local replica ID in the sync group. For now we pass None to parent, and
    # raise an explicit error when it is accessed.
    super().__init__(strategy, replica_id_in_sync_group=None)

  def __enter__(self):
    # This is a copy of parent class, without any check about whether the
    # current replica is the first one (since DTensor only has one).
    distribute_lib._push_per_thread_mode(self._thread_context)  # # pylint: disable=protected-access

    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    self._summary_recording_distribution_strategy = (
        summary_state.is_recording_distribution_strategy)
    summary_state.is_recording_distribution_strategy = True

  @property
  def replica_id_in_sync_group(self):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  @property
  def _replica_id(self):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def merge_call(self, merge_fn, args=(), kwargs=None):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def all_reduce(self, reduce_op, value, options=None):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def all_gather(self, value, axis, options=None):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def _update(self, var, fn, args=(), kwargs=None, group=True):
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)
