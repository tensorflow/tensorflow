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
from tensorflow.python.distribute import values


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
