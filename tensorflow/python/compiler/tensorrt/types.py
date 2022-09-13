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
# =============================================================================
"""Exposes utility classes for TF-TRT."""

from enum import Enum
from enum import EnumMeta
from packaging import version

from tensorflow.python.compiler.tensorrt import gen_trt_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import resource


class _EnumMeta(EnumMeta):
  def __call__(cls, value, *args, **kwargs):
    obj = super().__call__(value, *args, **kwargs)
    obj.__validate__()
    return obj


class ExtendedEnum(Enum, metaclass=_EnumMeta):

  @classmethod
  def to_dict(cls):
    """Returns a dictionary representation of the enum."""
    return {e.name: e.value for e in cls}

  @classmethod
  def values(cls):
    """Returns a list of all the enum values."""
    return list(cls._value2member_map_.keys())

  @classmethod
  def keys(cls):
    """Returns a list of all the enum values."""
    return list(cls._value2member_map_.keys())

  @classmethod
  def items(cls):
    """Returns a list of all the enum values."""
    return cls.to_dict().items()

  def __eq__(self, other):
    return self.value == other

  def __validate__(self):
    pass

  def lower(self):
    return self.value.lower()

  def upper(self):
    return self.value.upper()


class TrtVersion(version.Version):
  def __init__(self, version):
    if isinstance(version, tuple):
      if len(version) != 3:
        raise ValueError(f"A tuple of size 3 was expected, received: {version}")
      version = ".".join([str(s) for s in version])

    if not isinstance(version, str):
      raise ValueError(f"Expected tuple of size 3 or str, received: {version}")

    super().__init__(version)


class TRTEngineResource(resource.TrackableResource):
  """Class to track the serialized engines resource."""

  def __init__(
      self, resource_name, filename, maximum_cached_engines, device="GPU"
  ):
    super(TRTEngineResource, self).__init__(device=device)
    self._resource_name = resource_name
    # Track the serialized engine file in the SavedModel.
    self._filename = self._track_trackable(
        asset.Asset(filename), "_serialized_trt_resource_filename")
    self._maximum_cached_engines = maximum_cached_engines

  @staticmethod
  def get_resource_handle(name, device):
    with ops.device(device):
      return gen_trt_ops.create_trt_resource_handle(resource_name=name)

  def _create_resource(self):
    return TRTEngineResource.get_resource_handle(
        self._resource_name, self._resource_device)

  def _initialize(self):
    gen_trt_ops.initialize_trt_resource(
        self.resource_handle,
        self._filename,
        max_cached_engines_count=self._maximum_cached_engines)

  def _destroy_resource(self):
    handle = TRTEngineResource.get_resource_handle(
        self._resource_name, self._resource_device)
    with ops.device(self._resource_device):
      gen_resource_variable_ops.destroy_resource_op(
          handle, ignore_lookup_error=True)
