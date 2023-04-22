# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains GPU utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re


# Matches the DeviceAttributes.physical_device_desc field.
_PHYSICAL_DEVICE_DESCRIPTION_REGEX = re.compile(
    r'name: ([^,]*), (?:.*compute capability: (\d+)\.(\d+))?')


# compute_capability is a (major version, minor version) pair, or None if this
# is not an Nvidia GPU.
GpuInfo = collections.namedtuple('gpu_info', ['name', 'compute_capability'])


def compute_capability_from_device_desc(device_attrs):
  """Returns the GpuInfo given a DeviceAttributes proto.

  Args:
    device_attrs: A DeviceAttributes proto.

  Returns
    A gpu_info tuple. Both fields are None if `device_attrs` does not have a
    valid physical_device_desc field.
  """
  # TODO(jingyue): The device description generator has to be in sync with
  # this file. Another option is to put compute capability in
  # DeviceAttributes, but I avoided that to keep DeviceAttributes
  # target-independent. Reconsider this option when we have more things like
  # this to keep in sync.
  # LINT.IfChange
  match = _PHYSICAL_DEVICE_DESCRIPTION_REGEX.search(
      device_attrs.physical_device_desc)
  # LINT.ThenChange(//tensorflow/core/common_runtime/gpu/gpu_device.cc)
  if not match:
    return GpuInfo(None, None)
  cc = (int(match.group(2)), int(match.group(3))) if match.group(2) else None
  return GpuInfo(match.group(1), cc)
