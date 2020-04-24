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
"""Utility for eagerly executing operations in parallel on multiple devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import threading

from tensorflow.python import _pywrap_parallel_device
from tensorflow.python.distribute.parallel_device import saving
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.tpu.ops import tpu_ops

_next_device_number = 0
_next_device_number_lock = threading.Lock()


# TODO(allenl): Expand this docstring once things like getting components on and
# off the device are stable.
class ParallelDevice(object):
  """A device which executes operations in parallel."""

  def __init__(self, components):
    """Creates a device which executes operations in parallel on `components`.

    Args:
      components: A list of device names. Each operation executed on the
        returned device executes on these component devices.

    Returns:
      A string with the name of the newly created device.
    """
    global _next_device_number, _next_device_number_lock
    self.components = tuple(components)
    ctx = context.context()
    with _next_device_number_lock:
      # TODO(allenl): Better names for parallel devices (right now "CUSTOM" is
      # special-cased).
      self.name = "{}/device:CUSTOM:{}".format(
          ctx.host_address_space(), _next_device_number)
      _next_device_number += 1
    device, device_info = _pywrap_parallel_device.GetParallelDeviceCapsules(
        self.name, self.components)
    context.register_custom_device(device, self.name, device_info)

  def pack(self, tensors):
    """Create a tensor on the parallel device from a sequence of tensors.

    Args:
      tensors: A flat list of tensors, one per device in `self.components`.

    Returns:
      A single tensor placed on `self.name`.
    """
    with ops.device(self.name):
      return tpu_ops.tpu_replicated_input(inputs=tensors)

  def unpack(self, parallel_tensor):
    """Unpack a parallel tensor into its components.

    Args:
      parallel_tensor: A tensor placed on `self.name`.

    Returns:
      A flat list of tensors, one per `self.components`.
    """
    with ops.device(self.name):
      return tpu_ops.tpu_replicated_output(
          parallel_tensor, num_replicas=len(self.components))

  # TODO(allenl): Fixing saving in Python is a bit odd. One alternative would be
  # to provide a hook for the custom device to create save specs/etc., then call
  # that hook from the default variable implementation if the variable is on a
  # custom device. We'll likely want similar hooks for repr() and such.
  @contextlib.contextmanager
  def scope(self):
    """Runs ops in parallel, makes variables which save independent buffers."""
    with ops.device(self.name), saving.independent_buffers(self):
      yield
