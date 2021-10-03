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

import threading
import weakref

from tensorflow.python import _pywrap_parallel_device
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.parallel_device import saving
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest

_next_device_number = 0
_next_device_number_lock = threading.Lock()

_all_parallel_devices = weakref.WeakValueDictionary()


def unpack(tensor):
  """Finds `tensor`'s parallel device and unpacks its components."""
  parallel_device = _all_parallel_devices.get(tensor.device, None)
  if parallel_device is None:
    raise ValueError("{} is not a parallel device".format(tensor.device))
  return parallel_device.unpack(tensor)


# TODO(allenl): Expand this docstring once things like getting components on and
# off the device are stable.
#
# TODO(allenl): Make multi-client work; we need an offset for device IDs, and an
# indication of how many other devices there are total for collectives which
# don't have a number of participants hard-coded in their attributes.
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
    self.components = tuple(device_util.canonicalize(d) for d in components)
    if not self.components:
      raise ValueError("ParallelDevice requires at least one component.")
    ctx = context.context()
    with _next_device_number_lock:
      # TODO(allenl): Better names for parallel devices (right now "CUSTOM" is
      # special-cased).
      self._name = "{}/device:CUSTOM:{}".format(ctx.host_address_space(),
                                                _next_device_number)
      _next_device_number += 1
    device, device_info = _pywrap_parallel_device.GetParallelDeviceCapsules(
        self._name, self.components)
    context.register_custom_device(device, self._name, device_info)
    self._device_ids = None
    self._device_scope = None
    self._saving_scope = None
    _all_parallel_devices[self._name] = self

  def _pack_tensor(self, *tensors):
    """Helper to pack plain-old-tensors, not structures or composites."""
    for tensor in tensors:
      if not isinstance(tensor, (ops.Tensor, composite_tensor.CompositeTensor,
                                 variables.Variable)):
        raise ValueError(
            ("Every component to pack onto the ParallelDevice must already be "
             "a tensor, got {}. Consider running `tf.constant` or "
             "`tf.convert_to_tensor` first on literal values.")
            .format(tensors))
    with ops.device(None):
      # Explicitly read variable values. This can not be done on the parallel
      # device since the tensors are to be packed.
      tensors = [t.read_value() if isinstance(t, variables.Variable)
                 else t for t in tensors]
    with ops.device(self._name):
      return tpu_ops.tpu_replicated_input(inputs=tensors)

  def pack(self, tensors):
    """Create a tensor on the parallel device from a sequence of tensors.

    Args:
      tensors: A list of tensors, one per device in `self.components`. The list
        can contain composite tensors and nests (lists, dicts, etc. supported by
        `tf.nest`) with the same structure for each device, but every component
        of nests must already be a `tf.Tensor` or composite. Passing
        `tf.Variable` objects reads their value, it does not share a mutable
        reference between the packed and unpacked forms.

    Returns:
      A tensor placed on the ParallelDevice. For nested structures, returns a
      single structure containing tensors placed on the ParallelDevice (same
      structure as each component of `tensors`).

    Raises:
      ValueError: If the length of `tensors` does not match the number of
        component devices, or if there are non-tensor inputs.

    """
    self._assert_eager()
    if len(tensors) != len(self.components):
      raise ValueError(
          ("Creating a parallel tensor requires one tensor per component. "
           "Got {} but was expecting {}.")
          .format(len(tensors), len(self.components)))
    return nest.map_structure(self._pack_tensor, *tensors,
                              expand_composites=True)

  def _unpack_tensor(self, parallel_tensor):
    """Helper to unpack a single tensor."""
    if not isinstance(parallel_tensor, (
        ops.Tensor, composite_tensor.CompositeTensor, variables.Variable)):
      raise ValueError(
          "Expected a tensor, got {}.".format(parallel_tensor))
    with ops.device(self._name):
      return tpu_ops.tpu_replicated_output(
          parallel_tensor, num_replicas=len(self.components))

  def unpack(self, parallel_tensor):
    """Unpack a parallel tensor into its components.

    Args:
      parallel_tensor: A tensor, composite tensor, or `tf.nest` of such placed
        on the ParallelDevice. Passing `tf.Variable` objects reads their value,
        it does not share a mutable reference between the packed and unpacked
        forms.

    Returns:
      A list with the same length as `self.components` each with the same
      structure as `parallel_tensor`, containing component tensors.

    """
    self._assert_eager()
    unpacked_components = [[] for _ in range(len(self.components))]
    for tensor in nest.flatten(parallel_tensor, expand_composites=True):
      for accumulator, unpacked_tensor in zip(
          unpacked_components, self._unpack_tensor(tensor)):
        accumulator.append(unpacked_tensor)
    return [nest.pack_sequence_as(parallel_tensor, unpacked,
                                  expand_composites=True)
            for unpacked in unpacked_components]

  @property
  def device_ids(self):
    """A parallel tensor with scalar integers numbering component devices.

    Each device ID is placed on its corresponding device, in the same order as
    the `components` constructor argument.

    Returns:
      A parallel tensor containing 0 on the first device, 1 on the second, etc.
    """
    if self._device_ids is None:
      # device_ids may be called from inside a tf.function, in which case the
      # function captures the eager tensor. We can't pack tensors in a function
      # at the moment, and even if we could we don't want to hold on to a
      # symbolic tensor, so we need to init_scope out of the function
      # temporarily.
      with ops.init_scope():
        # TODO(allenl): Functions which capture eager device ID tensors won't be
        # saveable in SavedModels. Ideally we'd run a DeviceID op every time
        # device IDs are required, with functions using the op in their bodies
        # but not hard-coding a fixed number of devices (so they can be re-used
        # with a different replica count).
        device_ids_list = []
        for index, device in enumerate(self.components):
          with ops.device(device):
            # The identity op ensures each device ID tensor is placed on its
            # device.
            device_ids_list.append(
                array_ops.identity(constant_op.constant(index)))
        self._device_ids = self.pack(device_ids_list)

    return self._device_ids

  def _assert_eager(self):
    """Verifies that tracing is not active."""
    if not context.executing_eagerly():
      raise NotImplementedError(
          "ParallelDevice is currently not supported inside `tf.function`. It "
          "can however run calls to a `tf.function` in parallel:\n\n"
          "with ParallelDevice() as p:\n  f()")

  def __enter__(self):
    """Runs ops in parallel, makes variables which save independent buffers."""
    if (self._device_scope is not None or self._saving_scope is not None):
      raise AssertionError(
          "Re-entered a ParallelDevice scope without first exiting it.")
    self._assert_eager()
    self._device_scope = ops.device(self._name)
    self._saving_scope = saving.independent_buffers(self)
    self._device_scope.__enter__()
    # TODO(allenl): Fixing saving in Python is a bit odd. One alternative would
    # be to provide a hook for the custom device to create save specs/etc., then
    # call that hook from the default variable implementation if the variable is
    # on a custom device. We'll likely want similar hooks for repr() and such.
    self._saving_scope.__enter__()
    return self

  def __exit__(self, typ, exc, tb):
    self._device_scope.__exit__(typ, exc, tb)
    self._saving_scope.__exit__(typ, exc, tb)
    self._device_scope = None
    self._saving_scope = None
