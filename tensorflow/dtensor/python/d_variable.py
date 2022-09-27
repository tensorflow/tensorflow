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
# ==============================================================================
"""DTensor variable and saveable."""

import contextlib
import functools

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util.tf_export import tf_export


class DSaveSpec(saveable_object.SaveSpec):
  """DTensor SaveSpec that additionaly captures global_shape and layout."""

  def __init__(self,
               tensor,
               slice_spec,
               name,
               global_shape,
               layout,
               dtype=None,
               device=None):
    super().__init__(
        tensor=tensor,
        slice_spec=slice_spec,
        name=name,
        dtype=dtype,
        device=device)
    self.global_shape = global_shape
    self.layout = layout


class _DVariableSaveable(saveable_object.SaveableObject):
  """Class for defining how to save/restore DTensor variable."""

  def __init__(self, dvariable, name):
    with ops.device(dvariable.device):
      original_layout = api.fetch_layout(dvariable)
    # Record original layout to allow restore.
    self._original_layout = original_layout
    self._dvariable = dvariable

    def pack(tensors, layout):
      with ops.device(dvariable.device):
        return api.pack(tensors, layout)

    host_layout = layout_lib.Layout(original_layout.sharding_specs,
                                    original_layout.mesh.host_mesh())

    def get_host_dtensor():
      # Copy to host mesh if needed.
      if original_layout.mesh.device_type().upper() != 'CPU':
        # Prefer pack and unpack in eager mode because it supports sharded
        # layouts.
        if context.executing_eagerly():
          host_dtensor = api.pack(
              api.unpack(dvariable.read_value()), host_layout)
        else:
          host_dtensor = api.copy_to_mesh(dvariable.read_value(), host_layout)
      else:
        host_dtensor = dvariable.read_value()
      return (math_ops.cast(host_dtensor, dtypes.bfloat16)
              if self.should_cast(host_dtensor) else host_dtensor)

    num_local_devices = original_layout.mesh.num_local_devices()
    super(_DVariableSaveable, self).__init__(
        None,
        [
            DSaveSpec(
                tensor=get_host_dtensor,
                slice_spec=pack([''] * num_local_devices,
                                layout_lib.Layout.replicated(
                                    original_layout.mesh.host_mesh(), rank=0)),
                name=pack([name] * num_local_devices,
                          layout_lib.Layout.replicated(
                              original_layout.mesh.host_mesh(), rank=0)),
                global_shape=dvariable.shape,
                # Layout is attached as attribute, no need to put it as a
                # Tensor on DTensorDevice.
                layout=host_layout.to_string(),
                dtype=dtypes.bfloat16
                if self.should_cast(dvariable) else dvariable.dtype,
                device=dvariable.device)
        ],
        name)

  def should_cast(self, v):
    """Returns True if v has float32 dtype and is intructed to save as bf16.

    Args:
      v : The variable that determines whether to cast.

    Returns:
      True if current savable DVariable is instructed to save as bfloat16 and
        the variable has dtype float32.
    """
    return self._dvariable.save_as_bf16 and v.dtype == dtypes.float32

  def restore(self, restored_tensors, restored_shapes):
    """Restores the same value into all variables."""
    tensor, = restored_tensors

    @def_function.function
    def _restore(t):
      with ops.device(self._dvariable.device):
        return api.copy_to_mesh(t, self._original_layout)

    # This assign establishes connections from restored tensor and tensors
    # being restored to -- so that restore in SPMD can backtrack the DVariable
    # and its layout, given that we're using tf.function style restore.
    # Note that the restored dvaraible is on CPU no matter what as the restoreV2
    # op must run on CPU.
    # TODO(b/159035705): Allow restore for Tensor objects as well?
    # Restore the dvariable back to original layout.
    if self._original_layout.mesh.device_type().upper() != 'CPU':
      tensor = _restore(tensor)
    return self._dvariable.assign(
        math_ops.cast(tensor, dtype=self._dvariable.dtype) if self._dvariable
        .save_as_bf16 else tensor)


@tf_export('experimental.dtensor.DVariable', v1=[])
class DVariable(resource_variable_ops.ResourceVariable):
  """A replacement for tf.Variable which follows initial value placement.

    The class also handles restore/save operations in DTensor. Note that,
    DVariable may fall back to normal tf.Variable at this moment if
    `initial_value` is not a DTensor.
  """

  def __init__(self, initial_value, *args, dtype=None, **kwargs):
    """Overrides tf.Variable to fix VarHandleOp placements."""
    # Variables by default use the current device scope for placement. This
    # wrapper has them follow the initial value's placement instead (which will
    # be the DTensor device if the initial value has a layout).

    # Pop layout from kwargs since keras make_variable may pass a 'layout'
    # keyword argument. We need to pop it because we are passing kwargs to
    # super class constructor.
    layout = kwargs.pop('layout', None)
    shape = kwargs.get('shape', None)

    if callable(initial_value):
      unwrapped = initial_value
      if issubclass(type(initial_value), functools.partial):
        unwrapped = initial_value.func

      # If wrapped is a CheckpointInitialValueCallable, this means that
      # we are creating a Variable during a checkpoint restore.
      # Thus the restore will happen now through this callable
      # and we will create the DVariable with the restored dtensor.
      if issubclass(type(unwrapped), trackable.CheckpointInitialValueCallable):
        if not shape or not layout:
          raise ValueError('Expected shape and layout to be not None.')

        # CheckpointInitialValueCallable will call an eager tf.RestoreV2,
        # which does not have any shape information or layout information
        # attached. Thus we will do two things to have them correctly specified:
        #
        # The default layout scope allows us to correctly specify the output
        # layout of the tf.RestoreV2 that will be called
        #
        # Passing shard_info with the correct shape allows the tf.RestoreV2
        # ShapeInference to extract the shape.
        initial_value = api.call_with_layout(
            initial_value,
            layout,
            shard_info=trackable.ShardInfo(
                shape=shape, offset=[0] * len(shape)))
      else:
        initial_value = initial_value()

    # When the initial value came from a Checkpoint restoration, fetch tensor.
    if isinstance(initial_value, trackable.CheckpointInitialValue):
      initial_value = initial_value.wrapped_value

    initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
    variable_device = initial_value.device
    self._save_as_bf16 = False
    # TODO(b/159035705): The following code enables variable creation inside
    # a tf.function. However, it requires a global dtensor device.
    # if not variable_device and not tf.executing_eagerly():
    #   try:
    #     initial_value.op.get_attr("_layout")
    #   except ValueError:
    #     pass
    #   else:
    #     # The initial value is a DTensor, but because the DTensor device is
    #     # only active during eager execution at the moment we need to
    #     # translate that into a placement for the eager VarHandleOp.
    #     variable_device = _dtensor_device().name
    with ops.device(variable_device):
      # If initial tensor assigned to DVariable is DTensor, record the layout of
      # the resource so that this can be queried.
      self.layout = None
      if context.executing_eagerly():
        try:
          self.layout = api.fetch_layout(initial_value)
        except (errors.InvalidArgumentError, errors.NotFoundError):
          # For Non-DTensor tensors, fetch layout results in expected
          # InvalidArgument or NotFoundError depending on whether the API
          # is called within DTensor device scope or not.
          self.layout = None
          pass
      mesh = self.layout.mesh if self.layout else None
      with api.run_on(mesh) if mesh else contextlib.nullcontext():
        super(DVariable, self).__init__(
            initial_value, *args, dtype=dtype, **kwargs)

  @property
  def save_as_bf16(self):
    return self._save_as_bf16

  @save_as_bf16.setter
  def save_as_bf16(self, save_as_bf16):
    """Enables saving float32 as bfloat16."""
    self._save_as_bf16 = save_as_bf16 and self.dtype == dtypes.float32

  def _gather_saveables_for_checkpoint(self):
    return {
        trackable.VARIABLE_VALUE_KEY:
            functools.partial(_DVariableSaveable, self)
    }
