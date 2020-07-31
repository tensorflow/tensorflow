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
"""Special-cased checkpointing for variables on a parallel device."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training.saving import saveable_object


def _read_component(handle, dtype, replica_id, parallel_device):
  """Read one component of a parallel variable and discard the rest."""
  with ops.device(handle.device):
    read = gen_resource_variable_ops.read_variable_op(
        resource=handle, dtype=dtype)
  all_components = parallel_device.unpack(read)
  # We're pretending that parallel variables have a first axis with length
  # num_components, so we need to add a dummy first axis to the shape that gets
  # saved.
  return all_components[replica_id][None, ...]


class _ParallelDeviceSaveable(saveable_object.SaveableObject):
  """Saves and restores a parallel variable."""

  def __init__(self, name, handle, dtype, component_shape, parallel_device):
    # Each component device gets one spec with a tensor to save.
    specs = []
    for replica_id, device_name in enumerate(parallel_device.components):
      # TODO(b/151773535): SaveableObjects with SaveSpecs on different devices
      # will cause extra copying at the moment. We should fix that before doing
      # anything serious with this code.
      specs.append(
          saveable_object.SaveSpec(
              tensor=functools.partial(
                  _read_component,
                  handle=handle,
                  dtype=dtype,
                  replica_id=replica_id,
                  parallel_device=parallel_device),
              slice_spec=variables.Variable.SaveSliceInfo(
                  full_shape=([len(parallel_device.components)] +
                              component_shape),
                  var_offset=[replica_id] + [0] * len(component_shape),
                  var_shape=[1] + component_shape).spec,
              device=device_name,
              dtype=dtype,
              name=name))
    self._handle = handle
    self._parallel_device = parallel_device
    self._component_shape = component_shape
    super(_ParallelDeviceSaveable, self).__init__(None, specs, name)

  def restore(self, tensors, restored_shapes=None):
    with ops.device(self._handle.device):
      # Combine the restored tensors into one parallel tensor to assign.
      bundled = self._parallel_device.pack(tensors)
      gen_resource_variable_ops.assign_variable_op(
          resource=self._handle,
          # Squeeze out the dummy first axis we added when saving.
          value=array_ops.squeeze(bundled, axis=0))


class VariableWithFixedCheckpointing(resource_variable_ops.ResourceVariable):
  """Overrides checkpointing behavior to save like a partitioned variable."""

  def __init__(self, parallel_device, **kwargs):
    self._parallel_device = parallel_device
    kwargs = {k: v for k, v in kwargs.items()
              if k not in ["use_resource", "expected_shape"]}
    super(VariableWithFixedCheckpointing, self).__init__(**kwargs)

  def _gather_saveables_for_checkpoint(self):
    # Note VARIABLE_VALUE is the usual attribute name for variables. Using
    # something different means (a) the checkpointing infrastructure won't try
    # doing restore-on-create (which has shape issues), and (b) the saved
    # variables won't be compatible with regular variables. Both of those are
    # good in this case.
    return dict(
        PARALLEL_VARIABLE_VALUE=functools.partial(
            _ParallelDeviceSaveable,
            handle=self.handle,
            dtype=self.dtype,
            component_shape=self.shape,
            parallel_device=self._parallel_device))


def _variable_creator(next_creator, parallel_device, **kwargs):
  del next_creator
  return VariableWithFixedCheckpointing(
      parallel_device=parallel_device, **kwargs)


@contextlib.contextmanager
def independent_buffers(parallel_device):
  """Context manager which saves parallel buffers independently.

  Creates a ParallelDevice-aware variable subclass which saves buffers for each
  device separately.

  Args:
    parallel_device: A ParallelDevice object on which variables are placed.

  Yields:
    Nothing.
  """
  with variable_scope.variable_creator_scope(
      functools.partial(_variable_creator, parallel_device=parallel_device)):
    yield
