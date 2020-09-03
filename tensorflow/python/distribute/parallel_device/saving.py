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

from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training.saving import saveable_object


class _ParallelComponentSaveable(saveable_object.SaveableObject):
  """Saves and restores one component of a parallel variable."""

  def __init__(self, name, handle, dtype, shape):
    specs = [saveable_object.SaveSpec(
        tensor=functools.partial(gen_resource_variable_ops.read_variable_op,
                                 resource=handle, dtype=dtype),
        slice_spec="",
        device=handle.device,
        dtype=dtype,
        name=name)]
    self._handle = handle
    super(_ParallelComponentSaveable, self).__init__(handle, specs, name)

  def restore(self, tensors, restored_shapes=None):
    restored_tensor, = tensors
    gen_resource_variable_ops.assign_variable_op(
        resource=self._handle, value=restored_tensor)


class ParallelSavingMixin(resource_variable_ops.BaseResourceVariable):
  """Mixin to to override variable checkpointing, saving each component."""

  def __init__(self, parallel_device, expected_shape=None, use_resource=None,
               **kwargs):
    del expected_shape, use_resource
    self._parallel_device = parallel_device
    super(ParallelSavingMixin, self).__init__(**kwargs)

  # TODO(allenl): Consider either adding a boolean argument for
  # save-primary-only or looking at synchronization/aggregation properties.
  def _gather_saveables_for_checkpoint(self):
    """Generate SaveableObjects for each component device."""
    component_saveables = {}
    # Create one SaveableObject per device, each one of which looks like a
    # regular ResourceVariable saveable.
    for index, handle in enumerate(self._parallel_device.unpack(self.handle)):
      if index == 0:
        # This is the name regular tf.Variables use to save. Using it for the
        # component on the first device means non-parallel tf.Variable objects
        # will use this value when pointed at a parallel checkpoint.
        attribute = "VARIABLE_VALUE"
      else:
        attribute = "parallel_component_{}".format(index)
      component_saveables[attribute] = (
          functools.partial(
              _ParallelComponentSaveable,
              handle=handle,
              dtype=self.dtype,
              shape=self.shape))
    return component_saveables


class ParallelVariable(
    ParallelSavingMixin, resource_variable_ops.ResourceVariable):
  pass


class UninitializedParallelVariable(
    ParallelSavingMixin, resource_variable_ops.UninitializedVariable):
  pass


def _variable_creator(next_creator, parallel_device, initial_value=None,
                      **kwargs):
  del next_creator
  if initial_value is not None:
    return ParallelVariable(
        parallel_device=parallel_device, initial_value=initial_value, **kwargs)
  else:
    # SavedModel loading does not pass an initial value.
    return UninitializedParallelVariable(
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
