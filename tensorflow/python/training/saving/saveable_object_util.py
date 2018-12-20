# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for working with and creating SaveableObjects."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.training.saving import saveable_object


# Op names which identify variable reads which should be saved.
_VARIABLE_OPS = set(["Variable",
                     "VariableV2",
                     "AutoReloadVariable",
                     "VarHandleOp",
                     "ReadVariableOp"])


def set_cpu0(device_string):
  """Creates a new device string based on `device_string` but using /CPU:0.

  If the device is already on /CPU:0, this is a no-op.

  Args:
    device_string: A device string.

  Returns:
    A device string.
  """
  parsed_device = pydev.DeviceSpec.from_string(device_string)
  parsed_device.device_type = "CPU"
  parsed_device.device_index = 0
  return parsed_device.to_string()


class ReferenceVariableSaveable(saveable_object.SaveableObject):
  """SaveableObject implementation that handles reference variables."""

  def __init__(self, var, slice_spec, name):
    spec = saveable_object.SaveSpec(var, slice_spec, name, dtype=var.dtype)
    super(ReferenceVariableSaveable, self).__init__(var, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    restored_tensor = restored_tensors[0]
    if restored_shapes is not None:
      restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
    return state_ops.assign(
        self.op,
        restored_tensor,
        validate_shape=restored_shapes is None and
        self.op.get_shape().is_fully_defined())


class ResourceVariableSaveable(saveable_object.SaveableObject):
  """SaveableObject implementation that handles ResourceVariables."""

  def __init__(self, var, slice_spec, name):
    self._var_device = var.device
    self._var_shape = var.shape
    if isinstance(var, ops.Tensor):
      self.handle_op = var.op.inputs[0]
      tensor = var
    elif isinstance(var, resource_variable_ops.ResourceVariable):

      def _read_variable_closure(v):
        def f():
          with ops.device(v.device):
            x = v.read_value()
            # To allow variables placed on non-CPU devices to be checkpointed,
            # we copy them to CPU on the same machine first.
            with ops.device("/device:CPU:0"):
              return array_ops.identity(x)
        return f

      self.handle_op = var.handle
      tensor = _read_variable_closure(var)
    else:
      raise ValueError(
          "Saveable is neither a resource variable nor a read operation."
          " Got: %s" % repr(var))
    spec = saveable_object.SaveSpec(tensor, slice_spec, name,
                                    dtype=var.dtype)
    super(ResourceVariableSaveable, self).__init__(var, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    restored_tensor = restored_tensors[0]
    if restored_shapes is not None:
      restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
    # Copy the restored tensor to the variable's device.
    with ops.device(self._var_device):
      restored_tensor = array_ops.identity(restored_tensor)
      return resource_variable_ops.shape_safe_assign_variable_handle(
          self.handle_op, self._var_shape, restored_tensor)


def _tensor_comes_from_variable(v):
  return isinstance(v, ops.Tensor) and v.op.type in _VARIABLE_OPS


def saveable_objects_for_op(op, name):
  """Create `SaveableObject`s from an operation.

  Args:
    op: A variable, operation, or SaveableObject to coerce into a
      SaveableObject.
    name: A string name for the SaveableObject.

  Yields:
    `SaveableObject`s which together save/restore `op`.

  Raises:
    TypeError: If `name` is not a string.
    ValueError: For operations with no known conversion to SaveableObject.
  """
  if not isinstance(name, six.string_types):
    raise TypeError(
        "names_to_saveables must be a dict mapping string names to "
        "checkpointable operations. Name is not a string: %s" % name)
  if isinstance(op, saveable_object.SaveableObject):
    yield op
  elif isinstance(op, (list, tuple, variables.PartitionedVariable)):
    if isinstance(op, variables.PartitionedVariable):
      op = list(op)
    # A set of slices.
    slice_name = None
    # pylint: disable=protected-access
    for variable in op:
      if not isinstance(variable, variables.Variable):
        raise ValueError("Slices must all be Variables: %s" % variable)
      if not variable._save_slice_info:
        raise ValueError("Slices must all be slices: %s" % variable)
      if slice_name is None:
        slice_name = variable._save_slice_info.full_name
      elif slice_name != variable._save_slice_info.full_name:
        raise ValueError(
            "Slices must all be from the same tensor: %s != %s" %
            (slice_name, variable._save_slice_info.full_name))
      if variable.op.type in ["Variable", "VariableV2",
                              "AutoReloadVariable"]:
        yield ReferenceVariableSaveable(
            variable, variable._save_slice_info.spec, name)
      else:
        yield ResourceVariableSaveable(
            variable, variable._save_slice_info.spec, name)
    # pylint: enable=protected-access
  elif isinstance(op, checkpointable.CheckpointableBase) and not isinstance(
      op, variables.Variable):
    # pylint: disable=protected-access
    for attr, factory in op._gather_saveables_for_checkpoint().items():
      if attr == checkpointable.VARIABLE_VALUE_KEY:
        # Keep original name for classes masquerading as variables.
        full_name = name
      else:
        full_name = name + "_" + attr
      op = (factory(full_name) if callable(factory) else factory)
      for op in saveable_objects_for_op(op, op.name):
        yield op
    # pylint: enable=protected-access
  else:
    # A variable or tensor.
    if isinstance(op, resource_variable_ops.ResourceVariable):
      # pylint: disable=protected-access
      if op._in_graph_mode:
        variable = op._graph_element
      else:
        variable = op
      # pylint: enable=protected-access
      yield ResourceVariableSaveable(variable, "", name)
    else:
      with ops.init_scope():
        if context.executing_eagerly():
          raise ValueError("Can only save/restore ResourceVariables when "
                           "executing eagerly, got type: %s." % type(op))

      variable = ops.internal_convert_to_tensor(op, as_ref=True)
      if not _tensor_comes_from_variable(variable):
        raise TypeError("names_to_saveables must be a dict mapping string "
                        "names to Tensors/Variables. Not a variable: %s" %
                        variable)
      if variable.op.type in ["Variable", "VariableV2",
                              "AutoReloadVariable"]:
        yield ReferenceVariableSaveable(variable, "", name)
      else:
        yield ResourceVariableSaveable(
            variable, "", name)


def op_list_to_dict(op_list, convert_variable_to_tensor=True):
  """Create a dictionary of names to operation lists.

  Args:
    op_list: A list, tuple, or set of Variables or SaveableObjects.
    convert_variable_to_tensor: Whether or not to convert single Variables
      with no slice info into Tensors.

  Returns:
    A dictionary of names to the operations that must be saved under
    that name.  Variables with save_slice_info are grouped together under the
    same key in no particular order.

  Raises:
    TypeError: If the type of op_list or its elements is not supported.
    ValueError: If at least two saveables share the same name.
  """
  if not isinstance(op_list, (list, tuple, set)):
    raise TypeError("Variables to save should be passed in a dict or a "
                    "list: %s" % op_list)
  # When ResourceVariables are converted to Tensors, read ops are added to the
  # graph. Sorting the op_list ensures that the resulting graph is always
  # constructed in a deterministic way:
  op_list = sorted(op_list, key=lambda x: x.name)
  names_to_saveables = {}
  # pylint: disable=protected-access
  for var in op_list:
    if isinstance(var, saveable_object.SaveableObject):
      names_to_saveables[var.name] = var
    elif isinstance(var, variables.PartitionedVariable):
      if var.name in names_to_saveables:
        raise ValueError("At least two variables have the same name: %s" %
                         var.name)
      names_to_saveables[var.name] = var
    elif isinstance(var, variables.Variable) and var._save_slice_info:
      name = var._save_slice_info.full_name
      if name in names_to_saveables:
        if not isinstance(names_to_saveables[name], list):
          raise ValueError("Mixing slices and non-slices with the same name: "
                           "%s" % name)
        names_to_saveables[name].append(var)
      else:
        names_to_saveables[name] = [var]
    elif (isinstance(var, checkpointable.CheckpointableBase)
          and not isinstance(var, variables.Variable)):
      checkpointable_saveables = [
          (factory() if callable(factory) else factory)
          for factory in var._gather_saveables_for_checkpoint().values()]
      names_to_saveables.update(
          op_list_to_dict(checkpointable_saveables))
    else:
      if context.executing_eagerly():
        if not isinstance(var, resource_variable_ops.ResourceVariable):
          raise ValueError(
              "Can only save/restore ResourceVariables when eager execution "
              "is enabled, type: %s." % type(var))
        set_var = names_to_saveables.setdefault(var._shared_name, var)
        if set_var is not var:
          raise ValueError(
              ("Two different ResourceVariable objects with the same "
               "shared_name '%s' were passed to the Saver. This likely means "
               "that they were created in different Graphs or isolation "
               "contexts, and may not be checkpointed together.") %
              (var._shared_name,))
      else:
        if convert_variable_to_tensor:
          if isinstance(var, resource_variable_ops.ResourceVariable):
            var = var._graph_element  # pylint: disable=protected-access
          else:
            var = ops.internal_convert_to_tensor(var, as_ref=True)
          if not _tensor_comes_from_variable(var):
            raise TypeError("Variable to save is not a Variable: %s" % var)
        if var.op.type == "ReadVariableOp":
          name = var.op.inputs[0].op.name
        else:
          name = var.op.name
        if name in names_to_saveables:
          raise ValueError("At least two variables have the same name: %s" %
                           name)
        names_to_saveables[name] = var

    # pylint: enable=protected-access
  return names_to_saveables


def _add_saveable(saveables, seen_ops, saveable):
  """Adds the saveable to the saveables list.

  Args:
    saveables: List to append the SaveableObject to.
    seen_ops: Set of the ops of the saveables already processed.  Used to
      check that each saveable is only saved once.
    saveable: The saveable.

  Raises:
    ValueError: If the saveable has already been processed.
  """
  if saveable.op in seen_ops:
    raise ValueError("The same saveable will be restored with two names: %s" %
                     saveable.name)
  saveables.append(saveable)
  seen_ops.add(saveable.op)


def validate_and_slice_inputs(names_to_saveables):
  """Returns the variables and names that will be used for a Saver.

  Args:
    names_to_saveables: A dict (k, v) where k is the name of an operation and
       v is an operation to save or a BaseSaverBuilder.Saver.

  Returns:
    A list of SaveableObjects.

  Raises:
    TypeError: If any of the keys are not strings or any of the
      values are not one of Tensor or Variable or a checkpointable operation.
    ValueError: If the same operation is given in more than one value
      (this also applies to slices of SlicedVariables).
  """
  if not isinstance(names_to_saveables, dict):
    names_to_saveables = op_list_to_dict(names_to_saveables)

  saveables = []
  seen_ops = set()
  for name, op in sorted(names_to_saveables.items(),
                         # Avoid comparing ops, sort only by name.
                         key=lambda x: x[0]):
    for converted_saveable_object in saveable_objects_for_op(op, name):
      _add_saveable(saveables, seen_ops, converted_saveable_object)
  return saveables
