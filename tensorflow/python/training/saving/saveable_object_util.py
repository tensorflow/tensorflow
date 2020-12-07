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

import functools
import six

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec


from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity


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
  parsed_device = parsed_device.replace(device_type="CPU", device_index=0)
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
    elif resource_variable_ops.is_resource_variable(var):

      def _read_variable_closure(v):
        def f():
          with ops.device(v.device):
            if context.executing_eagerly() and not v.is_initialized():
              # A SaveSpec tensor value of `None` indicates that the variable is
              # uninitialized.
              return None
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
                                    dtype=var.dtype, device=var.device)
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
        "trackable operations. Name is not a string: %s" % name)
  if isinstance(op, saveable_object.SaveableObject):
    yield op
  elif isinstance(op, (list, tuple, variables.PartitionedVariable)):
    if isinstance(op, variables.PartitionedVariable):
      op = list(op)
    # A set of slices.
    slice_name = None
    # pylint: disable=protected-access
    for variable in op:
      if isinstance(variable, saveable_object.SaveableObject):
        yield variable
        continue
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
        yield ResourceVariableSaveable(variable, variable._save_slice_info.spec,
                                       name)
    # pylint: enable=protected-access
  elif isinstance(op, trackable.Trackable) and not isinstance(
      op, variables.Variable):
    # pylint: disable=protected-access
    for attr, factory in op._gather_saveables_for_checkpoint().items():
      if attr == trackable.VARIABLE_VALUE_KEY:
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
    if isinstance(op, resource_variable_ops.BaseResourceVariable):
      if op._in_graph_mode:  # pylint: disable=protected-access
        variable = op._graph_element  # pylint: disable=protected-access
      else:
        variable = op
      yield ResourceVariableSaveable(variable, "", name)
    else:
      if context.executing_eagerly():
        raise ValueError("Can only save/restore ResourceVariables when "
                         "executing eagerly, got type: %s." % type(op))

      variable = ops.convert_to_tensor(op, as_ref=True)
      if not _tensor_comes_from_variable(variable):
        raise TypeError("names_to_saveables must be a dict mapping string "
                        "names to Tensors/Variables. Not a variable: %s" %
                        variable)
      if variable.op.type in ["Variable", "VariableV2",
                              "AutoReloadVariable"]:
        yield ReferenceVariableSaveable(variable, "", name)
      else:
        yield ResourceVariableSaveable(variable, "", name)


def op_list_to_dict(op_list, convert_variable_to_tensor=True):
  """Create a dictionary of names to operation lists.

  Args:
    op_list: A (nested) list, tuple, or set of Variables or SaveableObjects.
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
  # List casting is necessary to support sets.
  op_list = nest.flatten(list(op_list))
  # When ResourceVariables are converted to Tensors, read ops are added to the
  # graph. Sorting the op_list ensures that the resulting graph is always
  # constructed in a deterministic way:
  op_list = sorted(op_list, key=lambda x: x.name)
  names_to_saveables = {}
  # pylint: disable=protected-access
  for var in op_list:
    resource_or_ref_variable = (
        isinstance(var, resource_variable_ops.BaseResourceVariable) or
        isinstance(var, variables.RefVariable))

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
    elif isinstance(var, trackable.Trackable) and not resource_or_ref_variable:
      trackable_saveables = [
          (factory() if callable(factory) else factory)
          for factory in var._gather_saveables_for_checkpoint().values()]
      names_to_saveables.update(
          op_list_to_dict(trackable_saveables))
    else:
      # Variables (reference and resource) have an _in_graph_mode property
      # indicating whether they were created in a graph building context. We
      # also get Tensors when graph building, which do not have this property.
      if not getattr(var, "_in_graph_mode", True):
        if not isinstance(var, resource_variable_ops.BaseResourceVariable):
          raise ValueError(
              "Can only save/restore ResourceVariables when eager execution "
              "is enabled, type: %s." % type(var))
        set_var = names_to_saveables.setdefault(var._shared_name, var)
        if set_var is not var:
          raise ValueError(
              ("Two different ResourceVariable objects with the same "
               "shared_name '%s' were passed to the Saver. This likely means "
               "that they were created in different Graphs or isoWlation "
               "contexts, and may not be checkpointed together.") %
              (var._shared_name,))
      else:
        if convert_variable_to_tensor:
          if isinstance(var, resource_variable_ops.BaseResourceVariable):
            var = var._graph_element  # pylint: disable=protected-access
          else:
            var = ops.convert_to_tensor(var, as_ref=True)
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
  if saveable.op is not None and saveable.op in seen_ops:
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
      values are not one of Tensor or Variable or a trackable operation.
    ValueError: If the same operation is given in more than one value
      (this also applies to slices of SlicedVariables).
  """
  if not isinstance(names_to_saveables, dict):
    names_to_saveables = op_list_to_dict(names_to_saveables)

  saveables = []
  seen_ops = object_identity.ObjectIdentitySet()
  for name, op in sorted(names_to_saveables.items(),
                         # Avoid comparing ops, sort only by name.
                         key=lambda x: x[0]):
    for converted_saveable_object in saveable_objects_for_op(op, name):
      _add_saveable(saveables, seen_ops, converted_saveable_object)
  return saveables


def trace_save_restore_functions(object_to_save):
  """Gathers all SaveableObjects and traces the save and restore ops."""
  saveable_map = {}  # Maps name -> (save function, restore function)
  for name, saveable_factory in (
      object_to_save._gather_saveables_for_checkpoint().items()):  # pylint: disable=protected-access
    if not callable(saveable_factory):
      if isinstance(saveable_factory, saveable_object.SaveableObject):
        logging.debug(
            "Trackable {} should return callable factories, not SaveableObjects"
            " in `_gather_saveables_for_checkpoint`. This could lead to "
            "problems loading the SavedModel back into Python."
            .format(object_to_save))
      continue

    if is_factory_for_restored_saveable_object(saveable_factory):
      saveable_map[name] = (saveable_factory.keywords["save_function"],
                            saveable_factory.keywords["restore_function"])
    else:
      concrete_save_fn, concrete_restore_fn = _trace_save_and_restore_function(
          saveable_factory, object_to_save)
      if concrete_save_fn is not None:
        saveable_map[name] = (concrete_save_fn, concrete_restore_fn)
  return saveable_map


def _trace_save_and_restore_function(saveable_factory, object_to_save):
  """Traces the save and restore concrete functions."""
  saveables = []

  @def_function.function(
      input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
  def save_fn(checkpoint_key):
    maybe_saveable = saveable_factory(name=checkpoint_key)
    if isinstance(maybe_saveable, saveable_object.SaveableObject):
      maybe_saveable = [maybe_saveable]
    saveables[:] = maybe_saveable

    # Return list of all SaveSpecs created by the factory.
    ret = []
    for saveable in saveables:
      for spec in saveable.specs:
        ret.append({"name": spec.name, "tensor": spec.tensor,
                    "slice_spec": spec.slice_spec})
    return ret

  concrete_save_fn = save_fn.get_concrete_function()
  if any(isinstance(saveable, trackable.PythonStateSaveable)
         for saveable in saveables):
    logging.warn(
        "Note that object {} stores python values into the checkpoint. "
        "These values will not be restored when loading the SavedModel "
        "into python.".format(object_to_save))
    return None, None
  if any(isinstance(saveable, trackable.NoRestoreSaveable)
         for saveable in saveables):
    return None, None

  restored_type_specs = []
  tensor_structure = []
  for saveable in saveables:
    saveable_tensor_structure = []
    tensor_structure.append(saveable_tensor_structure)
    for spec in saveable.specs:
      restored_type_specs.append(type_spec.type_spec_from_value(spec.tensor))
      saveable_tensor_structure.append(spec.name)

  @def_function.function(input_signature=restored_type_specs)
  def restore_fn(*restored_tensors):
    structured_restored_tensors = nest.pack_sequence_as(
        tensor_structure, restored_tensors)
    for saveable, restored_tensors in zip(saveables,
                                          structured_restored_tensors):
      saveable.restore(restored_tensors, restored_shapes=None)
    return 1

  concrete_restore_fn = restore_fn.get_concrete_function()
  return concrete_save_fn, concrete_restore_fn


class RestoredSaveableObject(saveable_object.SaveableObject):
  """SaveableObject restored from SavedModel using the traced save/restore."""

  def __init__(self, save_function, restore_function, name):
    self.save_function = save_function
    self.restore_function = restore_function

    if tensor_util.is_tensor(name):
      name_tensor = name
    else:
      with ops.init_scope():
        name_tensor = constant_op.constant(name)
    tensors = save_function(name_tensor)
    specs = [saveable_object.SaveSpec(x["tensor"], x["slice_spec"], x["name"])
             for x in tensors]
    super(RestoredSaveableObject, self).__init__(None, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    del restored_shapes  # unused
    return self.restore_function(
        *[restored_tensors[i] for i in range(len(self.specs))])


def restored_saved_object_factory(save_function, restore_function):
  return functools.partial(RestoredSaveableObject,
                           save_function=save_function,
                           restore_function=restore_function)


def create_saveable_object(factory, name, call_with_mapped_captures):
  """Creates a SaveableObject while potentially in a different graph.

  When creating the frozen saver for SavedModel, the save and restore ops are
  placed in a separate graph. Since RestoredSaveableObject uses tf.functions to
  save and restore, the function captures must be mapped to the new graph.

  Args:
    factory: Factory method for creating the SaveableObject.
    name: Checkpoint key of this SaveableObject.
    call_with_mapped_captures: Helper that calls a tf.function while remapping
      the captures.

  Returns:
    a SaveableObject.
  """
  if (call_with_mapped_captures is None or
      not is_factory_for_restored_saveable_object(factory)):
    return factory(name=name)

  concrete_save_fn = factory.keywords["save_function"]
  def save_fn(name):
    return call_with_mapped_captures(concrete_save_fn, [name])

  concrete_restore_fn = factory.keywords["restore_function"]
  def restore_fn(*restored_tensors):
    return call_with_mapped_captures(concrete_restore_fn, restored_tensors)

  return factory(save_function=save_fn, restore_function=restore_fn, name=name)


def is_factory_for_restored_saveable_object(factory):
  return (isinstance(factory, functools.partial) and
          factory.func is RestoredSaveableObject)
