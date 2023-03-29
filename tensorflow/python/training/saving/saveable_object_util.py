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
import functools

from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.client import session
from tensorflow.python.eager import context

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export

# Op names which identify variable reads which should be saved.
_VARIABLE_OPS = set(["Variable",
                     "VariableV2",
                     "AutoReloadVariable",
                     "VarHandleOp",
                     "ReadVariableOp"])

_REF_VARIABLE_OPS = frozenset(["Variable", "VariableV2", "AutoReloadVariable"])


def set_cpu0(device_string):
  """Creates a new device string based on `device_string` but using /CPU:0.

  If the device is already on /CPU:0 or it is a custom device, this is a no-op.

  Args:
    device_string: A device string.

  Returns:
    A device string.
  """
  if context.is_custom_device(device_string):
    return device_string
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
            # Read the variable without making a copy to limit memory usage.
            x = v.read_value_no_copy()
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
          f" Got: {repr(var)}")
    spec = saveable_object.SaveSpec(tensor, slice_spec, name,
                                    dtype=var.dtype, device=var.device)
    super(ResourceVariableSaveable, self).__init__(var, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    """Restores tensors. Raises ValueError if incompatible shape found."""
    restored_tensor = restored_tensors[0]
    if restored_shapes is not None:
      restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
    # Copy the restored tensor to the variable's device.
    with ops.device(self._var_device):
      restored_tensor = array_ops.identity(restored_tensor)
      try:
        assigned_variable = resource_variable_ops.shape_safe_assign_variable_handle(
            self.handle_op, self._var_shape, restored_tensor)
      except ValueError as e:
        raise ValueError(
            f"Received incompatible tensor with shape {restored_tensor.shape} "
            f"when attempting to restore variable with shape {self._var_shape} "
            f"and name {self.name}.") from e
      return assigned_variable


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
  if not isinstance(name, str):
    raise TypeError(
        "names_to_saveables must be a dict mapping string names to "
        f"trackable operations. Name is not a string: {name}")
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
        raise ValueError(f"Slices must all be Variables: {variable}")
      if not variable._save_slice_info:
        raise ValueError(f"Slices must all be slices: {variable}")
      if slice_name is None:
        slice_name = variable._save_slice_info.full_name
      elif slice_name != variable._save_slice_info.full_name:
        raise ValueError(
            f"Slices must all be from the same tensor: {slice_name} != "
            f"{variable._save_slice_info.full_name}")
      if variable.op.type in _REF_VARIABLE_OPS:
        yield ReferenceVariableSaveable(
            variable, variable._save_slice_info.spec, name)
      else:
        yield ResourceVariableSaveable(variable, variable._save_slice_info.spec,
                                       name)
    # pylint: enable=protected-access
  elif isinstance(op, trackable.Trackable) and not isinstance(
      op, variables.Variable):
    # pylint: disable=protected-access
    for attr, factory in saveable_objects_from_trackable(
        op, tf1_saver=True).items():
      if attr == trackable.VARIABLE_VALUE_KEY:
        # Keep original name for classes masquerading as variables and
        # Trackables that define _serialize_to_tensors.
        full_name = name
      elif attr == trackable_utils.SERIALIZE_TO_TENSORS_NAME:
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
                         f"executing eagerly, got type: {type(op)}.")

      variable = ops.convert_to_tensor(op, as_ref=True)
      if not _tensor_comes_from_variable(variable):
        raise TypeError(
            "names_to_saveables must be a dict mapping string "
            f"names to Tensors/Variables. Not a variable: {variable}")
      if variable.op.type in _REF_VARIABLE_OPS:
        yield ReferenceVariableSaveable(variable, "", name)
      else:
        yield ResourceVariableSaveable(variable, "", name)


def op_list_to_dict(op_list, convert_variable_to_tensor=True):
  """Create a dictionary of names to operation lists.

  This method is only used when the variable name matters (e.g. when saving
  or restoring from a TF1 name-based checkpoint). In TF2, this can be called
  from `tf.train.Checkpoint.restore` when loading from a name-based checkpoint.

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
                    f"list. Got {op_list}")
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
        raise ValueError(
            f"At least two variables have the same name: {var.name}")
      names_to_saveables[var.name] = var
    elif isinstance(var, variables.Variable) and var._save_slice_info:
      name = var._save_slice_info.full_name
      if name in names_to_saveables:
        if not isinstance(names_to_saveables[name], list):
          raise ValueError("Mixing slices and non-slices with the same name: "
                           f"{name}")
        names_to_saveables[name].append(var)
      else:
        names_to_saveables[name] = [var]
    elif isinstance(var, trackable.Trackable) and not resource_or_ref_variable:
      trackable_saveables = [
          (factory() if callable(factory) else factory)
          for factory in (
              saveable_objects_from_trackable(var, tf1_saver=True).values())]
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
              f"is enabled. Got type: {type(var)}.")
        set_var = names_to_saveables.setdefault(var._shared_name, var)
        if set_var is not var:
          raise ValueError(
              "Two different ResourceVariable objects with the same "
              f"shared_name '{var._shared_name}' were passed to the Saver. This"
              " likely means that they were created in different Graphs or "
              "isolated contexts, and may not be checkpointed together.")
      else:
        if convert_variable_to_tensor:
          if isinstance(var, resource_variable_ops.BaseResourceVariable):
            var = var._graph_element  # pylint: disable=protected-access
          else:
            var = ops.convert_to_tensor(var, as_ref=True)
          if not _tensor_comes_from_variable(var):
            raise TypeError(f"Variable to save is not a Variable: {var}")
        if var.op.type == "ReadVariableOp":
          name = var.op.inputs[0].op.name
        else:
          name = var.op.name
        if name in names_to_saveables:
          raise ValueError(f"At least two variables have the same name: {name}")
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
    raise ValueError("The same saveable will be restored with two names: "
                     f"{saveable.name}")
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
  saveables = []
  seen_ops = object_identity.ObjectIdentitySet()
  for name, op in sorted(names_to_saveables.items(),
                         # Avoid comparing ops, sort only by name.
                         key=lambda x: x[0]):
    for converted_saveable_object in saveable_objects_for_op(op, name):
      _add_saveable(saveables, seen_ops, converted_saveable_object)
  return saveables


def validate_saveables_for_saved_model(saveables, obj):
  """Makes sure SaveableObjects are compatible with SavedModel."""
  if isinstance(obj, python_state.PythonState):
    logging.warn(
        f"Note that object {obj} stores python values into the checkpoint. "
        "These values will not be restored when loading the SavedModel "
        "into python.")
    return []
  if any(isinstance(saveable, trackable.NoRestoreSaveable)
         for saveable in saveables):
    return []
  return saveables


class RestoredSaveableObject(saveable_object.SaveableObject):
  """SaveableObject restored from SavedModel using the traced save/restore."""

  def __init__(self, names_and_slices, save_function, restore_function, name):
    self.save_function = save_function
    self.restore_function = restore_function

    if tensor_util.is_tf_type(name):
      name_tensor = name
    else:
      with ops.init_scope():
        name_tensor = constant_op.constant(name)
    tensors = save_function(name_tensor)
    specs = []
    for (str_name, str_slice), tensor_info in zip(names_and_slices, tensors):
      specs.append(saveable_object.SaveSpec(tensor_info["tensor"], str_slice,
                                            name + str_name))
    super(RestoredSaveableObject, self).__init__(None, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    del restored_shapes  # unused
    return self.restore_function(
        *[restored_tensors[i] for i in range(len(self.specs))])


def recreate_saveable_objects(saveable_fn_by_name, temp_session):
  """Returns a dict of SaveableObject factories generated from loaded fns."""

  names_and_slices = []

  with ops.init_scope():

    for save_fn, _ in saveable_fn_by_name.values():
      for tensor_info in save_fn(""):
        name = tensor_info["name"]
        slice_spec = tensor_info["slice_spec"]
        if not context.executing_eagerly():
          sess = ops.get_default_session()
          if sess is None:
            if temp_session[0] is not None:
              sess = temp_session[0]
            else:
              sess = temp_session[0] = session.Session()
          name, slice_spec = sess.run([name, slice_spec])
        names_and_slices.append((
            _convert_to_string(name),
            _convert_to_string(slice_spec)))

  saveable_factories = {}
  for name, (save_fn, restore_fn) in saveable_fn_by_name.items():
    saveable_factories[name] = functools.partial(
        RestoredSaveableObject,
        names_and_slices=names_and_slices,
        save_function=save_fn,
        restore_function=restore_fn)
  return saveable_factories


def create_saveable_object(name, key, factory, call_with_mapped_captures):
  """Creates a SaveableObject while potentially in a different graph.

  When creating the frozen saver for SavedModel, the save and restore ops are
  placed in a separate graph. Since RestoredSaveableObject uses tf.functions to
  save and restore, the function captures must be mapped to the new graph.

  Args:
    name: Name of SaveableObject factory.
    key: Checkpoint key of this SaveableObject.
    factory: Factory method for creating the SaveableObject.
    call_with_mapped_captures: Helper that calls a tf.function while remapping
      the captures.

  Returns:
    a SaveableObject.
  """
  if call_with_mapped_captures is None:
    return factory(name=key)
  if name == trackable_utils.SERIALIZE_TO_TENSORS_NAME:
    return factory(name=key,
                   call_with_mapped_captures=call_with_mapped_captures)
  elif is_factory_for_restored_saveable_object(factory):
    concrete_save_fn = factory.keywords["save_function"]

    def save_fn(name):
      return call_with_mapped_captures(concrete_save_fn, [name])

    concrete_restore_fn = factory.keywords["restore_function"]

    def restore_fn(*restored_tensors):
      return call_with_mapped_captures(concrete_restore_fn, restored_tensors)

    return factory(save_function=save_fn, restore_function=restore_fn,
                   name=key)
  else:
    return factory(name=key)


def is_factory_for_restored_saveable_object(factory):
  return (isinstance(factory, functools.partial) and
          factory.func is RestoredSaveableObject)


@tf_export("__internal__.tracking.saveable_objects_from_trackable", v1=[])
def saveable_objects_from_trackable(obj, tf1_saver=False):
  """Returns SaveableObject factory dict from a Trackable.

  Args:
    obj: A `Trackable`
    tf1_saver: Boolean, whether this is being called from a TF1 Saver (
        `tf.compat.v1.train.Saver`). When this is True, the SaveableObject will
        be generated from `obj`'s legacy `_gather_saveables_for_checkpoint` fn.
        When saving with TF2, `Trackable._serialize_from_tensors` is preferred.

  Returns:
    A dict mapping attribute names to SaveableObject factories (callables that
    produce a SaveableObject).
  """
  if isinstance(obj, python_state.PythonState):
    return {
        python_state.PYTHON_STATE:
            functools.partial(
                _PythonStringStateSaveable,
                state_callback=obj.serialize,
                restore_callback=obj.deserialize)
    }

  if tf1_saver:
    saveable_factories = obj._gather_saveables_for_checkpoint()  # pylint: disable=protected-access
    if saveable_factories:
      return saveable_factories

  if trackable_has_serialize_to_tensor(obj):

    def create_saveable(name="", call_with_mapped_captures=None):
      save_fn = obj._serialize_to_tensors  # pylint: disable=protected-access
      if (call_with_mapped_captures and
          isinstance(save_fn, core.ConcreteFunction)):
        tensor_dict = call_with_mapped_captures(save_fn, [])
      else:
        tensor_dict = save_fn()

      specs = []
      local_names = []
      for tensor_name, maybe_tensor in tensor_dict.items():
        local_names.append(tensor_name)

        if not isinstance(maybe_tensor, dict):
          maybe_tensor = {"": maybe_tensor}

        spec_name = name + trackable_utils.escape_local_name(tensor_name)
        # Create separate specs for each slice spec.
        for slice_spec, tensor in maybe_tensor.items():
          if isinstance(tensor, saveable_object.SaveSpec):
            spec = tensor
            spec.name = spec_name
            spec.slice_spec = slice_spec
          else:
            spec = saveable_object.SaveSpec(tensor, slice_spec, spec_name)
          specs.append(spec)

      return TrackableSaveable(
          obj=obj,
          specs=specs,
          name=name,
          local_names=local_names,
          prefix=saveable_compat.get_saveable_name(obj) or "",
          call_with_mapped_captures=call_with_mapped_captures)

    return {trackable_utils.SERIALIZE_TO_TENSORS_NAME: create_saveable}
  else:
    return obj._gather_saveables_for_checkpoint()  # pylint: disable=protected-access


class TrackableSaveable(saveable_object.SaveableObject):
  """A SaveableObject that defines `Trackable` checkpointing steps."""

  def __init__(self, obj, specs, name, local_names, prefix,
               call_with_mapped_captures=None):
    self._prefix = prefix
    self._local_names = local_names
    self._trackable = obj
    self._call_with_mapped_captures = call_with_mapped_captures
    super(TrackableSaveable, self).__init__(obj, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    del restored_shapes  # Unused.
    restored_tensor_dict = {}
    for n, local_name in enumerate(self._local_names):
      restored_tensor_dict[local_name] = restored_tensors[n]

    restore_fn = self._trackable._restore_from_tensors  # pylint: disable=protected-access

    # When restoring a RefVariable, call the restore function directly.
    # pylint: disable=protected-access
    if not ops.executing_eagerly_outside_functions() and any([
        spec._tensor.op.type in _REF_VARIABLE_OPS
        for spec in self.specs
        if isinstance(spec._tensor, ops.Tensor)]):
      return restore_fn(restored_tensor_dict)
    # pylint: enable=protected-access

    if (self._call_with_mapped_captures and
        isinstance(restore_fn, core.ConcreteFunction)):
      ret = self._call_with_mapped_captures(restore_fn, [restored_tensor_dict])
    else:
      ret = restore_fn(restored_tensor_dict)
    if ret is not None:
      return ret
    return gen_control_flow_ops.no_op()

  def get_proto_names_and_checkpoint_keys(self):
    return [(self._prefix + local_name, spec.name)
            for local_name, spec in zip(self._local_names, self.specs)]


class _PythonStringStateSaveable(saveable_object.SaveableObject):
  """Saves Python state in a checkpoint."""

  def __init__(self, name, state_callback, restore_callback):
    """Configure saving.

    Args:
      name: The checkpoint key to write to.
      state_callback: A function taking no arguments which returns a string.
        This function is run every time a checkpoint is written.
      restore_callback: A function taking a Python string, used to restore
        state.
    """

    def _state_callback_wrapper():
      with ops.init_scope():
        return state_callback()

    self._state_callback = _state_callback_wrapper
    self._restore_callback = restore_callback
    with ops.device("/cpu:0"):
      self._save_string = constant_op.constant("", dtype=dtypes.string)
    spec = saveable_object.SaveSpec(
        self._save_string, "", name, dtype=dtypes.string)
    super(_PythonStringStateSaveable, self).__init__(self._save_string, [spec],
                                                     name)

  def feed_dict_additions(self):
    """When running a graph, indicates fresh state to feed."""
    return {self._save_string: self._state_callback()}

  def freeze(self):
    """Create a frozen `SaveableObject` which saves the current state."""

    def _constant_state():
      return constant_op.constant(self._state_callback(), dtype=dtypes.string)

    return trackable.NoRestoreSaveable(
        tensor=_constant_state,
        dtype=dtypes.string,
        name=self.name,
        device="cpu:0")


def trackable_has_serialize_to_tensor(obj):
  """Returns whether obj's class has `_serialize_to_tensors` defined."""
  try:
    if "_serialize_to_tensors" in obj.__dict__:
      # In some cases (e.g. restored objects), the object may have
      # `_serialize_to_tensors` even if the class does not.
      return True
  except AttributeError:  # Data structure proxy wrappers don't have __dict__.
    pass

  # Use MRO so that if a parent class has `_serialize_to_tensors`, but the
  # object class has not yet been migrated, we'll continue to use the obj
  # class's `_gather_saveables_for_checkpoint` method.
  for t in type(obj).mro():
    if t is trackable.Trackable:
      # Base case. Return False since _serialize_to_tensors will raise a
      # NotImplemented Error.
      return False
    elif "_serialize_to_tensors" in t.__dict__:
      return True
    elif "_gather_saveables_for_checkpoint" in t.__dict__:
      return False
  return False


def _convert_to_string(x):
  return compat.as_str(tensor_util.constant_value(x))


class SaveableCompatibilityConverter(trackable.Trackable):
  """Converts object's `SaveableObjects` to functions used in TF2 checkpointing.

  A class that converts a Trackable object's `SaveableObjects` to save and
  restore functions with the same signatures as
  `Trackable._serialize_to_tensors` and `Trackable._restore_from_tensors`.
  This class also produces a method for filling the object proto.
  """

  __slots__ = ("_obj", "_saveables")

  def __init__(self, obj, saveables):
    """Constructor.

    Args:
      obj: A Trackable object.
      saveables: A list of saveables for `obj`.
    """
    self._obj = obj
    self._saveables = saveables

  @property
  def obj(self):
    return self._obj

  @property
  def saveables(self):
    """Returns a list of SaveableObjects generated from the Trackable object."""
    return self._saveables

  def _serialize_to_tensors(self):
    """Returns a dict of tensors to serialize."""
    return saveable_object_to_tensor_dict(self.saveables)

  def _restore_from_tensors(self, restored_tensors):
    """Returns the restore ops defined in the Saveables."""
    # Map restored tensors to the corresponding SaveableObjects, then call
    # restore. There must be an exact match between restored tensors and the
    # expected attributes.
    expected_keys = []
    for saveable in self.saveables:
      expected_keys.extend(
          trackable_utils.extract_local_name(_convert_to_string(spec.name))
          for spec in saveable.specs)
    if set(expected_keys) != restored_tensors.keys():
      raise ValueError(f"Could not restore object {self._obj} because not all "
                       "expected tensors were in the checkpoint."
                       f"\n\tExpected: {expected_keys}"
                       f"\n\tGot: {list(restored_tensors.keys())}")

    return saveable_object_to_restore_fn(self.saveables)(restored_tensors)


def saveable_object_to_tensor_dict(saveables):
  """Converts a list of SaveableObjects to a tensor dictionary."""
  tensor_dict = {}
  for saveable in saveables:
    for spec in saveable.specs:
      name = _convert_to_string(spec.name)
      slice_spec = _convert_to_string(spec.slice_spec)
      # Currently, tensor dict cannot handle callable tensor values (which
      # are needed for uninitialized variables), so keep using SaveSpec.
      tensor = spec if callable(spec._tensor) else spec._tensor  # pylint: disable=protected-access
      if slice_spec:
        tensor_dict.setdefault(name, {})[slice_spec] = tensor
      else:
        tensor_dict[name] = tensor
  return tensor_dict


def saveable_object_to_restore_fn(saveables):
  """Generates `Trackable._restore_from_tensors` from SaveableObjects."""

  def _restore_from_tensors(restored_tensors):
    restore_ops = {}

    for saveable in saveables:
      saveable_restored_tensors = []
      for spec in saveable.specs:
        name = trackable_utils.extract_local_name(_convert_to_string(spec.name))
        slice_spec = _convert_to_string(spec.slice_spec)

        maybe_tensor = restored_tensors[name]
        if not isinstance(maybe_tensor, dict):
          maybe_tensor = {"": maybe_tensor}

        saveable_restored_tensors.append(maybe_tensor[slice_spec])
      restore_ops[saveable.name] = saveable.restore(
          saveable_restored_tensors, restored_shapes=None)
    return restore_ops

  return _restore_from_tensors


def serialized_tensors_to_saveable_cache(serialized_tensors):
  """Converts a tensor dict to a SaveableObject cache.

  Args:
    serialized_tensors: Map from Trackable to a tensor dict. The tensor dict
      maps checkpoint key (-> slice_spec) -> Tensor

  Returns:
    A dict mapping Trackable objects to a map from local savable name to
    SaveableObject.
  """
  saveables_cache = object_identity.ObjectIdentityWeakKeyDictionary()

  for obj, tensor_dict in serialized_tensors.items():
    if not tensor_dict: continue
    if isinstance(obj, SaveableCompatibilityConverter):
      trackable_obj = obj.obj
      saveables_cache[trackable_obj] = {}
      for saveable in obj.saveables:
        local_name = trackable_utils.extract_local_name(saveable.name)
        saveables_cache[trackable_obj][local_name] = [saveable]
      continue

    specs = []
    # The local names and prefixes are computed to ensure that the generated
    # SaveableObject can call `Trackable._restore_from_tensors()`
    local_names = []
    prefix = saveable_compat.get_saveable_name(obj) or ""
    for checkpoint_key, maybe_tensor in tensor_dict.items():
      # Make sure that `maybe_tensor` is a dict from `slice_spec` to `tensor`.
      if not isinstance(maybe_tensor, dict):
        maybe_tensor = {"": maybe_tensor}

      for slice_spec, tensor in maybe_tensor.items():
        if isinstance(tensor, saveable_object.SaveSpec):
          specs.append(tensor)
        else:
          specs.append(saveable_object.SaveSpec(tensor,
                                                slice_spec,
                                                checkpoint_key))
      local_names.append(trackable_utils.extract_local_name(checkpoint_key,
                                                            prefix))

    object_name = trackable_utils.extract_object_name(
        next(iter(tensor_dict.keys())))
    saveables_cache[obj] = {
        trackable_utils.SERIALIZE_TO_TENSORS_NAME: [TrackableSaveable(
            obj, specs, object_name, local_names=local_names, prefix=prefix)]}
  return saveables_cache
