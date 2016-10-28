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

# pylint: disable=invalid-name
"""Save and restore variables."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import re
import time
import uuid

import numpy as np
import six

from google.protobuf.any_pb2 import Any
from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat


def ops_used_by_graph_def(graph_def):
  """Collect the list of ops used by a graph.

  Does not validate that the ops are all registered.

  Args:
    graph_def: A `GraphDef` proto, as from `graph.as_graph_def()`.

  Returns:
    A list of strings, each naming an op used by the graph.
  """
  # Map function names to definitions
  name_to_function = {}
  for fun in graph_def.library.function:
    name_to_function[fun.signature.name] = fun

  # Collect the list of op names.  Since functions can reference functions, we
  # need a recursive traversal.
  used_ops = set()  # Includes both primitive ops and functions
  functions_to_process = []  # A subset of used_ops

  def mark_op_as_used(op):
    if op not in used_ops and op in name_to_function:
      functions_to_process.append(name_to_function[op])
    used_ops.add(op)

  for node in graph_def.node:
    mark_op_as_used(node.op)
  while functions_to_process:
    fun = functions_to_process.pop()
    if fun.node_def:
      for node in fun.node_def:
        mark_op_as_used(node.op)
    else:  # TODO(josh11b): Eventually remove this case.
      for node in fun.node:
        mark_op_as_used(node.op)

  return [op for op in used_ops if op not in name_to_function]


def stripped_op_list_for_graph(graph_def):
  """Collect the stripped OpDefs for ops used by a graph.

  This function computes the `stripped_op_list` field of `MetaGraphDef` and
  similar protos.  The result can be communicated from the producer to the
  consumer, which can then use the C++ function
  `RemoveNewDefaultAttrsFromGraphDef` to improve forwards compatibility.

  Args:
    graph_def: A `GraphDef` proto, as from `graph.as_graph_def()`.

  Returns:
    An `OpList` of ops used by the graph.

  Raises:
    ValueError: If an unregistered op is used.
  """
  # This is the Python equivalent of StrippedOpListForGraph in C++.
  # Unfortunately, since the Python op registry can differ from that in C++, we
  # can't remove the duplication using swig (at least naively).
  # TODO(irving): Support taking graphs directly.

  used_ops = ops_used_by_graph_def(graph_def)

  # Verify that all used ops are registered.
  registered_ops = op_def_registry.get_registered_ops()
  # These internal ops used by functions are not registered, so we need to
  # whitelist them.  # TODO(irving): Do something better here.
  op_whitelist = ("_Arg", "_Retval", "_ListToArray", "_ArrayToList")
  for op in used_ops:
    if op not in registered_ops and op not in op_whitelist:
      raise ValueError("Op %s is used by the graph, but is not registered" % op)

  # Build the stripped op list in sorted order
  return op_def_pb2.OpList(op=[registered_ops[op] for op in sorted(used_ops)
                               if op in registered_ops])


class BaseSaverBuilder(object):
  """Base class for Savers.

  Can be extended to create different Ops.
  """

  _CHECKPOINT_FORMAT_VERSION = saver_pb2.SaverDef.V1

  class SaveSpec(object):
    """Class used to describe tensor slices that need to be saved."""

    def __init__(self, tensor, slice_spec, name):
      """Creates a `SaveSpec` object.

      Args:
        tensor: the tensor to save.
        slice_spec: the slice to be saved. See `Variable.SaveSliceInfo`.
        name: the name to save the tensor under.
      """
      self.tensor = tensor
      self.slice_spec = slice_spec
      self.name = name

  class SaveableObject(object):
    """Base class for saving and restoring saveable objects."""

    def __init__(self, op, specs, name):
      """Creates a `SaveableObject` object.

      Args:
        op: the "producer" object that this class wraps; it produces a list of
          tensors to save.  E.g., a "Variable" object saving its backing tensor.
        specs: a list of SaveSpec, each element of which describes one tensor to
          save under this object.
        name: the name to save the object under.
      """
      self.op = op
      self.specs = specs
      self.name = name
      # The device of this saveable. All tensors must be on the same device.
      self.device = specs[0].tensor.device

    def restore(self, restored_tensors, restored_shapes):
      """Restores this object from 'restored_tensors'.

      Args:
        restored_tensors: the tensors that were loaded from a checkpoint
        restored_shapes: the shapes this object should conform to after
          restore, or None.

      Returns:
        An operation that restores the state of the object.

      Raises:
        ValueError: If the object cannot be restored using the provided
          parameters.
      """
      # pylint: disable=unused-argument
      raise ValueError("Calling an abstract method.")

  class VariableSaveable(SaveableObject):
    """SaveableObject implementation that handles Variables."""

    def __init__(self, var, slice_spec, name):
      spec = BaseSaverBuilder.SaveSpec(var, slice_spec, name)
      super(BaseSaverBuilder.VariableSaveable, self).__init__(var, [spec], name)

    def restore(self, restored_tensors, restored_shapes):
      restored_tensor = restored_tensors[0]
      if restored_shapes is not None:
        restored_tensor = array_ops.reshape(restored_tensor, restored_shapes[0])
      return state_ops.assign(
          self.op,
          restored_tensor,
          validate_shape=restored_shapes is None and
          self.op.get_shape().is_fully_defined())

  def __init__(self):
    pass

  def save_op(self, filename_tensor, saveables):
    """Create an Op to save 'saveables'.

    This is intended to be overridden by subclasses that want to generate
    different Ops.

    Args:
      filename_tensor: String Tensor.
      saveables: A list of BaseSaverBuilder.SaveableObject objects.

    Returns:
      An Operation that save the variables.
    """
    # pylint: disable=protected-access
    tensor_names = []
    tensors = []
    tensor_slices = []
    for saveable in saveables:
      for spec in saveable.specs:
        tensor_names.append(spec.name)
        tensors.append(spec.tensor)
        tensor_slices.append(spec.slice_spec)

    return io_ops._save(
        filename=filename_tensor,
        tensor_names=tensor_names,
        tensors=tensors,
        tensor_slices=tensor_slices)

  def restore_op(self, filename_tensor, saveable, preferred_shard):
    """Create ops to restore 'saveable'.

    This is intended to be overridden by subclasses that want to generate
    different Ops.

    Args:
      filename_tensor: String Tensor.
      saveable: A BaseSaverBuilder.SaveableObject object.
      preferred_shard: Int.  Shard to open first when loading a sharded file.

    Returns:
      A list of Tensors resulting from reading 'saveable' from
        'filename'.
    """
    # pylint: disable=protected-access
    tensors = []
    for spec in saveable.specs:
      tensors.append(
          io_ops._restore_slice(
              filename_tensor,
              spec.name,
              spec.slice_spec,
              spec.tensor.dtype,
              preferred_shard=preferred_shard))

    return tensors

  def sharded_filename(self, filename_tensor, shard, num_shards):
    """Append sharding information to a filename.

    Args:
      filename_tensor: A string tensor.
      shard: Integer.  The shard for the filename.
      num_shards: An int Tensor for the number of shards.

    Returns:
      A string tensor.
    """
    # pylint: disable=protected-access
    return gen_io_ops._sharded_filename(filename_tensor, shard, num_shards)

  def _AddSaveOps(self, filename_tensor, saveables):
    """Add ops to save variables that are on the same shard.

    Args:
      filename_tensor: String Tensor.
      saveables: A list of SaveableObject objects.

    Returns:
      A tensor with the filename used to save.
    """
    save = self.save_op(filename_tensor, saveables)
    return control_flow_ops.with_dependencies([save], filename_tensor)

  def _AddShardedSaveOps(self, filename_tensor, per_device):
    """Add ops to save the params per shard.

    Args:
      filename_tensor: a scalar String Tensor.
      per_device: A list of (device, BaseSaverBuilder.SaveableObject) pairs, as
        returned by _GroupByDevices().

    Returns:
      An op to save the variables.
    """
    num_shards = len(per_device)
    sharded_saves = []
    num_shards_tensor = constant_op.constant(num_shards, name="num_shards")
    for shard, (device, saveables) in enumerate(per_device):
      with ops.device(device):
        sharded_filename = self.sharded_filename(filename_tensor, shard,
                                                 num_shards_tensor)
        sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
    # Return the sharded name for the save path.
    with ops.control_dependencies([x.op for x in sharded_saves]):
      # pylint: disable=protected-access
      return gen_io_ops._sharded_filespec(filename_tensor, num_shards_tensor)

  def _AddRestoreOps(self,
                     filename_tensor,
                     saveables,
                     restore_sequentially,
                     reshape,
                     preferred_shard=-1,
                     name="restore_all"):
    """Add operations to restore saveables.

    Args:
      filename_tensor: Tensor for the path of the file to load.
      saveables: A list of SaveableObject objects.
      restore_sequentially: True if we want to restore variables sequentially
        within a shard.
      reshape: True if we want to reshape loaded tensors to the shape of
        the corresponding variable.
      preferred_shard: Shard to open first when loading a sharded file.
      name: Name for the returned op.

    Returns:
      An Operation that restores the variables.
    """
    assign_ops = []
    for saveable in saveables:
      restore_control_inputs = assign_ops[-1:] if restore_sequentially else []
      # Load and optionally reshape on the CPU, as string tensors are not
      # available on the GPU.
      # TODO(touts): Re-enable restore on GPU when we can support annotating
      # string tensors as "HostMemory" inputs.
      with ops.device(
          graph_util.set_cpu0(saveable.device) if saveable.device else None):
        with ops.control_dependencies(restore_control_inputs):
          tensors = self.restore_op(filename_tensor, saveable, preferred_shard)
          shapes = None
          if reshape:
            # Compute the shapes, let the restore op decide if and how to do
            # the reshape.
            shapes = []
            for spec in saveable.specs:
              v = spec.tensor
              shape = v.get_shape()
              if not shape.is_fully_defined():
                shape = array_ops.shape(v)
              shapes.append(shape)
          assign_ops.append(saveable.restore(tensors, shapes))

      # Create a Noop that has control dependencies from all the updates.
    return control_flow_ops.group(*assign_ops, name=name)

  def _AddShardedRestoreOps(self, filename_tensor, per_device,
                            restore_sequentially, reshape):
    """Add Ops to restore variables from multiple devices.

    Args:
      filename_tensor: Tensor for the path of the file to load.
      per_device: A list of (device, SaveableObject) pairs, as
        returned by _GroupByDevices().
      restore_sequentially: True if we want to restore variables sequentially
        within a shard.
      reshape: True if we want to reshape loaded tensors to the shape of
        the corresponding variable.

    Returns:
      An Operation that restores the variables.
    """
    sharded_restores = []
    for shard, (device, saveables) in enumerate(per_device):
      with ops.device(device):
        sharded_restores.append(
            self._AddRestoreOps(
                filename_tensor,
                saveables,
                restore_sequentially,
                reshape,
                preferred_shard=shard,
                name="restore_shard"))
    return control_flow_ops.group(*sharded_restores, name="restore_all")

  @staticmethod
  def _IsVariable(v):
    return isinstance(v, ops.Tensor) and (v.op.type == "Variable" or
                                          v.op.type == "AutoReloadVariable")

  def _GroupByDevices(self, saveables):
    """Group Variable tensor slices per device.

    TODO(touts): Make sure that all the devices found are on different
    job/replica/task/cpu|gpu.  It would be bad if 2 were on the same device.
    It can happen if the devices are unspecified.

    Args:
      saveables: A list of BaseSaverBuilder.SaveableObject objects.

    Returns:
      A list of tuples: (device_name, BaseSaverBuilder.SaveableObject) tuples.
      The list is sorted by ascending device_name.

    Raises:
      ValueError: If the tensors of a saveable are on different devices.
    """
    per_device = collections.defaultdict(lambda: [])
    for saveable in saveables:
      canonical_device = set(
          pydev.canonical_name(spec.tensor.device) for spec in saveable.specs)
      if len(canonical_device) != 1:
        raise ValueError("All tensors of a saveable object must be "
                         "on the same device: %s" % saveable.name)
      per_device[canonical_device.pop()].append(saveable)
    return sorted(per_device.items(), key=lambda t: t[0])

  @staticmethod
  def OpListToDict(op_list):
    """Create a dictionary of names to operation lists.

    Args:
      op_list: A list, tuple, or set of Variables or SaveableObjects.

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
    op_list = set(op_list)
    names_to_saveables = {}
    # pylint: disable=protected-access
    for var in op_list:
      if isinstance(var, BaseSaverBuilder.SaveableObject):
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
      else:
        var = ops.convert_to_tensor(var, as_ref=True)
        if not BaseSaverBuilder._IsVariable(var):
          raise TypeError("Variable to save is not a Variable: %s" % var)
        name = var.op.name
        if name in names_to_saveables:
          raise ValueError("At least two variables have the same name: %s" %
                           name)
        names_to_saveables[name] = var
      # pylint: enable=protected-access
    return names_to_saveables

  def _ValidateAndSliceInputs(self, names_to_saveables):
    """Returns the variables and names that will be used for a Saver.

    Args:
      names_to_saveables: A dict (k, v) where k is the name of an operation and
         v is an operation to save or a BaseSaverBuilder.Saver.

    Returns:
      A list of BaseSaverBuilder.SaveableObject objects.

    Raises:
      TypeError: If any of the keys are not strings or any of the
        values are not one of Tensor or Variable or a checkpointable operation.
      ValueError: If the same operation is given in more than one value
        (this also applies to slices of SlicedVariables).
    """
    if not isinstance(names_to_saveables, dict):
      names_to_saveables = BaseSaverBuilder.OpListToDict(names_to_saveables)

    saveables = []
    seen_ops = set()
    for name in sorted(names_to_saveables.keys()):
      if not isinstance(name, six.string_types):
        raise TypeError(
            "names_to_saveables must be a dict mapping string names to "
            "checkpointable operations. Name is not a string: %s" % name)
      op = names_to_saveables[name]
      if isinstance(op, BaseSaverBuilder.SaveableObject):
        self._AddSaveable(saveables, seen_ops, op)
      elif isinstance(op, (list, tuple)):
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
          saveable = BaseSaverBuilder.VariableSaveable(
              variable, variable._save_slice_info.spec, name)
          self._AddSaveable(saveables, seen_ops, saveable)
        # pylint: enable=protected-access
      else:
        # A variable or tensor.
        variable = ops.convert_to_tensor(op, as_ref=True)
        if not BaseSaverBuilder._IsVariable(variable):
          raise TypeError("names_to_saveables must be a dict mapping string "
                          "names to Tensors/Variables. Not a variable: %s" %
                          variable)
        saveable = BaseSaverBuilder.VariableSaveable(variable, "", name)
        self._AddSaveable(saveables, seen_ops, saveable)
    return saveables

  def _AddSaveable(self, saveables, seen_ops, saveable):
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

  def build(self,
            names_to_saveables,
            reshape=False,
            sharded=False,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            filename="model"):
    """Adds save/restore nodes to the graph and creates a SaverDef proto.

    Args:
      names_to_saveables: A dictionary mapping name to a Variable or
        SaveableObject. Each name will be associated with the
        corresponding variable in the checkpoint.
      reshape: If True, allow restoring parameters from a checkpoint
        that where the parameters have a different shape.  This is
        only needed when you try to restore from a Dist-Belief checkpoint,
        and only some times.
      sharded: If True, shard the checkpoints, one per device that has
        Variable nodes.
      max_to_keep: Maximum number of checkpoints to keep.  As new checkpoints
        are created, old ones are deleted.  If None or 0, no checkpoints are
        deleted from the filesystem but only the last one is kept in the
        `checkpoint` file.  Presently the number is only roughly enforced.  For
        example in case of restarts more than max_to_keep checkpoints may be
        kept.
      keep_checkpoint_every_n_hours: How often checkpoints should be kept.
        Defaults to 10,000 hours.
      name: String.  Optional name to use as a prefix when adding operations.
      restore_sequentially: A Bool, which if true, causes restore of different
        variables to happen sequentially within each device.
      filename: If known at graph construction time, filename used for variable
        loading/saving.

    Returns:
      A SaverDef proto.

    Raises:
      TypeError: If 'names_to_saveables' is not a dictionary mapping string
        keys to variable Tensors.
      ValueError: If any of the keys or values in 'names_to_saveables' is not
        unique.
    """
    saveables = self._ValidateAndSliceInputs(names_to_saveables)
    if max_to_keep is None:
      max_to_keep = 0

    with ops.name_scope(name, "save",
                        [saveable.op for saveable in saveables]) as name:
      # Add the Constant string tensor for the filename.
      filename_tensor = constant_op.constant(filename)

      # Add the save ops.
      if sharded:
        per_device = self._GroupByDevices(saveables)
        save_tensor = self._AddShardedSaveOps(filename_tensor, per_device)
        restore_op = self._AddShardedRestoreOps(filename_tensor, per_device,
                                                restore_sequentially, reshape)
      else:
        save_tensor = self._AddSaveOps(filename_tensor, saveables)
        restore_op = self._AddRestoreOps(filename_tensor, saveables,
                                         restore_sequentially, reshape)

    # In the following use case, it's possible to have restore_ops be called
    # something else:
    # - Build inference graph and export a meta_graph.
    # - Import the inference meta_graph
    # - Extend the inference graph to a train graph.
    # - Export a new meta_graph.
    # Now the second restore_op will be called "restore_all_1".
    # As such, comment out the assert for now until we know whether supporting
    # such usage model makes sense.
    #
    # assert restore_op.name.endswith("restore_all"), restore_op.name

    return saver_pb2.SaverDef(
        filename_tensor_name=filename_tensor.name,
        save_tensor_name=save_tensor.name,
        restore_op_name=restore_op.name,
        max_to_keep=max_to_keep,
        sharded=sharded,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        version=self._CHECKPOINT_FORMAT_VERSION)


def _GetCheckpointFilename(save_dir, latest_filename):
  """Returns a filename for storing the CheckpointState.

  Args:
    save_dir: The directory for saving and restoring checkpoints.
    latest_filename: Name of the file in 'save_dir' that is used
      to store the CheckpointState.

  Returns:
    The path of the file that contains the CheckpointState proto.
  """
  if latest_filename is None:
    latest_filename = "checkpoint"
  return os.path.join(save_dir, latest_filename)


def generate_checkpoint_state_proto(save_dir,
                                    model_checkpoint_path,
                                    all_model_checkpoint_paths=None):
  """Generates a checkpoint state proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.

  Returns:
    CheckpointState proto with model_checkpoint_path and
    all_model_checkpoint_paths updated to either absolute paths or
    relative paths to the current save_dir.
  """
  if all_model_checkpoint_paths is None:
    all_model_checkpoint_paths = []

  if (not all_model_checkpoint_paths or
      all_model_checkpoint_paths[-1] != model_checkpoint_path):
    logging.info("%s is not in all_model_checkpoint_paths. Manually adding it.",
                 model_checkpoint_path)
    all_model_checkpoint_paths.append(model_checkpoint_path)

  # Relative paths need to be rewritten to be relative to the "save_dir"
  # if model_checkpoint_path already contains "save_dir".
  if not os.path.isabs(save_dir):
    if not os.path.isabs(model_checkpoint_path):
      model_checkpoint_path = os.path.relpath(model_checkpoint_path, save_dir)
    for i in range(len(all_model_checkpoint_paths)):
      p = all_model_checkpoint_paths[i]
      if not os.path.isabs(p):
        all_model_checkpoint_paths[i] = os.path.relpath(p, save_dir)

  coord_checkpoint_proto = CheckpointState(
      model_checkpoint_path=model_checkpoint_path,
      all_model_checkpoint_paths=all_model_checkpoint_paths)

  return coord_checkpoint_proto


def update_checkpoint_state(save_dir,
                            model_checkpoint_path,
                            all_model_checkpoint_paths=None,
                            latest_filename=None):
  """Updates the content of the 'checkpoint' file.

  This updates the checkpoint file containing a CheckpointState
  proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.

  Raises:
    RuntimeError: If the save paths conflict.
  """
  # Writes the "checkpoint" file for the coordinator for later restoration.
  coord_checkpoint_filename = _GetCheckpointFilename(save_dir, latest_filename)
  ckpt = generate_checkpoint_state_proto(
      save_dir,
      model_checkpoint_path,
      all_model_checkpoint_paths=all_model_checkpoint_paths)

  if coord_checkpoint_filename == ckpt.model_checkpoint_path:
    raise RuntimeError("Save path '%s' conflicts with path used for "
                       "checkpoint state.  Please use a different save path." %
                       model_checkpoint_path)

  # Saves to a tmp file first.  On success, *atomically* renames it back.
  # This prevents a potential read/write race between this function and
  # get_checkpoint_state().
  temp_pathname = coord_checkpoint_filename + ".tmp." + uuid.uuid4().hex
  file_io.write_string_to_file(temp_pathname, text_format.MessageToString(ckpt))
  file_io.rename(temp_pathname, coord_checkpoint_filename, overwrite=True)


def get_checkpoint_state(checkpoint_dir, latest_filename=None):
  """Returns CheckpointState proto from the "checkpoint" file.

  If the "checkpoint" file contains a valid CheckpointState
  proto, returns it.

  Args:
    checkpoint_dir: The directory of checkpoints.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.

  Returns:
    A CheckpointState if the state was available, None
    otherwise.

  Raises:
    ValueError: if the checkpoint read doesn't have model_checkpoint_path set.
  """
  ckpt = None
  coord_checkpoint_filename = _GetCheckpointFilename(checkpoint_dir,
                                                     latest_filename)
  f = None
  try:
    # Check that the file exists before opening it to avoid
    # many lines of errors from colossus in the logs.
    if file_io.file_exists(coord_checkpoint_filename):
      file_content = file_io.read_file_to_string(
          coord_checkpoint_filename).decode("utf-8")
      ckpt = CheckpointState()
      text_format.Merge(file_content, ckpt)
      if not ckpt.model_checkpoint_path:
        raise ValueError("Invalid checkpoint state loaded from %s",
                         checkpoint_dir)
      # For relative model_checkpoint_path and all_model_checkpoint_paths,
      # prepend checkpoint_dir.
      if not os.path.isabs(ckpt.model_checkpoint_path):
        ckpt.model_checkpoint_path = os.path.join(checkpoint_dir,
                                                  ckpt.model_checkpoint_path)
      for i in range(len(ckpt.all_model_checkpoint_paths)):
        p = ckpt.all_model_checkpoint_paths[i]
        if not os.path.isabs(p):
          ckpt.all_model_checkpoint_paths[i] = os.path.join(checkpoint_dir, p)
  except IOError as e:
    # It's ok if the file cannot be read
    logging.warning(str(e))
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  except text_format.ParseError as e:
    logging.warning(str(e))
    logging.warning("%s: Checkpoint ignored", coord_checkpoint_filename)
    return None
  finally:
    if f:
      f.close()
  return ckpt


class Saver(object):
  """Saves and restores variables.

  See [Variables](../../how_tos/variables/index.md)
  for an overview of variables, saving and restoring.

  The `Saver` class adds ops to save and restore variables to and from
  *checkpoints*.  It also provides convenience methods to run these ops.

  Checkpoints are binary files in a proprietary format which map variable names
  to tensor values.  The best way to examine the contents of a checkpoint is to
  load it using a `Saver`.

  Savers can automatically number checkpoint filenames with a provided counter.
  This lets you keep multiple checkpoints at different steps while training a
  model.  For example you can number the checkpoint filenames with the training
  step number.  To avoid filling up disks, savers manage checkpoint files
  automatically. For example, they can keep only the N most recent files, or
  one checkpoint for every N hours of training.

  You number checkpoint filenames by passing a value to the optional
  `global_step` argument to `save()`:

  ```python
  saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
  ...
  saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
  ```

  Additionally, optional arguments to the `Saver()` constructor let you control
  the proliferation of checkpoint files on disk:

  * `max_to_keep` indicates the maximum number of recent checkpoint files to
    keep.  As new files are created, older files are deleted.  If None or 0,
    all checkpoint files are kept.  Defaults to 5 (that is, the 5 most recent
    checkpoint files are kept.)

  * `keep_checkpoint_every_n_hours`: In addition to keeping the most recent
    `max_to_keep` checkpoint files, you might want to keep one checkpoint file
    for every N hours of training.  This can be useful if you want to later
    analyze how a model progressed during a long training session.  For
    example, passing `keep_checkpoint_every_n_hours=2` ensures that you keep
    one checkpoint file for every 2 hours of training.  The default value of
    10,000 hours effectively disables the feature.

  Note that you still have to call the `save()` method to save the model.
  Passing these arguments to the constructor will not save variables
  automatically for you.

  A training program that saves regularly looks like:

  ```python
  ...
  # Create a saver.
  saver = tf.train.Saver(...variables...)
  # Launch the graph and train, saving the model every 1,000 steps.
  sess = tf.Session()
  for step in xrange(1000000):
      sess.run(..training_op..)
      if step % 1000 == 0:
          # Append the step number to the checkpoint name:
          saver.save(sess, 'my-model', global_step=step)
  ```

  In addition to checkpoint files, savers keep a protocol buffer on disk with
  the list of recent checkpoints. This is used to manage numbered checkpoint
  files and by `latest_checkpoint()`, which makes it easy to discover the path
  to the most recent checkpoint. That protocol buffer is stored in a file named
  'checkpoint' next to the checkpoint files.

  If you create several savers, you can specify a different filename for the
  protocol buffer file in the call to `save()`.

  @@__init__
  @@save
  @@restore

  Other utility methods.

  @@last_checkpoints
  @@set_last_checkpoints_with_time
  @@recover_last_checkpoints
  @@as_saver_def
  """

  def __init__(self,
               var_list=None,
               reshape=False,
               sharded=False,
               max_to_keep=5,
               keep_checkpoint_every_n_hours=10000.0,
               name=None,
               restore_sequentially=False,
               saver_def=None,
               builder=None,
               defer_build=False,
               allow_empty=False):
    """Creates a `Saver`.

    The constructor adds ops to save and restore variables.

    `var_list` specifies the variables that will be saved and restored. It can
    be passed as a `dict` or a list:

    * A `dict` of names to variables: The keys are the names that will be
      used to save or restore the variables in the checkpoint files.
    * A list of variables: The variables will be keyed with their op name in
      the checkpoint files.

    For example:

    ```python
    v1 = tf.Variable(..., name='v1')
    v2 = tf.Variable(..., name='v2')

    # Pass the variables as a dict:
    saver = tf.train.Saver({'v1': v1, 'v2': v2})

    # Or pass them as a list.
    saver = tf.train.Saver([v1, v2])
    # Passing a list is equivalent to passing a dict with the variable op names
    # as keys:
    saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
    ```

    The optional `reshape` argument, if `True`, allows restoring a variable from
    a save file where the variable had a different shape, but the same number
    of elements and type.  This is useful if you have reshaped a variable and
    want to reload it from an older checkpoint.

    The optional `sharded` argument, if `True`, instructs the saver to shard
    checkpoints per device.

    Args:
      var_list: A list of `Variable`/`SaveableObject`, or a dictionary mapping
        names to `SaveableObject`s. If `None`, defaults to the list of all
        saveable objects.
      reshape: If `True`, allows restoring parameters from a checkpoint
        where the variables have a different shape.
      sharded: If `True`, shard the checkpoints, one per device.
      max_to_keep: Maximum number of recent checkpoints to keep.
        Defaults to 5.
      keep_checkpoint_every_n_hours: How often to keep checkpoints.
        Defaults to 10,000 hours.
      name: String.  Optional name to use as a prefix when adding operations.
      restore_sequentially: A `Bool`, which if true, causes restore of different
        variables to happen sequentially within each device.  This can lower
        memory usage when restoring very large models.
      saver_def: Optional `SaverDef` proto to use instead of running the
        builder. This is only useful for specialty code that wants to recreate
        a `Saver` object for a previously built `Graph` that had a `Saver`.
        The `saver_def` proto should be the one returned by the
        `as_saver_def()` call of the `Saver` that was created for that `Graph`.
      builder: Optional `SaverBuilder` to use if a `saver_def` was not provided.
        Defaults to `BaseSaverBuilder()`.
      defer_build: If `True`, defer adding the save and restore ops to the
        `build()` call. In that case `build()` should be called before
        finalizing the graph or using the saver.
      allow_empty: If `False` (default) raise an error if there are no
        variables in the graph. Otherwise, construct the saver anyway and make
        it a no-op.

    Raises:
      TypeError: If `var_list` is invalid.
      ValueError: If any of the keys or values in `var_list` are not unique.
    """
    if defer_build and var_list:
      raise ValueError(
          "If `var_list` is provided then build cannot be deferred. "
          "Either set defer_build=False or var_list=None.")
    self._var_list = var_list
    self._reshape = reshape
    self._sharded = sharded
    self._max_to_keep = max_to_keep
    self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self._name = name
    self._restore_sequentially = restore_sequentially
    self.saver_def = saver_def
    self._builder = builder
    self._is_built = False
    self._allow_empty = allow_empty
    self._is_empty = None
    if not defer_build:
      self.build()
    if self.saver_def:
      self._check_saver_def()

  def build(self):
    """Builds saver_def."""
    if self._is_built:
      return
    self._is_built = True
    if not self.saver_def:
      if self._builder is None:
        self._builder = BaseSaverBuilder()
      if self._var_list is None:
        # pylint: disable=protected-access
        self._var_list = variables._all_saveable_objects()
      if not self._var_list:
        if self._allow_empty:
          self._is_empty = True
          return
        else:
          raise ValueError("No variables to save")
      self._is_empty = False
      self.saver_def = self._builder.build(
          self._var_list,
          reshape=self._reshape,
          sharded=self._sharded,
          max_to_keep=self._max_to_keep,
          keep_checkpoint_every_n_hours=self._keep_checkpoint_every_n_hours,
          name=self._name,
          restore_sequentially=self._restore_sequentially)
    self._check_saver_def()
    # Updates next checkpoint time.
    self._next_checkpoint_time = (
        time.time() + self.saver_def.keep_checkpoint_every_n_hours * 3600)
    self._last_checkpoints = []

  def _check_saver_def(self):
    if not isinstance(self.saver_def, saver_pb2.SaverDef):
      raise ValueError("saver_def must be a saver_pb2.SaverDef: %s" %
                       self.saver_def)
    if not self.saver_def.save_tensor_name:
      raise ValueError("saver_def must specify the save_tensor_name: %s" %
                       str(self.saver_def))
    if not self.saver_def.restore_op_name:
      raise ValueError("saver_def must specify the restore_op_name: %s" %
                       str(self.saver_def))

  def _CheckpointFilename(self, p):
    """Returns the checkpoint filename given a `(filename, time)` pair.

    Args:
      p: (filename, time) pair.

    Returns:
      Checkpoint file name.
    """
    name, _ = p
    return name

  def _MetaGraphFilename(self, checkpoint_filename, meta_graph_suffix="meta"):
    """Returns the meta graph filename.

    Args:
      checkpoint_filename: Name of the checkpoint file.
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.

    Returns:
      MetaGraph file name.
    """
    # If the checkpoint_filename is sharded, the checkpoint_filename could
    # be of format model.ckpt-step#-?????-of-shard#. For example,
    # model.ckpt-123456-?????-of-00005, or model.ckpt-123456-00001-of-00002.
    basename = re.sub(r"-[\d\?]+-of-\d+$", "", checkpoint_filename)
    meta_graph_filename = ".".join([basename, meta_graph_suffix])
    return meta_graph_filename

  def _MaybeDeleteOldCheckpoints(self,
                                 latest_save_path,
                                 meta_graph_suffix="meta"):
    """Deletes old checkpoints if necessary.

    Always keep the last `max_to_keep` checkpoints.  If
    `keep_checkpoint_every_n_hours` was specified, keep an additional checkpoint
    every `N` hours. For example, if `N` is 0.5, an additional checkpoint is
    kept for every 0.5 hours of training; if `N` is 10, an additional
    checkpoint is kept for every 10 hours of training.

    Args:
      latest_save_path: Name including path of checkpoint file to save.
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
    """
    if not self.saver_def.max_to_keep:
      return
    # Remove first from list if the same name was used before.
    for p in self._last_checkpoints:
      if latest_save_path == self._CheckpointFilename(p):
        self._last_checkpoints.remove(p)
    # Append new path to list
    self._last_checkpoints.append((latest_save_path, time.time()))
    # If more than max_to_keep, remove oldest.
    if len(self._last_checkpoints) > self.saver_def.max_to_keep:
      p = self._last_checkpoints.pop(0)
      # Do not delete the file if we keep_checkpoint_every_n_hours is set and we
      # have reached N hours of training.
      should_keep = p[1] > self._next_checkpoint_time
      if should_keep:
        self._next_checkpoint_time += (
            self.saver_def.keep_checkpoint_every_n_hours * 3600)
        return

      # Otherwise delete the files.
      try:
        checkpoint_prefix = self._CheckpointFilename(p)
        self._delete_file_if_exists(
            self._MetaGraphFilename(checkpoint_prefix, meta_graph_suffix))
        if self.saver_def.version == saver_pb2.SaverDef.V2:
          # V2 has a metadata file and some data files.
          self._delete_file_if_exists(checkpoint_prefix + ".index")
          self._delete_file_if_exists(checkpoint_prefix +
                                      ".data-?????-of-?????")
        else:
          # V1, Legacy.  Exact match on the data file.
          self._delete_file_if_exists(checkpoint_prefix)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("Ignoring: %s", str(e))

  def _delete_file_if_exists(self, filespec):
    for pathname in file_io.get_matching_files(filespec):
      file_io.delete_file(pathname)

  def as_saver_def(self):
    """Generates a `SaverDef` representation of this saver.

    Returns:
      A `SaverDef` proto.
    """
    return self.saver_def

  def to_proto(self):
    """Converts this `Saver` to a `SaverDef` protocol buffer.

    Returns:
      A `SaverDef` protocol buffer.
    """
    return self.saver_def

  @staticmethod
  def from_proto(saver_def):
    """Returns a `Saver` object created from `saver_def`."""
    return Saver(saver_def=saver_def)

  @property
  def last_checkpoints(self):
    """List of not-yet-deleted checkpoint filenames.

    You can pass any of the returned values to `restore()`.

    Returns:
      A list of checkpoint filenames, sorted from oldest to newest.
    """
    return list(self._CheckpointFilename(p) for p in self._last_checkpoints)

  def set_last_checkpoints(self, last_checkpoints):
    """DEPRECATED: Use set_last_checkpoints_with_time.

    Sets the list of old checkpoint filenames.

    Args:
      last_checkpoints: A list of checkpoint filenames.

    Raises:
      AssertionError: If last_checkpoints is not a list.
    """
    assert isinstance(last_checkpoints, list)
    # We use a timestamp of +inf so that this checkpoint will never be
    # deleted.  This is both safe and backwards compatible to a previous
    # version of the code which used s[1] as the "timestamp".
    self._last_checkpoints = [(s, np.inf) for s in last_checkpoints]

  def set_last_checkpoints_with_time(self, last_checkpoints_with_time):
    """Sets the list of old checkpoint filenames and timestamps.

    Args:
      last_checkpoints_with_time: A list of tuples of checkpoint filenames and
        timestamps.

    Raises:
      AssertionError: If last_checkpoints_with_time is not a list.
    """
    assert isinstance(last_checkpoints_with_time, list)
    self._last_checkpoints = last_checkpoints_with_time

  def recover_last_checkpoints(self, checkpoint_paths):
    """Recovers the internal saver state after a crash.

    This method is useful for recovering the "self._last_checkpoints" state.

    Globs for the checkpoints pointed to by `checkpoint_paths`.  If the files
    exist, use their mtime as the checkpoint timestamp.

    Args:
      checkpoint_paths: a list of checkpoint paths.
    """
    last_checkpoints = []
    for checkpoint_prefix in checkpoint_paths:
      pathname = _prefix_to_checkpoint_path(checkpoint_prefix,
                                            self.saver_def.version)
      fnames = file_io.get_matching_files(pathname)
      if fnames:
        mtime = int(file_io.stat(fnames[0]).mtime_nsec / 1e9)
        last_checkpoints.append((checkpoint_prefix, mtime))
    self.set_last_checkpoints_with_time(last_checkpoints)

  def save(self,
           sess,
           save_path,
           global_step=None,
           latest_filename=None,
           meta_graph_suffix="meta",
           write_meta_graph=True):
    """Saves variables.

    This method runs the ops added by the constructor for saving variables.
    It requires a session in which the graph was launched.  The variables to
    save must also have been initialized.

    The method returns the path of the newly created checkpoint file.  This
    path can be passed directly to a call to `restore()`.

    Args:
      sess: A Session to use to save the variables.
      save_path: String.  Path to the checkpoint filename.  If the saver is
        `sharded`, this is the prefix of the sharded checkpoint filename.
      global_step: If provided the global step number is appended to
        `save_path` to create the checkpoint filename. The optional argument
        can be a `Tensor`, a `Tensor` name or an integer.
      latest_filename: Optional name for the protocol buffer file that will
        contains the list of most recent checkpoint filenames.  That file,
        kept in the same directory as the checkpoint files, is automatically
        managed by the saver to keep track of recent checkpoints.  Defaults to
        'checkpoint'.
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
      write_meta_graph: `Boolean` indicating whether or not to write the meta
        graph file.

    Returns:
      A string: path at which the variables were saved.  If the saver is
        sharded, this string ends with: '-?????-of-nnnnn' where 'nnnnn'
        is the number of shards created.
      If the saver is empty, returns None.

    Raises:
      TypeError: If `sess` is not a `Session`.
      ValueError: If `latest_filename` contains path components, or if it
        collides with `save_path`.
      RuntimeError: If save and restore ops weren't built.
    """
    if not self._is_built:
      raise RuntimeError(
          "`build()` should be called before save if defer_build==True")
    if latest_filename is None:
      latest_filename = "checkpoint"

    if os.path.split(latest_filename)[0]:
      raise ValueError("'latest_filename' must not contain path components")

    if global_step is not None:
      if not isinstance(global_step, compat.integral_types):
        global_step = training_util.global_step(sess, global_step)
      checkpoint_file = "%s-%d" % (save_path, global_step)
    else:
      checkpoint_file = save_path
      if os.path.basename(
          save_path) == latest_filename and not self.saver_def.sharded:
        # Guard against collision between data file and checkpoint state file.
        raise ValueError(
            "'latest_filename' collides with 'save_path': '%s' and '%s'" %
            (latest_filename, save_path))

    if not gfile.IsDirectory(os.path.dirname(save_path)):
      raise ValueError(
          "Parent directory of {} doesn't exist, can't save.".format(save_path))

    save_path = os.path.dirname(save_path)
    if not isinstance(sess, session.SessionInterface):
      raise TypeError("'sess' must be a Session; %s" % sess)

    if not self._is_empty:
      model_checkpoint_path = sess.run(
          self.saver_def.save_tensor_name,
          {self.saver_def.filename_tensor_name: checkpoint_file})
      model_checkpoint_path = compat.as_str(model_checkpoint_path)
      self._MaybeDeleteOldCheckpoints(
          model_checkpoint_path, meta_graph_suffix=meta_graph_suffix)
      update_checkpoint_state(save_path, model_checkpoint_path,
                              self.last_checkpoints, latest_filename)

    if write_meta_graph:
      meta_graph_filename = self._MetaGraphFilename(
          checkpoint_file, meta_graph_suffix=meta_graph_suffix)
      with sess.graph.as_default():
        self.export_meta_graph(meta_graph_filename)

    if self._is_empty:
      return None
    else:
      return model_checkpoint_path

  def export_meta_graph(self,
                        filename=None,
                        collection_list=None,
                        as_text=False):
    """Writes `MetaGraphDef` to save_path/filename.

    Args:
      filename: Optional meta_graph filename including the path.
      collection_list: List of string keys to collect.
      as_text: If `True`, writes the meta_graph as an ASCII proto.

    Returns:
      A `MetaGraphDef` proto.
    """
    return export_meta_graph(
        filename=filename,
        graph_def=ops.get_default_graph().as_graph_def(add_shapes=True),
        saver_def=self.saver_def,
        collection_list=collection_list,
        as_text=as_text)

  def restore(self, sess, save_path):
    """Restores previously saved variables.

    This method runs the ops added by the constructor for restoring variables.
    It requires a session in which the graph was launched.  The variables to
    restore do not have to have been initialized, as restoring is itself a way
    to initialize variables.

    The `save_path` argument is typically a value previously returned from a
    `save()` call, or a call to `latest_checkpoint()`.

    Args:
      sess: A `Session` to use to restore the parameters.
      save_path: Path where parameters were previously saved.

    Raises:
      ValueError: If the given `save_path` does not point to a file.
    """
    if self._is_empty:
      return

    # Performs this check only for V1, as the V2 restore op can read either a
    # V1 ckpt or a V2 ckpt, making this check invalid.
    if self.saver_def.version == saver_pb2.SaverDef.V1:
      file_path = _prefix_to_checkpoint_path(save_path, self.saver_def.version)
      if not file_io.get_matching_files(file_path):
        raise ValueError("Restore called with invalid save path: %r. "
                         "File path is: %r" % (save_path, file_path))

    sess.run(self.saver_def.restore_op_name,
             {self.saver_def.filename_tensor_name: save_path})

  @staticmethod
  def _add_collection_def(meta_graph_def, key):
    """Adds a collection to MetaGraphDef protocol buffer.

    Args:
      meta_graph_def: MetaGraphDef protocol buffer.
      key: One of the GraphKeys or user-defined string.
    """
    _add_collection_def(meta_graph_def, key)


def _prefix_to_checkpoint_path(prefix, format_version=saver_pb2.SaverDef.V1):
  """Returns the pathname of a checkpoint file, given the checkpoint prefix.

  For V1 checkpoint, simply returns the prefix itself (the data file).  For V2,
  returns the pathname to the index file.

  Args:
    prefix: a string, the prefix of a checkpoint.
    format_version: the checkpoint format version that corresponds to the
      prefix.
  Returns:
    The pathname of a checkpoint file, taking into account the checkpoint
      format version.
  """
  if format_version == saver_pb2.SaverDef.V2:
    return prefix + ".index"  # The index file identifies a checkpoint.
  return prefix  # Just the data file.


def latest_checkpoint(checkpoint_dir, latest_filename=None):
  """Finds the filename of latest saved checkpoint file.

  Args:
    checkpoint_dir: Directory where the variables were saved.
    latest_filename: Optional name for the protocol buffer file that
      contains the list of most recent checkpoint filenames.
      See the corresponding argument to `Saver.save()`.

  Returns:
    The full path to the latest checkpoint or `None` if no checkpoint was found.
  """
  # Pick the latest checkpoint based on checkpoint state.
  ckpt = get_checkpoint_state(checkpoint_dir, latest_filename)
  if ckpt and ckpt.model_checkpoint_path:
    # Look for either a V2 path or a V1 path, with priority for V2.
    v2_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path,
                                         saver_pb2.SaverDef.V2)
    v1_path = _prefix_to_checkpoint_path(ckpt.model_checkpoint_path,
                                         saver_pb2.SaverDef.V1)
    if file_io.get_matching_files(v2_path) or file_io.get_matching_files(
        v1_path):
      return ckpt.model_checkpoint_path
    else:
      logging.error("Couldn't match files for checkpoint %s",
                    ckpt.model_checkpoint_path)
  return None


def _get_kind_name(item):
  """Returns the kind name in CollectionDef.

  Args:
    item: A data item.

  Returns:
    The string representation of the kind in CollectionDef.
  """
  if isinstance(item, (six.string_types, six.binary_type)):
    kind = "bytes_list"
  elif isinstance(item, six.integer_types):
    kind = "int64_list"
  elif isinstance(item, float):
    kind = "float_list"
  elif isinstance(item, Any):
    kind = "any_list"
  else:
    kind = "node_list"
  return kind


def _add_collection_def(meta_graph_def, key):
  """Adds a collection to MetaGraphDef protocol buffer.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer.
    key: One of the GraphKeys or user-defined string.
  """
  if not isinstance(key, six.string_types) and not isinstance(key, bytes):
    logging.warning("Only collections with string type keys will be "
                    "serialized. This key has %s", type(key))
    return
  collection_list = ops.get_collection(key)
  if not collection_list:
    return
  try:
    col_def = meta_graph_def.collection_def[key]
    to_proto = ops.get_to_proto_function(key)
    proto_type = ops.get_collection_proto_type(key)
    if to_proto:
      kind = "bytes_list"
      for x in collection_list:
        # Additional type check to make sure the returned proto is indeed
        # what we expect.
        proto = to_proto(x)
        assert isinstance(proto, proto_type)
        getattr(col_def, kind).value.append(proto.SerializeToString())
    else:
      kind = _get_kind_name(collection_list[0])
      if kind == "node_list":
        getattr(col_def, kind).value.extend([x.name for x in collection_list])
      elif kind == "bytes_list":
        # NOTE(opensource): This force conversion is to work around the fact
        # that Python3 distinguishes between bytes and strings.
        getattr(col_def, kind).value.extend(
            [compat.as_bytes(x) for x in collection_list])
      else:
        getattr(col_def, kind).value.extend([x for x in collection_list])
  except Exception as e:  # pylint: disable=broad-except
    logging.warning("Error encountered when serializing %s.\n"
                    "Type is unsupported, or the types of the items don't "
                    "match field type in CollectionDef.\n%s", key, str(e))
    if key in meta_graph_def.collection_def:
      del meta_graph_def.collection_def[key]
    return


def _as_meta_graph_def(meta_info_def=None,
                       graph_def=None,
                       saver_def=None,
                       collection_list=None):
  """Construct and returns a `MetaGraphDef` protocol buffer.

  Args:
    meta_info_def: `MetaInfoDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    saver_def: `SaverDef` protocol buffer.
    collection_list: List of string keys to collect.

  Returns:
    MetaGraphDef protocol buffer.

  Raises:
    TypeError: If the arguments are not of the correct proto buffer type.
  """
  # Type check.
  if meta_info_def and not isinstance(meta_info_def,
                                      meta_graph_pb2.MetaGraphDef.MetaInfoDef):
    raise TypeError("meta_info_def must be of type MetaInfoDef, not %s",
                    type(meta_info_def))
  if graph_def and not isinstance(graph_def, graph_pb2.GraphDef):
    raise TypeError("graph_def must be of type GraphDef, not %s",
                    type(graph_def))
  if saver_def and not isinstance(saver_def, saver_pb2.SaverDef):
    raise TypeError("saver_def must be of type SaverDef, not %s",
                    type(saver_def))

  # Creates a MetaGraphDef proto.
  meta_graph_def = meta_graph_pb2.MetaGraphDef()
  # Adds meta_info_def.
  if meta_info_def:
    meta_graph_def.meta_info_def.MergeFrom(meta_info_def)

  # Adds graph_def or the default.
  if not graph_def:
    meta_graph_def.graph_def.MergeFrom(ops.get_default_graph().as_graph_def(
        add_shapes=True))
  else:
    meta_graph_def.graph_def.MergeFrom(graph_def)

  # Fills in meta_info_def.stripped_op_list using the ops from graph_def.
  # pylint: disable=g-explicit-length-test
  if len(meta_graph_def.meta_info_def.stripped_op_list.op) == 0:
    meta_graph_def.meta_info_def.stripped_op_list.MergeFrom(
        stripped_op_list_for_graph(meta_graph_def.graph_def))
  # pylint: enable=g-explicit-length-test

  # Adds saver_def.
  if saver_def:
    meta_graph_def.saver_def.MergeFrom(saver_def)

  # Adds collection_list.
  if collection_list:
    clist = collection_list
  else:
    clist = ops.get_all_collection_keys()
  for ctype in clist:
    _add_collection_def(meta_graph_def, ctype)
  return meta_graph_def


def read_meta_graph_file(filename):
  """Reads a file containing `MetaGraphDef` and returns the protocol buffer.

  Args:
    filename: `meta_graph_def` filename including the path.

  Returns:
    A `MetaGraphDef` protocol buffer.

  Raises:
    IOError: If the file doesn't exist, or cannot be successfully parsed.
  """
  meta_graph_def = meta_graph_pb2.MetaGraphDef()
  if not file_io.file_exists(filename):
    raise IOError("File %s does not exist." % filename)
  # First try to read it as a binary file.
  file_content = file_io.read_file_to_string(filename)
  try:
    meta_graph_def.ParseFromString(file_content)
    return meta_graph_def
  except Exception:  # pylint: disable=broad-except
    pass

  # Next try to read it as a text file.
  try:
    text_format.Merge(file_content.decode("utf-8"), meta_graph_def)
  except text_format.ParseError as e:
    raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

  return meta_graph_def


def _import_meta_graph_def(meta_graph_def, clear_devices):
  """Recreates a Graph saved in a `MetaGraphDef` proto.

  This function adds all the nodes from the meta graph def proto to the current
  graph, recreates all the collections, and returns a saver from saver_def.

  Args:
    meta_graph_def: `MetaGraphDef` protocol buffer.
    clear_devices: Boolean which controls whether to clear device information
        from graph_def.

  Returns:
    A saver constructed from `saver_def` in `meta_graph_def` or None.

    A None value is returned if no variables exist in the `meta_graph_def`
    (i.e., no variables to restore).
  """
  # Gathers the list of nodes we are interested in.
  producer_op_list = None
  if meta_graph_def.meta_info_def.HasField("stripped_op_list"):
    producer_op_list = meta_graph_def.meta_info_def.stripped_op_list
  input_graph_def = meta_graph_def.graph_def
  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ""
  importer.import_graph_def(
      input_graph_def, name="", producer_op_list=producer_op_list)

  # Restores all the other collections.
  for key, col_def in meta_graph_def.collection_def.items():
    kind = col_def.WhichOneof("kind")
    if kind is None:
      logging.error("Cannot identify data type for collection %s. Skipping.",
                    key)
      continue
    from_proto = ops.get_from_proto_function(key)
    if from_proto:
      assert kind == "bytes_list"
      proto_type = ops.get_collection_proto_type(key)
      for value in col_def.bytes_list.value:
        proto = proto_type()
        proto.ParseFromString(value)
        ops.add_to_collection(key, from_proto(proto))
    else:
      field = getattr(col_def, kind)
      if kind == "node_list":
        for value in field.value:
          col_op = ops.get_default_graph().as_graph_element(value)
          ops.add_to_collection(key, col_op)
      elif kind == "int64_list":
        # NOTE(opensource): This force conversion is to work around the fact
        # that Python2 distinguishes between int and long, while Python3 has
        # only int.
        for value in field.value:
          ops.add_to_collection(key, int(value))
      else:
        for value in field.value:
          ops.add_to_collection(key, value)

  if meta_graph_def.HasField("saver_def"):
    return Saver(saver_def=meta_graph_def.saver_def)
  else:
    if variables._all_saveable_objects():  # pylint: disable=protected-access
      # Return the default saver instance for all graph variables.
      return Saver()
    else:
      # If not graph variables exist, then a Saver cannot be constructed.
      logging.info("Saver not created because there are no variables in the"
                   " graph to restore")
      return None


def import_meta_graph(meta_graph_or_file, clear_devices=False):
  """Recreates a Graph saved in a `MetaGraphDef` proto.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates all the collections, and returns a saver
  constructed from the `saver_def` field.

  In combination with `export_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  ```Python
  ...
  # Create a saver.
  saver = tf.train.Saver(...variables...)
  # Remember the training_op we want to run by adding it to a collection.
  tf.add_to_collection('train_op', train_op)
  sess = tf.Session()
  for step in xrange(1000000):
      sess.run(train_op)
      if step % 1000 == 0:
          # Saves checkpoint, which by default also exports a meta_graph
          # named 'my-model-global_step.meta'.
          saver.save(sess, 'my-model', global_step=step)
  ```

  Later we can continue training from this saved `meta_graph` without building
  the model from scratch.

  ```Python
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver.restore(sess, 'my-save-dir/my-model-10000')
    # tf.get_collection() returns a list. In this example we only want the
    # first one.
    train_op = tf.get_collection('train_op')[0]
    for step in xrange(1000000):
      sess.run(train_op)
  ```

  NOTE: Restarting training from saved `meta_graph` only works if the
  device assignments have not changed.

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.
    clear_devices: Boolean which controls whether to clear device information
      from graph_def. Default false.

  Returns:
    A saver constructed from `saver_def` in `MetaGraphDef` or None.

    A None value is returned if no variables exist in the `MetaGraphDef`
    (i.e., there are no variables to restore).
  """
  if isinstance(meta_graph_or_file, meta_graph_pb2.MetaGraphDef):
    return _import_meta_graph_def(meta_graph_or_file, clear_devices)
  else:
    return _import_meta_graph_def(
        read_meta_graph_file(meta_graph_or_file), clear_devices)


def export_meta_graph(filename=None,
                      meta_info_def=None,
                      graph_def=None,
                      saver_def=None,
                      collection_list=None,
                      as_text=False):
  """Returns `MetaGraphDef` proto. Optionally writes it to filename.

  This function exports the graph, saver, and collection objects into
  `MetaGraphDef` protocol buffer with the intention of it being imported
  at a later time or location to restart training, run inference, or be
  a subgraph.

  Args:
    filename: Optional filename including the path for writing the
      generated `MetaGraphDef` protocol buffer.
    meta_info_def: `MetaInfoDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    saver_def: `SaverDef` protocol buffer.
    collection_list: List of string keys to collect.
    as_text: If `True`, writes the `MetaGraphDef` as an ASCII proto.

  Returns:
    A `MetaGraphDef` proto.
  """
  meta_graph_def = _as_meta_graph_def(
      meta_info_def=meta_info_def,
      graph_def=graph_def,
      saver_def=saver_def,
      collection_list=collection_list)
  if filename:
    training_util.write_graph(
        meta_graph_def,
        os.path.dirname(filename),
        os.path.basename(filename),
        as_text=as_text)
  return meta_graph_def


ops.register_proto_function(
    ops.GraphKeys.SAVERS,
    proto_type=saver_pb2.SaverDef,
    to_proto=Saver.to_proto,
    from_proto=Saver.from_proto)
