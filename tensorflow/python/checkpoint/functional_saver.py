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
"""Saves and restore variables inside traced @tf.functions."""

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import registration
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest


class _SingleDeviceSaver(object):
  """Saves and restores checkpoints from the current device."""

  __slots__ = ["_saveable_objects"]

  def __init__(self, saveable_objects):
    """Specify a list of `SaveableObject`s to save and restore.

    Args:
      saveable_objects: A list of `SaveableObject`s.
    """
    saveable_objects = list(saveable_objects)
    for saveable in saveable_objects:
      if not isinstance(saveable, saveable_object.SaveableObject):
        raise ValueError(f"Expected a list of SaveableObjects, got {saveable}.")
    self._saveable_objects = saveable_objects

  def save(self, file_prefix, options=None):
    """Save the saveable objects to a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix to
        save under.
      options: Optional `CheckpointOptions` object.
    Returns:
      An `Operation`, or None when executing eagerly.
    """
    options = options or checkpoint_options.CheckpointOptions()
    tensor_names = []
    tensors = []
    tensor_slices = []
    for saveable in self._saveable_objects:
      for spec in saveable.specs:
        tensor = spec.tensor
        # A tensor value of `None` indicates that this SaveableObject gets
        # recorded in the object graph, but that no value is saved in the
        # checkpoint.
        if tensor is not None:
          tensor_names.append(spec.name)
          tensors.append(tensor)
          tensor_slices.append(spec.slice_spec)
    save_device = options.experimental_io_device or "cpu:0"
    with ops.device(save_device):
      return io_ops.save_v2(file_prefix, tensor_names, tensor_slices, tensors)

  def restore(self, file_prefix, options=None):
    """Restore the saveable objects from a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix for
        files to read from.
      options: Optional `CheckpointOptions` object.

    Returns:
      A dictionary mapping from SaveableObject names to restore operations.
    """
    options = options or checkpoint_options.CheckpointOptions()
    restore_specs = []
    tensor_structure = []
    for saveable in self._saveable_objects:
      saveable_tensor_structure = []
      tensor_structure.append(saveable_tensor_structure)
      for spec in saveable.specs:
        saveable_tensor_structure.append(spec.name)
        restore_specs.append((spec.name, spec.slice_spec, spec.dtype))
    tensor_names, tensor_slices, tensor_dtypes = zip(*restore_specs)
    restore_device = options.experimental_io_device or "cpu:0"
    with ops.device(restore_device):
      restored_tensors = io_ops.restore_v2(
          file_prefix, tensor_names, tensor_slices, tensor_dtypes)
    structured_restored_tensors = nest.pack_sequence_as(
        tensor_structure, restored_tensors)
    restore_ops = {}
    for saveable, restored_tensors in zip(self._saveable_objects,
                                          structured_restored_tensors):
      restore_ops[saveable.name] = saveable.restore(
          restored_tensors, restored_shapes=None)
    return restore_ops


def sharded_filename(filename_tensor, shard, num_shards):
  """Append sharding information to a filename.

  Args:
    filename_tensor: A string tensor.
    shard: Integer.  The shard for the filename.
    num_shards: An int Tensor for the number of shards.

  Returns:
    A string tensor.
  """
  return gen_io_ops.sharded_filename(filename_tensor, shard, num_shards)


def registered_saver_filename(filename_tensor, saver_name):
  return string_ops.string_join(
      [filename_tensor, constant_op.constant(f"-{saver_name}")])


def _get_mapped_registered_save_fn(fn, trackables, call_with_mapped_captures):
  """Converts the function to a python or tf.function with a single file arg."""

  def save_fn(file_prefix):
    return fn(trackables=trackables, file_prefix=file_prefix)
  if call_with_mapped_captures is None:
    return save_fn
  else:
    tf_fn = def_function.function(save_fn, autograph=False)
    concrete = tf_fn.get_concrete_function(
        file_prefix=tensor_spec.TensorSpec(shape=(), dtype=dtypes.string))

    def save_fn_with_replaced_captures(file_prefix):
      return call_with_mapped_captures(concrete, [file_prefix])

    return save_fn_with_replaced_captures


def _get_mapped_registered_restore_fn(fn, trackables,
                                      call_with_mapped_captures):
  """Converts the function to a python or tf.function with a single file arg."""

  def restore_fn(merged_prefix):
    return fn(trackables=trackables, merged_prefix=merged_prefix)
  if call_with_mapped_captures is None:
    return restore_fn
  else:
    tf_fn = def_function.function(restore_fn, autograph=False)
    concrete = tf_fn.get_concrete_function(
        merged_prefix=tensor_spec.TensorSpec(shape=(), dtype=dtypes.string))

    def restore_fn_with_replaced_captures(merged_prefix):
      return call_with_mapped_captures(concrete, [merged_prefix])

    return restore_fn_with_replaced_captures


class MultiDeviceSaver(object):
  """Saves checkpoints directly from multiple devices.

  Note that this is a low-level utility which stores Tensors in the keys
  specified by `SaveableObject`s. Higher-level utilities for object-based
  checkpointing are built on top of it.
  """

  def __init__(self,
               saveable_objects,
               registered_savers=None,
               call_with_mapped_captures=None):
    """Specify a list of `SaveableObject`s to save and restore.

    Args:
      saveable_objects: A list of `SaveableObject`s.
        Objects extending `SaveableObject` will be saved and restored.
      registered_savers: A dictionary mapping `registration.RegisteredSaver`
        namedtuples to a dictionary of named Trackables. The keys of the
        Trackable dictionary are string names that uniquely identify the
        Trackable in the checkpoint.
      call_with_mapped_captures: TODO
    """
    saveable_objects = list(saveable_objects)
    saveables_by_device = {}
    for saveable in saveable_objects:
      is_saveable = isinstance(saveable, saveable_object.SaveableObject)

      if not is_saveable:
        raise ValueError(
            f"Expected a dictionary of SaveableObjects, got {saveable}.")

      host_device = saveable_object_util.set_cpu0(saveable.device)
      saveables_by_device.setdefault(host_device, []).append(saveable)

    self._single_device_savers = {
        device: _SingleDeviceSaver(saveables)
        for device, saveables in saveables_by_device.items()}

    self._registered_savers = {}
    if registered_savers:
      for registered_name, trackables in registered_savers.items():
        save_fn = _get_mapped_registered_save_fn(
            registration.get_save_function(registered_name),
            trackables, call_with_mapped_captures)
        restore_fn = _get_mapped_registered_restore_fn(
            registration.get_restore_function(registered_name),
            trackables, call_with_mapped_captures)
        self._registered_savers[registered_name] = (save_fn, restore_fn)

  def to_proto(self):
    """Serializes to a SaverDef referencing the current graph."""
    filename_tensor = array_ops.placeholder(
        shape=[], dtype=dtypes.string, name="saver_filename")
    save_tensor = self._traced_save(filename_tensor)
    restore_op = self._traced_restore(filename_tensor).op
    return saver_pb2.SaverDef(
        filename_tensor_name=filename_tensor.name,
        save_tensor_name=save_tensor.name,
        restore_op_name=restore_op.name,
        version=saver_pb2.SaverDef.V2)

  @def_function.function(
      input_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.string),),
      autograph=False)
  def _traced_save(self, file_prefix):
    save_op = self.save(file_prefix)
    with ops.device("cpu:0"):
      with ops.control_dependencies([save_op]):
        return array_ops.identity(file_prefix)

  @def_function.function(
      input_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.string),),
      autograph=False)
  def _traced_restore(self, file_prefix):
    restore_ops = self.restore(file_prefix)
    with ops.device("cpu:0"):
      with ops.control_dependencies(restore_ops.values()):
        return array_ops.identity(file_prefix)

  def save(self, file_prefix, options=None):
    """Save the saveable objects to a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix to
        save under.
      options: Optional `CheckpointOptions` object.
    Returns:
      An `Operation`, or None when executing eagerly.
    """
    options = options or checkpoint_options.CheckpointOptions()

    # IMPLEMENTATION DETAILS: most clients should skip.
    #
    # Suffix for any well-formed "checkpoint_prefix", when sharded.
    # Transformations:
    # * Users pass in "save_path" in save() and restore().  Say "myckpt".
    # * checkpoint_prefix gets fed <save_path><sharded_suffix>.
    #
    # Example:
    #   During runtime, a temporary directory is first created, which contains
    #   files
    #
    #     <train dir>/myckpt_temp/
    #        part-?????-of-?????{.index, .data-00000-of-00001}
    #
    #   Before .save() finishes, they will be (hopefully, atomically) renamed to
    #
    #     <train dir>/
    #        myckpt{.index, .data-?????-of-?????}
    #
    #   Filesystems with eventual consistency (such as S3), don't need a
    #   temporary location. Using a temporary directory in those cases might
    #   cause situations where files are not available during copy.
    #
    # Users only need to interact with the user-specified prefix, which is
    # "<train dir>/myckpt" in this case.  Save() and Restore() work with the
    # prefix directly, instead of any physical pathname.  (On failure and
    # subsequent restore, an outdated and orphaned temporary directory can be
    # safely removed.)
    with ops.device("CPU"):
      sharded_suffix = array_ops.where(
          string_ops.regex_full_match(file_prefix, "^s3://.*"),
          constant_op.constant(".part"),
          constant_op.constant("_temp/part"))
      tmp_checkpoint_prefix = string_ops.string_join(
          [file_prefix, sharded_suffix])
      registered_paths = {
          saver_name: registered_saver_filename(file_prefix, saver_name)
          for saver_name in self._registered_savers
      }

    def save_fn():
      saved_prefixes = []
      # Save with the registered savers. These run before default savers due to
      # the API contract.
      for saver_name, (save_fn, _) in self._registered_savers.items():
        maybe_saved_prefixes = save_fn(registered_paths[saver_name])
        if maybe_saved_prefixes is not None:
          flattened_saved_prefixes = nest.flatten(maybe_saved_prefixes)
          if not all(
              tensor_util.is_tf_type(x) and x.dtype == dtypes.string
              for x in flattened_saved_prefixes):
            raise ValueError(
                "Registered saver must return a (maybe empty) list of "
                f"string type tensors. Got {maybe_saved_prefixes}.")
          saved_prefixes.extend(flattened_saved_prefixes)

      # (Default saver) Save with single device savers.
      num_shards = len(self._single_device_savers)
      sharded_saves = []
      num_shards_tensor = constant_op.constant(num_shards, name="num_shards")
      last_device = None
      for shard, (device, saver) in enumerate(
          sorted(self._single_device_savers.items())):
        last_device = device
        with ops.device(saveable_object_util.set_cpu0(device)):
          shard_prefix = sharded_filename(tmp_checkpoint_prefix, shard,
                                          num_shards_tensor)
        saved_prefixes.append(shard_prefix)
        with ops.device(device):
          # _SingleDeviceSaver will use the CPU device when necessary, but
          # initial read operations should be placed on the SaveableObject's
          # device.
          sharded_saves.append(saver.save(shard_prefix, options))

      with ops.control_dependencies(sharded_saves):
        # Merge on the io_device if specified, otherwise co-locates the merge op
        # with the last device used.
        merge_device = (
            options.experimental_io_device or
            saveable_object_util.set_cpu0(last_device))
        with ops.device(merge_device):
          # V2 format write path consists of a metadata merge step.  Once
          # merged, attempts to delete the temporary directory,
          # "<user-fed prefix>_temp".
          return gen_io_ops.merge_v2_checkpoints(
              saved_prefixes, file_prefix, delete_old_dirs=True)

    # Since this will causes a function re-trace on each save, limit this to the
    # cases where it is needed: eager and when there are multiple tasks/single
    # device savers. Note that the retrace is needed to ensure we pickup the
    # latest values of options like experimental_io_device.
    if context.executing_eagerly() and len(self._single_device_savers) > 1:
      # Explicitly place the identity op on the first device.
      @def_function.function(jit_compile=False)
      def tf_function_save():
        save_fn()
      tf_function_save()
    else:
      return save_fn()

  def restore(self, file_prefix, options=None):
    """Restore the saveable objects from a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix for
        files to read from.
      options: Optional `CheckpointOptions` object.

    Returns:
      When not run eagerly or when saving on a single device, returns a
      dictionary mapping from SaveableObject names to restore operations;
      otherwise, returns an empty dict.
    """
    options = options or checkpoint_options.CheckpointOptions()

    def restore_fn():
      restore_ops = {}
      # Sort by device name to avoid propagating non-deterministic dictionary
      # ordering in some Python versions.
      for device, saver in sorted(self._single_device_savers.items()):
        with ops.device(device):
          restore_ops.update(saver.restore(file_prefix, options))
      # Run registered restore methods after the default restore ops.
      for _, (_, restore_fn) in self._registered_savers.items():
        restore_fn(file_prefix)
      return restore_ops

    restore_device = options.experimental_io_device or "cpu:0"

    # Since this will causes a function re-trace on each restore, limit this to
    # cases where it is needed: eager and when there are multiple tasks/single
    # device savers. Note that the retrace is needed to ensure we pickup the
    # latest values of options like experimental_io_device.
    if context.executing_eagerly() and (len(self._single_device_savers) > 1 or
                                        options.experimental_io_device):
      @def_function.function(jit_compile=False)
      def tf_function_restore():
        restore_fn()
        return {}

      with ops.device(restore_device):
        restore_ops = tf_function_restore()
    else:
      restore_ops = restore_fn()

    return restore_ops
