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

import dataclasses
import math
import time
from typing import Callable, Mapping, MutableMapping, MutableSequence, Sequence

from absl import logging

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint.sharding import sharding_policies
from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity


RegisteredSaversDict = Mapping[
    registration.RegisteredSaver, Mapping[str, base.Trackable]]
MappedCapturesCallable = Callable[
    [core.ConcreteFunction, Sequence[tensor_lib.Tensor]], tensor_lib.Tensor]


def _single_shard_save(
    file_prefix: tensor_lib.Tensor,
    shard: sharding_util.Shard,
    task: device_lib.DeviceSpec,
    options: "checkpoint_options.CheckpointOptions | None" = None,
) -> ops.Operation:
  """Save the saveable objects to a checkpoint with `file_prefix`.

  Args:
    file_prefix: A string or scalar string Tensor containing the prefix to
      save under.
    shard: Dict containing tensors. {checkpoint key: {slice_spec: tensor} }
    task: The device spec task of the tensors in the shard.
    options: Optional `CheckpointOptions` object.

  Returns:
    An `Operation`, or None when executing eagerly.
  """
  options = options or checkpoint_options.CheckpointOptions()

  tensor_names = []
  tensors = []
  slice_specs = []
  for checkpoint_key, tensor_slices in shard.items():
    for slice_spec, tensor in tensor_slices.items():
      # A tensor value of `None` indicates that this SaveableObject gets
      # recorded in the object graph, but that no value is saved in the
      # checkpoint.
      if tensor is not None:
        # See `MultiDeviceSaver._get_shards_by_task` for an explanation on the
        # wrapped properties.
        name = (tensor._wrapped_name  # pylint: disable=protected-access
                if hasattr(tensor, "_wrapped_name")
                else checkpoint_key)
        spec = (tensor._wrapped_slice_spec  # pylint: disable=protected-access
                if hasattr(tensor, "_wrapped_slice_spec")
                else slice_spec)

        tensor_names.append(name)
        tensors.append(tensor)
        slice_specs.append(spec)

  save_device = options.experimental_io_device or (tensors and task)
  with ops.device(save_device or "CPU:0"):
    return io_ops.save_v2(file_prefix, tensor_names, slice_specs, tensors)


def _single_shard_restore(
    file_prefix: tensor_lib.Tensor,
    shardable_tensors: Sequence[sharding_util.ShardableTensor],
    options: "checkpoint_options.CheckpointOptions | None" = None
) -> sharding_util.Shard:
  """Restore the saveable objects from a checkpoint with `file_prefix`.

  Args:
    file_prefix: A string or scalar string Tensor containing the prefix for
      files to read from.
    shardable_tensors: A list of ShardableTensors to restore.
    options: Optional `CheckpointOptions` object.

  Returns:
    A restored tensor dict (maps checkpoint_key -> slice_spec -> tensor).
  """
  options = options or checkpoint_options.CheckpointOptions()

  tensor_names = []
  tensor_dtypes = []
  slice_specs = []
  for shardable_tensor in shardable_tensors:
    if shardable_tensor._tensor_save_spec:  # pylint: disable=protected-access
      name = shardable_tensor._tensor_save_spec.name  # pylint: disable=protected-access
      spec = shardable_tensor._tensor_save_spec.slice_spec  # pylint: disable=protected-access
    else:
      name, spec = shardable_tensor.checkpoint_key, shardable_tensor.slice_spec
    tensor_names.append(name)
    slice_specs.append(spec)
    tensor_dtypes.append(shardable_tensor.dtype)

  restore_device = options.experimental_io_device or "cpu:0"
  with ops.device(restore_device):
    restored_tensors = io_ops.restore_v2(
        file_prefix, tensor_names, slice_specs, tensor_dtypes)

  restored_tensor_dict = {}
  for shardable_tensor in shardable_tensors:
    restored_tensor = restored_tensors.pop(0)
    (restored_tensor_dict
     .setdefault(shardable_tensor.checkpoint_key, {}
                 )[shardable_tensor.slice_spec]) = restored_tensor
  return restored_tensor_dict


def sharded_filename(
    filename_tensor: tensor_lib.Tensor,
    shard: int,
    num_shards: tensor_lib.Tensor
) -> tensor_lib.Tensor:
  """Append sharding information to a filename.

  Args:
    filename_tensor: A string tensor.
    shard: Integer.  The shard for the filename.
    num_shards: An int Tensor for the number of shards.

  Returns:
    A string tensor.
  """
  return gen_io_ops.sharded_filename(filename_tensor, shard, num_shards)


def registered_saver_filename(
    filename_tensor: tensor_lib.Tensor,
    saver_name: registration.RegisteredSaver
) -> tensor_lib.Tensor:
  return string_ops.string_join(
      [filename_tensor, constant_op.constant(f"-{saver_name}")])


def _get_mapped_registered_save_fn(
    fn: Callable[..., tensor_lib.Tensor],
    trackables: Sequence[base.Trackable],
    call_with_mapped_captures: MappedCapturesCallable
) -> Callable[[tensor_lib.Tensor], MappedCapturesCallable]:
  """Converts the function to a python or tf.function with a single file arg."""

  def save_fn(file_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
    return fn(trackables=trackables, file_prefix=file_prefix)
  if call_with_mapped_captures is None:
    return save_fn
  else:
    tf_fn = def_function.function(save_fn, autograph=False)
    concrete = tf_fn.get_concrete_function(
        file_prefix=tensor_spec.TensorSpec(shape=(), dtype=dtypes.string))

    def save_fn_with_replaced_captures(
        file_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
      return call_with_mapped_captures(concrete, [file_prefix])

    return save_fn_with_replaced_captures


def _get_mapped_registered_restore_fn(
    fn: Callable[..., tensor_lib.Tensor],
    trackables: Sequence[base.Trackable],
    call_with_mapped_captures: MappedCapturesCallable
) -> Callable[..., tensor_lib.Tensor]:
  """Converts the function to a python or tf.function with a single file arg."""

  def restore_fn(merged_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
    return fn(trackables=trackables, merged_prefix=merged_prefix)
  if call_with_mapped_captures is None:
    return restore_fn
  else:
    tf_fn = def_function.function(restore_fn, autograph=False)
    concrete = tf_fn.get_concrete_function(
        merged_prefix=tensor_spec.TensorSpec(shape=(), dtype=dtypes.string))

    def restore_fn_with_replaced_captures(
        merged_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
      return call_with_mapped_captures(concrete, [merged_prefix])

    return restore_fn_with_replaced_captures


_restore_noop = lambda *args, **kwargs: None

TensorKeyAndSliceSpec = tuple[str, str]
RestoreFn = Callable[[Mapping[str, tensor_lib.Tensor]], ops.Operation]


class MultiDeviceSaver:
  """Saves checkpoints directly from multiple devices.

  Note that this is a low-level utility which stores Tensors in the keys
  specified by `SaveableObject`s. Higher-level utilities for object-based
  checkpointing are built on top of it.
  """

  def __init__(
      self,
      serialized_tensors: Mapping[base.Trackable, sharding_util.Shard],
      registered_savers: "RegisteredSaversDict | None" = None,
      call_with_mapped_captures: "MappedCapturesCallable | None" = None):
    """Specify a list of `SaveableObject`s to save and restore.

    Args:
      serialized_tensors: A dictionary mapping `Trackable` to a tensor dict,
        which maps checkpoint_key -> (slice_spec ->) -> Tensor/SaveSpec. The
        `Trackable` key is used to get the `restore_from_tensors` function,
        and may be `None` if the tensor is not meant to be restored.
      registered_savers: A dictionary mapping `registration.RegisteredSaver`
        namedtuples to a dictionary of named Trackables. The keys of the
        Trackable dictionary are string names that uniquely identify the
        Trackable in the checkpoint.
      call_with_mapped_captures: TODO
    """
    self._shardable_tensors_by_task: MutableMapping[
        device_lib.DeviceSpec,
        MutableSequence[sharding_util.ShardableTensor]] = {}
    # Keep these two data structures so that we can map restored tensors to
    # the Trackable restore functions.
    self._keys_to_restore_fn: MutableMapping[
        TensorKeyAndSliceSpec, RestoreFn] = {}
    self._restore_fn_to_keys: MutableMapping[
        RestoreFn, MutableSequence[TensorKeyAndSliceSpec]] = {}

    unique_tasks = set()
    for obj, tensor_dict in serialized_tensors.items():
      restore_fn = _restore_noop if obj is None else obj._restore_from_tensors

      # Divide tensor_dict by task.
      for checkpoint_key, tensor_slice_dict in tensor_dict.items():
        if not isinstance(tensor_slice_dict, dict):
          # Make sure that maybe_tensor is structured as {slice_spec -> tensor}.
          tensor_slice_dict = {"": tensor_slice_dict}

        for slice_spec, tensor_save_spec in tensor_slice_dict.items():
          tensor_value = None
          if not isinstance(tensor_save_spec, saveable_object.SaveSpec):
            tensor_value = tensor_save_spec
            tensor_save_spec = saveable_object.SaveSpec(
                tensor=tensor_value,
                slice_spec=slice_spec,
                name=checkpoint_key,
                dtype=tensor_save_spec.dtype,
                device=tensor_save_spec.device)

          if (checkpoint_key, slice_spec) in self._keys_to_restore_fn:
            raise ValueError(
                "Recieved multiple tensors with the same checkpoint key and "
                "slice spec. This is invalid because one will overwrite the "
                "other in the checkpoint. This indicates a bug in the "
                "Checkpoint key-generation.")
          self._keys_to_restore_fn[(checkpoint_key, slice_spec)] = restore_fn
          self._restore_fn_to_keys.setdefault(restore_fn, []).append(
              (checkpoint_key, slice_spec))

          if isinstance(tensor_save_spec.device, str):
            device = device_lib.DeviceSpec.from_string(tensor_save_spec.device)
            task = device_lib.DeviceSpec.from_string(
                saveable_object_util.set_cpu0(tensor_save_spec.device))
          else:
            device = tensor_save_spec.device
            task = device_lib.DeviceSpec.from_string(
                saveable_object_util.set_cpu0(device.to_string()))

          self._shardable_tensors_by_task.setdefault(task, []).append(
              sharding_util.ShardableTensor(
                  _tensor_save_spec=tensor_save_spec,
                  tensor=tensor_value,
                  dtype=tensor_save_spec.dtype,
                  device=device,
                  name=tensor_save_spec.name,
                  shape=None,
                  slice_spec=slice_spec.strip(),
                  checkpoint_key=checkpoint_key,
                  trackable=obj))
          unique_tasks.add(
              saveable_object_util.set_cpu0(device.to_string()))

    self._num_unique_tasks = len(unique_tasks)

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

  @classmethod
  def from_saveables(
      cls,
      saveables: Sequence[base.Trackable],
      registered_savers: "RegisteredSaversDict | None" = None,
      call_with_mapped_captures: "MappedCapturesCallable | None" = None
  ) -> "MultiDeviceSaver":
    """Constructs a MultiDeviceSaver from a list of `SaveableObject`s."""
    serialized_tensors = object_identity.ObjectIdentityDictionary()
    for saveable in saveables:
      trackable = saveable_object_util.SaveableCompatibilityConverter(
          saveable, saveables=[saveable])
      serialized_tensors[trackable] = trackable._serialize_to_tensors()  # pylint: disable=protected-access
    return cls(serialized_tensors, registered_savers, call_with_mapped_captures)

  def to_proto(self) -> saver_pb2.SaverDef:
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
  def _traced_save(self, file_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
    save_op = self.save(file_prefix)
    with ops.device("cpu:0"):
      with ops.control_dependencies([save_op]):
        return array_ops.identity(file_prefix)

  @def_function.function(
      input_signature=(tensor_spec.TensorSpec(shape=(), dtype=dtypes.string),),
      autograph=False)
  def _traced_restore(
      self, file_prefix: tensor_lib.Tensor) -> tensor_lib.Tensor:
    restore_ops = self.restore(file_prefix)
    with ops.device("cpu:0"):
      with ops.control_dependencies(restore_ops.values()):
        return array_ops.identity(file_prefix)

  def _get_shards_by_task(
      self,
      sharding_callback: sharding_util.ShardingCallback
  ) -> Sequence[tuple[str, Sequence[sharding_util.Shard]]]:
    """Calls the sharding callback with shardable_tensors.

    Args:
      sharding_callback: ShardingCallback. The callback function wrapper that
        splits shardable_tensors into shards.

    Returns:
      A list of (task, shards) tuples.
    """
    def wrap_tensor(shardable_tensor):
      tensor_val = shardable_tensor.tensor
      tensor_shape = shardable_tensor.shape
      save_spec = shardable_tensor._tensor_save_spec  # pylint: disable=protected-access
      with ops.device(shardable_tensor.device):
        save_spec_tensor = save_spec.tensor

      if tensor_val is None and save_spec_tensor is None:
        # A tensor value of `None` indicates that this SaveableObject gets
        # recorded in the object graph, but that no value is saved in the
        # checkpoint.
        return None
      elif save_spec_tensor is not None:
        # Pull the tensor value from _tensor_save_spec.
        tensor_val = save_spec_tensor
        tensor_shape = save_spec_tensor.shape

        # Propagate the save spec name and/or slice spec when they are tensors.
        # This makes sure properties like `layout` for dtensor names/slice specs
        # are preserved during sharding.
        if isinstance(save_spec.name, tensor_lib.Tensor):
          tensor_val._wrapped_name = save_spec.name  # pylint: disable=protected-access
        if isinstance(shardable_tensor.slice_spec, tensor_lib.Tensor):
          tensor_val._wrapped_slice_spec = save_spec.slice_spec  # pylint: disable=protected-access

      return dataclasses.replace(
          shardable_tensor,
          tensor=tensor_val,
          shape=tensor_shape)

    shardable_tensors_by_task = {
        task: [shardable_tensor
               for shardable_tensor in map(wrap_tensor, shardable_tensors)
               if shardable_tensor is not None]
        for task, shardable_tensors in self._shardable_tensors_by_task.items()}

    sharding_callback = (
        sharding_callback or sharding_policies.ShardByTaskPolicy())
    metrics.SetShardingCallbackDescription(
        description=sharding_callback.description)

    callback_start_time = time.time() * 1e6
    shards_by_task = []
    for task, shardable_tensors in shardable_tensors_by_task.items():
      shards_by_task.append((task, sharding_callback(shardable_tensors)))
    callback_end_time = time.time() * 1e6

    callback_duration = math.ceil(callback_end_time - callback_start_time)
    metrics.AddShardingCallbackDuration(
        callback_duration=max(1, callback_duration))  # in microseconds
    logging.info("Sharding callback duration: %s", callback_duration)

    return shards_by_task

  def save(
      self,
      file_prefix: tensor_lib.Tensor,
      options: "checkpoint_options.CheckpointOptions | None" = None
  ) -> ops.Operation:
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

    def save_fn() -> ops.Operation:
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

      shards_by_task = self._get_shards_by_task(
          options.experimental_sharding_callback)
      num_shards = sum([len(shards) for _, shards in shards_by_task])
      metrics.AddNumCheckpointShardsWritten(num_shards=num_shards)
      num_shards_tensor = constant_op.constant(num_shards, name="num_shards")
      sharded_saves = []

      shard_idx = 0
      for task, shards in shards_by_task:
        for shard in shards:
          with ops.device(task):
            shard_prefix = sharded_filename(tmp_checkpoint_prefix, shard_idx,
                                            num_shards_tensor)
            shard_idx += 1
          saved_prefixes.append(shard_prefix)
          sharded_saves.append(
              _single_shard_save(shard_prefix, shard, task, options))

      with ops.control_dependencies(sharded_saves):
        # Merge on the io_device if specified, otherwise co-locates the merge op
        # with the last device used.
        tensor_device_spec = list(self._shardable_tensors_by_task.keys())[-1]
        merge_device_spec = (
            options.experimental_io_device or
            saveable_object_util.set_cpu0(tensor_device_spec.to_string()))
        with ops.device(merge_device_spec):
          # V2 format write path consists of a metadata merge step.  Once
          # merged, attempts to delete the temporary directory,
          # "<user-fed prefix>_temp".
          return gen_io_ops.merge_v2_checkpoints(
              saved_prefixes, file_prefix, delete_old_dirs=True)

    # Since this will causes a function re-trace on each save, limit this to the
    # cases where it is needed: eager and when there are multiple tasks. Note
    # that the retrace is needed to ensure we pickup the latest values of
    # options like experimental_io_device.
    if context.executing_eagerly() and self._num_unique_tasks > 1:
      # Explicitly place the identity op on the first device.
      @def_function.function(jit_compile=False)
      def tf_function_save() -> None:
        save_fn()
      tf_function_save()
    else:
      return save_fn()

  def restore(
      self,
      file_prefix: tensor_lib.Tensor,
      options: "checkpoint_options.CheckpointOptions | None" = None
  ) -> Mapping[str, ops.Operation]:
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

    def restore_fn() -> Mapping[str, ops.Operation]:
      restore_fn_inputs = {}
      restore_fn_input_count = {
          fn: len(keys) for fn, keys in self._restore_fn_to_keys.items()}

      restore_ops = {}

      for task, shard in self._shardable_tensors_by_task.items():
        with ops.device(task):
          # Load values from checkpoint
          restored_tensor_dict = _single_shard_restore(
              file_prefix, shard, options)

          # Map restored tensors to the corresponding restore_fn, and see if
          # all inputs have all been loaded. Call `restore_fn` if that is the
          # case.
          for ckpt_key, slice_and_tensor in restored_tensor_dict.items():
            for slice_spec, tensor in slice_and_tensor.items():
              restore_fn = self._keys_to_restore_fn[(ckpt_key,
                                                     slice_spec)]

              # Processing the returned restored_tensor_dict to prepare for
              # the Trackable `restore` function. The `restore` function
              # expects a map of `string name (checkpoint_key) -> Tensor`.
              # Unless there is a slice_spec, in which case the map will be of
              # `string name (checkpoint_key)-> slice_spec -> Tensor`.
              if slice_spec:
                (restore_fn_inputs.setdefault(restore_fn, {}).setdefault(
                    ckpt_key, {})[slice_spec]) = tensor
              else:
                restore_fn_inputs.setdefault(restore_fn,
                                             {})[ckpt_key] = tensor
              restore_fn_input_count[restore_fn] -= 1

              if restore_fn_input_count[restore_fn] == 0:
                restored_tensors = {}
                # Extracts the substring after the "/.ATTRIBUTES/" in the
                # ckpt_key from restore_fn_inputs[restore_fn] to
                # restored_tensors. For example, if
                # restore_fn_input[restore_fn] is dict
                # { "/.ATTIBUTES/a": Tensor}, restored_tensors will be
                # changed to dict {"a": Tensor}
                for ckpt_key, tensor in restore_fn_inputs[restore_fn].items():
                  restored_tensors[trackable_utils.extract_local_name(
                      ckpt_key)] = tensor
                ret = restore_fn(restored_tensors)
                if isinstance(ret, dict):
                  restore_ops.update(ret)
      # Run registered restore methods after the default restore ops.
      for _, (_, restore_fn) in self._registered_savers.items():
        restore_fn(file_prefix)
      return restore_ops

    has_custom_device_saver = False
    for sts in self._shardable_tensors_by_task.values():
      if any([context.is_custom_device(st.device.to_string()) for st in sts]):
        has_custom_device_saver = True
        break
    # Since this will cause a function re-trace on each restore, limit this to
    # cases where it is needed: eager and when there are multiple tasks or any
    # device_spec is a custom device. Note that the retrace is needed to ensure
    # we pickup the latest values of options like experimental_io_device.
    #
    # We run in a function when there is a custom device saver because custom
    # devices, such as DTensor, usually do a sharded save and restore.
    # Doing a sharded save and restore requires knowledge about what shards
    # of variables we are restoring to. In practice, this means that custom
    # devices need the AssignVariableOps along with the Restore op within the
    # same graph to infer shapes and shard specs for Restore op.
    if context.executing_eagerly() and (self._num_unique_tasks > 1 or
                                        has_custom_device_saver):
      @def_function.function(jit_compile=False, autograph=False)
      def tf_function_restore() -> Mapping[str, ops.Operation]:
        restore_fn()
        return {}

      restore_ops = tf_function_restore()
    else:
      restore_ops = restore_fn()

    return restore_ops
