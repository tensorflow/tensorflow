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
"""Contains functionaility for Checkpoint/SavedModel in DTensor."""

import collections
import logging
from typing import Dict, List, Union

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import tf_export


def sharded_prefix(
    mesh: layout_lib.Mesh,
    prefix: List[str],
    tensor_names: List[str],
    shape_and_slices: List[str],
    tensors: List[ops.Tensor],
):
  """Generates all sharded prefix in distributed Save.

  DTensor SaveV2 SPMD would generate multiple SaveV2 ops on saving devices,
  and it is desired to not save with same shard_prefix so that content will
  not be overwritten.

  (prefix, tensor_names, tensors(with layouts)) and saving mesh collectively
  defines a unique set of shard prefix that is generated for all the Save ops.
  Usually, (prefix, tensor_names, shape_and_slices, tensors) should match what
  is used in save.

  Args:
    mesh: The mesh that is used in save op. Usually a CPU mesh, and matches what
      is used in Save op.
    prefix: The prefix of saving files.
    tensor_names: a list of tensor names used in save op.
    shape_and_slices: a list of shape and slice specification used in save op.
      The only supported value is "" as we don't support distributed saving with
      slices yet.
    tensors: a list of tensors used in save op. The order should match
      tensor_names.

  Returns:
    A one d string tensor that represents all shard_prefix generated.
  """
  layout_str = array_ops.stack(
      [api.fetch_layout(tensor).to_string() for tensor in tensors], axis=0)
  layouts = api.pack([layout_str] * mesh.num_local_devices(),
                     layout_lib.Layout.replicated(mesh, rank=1))

  mesh_str_tensor = api.pack([mesh.to_string()] * mesh.num_local_devices(),
                             layout_lib.Layout.replicated(mesh, rank=0))
  return gen_dtensor_ops.d_tensor_sharded_prefix(
      prefix,
      tensor_names,
      shape_and_slices,
      mesh_str_tensor,
      layouts=layouts,
      tensors=tensors)


@tf_export('experimental.dtensor.sharded_save', v1=[])
def sharded_save(
    mesh: layout_lib.Mesh,
    file_prefix: Union[str, ops.Tensor],
    tensor_names: Union[List[str], ops.Tensor],
    shape_and_slices: Union[List[str], ops.Tensor],
    tensors: List[Union[ops.Tensor, tf_variables.Variable]],
):
  """Saves given named tensor slices in a sharded, multi-client safe fashion.

  The method makes sure the checkpoint directory state is correct in a sharded
  mutli-client saving. Namely, we place a barrier after SaveV2 to make sure
  every client has done writing the files. And another one after
  MergeV2Checkpoints to make sure all Metadata is properly merged.

  Upon existing, the checkpoint is completed and the all directory operations
  are done.

  Args:
    mesh: The Mesh that contains the Tensors to save.
    file_prefix: The prefix of checkpoint.
    tensor_names: a list of tensor names used in save op.
    shape_and_slices: a list of shape and slice specification used in save op.
      The only supported value is "" as we don't support distributed saving with
      slices yet.
    tensors: a list of tensors used in save op. The order should match
      tensor_names.

  Returns:
    A MergeV2Checkpoints op that merged all Metadata.
  """
  with ops.device(api.device_name()):
    io_ops.save_v2(file_prefix, tensor_names, shape_and_slices, tensors)

  # Query generated shards and generate MergeV2.
  generated_shards = sharded_prefix(mesh.host_mesh(), [file_prefix],
                                    tensor_names, shape_and_slices, tensors)
  # api.py is still visible to external users but the _global_barrier() isn't
  # intended for public usage.
  # Once we locked down api.py visibility, we shall be able to make the `_`
  # prefix on these APIs go away.

  # Make sure all clients have written the files
  _global_barrier(mesh.host_mesh(), 'SaveV2')  # pylint: disable=protected-access

  with ops.device(api.device_name()):
    merge_op = io_ops.MergeV2Checkpoints(
        checkpoint_prefixes=generated_shards,
        destination_prefix=file_prefix,
        delete_old_dirs=True)

  # Make sure first device in first host has finished merge.
  # pylint: disable=protected-access
  _global_barrier(mesh.host_mesh(), 'MergeV2Checkpoints')
  # pylint: enable=protected-access

  return merge_op


@tf_export('experimental.dtensor.enable_save_as_bf16', v1=[])
def enable_save_as_bf16(variables: List[tf_variables.Variable]):
  """Allows float32 DVariables to be checkpointed and restored as bfloat16.

  The method only affects the DVariable part inside the model and leaves
  non-DTensor Variables/Tensors untouched.

  Args:
    variables: A list of tf.Variable to be enabled with bfloat16 save/restore.
      Only has effect on DTensor Variables as they go through d_variables with
      DTensor Specific logis.
  """
  for v in variables:
    if isinstance(v, d_variable.DVariable):
      v.save_as_bf16 = True


@tf_export('experimental.dtensor.name_based_restore', v1=[])
def name_based_restore(
    mesh: layout_lib.Mesh,
    checkpoint_prefix: str,
    name_tensor_dict: Dict[str, Union[ops.Tensor, tf_variables.Variable]],
):
  """Restores from checkpoint_prefix to name based DTensors.

  It is required to have already-initialized DTensor variables that have same
  shape/dtype for the tensors being restored.

  Also, we currently only support a named based restore on a single mesh.

  Args:
    mesh: The single mesh that all Tensors would be restored to.
    checkpoint_prefix : The prefix of checkpoint to be restored.
    name_tensor_dict: A ordered dictionary of tensor_names to a DTensor. The
      DTensor shape/dtype must match the tensors being saved/restored for now.

  Returns:
    A dictionary of name to its restored DTensor value.
  """
  if not context.executing_eagerly():
    raise ValueError('name based restore must run eagerly.')

  ordered_name_tensor_dict = name_tensor_dict
  if not isinstance(name_tensor_dict, collections.OrderedDict):
    ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)

  # Make sure that all tensors are on CPU mesh for now.
  # This might not be a hard limitation in the future.
  for name, tensor in ordered_name_tensor_dict.items():
    try:
      if api.fetch_layout(tensor).mesh.device_type().upper() != 'CPU':
        raise ValueError(
            'Restoring a non CPU Tensor is not supported currently. Offending '
            'tensor name : {tensor_name}'.format(tensor_name=name))
    except errors_impl.OpError as op_error:
      raise ValueError(
          'Saving/Restoring tensor must be a DTensor') from op_error

  # Now that we have all tensors on CPU mesh, do a DTensorRestoreV2.
  checkpoint_prefix = api.pack(
      [checkpoint_prefix] * mesh.num_local_devices(),
      layout_lib.Layout.replicated(mesh.host_mesh(), rank=0))
  # Explicitly pack to mesh to avoid implicit small constant extraction, which
  # does not work larger restores that has lots of names.
  tensor_names = api.pack(
      [list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(),
      layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
  shape_and_slices = api.pack(
      [[''] * len(ordered_name_tensor_dict)] * mesh.num_local_devices(),
      layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
  # A list of TensorShape representing all shapes for the input tensors.
  input_shapes = [tensor.shape for tensor in ordered_name_tensor_dict.values()]
  input_layouts = [
      api.fetch_layout(tensor).to_string()
      for tensor in ordered_name_tensor_dict.values()
  ]

  with ops.device(api.device_name()):
    restored_cpu_tensors = gen_dtensor_ops.d_tensor_restore_v2(
        prefix=checkpoint_prefix,
        tensor_names=tensor_names,
        shape_and_slices=shape_and_slices,
        input_shapes=input_shapes,
        input_layouts=input_layouts,
        dtypes=[tensor.dtype for tensor in ordered_name_tensor_dict.values()])

  return collections.OrderedDict(
      zip(ordered_name_tensor_dict.keys(), restored_cpu_tensors))


@tf_export('experimental.dtensor.name_based_save', v1=[])
def name_based_save(mesh: layout_lib.Mesh, checkpoint_prefix: Union[str,
                                                                    ops.Tensor],
                    name_tensor_dict: Dict[str, Union[ops.Tensor,
                                                      tf_variables.Variable]]):
  """Saves name based Tensor into a Checkpoint.

  The function prepares the input dictionary to the format of a `sharded_save`,
  so that it can take advantage of DTensor SPMD based distributed save.

  Same as restore, the function only supports saving on the single mesh.

  Args:
    mesh: The single mesh that all Tensors would be restored to.
    checkpoint_prefix : The prefix of checkpoint to be restored.
    name_tensor_dict: A ordered dictionary of tensor_names to a DTensor. The
      DTensor shape/dtype must match the tensors being saved/restored for now.
  """
  if not context.executing_eagerly():
    raise ValueError('name based save must run eagerly.')

  ordered_name_tensor_dict = name_tensor_dict
  if not isinstance(name_tensor_dict, collections.OrderedDict):
    ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)

  # Current _dtensor_device() in api.py is the correct way of specifying
  # DTensor device singletons. The API itself will be eventually be moved to
  # a public API and provides global singleton in DTensor context.
  # For now, we just use the current `internal` API and aim at migrating in
  # one shot later.
  # TODO(hthu): Provide _dtensor_device() singleton as a public API.
  # pylint: disable=protected-access
  checkpoint_prefix = api.pack([checkpoint_prefix] * mesh.num_local_devices(),
                               layout_lib.Layout.replicated(
                                   mesh.host_mesh(), rank=0))
  tensor_names = api.pack(
      [list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(),
      layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))

  sharded_save(
      mesh,
      file_prefix=checkpoint_prefix,
      tensor_names=tensor_names,
      shape_and_slices=[''] * len(ordered_name_tensor_dict),
      tensors=list(ordered_name_tensor_dict.values()))


def _global_barrier(mesh: layout_lib.Mesh, last_op_name: str):
  """Runs a global barrier on the mesh.

  Upon returning from the barrier, all operations run before the barrier
  would have completed across all clients.

  Currently we allocate a fully sharded tensor with mesh shape and run a
  all_reduce on it.

  Args:
    mesh: The mesh to run the global barrier on.
    last_op_name: The last op run before the global_barrier. mainly used for
      logging purpose.
  """
  logging.info('entering global barrier before op: %s', last_op_name)

  # Make sure all ops are consumed before running the sync.
  context.async_wait()

  shape = api._dtensor_device().pack(  # pylint: disable=protected-access
      [mesh.shape()] * mesh.num_local_devices(),
      layout_lib.Layout.replicated(mesh, rank=1))
  ones = api.call_with_layout(
      array_ops.ones,
      layout_lib.Layout(mesh.dim_names, mesh),
      shape=shape,
      dtype=dtypes.float32)
  mesh_size = math_ops.reduce_sum(ones)
  if mesh_size != mesh.size:
    raise ValueError(
        'Global barrier produced wrong mesh size : {0} while mesh has actual'
        'size : {1}'.format(mesh_size, mesh.size))

  # TODO(hthu): This isn't strictly needed but might cause confusing behaviors
  # from users. Consider dropping this if there is a `big` performance hit.
  context.async_wait()

  logging.info(
      'finished running global barrier across all clients after '
      'op: %s', last_op_name)
