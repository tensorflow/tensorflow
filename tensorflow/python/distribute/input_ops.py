# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Input-pipeline utilities for Distribution strategies."""

from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.util import traverse
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.types import data as data_types
from tensorflow.python.types import distribute as distribute_types


# pylint: disable=protected-access
def auto_shard_dataset(dataset, num_shards, index, num_replicas_in_sync=None):
  """Shard the input pipeline by sharding the underlying list of files.

  Args:
    dataset: A `tf.data.Dataset` instance, typically the result of a bunch of
      dataset transformations.
    num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel. Same usage as in `tf.data.Dataset.shard`.
    index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.
      Same usage as in `tf.data.Dataset.shard`.
    num_replicas_in_sync: An integer representing the total number of replicas
      across all workers. This is used in the rewrite when sharding by data.

  Returns:
    A modified `Dataset` obtained by updating the pipeline sharded by the
    files. The input dataset will be returned if we cannot automatically
    determine a good way to shard the input dataset.
  """
  if isinstance(dataset, distribute_types.DistributedDatasetInterface):
    return dataset.auto_shard(num_shards, index)
  if (dataset.options().experimental_distribute.auto_shard_policy !=
      AutoShardPolicy.OFF):
    if num_replicas_in_sync is None:
      num_replicas_in_sync = 1
    if isinstance(dataset, data_types.DatasetV1):
      return distribute._AutoShardDatasetV1(dataset, num_shards, index,
                                            num_replicas_in_sync)
    else:
      return distribute._AutoShardDataset(dataset, num_shards, index,
                                          num_replicas_in_sync)
  else:
    return dataset


def _clone_dataset(dataset):
  """Returns a cloned version of `dataset`."""
  variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(dataset)
  remap_dict = _clone_helper(dataset._variant_tensor.op, variant_tensor_ops)
  new_variant_tensor = remap_dict[dataset._variant_tensor.op].outputs[0]
  return dataset_ops._VariantDataset(new_variant_tensor, dataset.element_spec)


def _get_op_def(op):
  return op.op_def or op_def_registry.get(op.type)


def _clone_helper(op_to_clone, variant_tensor_ops):
  """Helper method that recursively clones `op_to_clone`.

  Args:
    op_to_clone: The op we want to clone.
    variant_tensor_ops: A list of ops that we have to clone along the way.

  Returns:
    A dictionary mapping old_ops to new_ops created. Includes op_to_clone
    as a key.
  """
  remap_dict = {}
  for input_tensor in op_to_clone.inputs:
    input_tensor_op = input_tensor.op
    if input_tensor_op in variant_tensor_ops:
      recursive_map = _clone_helper(input_tensor_op, variant_tensor_ops)
      remap_dict.update(recursive_map)
  inputs_list = []
  for input_tensor in op_to_clone.inputs:
    input_tensor_op = input_tensor.op
    if input_tensor_op in remap_dict:
      remapped_input = remap_dict[input_tensor_op].outputs[0]
      inputs_list.append(remapped_input)
    else:
      inputs_list.append(input_tensor_op.outputs[input_tensor.value_index])
  g = ops.get_default_graph()
  new_op = g.create_op(
      op_to_clone.type,
      inputs_list, [o.dtype for o in op_to_clone.outputs],
      name=op_to_clone.name,
      attrs=op_to_clone.node_def.attr,
      op_def=_get_op_def(op_to_clone))
  remap_dict[op_to_clone] = new_op
  return remap_dict
