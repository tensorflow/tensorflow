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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import traverse
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging


# TODO(priyag): Any other reader datasets to consider here?
_READER_DATASET_OPS = [
    "TextLineDataset", "TFRecordDataset", "FixedLengthRecordDataset",
    "FixedLengthRecordDatasetV2"
]


# pylint: disable=protected-access
def auto_shard_dataset(dataset, num_shards, index):
  """Shard the input pipeline by sharding the underlying list of files.

  Args:
    dataset: A `tf.data.Dataset` instance, typically the result of a bunch of
      dataset transformations.
    num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel. Same usage as in
        `tf.data.experimental.filter_for_shard`.
    index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.
      Same usage as in `Dataset.shard`.

  Returns:
    A modified `Dataset` obtained by updating the pipeline sharded by the
    files. The input dataset will be returned if we cannot automatically
    determine a good way to shard the input dataset.
  """

  # TODO(rohanj): b/120673685 to track re-enabling auto sharding.
  tf_logging.warn("Autosharding is currently disabled. Please shard your input "
                  "manually.")
  del num_shards, index
  return dataset


def _clone_dataset(dataset):
  """Returns a cloned version of `dataset`."""
  variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(dataset)
  remap_dict = _clone_helper(dataset._variant_tensor.op, variant_tensor_ops)
  new_variant_tensor = remap_dict[dataset._variant_tensor.op].outputs[0]
  return dataset_ops._VariantDataset(new_variant_tensor,
                                     dataset._element_structure)


def _get_op_def(op):
  return op.op_def or op_def_registry.get_registered_ops()[op.type]


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
