# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Operations for TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform

from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging

if platform.system() != "Windows":
  # pylint: disable=wildcard-import,unused-import,g-import-not-at-top
  from tensorflow.contrib.tpu.ops import gen_tpu_ops
  from tensorflow.contrib.tpu.ops.gen_tpu_ops import *

  from tensorflow.contrib.util import loader
  from tensorflow.python.platform import resource_loader
  # pylint: enable=wildcard-import,unused-import,g-import-not-at-top

  _tpu_ops = loader.load_op_library(
      resource_loader.get_path_to_datafile("_tpu_ops.so"))

  def _create_default_group_assignment():
    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards is None:
      logging.warning(
          "cross_replica_sum should be used within a tpu_shard_context, but "
          "got unset number_of_shards. Assuming 1.")
      num_shards = 1
    group_assignment = [list(range(num_shards))]
    return group_assignment

  def all_to_all(x,
                 concat_dimension,
                 split_dimension,
                 split_count,
                 group_assignment=None,
                 name=None):
    """Exchange data across TPU replicas.

    Args:
      x: The local tensor.
      concat_dimension: The dimension number to concatenate.
      split_dimension: The dimension number to split.
      split_count: The number of splits, this number must equal to the sub-group
        size(group_assignment.get_shape()[1])
      group_assignment: Optional 2d int32 lists with shape [num_groups,
        num_replicas_per_group]. `group_assignment[i]` represents the replica
        ids in the ith subgroup.
      name: Optional op name.

    Returns:
      A `Tensor` which is concatenated by data from different replicas.
    """
    if group_assignment is None:
      group_assignment = _create_default_group_assignment()
    return gen_tpu_ops.all_to_all(
        x,
        group_assignment,
        concat_dimension=concat_dimension,
        split_dimension=split_dimension,
        split_count=split_count,
        name=name)

  @ops.RegisterGradient("AllToAll")
  def _all_to_all_grad(op, grad):
    # The gradient of a all-to-all is also a all-to-all but the
    # split_dimension and concat_dimension is swapped.
    # The graident with respect to group_assignment is None.
    return [
        gen_tpu_ops.all_to_all(
            grad,
            op.inputs[1],
            concat_dimension=op.get_attr("split_dimension"),
            split_dimension=op.get_attr("concat_dimension"),
            split_count=op.get_attr("split_count")), None
    ]

  def cross_replica_sum(x, group_assignment=None, name=None):
    """Sum the input tensor across replicas according to group_assignment.

    Args:
      x: The local tensor to the sum.
      group_assignment: Optional 2d int32 lists with shape [num_groups,
        num_replicas_per_group]. `group_assignment[i]` represents the replica
        ids in the ith subgroup.
      name: Optional op name.

    Returns:
      A `Tensor` which is summed across replicas.
    """
    if group_assignment is None:
      group_assignment = _create_default_group_assignment()

    return gen_tpu_ops.cross_replica_sum(x, group_assignment, name=name)

  def collective_permute(x, source_target_pairs, name=None):
    """Permute the input tensor across replicas given source_target_pairs.

    For each source_target_pair <a, b>, we send replica a's input to replica b.
    Each replica id must only appear once in the source column. Also it must
    only appear once in the target column.
    For the replica id not in the target column, this op returns a zero tensor
    with the same shape and dtype of the input x.

    For example, suppose there are 4 TPU instances: `[A, B, C, D]`. Passing
    source_target_pairs=`[[0,1],[1,2],[2,3]]` gets the outputs:
    `[0, A, B, C]`.

    Args:
      x: The local tensor to be permuted.
      source_target_pairs: 2d int lists with shape [num_pairs, 2].
        source_target_pairs[i][0] represents the source replica id and
        source_target_pairs[i][1] represents the target replica id.
      name: Optional op name.

    Returns:
      A `Tensor` which is permuted.
    """
    return gen_tpu_ops.collective_permute(x, source_target_pairs, name=name)

  @ops.RegisterGradient("CrossReplicaSum")
  def _cross_replica_sum_grad(op, grad):
    # The gradient of a cross replica sum is also a cross-replica sum.
    # The graident with respect to group_assignment is None.
    return [gen_tpu_ops.cross_replica_sum(grad, op.inputs[1]), None]

  # This extra type checking exists to give a more helpful error message in
  # the common case that uint8 and int64 values are infed. Remove when both
  # types are supported.

  _SUPPORTED_INFEED_DTYPES = set([
      dtypes.bool, dtypes.int32, dtypes.int64, dtypes.bfloat16, dtypes.float32,
      dtypes.complex64
  ])

  def infeed_dequeue(dtype, shape, name=None):
    """A placeholder op for a value that will be fed into the computation.

    Args:
      dtype: A `tf.DType`. The type of elements in the tensor.
      shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `dtype`.
      A tensor that will be provided using the infeed mechanism.

    Raises:
      TypeError: If 'dtype` is not a supported infeed type.
    """
    if dtype not in _SUPPORTED_INFEED_DTYPES:
      raise TypeError(
          "{} is not a supported TPU infeed type. Supported types are: "
          "{}".format(dtype, list(_SUPPORTED_INFEED_DTYPES)))

    return gen_tpu_ops.infeed_dequeue(dtype, shape, name=name)

  # pylint: disable=redefined-outer-name
  def infeed_dequeue_tuple(dtypes, shapes, name=None):
    """A placeholder op for values fed into the TPU simultaneously as a tuple.

    Args:
      dtypes: A list of `tf.DType`s that has length `>= 1`.
        The element types of each element in `outputs`.
      shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
        The shapes of each tensor in `outputs`.
      name: A name for the operation (optional).

    Returns:
      A list of `Tensor` objects of type `dtypes`.
      A list of tensors that will be provided using the infeed mechanism.

    Raises:
      TypeError: If a type in 'dtypes` is not a supported infeed type.
    """
    for dtype in dtypes:
      if dtype not in _SUPPORTED_INFEED_DTYPES:
        raise TypeError(
            "{} is not a supported TPU infeed type. Supported types are: "
            "{}".format(dtype, list(_SUPPORTED_INFEED_DTYPES)))
    return gen_tpu_ops.infeed_dequeue_tuple(dtypes, shapes, name=name)
  # pylint: enable=redefined-outer-name

  # pylint: disable=protected-access
  def send_tpu_embedding_gradients(inputs,
                                   config,
                                   learning_rates=None,
                                   name=None):
    """A placeholder op for feeding per-sample gradients to the embedding layer.

    Args:
      inputs: A TensorList of gradients with which to update embedding tables.
        Contains one tensor per embedding table in the model.
      config: Serialized TPUEmbeddingConfiguration proto.
      learning_rates: A TensorList of float32 scalars, one for each embedding
        table, containing the learning rates for each table when dynamic
        learning rate is enabled through the OptimizationParameters in
        TPUEmbeddingConfiguration. When the learning rate is constant, the list
        should be empty (optional).
      name: A name for the operation (optional).

    Returns:
      A SendTPUEmbeddingGradients operation.
    """
    if learning_rates is None:
      learning_rates = []
    return gen_tpu_ops._send_tpu_embedding_gradients(
        inputs=inputs, learning_rates=learning_rates, config=config, name=name)


  send_tpu_embedding_gradients.__doc__ = (
      gen_tpu_ops._send_tpu_embedding_gradients.__doc__)

  # pylint: disable=protected-access
  def enqueue_tpu_embedding_integer_batch(batch,
                                          device_ordinal,
                                          mode_override=None,
                                          name=None):
    """A placeholder op for enqueueing embedding IDs to the TPU.

    Args:
      batch: A list of 1D tensors, one for each embedding table, containing the
        indices into the tables.
      device_ordinal: The TPU device to use. Should be >= 0 and less than the
        number of TPU cores in the task on which the node is placed.
      mode_override: A string input that overrides the mode specified in the
        TPUEmbeddingConfiguration. Supported values are {'unspecified',
        'inference', 'training', 'backward_pass_only'}. When set to
        'unspecified', the mode set in TPUEmbeddingConfiguration is used,
        otherwise mode_override is used (optional).
      name: A name for the operation (optional).

    Returns:
      An EnqueueTPUEmbeddingIntegerBatch operation.
    """
    if mode_override is None:
      mode_override = "unspecified"
    return gen_tpu_ops._enqueue_tpu_embedding_integer_batch(
        batch=batch,
        device_ordinal=device_ordinal,
        mode_override=mode_override,
        name=name)

  enqueue_tpu_embedding_integer_batch.__doc__ = (
      gen_tpu_ops._enqueue_tpu_embedding_integer_batch.__doc__)

  # pylint: disable=protected-access
  def enqueue_tpu_embedding_sparse_batch(sample_indices,
                                         embedding_indices,
                                         aggregation_weights,
                                         device_ordinal,
                                         combiners=None,
                                         mode_override=None,
                                         name=None):
    """A placeholder op for enqueueing embedding IDs to the TPU.

    Args:
      sample_indices: A list of rank 1 Tensors specifying the training example
        and feature to which the corresponding embedding_indices and
        aggregation_weights values belong. sample_indices[i] must equal b * nf +
        f, where nf is the number of features from the corresponding table, f is
        in [0, nf), and b is in [0, batch size).
      embedding_indices: A list of rank 1 Tensors, indices into the embedding
        tables.
      aggregation_weights: A list of rank 1 Tensors containing per sample --
        i.e. per (training example, feature) -- aggregation weights.
      device_ordinal: The TPU device to use. Should be >= 0 and less than the
        number of TPU cores in the task on which the node is placed.
      combiners: A list of string scalars, one for each embedding table that
        specify how to normalize the embedding activations after weighted
        summation. Supported combiners are 'mean', 'sum', or 'sqrtn'. It is
        invalid to have the sum of the weights be 0 for 'mean' or the sum of the
        squared weights be 0 for 'sqrtn'. If combiners isn't passed, the default
        is to use 'sum' for all tables (optional).
      mode_override: A string input that overrides the mode specified in the
        TPUEmbeddingConfiguration. Supported values are {'unspecified',
        'inference', 'training', 'backward_pass_only'}. When set to
        'unspecified', the mode set in TPUEmbeddingConfiguration is used,
        otherwise mode_override is used (optional).
      name: A name for the operation (optional).

    Returns:
      An EnqueueTPUEmbeddingSparseBatch operation.
    """
    if mode_override is None:
      mode_override = "unspecified"
    return gen_tpu_ops._enqueue_tpu_embedding_sparse_batch(
        sample_indices=sample_indices,
        embedding_indices=embedding_indices,
        aggregation_weights=aggregation_weights,
        device_ordinal=device_ordinal,
        combiners=combiners,
        mode_override=mode_override,
        name=name)

  enqueue_tpu_embedding_sparse_batch.__doc__ = (
      gen_tpu_ops._enqueue_tpu_embedding_sparse_batch.__doc__)

  # pylint: disable=protected-access
  def enqueue_tpu_embedding_sparse_tensor_batch(sample_indices,
                                                embedding_indices,
                                                aggregation_weights,
                                                table_ids,
                                                device_ordinal,
                                                combiners=None,
                                                mode_override=None,
                                                name=None):
    """A placeholder op for enqueueing embedding IDs to the TPU.

    Args:
      sample_indices: A list of rank 1 Tensors specifying the training example
        to which the corresponding embedding_indices and aggregation_weights
        values
        belong. It corresponds to sp_ids.indices[:,0] in
          embedding_lookup_sparse().
      embedding_indices: A list of rank 1 Tensors, indices into the embedding
        tables. It corresponds to sp_ids.values in embedding_lookup_sparse().
      aggregation_weights: A list of rank 1 Tensors containing per training
        example aggregation weights. It corresponds to sp_weights.values in
        embedding_lookup_sparse().
      table_ids: A list of integers specifying the identifier of the embedding
        table (offset of TableDescriptor in the TPUEmbeddingConfiguration) to
        lookup the corresponding input. The ith input is looked up using
        table_ids[i]. The size of the table_ids list must be equal to that of
        sample_indices, embedding_indices and aggregation_weights.
      device_ordinal: The TPU device to use. Should be >= 0 and less than the
        number of TPU cores in the task on which the node is placed.
      combiners: A list of string scalars, one for each embedding table that
        specify how to normalize the embedding activations after weighted
        summation. Supported combiners are 'mean', 'sum', or 'sqrtn'. It is
        invalid to have the sum of the weights be 0 for 'mean' or the sum of the
        squared weights be 0 for 'sqrtn'. If combiners isn't passed, the default
        is to use 'sum' for all tables (optional).
      mode_override: A string input that overrides the mode specified in the
        TPUEmbeddingConfiguration. Supported values are {'unspecified',
        'inference', 'training', 'backward_pass_only'}. When set to
        'unspecified', the mode set in TPUEmbeddingConfiguration is used,
        otherwise mode_override is used (optional).
      name: A name for the operation (optional).

    Returns:
      An EnqueueTPUEmbeddingSparseTensorBatch operation.
    """
    if mode_override is None:
      mode_override = "unspecified"
    return gen_tpu_ops._enqueue_tpu_embedding_sparse_tensor_batch(
        sample_indices=sample_indices,
        embedding_indices=embedding_indices,
        aggregation_weights=aggregation_weights,
        table_ids=table_ids,
        device_ordinal=device_ordinal,
        combiners=combiners,
        mode_override=mode_override,
        name=name)

  enqueue_tpu_embedding_sparse_tensor_batch.__doc__ = (
      gen_tpu_ops._enqueue_tpu_embedding_sparse_tensor_batch.__doc__)

else:
  # We have already built the appropriate libraries into the binary via CMake
  # if we have built contrib, so we don't need this
  pass
