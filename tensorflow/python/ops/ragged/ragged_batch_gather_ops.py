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
"""Batch gather operations for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util


#===============================================================================
# ragged.batch_gather
#===============================================================================
def batch_gather(params, indices, name=None):
  """Gathers slices from `params` according to `indices` with batch dims.

  This operation is similar to `gather`, but it assumes that the leading `N`
  dimensions of `indices` and `params` are batch dimensions, and performs a
  gather within each batch.  In particular, when using this operation with `N`
  batch dimensions `B1...BN`:

  * `indices` has shape `[B1...BN, I]`
  * `params` has shape `[B1...BN, P1...PM]`.
  * `result` has shape `[B1...BN, I, P2...PM]`.
  * `result[b1...bN, i, p2...pM] =
    params[b1...bN, indices[b1...bN, i], p2...pM]`

  Args:
    params: A potentially ragged tensor with shape `[B1...BN, P1...PM]` (`N>=0`,
      `M>0`).
    indices: A potentially ragged tensor with shape `[B1...BN, I]` (`N>=0`).
    name: A name for the operation (optional).

  Returns:
    A potentially ragged tensor with shape `[B1...BN, I, P2...PM]`.
    `result.ragged_rank = max(indices.ragged_rank, params.ragged_rank)`.

  #### Example:

  >>> params = tf.ragged.constant([['a', 'b', 'c'], ['d'], [], ['e']])
  >>> indices = tf.ragged.constant([[1, 2, 0], [], [], [0, 0]])
  >>> tf.compat.v1.batch_gather(params, indices)
  <tf.RaggedTensor [[b'b', b'c', b'a'], [], [], [b'e', b'e']]>
  """
  if not (ragged_tensor.is_ragged(params) or ragged_tensor.is_ragged(indices)):
    return array_ops.batch_gather(params, indices, name)

  with ops.name_scope(name, 'RaggedBatchGather', [params, indices]):
    params = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        params, name='params')
    indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        indices, name='indices')
    params, indices = ragged_tensor.match_row_splits_dtypes(params, indices)
    indices_ndims = indices.shape.ndims
    if indices_ndims is None:
      raise ValueError(
          'batch_gather does not allow indices with unknown shape.')
    if indices_ndims == 0:
      raise ValueError('indices.rank must be at least 1.')

    if ragged_tensor.is_ragged(indices):
      # If the outermost ragged dimension is a batch dimension, recurse.
      if indices_ndims > 2:
        if not ragged_tensor.is_ragged(params):
          raise ValueError('batch shape from indices does '
                           'not match params shape')
        checks = [check_ops.assert_equal(params.row_splits, indices.row_splits)]
        with ops.control_dependencies(checks):
          return ragged_tensor.RaggedTensor.from_row_splits(
              batch_gather(params.values, indices.values), indices.row_splits,
              validate=False)

      # Otherwise, indices is a 2D ragged tensor with 1 ragged dimension.
      else:
        # Ensure that `params` is ragged and has at least 2 dimensions.
        if not ragged_tensor.is_ragged(params):
          if params.shape.ndims is not None and params.shape.ndims < 2:
            raise ValueError('batch shape from indices does '
                             'not match params shape')
          params = ragged_tensor.RaggedTensor.from_tensor(
              params, ragged_rank=1,
              row_splits_dtype=indices.row_splits.dtype)

        # Adjust indices from within-batch to global (in params.values), and
        # then use ragged.gather to gather them.
        num_indices = indices.row_lengths()
        params_starts = params.row_starts()
        adjustments = ragged_util.repeat(params_starts, num_indices, axis=0)
        adjusted_index_values = (
            math_ops.cast(indices.values, adjustments.dtype) + adjustments)
        return ragged_tensor.RaggedTensor.from_row_splits(
            ragged_gather_ops.gather(params.values, adjusted_index_values),
            indices.row_splits, validate=False)

    else:  # params is a RaggedTensor and indices is a Tensor.
      if indices_ndims == 1:
        return ragged_gather_ops.gather(params, indices)
      elif indices_ndims == 2:
        # Adjust indices from batch-local to global (in params.values)
        adjustments = array_ops.expand_dims(params.row_starts(), 1)
        adjusted_indices = (
            math_ops.cast(indices, adjustments.dtype) + adjustments)
        return ragged_gather_ops.gather(params.values, adjusted_indices)
      else:
        raise ValueError('batch shape from indices does not match params shape')
