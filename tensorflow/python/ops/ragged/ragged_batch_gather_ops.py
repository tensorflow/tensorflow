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

from tensorflow.python.ops.ragged import ragged_gather_ops


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
  return ragged_gather_ops.gather(params, indices, batch_dims=-1, name=name)
