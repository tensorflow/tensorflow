# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""
Popnn embedding operator
~~~~~~~~~~~~~~~~~~~~~~~~
"""

from functools import reduce
from operator import mul
import tensorflow as tf


def embedding_lookup(params, ids, name=None):
  """Looks up `ids` in a list of embedding tensors.

    This is designed to be a drop-in replacement for the typical use cases with
    `tf.nn.embedding_lookup` for the IPU.

    Args:
        params: A single tensor representing the complete embedding tensor.
        ids: A `Tensor` with type `int32` containing the ids to be looked up in `params`.
        name: A name for the operation.
    Returns:
        A `Tensor` with the same type as the tensors in `params`.
    """
  name = name or "embedding_lookup"
  M = reduce(mul, ids.shape, 1)
  K = params.shape[0]
  N = params.shape[1]
  ids_one_hot = tf.one_hot(ids, K, name=name + "_one_hot", dtype=params.dtype)
  ids_one_hot = tf.reshape(ids_one_hot, [M, K])
  result = tf.matmul(ids_one_hot, params, name=name + "_lookup")
  return tf.reshape(result, list(ids.shape) + [N])
