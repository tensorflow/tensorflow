# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Embedding functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.framework.deprecation import deprecated
from tensorflow.contrib.layers import embedding_ops as embedding_ops

__all__ = ["safe_embedding_lookup_sparse",]


@deprecated("2016-09-01",
            "Please use tf.contrib.layers.safe_embedding_lookup_sparse.")
def safe_embedding_lookup_sparse(embedding_weights,
                                 sparse_ids,
                                 sparse_weights=None,
                                 combiner="mean",
                                 default_id=None,
                                 name=None,
                                 partition_strategy="div"):
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional. Embeddings are always aggregated
  along the last dimension.

  Args:
    embedding_weights:  A list of `P` float tensors or values representing
        partitioned embedding tensors.  The total unpartitioned shape should be
        `[e_0, e_1, ..., e_m]`, where `e_0` represents the vocab size and
        `e_1, ..., e_m` are the embedding dimensions.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
        ids. `d_0` is typically batch size.
    sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
        float weights corresponding to `sparse_ids`, or `None` if all weights
        are be assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
        entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
        the default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy.
        Currently `"div"` and `"mod"` are supported. Default is `"div"`.


  Returns:
    Dense tensor of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  return embedding_ops.safe_embedding_lookup_sparse(
      embedding_weights,
      sparse_ids,
      sparse_weights=sparse_weights,
      combiner=combiner,
      default_id=default_id,
      name=name,
      partition_strategy=partition_strategy)
