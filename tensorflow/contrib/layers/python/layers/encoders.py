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
"""Encoders to transform sequence of symbols into vector representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import embedding_ops as contrib_embedding_ops
from tensorflow.contrib.layers.python.ops import sparse_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope

__all__ = ['bow_encoder', 'embed_sequence']


def bow_encoder(ids,
                vocab_size,
                embed_dim,
                sparse_lookup=True,
                initializer=None,
                regularizer=None,
                trainable=True,
                scope=None,
                reuse=None):
  """Maps a sequence of symbols to a vector per example by averaging embeddings.

  Args:
    ids: `[batch_size, doc_length]` `Tensor` or `SparseTensor` of type
      `int32` or `int64` with symbol ids.
    vocab_size: Integer number of symbols in vocabulary.
    embed_dim: Integer number of dimensions for embedding matrix.
    sparse_lookup: `bool`, if `True`, converts ids to a `SparseTensor`
        and performs a sparse embedding lookup. This is usually faster,
        but not desirable if padding tokens should have an embedding. Empty rows
        are assigned a special embedding.
    initializer: An initializer for the embeddings, if `None` default for
        current scope is used.
    regularizer: Optional regularizer for the embeddings.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional string specifying the variable scope for the op, required
        if `reuse=True`.
    reuse: If `True`, variables inside the op will be reused.

  Returns:
    Encoding `Tensor` `[batch_size, embed_dim]` produced by
    averaging embeddings.

  Raises:
    ValueError: If `embed_dim` or `vocab_size` are not specified.
  """
  if not vocab_size or not embed_dim:
    raise ValueError('Must specify vocab size and embedding dimension')
  with variable_scope.variable_scope(
      scope, 'bow_encoder', [ids], reuse=reuse):
    embeddings = variables.model_variable(
        'embeddings', shape=[vocab_size, embed_dim],
        initializer=initializer, regularizer=regularizer,
        trainable=trainable)
    if sparse_lookup:
      if isinstance(ids, sparse_tensor.SparseTensor):
        sparse_ids = ids
      else:
        sparse_ids = sparse_ops.dense_to_sparse_tensor(ids)
      return contrib_embedding_ops.safe_embedding_lookup_sparse(
          [embeddings], sparse_ids, combiner='mean', default_id=0)
    else:
      if isinstance(ids, sparse_tensor.SparseTensor):
        raise TypeError('ids are expected to be dense Tensor, got: %s', ids)
      return math_ops.reduce_mean(
          embedding_ops.embedding_lookup(embeddings, ids),
          reduction_indices=1)


def embed_sequence(ids,
                   vocab_size=None,
                   embed_dim=None,
                   unique=False,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   scope=None,
                   reuse=None):
  """Maps a sequence of symbols to a sequence of embeddings.

  Typical use case would be reusing embeddings between an encoder and decoder.

  Args:
    ids: `[batch_size, doc_length]` `Tensor` of type `int32` or `int64`
      with symbol ids.
    vocab_size: Integer number of symbols in vocabulary.
    embed_dim: Integer number of dimensions for embedding matrix.
    unique: If `True`, will first compute the unique set of indices, and then
         lookup each embedding once, repeating them in the output as needed.
    initializer: An initializer for the embeddings, if `None` default for
        current scope is used.
    regularizer: Optional regularizer for the embeddings.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional string specifying the variable scope for the op, required
        if `reuse=True`.
    reuse: If `True`, variables inside the op will be reused.

  Returns:
    `Tensor` of `[batch_size, doc_length, embed_dim]` with embedded sequences.

  Raises:
    ValueError: if `embed_dim` or `vocab_size` are not specified when
      `reuse` is `None` or `False`.
  """
  if not (reuse or (vocab_size and embed_dim)):
    raise ValueError('Must specify vocab size and embedding dimension when not'
                     'reusing. Got vocab_size=%s and embed_dim=%s' % (
                         vocab_size, embed_dim))
  with variable_scope.variable_scope(
      scope, 'EmbedSequence', [ids], reuse=reuse):
    shape = [vocab_size, embed_dim]
    if reuse and vocab_size is None or embed_dim is None:
      shape = None
    embeddings = variables.model_variable(
        'embeddings', shape=shape,
        initializer=initializer, regularizer=regularizer,
        trainable=trainable)
    if unique:
      return contrib_embedding_ops.embedding_lookup_unique(embeddings, ids)
    return embedding_ops.embedding_lookup(embeddings, ids)
