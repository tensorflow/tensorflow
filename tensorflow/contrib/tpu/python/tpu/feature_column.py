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
# ===================================================================
"""TPU Feature Column Library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import math

from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
# pylint: disable=protected-access


_TPU_FC_TO_SCOPE = '_tpu_feature_column_scope'
_SUPPORTED_CATEGORICAL_COLUMNS = (fc._IdentityCategoricalColumn,
                                  fc._VocabularyFileCategoricalColumn,
                                  fc._VocabularyListCategoricalColumn,
                                  fc._WeightedCategoricalColumn,
                                  fc_lib.IdentityCategoricalColumn,
                                  fc_lib.VocabularyFileCategoricalColumn,
                                  fc_lib.VocabularyListCategoricalColumn,
                                  fc_lib.WeightedCategoricalColumn)


def embedding_column(categorical_column,
                     dimension,
                     combiner='mean',
                     initializer=None):
  """TPU embedding_column for `tf.feature_column.embedding_column`.

  Note that the interface for TPU embedding_column is different from the non-TPU
  version. The following args available for the non-TPU version are NOT
  supported: ckpt_to_load_from, tensor_name_in_ckp, max_norm and trainable.

  Args:
    categorical_column: A categorical_column returned from
        categorical_column_with_identity,  weighted_categorical_column,
        categorical_column_with_vocabulary_list or
        categorical_column_with_vocabulary_file.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.

  Returns:
    A  _TPUEmbeddingColumn.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
  """
  if not isinstance(categorical_column, _SUPPORTED_CATEGORICAL_COLUMNS):
    raise TypeError(
        'categorical_column for tpu '
        ' embedding_column must be type %s, got %s.' % (' or '.join([
            cc.__name__ for cc in _SUPPORTED_CATEGORICAL_COLUMNS
        ]), type(categorical_column)))
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. '
                     'Embedding of column_name: {}'.format(
                         categorical_column.name))
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))

  embedding_shape = categorical_column._num_buckets, dimension  # pylint: disable=protected-access

  def _creator(weight_collections, scope):
    embedding_column_layer = fc._EmbeddingColumnLayer(
        embedding_shape=embedding_shape,
        initializer=initializer,
        weight_collections=weight_collections,
        trainable=True,
        name='embedding_column_layer')
    return embedding_column_layer(None, scope=scope)  # pylint: disable=not-callable

  column = _TPUEmbeddingColumn(
      categorical_column=categorical_column,
      dimension=dimension,
      combiner=combiner,
      layer_creator=_creator,
      ckpt_to_load_from=None,
      tensor_name_in_ckpt=None,
      max_norm=None,
      trainable=True)
  # For Embedding column, the initializer is hidden inside the creator Fn, which
  # is not accessiable later. So, we attach it to a speicial field. Also note
  # that non-TPU Embedding column and non-TPU shared Embedding column handle the
  # initializer differently. See shared_embedding_columns for details.
  column._tpu_initializer = initializer
  return column


def shared_embedding_columns(categorical_columns,
                             dimension,
                             combiner='mean',
                             initializer=None,
                             shared_embedding_collection_name=None):
  """List of dense columns that convert from sparse, categorical input."""
  for categorical_column in categorical_columns:
    if not isinstance(categorical_column, _SUPPORTED_CATEGORICAL_COLUMNS):
      raise TypeError(
          'categorical_column for tpu '
          ' shared_embedding_columns must be type %s, got %s.' % (' or '.join([
              cc.__name__ for cc in _SUPPORTED_CATEGORICAL_COLUMNS
          ]), type(categorical_column)))
  columns = fc_lib.shared_embedding_columns(
      categorical_columns,
      dimension,
      combiner=combiner,
      initializer=initializer,
      shared_embedding_collection_name=shared_embedding_collection_name,
      ckpt_to_load_from=None,
      tensor_name_in_ckpt=None,
      max_norm=None,
      trainable=True)

  # Use the initializer and shared_embedding_collection_name to create TPU
  # version
  initializer = columns[0].initializer
  shared_embedding_collection_name = columns[0].shared_embedding_collection_name
  tpu_columns = []

  # Create the state (_SharedEmbeddingColumnLayer) here.
  for categorical_column in categorical_columns:
    column = _TPUSharedEmbeddingColumn(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner=combiner,
        initializer=initializer,
        shared_embedding_collection_name=shared_embedding_collection_name,
        ckpt_to_load_from=None,
        tensor_name_in_ckpt=None,
        max_norm=None,
        trainable=True)
    tpu_columns.append(column)

  return tpu_columns


class _TPUBaseEmbeddingColumn(object):
  """Base class for TPU Embedding Column."""

  def __init__(self, categorical_column):
    self._tpu_categorical_column = categorical_column

  def get_combiner(self):
    """Returns the embedding combiner."""
    raise NotImplementedError('not implemented')

  def get_embedding_table_size(self):
    """Returns the embedding table size, tuple of vocab size and dimension."""
    raise NotImplementedError('not implemented')

  def get_feature_key_name(self):
    """Returns the feature key name in the features dict."""
    raise NotImplementedError('not impl')

  def get_weight_key_name(self):
    """Return the key name for weights."""
    raise NotImplementedError('not impl')

  def get_embedding_var_name(self):
    """Returns the embedding variable name.

    Feature key name and embedding variable name are usually one-to-one mapping.
    But for shared embedding columns, it is many-to-one mapping.
    """
    raise NotImplementedError('not impl')

  def get_initializer(self):
    """Returns the initializer."""
    raise NotImplementedError('not impl')

  def is_categorical_column_weighted(self):
    """Check if the categorical column of the embedding column is weighted."""
    raise NotImplementedError('not impl')


class _TPUEmbeddingColumn(_TPUBaseEmbeddingColumn, fc._EmbeddingColumn):
  """Core Embedding Column."""

  def __new__(cls,
              categorical_column,
              dimension,
              combiner='mean',
              layer_creator=None,
              ckpt_to_load_from=None,
              tensor_name_in_ckpt=None,
              max_norm=None,
              trainable=True):
    # Note, args ckpt_to_load_from, tensor_name_in_ckpt, max_norm and trainable
    # are not supported on TPU. They are solely for matching the signature of
    # __new__ of parent class fc._EmbeddingColumn.
    return fc._EmbeddingColumn.__new__(
        cls,
        categorical_column,
        dimension,
        combiner=combiner,
        layer_creator=layer_creator,
        ckpt_to_load_from=ckpt_to_load_from,
        tensor_name_in_ckpt=tensor_name_in_ckpt,
        max_norm=max_norm,
        trainable=trainable)

  def __init__(self,
               categorical_column,
               dimension,
               combiner='mean',
               layer_creator=None,
               ckpt_to_load_from=None,
               tensor_name_in_ckpt=None,
               max_norm=None,
               trainable=True):
    _TPUBaseEmbeddingColumn.__init__(self, categorical_column)
    self._key = None

  def get_combiner(self):
    return self.combiner

  def get_embedding_table_size(self):
    """Returns num_ids and width."""
    return (self.categorical_column._num_buckets, self.dimension)

  def get_feature_key_name(self):
    """get_feature_key_name."""
    if self.is_categorical_column_weighted():
      return self.categorical_column.categorical_column.name
    return self.categorical_column.name

  def get_weight_key_name(self):
    """get_weight_key_name."""
    if self.is_categorical_column_weighted():
      return self.categorical_column.weight_feature_key
    return None

  def get_embedding_var_name(self):
    """get_embedding_var_name."""
    return self.categorical_column.name

  def get_initializer(self):
    return self._tpu_initializer

  def is_categorical_column_weighted(self):
    """Check if the categorical column of the embedding column is weighted."""
    if isinstance(
        self.categorical_column,
        (
            fc._WeightedCategoricalColumn,  # pylint: disable=protected-access
            fc_lib.WeightedCategoricalColumn)):
      return True
    return False

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    if tpu.under_tpu_inference_context():
      # TODO(shizhiw, b/112012627, b/112336539): Replace _outside_all_rewrites()
      # with outside compilation.
      with _outside_all_rewrites():
        return fc._EmbeddingColumn._get_dense_tensor(
            self, inputs, weight_collections, trainable)

    if _is_running_on_cpu():
      return fc._EmbeddingColumn._get_dense_tensor(
          self, inputs, weight_collections, trainable)

    # TPU mode
    # Get the embeddings from the LazyBuilder.
    tensor = inputs.get(self.get_feature_key_name())

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(self.get_embedding_var_name(),
                                    'embedding_weights')

    return tensor


@contextlib.contextmanager
def _outside_all_rewrites():
  """'Break out' of a tpu.rewrite() (or shard(), etc.)."""
  with ops.control_dependencies(None):
    yield


class _TPUSharedEmbeddingColumn(_TPUBaseEmbeddingColumn,
                                fc._SharedEmbeddingColumn):
  """Core Shared Embedding Column."""

  def __new__(cls,
              categorical_column,
              dimension,
              combiner='mean',
              initializer=None,
              shared_embedding_collection_name=None,
              ckpt_to_load_from=None,
              tensor_name_in_ckpt=None,
              max_norm=None,
              trainable=True):
    return fc._SharedEmbeddingColumn.__new__(
        cls,
        categorical_column,
        dimension,
        combiner=combiner,
        initializer=initializer,
        shared_embedding_collection_name=shared_embedding_collection_name,
        ckpt_to_load_from=ckpt_to_load_from,
        tensor_name_in_ckpt=tensor_name_in_ckpt,
        max_norm=max_norm,
        trainable=trainable)

  def __init__(self,
               categorical_column,
               dimension,
               combiner='mean',
               initializer=None,
               shared_embedding_collection_name=None,
               ckpt_to_load_from=None,
               tensor_name_in_ckpt=None,
               max_norm=None,
               trainable=True):

    _TPUBaseEmbeddingColumn.__init__(self, categorical_column)
    self._key = None

  def get_combiner(self):
    return self.combiner

  def get_embedding_table_size(self):
    """Returns num_ids and width."""
    return (self.categorical_column._num_buckets, self.dimension)

  def get_feature_key_name(self):
    """get_feature_key_name."""
    if self.is_categorical_column_weighted():
      return self.categorical_column.categorical_column.name
    return self.categorical_column.name

  def get_weight_key_name(self):
    """get_weight_key_name."""
    if self.is_categorical_column_weighted():
      return self.categorical_column.weight_feature_key
    return None

  def get_embedding_var_name(self):
    """get_embedding_var_name."""
    return self.shared_embedding_collection_name

  def get_initializer(self):
    return self.initializer

  def is_categorical_column_weighted(self):
    """Check if the categorical column of the embedding column is weighted."""
    if isinstance(
        self.categorical_column,
        (
            fc._WeightedCategoricalColumn,  # pylint: disable=protected-access
            fc_lib.WeightedCategoricalColumn)):
      return True
    return False

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    if tpu.under_tpu_inference_context():
      # TODO(shizhiw, b/112012627, b/112336539): Replace _outside_all_rewrites()
      # with outside compilation.
      with _outside_all_rewrites():
        return fc._SharedEmbeddingColumn._get_dense_tensor(
            self, inputs, weight_collections, trainable)

    if _is_running_on_cpu():
      return fc._SharedEmbeddingColumn._get_dense_tensor(
          self, inputs, weight_collections, trainable)

    # TPU mode
    # Get the embeddings from the LazyBuilder.
    tensor = inputs.get(self.get_feature_key_name())

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(
        self.get_embedding_var_name(),
        'embedding_weights',
        is_shared_embedding=True)
    return tensor


def _record_variable_scope_and_name(embedding_var_name,
                                    embedding_var_name_in_fc,
                                    is_shared_embedding=False):
  """Add embedding variable name and scope to collection."""
  g = ops.get_default_graph()
  collection = g.get_collection_ref(_TPU_FC_TO_SCOPE)
  if not collection:
    collection.append({})

  var_def_dict = collection[0]

  captured_scope = None

  if is_shared_embedding and (embedding_var_name in var_def_dict):
    if var_def_dict[embedding_var_name][1] != embedding_var_name_in_fc:
      raise ValueError(
          'For embedding var name {}, the shared embedding name is different, '
          'got {}; expected {}'.format(embedding_var_name,
                                       embedding_var_name_in_fc,
                                       var_def_dict[embedding_var_name][1]))
  else:
    # scope contains var_scope_name.
    captured_scope = variable_scope.get_variable_scope()
    var_def_dict[embedding_var_name] = (captured_scope,
                                        embedding_var_name_in_fc)


def _is_running_on_cpu():
  """Returns True if the current context is CPU model."""
  return tpu_function.get_tpu_context().number_of_shards is None
