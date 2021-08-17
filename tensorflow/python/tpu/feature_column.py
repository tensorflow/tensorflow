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

import math

from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_function
# pylint: disable=protected-access


_TPU_FC_TO_SCOPE = '_tpu_feature_column_scope'
_SUPPORTED_SEQUENCE_COLUMNS = (fc._SequenceCategoricalColumn,
                               fc_lib.SequenceCategoricalColumn)


# For V2 columns, we support anything that inherits from CategoricalColumn
# other than those in the denylist. User-provided columns that inherit from
# CategoricalColumn may or may not be compatible; it is up to the user to
# manage TPU compatibility for custom columns.
_SUPPORTED_CATEGORICAL_COLUMNS_V2 = (fc_lib.CategoricalColumn,)
_DENYLISTED_CATEGORICAL_COLUMNS_V2 = (fc_lib.HashedCategoricalColumn,
                                      fc_lib.BucketizedColumn,
                                      fc_lib.CrossedColumn)
_SUPPORTED_CATEGORICAL_COLUMNS = (fc._IdentityCategoricalColumn,
                                  fc._VocabularyFileCategoricalColumn,
                                  fc._VocabularyListCategoricalColumn,
                                  fc._WeightedCategoricalColumn,
                                  fc._SequenceCategoricalColumn
                                 ) + _SUPPORTED_CATEGORICAL_COLUMNS_V2
_SEQUENCE_FEATURE_LENGTH_POSTFIX = '_seq_length_'


def embedding_column(categorical_column,
                     dimension,
                     combiner='mean',
                     initializer=None,
                     max_sequence_length=0,
                     learning_rate_fn=None,
                     use_safe_embedding_lookup=True):
  """TPU embedding_column for `tf.feature_column.embedding_column`.

  Note that the interface for TPU embedding_column is different from the non-TPU
  version. The following args available for the non-TPU version are NOT
  supported: ckpt_to_load_from, tensor_name_in_ckp, max_norm and trainable.

  Args:
    categorical_column: A categorical_column returned from
        categorical_column_with_identity, weighted_categorical_column,
        categorical_column_with_vocabulary_file,
        categorical_column_with_vocabulary_list,
        sequence_categorical_column_with_identity,
        sequence_categorical_column_with_vocabulary_file,
        sequence_categorical_column_with_vocabulary_list
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.compat.v1.truncated_normal_initializer` with mean `0.0` and
      standard deviation `1/sqrt(dimension)`.
    max_sequence_length: An non-negative integer specifying the max sequence
      length. Any sequence shorter then this will be padded with 0 embeddings
      and any sequence longer will be truncated. This must be positive for
      sequence features and 0 for non-sequence features.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table. If you intend to use the same learning rate
      for multiple embedding tables, please ensure that you pass the exact same
      python function to all calls of embedding_column, otherwise performence
      may suffer.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A  _TPUEmbeddingColumn.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
    TypeError: if categorical_column is not a supported type.
  """
  if isinstance(categorical_column, _DENYLISTED_CATEGORICAL_COLUMNS_V2):
    raise TypeError('categorical_column for tpu '
                    ' embedding_column was denylisted type %s' %
                    type(categorical_column))
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
      trainable=True,
      max_sequence_length=max_sequence_length,
      learning_rate_fn=learning_rate_fn,
      use_safe_embedding_lookup=use_safe_embedding_lookup)
  # For Embedding column, the initializer is hidden inside the creator Fn, which
  # is not accessible later. So, we attach it to a special field. Also note
  # that non-TPU Embedding column and non-TPU shared Embedding column handle the
  # initializer differently. See shared_embedding_columns for details.
  column._tpu_initializer = initializer
  return column


def shared_embedding_columns(categorical_columns,
                             dimension,
                             combiner='mean',
                             initializer=None,
                             shared_embedding_collection_name=None,
                             max_sequence_lengths=None,
                             learning_rate_fn=None,
                             use_safe_embedding_lookup=True):
  """List of dense columns that convert from sparse, categorical input.

  Note that the interface for TPU embedding_column is different from the non-TPU
  version. The following args available for the non-TPU version are NOT
  supported: ckpt_to_load_from, tensor_name_in_ckp, max_norm and trainable.

  Args:
    categorical_columns: A list of categorical_columns returned from
        categorical_column_with_identity, weighted_categorical_column,
        categorical_column_with_vocabulary_file,
        categorical_column_with_vocabulary_list,
        sequence_categorical_column_with_identity,
        sequence_categorical_column_with_vocabulary_file,
        sequence_categorical_column_with_vocabulary_list
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    max_sequence_lengths: An list of non-negative integers, either None or
      empty or the same length as the argument categorical_columns. Entries
      corresponding to non-sequence columns must be 0 and entries corresponding
      to sequence columns specify the max sequence length for the column. Any
      sequence shorter then this will be padded with 0 embeddings and any
      sequence longer will be truncated.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table. If you intend to use the same learning rate
      for multiple embedding tables, please ensure that you pass the exact same
      python function to all calls of shared_embedding_columns, otherwise
      performence may suffer.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A  _TPUEmbeddingColumn.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
    ValueError: if `max_sequence_lengths` is specified and not the same length
      as `categorical_columns`.
    ValueError: if `max_sequence_lengths` is positive for a non sequence column
      or 0 for a sequence column.
  """
  for categorical_column in categorical_columns:
    if isinstance(categorical_column, _DENYLISTED_CATEGORICAL_COLUMNS_V2):
      raise TypeError('categorical_column for tpu '
                      ' embedding_column was denylisted type %s' %
                      type(categorical_column))
    if not isinstance(categorical_column, _SUPPORTED_CATEGORICAL_COLUMNS):
      raise TypeError(
          'categorical_column for tpu '
          ' shared_embedding_columns must be type %s, got %s.' % (' or '.join([
              cc.__name__ for cc in _SUPPORTED_CATEGORICAL_COLUMNS
          ]), type(categorical_column)))

  if not max_sequence_lengths:
    max_sequence_lengths = [0] * len(categorical_columns)
  if len(max_sequence_lengths) != len(categorical_columns):
    raise ValueError('max_sequence_lengths and categorical_columns must be of '
                     'the same length. len(max_sequence_lengths)={} '
                     'len(categorical_columns)={}.'.format(
                         len(max_sequence_lengths), len(categorical_columns)))

  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. ')
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))

  # Sort the columns so the default collection name is deterministic even if the
  # user passes columns from an unsorted collection, such as dict.values().
  sorted_columns = sorted(categorical_columns, key=lambda x: x.name)
  num_buckets = sorted_columns[0]._num_buckets  # pylint: disable=protected-access

  for c in sorted_columns[1:]:
    if num_buckets != c._num_buckets:  # pylint: disable=protected-access
      raise ValueError(
          'To use shared_embedding_column, all categorical_columns must have '
          'the same number of buckets. Given column: {} with buckets: {} does  '
          'not match column: {} with buckets: {}'.format(
              sorted_columns[0], num_buckets, c, c._num_buckets))  # pylint: disable=protected-access

  if not shared_embedding_collection_name:
    shared_embedding_collection_name = '_'.join(c.name for c in sorted_columns)
    shared_embedding_collection_name += '_shared_embedding'

  tpu_columns = []

  # Create the state (_SharedEmbeddingColumnLayer) here.
  for categorical_column, max_sequence_length in zip(
      categorical_columns, max_sequence_lengths):
    column = _TPUSharedEmbeddingColumn(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner=combiner,
        initializer=initializer,
        shared_embedding_collection_name=shared_embedding_collection_name,
        ckpt_to_load_from=None,
        tensor_name_in_ckpt=None,
        max_norm=None,
        trainable=True,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn,
        use_safe_embedding_lookup=use_safe_embedding_lookup)
    tpu_columns.append(column)

  return tpu_columns


class _TPUBaseEmbeddingColumn(object):
  """Base class for TPU Embedding Column."""

  def __init__(self,
               categorical_column,
               max_sequence_length=0,
               learning_rate_fn=None):
    self._tpu_categorical_column = categorical_column
    self._max_sequence_length = max_sequence_length
    self._learning_rate_fn = learning_rate_fn
    if (self.is_sequence_column() and max_sequence_length < 1):
      raise ValueError('max_sequence_length must be greater than 0 for '
                       'sequence columns. Got max_sequence_length={} for '
                       'sequence column {}.'.format(max_sequence_length,
                                                    categorical_column.name))
    if (not self.is_sequence_column() and max_sequence_length != 0):
      raise ValueError('Non zero max_seq_length={} specified for non '
                       'sequence column {}.'.format(max_sequence_length,
                                                    categorical_column.name))

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

  def is_sequence_column(self):
    return isinstance(self._tpu_categorical_column, _SUPPORTED_SEQUENCE_COLUMNS)

  def get_max_sequence_length(self):
    return self._max_sequence_length

  def get_learning_rate_fn(self):
    return self._learning_rate_fn

  def get_sequence_length_feature_key_name(self):
    """Get the key for the associated sequence length feature."""
    return get_sequence_length_feature_key_name_from_feature_key_name(
        self.get_feature_key_name())


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
              trainable=True,
              max_sequence_length=0,
              learning_rate_fn=None,
              use_safe_embedding_lookup=True,
              bypass_scope_validation=False):
    # Note, args ckpt_to_load_from, tensor_name_in_ckpt, max_norm and trainable
    # are not supported on TPU. They are solely for matching the signature of
    # __new__ of parent class fc._EmbeddingColumn.
    del bypass_scope_validation
    # pylint: disable=redundant-keyword-arg
    return fc._EmbeddingColumn.__new__(
        cls,
        categorical_column,
        dimension,
        combiner=combiner,
        layer_creator=layer_creator,
        ckpt_to_load_from=ckpt_to_load_from,
        tensor_name_in_ckpt=tensor_name_in_ckpt,
        max_norm=max_norm,
        trainable=trainable,
        use_safe_embedding_lookup=use_safe_embedding_lookup)

  def __init__(self,
               categorical_column,
               dimension,
               combiner='mean',
               layer_creator=None,
               ckpt_to_load_from=None,
               tensor_name_in_ckpt=None,
               max_norm=None,
               trainable=True,
               max_sequence_length=0,
               learning_rate_fn=None,
               use_safe_embedding_lookup=True,
               bypass_scope_validation=False):
    _TPUBaseEmbeddingColumn.__init__(
        self,
        categorical_column,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn)
    self._key = None
    # If true, scope validation is skipped to allow the same column to be used
    # in multiple variable scopes. By default, this is False, and we expect a
    # 1:1 mapping between feature columns and scopes.
    self._bypass_scope_validation = bypass_scope_validation

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
      def host_computation():
        return fc._EmbeddingColumn._get_dense_tensor(
            self, inputs, weight_collections, trainable)
      return tpu.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc._EmbeddingColumn._get_dense_tensor(
          self, inputs, weight_collections, trainable)

    # TPU mode
    # Get the embeddings from the LazyBuilder.
    tensor = inputs.get(self.get_feature_key_name())

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(
        self.get_embedding_var_name(),
        'embedding_weights',
        bypass_scope_validation=self._bypass_scope_validation)

    return tensor

  def _get_sequence_dense_tensor(
      self, inputs, weight_collections=None, trainable=None):
    if tpu.under_tpu_inference_context():
      def host_computation():
        return fc._EmbeddingColumn._get_sequence_dense_tensor(
            self, inputs, weight_collections, trainable)
      return tpu.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc._EmbeddingColumn._get_sequence_dense_tensor(
          self, inputs, weight_collections, trainable)

    tensor = inputs.get(self.get_feature_key_name())
    tensor_lengths = inputs.get(self.get_sequence_length_feature_key_name())

    # inputs is a _LazyBuilder and for rank 1 tensors, it calls expand_dims(-1).
    # We need to undo this to match the standard CPU sequence embedding.
    tensor_lengths = array_ops.squeeze(tensor_lengths, -1)

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(
        self.get_embedding_var_name(),
        'embedding_weights',
        bypass_scope_validation=self._bypass_scope_validation)

    return fc._SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=tensor, sequence_length=tensor_lengths)


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
              trainable=True,
              max_sequence_length=0,
              learning_rate_fn=None,
              use_safe_embedding_lookup=True):
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
        trainable=trainable,
        use_safe_embedding_lookup=use_safe_embedding_lookup)

  def __init__(self,
               categorical_column,
               dimension,
               combiner='mean',
               initializer=None,
               shared_embedding_collection_name=None,
               ckpt_to_load_from=None,
               tensor_name_in_ckpt=None,
               max_norm=None,
               trainable=True,
               max_sequence_length=0,
               learning_rate_fn=None,
               use_safe_embedding_lookup=True):

    _TPUBaseEmbeddingColumn.__init__(
        self,
        categorical_column,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn)
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
      def host_computation():
        return fc._SharedEmbeddingColumn._get_dense_tensor(
            self, inputs, weight_collections, trainable)
      return tpu.outside_compilation(host_computation)

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

  def _get_sequence_dense_tensor(
      self, inputs, weight_collections=None, trainable=None):
    if tpu.under_tpu_inference_context():
      def host_computation():
        return fc._SharedEmbeddingColumn._get_sequence_dense_tensor(
            self, inputs, weight_collections, trainable)
      return tpu.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc._SharedEmbeddingColumn._get_sequence_dense_tensor(
          self, inputs, weight_collections, trainable)

    tensor = inputs.get(self.get_feature_key_name())
    tensor_lengths = inputs.get(self.get_sequence_length_feature_key_name())

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(
        self.get_embedding_var_name(),
        'embedding_weights',
        is_shared_embedding=True)

    return fc._SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=tensor, sequence_length=tensor_lengths)


def _record_variable_scope_and_name(embedding_var_name,
                                    embedding_var_name_in_fc,
                                    is_shared_embedding=False,
                                    bypass_scope_validation=False):
  """Add embedding variable name and scope to collection."""
  g = ops.get_default_graph()
  collection = g.get_collection_ref(_TPU_FC_TO_SCOPE)
  if not collection:
    collection.append({})

  var_def_dict = collection[0]

  captured_scope = variable_scope.get_variable_scope()
  captured_scope_name = captured_scope.name

  if embedding_var_name in var_def_dict:
    if (var_def_dict[embedding_var_name][0] != captured_scope_name and
        not is_shared_embedding and not bypass_scope_validation):
      raise ValueError(
          'For embedding var name {}, the variable scope name is different, '
          'got {}; expected {}'.format(embedding_var_name,
                                       captured_scope_name,
                                       var_def_dict[embedding_var_name][0]))
    if var_def_dict[embedding_var_name][1] != embedding_var_name_in_fc:
      raise ValueError(
          'For embedding var name {}, the embedding name is different, '
          'got {}; expected {}'.format(embedding_var_name,
                                       embedding_var_name_in_fc,
                                       var_def_dict[embedding_var_name][1]))
  else:
    var_def_dict[embedding_var_name] = (captured_scope_name,
                                        embedding_var_name_in_fc)


def _is_running_on_cpu():
  """Returns True if the current context is CPU model."""
  return tpu_function.get_tpu_context().number_of_shards is None


def get_sequence_length_feature_key_name_from_feature_key_name(feature_name):
  """Gets the name of the sequence length feature from that of the base feature.

  Args:
    feature_name: The feature key of a sequence column.

  Returns:
    A string which is the feature key for the associated feature length column.
  """
  return feature_name + _SEQUENCE_FEATURE_LENGTH_POSTFIX


def split_sequence_columns(feature_columns):
  """Split a list of _TPUEmbeddingColumn into sequence and non-sequence columns.

  For use in a TPUEstimator model_fn function. E.g.

  def model_fn(features):
    sequence_columns, feature_columns = (
        tf.tpu.feature_column.split_sequence_columns(feature_columns))
    input = tf.feature_column.input_layer(
        features=features, feature_columns=feature_columns)
    sequence_features, sequence_lengths = (
        tf.contrib.feature_column.sequence_input_layer(
            features=features, feature_columns=sequence_columns))

  Args:
    feature_columns: A list of _TPUEmbeddingColumns to split.

  Returns:
    Two lists of _TPUEmbeddingColumns, the first is the sequence columns and the
    second is the non-sequence columns.
  """
  sequence_columns = []
  non_sequence_columns = []
  for column in feature_columns:
    if not isinstance(column, (_TPUEmbeddingColumn, _TPUSharedEmbeddingColumn)):
      raise TypeError(
          'column must be a _TPUEmbeddingColumn or  _TPUSharedEmbeddingColumn '
          'but got %s instead.' % (type(column)))
    if column.is_sequence_column():
      sequence_columns.append(column)
    else:
      non_sequence_columns.append(column)
  return sequence_columns, non_sequence_columns
