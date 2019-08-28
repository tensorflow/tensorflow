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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu.feature_column import _is_running_on_cpu
from tensorflow.python.tpu.feature_column import _record_variable_scope_and_name
from tensorflow.python.tpu.feature_column import _SUPPORTED_CATEGORICAL_COLUMNS_V2
from tensorflow.python.tpu.feature_column import _TPUBaseEmbeddingColumn
from tensorflow.python.util.tf_export import tf_export
# pylint: disable=protected-access


@tf_export(v1=['tpu.experimental.embedding_column'])
def embedding_column_v2(categorical_column,
                        dimension,
                        combiner='mean',
                        initializer=None,
                        max_sequence_length=0,
                        learning_rate_fn=None):
  """TPU version of `tf.compat.v1.feature_column.embedding_column`.

  Note that the interface for `tf.tpu.experimental.embedding_column` is
  different from that of `tf.compat.v1.feature_column.embedding_column`: The
  following arguments are NOT supported: `ckpt_to_load_from`,
  `tensor_name_in_ckpt`, `max_norm` and `trainable`.

  Use this function in place of `tf.compat.v1.feature_column.embedding_column`
  when you want to use the TPU to accelerate your embedding lookups via TPU
  embeddings.

  ```
  column = tf.feature_column.categorical_column_with_identity(...)
  tpu_column = tf.tpu.experimental.embedding_column(column, 10)
  ...
  def model_fn(features):
    dense_feature = tf.keras.layers.DenseFeature(tpu_column)
    embedded_feature = dense_feature(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        column=[tpu_column],
        ...))
  ```

  Args:
    categorical_column: A categorical column returned from
        `categorical_column_with_identity`, `weighted_categorical_column`,
        `categorical_column_with_vocabulary_file`,
        `categorical_column_with_vocabulary_list`,
        `sequence_categorical_column_with_identity`,
        `sequence_categorical_column_with_vocabulary_file`,
        `sequence_categorical_column_with_vocabulary_list`
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
      rate for the embedding table.

  Returns:
    A  `_TPUEmbeddingColumnV2`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
  """

  if not isinstance(categorical_column, _SUPPORTED_CATEGORICAL_COLUMNS_V2):
    raise TypeError(
        'categorical_column for tpu '
        ' embedding_column must be type %s, got %s.' % (' or '.join([
            cc.__name__ for cc in _SUPPORTED_CATEGORICAL_COLUMNS_V2
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

  column = _TPUEmbeddingColumnV2(
      categorical_column=categorical_column,
      dimension=dimension,
      combiner=combiner,
      initializer=initializer,
      max_sequence_length=max_sequence_length,
      learning_rate_fn=learning_rate_fn)
  return column


@tf_export(v1=['tpu.experimental.shared_embedding_columns'])
def shared_embedding_columns_v2(categorical_columns,
                                dimension,
                                combiner='mean',
                                initializer=None,
                                shared_embedding_collection_name=None,
                                max_sequence_lengths=None,
                                learning_rate_fn=None):
  """TPU version of `tf.compat.v1.feature_column.shared_embedding_columns`.

  Note that the interface for `tf.tpu.experimental.shared_embedding_columns` is
  different from that of `tf.compat.v1.feature_column.shared_embedding_columns`:
  The following arguments are NOT supported: `ckpt_to_load_from`,
  `tensor_name_in_ckpt`, `max_norm` and `trainable`.

  Use this function in place of
  tf.compat.v1.feature_column.shared_embedding_columns` when you want to use the
  TPU to accelerate your embedding lookups via TPU embeddings.

  ```
  column_a = tf.feature_column.categorical_column_with_identity(...)
  column_b = tf.feature_column.categorical_column_with_identity(...)
  tpu_columns = tf.tpu.experimental.shared_embedding_columns(
      [column_a, column_b], 10)
  ...
  def model_fn(features):
    dense_feature = tf.keras.layers.DenseFeature(tpu_columns)
    embedded_feature = dense_feature(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          column=tpu_columns,
          ...))
  ```

  Args:
    categorical_columns: A list of categorical columns returned from
        `categorical_column_with_identity`, `weighted_categorical_column`,
        `categorical_column_with_vocabulary_file`,
        `categorical_column_with_vocabulary_list`,
        `sequence_categorical_column_with_identity`,
        `sequence_categorical_column_with_vocabulary_file`,
        `sequence_categorical_column_with_vocabulary_list`
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
      rate for the embedding table.

  Returns:
    A  list of `_TPUSharedEmbeddingColumnV2`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
    ValueError: if `max_sequence_lengths` is specified and not the same length
      as `categorical_columns`.
    ValueError: if `max_sequence_lengths` is positive for a non sequence column
      or 0 for a sequence column.
  """

  for categorical_column in categorical_columns:
    if not isinstance(categorical_column, _SUPPORTED_CATEGORICAL_COLUMNS_V2):
      raise TypeError(
          'categorical_column for tpu '
          ' shared_embedding_columns must be type %s, got %s.' % (' or '.join([
              cc.__name__ for cc in _SUPPORTED_CATEGORICAL_COLUMNS_V2
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

  column_creator = fc_lib.SharedEmbeddingColumnCreator(
      dimension=dimension, initializer=initializer, ckpt_to_load_from=None,
      tensor_name_in_ckpt=None, num_buckets=num_buckets, trainable=None,
      name=shared_embedding_collection_name)

  # Create the state (_SharedEmbeddingColumnLayer) here.
  for categorical_column, max_sequence_length in zip(
      categorical_columns, max_sequence_lengths):
    column = _TPUSharedEmbeddingColumnV2(
        categorical_column=categorical_column,
        shared_embedding_column_creator=column_creator,
        combiner=combiner,
        initializer=initializer,
        shared_embedding_collection_name=shared_embedding_collection_name,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn)
    tpu_columns.append(column)

  return tpu_columns


class _TPUEmbeddingColumnV2(_TPUBaseEmbeddingColumn, fc_lib.EmbeddingColumn):
  """Core Embedding Column."""

  def __new__(cls,
              categorical_column,
              dimension,
              combiner='mean',
              initializer=None,
              max_sequence_length=0,
              learning_rate_fn=None):
    return fc_lib.EmbeddingColumn.__new__(
        cls,
        categorical_column,
        dimension,
        combiner=combiner,
        initializer=initializer,
        ckpt_to_load_from=None,
        tensor_name_in_ckpt=None,
        max_norm=None,
        trainable=True)

  def __init__(self,
               categorical_column,
               dimension,
               combiner='mean',
               initializer=None,
               max_sequence_length=0,
               learning_rate_fn=None):
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
    return self.categorical_column.name

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
        return fc_lib.EmbeddingColumn._get_dense_tensor(
            self, inputs, weight_collections, trainable)
      return tpu.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc_lib.EmbeddingColumn._get_dense_tensor(
          self, inputs, weight_collections, trainable)

    # TPU mode
    # Get the embeddings from the LazyBuilder.
    tensor = inputs.get(self.get_feature_key_name())

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(self.get_embedding_var_name(),
                                    'embedding_weights')

    return tensor

  def create_state(self, state_manager):
    if _is_running_on_cpu():
      return fc_lib.EmbeddingColumn.create_state(
          self, state_manager)

    # Create state is called for the EmbeddingColumn to create its embedding
    # variables under feature column V2, if we are on TPU so record the scope
    # here.
    _record_variable_scope_and_name(self.get_embedding_var_name(),
                                    'embedding_weights')

  def get_dense_tensor(self, transformation_cache, state_manager):
    if tpu.under_tpu_inference_context():
      def host_computation():
        return fc_lib.EmbeddingColumn.get_dense_tensor(
            self, transformation_cache, state_manager)
      return tpu.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc_lib.EmbeddingColumn.get_dense_tensor(
          self, transformation_cache, state_manager)

    # TPU mode
    # Get the embeddings from the FeatureTransformationCache.
    tensor = transformation_cache.get(self.get_feature_key_name(),
                                      state_manager)

    return tensor

  def _get_sequence_dense_tensor(
      self, inputs, weight_collections=None, trainable=None):
    if tpu.under_tpu_inference_context():
      def host_computation():
        return fc_lib.EmbeddingColumn._get_sequence_dense_tensor(
            self, inputs, weight_collections, trainable)
      return tpu.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc_lib.EmbeddingColumn._get_sequence_dense_tensor(
          self, inputs, weight_collections, trainable)

    tensor = inputs.get(self.get_feature_key_name())
    tensor_lengths = inputs.get(self.get_sequence_length_feature_key_name())

    # inputs is a _LazyBuilder and for rank 1 tensors, it calls expand_dims(-1).
    # We need to undo this to match the standard CPU sequence embedding.
    tensor_lengths = array_ops.squeeze(tensor_lengths, -1)

    # Add to collection for _create_tpu_embedding_variables_and_ops
    _record_variable_scope_and_name(self.get_embedding_var_name(),
                                    'embedding_weights')

    return fc_lib.SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=tensor, sequence_length=tensor_lengths)

  def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    if tpu.under_tpu_inference_context():
      def host_computation():
        return fc_lib.EmbeddingColumn.get_sequence_dense_tensor(
            self, transformation_cache, state_manager)
      return tpu.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc_lib.EmbeddingColumn.get_sequence_dense_tensor(
          self, transformation_cache, state_manager)

    tensor = transformation_cache.get(self.get_feature_key_name(),
                                      state_manager)
    tensor_lengths = transformation_cache.get(
        self.get_sequence_length_feature_key_name(),
        state_manager)

    # FeatureTransformationCache expands rank 1 tensors (like sequence length)
    # to rank 2. We need to undo this to match the standard CPU sequence
    # embedding.
    tensor_lengths = array_ops.squeeze(tensor_lengths, -1)

    return fc_lib.SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=tensor, sequence_length=tensor_lengths)


class _TPUSharedEmbeddingColumnV2(_TPUBaseEmbeddingColumn,
                                  fc_lib.SharedEmbeddingColumn):
  """Core Shared Embedding Column."""

  def __new__(cls,
              categorical_column,
              shared_embedding_column_creator,
              combiner='mean',
              initializer=None,
              shared_embedding_collection_name=None,
              max_sequence_length=0,
              learning_rate_fn=None):
    return fc_lib.SharedEmbeddingColumn.__new__(
        cls,
        categorical_column,
        combiner=combiner,
        shared_embedding_column_creator=shared_embedding_column_creator,
        max_norm=None)

  def __init__(self,
               categorical_column,
               shared_embedding_column_creator,
               combiner='mean',
               initializer=None,
               shared_embedding_collection_name=None,
               max_sequence_length=0,
               learning_rate_fn=None):

    _TPUBaseEmbeddingColumn.__init__(
        self,
        categorical_column,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn)
    self._initializer = initializer
    self._shared_embedding_collection_name = shared_embedding_collection_name

  def get_combiner(self):
    return self.combiner

  def get_embedding_table_size(self):
    """Returns num_ids and width."""
    return (self.categorical_column._num_buckets,
            self.shared_embedding_column_creator.dimension)

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
    return self._shared_embedding_collection_name

  def get_initializer(self):
    return self._initializer

  def is_categorical_column_weighted(self):
    """Check if the categorical column of the embedding column is weighted."""
    if isinstance(
        self.categorical_column,
        (
            fc._WeightedCategoricalColumn,  # pylint: disable=protected-access
            fc_lib.WeightedCategoricalColumn)):
      return True
    return False

  def _get_dense_tensor_internal(
      self, transformation_cache, state_manager):
    if tpu.under_tpu_inference_context():
      def host_computation():
        return fc_lib.SharedEmbeddingColumn._get_dense_tensor_internal(
            self, transformation_cache, state_manager)
      return tpu.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc_lib.SharedEmbeddingColumn._get_dense_tensor_internal(
          self, transformation_cache, state_manager)

    # TPU mode
    # Get the embeddings from the FeatureTransformationCache.
    tensor = transformation_cache.get(self.get_feature_key_name(),
                                      state_manager)

    # Add to collection for _create_tpu_embedding_variables_and_ops
    # Note that in Feature Column V2, shared embeddings have no scope.
    _record_variable_scope_and_name(
        self.get_embedding_var_name(),
        self.shared_embedding_column_creator._name,
        is_shared_embedding=True)
    return tensor

  def get_sequence_dense_tensor(
      self, transformation_cache, state_manager):
    if tpu.under_tpu_inference_context():
      def host_computation():
        return fc_lib.SharedEmbeddingColumn.get_sequence_dense_tensor(
            self, transformation_cache, state_manager)
      return tpu.outside_compilation(host_computation)

    if _is_running_on_cpu():
      return fc_lib.SharedEmbeddingColumn.get_sequence_dense_tensor(
          self, transformation_cache, state_manager)

    tensor = self._get_dense_tensor_internal(
        transformation_cache, state_manager)
    tensor_lengths = transformation_cache.get(
        self.get_sequence_length_feature_key_name(),
        state_manager)

    # FeatureTransformationCache expands rank 1 tensors (like sequence length)
    # to rank 2. We need to undo this to match the standard CPU sequence
    # embedding.
    tensor_lengths = array_ops.squeeze(tensor_lengths, -1)

    return fc_lib.SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=tensor, sequence_length=tensor_lengths)


def split_sequence_columns_v2(feature_columns):
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
    if not isinstance(column, (_TPUEmbeddingColumnV2,
                               _TPUSharedEmbeddingColumnV2)):
      raise TypeError(
          'column must be a _TPUEmbeddingColumnV2 or '
          '_TPUSharedEmbeddingColumnV2 but got %s instead.' % (type(column)))
    if column.is_sequence_column():
      sequence_columns.append(column)
    else:
      non_sequence_columns.append(column)
  return sequence_columns, non_sequence_columns
