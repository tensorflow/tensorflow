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

import enum

from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu.feature_column import _is_running_on_cpu
from tensorflow.python.tpu.feature_column import _record_variable_scope_and_name
from tensorflow.python.tpu.feature_column import _SUPPORTED_CATEGORICAL_COLUMNS_V2
from tensorflow.python.tpu.feature_column import _TPUBaseEmbeddingColumn
from tensorflow.python.util.tf_export import tf_export
# pylint: disable=protected-access

_ALLOWED_DEVICES = ['cpu', 'tpu_tensor_core', 'tpu_embedding_core']


class EmbeddingDevice(enum.Enum):
  CPU = 1
  TPU_TENSOR_CORE = 2
  TPU_EMBEDDING_CORE = 3


@tf_export(v1=['tpu.experimental.embedding_column'])
def embedding_column_v2(categorical_column,
                        dimension,
                        combiner='mean',
                        initializer=None,
                        max_sequence_length=0,
                        learning_rate_fn=None,
                        embedding_lookup_device=None,
                        tensor_core_shape=None):
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
    embedding_lookup_device: The device on which to run the embedding lookup.
      Valid options are "cpu", "tpu_tensor_core", and "tpu_embedding_core".
      If specifying "tpu_tensor_core", a tensor_core_shape must be supplied.
      If not specified, the default behavior is embedding lookup on
      "tpu_embedding_core" for training and "cpu" for inference.
      Valid options for training : ["tpu_embedding_core", "tpu_tensor_core"]
      Valid options for serving :  ["cpu", "tpu_tensor_core"]
      For training, tpu_embedding_core is good for large embedding vocab (>1M),
      otherwise, tpu_tensor_core is often sufficient.
      For serving, doing embedding lookup on tpu_tensor_core during serving is
      a way to reduce host cpu usage in cases where that is a bottleneck.
    tensor_core_shape: If supplied, a list of integers which specifies
      the intended dense shape to run embedding lookup for this feature on
      TensorCore. The batch dimension can be left None or -1 to indicate
      a dynamic shape. Only rank 2 shapes currently supported.

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
  if tensor_core_shape and len(tensor_core_shape) != 2:
    raise ValueError(
        'tensor_core_shape must be size 2. Got {}.'.format(tensor_core_shape))

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. '
                     'Embedding of column_name: {}'.format(
                         categorical_column.name))
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))

  if (embedding_lookup_device and
      embedding_lookup_device not in _ALLOWED_DEVICES):
    raise ValueError('If set, embedding_lookup_device must be in ',
                     _ALLOWED_DEVICES)

  if embedding_lookup_device == 'cpu':
    embedding_lookup_device = EmbeddingDevice.CPU
  elif embedding_lookup_device == 'tpu_tensor_core':
    embedding_lookup_device = EmbeddingDevice.TPU_TENSOR_CORE
  elif embedding_lookup_device == 'tpu_embedding_core':
    embedding_lookup_device = EmbeddingDevice.TPU_EMBEDDING_CORE

  if (embedding_lookup_device == EmbeddingDevice.TPU_TENSOR_CORE and
      not tensor_core_shape):
    raise ValueError('Using embedding_lookup_device=tpu_tensor_core requires '
                     'tensor_core_shape to be set.')

  if not embedding_lookup_device:
    return _TPUEmbeddingColumnV2(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner=combiner,
        initializer=initializer,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn)
  else:
    return _TPUDeviceSpecificEmbeddingColumnV2(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner=combiner,
        initializer=initializer,
        max_sequence_length=max_sequence_length,
        learning_rate_fn=learning_rate_fn,
        embedding_lookup_device=embedding_lookup_device,
        tensor_core_shape=tensor_core_shape)


@tf_export(v1=['tpu.experimental.shared_embedding_columns'])
def shared_embedding_columns_v2(categorical_columns,
                                dimension,
                                combiner='mean',
                                initializer=None,
                                shared_embedding_collection_name=None,
                                max_sequence_lengths=None,
                                learning_rate_fn=None,
                                embedding_lookup_device=None,
                                tensor_core_shape=None):
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
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    max_sequence_lengths: An list of non-negative integers, either None or empty
      or the same length as the argument categorical_columns. Entries
      corresponding to non-sequence columns must be 0 and entries corresponding
      to sequence columns specify the max sequence length for the column. Any
      sequence shorter then this will be padded with 0 embeddings and any
      sequence longer will be truncated.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table.
    embedding_lookup_device: The device on which to run the embedding lookup.
      Valid options are "cpu", "tpu_tensor_core", and "tpu_embedding_core". If
      specifying "tpu_tensor_core", a tensor_core_shape must be supplied.
      Defaults to "cpu". If not specified, the default behavior is embedding
      lookup on "tpu_embedding_core" for training and "cpu" for inference.
      Valid options for training : ["tpu_embedding_core", "tpu_tensor_core"]
      Valid options for serving :  ["cpu", "tpu_tensor_core"]
      For training, tpu_embedding_core is good for large embedding vocab (>1M),
      otherwise, tpu_tensor_core is often sufficient.
      For serving, doing embedding lookup on tpu_tensor_core during serving is
      a way to reduce host cpu usage in cases where that is a bottleneck.
    tensor_core_shape: If supplied, a list of integers which specifies the
      intended dense shape to run embedding lookup for this feature on
      TensorCore. The batch dimension can be left None or -1 to indicate a
      dynamic shape. Only rank 2 shapes currently supported.

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
  if tensor_core_shape and len(tensor_core_shape) != 2:
    raise ValueError(
        'tensor_core_shape must be size 2. Got {}.'.format(tensor_core_shape))

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

  if (embedding_lookup_device and
      embedding_lookup_device not in _ALLOWED_DEVICES):
    raise ValueError('If set, embedding_lookup_device must be in ',
                     _ALLOWED_DEVICES)

  if embedding_lookup_device == 'cpu':
    embedding_lookup_device = EmbeddingDevice.CPU
  elif embedding_lookup_device == 'tpu_tensor_core':
    embedding_lookup_device = EmbeddingDevice.TPU_TENSOR_CORE
  elif embedding_lookup_device == 'tpu_embedding_core':
    embedding_lookup_device = EmbeddingDevice.TPU_EMBEDDING_CORE

  if (embedding_lookup_device == EmbeddingDevice.TPU_EMBEDDING_CORE and
      not tensor_core_shape):
    raise ValueError('Using embedding_lookup_device=tpu_tensor_core requires '
                     'tensor_core_shape to be set.')

  # Create the state (_SharedEmbeddingColumnLayer) here.
  for categorical_column, max_sequence_length in zip(
      categorical_columns, max_sequence_lengths):
    if not embedding_lookup_device:
      column = _TPUSharedEmbeddingColumnV2(
          categorical_column=categorical_column,
          shared_embedding_column_creator=column_creator,
          combiner=combiner,
          initializer=initializer,
          shared_embedding_collection_name=shared_embedding_collection_name,
          max_sequence_length=max_sequence_length,
          learning_rate_fn=learning_rate_fn)
    else:
      column = _TPUSharedDeviceSpecificEmbeddingColumnV2(
          categorical_column=categorical_column,
          shared_embedding_column_creator=column_creator,
          combiner=combiner,
          initializer=initializer,
          shared_embedding_collection_name=shared_embedding_collection_name,
          max_sequence_length=max_sequence_length,
          learning_rate_fn=learning_rate_fn,
          embedding_lookup_device=embedding_lookup_device,
          tensor_core_shape=tensor_core_shape)
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


def sparse_embedding_aggregate_slice(params,
                                     values_and_values_mask,
                                     combiner='mean',
                                     name='sparse_embedding_aggregate_slice'):
  """Uses XLA's dynamic slice operations to perform embedding lookups.

  From third_party/cloud_tpu/models/movielens/tpu_embedding.py

  Args:
    params: Tensor of embedding table. Rank 2 (table_size x embedding dim)
    values_and_values_mask: is a two-tuple that contains: values - Tensor of
      embedding indices. Rank 2 (batch x n_indices) values_mask - Tensor of mask
      / weights. Rank 2 (batch x n_indices)
    combiner: The combiner to use for the embedding lookup. Currently supports
      'sum' and 'mean'.
    name: Optional name scope for created ops

  Returns:
    Rank 2 tensor of aggregated (per batch element) embedding vectors.

  Raises:
    ValueError: Combiner is not supported.
  """
  values, values_mask = values_and_values_mask  # unpack the two-tuple
  with ops.name_scope(name):
    _, embedding_dimension = params.get_shape().as_list()
    n_batch, n_indices_padded = values.get_shape().as_list()
    if not n_batch:
      n_batch = -1

    emb_lookup = array_ops.reshape(
        embedding_ops.embedding_lookup(
            params, array_ops.reshape(values, [n_batch, n_indices_padded])),
        [n_batch, n_indices_padded, embedding_dimension])

    values_mask_broadcast = array_ops.reshape(values_mask,
                                              [n_batch, n_indices_padded, 1])
    aggregate_emb = math_ops.reduce_sum(
        emb_lookup * values_mask_broadcast, axis=1)
    if combiner == 'sum':
      return aggregate_emb
    elif combiner == 'mean':
      return aggregate_emb / math_ops.reduce_sum(values_mask_broadcast, axis=1)
    else:
      raise ValueError('Dense TPU Embedding does not support combiner '
                       'other than sum and mean.')


def pad_sparse_embedding_lookup_indices(sparse_indices, padded_size):
  """Creates statically-sized Tensors containing indices and weights.

  From third_party/cloud_tpu/models/movielens/tpu_embedding.py

  Also computes sparse_indices.values % embedding_table_size, for equivalent
  functionality to sparse_column_with_integerized_feature. The returned
  padded weight Tensor also doubles as a mask indicating which values in
  the returned padded indices Tensor are indices versus padded zeros.

  Args:
    sparse_indices: SparseTensor of embedding lookup indices.
    padded_size: Number of columns of the returned Tensors. Indices which fall
      out of bounds will be truncated to the padded size.

  Returns:
    (sparse_indices.values padded to the specified size,
     a mask the same size as the returned padded values in which 0s
     indicate padded locations and 1s (or values from sparse_weights)
     indicate actual values)
  """
  batch_size = sparse_indices.dense_shape[0]
  sparse_indices = sparse_ops.sparse_slice(sparse_indices, [0, 0],
                                           [batch_size, padded_size])
  indices, values = sparse_indices.indices, sparse_indices.values

  padded_values = array_ops.scatter_nd(
      indices,
      math_ops.cast(values, dtypes.int32),
      shape=(batch_size, padded_size))

  weights = array_ops.ones_like(values, dtype=dtypes.float32)
  padded_mask = array_ops.scatter_nd(
      indices, weights, shape=(batch_size, padded_size))

  return padded_values, padded_mask


class _TPUDeviceSpecificEmbeddingColumnV2(_TPUEmbeddingColumnV2):
  """TPUEmbeddingColumn which allows serving on TensorCore."""

  def __new__(cls, *args, **kwargs):
    # For __new__, just capture the inference dense shape and call parent.
    if 'tensor_core_shape' in kwargs:
      cls._tensor_core_shape = kwargs['tensor_core_shape']
      del kwargs['tensor_core_shape']
    if 'embedding_lookup_device' in kwargs:
      cls._embedding_lookup_device = kwargs['embedding_lookup_device']
      del kwargs['embedding_lookup_device']
    return _TPUEmbeddingColumnV2.__new__(cls, *args, **kwargs)

  def __init__(self, *args, **kwargs):
    # For __init__, just capture the inference dense shape and call parent.
    if 'tensor_core_shape' in kwargs:
      self._tensor_core_shape = kwargs['tensor_core_shape']
      del kwargs['tensor_core_shape']
    if 'embedding_lookup_device' in kwargs:
      self._embedding_lookup_device = kwargs['embedding_lookup_device']
      del kwargs['embedding_lookup_device']
    _TPUEmbeddingColumnV2.__init__(self, *args, **kwargs)

  def create_state(self, state_manager):
    if (tpu.under_tpu_inference_context() and
        self._embedding_lookup_device == EmbeddingDevice.TPU_EMBEDDING_CORE):
      raise ValueError(
          'Using embedding_lookup_device=tpu_embedding_core during inference '
          'is not supported.')
    if self._embedding_lookup_device == EmbeddingDevice.CPU:
      if tpu.under_tpu_inference_context():
        return fc_lib.EmbeddingColumn.create_state(self, state_manager)
      else:
        raise ValueError(
            'Using TPUEmbeddingColumn with embedding_lookup_device="cpu" '
            'during training is not supported.')

    return super(_TPUDeviceSpecificEmbeddingColumnV2,
                 self).create_state(state_manager)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Private method that follows get_dense_tensor."""

    # If we aren't inferencing on TensorCore, just delegate to parent.
    if not tpu.under_tpu_inference_context() or not self._tensor_core_shape:
      return super(_TPUDeviceSpecificEmbeddingColumnV2,
                   self).get_dense_tensor(transformation_cache, state_manager)
    sparse_tensor = transformation_cache.get(self.categorical_column.name,
                                             state_manager)

    # Use outside compile to densify and pad the input tensors.
    def host_computation():
      return pad_sparse_embedding_lookup_indices(sparse_tensor,
                                                 self._tensor_core_shape[1])

    values, mask = tpu.outside_compilation(host_computation)

    # Do a dense embedding lookup on TensorCore.
    embedding_weights = state_manager.get_variable(self, 'embedding_weights')
    embedding = sparse_embedding_aggregate_slice(embedding_weights,
                                                 (values, mask),
                                                 self.get_combiner())
    return embedding


class _TPUSharedDeviceSpecificEmbeddingColumnV2(_TPUSharedEmbeddingColumnV2):
  """TPUSharedEmbeddingColumnV2 which allows serving on TensorCore."""

  def __new__(cls, *args, **kwargs):
    # For __new__, just capture the inference dense shape and call parent.
    if 'tensor_core_shape' in kwargs:
      cls._tensor_core_shape = kwargs['tensor_core_shape']
      del kwargs['tensor_core_shape']
    if 'embedding_lookup_device' in kwargs:
      cls._embedding_lookup_device = kwargs['embedding_lookup_device']
      del kwargs['embedding_lookup_device']

    return _TPUSharedEmbeddingColumnV2.__new__(cls, *args, **kwargs)

  def __init__(self, *args, **kwargs):
    # For __init__, just capture the inference dense shape and call parent.
    if 'tensor_core_shape' in kwargs:
      self._tensor_core_shape = kwargs['tensor_core_shape']
      del kwargs['tensor_core_shape']
    if 'embedding_lookup_device' in kwargs:
      self._embedding_lookup_device = kwargs['embedding_lookup_device']
      del kwargs['embedding_lookup_device']
    _TPUSharedEmbeddingColumnV2.__init__(self, *args, **kwargs)

  def _get_dense_tensor_internal(self, transformation_cache, state_manager):
    """Private method that follows _get_dense_tensor_internal."""
    if (tpu.under_tpu_inference_context() and
        self._embedding_lookup_device == EmbeddingDevice.TPU_EMBEDDING_CORE):
      raise ValueError('Using embedding_lookup_device=tpu_embedding_core '
                       'during inference is not supported.')
    if self._embedding_lookup_device == EmbeddingDevice.CPU:
      if tpu.under_tpu_inference_context():
        return super(_TPUSharedDeviceSpecificEmbeddingColumnV2,
                     self)._get_dense_tensor_internal(transformation_cache,
                                                      state_manager)
      else:
        raise ValueError(
            'Using TPUSharedEmbeddingColumn with '
            'embedding_lookup_device="cpu" during training is not supported.')
    sparse_tensor = transformation_cache.get(self.categorical_column.name,
                                             state_manager)

    # Use outside compile to densify and pad the input tensors.
    def host_computation():
      return pad_sparse_embedding_lookup_indices(sparse_tensor,
                                                 self._tensor_core_shape[1])

    values, mask = tpu.outside_compilation(host_computation)

    # Do a dense embedding lookup on TensorCore.
    embedding_weights = self.shared_embedding_column_creator.embedding_weights
    embedding = sparse_embedding_aggregate_slice(embedding_weights,
                                                 (values, mask),
                                                 self.get_combiner())
    return embedding
