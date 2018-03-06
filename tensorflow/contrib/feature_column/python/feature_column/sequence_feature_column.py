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
"""Experimental methods for tf.feature_column sequence input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import collections


from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope

# pylint: disable=protected-access
# TODO(b/73827486): Support SequenceExample.


def sequence_input_layer(
    features,
    feature_columns,
    weight_collections=None,
    trainable=True):
  """"Builds input layer for sequence input.

  All `feature_columns` must be sequence dense columns with the same
  `sequence_length`. The output of this method can be fed into sequence
  networks, such as RNN.

  The output of this method is a 3D `Tensor` of shape `[batch_size, T, D]`.
  `T` is the maximum sequence length for this batch, which could differ from
  batch to batch.

  If multiple `feature_columns` are given with `Di` `num_elements` each, their
  outputs are concatenated. So, the final `Tensor` has shape
  `[batch_size, T, D0 + D1 + ... + Dn]`.

  Example:

  ```python
  rating = sequence_numeric_column('rating')
  watches = sequence_categorical_column_with_identity(
      'watches', num_buckets=1000)
  watches_embedding = embedding_column(watches, dimension=10)
  columns = [rating, watches]

  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    features: A dict mapping keys to tensors.
    feature_columns: An iterable of dense sequence columns. Valid columns are
      - `embedding_column` that wraps a `sequence_categorical_column_with_*`
      - `sequence_numeric_column`.
    weight_collections: A list of collection names to which the Variable will be
      added. Note that variables will also be added to collections
      `tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES`.

  Returns:
    An `(input_layer, sequence_length)` tuple where:
    - input_layer: A float `Tensor` of shape `[batch_size, T, D]`.
        `T` is the maximum sequence length for this batch, which could differ
        from batch to batch. `D` is the sum of `num_elements` for all
        `feature_columns`.
    - sequence_length: An int `Tensor` of shape `[batch_size]`. The sequence
        length for each example.

  Raises:
    ValueError: If any of the `feature_columns` is the wrong type.
  """
  feature_columns = fc._clean_feature_columns(feature_columns)
  for c in feature_columns:
    if not isinstance(c, _SequenceDenseColumn):
      raise ValueError(
          'All feature_columns must be of type _SequenceDenseColumn. '
          'Given (type {}): {}'.format(type(c), c))

  with variable_scope.variable_scope(
      None, default_name='sequence_input_layer', values=features.values()):
    builder = fc._LazyBuilder(features)
    output_tensors = []
    sequence_lengths = []
    ordered_columns = []
    for column in sorted(feature_columns, key=lambda x: x.name):
      ordered_columns.append(column)
      with variable_scope.variable_scope(
          None, default_name=column._var_scope_name):
        dense_tensor, sequence_length = column._get_sequence_dense_tensor(
            builder,
            weight_collections=weight_collections,
            trainable=trainable)
        # Flattens the final dimension to produce a 3D Tensor.
        num_elements = column._variable_shape.num_elements()
        shape = array_ops.shape(dense_tensor)
        output_tensors.append(
            array_ops.reshape(
                dense_tensor,
                shape=array_ops.concat([shape[:2], [num_elements]], axis=0)))
        sequence_lengths.append(sequence_length)
    fc._verify_static_batch_size_equality(output_tensors, ordered_columns)
    fc._verify_static_batch_size_equality(sequence_lengths, ordered_columns)
    sequence_length = _assert_all_equal_and_return(sequence_lengths)
    return array_ops.concat(output_tensors, -1), sequence_length


def sequence_categorical_column_with_identity(
    key, num_buckets, default_value=None):
  """Returns a feature column that represents sequences of integers.

  Example:

  ```python
  watches = sequence_categorical_column_with_identity(
      'watches', num_buckets=1000)
  watches_embedding = embedding_column(watches, dimension=10)
  columns = [watches_embedding]

  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input feature.
    num_buckets: Range of inputs. Namely, inputs are expected to be in the
      range `[0, num_buckets)`.
    default_value: If `None`, this column's graph operations will fail for
      out-of-range inputs. Otherwise, this value must be in the range
      `[0, num_buckets)`, and will replace out-of-range inputs.

  Returns:
    A `_SequenceCategoricalColumn`.
  """
  return _SequenceCategoricalColumn(
      fc.categorical_column_with_identity(
          key=key,
          num_buckets=num_buckets,
          default_value=default_value))


def sequence_categorical_column_with_hash_bucket(
    key, hash_bucket_size, dtype=dtypes.string):
  """A sequence of categorical terms where ids are set by hashing.

  Example:

  ```python
  tokens = sequence_categorical_column_with_hash_bucket(
      'tokens', hash_bucket_size=1000)
  tokens_embedding = embedding_column(tokens, dimension=10)
  columns = [tokens_embedding]

  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input feature.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `_SequenceCategoricalColumn`.
  """
  return _SequenceCategoricalColumn(
      fc.categorical_column_with_hash_bucket(
          key=key,
          hash_bucket_size=hash_bucket_size,
          dtype=dtype))


def sequence_categorical_column_with_vocabulary_file(
    key, vocabulary_file, vocabulary_size=None, num_oov_buckets=0,
    default_value=None, dtype=dtypes.string):
  """A sequence of categorical terms where ids use a vocabulary file.

  Example:

  ```python
  states = sequence_categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
      num_oov_buckets=5)
  states_embedding = embedding_column(states, dimension=10)
  columns = [states_embedding]

  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input feature.
    vocabulary_file: The vocabulary file name.
    vocabulary_size: Number of the elements in the vocabulary. This must be no
      greater than length of `vocabulary_file`, if less than length, later
      values are ignored. If None, it is set to the length of `vocabulary_file`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
      the input value. A positive `num_oov_buckets` can not be specified with
      `default_value`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `_SequenceCategoricalColumn`.
  """
  return _SequenceCategoricalColumn(
      fc.categorical_column_with_vocabulary_file(
          key=key,
          vocabulary_file=vocabulary_file,
          vocabulary_size=vocabulary_size,
          num_oov_buckets=num_oov_buckets,
          default_value=default_value,
          dtype=dtype))


def sequence_categorical_column_with_vocabulary_list(
    key, vocabulary_list, dtype=None, default_value=-1, num_oov_buckets=0):
  """A sequence of categorical terms where ids use an in-memory list.

  Example:

  ```python
  colors = sequence_categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
      num_oov_buckets=2)
  colors_embedding = embedding_column(colors, dimension=3)
  columns = [colors_embedding]

  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input feature.
    vocabulary_list: An ordered iterable defining the vocabulary. Each feature
      is mapped to the index of its value (if present) in `vocabulary_list`.
      Must be castable to `dtype`.
    dtype: The type of features. Only string and integer types are supported.
      If `None`, it will be inferred from `vocabulary_list`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets)` based on a
      hash of the input value. A positive `num_oov_buckets` can not be specified
      with `default_value`.

  Returns:
    A `_SequenceCategoricalColumn`.
  """
  return _SequenceCategoricalColumn(
      fc.categorical_column_with_vocabulary_list(
          key=key,
          vocabulary_list=vocabulary_list,
          dtype=dtype,
          default_value=default_value,
          num_oov_buckets=num_oov_buckets))


# TODO(b/73160931): Merge with embedding_column
def _sequence_embedding_column(
    categorical_column, dimension, initializer=None, ckpt_to_load_from=None,
    tensor_name_in_ckpt=None, max_norm=None, trainable=True):
  """Returns a feature column that represents sequences of embeddings.

  Use this to convert sequence categorical data into dense representation for
  input to sequence NN, such as RNN.

  Example:

  ```python
  watches = sequence_categorical_column_with_identity(
      'watches', num_buckets=1000)
  watches_embedding = _sequence_embedding_column(watches, dimension=10)
  columns = [watches]

  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    categorical_column: A `_SequenceCategoricalColumn` created with a
      `sequence_cateogrical_column_with_*` function.
    dimension: Integer dimension of the embedding.
    initializer: Initializer function used to initialize the embeddings.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from
      which to restore the column weights. Required if `ckpt_to_load_from` is
      not `None`.
    max_norm: If not `None`, embedding values are l2-normalized to this value.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    A `_SequenceCategoricalToDenseColumn`.

  Raises:
    ValueError: If `categorical_column` is not the right type.
  """
  if not isinstance(categorical_column, _SequenceCategoricalColumn):
    raise ValueError(
        'categorical_column must be of type _SequenceCategoricalColumn. '
        'Given (type {}): {}'.format(
            type(categorical_column), categorical_column))
  return _SequenceCategoricalToDenseColumn(
      fc.embedding_column(
          categorical_column,
          dimension=dimension,
          initializer=initializer,
          ckpt_to_load_from=ckpt_to_load_from,
          tensor_name_in_ckpt=tensor_name_in_ckpt,
          max_norm=max_norm,
          trainable=trainable))


# TODO(b/73160931): Merge with indicator_column
def _sequence_indicator_column(categorical_column):
  """Returns a feature column that represents sequences of multi-hot tensors.

  Use this to convert sequence categorical data into dense representation for
  input to sequence NN, such as RNN.

  Example:

  ```python
  colors = sequence_categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('R', 'G', 'B', 'Y'))
  colors_indicator = _sequence_indicator_column(colors)
  columns = [colors]

  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    categorical_column: A `_SequenceCategoricalColumn` created with a
      `sequence_cateogrical_column_with_*` function.

  Returns:
    A `_SequenceCategoricalToDenseColumn`.

  Raises:
    ValueError: If `categorical_column` is not the right type.
  """
  if not isinstance(categorical_column, _SequenceCategoricalColumn):
    raise ValueError(
        'categorical_column must be of type _SequenceCategoricalColumn. '
        'Given (type {}): {}'.format(
            type(categorical_column), categorical_column))
  return _SequenceCategoricalToDenseColumn(
      fc.indicator_column(categorical_column))


def sequence_numeric_column(
    key,
    shape=(1,),
    default_value=0.,
    dtype=dtypes.float32):
  """Returns a feature column that represents sequences of numeric data.

  Example:

  ```python
  temperature = sequence_numeric_column('temperature')
  columns = [temperature]

  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input features.
    shape: The shape of the input data per sequence id. E.g. if `shape=(2,)`,
      each example must contain `2 * sequence_length` values.
    default_value: A single value compatible with `dtype` that is used for
      padding the sparse data into a dense `Tensor`.
    dtype: The type of values.

  Returns:
    A `_SequenceNumericColumn`.
  """
  # TODO(b/73160931): Add validations.
  return _SequenceNumericColumn(
      key,
      shape=shape,
      default_value=default_value,
      dtype=dtype)


def _assert_all_equal_and_return(tensors, name=None):
  """Asserts that all tensors are equal and returns the first one."""
  with ops.name_scope(name, 'assert_all_equal', values=tensors):
    if len(tensors) == 1:
      return tensors[0]
    assert_equal_ops = []
    for t in tensors[1:]:
      assert_equal_ops.append(check_ops.assert_equal(tensors[0], t))
    with ops.control_dependencies(assert_equal_ops):
      return array_ops.identity(tensors[0])


class _SequenceDenseColumn(fc._FeatureColumn):
  """Represents dense sequence data."""

  __metaclass__ = abc.ABCMeta

  TensorSequenceLengthPair = collections.namedtuple(  # pylint: disable=invalid-name
      'TensorSequenceLengthPair', ['dense_tensor', 'sequence_length'])

  @abc.abstractproperty
  def _variable_shape(self):
    """`TensorShape` without batch and sequence dimensions."""
    pass

  @abc.abstractmethod
  def _get_sequence_dense_tensor(
      self, inputs, weight_collections=None, trainable=None):
    """Returns a `TensorSequenceLengthPair`."""
    pass


def _sequence_length_from_sparse_tensor(sp_tensor, num_elements=1):
  with ops.name_scope(None, 'sequence_length') as name_scope:
    row_ids = sp_tensor.indices[:, 0]
    column_ids = sp_tensor.indices[:, 1]
    column_ids += array_ops.ones_like(column_ids)
    seq_length = math_ops.to_int64(
        math_ops.segment_max(column_ids, segment_ids=row_ids) / num_elements)
    # If the last n rows do not have ids, seq_length will have shape
    # [batch_size - n]. Pad the remaining values with zeros.
    n_pad = array_ops.shape(sp_tensor)[:1] - array_ops.shape(seq_length)[:1]
    padding = array_ops.zeros(n_pad, dtype=seq_length.dtype)
    return array_ops.concat([seq_length, padding], axis=0, name=name_scope)


class _SequenceCategoricalColumn(
    fc._CategoricalColumn,
    collections.namedtuple(
        '_SequenceCategoricalColumn', ['categorical_column'])):
  """Represents sequences of categorical data."""

  @property
  def name(self):
    return self.categorical_column.name

  @property
  def _parse_example_spec(self):
    return self.categorical_column._parse_example_spec

  def _transform_feature(self, inputs):
    return self.categorical_column._transform_feature(inputs)

  @property
  def _num_buckets(self):
    return self.categorical_column._num_buckets

  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)
    id_tensor = sparse_tensors.id_tensor
    weight_tensor = sparse_tensors.weight_tensor
    # Expands final dimension, so that embeddings are not combined during
    # embedding lookup.
    check_id_rank = check_ops.assert_equal(
        array_ops.rank(id_tensor), 2,
        data=[
            'Column {} expected ID tensor of rank 2. '.format(self.name),
            'id_tensor shape: ', array_ops.shape(id_tensor)])
    with ops.control_dependencies([check_id_rank]):
      id_tensor = sparse_ops.sparse_reshape(
          id_tensor,
          shape=array_ops.concat([id_tensor.dense_shape, [1]], axis=0))
    if weight_tensor is not None:
      check_weight_rank = check_ops.assert_equal(
          array_ops.rank(weight_tensor), 2,
          data=[
              'Column {} expected weight tensor of rank 2.'.format(self.name),
              'weight_tensor shape:', array_ops.shape(weight_tensor)])
      with ops.control_dependencies([check_weight_rank]):
        weight_tensor = sparse_ops.sparse_reshape(
            weight_tensor,
            shape=array_ops.concat([weight_tensor.dense_shape, [1]], axis=0))
    return fc._CategoricalColumn.IdWeightPair(id_tensor, weight_tensor)

  def _sequence_length(self, inputs):
    sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)
    return _sequence_length_from_sparse_tensor(sparse_tensors.id_tensor)


class _SequenceCategoricalToDenseColumn(
    _SequenceDenseColumn,
    collections.namedtuple(
        '_SequenceCategoricalToDenseColumn', ['dense_column'])):
  """Densifies a _SequenceCategoricalColumn using the specified column."""

  @property
  def name(self):
    return self.dense_column.name

  @property
  def _parse_example_spec(self):
    return self.dense_column._parse_example_spec

  def _transform_feature(self, inputs):
    return self.dense_column._transform_feature(inputs)

  @property
  def _variable_shape(self):
    return self.dense_column._variable_shape

  def _get_sequence_dense_tensor(
      self, inputs, weight_collections=None, trainable=None):
    dense_tensor = self.dense_column._get_dense_tensor(
        inputs=inputs,
        weight_collections=weight_collections,
        trainable=trainable)
    sequence_length = self.dense_column.categorical_column._sequence_length(
        inputs)
    return _SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)


class _SequenceNumericColumn(
    _SequenceDenseColumn,
    collections.namedtuple(
        '_SequenceNumericColumn',
        ['key', 'shape', 'default_value', 'dtype'])):
  """Represents sequences of numeric data."""

  @property
  def name(self):
    return self.key

  @property
  def _parse_example_spec(self):
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  def _transform_feature(self, inputs):
    return inputs.get(self.key)

  @property
  def _variable_shape(self):
    return tensor_shape.TensorShape(self.shape)

  def _get_sequence_dense_tensor(
      self, inputs, weight_collections=None, trainable=None):
    # Do nothing with weight_collections and trainable since no variables are
    # created in this function.
    del weight_collections
    del trainable
    sp_tensor = inputs.get(self)
    dense_tensor = sparse_ops.sparse_tensor_to_dense(
        sp_tensor, default_value=self.default_value)
    # Reshape into [batch_size, T, variable_shape].
    dense_shape = array_ops.concat(
        [array_ops.shape(dense_tensor)[:1], [-1], self._variable_shape],
        axis=0)
    dense_tensor = array_ops.reshape(dense_tensor, shape=dense_shape)
    sequence_length = _sequence_length_from_sparse_tensor(
        sp_tensor, num_elements=self._variable_shape.num_elements())
    return _SequenceDenseColumn.TensorSequenceLengthPair(
        dense_tensor=dense_tensor, sequence_length=sequence_length)

# pylint: enable=protected-access
