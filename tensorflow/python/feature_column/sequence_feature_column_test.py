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
"""Tests for sequential_feature_column."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.feature_column import serialization
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


def _initialized_session(config=None):
  sess = session.Session(config=config)
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


@test_util.run_all_in_graph_and_eager_modes
class ConcatenateContextInputTest(test.TestCase, parameterized.TestCase):
  """Tests the utility fn concatenate_context_input."""

  def test_concatenate_context_input(self):
    seq_input = ops.convert_to_tensor(np.arange(12).reshape(2, 3, 2))
    context_input = ops.convert_to_tensor(np.arange(10).reshape(2, 5))
    seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
    context_input = math_ops.cast(context_input, dtype=dtypes.float32)
    input_layer = sfc.concatenate_context_input(context_input, seq_input)

    expected = np.array([
        [[0, 1, 0, 1, 2, 3, 4], [2, 3, 0, 1, 2, 3, 4], [4, 5, 0, 1, 2, 3, 4]],
        [[6, 7, 5, 6, 7, 8, 9], [8, 9, 5, 6, 7, 8, 9], [10, 11, 5, 6, 7, 8, 9]]
    ], dtype=np.float32)
    output = self.evaluate(input_layer)
    self.assertAllEqual(expected, output)

  @parameterized.named_parameters(
      {'testcase_name': 'rank_lt_3',
       'seq_input_arg': np.arange(100).reshape(10, 10)},
      {'testcase_name': 'rank_gt_3',
       'seq_input_arg': np.arange(100).reshape(5, 5, 2, 2)}
      )
  def test_sequence_input_throws_error(self, seq_input_arg):
    seq_input = ops.convert_to_tensor(seq_input_arg)
    context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
    seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
    context_input = math_ops.cast(context_input, dtype=dtypes.float32)
    with self.assertRaisesRegex(ValueError, 'sequence_input must have rank 3'):
      sfc.concatenate_context_input(context_input, seq_input)

  @parameterized.named_parameters(
      {'testcase_name': 'rank_lt_2',
       'context_input_arg': np.arange(100)},
      {'testcase_name': 'rank_gt_2',
       'context_input_arg': np.arange(100).reshape(5, 5, 4)}
      )
  def test_context_input_throws_error(self, context_input_arg):
    context_input = ops.convert_to_tensor(context_input_arg)
    seq_input = ops.convert_to_tensor(np.arange(100).reshape(5, 5, 4))
    seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
    context_input = math_ops.cast(context_input, dtype=dtypes.float32)
    with self.assertRaisesRegex(ValueError, 'context_input must have rank 2'):
      sfc.concatenate_context_input(context_input, seq_input)

  def test_integer_seq_input_throws_error(self):
    seq_input = ops.convert_to_tensor(np.arange(100).reshape(5, 5, 4))
    context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
    context_input = math_ops.cast(context_input, dtype=dtypes.float32)
    with self.assertRaisesRegex(TypeError,
                                'sequence_input must have dtype float32'):
      sfc.concatenate_context_input(context_input, seq_input)

  def test_integer_context_input_throws_error(self):
    seq_input = ops.convert_to_tensor(np.arange(100).reshape(5, 5, 4))
    context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
    seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
    with self.assertRaisesRegex(TypeError,
                                'context_input must have dtype float32'):
      sfc.concatenate_context_input(context_input, seq_input)


def _assert_sparse_tensor_value(test_case, expected, actual):
  _assert_sparse_tensor_indices_shape(test_case, expected, actual)

  test_case.assertEqual(
      np.array(expected.values).dtype, np.array(actual.values).dtype)
  test_case.assertAllEqual(expected.values, actual.values)


def _assert_sparse_tensor_indices_shape(test_case, expected, actual):
  test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
  test_case.assertAllEqual(expected.indices, actual.indices)

  test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
  test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


def _get_sequence_dense_tensor(column, features):
  return column.get_sequence_dense_tensor(
      fc.FeatureTransformationCache(features), None)


def _get_sequence_dense_tensor_state(column, features):
  state_manager = fc._StateManagerImpl(
      fc_lib.DenseFeatures(column), trainable=True)
  column.create_state(state_manager)
  dense_tensor, lengths = column.get_sequence_dense_tensor(
      fc.FeatureTransformationCache(features), state_manager)
  return dense_tensor, lengths, state_manager


def _get_sparse_tensors(column, features):
  return column.get_sparse_tensors(
      fc.FeatureTransformationCache(features), None)


@test_util.run_all_in_graph_and_eager_modes
class SequenceCategoricalColumnWithIdentityTest(
    test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (1, 2, 0),
           'dense_shape': (2, 2)},
       'expected_args': {
           'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
           'values': np.array((1, 2, 0), dtype=np.int64),
           'dense_shape': (2, 2, 1)}},
      {'testcase_name': '3D',
       'inputs_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': (6, 7, 8),
           'dense_shape': (2, 2, 2)},
       'expected_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': np.array((6, 7, 8), dtype=np.int64),
           'dense_shape': (2, 2, 2)}}
      )
  def test_get_sparse_tensors(self, inputs_args, expected_args):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    expected = sparse_tensor.SparseTensorValue(**expected_args)
    column = sfc.sequence_categorical_column_with_identity('aaa', num_buckets=9)

    id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

    self.assertIsNone(id_weight_pair.weight_tensor)
    _assert_sparse_tensor_value(
        self, expected, self.evaluate(id_weight_pair.id_tensor))

  def test_serialization(self):
    """Tests that column can be serialized."""
    parent = sfc.sequence_categorical_column_with_identity(
        'animal', num_buckets=4)
    animal = fc.indicator_column(parent)

    config = animal.get_config()
    self.assertEqual(
        {
            'categorical_column': {
                'class_name': 'SequenceCategoricalColumn',
                'config': {
                    'categorical_column': {
                        'class_name': 'IdentityCategoricalColumn',
                        'config': {
                            'default_value': None,
                            'key': 'animal',
                            'number_buckets': 4
                        }
                    }
                }
            }
        }, config)

    new_animal = fc.IndicatorColumn.from_config(config)
    self.assertEqual(animal, new_animal)
    self.assertIsNot(parent, new_animal.categorical_column)

    new_animal = fc.IndicatorColumn.from_config(
        config,
        columns_by_name={
            serialization._column_name_with_class_name(parent): parent
        })
    self.assertEqual(animal, new_animal)
    self.assertIs(parent, new_animal.categorical_column)


@test_util.run_all_in_graph_and_eager_modes
class SequenceCategoricalColumnWithHashBucketTest(
    test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': ('omar', 'stringer', 'marlo'),
           'dense_shape': (2, 2)},
       'expected_args': {
           'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
           # Ignored to avoid hash dependence in test.
           'values': np.array((0, 0, 0), dtype=np.int64),
           'dense_shape': (2, 2, 1)}},
      {'testcase_name': '3D',
       'inputs_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': ('omar', 'stringer', 'marlo'),
           'dense_shape': (2, 2, 2)},
       'expected_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           # Ignored to avoid hash dependence in test.
           'values': np.array((0, 0, 0), dtype=np.int64),
           'dense_shape': (2, 2, 2)}}
      )
  def test_get_sparse_tensors(self, inputs_args, expected_args):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    expected = sparse_tensor.SparseTensorValue(**expected_args)
    column = sfc.sequence_categorical_column_with_hash_bucket(
        'aaa', hash_bucket_size=10)

    id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

    self.assertIsNone(id_weight_pair.weight_tensor)
    _assert_sparse_tensor_indices_shape(
        self, expected, self.evaluate(id_weight_pair.id_tensor))


@test_util.run_all_in_graph_and_eager_modes
class SequenceCategoricalColumnWithVocabularyFileTest(
    test.TestCase, parameterized.TestCase):

  def _write_vocab(self, vocab_strings, file_name):
    vocab_file = os.path.join(self.get_temp_dir(), file_name)
    with open(vocab_file, 'w') as f:
      f.write('\n'.join(vocab_strings))
    return vocab_file

  def setUp(self):
    super(SequenceCategoricalColumnWithVocabularyFileTest, self).setUp()

    vocab_strings = ['omar', 'stringer', 'marlo']
    self._wire_vocabulary_file_name = self._write_vocab(vocab_strings,
                                                        'wire_vocabulary.txt')
    self._wire_vocabulary_size = 3

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': ('marlo', 'skywalker', 'omar'),
           'dense_shape': (2, 2)},
       'expected_args': {
           'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
           'values': np.array((2, -1, 0), dtype=np.int64),
           'dense_shape': (2, 2, 1)}},
      {'testcase_name': '3D',
       'inputs_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': ('omar', 'skywalker', 'marlo'),
           'dense_shape': (2, 2, 2)},
       'expected_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': np.array((0, -1, 2), dtype=np.int64),
           'dense_shape': (2, 2, 2)}}
      )
  def test_get_sparse_tensors(self, inputs_args, expected_args):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    expected = sparse_tensor.SparseTensorValue(**expected_args)
    column = sfc.sequence_categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)

    id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

    self.assertIsNone(id_weight_pair.weight_tensor)
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())
    _assert_sparse_tensor_value(
        self, expected, self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_dynamic_zero_length(self):
    """Tests _get_sparse_tensors with a dynamic sequence length."""
    with ops.Graph().as_default():
      inputs = sparse_tensor.SparseTensorValue(
          indices=np.zeros((0, 2)), values=[], dense_shape=(2, 0))
      expected = sparse_tensor.SparseTensorValue(
          indices=np.zeros((0, 3)),
          values=np.array((), dtype=np.int64),
          dense_shape=(2, 0, 1))
      column = sfc.sequence_categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size)
      input_placeholder_shape = list(inputs.dense_shape)
      # Make second dimension (sequence length) dynamic.
      input_placeholder_shape[1] = None
      input_placeholder = array_ops.sparse_placeholder(
          dtypes.string, shape=input_placeholder_shape)
      id_weight_pair = _get_sparse_tensors(column, {'aaa': input_placeholder})

      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session() as sess:
        result = id_weight_pair.id_tensor.eval(
            session=sess, feed_dict={input_placeholder: inputs})
        _assert_sparse_tensor_value(
            self, expected, result)


@test_util.run_all_in_graph_and_eager_modes
class SequenceCategoricalColumnWithVocabularyListTest(
    test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': ('marlo', 'skywalker', 'omar'),
           'dense_shape': (2, 2)},
       'expected_args': {
           'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
           'values': np.array((2, -1, 0), dtype=np.int64),
           'dense_shape': (2, 2, 1)}},
      {'testcase_name': '3D',
       'inputs_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': ('omar', 'skywalker', 'marlo'),
           'dense_shape': (2, 2, 2)},
       'expected_args': {
           'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
           'values': np.array((0, -1, 2), dtype=np.int64),
           'dense_shape': (2, 2, 2)}}
      )
  def test_get_sparse_tensors(self, inputs_args, expected_args):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    expected = sparse_tensor.SparseTensorValue(**expected_args)
    column = sfc.sequence_categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'))

    id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

    self.assertIsNone(id_weight_pair.weight_tensor)
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())
    _assert_sparse_tensor_value(
        self, expected, self.evaluate(id_weight_pair.id_tensor))


@test_util.run_all_in_graph_and_eager_modes
class SequenceEmbeddingColumnTest(
    test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           # example 2, ids []
           # example 3, ids [1]
           'indices': ((0, 0), (1, 0), (1, 1), (3, 0)),
           'values': (2, 0, 1, 1),
           'dense_shape': (4, 2)},
       'expected': [
           # example 0, ids [2]
           [[7., 11.], [0., 0.]],
           # example 1, ids [0, 1]
           [[1., 2.], [3., 5.]],
           # example 2, ids []
           [[0., 0.], [0., 0.]],
           # example 3, ids [1]
           [[3., 5.], [0., 0.]]]},
      {'testcase_name': '3D',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           # example 2, ids []
           # example 3, ids [[1], [0, 2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0),
                       (3, 0, 0), (3, 1, 0), (3, 1, 1)),
           'values': (2, 0, 1, 2, 1, 0, 2),
           'dense_shape': (4, 2, 2)},
       'expected': [
           # example 0, ids [[2]]
           [[7., 11.], [0., 0.]],
           # example 1, ids [[0, 1], [2]]
           [[2, 3.5], [7., 11.]],
           # example 2, ids []
           [[0., 0.], [0., 0.]],
           # example 3, ids [[1], [0, 2]]
           [[3., 5.], [4., 6.5]]]}
      )
  def test_get_sequence_dense_tensor(self, inputs_args, expected):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    vocabulary_size = 3
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        initializer=_initializer)

    embedding_lookup, _, state_manager = _get_sequence_dense_tensor_state(
        embedding_column, {'aaa': inputs})

    variables = state_manager._layer.weights
    self.evaluate(variables_lib.global_variables_initializer())
    self.assertCountEqual(
        ('embedding_weights:0',), tuple([v.name for v in variables]))
    self.assertAllEqual(embedding_values, self.evaluate(variables[0]))
    self.assertAllEqual(expected, self.evaluate(embedding_lookup))

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (2, 0, 1),
           'dense_shape': (2, 2)},
       'expected_sequence_length': [1, 2]},
      {'testcase_name': '3D',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
           'values': (2, 0, 1, 2),
           'dense_shape': (2, 2, 2)},
       'expected_sequence_length': [1, 2]}
      )
  def test_sequence_length(self, inputs_args, expected_sequence_length):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    vocabulary_size = 3

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=2)

    _, sequence_length, _ = _get_sequence_dense_tensor_state(
        embedding_column, {'aaa': inputs})

    sequence_length = self.evaluate(sequence_length)
    self.assertAllEqual(expected_sequence_length, sequence_length)
    self.assertEqual(np.int64, sequence_length.dtype)

  def test_sequence_length_with_empty_rows(self):
    """Tests _sequence_length when some examples do not have ids."""
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids []
        # example 1, ids [2]
        # example 2, ids [0, 1]
        # example 3, ids []
        # example 4, ids [1]
        # example 5, ids []
        indices=((1, 0), (2, 0), (2, 1), (4, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(6, 2))
    expected_sequence_length = [0, 1, 2, 0, 1, 0]

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=2)

    _, sequence_length, _ = _get_sequence_dense_tensor_state(
        embedding_column, {'aaa': sparse_input})

    self.assertAllEqual(
        expected_sequence_length, self.evaluate(sequence_length))


class SequenceSharedEmbeddingColumnTest(test.TestCase):

  def test_get_sequence_dense_tensor(self):
    vocabulary_size = 3
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    with ops.Graph().as_default():
      sparse_input_a = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          # example 2, ids []
          # example 3, ids [1]
          indices=((0, 0), (1, 0), (1, 1), (3, 0)),
          values=(2, 0, 1, 1),
          dense_shape=(4, 2))
      sparse_input_b = sparse_tensor.SparseTensorValue(
          # example 0, ids [1]
          # example 1, ids [0, 2]
          # example 2, ids [0]
          # example 3, ids []
          indices=((0, 0), (1, 0), (1, 1), (2, 0)),
          values=(1, 0, 2, 0),
          dense_shape=(4, 2))

      expected_lookups_a = [
          # example 0, ids [2]
          [[7., 11.], [0., 0.]],
          # example 1, ids [0, 1]
          [[1., 2.], [3., 5.]],
          # example 2, ids []
          [[0., 0.], [0., 0.]],
          # example 3, ids [1]
          [[3., 5.], [0., 0.]],
      ]

      expected_lookups_b = [
          # example 0, ids [1]
          [[3., 5.], [0., 0.]],
          # example 1, ids [0, 2]
          [[1., 2.], [7., 11.]],
          # example 2, ids [0]
          [[1., 2.], [0., 0.]],
          # example 3, ids []
          [[0., 0.], [0., 0.]],
      ]

      categorical_column_a = sfc.sequence_categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = sfc.sequence_categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      shared_embedding_columns = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer)

      embedding_lookup_a = _get_sequence_dense_tensor(
          shared_embedding_columns[0], {'aaa': sparse_input_a})[0]
      embedding_lookup_b = _get_sequence_dense_tensor(
          shared_embedding_columns[1], {'bbb': sparse_input_b})[0]

      self.evaluate(variables_lib.global_variables_initializer())
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertItemsEqual(('aaa_bbb_shared_embedding:0',),
                            tuple([v.name for v in global_vars]))
      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
      self.assertAllEqual(
          expected_lookups_a, self.evaluate(embedding_lookup_a))
      self.assertAllEqual(expected_lookups_b, self.evaluate(embedding_lookup_b))

  def test_sequence_length(self):
    with ops.Graph().as_default():
      vocabulary_size = 3

      sparse_input_a = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          indices=((0, 0), (1, 0), (1, 1)),
          values=(2, 0, 1),
          dense_shape=(2, 2))
      expected_sequence_length_a = [1, 2]
      categorical_column_a = sfc.sequence_categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)

      sparse_input_b = sparse_tensor.SparseTensorValue(
          # example 0, ids [0, 2]
          # example 1, ids [1]
          indices=((0, 0), (0, 1), (1, 0)),
          values=(0, 2, 1),
          dense_shape=(2, 2))
      expected_sequence_length_b = [2, 1]
      categorical_column_b = sfc.sequence_categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      shared_embedding_columns = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b], dimension=2)

      sequence_length_a = _get_sequence_dense_tensor(
          shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
      sequence_length_b = _get_sequence_dense_tensor(
          shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

      with _initialized_session() as sess:
        sequence_length_a = sess.run(sequence_length_a)
        self.assertAllEqual(expected_sequence_length_a, sequence_length_a)
        self.assertEqual(np.int64, sequence_length_a.dtype)
        sequence_length_b = sess.run(sequence_length_b)
        self.assertAllEqual(expected_sequence_length_b, sequence_length_b)
        self.assertEqual(np.int64, sequence_length_b.dtype)

  def test_sequence_length_with_empty_rows(self):
    """Tests _sequence_length when some examples do not have ids."""
    with ops.Graph().as_default():
      vocabulary_size = 3
      sparse_input_a = sparse_tensor.SparseTensorValue(
          # example 0, ids []
          # example 1, ids [2]
          # example 2, ids [0, 1]
          # example 3, ids []
          # example 4, ids [1]
          # example 5, ids []
          indices=((1, 0), (2, 0), (2, 1), (4, 0)),
          values=(2, 0, 1, 1),
          dense_shape=(6, 2))
      expected_sequence_length_a = [0, 1, 2, 0, 1, 0]
      categorical_column_a = sfc.sequence_categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)

      sparse_input_b = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids []
          # example 2, ids []
          # example 3, ids []
          # example 4, ids [1]
          # example 5, ids [0, 1]
          indices=((0, 0), (4, 0), (5, 0), (5, 1)),
          values=(2, 1, 0, 1),
          dense_shape=(6, 2))
      expected_sequence_length_b = [1, 0, 0, 0, 1, 2]
      categorical_column_b = sfc.sequence_categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)

      shared_embedding_columns = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b], dimension=2)

      sequence_length_a = _get_sequence_dense_tensor(
          shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
      sequence_length_b = _get_sequence_dense_tensor(
          shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

      with _initialized_session() as sess:
        self.assertAllEqual(
            expected_sequence_length_a, sequence_length_a.eval(session=sess))
        self.assertAllEqual(
            expected_sequence_length_b, sequence_length_b.eval(session=sess))


@test_util.run_all_in_graph_and_eager_modes
class SequenceIndicatorColumnTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           # example 2, ids []
           # example 3, ids [1]
           'indices': ((0, 0), (1, 0), (1, 1), (3, 0)),
           'values': (2, 0, 1, 1),
           'dense_shape': (4, 2)},
       'expected': [
           # example 0, ids [2]
           [[0., 0., 1.], [0., 0., 0.]],
           # example 1, ids [0, 1]
           [[1., 0., 0.], [0., 1., 0.]],
           # example 2, ids []
           [[0., 0., 0.], [0., 0., 0.]],
           # example 3, ids [1]
           [[0., 1., 0.], [0., 0., 0.]]]},
      {'testcase_name': '3D',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           # example 2, ids []
           # example 3, ids [[1], [2, 2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0),
                       (3, 0, 0), (3, 1, 0), (3, 1, 1)),
           'values': (2, 0, 1, 2, 1, 2, 2),
           'dense_shape': (4, 2, 2)},
       'expected': [
           # example 0, ids [[2]]
           [[0., 0., 1.], [0., 0., 0.]],
           # example 1, ids [[0, 1], [2]]
           [[1., 1., 0.], [0., 0., 1.]],
           # example 2, ids []
           [[0., 0., 0.], [0., 0., 0.]],
           # example 3, ids [[1], [2, 2]]
           [[0., 1., 0.], [0., 0., 2.]]]}
      )
  def test_get_sequence_dense_tensor(self, inputs_args, expected):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    vocabulary_size = 3

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column = fc.indicator_column(categorical_column)

    indicator_tensor, _ = _get_sequence_dense_tensor(
        indicator_column, {'aaa': inputs})

    self.assertAllEqual(expected, self.evaluate(indicator_tensor))

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (2, 0, 1),
           'dense_shape': (2, 2)},
       'expected_sequence_length': [1, 2]},
      {'testcase_name': '3D',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
           'values': (2, 0, 1, 2),
           'dense_shape': (2, 2, 2)},
       'expected_sequence_length': [1, 2]}
      )
  def test_sequence_length(self, inputs_args, expected_sequence_length):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    vocabulary_size = 3

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column = fc.indicator_column(categorical_column)

    _, sequence_length = _get_sequence_dense_tensor(
        indicator_column, {'aaa': inputs})

    sequence_length = self.evaluate(sequence_length)
    self.assertAllEqual(expected_sequence_length, sequence_length)
    self.assertEqual(np.int64, sequence_length.dtype)

  def test_sequence_length_with_empty_rows(self):
    """Tests _sequence_length when some examples do not have ids."""
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids []
        # example 1, ids [2]
        # example 2, ids [0, 1]
        # example 3, ids []
        # example 4, ids [1]
        # example 5, ids []
        indices=((1, 0), (2, 0), (2, 1), (4, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(6, 2))
    expected_sequence_length = [0, 1, 2, 0, 1, 0]

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column = fc.indicator_column(categorical_column)

    _, sequence_length = _get_sequence_dense_tensor(
        indicator_column, {'aaa': sparse_input})

    self.assertAllEqual(
        expected_sequence_length, self.evaluate(sequence_length))


@test_util.run_all_in_graph_and_eager_modes
class SequenceNumericColumnTest(test.TestCase, parameterized.TestCase):

  def test_defaults(self):
    a = sfc.sequence_numeric_column('aaa')
    self.assertEqual('aaa', a.key)
    self.assertEqual('aaa', a.name)
    self.assertEqual((1,), a.shape)
    self.assertEqual(0., a.default_value)
    self.assertEqual(dtypes.float32, a.dtype)
    self.assertIsNone(a.normalizer_fn)

  def test_shape_saved_as_tuple(self):
    a = sfc.sequence_numeric_column('aaa', shape=[1, 2])
    self.assertEqual((1, 2), a.shape)

  def test_shape_must_be_positive_integer(self):
    with self.assertRaisesRegex(TypeError, 'shape dimensions must be integer'):
      sfc.sequence_numeric_column('aaa', shape=[1.0])

    with self.assertRaisesRegex(ValueError,
                                'shape dimensions must be greater than 0'):
      sfc.sequence_numeric_column('aaa', shape=[0])

  def test_dtype_is_convertible_to_float(self):
    with self.assertRaisesRegex(ValueError,
                                'dtype must be convertible to float'):
      sfc.sequence_numeric_column('aaa', dtype=dtypes.string)

  def test_normalizer_fn_must_be_callable(self):
    with self.assertRaisesRegex(TypeError, 'must be a callable'):
      sfc.sequence_numeric_column('aaa', normalizer_fn='NotACallable')

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, values [0., 1]
           # example 1, [10.]
           'indices': ((0, 0), (0, 1), (1, 0)),
           'values': (0., 1., 10.),
           'dense_shape': (2, 2)},
       'expected': [
           [[0.], [1.]],
           [[10.], [0.]]]},
      {'testcase_name': '3D',
       'inputs_args': {
           # feature 0, ids [[20, 3], [5]]
           # feature 1, ids [[3], [8]]
           'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
           'values': (20, 3, 5., 3., 8.),
           'dense_shape': (2, 2, 2)},
       'expected': [
           [[20.], [3.], [5.], [0.]],
           [[3.], [0.], [8.], [0.]]]},
      )
  def test_get_sequence_dense_tensor(self, inputs_args, expected):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    numeric_column = sfc.sequence_numeric_column('aaa')

    dense_tensor, _ = _get_sequence_dense_tensor(
        numeric_column, {'aaa': inputs})
    self.assertAllEqual(expected, self.evaluate(dense_tensor))

  def test_get_sequence_dense_tensor_with_normalizer_fn(self):

    def _increment_two(input_sparse_tensor):
      return sparse_ops.sparse_add(
          input_sparse_tensor,
          sparse_tensor.SparseTensor(((0, 0), (1, 1)), (2.0, 2.0), (2, 2))
      )

    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values [[0.], [1]]
        # example 1, [[10.]]
        indices=((0, 0), (0, 1), (1, 0)),
        values=(0., 1., 10.),
        dense_shape=(2, 2))

    # Before _increment_two:
    #   [[0.], [1.]],
    #   [[10.], [0.]],
    # After _increment_two:
    #   [[2.], [1.]],
    #   [[10.], [2.]],
    expected_dense_tensor = [
        [[2.], [1.]],
        [[10.], [2.]],
    ]
    numeric_column = sfc.sequence_numeric_column(
        'aaa', normalizer_fn=_increment_two)

    dense_tensor, _ = _get_sequence_dense_tensor(
        numeric_column, {'aaa': sparse_input})

    self.assertAllEqual(
        expected_dense_tensor, self.evaluate(dense_tensor))

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'sparse_input_args': {
           # example 0, values [[[0., 1.],  [2., 3.]], [[4., 5.],  [6., 7.]]]
           # example 1, [[[10., 11.],  [12., 13.]]]
           'indices': ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                       (0, 7), (1, 0), (1, 1), (1, 2), (1, 3)),
           'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
           'dense_shape': (2, 8)},
       'expected_dense_tensor': [
           [[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
           [[[10., 11.], [12., 13.]], [[0., 0.], [0., 0.]]]]},
      {'testcase_name': '3D',
       'sparse_input_args': {
           'indices': ((0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6),
                       (0, 1, 0), (0, 1, 2), (0, 1, 4), (0, 1, 6),
                       (1, 0, 0), (1, 0, 2), (1, 0, 4), (1, 0, 6)),
           'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
           'dense_shape': (2, 2, 8)},
       'expected_dense_tensor': [
           [[[0., 0.], [1., 0.]], [[2., 0.], [3., 0.]],
            [[4., 0.], [5., 0.]], [[6., 0.], [7., 0.]]],
           [[[10., 0.], [11., 0.]], [[12., 0.], [13., 0.]],
            [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]]},
      )
  def test_get_dense_tensor_multi_dim(
      self, sparse_input_args, expected_dense_tensor):
    """Tests get_sequence_dense_tensor for multi-dim numeric_column."""
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)
    numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

    dense_tensor, _ = _get_sequence_dense_tensor(
        numeric_column, {'aaa': sparse_input})

    self.assertAllEqual(
        expected_dense_tensor, self.evaluate(dense_tensor))

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (2., 0., 1.),
           'dense_shape': (2, 2)},
       'expected_sequence_length': [1, 2],
       'shape': (1,)},
      {'testcase_name': '3D',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
           'values': (2., 0., 1., 2.),
           'dense_shape': (2, 2, 2)},
       'expected_sequence_length': [1, 2],
       'shape': (1,)},
      {'testcase_name': '2D_with_shape',
       'inputs_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (2., 0., 1.),
           'dense_shape': (2, 2)},
       'expected_sequence_length': [1, 1],
       'shape': (2,)},
      {'testcase_name': '3D_with_shape',
       'inputs_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
           'values': (2., 0., 1., 2.),
           'dense_shape': (2, 2, 2)},
       'expected_sequence_length': [1, 2],
       'shape': (2,)},
      )
  def test_sequence_length(self, inputs_args, expected_sequence_length, shape):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    numeric_column = sfc.sequence_numeric_column('aaa', shape=shape)

    _, sequence_length = _get_sequence_dense_tensor(
        numeric_column, {'aaa': inputs})

    sequence_length = self.evaluate(sequence_length)
    self.assertAllEqual(expected_sequence_length, sequence_length)
    self.assertEqual(np.int64, sequence_length.dtype)

  def test_sequence_length_with_empty_rows(self):
    """Tests _sequence_length when some examples do not have ids."""
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values []
        # example 1, values [[0.], [1.]]
        # example 2, [[2.]]
        # example 3, values []
        # example 4, [[3.]]
        # example 5, values []
        indices=((1, 0), (1, 1), (2, 0), (4, 0)),
        values=(0., 1., 2., 3.),
        dense_shape=(6, 2))
    expected_sequence_length = [0, 2, 1, 0, 1, 0]
    numeric_column = sfc.sequence_numeric_column('aaa')

    _, sequence_length = _get_sequence_dense_tensor(
        numeric_column, {'aaa': sparse_input})

    self.assertAllEqual(
        expected_sequence_length, self.evaluate(sequence_length))

  def test_serialization(self):
    """Tests that column can be serialized."""
    def _custom_fn(input_tensor):
      return input_tensor + 42

    column = sfc.sequence_numeric_column(
        key='my-key', shape=(2,), default_value=3, dtype=dtypes.int32,
        normalizer_fn=_custom_fn)
    configs = serialization.serialize_feature_column(column)
    column = serialization.deserialize_feature_column(
        configs, custom_objects={_custom_fn.__name__: _custom_fn})
    self.assertEqual(column.key, 'my-key')
    self.assertEqual(column.shape, (2,))
    self.assertEqual(column.default_value, 3)
    self.assertEqual(column.normalizer_fn(3), 45)
    with self.assertRaisesRegex(ValueError,
                                'Instance: 0 is not a FeatureColumn'):
      serialization.serialize_feature_column(int())

  def test_parents(self):
    """Tests parents attribute of column."""
    column = sfc.sequence_numeric_column(key='my-key')
    self.assertEqual(column.parents, ['my-key'])


if __name__ == '__main__':
  test.main()
