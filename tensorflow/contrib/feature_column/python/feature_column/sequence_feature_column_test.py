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
import numpy as np

from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session


class SequenceInputLayerTest(test.TestCase):

  def test_embedding_column(self):
    vocabulary_size = 3
    sparse_input_a = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))
    sparse_input_b = sparse_tensor.SparseTensorValue(
        # example 0, ids [1]
        # example 1, ids [2, 0]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(1, 2, 0),
        dense_shape=(2, 2))

    embedding_dimension_a = 2
    embedding_values_a = (
        (1., 2.),  # id 0
        (3., 4.),  # id 1
        (5., 6.)  # id 2
    )
    embedding_dimension_b = 3
    embedding_values_b = (
        (11., 12., 13.),  # id 0
        (14., 15., 16.),  # id 1
        (17., 18., 19.)  # id 2
    )
    def _get_initializer(embedding_dimension, embedding_values):
      def _initializer(shape, dtype, partition_info):
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values
      return _initializer

    expected_input_layer = [
        # example 0, ids_a [2], ids_b [1]
        [[5., 6., 14., 15., 16.], [0., 0., 0., 0., 0.]],
        # example 1, ids_a [0, 1], ids_b [2, 0]
        [[1., 2., 17., 18., 19.], [3., 4., 11., 12., 13.]],
    ]
    expected_sequence_length = [1, 2]

    categorical_column_a = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column_a = fc.embedding_column(
        categorical_column_a, dimension=embedding_dimension_a,
        initializer=_get_initializer(embedding_dimension_a, embedding_values_a))
    categorical_column_b = sfc.sequence_categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_column_b = fc.embedding_column(
        categorical_column_b, dimension=embedding_dimension_b,
        initializer=_get_initializer(embedding_dimension_b, embedding_values_b))

    input_layer, sequence_length = sfc.sequence_input_layer(
        features={
            'aaa': sparse_input_a,
            'bbb': sparse_input_b,
        },
        # Test that columns are reordered alphabetically.
        feature_columns=[embedding_column_b, embedding_column_a])

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('sequence_input_layer/aaa_embedding/embedding_weights:0',
         'sequence_input_layer/bbb_embedding/embedding_weights:0'),
        tuple([v.name for v in global_vars]))
    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(embedding_values_a, global_vars[0].eval(session=sess))
      self.assertAllEqual(embedding_values_b, global_vars[1].eval(session=sess))
      self.assertAllEqual(expected_input_layer, input_layer.eval(session=sess))
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))

  def test_embedding_column_with_non_sequence_categorical(self):
    """Tests that error is raised for non-sequence embedding column."""
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))

    categorical_column_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column_a = fc.embedding_column(
        categorical_column_a, dimension=2)

    with self.assertRaisesRegexp(
        ValueError,
        r'In embedding_column: aaa_embedding\. categorical_column must be of '
        r'type _SequenceCategoricalColumn to use sequence_input_layer\.'):
      _, _ = sfc.sequence_input_layer(
          features={'aaa': sparse_input},
          feature_columns=[embedding_column_a])

  def test_shared_embedding_column(self):
    vocabulary_size = 3
    sparse_input_a = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))
    sparse_input_b = sparse_tensor.SparseTensorValue(
        # example 0, ids [1]
        # example 1, ids [2, 0]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(1, 2, 0),
        dense_shape=(2, 2))

    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 4.),  # id 1
        (5., 6.)  # id 2
    )

    def _get_initializer(embedding_dimension, embedding_values):

      def _initializer(shape, dtype, partition_info):
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values

      return _initializer

    expected_input_layer = [
        # example 0, ids_a [2], ids_b [1]
        [[5., 6., 3., 4.], [0., 0., 0., 0.]],
        # example 1, ids_a [0, 1], ids_b [2, 0]
        [[1., 2., 5., 6.], [3., 4., 1., 2.]],
    ]
    expected_sequence_length = [1, 2]

    categorical_column_a = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = sfc.sequence_categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    # Test that columns are reordered alphabetically.
    shared_embedding_columns = fc.shared_embedding_columns(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension,
        initializer=_get_initializer(embedding_dimension, embedding_values))

    input_layer, sequence_length = sfc.sequence_input_layer(
        features={
            'aaa': sparse_input_a,
            'bbb': sparse_input_b,
        },
        feature_columns=shared_embedding_columns)

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('sequence_input_layer/aaa_bbb_shared_embedding/embedding_weights:0',),
        tuple([v.name for v in global_vars]))
    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(embedding_values, global_vars[0].eval(session=sess))
      self.assertAllEqual(expected_input_layer, input_layer.eval(session=sess))
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))

  def test_shared_embedding_column_with_non_sequence_categorical(self):
    """Tests that error is raised for non-sequence shared embedding column."""
    vocabulary_size = 3
    sparse_input_a = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))
    sparse_input_b = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))

    categorical_column_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    shared_embedding_columns = fc.shared_embedding_columns(
        [categorical_column_a, categorical_column_b], dimension=2)

    with self.assertRaisesRegexp(
        ValueError,
        r'In embedding_column: aaa_shared_embedding\. categorical_column must '
        r'be of type _SequenceCategoricalColumn to use sequence_input_layer\.'):
      _, _ = sfc.sequence_input_layer(
          features={
              'aaa': sparse_input_a,
              'bbb': sparse_input_b
          },
          feature_columns=shared_embedding_columns)

  def test_indicator_column(self):
    vocabulary_size_a = 3
    sparse_input_a = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))
    vocabulary_size_b = 2
    sparse_input_b = sparse_tensor.SparseTensorValue(
        # example 0, ids [1]
        # example 1, ids [1, 0]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(1, 1, 0),
        dense_shape=(2, 2))

    expected_input_layer = [
        # example 0, ids_a [2], ids_b [1]
        [[0., 0., 1., 0., 1.], [0., 0., 0., 0., 0.]],
        # example 1, ids_a [0, 1], ids_b [1, 0]
        [[1., 0., 0., 0., 1.], [0., 1., 0., 1., 0.]],
    ]
    expected_sequence_length = [1, 2]

    categorical_column_a = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size_a)
    indicator_column_a = fc.indicator_column(categorical_column_a)
    categorical_column_b = sfc.sequence_categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size_b)
    indicator_column_b = fc.indicator_column(categorical_column_b)
    input_layer, sequence_length = sfc.sequence_input_layer(
        features={
            'aaa': sparse_input_a,
            'bbb': sparse_input_b,
        },
        # Test that columns are reordered alphabetically.
        feature_columns=[indicator_column_b, indicator_column_a])

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(expected_input_layer, input_layer.eval(session=sess))
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))

  def test_indicator_column_with_non_sequence_categorical(self):
    """Tests that error is raised for non-sequence categorical column."""
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))

    categorical_column_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column_a = fc.indicator_column(categorical_column_a)

    with self.assertRaisesRegexp(
        ValueError,
        r'In indicator_column: aaa_indicator\. categorical_column must be of '
        r'type _SequenceCategoricalColumn to use sequence_input_layer\.'):
      _, _ = sfc.sequence_input_layer(
          features={'aaa': sparse_input},
          feature_columns=[indicator_column_a])

  def test_numeric_column(self):
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values [[0.], [1]]
        # example 1, [[10.]]
        indices=((0, 0), (0, 1), (1, 0)),
        values=(0., 1., 10.),
        dense_shape=(2, 2))
    expected_input_layer = [
        [[0.], [1.]],
        [[10.], [0.]],
    ]
    expected_sequence_length = [2, 1]
    numeric_column = sfc.sequence_numeric_column('aaa')

    input_layer, sequence_length = sfc.sequence_input_layer(
        features={'aaa': sparse_input},
        feature_columns=[numeric_column])

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(expected_input_layer, input_layer.eval(session=sess))
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))

  def test_numeric_column_multi_dim(self):
    """Tests sequence_input_layer for multi-dimensional numeric_column."""
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values [[[0., 1.],  [2., 3.]], [[4., 5.],  [6., 7.]]]
        # example 1, [[[10., 11.],  [12., 13.]]]
        indices=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                 (1, 0), (1, 1), (1, 2), (1, 3)),
        values=(0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
        dense_shape=(2, 8))
    # The output of numeric_column._get_dense_tensor should be flattened.
    expected_input_layer = [
        [[0., 1., 2., 3.], [4., 5., 6., 7.]],
        [[10., 11., 12., 13.], [0., 0., 0., 0.]],
    ]
    expected_sequence_length = [2, 1]
    numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

    input_layer, sequence_length = sfc.sequence_input_layer(
        features={'aaa': sparse_input},
        feature_columns=[numeric_column])

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(expected_input_layer, input_layer.eval(session=sess))
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))

  def test_sequence_length_not_equal(self):
    """Tests that an error is raised when sequence lengths are not equal."""
    # Input a with sequence_length = [2, 1]
    sparse_input_a = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (0, 1), (1, 0)),
        values=(0., 1., 10.),
        dense_shape=(2, 2))
    # Input b with sequence_length = [1, 1]
    sparse_input_b = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0)),
        values=(1., 10.),
        dense_shape=(2, 2))
    numeric_column_a = sfc.sequence_numeric_column('aaa')
    numeric_column_b = sfc.sequence_numeric_column('bbb')

    _, sequence_length = sfc.sequence_input_layer(
        features={
            'aaa': sparse_input_a,
            'bbb': sparse_input_b,
        },
        feature_columns=[numeric_column_a, numeric_column_b])

    with monitored_session.MonitoredSession() as sess:
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[Condition x == y did not hold element-wise:\] '
          r'\[x \(sequence_input_layer/aaa/sequence_length:0\) = \] \[2 1\] '
          r'\[y \(sequence_input_layer/bbb/sequence_length:0\) = \] \[1 1\]'):
        sess.run(sequence_length)


class InputLayerTest(test.TestCase):
  """Tests input_layer with sequence feature columns."""

  def test_embedding_column(self):
    """Tests that error is raised for sequence embedding column."""
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))

    categorical_column_a = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column_a = fc.embedding_column(
        categorical_column_a, dimension=2)

    with self.assertRaisesRegexp(
        ValueError,
        r'In embedding_column: aaa_embedding\. categorical_column must not be '
        r'of type _SequenceCategoricalColumn\.'):
      _ = fc.input_layer(
          features={'aaa': sparse_input},
          feature_columns=[embedding_column_a])

  def test_indicator_column(self):
    """Tests that error is raised for sequence indicator column."""
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))

    categorical_column_a = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column_a = fc.indicator_column(categorical_column_a)

    with self.assertRaisesRegexp(
        ValueError,
        r'In indicator_column: aaa_indicator\. categorical_column must not be '
        r'of type _SequenceCategoricalColumn\.'):
      _ = fc.input_layer(
          features={'aaa': sparse_input},
          feature_columns=[indicator_column_a])


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


class SequenceCategoricalColumnWithIdentityTest(test.TestCase):

  def test_get_sparse_tensors(self):
    column = sfc.sequence_categorical_column_with_identity(
        'aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(1, 2, 0),
        dense_shape=(2, 2))
    expected_sparse_ids = sparse_tensor.SparseTensorValue(
        indices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        values=np.array((1, 2, 0), dtype=np.int64),
        dense_shape=(2, 2, 1))

    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

    self.assertIsNone(id_weight_pair.weight_tensor)
    with monitored_session.MonitoredSession() as sess:
      _assert_sparse_tensor_value(
          self,
          expected_sparse_ids,
          id_weight_pair.id_tensor.eval(session=sess))

  def test_get_sparse_tensors_inputs3d(self):
    """Tests _get_sparse_tensors when the input is already 3D Tensor."""
    column = sfc.sequence_categorical_column_with_identity(
        'aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        values=(1, 2, 0),
        dense_shape=(2, 2, 1))

    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r'Column aaa expected ID tensor of rank 2\.\s*'
        r'id_tensor shape:\s*\[2 2 1\]'):
      id_weight_pair = column._get_sparse_tensors(
          _LazyBuilder({'aaa': inputs}))
      with monitored_session.MonitoredSession() as sess:
        id_weight_pair.id_tensor.eval(session=sess)


class SequenceCategoricalColumnWithHashBucketTest(test.TestCase):

  def test_get_sparse_tensors(self):
    column = sfc.sequence_categorical_column_with_hash_bucket(
        'aaa', hash_bucket_size=10)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))

    expected_sparse_ids = sparse_tensor.SparseTensorValue(
        indices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        # Ignored to avoid hash dependence in test.
        values=np.array((0, 0, 0), dtype=np.int64),
        dense_shape=(2, 2, 1))

    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

    self.assertIsNone(id_weight_pair.weight_tensor)
    with monitored_session.MonitoredSession() as sess:
      _assert_sparse_tensor_indices_shape(
          self,
          expected_sparse_ids,
          id_weight_pair.id_tensor.eval(session=sess))


class SequenceCategoricalColumnWithVocabularyFileTest(test.TestCase):

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

  def test_get_sparse_tensors(self):
    column = sfc.sequence_categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    expected_sparse_ids = sparse_tensor.SparseTensorValue(
        indices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        values=np.array((2, -1, 0), dtype=np.int64),
        dense_shape=(2, 2, 1))

    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

    self.assertIsNone(id_weight_pair.weight_tensor)
    with monitored_session.MonitoredSession() as sess:
      _assert_sparse_tensor_value(
          self,
          expected_sparse_ids,
          id_weight_pair.id_tensor.eval(session=sess))


class SequenceCategoricalColumnWithVocabularyListTest(test.TestCase):

  def test_get_sparse_tensors(self):
    column = sfc.sequence_categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    expected_sparse_ids = sparse_tensor.SparseTensorValue(
        indices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        values=np.array((2, -1, 0), dtype=np.int64),
        dense_shape=(2, 2, 1))

    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

    self.assertIsNone(id_weight_pair.weight_tensor)
    with monitored_session.MonitoredSession() as sess:
      _assert_sparse_tensor_value(
          self,
          expected_sparse_ids,
          id_weight_pair.id_tensor.eval(session=sess))


class SequenceEmbeddingColumnTest(test.TestCase):

  def test_get_sequence_dense_tensor(self):
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 2))

    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )
    def _initializer(shape, dtype, partition_info):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    expected_lookups = [
        # example 0, ids [2]
        [[7., 11.], [0., 0.]],
        # example 1, ids [0, 1]
        [[1., 2.], [3., 5.]],
        # example 2, ids []
        [[0., 0.], [0., 0.]],
        # example 3, ids [1]
        [[3., 5.], [0., 0.]],
    ]

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        initializer=_initializer)

    embedding_lookup, _ = embedding_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('embedding_weights:0',), tuple([v.name for v in global_vars]))
    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(embedding_values, global_vars[0].eval(session=sess))
      self.assertAllEqual(expected_lookups, embedding_lookup.eval(session=sess))

  def test_sequence_length(self):
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))
    expected_sequence_length = [1, 2]

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=2)

    _, sequence_length = embedding_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      sequence_length = sess.run(sequence_length)
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

    _, sequence_length = embedding_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))


class SequenceSharedEmbeddingColumnTest(test.TestCase):

  def test_get_sequence_dense_tensor(self):
    vocabulary_size = 3
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )

    def _initializer(shape, dtype, partition_info):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

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
    shared_embedding_columns = fc.shared_embedding_columns(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        initializer=_initializer)

    embedding_lookup_a = shared_embedding_columns[0]._get_sequence_dense_tensor(
        _LazyBuilder({
            'aaa': sparse_input_a
        }))[0]
    embedding_lookup_b = shared_embedding_columns[1]._get_sequence_dense_tensor(
        _LazyBuilder({
            'bbb': sparse_input_b
        }))[0]

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('embedding_weights:0',),
                          tuple([v.name for v in global_vars]))
    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(embedding_values, global_vars[0].eval(session=sess))
      self.assertAllEqual(
          expected_lookups_a, embedding_lookup_a.eval(session=sess))
      self.assertAllEqual(
          expected_lookups_b, embedding_lookup_b.eval(session=sess))

  def test_sequence_length(self):
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
    shared_embedding_columns = fc.shared_embedding_columns(
        [categorical_column_a, categorical_column_b], dimension=2)

    sequence_length_a = shared_embedding_columns[0]._get_sequence_dense_tensor(
        _LazyBuilder({
            'aaa': sparse_input_a
        }))[1]
    sequence_length_b = shared_embedding_columns[1]._get_sequence_dense_tensor(
        _LazyBuilder({
            'bbb': sparse_input_b
        }))[1]

    with monitored_session.MonitoredSession() as sess:
      sequence_length_a = sess.run(sequence_length_a)
      self.assertAllEqual(expected_sequence_length_a, sequence_length_a)
      self.assertEqual(np.int64, sequence_length_a.dtype)
      sequence_length_b = sess.run(sequence_length_b)
      self.assertAllEqual(expected_sequence_length_b, sequence_length_b)
      self.assertEqual(np.int64, sequence_length_b.dtype)

  def test_sequence_length_with_empty_rows(self):
    """Tests _sequence_length when some examples do not have ids."""
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

    shared_embedding_columns = fc.shared_embedding_columns(
        [categorical_column_a, categorical_column_b], dimension=2)

    sequence_length_a = shared_embedding_columns[0]._get_sequence_dense_tensor(
        _LazyBuilder({
            'aaa': sparse_input_a
        }))[1]
    sequence_length_b = shared_embedding_columns[1]._get_sequence_dense_tensor(
        _LazyBuilder({
            'bbb': sparse_input_b
        }))[1]

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_sequence_length_a, sequence_length_a.eval(session=sess))
      self.assertAllEqual(
          expected_sequence_length_b, sequence_length_b.eval(session=sess))


class SequenceIndicatorColumnTest(test.TestCase):

  def test_get_sequence_dense_tensor(self):
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 2))

    expected_lookups = [
        # example 0, ids [2]
        [[0., 0., 1.], [0., 0., 0.]],
        # example 1, ids [0, 1]
        [[1., 0., 0.], [0., 1., 0.]],
        # example 2, ids []
        [[0., 0., 0.], [0., 0., 0.]],
        # example 3, ids [1]
        [[0., 1., 0.], [0., 0., 0.]],
    ]

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column = fc.indicator_column(categorical_column)

    indicator_tensor, _ = indicator_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(expected_lookups, indicator_tensor.eval(session=sess))

  def test_sequence_length(self):
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))
    expected_sequence_length = [1, 2]

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    indicator_column = fc.indicator_column(categorical_column)

    _, sequence_length = indicator_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      sequence_length = sess.run(sequence_length)
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

    _, sequence_length = indicator_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))


class SequenceNumericColumnTest(test.TestCase):

  def test_defaults(self):
    a = sfc.sequence_numeric_column('aaa')
    self.assertEqual('aaa', a.key)
    self.assertEqual('aaa', a.name)
    self.assertEqual('aaa', a._var_scope_name)
    self.assertEqual((1,), a.shape)
    self.assertEqual(0., a.default_value)
    self.assertEqual(dtypes.float32, a.dtype)
    self.assertIsNone(a.normalizer_fn)

  def test_shape_saved_as_tuple(self):
    a = sfc.sequence_numeric_column('aaa', shape=[1, 2])
    self.assertEqual((1, 2), a.shape)

  def test_shape_must_be_positive_integer(self):
    with self.assertRaisesRegexp(TypeError, 'shape dimensions must be integer'):
      sfc.sequence_numeric_column('aaa', shape=[1.0])

    with self.assertRaisesRegexp(
        ValueError, 'shape dimensions must be greater than 0'):
      sfc.sequence_numeric_column('aaa', shape=[0])

  def test_dtype_is_convertible_to_float(self):
    with self.assertRaisesRegexp(
        ValueError, 'dtype must be convertible to float'):
      sfc.sequence_numeric_column('aaa', dtype=dtypes.string)

  def test_normalizer_fn_must_be_callable(self):
    with self.assertRaisesRegexp(TypeError, 'must be a callable'):
      sfc.sequence_numeric_column('aaa', normalizer_fn='NotACallable')

  def test_get_sequence_dense_tensor(self):
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values [[0.], [1]]
        # example 1, [[10.]]
        indices=((0, 0), (0, 1), (1, 0)),
        values=(0., 1., 10.),
        dense_shape=(2, 2))
    expected_dense_tensor = [
        [[0.], [1.]],
        [[10.], [0.]],
    ]
    numeric_column = sfc.sequence_numeric_column('aaa')

    dense_tensor, _ = numeric_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_dense_tensor, dense_tensor.eval(session=sess))

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

    dense_tensor, _ = numeric_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_dense_tensor, dense_tensor.eval(session=sess))

  def test_get_sequence_dense_tensor_with_shape(self):
    """Tests get_sequence_dense_tensor with shape !=(1,)."""
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values [[0., 1., 2.], [3., 4., 5.]]
        # example 1, [[10., 11., 12.]]
        indices=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                 (1, 0), (1, 1), (1, 2)),
        values=(0., 1., 2., 3., 4., 5., 10., 11., 12.),
        dense_shape=(2, 6))
    expected_dense_tensor = [
        [[0., 1., 2.], [3., 4., 5.]],
        [[10., 11., 12.], [0., 0., 0.]],
    ]
    numeric_column = sfc.sequence_numeric_column('aaa', shape=(3,))

    dense_tensor, _ = numeric_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_dense_tensor, dense_tensor.eval(session=sess))

  def test_get_dense_tensor_multi_dim(self):
    """Tests get_sequence_dense_tensor for multi-dim numeric_column."""
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values [[[0., 1.],  [2., 3.]], [[4., 5.],  [6., 7.]]]
        # example 1, [[[10., 11.],  [12., 13.]]]
        indices=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                 (1, 0), (1, 1), (1, 2), (1, 3)),
        values=(0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
        dense_shape=(2, 8))
    expected_dense_tensor = [
        [[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        [[[10., 11.], [12., 13.]], [[0., 0.], [0., 0.]]],
    ]
    numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

    dense_tensor, _ = numeric_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_dense_tensor, dense_tensor.eval(session=sess))

  def test_sequence_length(self):
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values [[0., 1., 2.], [3., 4., 5.]]
        # example 1, [[10., 11., 12.]]
        indices=((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                 (1, 0), (1, 1), (1, 2)),
        values=(0., 1., 2., 3., 4., 5., 10., 11., 12.),
        dense_shape=(2, 6))
    expected_sequence_length = [2, 1]
    numeric_column = sfc.sequence_numeric_column('aaa', shape=(3,))

    _, sequence_length = numeric_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      sequence_length = sess.run(sequence_length)
      self.assertAllEqual(expected_sequence_length, sequence_length)
      self.assertEqual(np.int64, sequence_length.dtype)

  def test_sequence_length_with_shape(self):
    """Tests _sequence_length with shape !=(1,)."""
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, values [[0.], [1]]
        # example 1, [[10.]]
        indices=((0, 0), (0, 1), (1, 0)),
        values=(0., 1., 10.),
        dense_shape=(2, 2))
    expected_sequence_length = [2, 1]
    numeric_column = sfc.sequence_numeric_column('aaa')

    _, sequence_length = numeric_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))

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

    _, sequence_length = numeric_column._get_sequence_dense_tensor(
        _LazyBuilder({'aaa': sparse_input}))

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))


if __name__ == '__main__':
  test.main()
