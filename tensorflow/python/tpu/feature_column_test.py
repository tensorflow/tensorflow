# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for python.tpu.feature_column."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.tpu import feature_column as tpu_fc


def _initialized_session():
  sess = session.Session()
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


class EmbeddingColumnTest(test.TestCase):

  def test_defaults(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = tpu_fc.embedding_column(
        categorical_column, dimension=embedding_dimension)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('mean', embedding_column.combiner)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual('aaa_embedding', embedding_column._var_scope_name)
    self.assertEqual((embedding_dimension,), embedding_column._variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column._parse_example_spec)

  def test_blacklisted_column(self):
    # HashedCategoricalColumn is blacklisted and so will raise an exception.
    categorical_column = fc_lib.categorical_column_with_hash_bucket(
        key='aaa', hash_bucket_size=3)
    embedding_dimension = 2
    with self.assertRaises(TypeError):
      tpu_fc.embedding_column(categorical_column, dimension=embedding_dimension)

  def test_custom_column(self):
    # This column is not in any whitelist but should succeed because
    # it inherits from V2 CategoricalColumn.
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=10)
    embedding_dimension = 2
    embedding_column = tpu_fc.embedding_column(
        categorical_column, dimension=embedding_dimension)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('mean', embedding_column.combiner)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual('aaa_embedding', embedding_column._var_scope_name)
    self.assertEqual((embedding_dimension,), embedding_column._variable_shape)
    self.assertEqual({'aaa': parsing_ops.VarLenFeature(dtypes.int64)},
                     embedding_column._parse_example_spec)

  def test_all_constructor_args(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = tpu_fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer')
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('my_combiner', embedding_column.combiner)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual('aaa_embedding', embedding_column._var_scope_name)
    self.assertEqual((embedding_dimension,), embedding_column._variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column._parse_example_spec)

  @test_util.deprecated_graph_mode_only
  def test_get_dense_tensor(self):
    # Inputs.
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 4), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 5))

    # Embedding variable.
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

    # Expected lookup result, using combiner='mean'.
    expected_lookups = (
        # example 0, ids [2], embedding = [7, 11]
        (7., 11.),
        # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
        (2., 3.5),
        # example 2, ids [], embedding = [0, 0]
        (0., 0.),
        # example 3, ids [1], embedding = [3, 5]
        (3., 5.),
    )

    # Build columns.
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = tpu_fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column._get_dense_tensor(
        fc._LazyBuilder({
            'aaa': sparse_input
        }))

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('embedding_weights:0',),
                          tuple([v.name for v in global_vars]))
    with _initialized_session():
      self.assertAllEqual(embedding_values, global_vars[0])
      self.assertAllEqual(expected_lookups, embedding_lookup)


class SharedEmbeddingColumnTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def test_defaults(self):
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=3)
    embedding_dimension = 2
    embedding_column_b, embedding_column_a = tpu_fc.shared_embedding_columns(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension)
    self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
    self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column_a.dimension)
    self.assertEqual(embedding_dimension, embedding_column_b.dimension)
    self.assertEqual('mean', embedding_column_a.combiner)
    self.assertEqual('mean', embedding_column_b.combiner)
    self.assertIsNotNone(embedding_column_a.initializer)
    self.assertIsNotNone(embedding_column_b.initializer)
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_a.shared_embedding_collection_name)
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_b.shared_embedding_collection_name)
    self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
    self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_a._var_scope_name)
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_b._var_scope_name)
    self.assertEqual((embedding_dimension,), embedding_column_a._variable_shape)
    self.assertEqual((embedding_dimension,), embedding_column_b._variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column_a._parse_example_spec)
    self.assertEqual({
        'bbb': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column_b._parse_example_spec)

  @test_util.deprecated_graph_mode_only
  def test_all_constructor_args(self):
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=3)
    embedding_dimension = 2
    embedding_column_a, embedding_column_b = tpu_fc.shared_embedding_columns(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer',
        shared_embedding_collection_name='var_scope_name')
    self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
    self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column_a.dimension)
    self.assertEqual(embedding_dimension, embedding_column_b.dimension)
    self.assertEqual('my_combiner', embedding_column_a.combiner)
    self.assertEqual('my_combiner', embedding_column_b.combiner)
    self.assertEqual('my_initializer', embedding_column_a.initializer())
    self.assertEqual('my_initializer', embedding_column_b.initializer())
    self.assertEqual('var_scope_name',
                     embedding_column_a.shared_embedding_collection_name)
    self.assertEqual('var_scope_name',
                     embedding_column_b.shared_embedding_collection_name)
    self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
    self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
    self.assertEqual('var_scope_name', embedding_column_a._var_scope_name)
    self.assertEqual('var_scope_name', embedding_column_b._var_scope_name)
    self.assertEqual((embedding_dimension,), embedding_column_a._variable_shape)
    self.assertEqual((embedding_dimension,), embedding_column_b._variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column_a._parse_example_spec)
    self.assertEqual({
        'bbb': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column_b._parse_example_spec)

  @test_util.deprecated_graph_mode_only
  def test_get_dense_tensor(self):
    # Inputs.
    vocabulary_size = 3
    # -1 values are ignored.
    input_a = np.array([
        [2, -1, -1],  # example 0, ids [2]
        [0, 1, -1]
    ])  # example 1, ids [0, 1]
    input_b = np.array([
        [0, -1, -1],  # example 0, ids [0]
        [-1, -1, -1]
    ])  # example 1, ids []
    input_features = {'aaa': input_a, 'bbb': input_b}

    # Embedding variable.
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

    # Expected lookup result, using combiner='mean'.
    expected_lookups_a = (
        # example 0:
        (7., 11.),  # ids [2], embedding = [7, 11]
        # example 1:
        (2., 3.5),  # ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
    )
    expected_lookups_b = (
        # example 0:
        (1., 2.),  # ids [0], embedding = [1, 2]
        # example 1:
        (0., 0.),  # ids [], embedding = [0, 0]
    )

    # Build columns.
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_column_a, embedding_column_b = tpu_fc.shared_embedding_columns(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    embedding_lookup_a = embedding_column_a._get_dense_tensor(
        fc._LazyBuilder(input_features))
    embedding_lookup_b = embedding_column_b._get_dense_tensor(
        fc._LazyBuilder(input_features))

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('embedding_weights:0',),
                          tuple([v.name for v in global_vars]))
    embedding_var = global_vars[0]
    with _initialized_session():
      self.assertAllEqual(embedding_values, embedding_var)
      self.assertAllEqual(expected_lookups_a, embedding_lookup_a)
      self.assertAllEqual(expected_lookups_b, embedding_lookup_b)


if __name__ == '__main__':
  test.main()
