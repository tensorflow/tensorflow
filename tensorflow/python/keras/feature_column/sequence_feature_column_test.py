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


from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras.feature_column import sequence_feature_column as ksfc
from tensorflow.python.keras.saving import model_config
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


def _initialized_session(config=None):
  sess = session.Session(config=config)
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


class SequenceFeaturesTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'sparse_input_args_a': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (2, 0, 1),
           'dense_shape': (2, 2)},
       'sparse_input_args_b': {
           # example 0, ids [1]
           # example 1, ids [2, 0]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (1, 2, 0),
           'dense_shape': (2, 2)},
       'expected_input_layer': [
           # example 0, ids_a [2], ids_b [1]
           [[5., 6., 14., 15., 16.], [0., 0., 0., 0., 0.]],
           # example 1, ids_a [0, 1], ids_b [2, 0]
           [[1., 2., 17., 18., 19.], [3., 4., 11., 12., 13.]],],
       'expected_sequence_length': [1, 2]},
      {'testcase_name': '3D',
       'sparse_input_args_a': {
           # feature 0, ids [[2], [0, 1]]
           # feature 1, ids [[0, 0], [1]]
           'indices': (
               (0, 0, 0), (0, 1, 0), (0, 1, 1),
               (1, 0, 0), (1, 0, 1), (1, 1, 0)),
           'values': (2, 0, 1, 0, 0, 1),
           'dense_shape': (2, 2, 2)},
       'sparse_input_args_b': {
           # feature 0, ids [[1, 1], [1]]
           # feature 1, ids [[2], [0]]
           'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
           'values': (1, 1, 1, 2, 0),
           'dense_shape': (2, 2, 2)},
       'expected_input_layer': [
           # feature 0, [a: 2, -, b: 1, 1], [a: 0, 1, b: 1, -]
           [[5., 6., 14., 15., 16.], [2., 3., 14., 15., 16.]],
           # feature 1, [a: 0, 0, b: 2, -], [a: 1, -, b: 0, -]
           [[1., 2., 17., 18., 19.], [3., 4., 11., 12., 13.]]],
       'expected_sequence_length': [2, 2]},
      )
  @test_util.run_in_graph_and_eager_modes
  def test_embedding_column(
      self, sparse_input_args_a, sparse_input_args_b, expected_input_layer,
      expected_sequence_length):

    sparse_input_a = sparse_tensor.SparseTensorValue(**sparse_input_args_a)
    sparse_input_b = sparse_tensor.SparseTensorValue(**sparse_input_args_b)
    vocabulary_size = 3
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

      def _initializer(shape, dtype, partition_info=None):
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values
      return _initializer

    categorical_column_a = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column_a = fc.embedding_column(
        categorical_column_a,
        dimension=embedding_dimension_a,
        initializer=_get_initializer(embedding_dimension_a, embedding_values_a))
    categorical_column_b = sfc.sequence_categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_column_b = fc.embedding_column(
        categorical_column_b,
        dimension=embedding_dimension_b,
        initializer=_get_initializer(embedding_dimension_b, embedding_values_b))

    # Test that columns are reordered alphabetically.
    sequence_input_layer = ksfc.SequenceFeatures(
        [embedding_column_b, embedding_column_a])
    input_layer, sequence_length = sequence_input_layer({
        'aaa': sparse_input_a, 'bbb': sparse_input_b,})

    self.evaluate(variables_lib.global_variables_initializer())
    weights = sequence_input_layer.weights
    self.assertCountEqual(
        ('sequence_features/aaa_embedding/embedding_weights:0',
         'sequence_features/bbb_embedding/embedding_weights:0'),
        tuple([v.name for v in weights]))
    self.assertAllEqual(embedding_values_a, self.evaluate(weights[0]))
    self.assertAllEqual(embedding_values_b, self.evaluate(weights[1]))
    self.assertAllEqual(expected_input_layer, self.evaluate(input_layer))
    self.assertAllEqual(
        expected_sequence_length, self.evaluate(sequence_length))

  @test_util.run_in_graph_and_eager_modes
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
    sequence_input_layer = ksfc.SequenceFeatures([embedding_column_a])
    with self.assertRaisesRegex(
        ValueError,
        r'In embedding_column: aaa_embedding\. categorical_column must be of '
        r'type SequenceCategoricalColumn to use SequenceFeatures\.'):
      _, _ = sequence_input_layer({'aaa': sparse_input})

  @test_util.run_in_graph_and_eager_modes
  def test_shared_embedding_column(self):
    with ops.Graph().as_default():
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

        def _initializer(shape, dtype, partition_info=None):
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
      shared_embedding_columns = fc.shared_embedding_columns_v2(
          [categorical_column_b, categorical_column_a],
          dimension=embedding_dimension,
          initializer=_get_initializer(embedding_dimension, embedding_values))

      sequence_input_layer = ksfc.SequenceFeatures(shared_embedding_columns)
      input_layer, sequence_length = sequence_input_layer({
          'aaa': sparse_input_a, 'bbb': sparse_input_b})

      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(
          ('aaa_bbb_shared_embedding:0',),
          tuple([v.name for v in global_vars]))
      with _initialized_session() as sess:
        self.assertAllEqual(embedding_values,
                            global_vars[0].eval(session=sess))
        self.assertAllEqual(expected_input_layer,
                            input_layer.eval(session=sess))
        self.assertAllEqual(
            expected_sequence_length, sequence_length.eval(session=sess))

  @test_util.run_deprecated_v1
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
    shared_embedding_columns = fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b], dimension=2)

    sequence_input_layer = ksfc.SequenceFeatures(shared_embedding_columns)
    with self.assertRaisesRegex(
        ValueError,
        r'In embedding_column: aaa_shared_embedding\. categorical_column must '
        r'be of type SequenceCategoricalColumn to use SequenceFeatures\.'):
      _, _ = sequence_input_layer({'aaa': sparse_input_a,
                                   'bbb': sparse_input_b})

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'sparse_input_args_a': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (2, 0, 1),
           'dense_shape': (2, 2)},
       'sparse_input_args_b': {
           # example 0, ids [1]
           # example 1, ids [1, 0]
           'indices': ((0, 0), (1, 0), (1, 1)),
           'values': (1, 1, 0),
           'dense_shape': (2, 2)},
       'expected_input_layer': [
           # example 0, ids_a [2], ids_b [1]
           [[0., 0., 1., 0., 1.], [0., 0., 0., 0., 0.]],
           # example 1, ids_a [0, 1], ids_b [1, 0]
           [[1., 0., 0., 0., 1.], [0., 1., 0., 1., 0.]]],
       'expected_sequence_length': [1, 2]},
      {'testcase_name': '3D',
       'sparse_input_args_a': {
           # feature 0, ids [[2], [0, 1]]
           # feature 1, ids [[0, 0], [1]]
           'indices': (
               (0, 0, 0), (0, 1, 0), (0, 1, 1),
               (1, 0, 0), (1, 0, 1), (1, 1, 0)),
           'values': (2, 0, 1, 0, 0, 1),
           'dense_shape': (2, 2, 2)},
       'sparse_input_args_b': {
           # feature 0, ids [[1, 1], [1]]
           # feature 1, ids [[1], [0]]
           'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
           'values': (1, 1, 1, 1, 0),
           'dense_shape': (2, 2, 2)},
       'expected_input_layer': [
           # feature 0, [a: 2, -, b: 1, 1], [a: 0, 1, b: 1, -]
           [[0., 0., 1., 0., 2.], [1., 1., 0., 0., 1.]],
           # feature 1, [a: 0, 0, b: 1, -], [a: 1, -, b: 0, -]
           [[2., 0., 0., 0., 1.], [0., 1., 0., 1., 0.]]],
       'expected_sequence_length': [2, 2]},
      )
  @test_util.run_in_graph_and_eager_modes
  def test_indicator_column(
      self, sparse_input_args_a, sparse_input_args_b, expected_input_layer,
      expected_sequence_length):
    sparse_input_a = sparse_tensor.SparseTensorValue(**sparse_input_args_a)
    sparse_input_b = sparse_tensor.SparseTensorValue(**sparse_input_args_b)

    vocabulary_size_a = 3
    vocabulary_size_b = 2

    categorical_column_a = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size_a)
    indicator_column_a = fc.indicator_column(categorical_column_a)
    categorical_column_b = sfc.sequence_categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size_b)
    indicator_column_b = fc.indicator_column(categorical_column_b)
    # Test that columns are reordered alphabetically.
    sequence_input_layer = ksfc.SequenceFeatures(
        [indicator_column_b, indicator_column_a])
    input_layer, sequence_length = sequence_input_layer({
        'aaa': sparse_input_a, 'bbb': sparse_input_b})

    self.assertAllEqual(expected_input_layer, self.evaluate(input_layer))
    self.assertAllEqual(
        expected_sequence_length, self.evaluate(sequence_length))

  @test_util.run_in_graph_and_eager_modes
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

    sequence_input_layer = ksfc.SequenceFeatures([indicator_column_a])
    with self.assertRaisesRegex(
        ValueError,
        r'In indicator_column: aaa_indicator\. categorical_column must be of '
        r'type SequenceCategoricalColumn to use SequenceFeatures\.'):
      _, _ = sequence_input_layer({'aaa': sparse_input})

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'sparse_input_args': {
           # example 0, values [0., 1]
           # example 1, [10.]
           'indices': ((0, 0), (0, 1), (1, 0)),
           'values': (0., 1., 10.),
           'dense_shape': (2, 2)},
       'expected_input_layer': [
           [[0.], [1.]],
           [[10.], [0.]]],
       'expected_sequence_length': [2, 1]},
      {'testcase_name': '3D',
       'sparse_input_args': {
           # feature 0, ids [[20, 3], [5]]
           # feature 1, ids [[3], [8]]
           'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
           'values': (20., 3., 5., 3., 8.),
           'dense_shape': (2, 2, 2)},
       'expected_input_layer': [
           [[20.], [3.], [5.], [0.]],
           [[3.], [0.], [8.], [0.]]],
       'expected_sequence_length': [2, 2]},
      )
  @test_util.run_in_graph_and_eager_modes
  def test_numeric_column(
      self, sparse_input_args, expected_input_layer, expected_sequence_length):
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)

    numeric_column = sfc.sequence_numeric_column('aaa')

    sequence_input_layer = ksfc.SequenceFeatures([numeric_column])
    input_layer, sequence_length = sequence_input_layer({'aaa': sparse_input})

    self.assertAllEqual(expected_input_layer, self.evaluate(input_layer))
    self.assertAllEqual(
        expected_sequence_length, self.evaluate(sequence_length))

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'sparse_input_args': {
           # example 0, values [0., 1.,  2., 3., 4., 5., 6., 7.]
           # example 1, [10., 11., 12., 13.]
           'indices': ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                       (0, 7), (1, 0), (1, 1), (1, 2), (1, 3)),
           'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
           'dense_shape': (2, 8)},
       'expected_input_layer': [
           # The output of numeric_column._get_dense_tensor should be flattened.
           [[0., 1., 2., 3.], [4., 5., 6., 7.]],
           [[10., 11., 12., 13.], [0., 0., 0., 0.]]],
       'expected_sequence_length': [2, 1]},
      {'testcase_name': '3D',
       'sparse_input_args': {
           # example 0, values [[0., 1., 2., 3.]], [[4., 5., 6., 7.]]
           # example 1, [[10., 11., 12., 13.], []]
           'indices': ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
                       (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
                       (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3)),
           'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
           'dense_shape': (2, 2, 4)},
       'expected_input_layer': [
           # The output of numeric_column._get_dense_tensor should be flattened.
           [[0., 1., 2., 3.], [4., 5., 6., 7.]],
           [[10., 11., 12., 13.], [0., 0., 0., 0.]]],
       'expected_sequence_length': [2, 1]},
      )
  @test_util.run_in_graph_and_eager_modes
  def test_numeric_column_multi_dim(
      self, sparse_input_args, expected_input_layer, expected_sequence_length):
    """Tests SequenceFeatures for multi-dimensional numeric_column."""
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)

    numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

    sequence_input_layer = ksfc.SequenceFeatures([numeric_column])
    input_layer, sequence_length = sequence_input_layer({'aaa': sparse_input})

    self.assertAllEqual(expected_input_layer, self.evaluate(input_layer))
    self.assertAllEqual(
        expected_sequence_length, self.evaluate(sequence_length))

  @test_util.run_in_graph_and_eager_modes
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

    sequence_input_layer = ksfc.SequenceFeatures(
        [numeric_column_a, numeric_column_b])

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'Condition x == y did not hold.*'):
      _, sequence_length = sequence_input_layer({
          'aaa': sparse_input_a,
          'bbb': sparse_input_b
      })
      self.evaluate(sequence_length)

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'sparse_input_args': {
           # example 0, values [[[0., 1.],  [2., 3.]], [[4., 5.],  [6., 7.]]]
           # example 1, [[[10., 11.],  [12., 13.]]]
           'indices': ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                       (0, 7), (1, 0), (1, 1), (1, 2), (1, 3)),
           'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
           'dense_shape': (2, 8)},
       'expected_shape': [2, 2, 4]},
      {'testcase_name': '3D',
       'sparse_input_args': {
           # example 0, values [[0., 1., 2., 3.]], [[4., 5., 6., 7.]]
           # example 1, [[10., 11., 12., 13.], []]
           'indices': ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
                       (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
                       (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3)),
           'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
           'dense_shape': (2, 2, 4)},
       'expected_shape': [2, 2, 4]},
      )
  @test_util.run_in_graph_and_eager_modes
  def test_static_shape_from_tensors_numeric(
      self, sparse_input_args, expected_shape):
    """Tests that we return a known static shape when we have one."""
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)
    numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

    sequence_input_layer = ksfc.SequenceFeatures([numeric_column])
    input_layer, _ = sequence_input_layer({'aaa': sparse_input})
    shape = input_layer.get_shape()
    self.assertEqual(shape, expected_shape)

  @parameterized.named_parameters(
      {'testcase_name': '2D',
       'sparse_input_args': {
           # example 0, ids [2]
           # example 1, ids [0, 1]
           # example 2, ids []
           # example 3, ids [1]
           'indices': ((0, 0), (1, 0), (1, 1), (3, 0)),
           'values': (2, 0, 1, 1),
           'dense_shape': (4, 2)},
       'expected_shape': [4, 2, 3]},
      {'testcase_name': '3D',
       'sparse_input_args': {
           # example 0, ids [[2]]
           # example 1, ids [[0, 1], [2]]
           # example 2, ids []
           # example 3, ids [[1], [0, 2]]
           'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0),
                       (3, 0, 0), (3, 1, 0), (3, 1, 1)),
           'values': (2, 0, 1, 2, 1, 0, 2),
           'dense_shape': (4, 2, 2)},
       'expected_shape': [4, 2, 3]}
      )
  @test_util.run_in_graph_and_eager_modes
  def test_static_shape_from_tensors_indicator(
      self, sparse_input_args, expected_shape):
    """Tests that we return a known static shape when we have one."""
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)
    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=3)
    indicator_column = fc.indicator_column(categorical_column)

    sequence_input_layer = ksfc.SequenceFeatures([indicator_column])
    input_layer, _ = sequence_input_layer({'aaa': sparse_input})
    shape = input_layer.get_shape()
    self.assertEqual(shape, expected_shape)

  @test_util.run_in_graph_and_eager_modes
  def test_compute_output_shape(self):
    price1 = sfc.sequence_numeric_column('price1', shape=2)
    price2 = sfc.sequence_numeric_column('price2')
    features = {
        'price1': sparse_tensor.SparseTensor(
            indices=[[0, 0, 0], [0, 0, 1],
                     [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1],
                     [2, 0, 0], [2, 0, 1],
                     [3, 0, 0], [3, 0, 1]],
            values=[0., 1., 10., 11., 100., 101., 200., 201., 300., 301.],
            dense_shape=(4, 3, 2)),
        'price2': sparse_tensor.SparseTensor(
            indices=[[0, 0],
                     [0, 1],
                     [1, 0],
                     [2, 0],
                     [3, 0]],
            values=[10., 11., 20., 30., 40.],
            dense_shape=(4, 3))}
    sequence_features = ksfc.SequenceFeatures([price1, price2])
    seq_input, seq_len = sequence_features(features)
    self.assertEqual(
        sequence_features.compute_output_shape((None, None)),
        (None, None, 3))
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllClose([[[0., 1., 10.], [10., 11., 11.], [0., 0., 0.]],
                         [[100., 101., 20.], [0., 0., 0.], [0., 0., 0.]],
                         [[200., 201., 30.], [0., 0., 0.], [0., 0., 0.]],
                         [[300., 301., 40.], [0., 0., 0.], [0., 0., 0.]]],
                        self.evaluate(seq_input))
    self.assertAllClose([2, 1, 1, 1], self.evaluate(seq_len))


@test_util.run_all_in_graph_and_eager_modes
class SequenceFeaturesSerializationTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('default', None, None),
                                  ('trainable', True, 'trainable'),
                                  ('not_trainable', False, 'frozen'))
  def test_get_config(self, trainable, name):
    cols = [sfc.sequence_numeric_column('a')]
    orig_layer = ksfc.SequenceFeatures(cols, trainable=trainable, name=name)
    config = orig_layer.get_config()

    self.assertEqual(config['name'], orig_layer.name)
    self.assertEqual(config['trainable'], trainable)
    self.assertLen(config['feature_columns'], 1)
    self.assertEqual(config['feature_columns'][0]['class_name'],
                     'SequenceNumericColumn')
    self.assertEqual(config['feature_columns'][0]['config']['shape'], (1,))

  @parameterized.named_parameters(('default', None, None),
                                  ('trainable', True, 'trainable'),
                                  ('not_trainable', False, 'frozen'))
  def test_from_config(self, trainable, name):
    cols = [sfc.sequence_numeric_column('a')]
    orig_layer = ksfc.SequenceFeatures(cols, trainable=trainable, name=name)
    config = orig_layer.get_config()

    new_layer = ksfc.SequenceFeatures.from_config(config)

    self.assertEqual(new_layer.name, orig_layer.name)
    self.assertEqual(new_layer.trainable, trainable)
    self.assertLen(new_layer._feature_columns, 1)
    self.assertEqual(new_layer._feature_columns[0].name, 'a')

  def test_serialization_sequence_features(self):
    rating = sfc.sequence_numeric_column('rating')
    sequence_feature = ksfc.SequenceFeatures([rating])
    config = keras.layers.serialize(sequence_feature)

    revived = keras.layers.deserialize(config)
    self.assertIsInstance(revived, ksfc.SequenceFeatures)


class SequenceFeaturesSavingTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_saving_with_sequence_features(self):
    cols = [
        sfc.sequence_numeric_column('a'),
        fc.indicator_column(
            sfc.sequence_categorical_column_with_vocabulary_list(
                'b', ['one', 'two']))
    ]
    input_layers = {
        'a':
            keras.layers.Input(shape=(None, 1), sparse=True, name='a'),
        'b':
            keras.layers.Input(
                shape=(None, 1), sparse=True, name='b', dtype='string')
    }

    fc_layer, _ = ksfc.SequenceFeatures(cols)(input_layers)
    # TODO(tibell): Figure out the right dtype and apply masking.
    # sequence_length_mask = array_ops.sequence_mask(sequence_length)
    # x = keras.layers.GRU(32)(fc_layer, mask=sequence_length_mask)
    x = keras.layers.GRU(32)(fc_layer)
    output = keras.layers.Dense(10)(x)

    model = keras.models.Model(input_layers, output)

    model.compile(
        loss=keras.losses.MSE,
        optimizer='rmsprop',
        metrics=[keras.metrics.categorical_accuracy])

    config = model.to_json()
    loaded_model = model_config.model_from_json(config)

    batch_size = 10
    timesteps = 1

    values_a = np.arange(10, dtype=np.float32)
    indices_a = np.zeros((10, 3), dtype=np.int64)
    indices_a[:, 0] = np.arange(10)
    inputs_a = sparse_tensor.SparseTensor(indices_a, values_a,
                                          (batch_size, timesteps, 1))

    values_b = np.zeros(10, dtype=np.str)
    indices_b = np.zeros((10, 3), dtype=np.int64)
    indices_b[:, 0] = np.arange(10)
    inputs_b = sparse_tensor.SparseTensor(indices_b, values_b,
                                          (batch_size, timesteps, 1))

    with self.cached_session():
      # Initialize tables for V1 lookup.
      if not context.executing_eagerly():
        self.evaluate(lookup_ops.tables_initializer())

      self.assertLen(
          loaded_model.predict({
              'a': inputs_a,
              'b': inputs_b
          }, steps=1), batch_size)


if __name__ == '__main__':
  test.main()
