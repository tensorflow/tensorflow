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

from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column_v2 as sfc
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.feature_column.feature_column_v2_test import _TestStateManager
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session


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
      def _initializer(shape, dtype, partition_info):
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
    sequence_input_layer = sfc.SequenceFeatures(
        [embedding_column_b, embedding_column_a])
    input_layer, sequence_length = sequence_input_layer({
        'aaa': sparse_input_a, 'bbb': sparse_input_b,})

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertCountEqual(
        ('sequence_features/aaa_embedding/embedding_weights:0',
         'sequence_features/bbb_embedding/embedding_weights:0'),
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
        r'type SequenceCategoricalColumn to use SequenceFeatures\.'):
      sequence_input_layer = sfc.SequenceFeatures([embedding_column_a])
      _, _ = sequence_input_layer({'aaa': sparse_input})

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
    shared_embedding_columns = fc.shared_embedding_columns_v2(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension,
        initializer=_get_initializer(embedding_dimension, embedding_values))

    sequence_input_layer = sfc.SequenceFeatures(shared_embedding_columns)
    input_layer, sequence_length = sequence_input_layer({
        'aaa': sparse_input_a, 'bbb': sparse_input_b})

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertCountEqual(
        ('aaa_bbb_shared_embedding:0',),
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
    shared_embedding_columns = fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b], dimension=2)

    with self.assertRaisesRegexp(
        ValueError,
        r'In embedding_column: aaa_shared_embedding\. categorical_column must '
        r'be of type SequenceCategoricalColumn to use SequenceFeatures\.'):
      sequence_input_layer = sfc.SequenceFeatures(shared_embedding_columns)
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
    sequence_input_layer = sfc.SequenceFeatures(
        [indicator_column_b, indicator_column_a])
    input_layer, sequence_length = sequence_input_layer({
        'aaa': sparse_input_a, 'bbb': sparse_input_b})

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
        r'type SequenceCategoricalColumn to use SequenceFeatures\.'):
      sequence_input_layer = sfc.SequenceFeatures([indicator_column_a])
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
  def test_numeric_column(
      self, sparse_input_args, expected_input_layer, expected_sequence_length):
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)

    numeric_column = sfc.sequence_numeric_column('aaa')

    sequence_input_layer = sfc.SequenceFeatures([numeric_column])
    input_layer, sequence_length = sequence_input_layer({'aaa': sparse_input})

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(expected_input_layer, input_layer.eval(session=sess))
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))

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
  def test_numeric_column_multi_dim(
      self, sparse_input_args, expected_input_layer, expected_sequence_length):
    """Tests SequenceFeatures for multi-dimensional numeric_column."""
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)

    numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

    sequence_input_layer = sfc.SequenceFeatures([numeric_column])
    input_layer, sequence_length = sequence_input_layer({'aaa': sparse_input})

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

    sequence_input_layer = sfc.SequenceFeatures(
        [numeric_column_a, numeric_column_b])
    _, sequence_length = sequence_input_layer({
        'aaa': sparse_input_a, 'bbb': sparse_input_b})

    with monitored_session.MonitoredSession() as sess:
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[Condition x == y did not hold element-wise:\] '
          r'\[x \(sequence_features/aaa/sequence_length:0\) = \] \[2 1\] '
          r'\[y \(sequence_features/bbb/sequence_length:0\) = \] \[1 1\]'):
        sess.run(sequence_length)

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
                       (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 2),
                       (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3)),
           'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
           'dense_shape': (2, 2, 4)},
       'expected_shape': [2, 2, 4]},
      )
  def test_static_shape_from_tensors_numeric(
      self, sparse_input_args, expected_shape):
    """Tests that we return a known static shape when we have one."""
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)
    numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

    sequence_input_layer = sfc.SequenceFeatures([numeric_column])
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
  def test_static_shape_from_tensors_indicator(
      self, sparse_input_args, expected_shape):
    """Tests that we return a known static shape when we have one."""
    sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)
    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=3)
    indicator_column = fc.indicator_column(categorical_column)

    sequence_input_layer = sfc.SequenceFeatures([indicator_column])
    input_layer, _ = sequence_input_layer({'aaa': sparse_input})
    shape = input_layer.get_shape()
    self.assertEqual(shape, expected_shape)

  def test_compute_output_shape(self):
    price1 = sfc.sequence_numeric_column('price1', shape=2)
    price2 = sfc.sequence_numeric_column('price2')
    with ops.Graph().as_default():
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
      sequence_features = sfc.SequenceFeatures([price1, price2])
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
    with monitored_session.MonitoredSession() as sess:
      output = sess.run(input_layer)
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
    with self.assertRaisesRegexp(ValueError, 'sequence_input must have rank 3'):
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
    with self.assertRaisesRegexp(ValueError, 'context_input must have rank 2'):
      sfc.concatenate_context_input(context_input, seq_input)

  def test_integer_seq_input_throws_error(self):
    seq_input = ops.convert_to_tensor(np.arange(100).reshape(5, 5, 4))
    context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
    context_input = math_ops.cast(context_input, dtype=dtypes.float32)
    with self.assertRaisesRegexp(
        TypeError, 'sequence_input must have dtype float32'):
      sfc.concatenate_context_input(context_input, seq_input)

  def test_integer_context_input_throws_error(self):
    seq_input = ops.convert_to_tensor(np.arange(100).reshape(5, 5, 4))
    context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
    seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
    with self.assertRaisesRegexp(
        TypeError, 'context_input must have dtype float32'):
      sfc.concatenate_context_input(context_input, seq_input)


class DenseFeaturesTest(test.TestCase):
  """Tests DenseFeatures with sequence feature columns."""

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
        r'of type SequenceCategoricalColumn\.'):
      input_layer = fc.DenseFeatures([embedding_column_a])
      _ = input_layer({'aaa': sparse_input})

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
        r'of type SequenceCategoricalColumn\.'):
      input_layer = fc.DenseFeatures([indicator_column_a])
      _ = input_layer({'aaa': sparse_input})


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
  state_manager = _TestStateManager()
  column.create_state(state_manager)
  return column.get_sequence_dense_tensor(
      fc.FeatureTransformationCache(features), state_manager)


def _get_sparse_tensors(column, features):
  return column.get_sparse_tensors(
      fc.FeatureTransformationCache(features), None)


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
           'values': (6, 7, 8),
           'dense_shape': (2, 2, 2)}}
      )
  def test_get_sparse_tensors(self, inputs_args, expected_args):
    inputs = sparse_tensor.SparseTensorValue(**inputs_args)
    expected = sparse_tensor.SparseTensorValue(**expected_args)
    column = sfc.sequence_categorical_column_with_identity('aaa', num_buckets=9)

    id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

    self.assertIsNone(id_weight_pair.weight_tensor)
    with monitored_session.MonitoredSession() as sess:
      _assert_sparse_tensor_value(
          self, expected, id_weight_pair.id_tensor.eval(session=sess))


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
    with monitored_session.MonitoredSession() as sess:
      _assert_sparse_tensor_indices_shape(
          self, expected, id_weight_pair.id_tensor.eval(session=sess))


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
    with monitored_session.MonitoredSession() as sess:
      _assert_sparse_tensor_value(
          self, expected, id_weight_pair.id_tensor.eval(session=sess))

  def test_get_sparse_tensors_dynamic_zero_length(self):
    """Tests _get_sparse_tensors with a dynamic sequence length."""
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
    with monitored_session.MonitoredSession() as sess:
      result = id_weight_pair.id_tensor.eval(
          session=sess, feed_dict={input_placeholder: inputs})
      _assert_sparse_tensor_value(
          self, expected, result)


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
    with monitored_session.MonitoredSession() as sess:
      _assert_sparse_tensor_value(
          self, expected, id_weight_pair.id_tensor.eval(session=sess))


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
    def _initializer(shape, dtype, partition_info):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    categorical_column = sfc.sequence_categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        initializer=_initializer)

    embedding_lookup, _ = _get_sequence_dense_tensor_state(
        embedding_column, {'aaa': inputs})

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertCountEqual(
        ('embedding_weights:0',), tuple([v.name for v in global_vars]))
    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(embedding_values, global_vars[0].eval(session=sess))
      self.assertAllEqual(expected, embedding_lookup.eval(session=sess))

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

    _, sequence_length = _get_sequence_dense_tensor_state(
        embedding_column, {'aaa': inputs})

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

    _, sequence_length = _get_sequence_dense_tensor_state(
        embedding_column, {'aaa': sparse_input})

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
    shared_embedding_columns = fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        initializer=_initializer)

    embedding_lookup_a = _get_sequence_dense_tensor(
        shared_embedding_columns[0], {'aaa': sparse_input_a})[0]
    embedding_lookup_b = _get_sequence_dense_tensor(
        shared_embedding_columns[1], {'bbb': sparse_input_b})[0]

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('aaa_bbb_shared_embedding:0',),
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
    shared_embedding_columns = fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b], dimension=2)

    sequence_length_a = _get_sequence_dense_tensor(
        shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
    sequence_length_b = _get_sequence_dense_tensor(
        shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

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

    shared_embedding_columns = fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b], dimension=2)

    sequence_length_a = _get_sequence_dense_tensor(
        shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
    sequence_length_b = _get_sequence_dense_tensor(
        shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_sequence_length_a, sequence_length_a.eval(session=sess))
      self.assertAllEqual(
          expected_sequence_length_b, sequence_length_b.eval(session=sess))


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

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(expected, indicator_tensor.eval(session=sess))

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

    _, sequence_length = _get_sequence_dense_tensor(
        indicator_column, {'aaa': sparse_input})

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))


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
    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(expected, dense_tensor.eval(session=sess))

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

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_dense_tensor, dense_tensor.eval(session=sess))

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

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_dense_tensor, dense_tensor.eval(session=sess))

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

    with monitored_session.MonitoredSession() as sess:
      sequence_length = sess.run(sequence_length)
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

    with monitored_session.MonitoredSession() as sess:
      self.assertAllEqual(
          expected_sequence_length, sequence_length.eval(session=sess))


if __name__ == '__main__':
  test.main()
