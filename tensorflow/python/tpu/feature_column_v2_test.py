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

import copy

from absl.testing import parameterized

from tensorflow.python.client import session
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.tpu import feature_column_v2 as tpu_fc
from tensorflow.python.tpu import tpu_function


def _initialized_session():
  sess = session.Session()
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


class _TestStateManager(fc_lib.StateManager):

  def __init__(self, trainable=True):
    self._all_variables = {}
    self._trainable = trainable

  def create_variable(self,
                      feature_column,
                      name,
                      shape,
                      dtype=None,
                      trainable=True,
                      use_resource=True,
                      initializer=None):
    if feature_column not in self._all_variables:
      self._all_variables[feature_column] = {}
    var_dict = self._all_variables[feature_column]
    if name in var_dict:
      return var_dict[name]
    else:
      var = variable_scope.get_variable(
          name=name,
          shape=shape,
          dtype=dtype,
          trainable=self._trainable and trainable,
          use_resource=use_resource,
          initializer=initializer)
      var_dict[name] = var
      return var

  def get_variable(self, feature_column, name):
    return self._all_variables[feature_column][name]


class EmbeddingColumnTestV2(test.TestCase, parameterized.TestCase):

  def test_defaults(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column, dimension=embedding_dimension)
    # Can't test default initializer as it's a random function.
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('mean', embedding_column.combiner)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)

  def test_all_constructor_args(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column,
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer')
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('my_combiner', embedding_column.combiner)
    self.assertEqual('my_initializer', embedding_column.initializer())
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column._parse_example_spec)

  def test_deepcopy(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column, dimension=2)
    embedding_column_copy = copy.deepcopy(embedding_column)
    self.assertEqual(embedding_column.dimension,
                     embedding_column_copy.dimension)
    self.assertEqual(embedding_column._max_sequence_length,
                     embedding_column_copy._max_sequence_length)

  def test_with_scope_validation(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    initializer = init_ops.truncated_normal_initializer(mean=0.0, stddev=.5)
    embedding_column = tpu_fc._TPUEmbeddingColumnV2(
        categorical_column=categorical_column,
        dimension=embedding_dimension,
        combiner='mean',
        initializer=initializer,
        max_sequence_length=0,
        learning_rate_fn=None,
        use_safe_embedding_lookup=True,
        bypass_scope_validation=False)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    state_manager = _TestStateManager()
    with tpu_function.tpu_shard_context(1):
      with variable_scope.variable_scope('tower1/scope1'):
        embedding_column.create_state(state_manager)
      with variable_scope.variable_scope('tower2/scope2'):
        # With default scope validation, the same column cannot be used in a new
        # variable scope.
        with self.assertRaisesRegex(ValueError,
                                    'the variable scope name is different'):
          embedding_column.create_state(state_manager)

  def test_bypass_scope_validation(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    initializer = init_ops.truncated_normal_initializer(mean=0.0, stddev=.5)
    embedding_column = tpu_fc._TPUEmbeddingColumnV2(
        categorical_column=categorical_column,
        dimension=embedding_dimension,
        combiner='mean',
        initializer=initializer,
        max_sequence_length=0,
        learning_rate_fn=None,
        use_safe_embedding_lookup=True,
        bypass_scope_validation=True)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    state_manager = _TestStateManager()
    with tpu_function.tpu_shard_context(1):
      with variable_scope.variable_scope('tower1/scope1'):
        embedding_column.create_state(state_manager)
      with variable_scope.variable_scope('tower2/scope2'):
        embedding_column.create_state(state_manager)

  def test_deepcopy_with_bypass_scope_validation(self):
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    initializer = init_ops.truncated_normal_initializer(mean=0.0, stddev=.5)
    embedding_column = tpu_fc._TPUEmbeddingColumnV2(
        categorical_column=categorical_column,
        dimension=embedding_dimension,
        combiner='mean',
        initializer=initializer,
        max_sequence_length=0,
        use_safe_embedding_lookup=False,
        bypass_scope_validation=True)
    embedding_column_copy = copy.deepcopy(embedding_column)
    self.assertEqual(embedding_dimension, embedding_column_copy.dimension)
    self.assertEqual(embedding_column._max_sequence_length,
                     embedding_column_copy._max_sequence_length)
    self.assertTrue(embedding_column_copy._bypass_scope_validation)
    self.assertFalse(embedding_column_copy.use_safe_embedding_lookup)


class SharedEmbeddingColumnTestV2(test.TestCase, parameterized.TestCase):

  @test_util.deprecated_graph_mode_only
  def test_defaults(self):
    vocabulary_size = 3
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_dimension = 2
    embedding_column_b, embedding_column_a = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension)
    self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
    self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual('mean', embedding_column_a.combiner)
    self.assertEqual('mean', embedding_column_b.combiner)
    self.assertIsNotNone(embedding_column_a.get_initializer())
    self.assertIsNotNone(embedding_column_b.get_initializer())
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_a.get_embedding_var_name())
    self.assertEqual('aaa_bbb_shared_embedding',
                     embedding_column_b.get_embedding_var_name())
    self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
    self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
    self.assertEqual((embedding_dimension,), embedding_column_a.variable_shape)
    self.assertEqual((embedding_dimension,), embedding_column_b.variable_shape)

  @test_util.deprecated_graph_mode_only
  def test_all_constructor_args(self):
    vocabulary_size = 3
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_dimension = 2
    embedding_column_a, embedding_column_b = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer',
        shared_embedding_collection_name='var_scope_name')
    self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
    self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual((vocabulary_size, embedding_dimension),
                     embedding_column_a.get_embedding_table_size())
    self.assertEqual('my_combiner', embedding_column_a.combiner)
    self.assertEqual('my_combiner', embedding_column_b.combiner)
    self.assertEqual('my_initializer', embedding_column_a.get_initializer()())
    self.assertEqual('my_initializer', embedding_column_b.get_initializer()())
    self.assertEqual('var_scope_name',
                     embedding_column_a.get_embedding_var_name())
    self.assertEqual('var_scope_name',
                     embedding_column_b.get_embedding_var_name())
    self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
    self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
    self.assertEqual((embedding_dimension,), embedding_column_a.variable_shape)
    self.assertEqual((embedding_dimension,), embedding_column_b.variable_shape)

  def test_deepcopy(self):
    vocabulary_size = 3
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_dimension = 2
    columns = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension)
    columns_copy = copy.deepcopy(columns)
    self.assertEqual(
        [column._shared_embedding_collection_name for column in columns],
        [column._shared_embedding_collection_name for column in columns_copy])


class DeviceSpecificEmbeddingColumnTestV2(test.TestCase,
                                          parameterized.TestCase):

  @test_util.deprecated_graph_mode_only
  def test_error_dense_shape_invalid(self):
    categorical_column_input = fc_lib.categorical_column_with_identity(
        key='inp', num_buckets=5)
    with self.assertRaisesRegex(ValueError, 'tensor_core_shape must be size 2'):
      tpu_fc.shared_embedding_columns_v2([categorical_column_input],
                                         dimension=20,
                                         tensor_core_shape=[None, 20, 15])


if __name__ == '__main__':
  test.main()
