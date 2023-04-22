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

import copy

from absl.testing import parameterized

from tensorflow.python.client import session
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.tpu import feature_column_v2 as tpu_fc
from tensorflow.python.tpu import tpu
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

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True,
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False,
      })
  @test_util.deprecated_graph_mode_only
  def test_feature_layer_cpu(self, use_safe_embedding_lookup):
    # Inputs.
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 2))

    # Embedding variable.
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
    expected_lookups_sequence = (
        # example 0, ids [2], embedding = [[7, 11], [0, 0]]
        ((7., 11.), (0., 0.),),
        # example 1, ids [0, 1], embedding = [[1, 2], [3. 5]]
        ((1., 2.), (3., 5.),),
        # example 2, ids [], embedding = [0, 0]
        ((0., 0.), (0., 0.),),
        # example 3, ids [1], embedding = [3, 5]
        ((3., 5.), (0., 0.),),
    )

    # Build columns.
    categorical_column = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    sequence_categorical_column = (
        fc_lib.sequence_categorical_column_with_identity(
            key='bbb', num_buckets=vocabulary_size))
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer,
        use_safe_embedding_lookup=use_safe_embedding_lookup)
    sequence_embedding_column = tpu_fc.embedding_column_v2(
        sequence_categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer,
        max_sequence_length=2,
        use_safe_embedding_lookup=use_safe_embedding_lookup)

    # Provide sparse input and get dense result.
    features = {'aaa': sparse_input, 'bbb': sparse_input}
    dense_features = fc_lib.DenseFeatures([embedding_column])
    sequence_features = fc_lib.SequenceFeatures([sequence_embedding_column])
    embedding_lookup = dense_features(features)
    sequence_embedding_lookup = sequence_features(features)

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('dense_features/aaa_embedding/embedding_weights:0',
         'sequence_features/bbb_embedding/embedding_weights:0',),
        tuple([v.name for v in global_vars]))
    with _initialized_session():
      self.assertAllEqual(embedding_values, global_vars[0])
      self.assertAllEqual(expected_lookups, embedding_lookup)
      self.assertAllEqual(expected_lookups_sequence,
                          sequence_embedding_lookup[0].eval())
      # The graph will still have SparseFillEmptyRows due to sequence being
      # a Rank3 embedding lookup.
      if use_safe_embedding_lookup:
        self.assertEqual(2, [
            x.type for x in ops.get_default_graph().get_operations()
        ].count('SparseFillEmptyRows'))
      else:
        self.assertEqual(1, [
            x.type for x in ops.get_default_graph().get_operations()
        ].count('SparseFillEmptyRows'))

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

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False
      })
  @test_util.deprecated_graph_mode_only
  def test_feature_layer_cpu(self, use_safe_embedding_lookup):
    # Inputs.
    vocabulary_size = 3
    input_a = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(2, 2))
    input_b = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 0, 1),
        dense_shape=(3, 2))
    input_features = {'aaa': input_a, 'bbb': input_b}

    # Embedding variable.
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

    # Expected lookup result, using combiner='mean'.
    expected_lookups_a = (
        # example 0:
        (7., 11.),  # ids [2], embedding = [7, 11]
        # example 1:
        (2., 3.5),  # ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
    )
    expected_lookups_b = (
        # example 0:
        ((7., 11.), (0., 0.),),  # ids [2], embedding = [[7, 11], [0, 0]]
        # example 1:
        ((1., 2.), (3., 5.),),  # ids [0, 1], embedding = [[1, 2], [3, 5]]
        # example 2:
        ((0., 0.), (0., 0.),),  # ids [], embedding = [[0, 0], [0, 0]]
    )

    # Build columns.
    categorical_column_a = fc_lib.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc_lib.sequence_categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_column_a, embedding_column_b = tpu_fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        initializer=_initializer,
        max_sequence_lengths=[0, 2],
        use_safe_embedding_lookup=use_safe_embedding_lookup)

    # Provide sparse input and get dense result.
    dense_features = fc_lib.DenseFeatures([embedding_column_a])
    sequence_features = fc_lib.SequenceFeatures([embedding_column_b])
    embedding_lookup_a = dense_features(input_features)
    embedding_lookup_b = sequence_features(input_features)

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('aaa_bbb_shared_embedding:0',),
        tuple([v.name for v in global_vars]))
    embedding_var = global_vars[0]
    with _initialized_session():
      self.assertAllEqual(embedding_values, embedding_var)
      self.assertAllEqual(expected_lookups_a, embedding_lookup_a)
      self.assertAllEqual(expected_lookups_b,
                          embedding_lookup_b[0].eval())
      # The graph will still have SparseFillEmptyRows due to sequence being
      # a Rank3 embedding lookup.
      if use_safe_embedding_lookup:
        self.assertEqual(2, [
            x.type for x in ops.get_default_graph().get_operations()
        ].count('SparseFillEmptyRows'))
      else:
        self.assertEqual(1, [
            x.type for x in ops.get_default_graph().get_operations()
        ].count('SparseFillEmptyRows'))

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

  @parameterized.named_parameters(
      {
          'testcase_name': 'invalid_shared',
          'shared': True,
      }, {
          'testcase_name': 'invalid_not_shared',
          'shared': False,
      })
  @test_util.deprecated_graph_mode_only
  def test_invalid_cases(self, shared):

    # Inputs.
    input_sparse_tensor = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (1, 4)),
        values=(2, 0, 1, 3),
        dense_shape=(2, 5))
    input_features = {'inp': input_sparse_tensor}

    # Build columns.
    categorical_column_input = fc_lib.categorical_column_with_identity(
        key='inp', num_buckets=3)

    # Training on TPU with cpu embedding lookups is not supported.
    if shared:
      embedding_column = tpu_fc.shared_embedding_columns_v2(
          [categorical_column_input],
          dimension=2,
          embedding_lookup_device='cpu',
          tensor_core_shape=[None, 3])
    else:
      embedding_column = tpu_fc.embedding_column_v2(
          categorical_column_input,
          dimension=2,
          embedding_lookup_device='cpu',
          tensor_core_shape=[None, 3])
    dense_features = fc_lib.DenseFeatures(embedding_column)
    with self.assertRaisesRegex(
        ValueError,
        r'.*embedding_lookup_device=\"cpu\" during training is not'):
      dense_features(input_features)

    # Inference on with TPU Embedding Hardware is not supported.
    if shared:
      embedding_column = tpu_fc.shared_embedding_columns_v2(
          [categorical_column_input],
          dimension=2,
          embedding_lookup_device='tpu_embedding_core',
          tensor_core_shape=[None, 3])
    else:
      embedding_column = tpu_fc.embedding_column_v2(
          categorical_column_input,
          dimension=2,
          embedding_lookup_device='tpu_embedding_core',
          tensor_core_shape=[None, 3])
    context = tpu._TPUInferenceContext('tpu_inference')
    context.Enter()
    dense_features = fc_lib.DenseFeatures(embedding_column)
    with self.assertRaisesRegex(
        ValueError,
        r'Using embedding_lookup_device=tpu_embedding_core during inference is '
    ):
      dense_features(input_features)
    context.Exit()

  @parameterized.named_parameters(
      {
          'testcase_name': 'combiner_mean_shared',
          'shared': True,
          'combiner': 'mean'
      }, {
          'testcase_name': 'combiner_sum_shared',
          'shared': True,
          'combiner': 'sum'
      }, {
          'testcase_name': 'combiner_sqrtn_shared',
          'shared': True,
          'combiner': 'sqrtn'
      }, {
          'testcase_name': 'combiner_mean_not_shared',
          'shared': False,
          'combiner': 'mean'
      }, {
          'testcase_name': 'combiner_sum_not_shared',
          'shared': False,
          'combiner': 'sum'
      }, {
          'testcase_name': 'combiner_sqrtn_not_shared',
          'shared': False,
          'combiner': 'sqrtn'
      })
  @test_util.deprecated_graph_mode_only
  def test_dense_embedding_lookup(self, shared, combiner):
    # Inputs.
    vocabulary_size = 3
    input_sparse_tensor = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1, 3]
        indices=((0, 0), (1, 0), (1, 1), (1, 4)),
        values=(2, 0, 1, 3),
        dense_shape=(2, 5))
    input_features = {'inp': input_sparse_tensor}

    # Embedding variable.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.),  # id 2
        (13., 17.)  # id 3
    )

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    # Build columns.
    categorical_column_input = fc_lib.categorical_column_with_identity(
        key='inp', num_buckets=vocabulary_size)

    # Set tensor_core_shape to be [None, 20] to ensure some padding and
    # dynamic batch size.
    if shared:
      embedding_column = tpu_fc.shared_embedding_columns_v2(
          [categorical_column_input],
          dimension=embedding_dimension,
          initializer=_initializer,
          combiner=combiner,
          embedding_lookup_device='tpu_tensor_core',
          tensor_core_shape=[None, 3])
    else:
      embedding_column = tpu_fc.embedding_column_v2(
          categorical_column_input,
          dimension=embedding_dimension,
          initializer=_initializer,
          combiner=combiner,
          embedding_lookup_device='tpu_tensor_core',
          tensor_core_shape=[None, 3])

    # Run in TPUContexts so that we hit the intended densification case.
    context = tpu._TPUInferenceContext('tpu_inference')
    context.Enter()
    with tpu_function.tpu_shard_context(1):
      dense_features = fc_lib.DenseFeatures(embedding_column)
      # Sqrtn combiner not supported for now.
      if combiner == 'sqrtn':
        with self.assertRaisesRegex(
            ValueError, 'Dense TPU Embedding does not support combiner'):
          embedding_lookup = dense_features(input_features)
        return
      if combiner == 'mean':
        expected_lookups = (
            # example 0:
            (7., 11.),  # ids [2], embedding = [7, 11]
            # example 1:
            (2., 3.5),  # ids [0, 1], embedding = mean([1, 2] + [3, 5]) =
            # [2, 3.5]
        )
      elif combiner == 'sum':
        expected_lookups = (
            # example 0:
            (7., 11.),  # ids [2], embedding = [7, 11]
            # example 1:
            (4., 7),  # ids [0, 1], embedding = sum([1, 2] + [3, 5]) = [4, 7]
        )

      embedding_lookup = dense_features(input_features)

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      if shared:
        self.assertCountEqual(('inp_shared_embedding:0',),
                              tuple([v.name for v in global_vars]))
      else:
        self.assertCountEqual(
            ('dense_features/inp_embedding/embedding_weights:0',),
            tuple([v.name for v in global_vars]))

      embedding_var = global_vars[0]
      with _initialized_session():
        self.assertAllEqual(embedding_values, embedding_var)
        eval_res = embedding_lookup.eval()
        self.assertAllEqual(expected_lookups, eval_res)
      context.Exit()

  @test_util.deprecated_graph_mode_only
  def test_empty_row(self):
    # Inputs.
    vocabulary_size = 3
    input_sparse_tensor = sparse_tensor.SparseTensorValue(
        # example 0, ids []
        # example 1, ids [0, 1, 3]
        indices=((1, 0), (1, 1), (1, 4)),
        values=(0, 1, 3),
        dense_shape=(2, 5))
    input_features = {'inp': input_sparse_tensor}

    # Embedding variable.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.),  # id 2
        (13., 17.)  # id 3
    )

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    # Build columns.
    categorical_column_input = fc_lib.categorical_column_with_identity(
        key='inp', num_buckets=vocabulary_size)

    # Set tensor_core_shape to be [None, 20] to ensure some padding and
    # dynamic batch size.
    embedding_column = tpu_fc.embedding_column_v2(
        categorical_column_input,
        dimension=embedding_dimension,
        initializer=_initializer,
        combiner='mean',
        embedding_lookup_device='tpu_tensor_core',
        tensor_core_shape=[None, 3])

    # Run in TPUContexts so that we hit the intended densification case.
    context = tpu._TPUInferenceContext('tpu_inference')
    context.Enter()
    with tpu_function.tpu_shard_context(1):
      dense_features = fc_lib.DenseFeatures(embedding_column)
      expected_lookups = (
          # example 0:
          (0., 0.),  # ids [], embedding = [0, 0]
          # example 1:
          (2., 3.5),  # ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
      )

      embedding_lookup = dense_features(input_features)

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(
          ('dense_features/inp_embedding/embedding_weights:0',),
          tuple([v.name for v in global_vars]))

      embedding_var = global_vars[0]
      with _initialized_session():
        self.assertAllEqual(embedding_values, embedding_var)
        eval_res = embedding_lookup.eval()
        self.assertAllEqual(expected_lookups, eval_res)
      context.Exit()

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
