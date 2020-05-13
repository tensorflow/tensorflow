# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for dense_features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras.feature_column import dense_features as df
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


def _initialized_session(config=None):
  sess = session.Session(config=config)
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


class DenseFeaturesTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_retrieving_input(self):
    features = {'a': [0.]}
    dense_features = df.DenseFeatures(fc.numeric_column('a'))
    inputs = self.evaluate(dense_features(features))
    self.assertAllClose([[0.]], inputs)

  def test_reuses_variables(self):
    with context.eager_mode():
      sparse_input = sparse_tensor.SparseTensor(
          indices=((0, 0), (1, 0), (2, 0)),
          values=(0, 1, 2),
          dense_shape=(3, 3))

      # Create feature columns (categorical and embedding).
      categorical_column = fc.categorical_column_with_identity(
          key='a', num_buckets=3)
      embedding_dimension = 2

      def _embedding_column_initializer(shape, dtype, partition_info=None):
        del shape  # unused
        del dtype  # unused
        del partition_info  # unused
        embedding_values = (
            (1, 0),  # id 0
            (0, 1),  # id 1
            (1, 1))  # id 2
        return embedding_values

      embedding_column = fc.embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_embedding_column_initializer)

      dense_features = df.DenseFeatures([embedding_column])
      features = {'a': sparse_input}

      inputs = dense_features(features)
      variables = dense_features.variables

      # Sanity check: test that the inputs are correct.
      self.assertAllEqual([[1, 0], [0, 1], [1, 1]], inputs)

      # Check that only one variable was created.
      self.assertEqual(1, len(variables))

      # Check that invoking dense_features on the same features does not create
      # additional variables
      _ = dense_features(features)
      self.assertEqual(1, len(variables))
      self.assertIs(variables[0], dense_features.variables[0])

  def test_dense_feature_with_partitioner(self):
    with context.eager_mode():
      sparse_input = sparse_tensor.SparseTensor(
          indices=((0, 0), (1, 0), (2, 0), (3, 0)),
          values=(0, 1, 3, 2),
          dense_shape=(4, 4))

      # Create feature columns (categorical and embedding).
      categorical_column = fc.categorical_column_with_identity(
          key='a', num_buckets=4)
      embedding_dimension = 2

      def _embedding_column_initializer(shape, dtype, partition_info=None):
        offset = partition_info._var_offset[0]
        del shape  # unused
        del dtype  # unused
        if offset == 0:
          embedding_values = (
              (1, 0),  # id 0
              (0, 1))  # id 1
        else:
          embedding_values = (
              (1, 1),  # id 2
              (2, 2))  # id 3
        return embedding_values

      embedding_column = fc.embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_embedding_column_initializer)

      dense_features = df.DenseFeatures(
          [embedding_column],
          partitioner=partitioned_variables.fixed_size_partitioner(2))
      features = {'a': sparse_input}

      inputs = dense_features(features)
      variables = dense_features.variables

      # Sanity check: test that the inputs are correct.
      self.assertAllEqual([[1, 0], [0, 1], [2, 2], [1, 1]], inputs)

      # Check that only one variable was created.
      self.assertEqual(2, len(variables))

      # Check that invoking dense_features on the same features does not create
      # additional variables
      _ = dense_features(features)
      self.assertEqual(2, len(variables))
      self.assertIs(variables[0], dense_features.variables[0])
      self.assertIs(variables[1], dense_features.variables[1])

  def test_feature_column_dense_features_gradient(self):
    with context.eager_mode():
      sparse_input = sparse_tensor.SparseTensor(
          indices=((0, 0), (1, 0), (2, 0)),
          values=(0, 1, 2),
          dense_shape=(3, 3))

      # Create feature columns (categorical and embedding).
      categorical_column = fc.categorical_column_with_identity(
          key='a', num_buckets=3)
      embedding_dimension = 2

      def _embedding_column_initializer(shape, dtype, partition_info=None):
        del shape  # unused
        del dtype  # unused
        del partition_info  # unused
        embedding_values = (
            (1, 0),  # id 0
            (0, 1),  # id 1
            (1, 1))  # id 2
        return embedding_values

      embedding_column = fc.embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_embedding_column_initializer)

      dense_features = df.DenseFeatures([embedding_column])
      features = {'a': sparse_input}

      def scale_matrix():
        matrix = dense_features(features)
        return 2 * matrix

      # Sanity check: Verify that scale_matrix returns the correct output.
      self.assertAllEqual([[2, 0], [0, 2], [2, 2]], scale_matrix())

      # Check that the returned gradient is correct.
      grad_function = backprop.implicit_grad(scale_matrix)
      grads_and_vars = grad_function()
      indexed_slice = grads_and_vars[0][0]
      gradient = grads_and_vars[0][0].values

      self.assertAllEqual([0, 1, 2], indexed_slice.indices)
      self.assertAllEqual([[2, 2], [2, 2], [2, 2]], gradient)

  def test_raises_if_empty_feature_columns(self):
    with self.assertRaisesRegexp(ValueError,
                                 'feature_columns must not be empty'):
      df.DenseFeatures(feature_columns=[])(features={})

  def test_should_be_dense_column(self):
    with self.assertRaisesRegexp(ValueError, 'must be a .*DenseColumn'):
      df.DenseFeatures(feature_columns=[
          fc.categorical_column_with_hash_bucket('wire_cast', 4)
      ])(
          features={
              'a': [[0]]
          })

  def test_does_not_support_dict_columns(self):
    with self.assertRaisesRegexp(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      df.DenseFeatures(feature_columns={'a': fc.numeric_column('a')})(
          features={
              'a': [[0]]
          })

  def test_bare_column(self):
    with ops.Graph().as_default():
      features = features = {'a': [0.]}
      net = df.DenseFeatures(fc.numeric_column('a'))(features)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0.]], self.evaluate(net))

  def test_column_generator(self):
    with ops.Graph().as_default():
      features = features = {'a': [0.], 'b': [1.]}
      columns = (fc.numeric_column(key) for key in features)
      net = df.DenseFeatures(columns)(features)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0., 1.]], self.evaluate(net))

  def test_raises_if_duplicate_name(self):
    with self.assertRaisesRegexp(
        ValueError, 'Duplicate feature column name found for columns'):
      df.DenseFeatures(
          feature_columns=[fc.numeric_column('a'),
                           fc.numeric_column('a')])(
                               features={
                                   'a': [[0]]
                               })

  def test_one_column(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      net = df.DenseFeatures([price])(features)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1.], [5.]], self.evaluate(net))

  def test_multi_dimension(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      net = df.DenseFeatures([price])(features)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_compute_output_shape(self):
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2', shape=4)
    with ops.Graph().as_default():
      features = {
          'price1': [[1., 2.], [5., 6.]],
          'price2': [[3., 4., 5., 6.], [7., 8., 9., 10.]]
      }
      dense_features = df.DenseFeatures([price1, price2])
      self.assertEqual((None, 6), dense_features.compute_output_shape((None,)))
      net = dense_features(features)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2., 3., 4., 5., 6.], [5., 6., 7., 8., 9., 10.]],
                          self.evaluate(net))

  def test_raises_if_shape_mismatch(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegexp(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        df.DenseFeatures([price])(features)

  def test_reshaping(self):
    price = fc.numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      net = df.DenseFeatures([price])(features)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_multi_column(self):
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      net = df.DenseFeatures([price1, price2])(features)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2., 3.], [5., 6., 4.]], self.evaluate(net))

  def test_cols_to_output_tensors(self):
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      cols_dict = {}
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      dense_features = df.DenseFeatures([price1, price2])
      net = dense_features(features, cols_dict)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]],
                          self.evaluate(cols_dict[price1]))
      self.assertAllClose([[3.], [4.]], self.evaluate(cols_dict[price2]))
      self.assertAllClose([[1., 2., 3.], [5., 6., 4.]], self.evaluate(net))

  def test_column_order(self):
    price_a = fc.numeric_column('price_a')
    price_b = fc.numeric_column('price_b')
    with ops.Graph().as_default():
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
      }
      net1 = df.DenseFeatures([price_a, price_b])(features)
      net2 = df.DenseFeatures([price_b, price_a])(features)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 3.]], self.evaluate(net1))
      self.assertAllClose([[1., 3.]], self.evaluate(net2))

  def test_fails_for_categorical_column(self):
    animal = fc.categorical_column_with_identity('animal', num_buckets=4)
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      with self.assertRaisesRegexp(Exception, 'must be a .*DenseColumn'):
        df.DenseFeatures([animal])(features)

  def test_static_batch_size_mismatch(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      with self.assertRaisesRegexp(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        df.DenseFeatures([price1, price2])(features)

  def test_subset_of_static_batch_size_mismatch(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    price3 = fc.numeric_column('price3')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]],  # batchsize = 2
          'price3': [[3.], [4.], [5.]]  # batchsize = 3
      }
      with self.assertRaisesRegexp(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        df.DenseFeatures([price1, price2, price3])(features)

  def test_runtime_batch_size_mismatch(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      net = df.DenseFeatures([price1, price2])(features)
      with _initialized_session() as sess:
        with self.assertRaisesRegexp(errors.OpError,
                                     'Dimensions of inputs should match'):
          sess.run(net, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
          'price2': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
      }
      net = df.DenseFeatures([price1, price2])(features)
      with _initialized_session() as sess:
        sess.run(
            net,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  def test_multiple_layers_with_same_embedding_column(self):
    some_sparse_column = fc.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = fc.embedding_column(
        some_sparse_column, dimension=10)

    with ops.Graph().as_default():
      features = {
          'sparse_feature': [['a'], ['x']],
      }
      all_cols = [some_embedding_column]
      df.DenseFeatures(all_cols)(features)
      df.DenseFeatures(all_cols)(features)
      # Make sure that 2 variables get created in this case.
      self.assertEqual(2,
                       len(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))
      expected_var_names = [
          'dense_features/sparse_feature_embedding/embedding_weights:0',
          'dense_features_1/sparse_feature_embedding/embedding_weights:0'
      ]
      self.assertItemsEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])

  @test_util.run_deprecated_v1
  def test_multiple_layers_with_same_shared_embedding_column(self):
    categorical_column_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_column_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=3)
    embedding_dimension = 2
    embedding_column_b, embedding_column_a = fc.shared_embedding_columns_v2(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension)

    with ops.Graph().as_default():
      features = {
          'aaa':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }
      all_cols = [embedding_column_a, embedding_column_b]
      df.DenseFeatures(all_cols)(features)
      df.DenseFeatures(all_cols)(features)
      # Make sure that only 1 variable gets created in this case.
      self.assertEqual(1,
                       len(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))
      self.assertItemsEqual(
          ['aaa_bbb_shared_embedding:0'],
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])

  @test_util.run_deprecated_v1
  def test_multiple_layers_with_same_shared_embedding_column_diff_graphs(self):
    categorical_column_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_column_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=3)
    embedding_dimension = 2
    embedding_column_b, embedding_column_a = fc.shared_embedding_columns_v2(
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension)
    all_cols = [embedding_column_a, embedding_column_b]

    with ops.Graph().as_default():
      features = {
          'aaa':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }
      df.DenseFeatures(all_cols)(features)
      # Make sure that only 1 variable gets created in this case.
      self.assertEqual(1,
                       len(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))

    with ops.Graph().as_default():
      features1 = {
          'aaa':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }

      df.DenseFeatures(all_cols)(features1)
      # Make sure that only 1 variable gets created in this case.
      self.assertEqual(1,
                       len(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))
      self.assertItemsEqual(
          ['aaa_bbb_shared_embedding:0'],
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])

  @test_util.run_deprecated_v1
  def test_with_1d_sparse_tensor(self):
    embedding_values = (
        (1., 2., 3., 4., 5.),  # id 0
        (6., 7., 8., 9., 10.),  # id 1
        (11., 12., 13., 14., 15.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      del shape, dtype, partition_info
      return embedding_values

    # price has 1 dimension in dense_features
    price = fc.numeric_column('price')

    # one_hot_body_style has 3 dims in dense_features.
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    one_hot_body_style = fc.indicator_column(body_style)

    # embedded_body_style has 5 dims in dense_features.
    country = fc.categorical_column_with_vocabulary_list(
        'country', vocabulary_list=['US', 'JP', 'CA'])
    embedded_country = fc.embedding_column(
        country, dimension=5, initializer=_initializer)

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price':
            constant_op.constant([
                11.,
                12.,
            ]),
        'body-style':
            sparse_tensor.SparseTensor(
                indices=((0,), (1,)),
                values=('sedan', 'hardtop'),
                dense_shape=(2,)),
        # This is dense tensor for the categorical_column.
        'country':
            constant_op.constant(['CA', 'US']),
    }
    self.assertEqual(1, features['price'].shape.ndims)
    self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])
    self.assertEqual(1, features['country'].shape.ndims)

    net = df.DenseFeatures([price, one_hot_body_style, embedded_country])(
        features)
    self.assertEqual(1 + 3 + 5, net.shape[1])
    with _initialized_session() as sess:

      # Each row is formed by concatenating `embedded_body_style`,
      # `one_hot_body_style`, and `price` in order.
      self.assertAllEqual([[0., 0., 1., 11., 12., 13., 14., 15., 11.],
                           [1., 0., 0., 1., 2., 3., 4., 5., 12.]],
                          sess.run(net))

  @test_util.run_deprecated_v1
  def test_with_1d_unknown_shape_sparse_tensor(self):
    embedding_values = (
        (1., 2.),  # id 0
        (6., 7.),  # id 1
        (11., 12.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      del shape, dtype, partition_info
      return embedding_values

    # price has 1 dimension in dense_features
    price = fc.numeric_column('price')

    # one_hot_body_style has 3 dims in dense_features.
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    one_hot_body_style = fc.indicator_column(body_style)

    # embedded_body_style has 5 dims in dense_features.
    country = fc.categorical_column_with_vocabulary_list(
        'country', vocabulary_list=['US', 'JP', 'CA'])
    embedded_country = fc.embedding_column(
        country, dimension=2, initializer=_initializer)

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price': array_ops.placeholder(dtypes.float32),
        'body-style': array_ops.sparse_placeholder(dtypes.string),
        # This is dense tensor for the categorical_column.
        'country': array_ops.placeholder(dtypes.string),
    }
    self.assertIsNone(features['price'].shape.ndims)
    self.assertIsNone(features['body-style'].get_shape().ndims)
    self.assertIsNone(features['country'].shape.ndims)

    price_data = np.array([11., 12.])
    body_style_data = sparse_tensor.SparseTensorValue(
        indices=((0,), (1,)), values=('sedan', 'hardtop'), dense_shape=(2,))
    country_data = np.array([['US'], ['CA']])

    net = df.DenseFeatures([price, one_hot_body_style, embedded_country])(
        features)
    self.assertEqual(1 + 3 + 2, net.shape[1])
    with _initialized_session() as sess:

      # Each row is formed by concatenating `embedded_body_style`,
      # `one_hot_body_style`, and `price` in order.
      self.assertAllEqual(
          [[0., 0., 1., 1., 2., 11.], [1., 0., 0., 11., 12., 12.]],
          sess.run(
              net,
              feed_dict={
                  features['price']: price_data,
                  features['body-style']: body_style_data,
                  features['country']: country_data
              }))

  @test_util.run_deprecated_v1
  def test_with_rank_0_feature(self):
    # price has 1 dimension in dense_features
    price = fc.numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegexp(ValueError, 'Feature .* cannot have rank 0'):
      df.DenseFeatures([price])(features)

    # Dynamic rank 0 should fail
    features = {
        'price': array_ops.placeholder(dtypes.float32),
    }
    net = df.DenseFeatures([price])(features)
    self.assertEqual(1, net.shape[1])
    with _initialized_session() as sess:
      with self.assertRaisesOpError('Feature .* cannot have rank 0'):
        sess.run(net, feed_dict={features['price']: np.array(1)})


class IndicatorColumnTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_dense_features(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      net = df.DenseFeatures([animal])(features)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0., 1., 1., 0.]], self.evaluate(net))


class EmbeddingColumnTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False
      })
  @test_util.run_deprecated_v1
  def test_dense_features(self, use_safe_embedding_lookup):
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

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer,
        use_safe_embedding_lookup=use_safe_embedding_lookup)

    # Provide sparse input and get dense result.
    l = df.DenseFeatures((embedding_column,))
    dense_features = l({'aaa': sparse_input})

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertCountEqual(('dense_features/aaa_embedding/embedding_weights:0',),
                          tuple([v.name for v in global_vars]))
    for v in global_vars:
      self.assertIsInstance(v, variables_lib.Variable)
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertCountEqual(('dense_features/aaa_embedding/embedding_weights:0',),
                          tuple([v.name for v in trainable_vars]))

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual(embedding_values, self.evaluate(trainable_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(dense_features))

    if use_safe_embedding_lookup:
      self.assertIn('SparseFillEmptyRows',
                    [x.type for x in ops.get_default_graph().get_operations()])
    else:
      self.assertNotIn(
          'SparseFillEmptyRows',
          [x.type for x in ops.get_default_graph().get_operations()])

  @test_util.run_deprecated_v1
  def test_dense_features_not_trainable(self):
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

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer,
        trainable=False)

    # Provide sparse input and get dense result.
    dense_features = df.DenseFeatures((embedding_column,))({
        'aaa': sparse_input
    })

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertCountEqual(('dense_features/aaa_embedding/embedding_weights:0',),
                          tuple([v.name for v in global_vars]))
    self.assertCountEqual([],
                          ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(dense_features))


class SharedEmbeddingColumnTest(test.TestCase, parameterized.TestCase):

  def _test_dense_features(self, trainable=True):
    # Inputs.
    vocabulary_size = 3
    sparse_input_a = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 0), (1, 0), (1, 4)),
        values=(2, 0, 1),
        dense_shape=(2, 5))
    sparse_input_b = sparse_tensor.SparseTensorValue(
        # example 0, ids [0]
        # example 1, ids []
        indices=((0, 0),),
        values=(0,),
        dense_shape=(2, 5))
    sparse_input_c = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        indices=((0, 1), (1, 1), (1, 3)),
        values=(2, 0, 1),
        dense_shape=(2, 5))
    sparse_input_d = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids []
        indices=((0, 1),),
        values=(2,),
        dense_shape=(2, 5))

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
        # example 0:
        # A ids [2], embedding = [7, 11]
        # B ids [0], embedding = [1, 2]
        # C ids [2], embedding = [7, 11]
        # D ids [2], embedding = [7, 11]
        (7., 11., 1., 2., 7., 11., 7., 11.),
        # example 1:
        # A ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
        # B ids [], embedding = [0, 0]
        # C ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
        # D ids [], embedding = [0, 0]
        (2., 3.5, 0., 0., 2., 3.5, 0., 0.),
    )

    # Build columns.
    categorical_column_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    categorical_column_c = fc.categorical_column_with_identity(
        key='ccc', num_buckets=vocabulary_size)
    categorical_column_d = fc.categorical_column_with_identity(
        key='ddd', num_buckets=vocabulary_size)

    embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        initializer=_initializer,
        trainable=trainable)
    embedding_column_c, embedding_column_d = fc.shared_embedding_columns_v2(
        [categorical_column_c, categorical_column_d],
        dimension=embedding_dimension,
        initializer=_initializer,
        trainable=trainable)

    features = {
        'aaa': sparse_input_a,
        'bbb': sparse_input_b,
        'ccc': sparse_input_c,
        'ddd': sparse_input_d
    }

    # Provide sparse input and get dense result.
    dense_features = df.DenseFeatures(
        feature_columns=(embedding_column_b, embedding_column_a,
                         embedding_column_c, embedding_column_d))(
                             features)

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertCountEqual(
        ['aaa_bbb_shared_embedding:0', 'ccc_ddd_shared_embedding:0'],
        tuple([v.name for v in global_vars]))
    for v in global_vars:
      self.assertIsInstance(v, variables_lib.Variable)
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    if trainable:
      self.assertCountEqual(
          ['aaa_bbb_shared_embedding:0', 'ccc_ddd_shared_embedding:0'],
          tuple([v.name for v in trainable_vars]))
    else:
      self.assertCountEqual([], tuple([v.name for v in trainable_vars]))
    shared_embedding_vars = global_vars

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual(embedding_values,
                        self.evaluate(shared_embedding_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(dense_features))

  @test_util.run_deprecated_v1
  def test_dense_features(self):
    self._test_dense_features()

  @test_util.run_deprecated_v1
  def test_dense_features_no_trainable(self):
    self._test_dense_features(trainable=False)


@test_util.run_all_in_graph_and_eager_modes
class DenseFeaturesSerializationTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('default', None, None),
      ('trainable', True, 'trainable'),
      ('not_trainable', False, 'frozen'))
  def test_get_config(self, trainable, name):
    cols = [fc.numeric_column('a'),
            fc.embedding_column(fc.categorical_column_with_identity(
                key='b', num_buckets=3), dimension=2)]
    orig_layer = df.DenseFeatures(
        cols, trainable=trainable, name=name)
    config = orig_layer.get_config()

    self.assertEqual(config['name'], orig_layer.name)
    self.assertEqual(config['trainable'], trainable)
    self.assertLen(config['feature_columns'], 2)
    self.assertEqual(
        config['feature_columns'][0]['class_name'], 'NumericColumn')
    self.assertEqual(config['feature_columns'][0]['config']['shape'], (1,))
    self.assertEqual(
        config['feature_columns'][1]['class_name'], 'EmbeddingColumn')

  @parameterized.named_parameters(
      ('default', None, None),
      ('trainable', True, 'trainable'),
      ('not_trainable', False, 'frozen'))
  def test_from_config(self, trainable, name):
    cols = [fc.numeric_column('a'),
            fc.embedding_column(fc.categorical_column_with_vocabulary_list(
                'b', vocabulary_list=['1', '2', '3']), dimension=2),
            fc.indicator_column(fc.categorical_column_with_hash_bucket(
                key='c', hash_bucket_size=3))]
    orig_layer = df.DenseFeatures(
        cols, trainable=trainable, name=name)
    config = orig_layer.get_config()

    new_layer = df.DenseFeatures.from_config(config)

    self.assertEqual(new_layer.name, orig_layer.name)
    self.assertEqual(new_layer.trainable, trainable)
    self.assertLen(new_layer._feature_columns, 3)
    self.assertEqual(new_layer._feature_columns[0].name, 'a')
    self.assertEqual(new_layer._feature_columns[1].initializer.mean, 0.0)
    self.assertEqual(new_layer._feature_columns[1].categorical_column.name, 'b')
    self.assertIsInstance(new_layer._feature_columns[2], fc.IndicatorColumn)

  def test_crossed_column(self):
    a = fc.categorical_column_with_vocabulary_list(
        'a', vocabulary_list=['1', '2', '3'])
    b = fc.categorical_column_with_vocabulary_list(
        'b', vocabulary_list=['1', '2', '3'])
    ab = fc.crossed_column([a, b], hash_bucket_size=2)
    cols = [fc.indicator_column(ab)]

    orig_layer = df.DenseFeatures(cols)
    config = orig_layer.get_config()

    new_layer = df.DenseFeatures.from_config(config)

    self.assertLen(new_layer._feature_columns, 1)
    self.assertEqual(new_layer._feature_columns[0].name, 'a_X_b_indicator')


@test_util.run_all_in_graph_and_eager_modes
class SequenceFeatureColumnsTest(test.TestCase):
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

    input_layer = df.DenseFeatures([embedding_column_a])
    with self.assertRaisesRegexp(
        ValueError,
        r'In embedding_column: aaa_embedding\. categorical_column must not be '
        r'of type SequenceCategoricalColumn\.'):
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

    input_layer = df.DenseFeatures([indicator_column_a])
    with self.assertRaisesRegexp(
        ValueError,
        r'In indicator_column: aaa_indicator\. categorical_column must not be '
        r'of type SequenceCategoricalColumn\.'):
      _ = input_layer({'aaa': sparse_input})


if __name__ == '__main__':
  test.main()
