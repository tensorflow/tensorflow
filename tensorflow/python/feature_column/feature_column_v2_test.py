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
# ==============================================================================
"""Tests for feature_column."""

import collections
import copy

from absl.testing import parameterized
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import feature_column_v2_types
from tensorflow.python.feature_column import serialization
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


def _initialized_session(config=None):
  sess = session.Session(config=config)
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


def get_linear_model_bias(name='linear_model'):
  with variable_scope.variable_scope(name, reuse=True):
    return variable_scope.get_variable('bias_weights')


def get_linear_model_column_var(column, name='linear_model'):
  return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                            name + '/' + column.name)[0]


class BaseFeatureColumnForTests(feature_column_v2_types.FeatureColumn):
  """A base FeatureColumn useful to avoid boiler-plate in tests.

  Provides dummy implementations for abstract methods that raise ValueError in
  order to avoid re-defining all abstract methods for each test sub-class.
  """

  @property
  def parents(self):
    raise ValueError('Should not use this method.')

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    raise ValueError('Should not use this method.')

  def get_config(self):
    raise ValueError('Should not use this method.')


class SortableFeatureColumnTest(test.TestCase):

  def test_sort_columns_by_string_representation(self):
    # These should be sorted lexicographically based on their string
    # representations. For FeatureColumns, this looks like
    # '<__main__.FeatureColumn object at ...>'.

    a = fc.numeric_column('first')  # '<__main__.NumericColumn ...>'
    b = fc.numeric_column('second')  # '<__main__.NumericColumn ...>'
    c = fc_old._numeric_column('third')  # '<__main__._NumericColumn ...>'

    sorted_sequence = ['0', a, b, c, 'd']
    reversed_sequence = sorted_sequence[::-1]
    self.assertAllEqual(sorted(reversed_sequence), sorted_sequence)

    # pylint: disable=g-generic-assert
    self.assertTrue(a < b)  # V2 < V2 feature columns.
    self.assertTrue(a < c)  # V2 < V1 feature columns.
    self.assertFalse(c < a)  # V1 < V2 feature columns.
    self.assertTrue('0' < a)  # string < V2 feature column.
    self.assertTrue(a < 'd')  # V2 feature column < string.
    # pylint: enable=g-generic-assert


class LazyColumnTest(test.TestCase):

  def test_transformations_called_once(self):

    class TransformCounter(BaseFeatureColumnForTests):

      def __init__(self):
        super(TransformCounter, self).__init__()
        self.num_transform = 0

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'TransformCounter'

      def transform_feature(self, transformation_cache, state_manager):
        self.num_transform += 1  # Count transform calls.
        return transformation_cache.get('a', state_manager)

      @property
      def parse_example_spec(self):
        pass

    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    column = TransformCounter()
    self.assertEqual(0, column.num_transform)
    transformation_cache.get(column, None)
    self.assertEqual(1, column.num_transform)
    transformation_cache.get(column, None)
    self.assertEqual(1, column.num_transform)

  def test_returns_transform_output(self):

    class Transformer(BaseFeatureColumnForTests):

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'Transformer'

      def transform_feature(self, transformation_cache, state_manager):
        return 'Output'

      @property
      def parse_example_spec(self):
        pass

    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    column = Transformer()
    self.assertEqual('Output', transformation_cache.get(column, None))
    self.assertEqual('Output', transformation_cache.get(column, None))

  def test_does_not_pollute_given_features_dict(self):

    class Transformer(BaseFeatureColumnForTests):

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'Transformer'

      def transform_feature(self, transformation_cache, state_manager):
        return 'Output'

      @property
      def parse_example_spec(self):
        pass

    features = {'a': [[2], [3.]]}
    transformation_cache = fc.FeatureTransformationCache(features=features)
    transformation_cache.get(Transformer(), None)
    self.assertEqual(['a'], list(features.keys()))

  def test_error_if_feature_is_not_found(self):
    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    with self.assertRaisesRegex(ValueError,
                                'bbb is not in features dictionary'):
      transformation_cache.get('bbb', None)
    with self.assertRaisesRegex(ValueError,
                                'bbb is not in features dictionary'):
      transformation_cache.get(u'bbb', None)

  def test_not_supported_feature_column(self):

    class NotAProperColumn(BaseFeatureColumnForTests):

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'NotAProperColumn'

      def transform_feature(self, transformation_cache, state_manager):
        # It should return not None.
        pass

      @property
      def parse_example_spec(self):
        pass

    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    with self.assertRaisesRegex(ValueError,
                                'NotAProperColumn is not supported'):
      transformation_cache.get(NotAProperColumn(), None)

  def test_key_should_be_string_or_feature_colum(self):

    class NotAFeatureColumn(object):
      pass

    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    with self.assertRaisesRegex(
        TypeError, '"key" must be either a "str" or "FeatureColumn".'):
      transformation_cache.get(NotAFeatureColumn(), None)

  def test_expand_dim_rank_1_sparse_tensor_empty_batch(self):
    # empty 1-D sparse tensor:
    transformation_cache = fc.FeatureTransformationCache(
        features={
            'a':
                sparse_tensor.SparseTensor(
                    indices=np.reshape(np.array([], dtype=np.int64), (0, 1)),
                    dense_shape=[0],
                    values=np.array([]))
        })

    spv = self.evaluate(transformation_cache.get('a', None))
    self.assertAllEqual(np.array([0, 1], dtype=np.int64), spv.dense_shape)
    self.assertAllEqual(
        np.reshape(np.array([], dtype=np.int64), (0, 2)), spv.indices)


class NumericColumnTest(test.TestCase):

  def test_defaults(self):
    a = fc.numeric_column('aaa')
    self.assertEqual('aaa', a.key)
    self.assertEqual('aaa', a.name)
    self.assertEqual((1,), a.shape)
    self.assertIsNone(a.default_value)
    self.assertEqual(dtypes.float32, a.dtype)
    self.assertIsNone(a.normalizer_fn)
    self.assertTrue(a._is_v2_column)

  def test_key_should_be_string(self):
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.numeric_column(key=('aaa',))

  def test_shape_saved_as_tuple(self):
    a = fc.numeric_column('aaa', shape=[1, 2], default_value=[[3, 2.]])
    self.assertEqual((1, 2), a.shape)

  def test_default_value_saved_as_tuple(self):
    a = fc.numeric_column('aaa', default_value=4.)
    self.assertEqual((4.,), a.default_value)
    a = fc.numeric_column('aaa', shape=[1, 2], default_value=[[3, 2.]])
    self.assertEqual(((3., 2.),), a.default_value)

  def test_shape_and_default_value_compatibility(self):
    a = fc.numeric_column('aaa', shape=[2], default_value=[1, 2.])
    self.assertEqual((1, 2.), a.default_value)
    with self.assertRaisesRegex(ValueError, 'The shape of default_value'):
      fc.numeric_column('aaa', shape=[2], default_value=[1, 2, 3.])
      a = fc.numeric_column(
          'aaa', shape=[3, 2], default_value=[[2, 3], [1, 2], [2, 3.]])
      self.assertEqual(((2, 3), (1, 2), (2, 3.)), a.default_value)
    with self.assertRaisesRegex(ValueError, 'The shape of default_value'):
      fc.numeric_column(
          'aaa', shape=[3, 1], default_value=[[2, 3], [1, 2], [2, 3.]])
    with self.assertRaisesRegex(ValueError, 'The shape of default_value'):
      fc.numeric_column(
          'aaa', shape=[3, 3], default_value=[[2, 3], [1, 2], [2, 3.]])

  def test_default_value_type_check(self):
    fc.numeric_column(
        'aaa', shape=[2], default_value=[1, 2.], dtype=dtypes.float32)
    fc.numeric_column(
        'aaa', shape=[2], default_value=[1, 2], dtype=dtypes.int32)
    with self.assertRaisesRegex(TypeError, 'must be compatible with dtype'):
      fc.numeric_column(
          'aaa', shape=[2], default_value=[1, 2.], dtype=dtypes.int32)
    with self.assertRaisesRegex(TypeError,
                                'default_value must be compatible with dtype'):
      fc.numeric_column('aaa', default_value=['string'])

  def test_shape_must_be_positive_integer(self):
    with self.assertRaisesRegex(TypeError, 'shape dimensions must be integer'):
      fc.numeric_column(
          'aaa', shape=[
              1.0,
          ])

    with self.assertRaisesRegex(ValueError,
                                'shape dimensions must be greater than 0'):
      fc.numeric_column(
          'aaa', shape=[
              0,
          ])

  def test_dtype_is_convertible_to_float(self):
    with self.assertRaisesRegex(ValueError,
                                'dtype must be convertible to float'):
      fc.numeric_column('aaa', dtype=dtypes.string)

  def test_scalar_default_value_fills_the_shape(self):
    a = fc.numeric_column('aaa', shape=[2, 3], default_value=2.)
    self.assertEqual(((2., 2., 2.), (2., 2., 2.)), a.default_value)

  def test_parse_spec(self):
    a = fc.numeric_column('aaa', shape=[2, 3], dtype=dtypes.int32)
    self.assertEqual({
        'aaa': parsing_ops.FixedLenFeature((2, 3), dtype=dtypes.int32)
    }, a.parse_example_spec)

  def test_parse_example_no_default_value(self):
    price = fc.numeric_column('price', shape=[2])
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'price':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([price]))
    self.assertIn('price', features)

    self.assertAllEqual([[20., 110.]], self.evaluate(features['price']))

  def test_parse_example_with_default_value(self):
    price = fc.numeric_column('price', shape=[2], default_value=11.)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'price':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.]))
            }))
    no_data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'something_else':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString(),
                    no_data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([price]))
    self.assertIn('price', features)

    self.assertAllEqual([[20., 110.], [11., 11.]],
                        self.evaluate(features['price']))

  def test_normalizer_fn_must_be_callable(self):
    with self.assertRaisesRegex(TypeError, 'must be a callable'):
      fc.numeric_column('price', normalizer_fn='NotACallable')

  def test_normalizer_fn_transform_feature(self):

    def _increment_two(input_tensor):
      return input_tensor + 2.

    price = fc.numeric_column('price', shape=[2], normalizer_fn=_increment_two)
    output = fc._transform_features_v2({
        'price': [[1., 2.], [5., 6.]]
    }, [price], None)

    self.assertAllEqual([[3., 4.], [7., 8.]], self.evaluate(output[price]))

  def test_get_dense_tensor(self):

    def _increment_two(input_tensor):
      return input_tensor + 2.

    price = fc.numeric_column('price', shape=[2], normalizer_fn=_increment_two)
    transformation_cache = fc.FeatureTransformationCache({
        'price': [[1., 2.], [5., 6.]]
    })
    self.assertAllEqual(
        transformation_cache.get(price, None),
        price.get_dense_tensor(transformation_cache, None))

  def test_sparse_tensor_not_supported(self):
    price = fc.numeric_column('price')
    transformation_cache = fc.FeatureTransformationCache({
        'price':
            sparse_tensor.SparseTensor(
                indices=[[0, 0]], values=[0.3], dense_shape=[1, 1])
    })
    with self.assertRaisesRegex(ValueError, 'must be a Tensor'):
      price.transform_feature(transformation_cache, None)

  def test_deep_copy(self):
    a = fc.numeric_column('aaa', shape=[1, 2], default_value=[[3., 2.]])
    a_copy = copy.deepcopy(a)
    self.assertEqual(a_copy.name, 'aaa')
    self.assertEqual(a_copy.shape, (1, 2))
    self.assertEqual(a_copy.default_value, ((3., 2.),))

  def test_numpy_default_value(self):
    a = fc.numeric_column(
        'aaa', shape=[1, 2], default_value=np.array([[3., 2.]]))
    self.assertEqual(a.default_value, ((3., 2.),))

  def test_old_linear_model(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc_old.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[10.], [50.]], self.evaluate(predictions))

  def test_serialization(self):

    def _increment_two(input_tensor):
      return input_tensor + 2.

    price = fc.numeric_column('price', normalizer_fn=_increment_two)
    self.assertEqual(['price'], price.parents)

    config = price.get_config()
    self.assertEqual({
        'key': 'price',
        'shape': (1,),
        'default_value': None,
        'dtype': 'float32',
        'normalizer_fn': '_increment_two'
    }, config)

    new_col = fc.NumericColumn.from_config(
        config, custom_objects={'_increment_two': _increment_two})
    self.assertEqual(price, new_col)
    self.assertEqual(new_col.shape, (1,))

    # Also test round trip through feature column serialization utils.
    new_col = serialization.deserialize_feature_column(
        serialization.serialize_feature_column(price),
        custom_objects={'_increment_two': _increment_two})
    self.assertEqual(price, new_col)


class BucketizedColumnTest(test.TestCase):

  def test_invalid_source_column_type(self):
    a = fc.categorical_column_with_hash_bucket('aaa', hash_bucket_size=10)
    with self.assertRaisesRegex(
        ValueError,
        'source_column must be a column generated with numeric_column'):
      fc.bucketized_column(a, boundaries=[0, 1])

  def test_invalid_source_column_shape(self):
    a = fc.numeric_column('aaa', shape=[2, 3])
    with self.assertRaisesRegex(ValueError,
                                'source_column must be one-dimensional column'):
      fc.bucketized_column(a, boundaries=[0, 1])

  def test_invalid_boundaries(self):
    a = fc.numeric_column('aaa')
    with self.assertRaisesRegex(ValueError, 'boundaries must not be empty'):
      fc.bucketized_column(a, boundaries=None)
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=1.)
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=[1, 0])
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=[1, 1])

  def test_name(self):
    a = fc.numeric_column('aaa', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    self.assertTrue(b._is_v2_column)
    self.assertEqual('aaa_bucketized', b.name)

  def test_is_v2_column_old_numeric(self):
    a = fc_old._numeric_column('aaa', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    self.assertFalse(b._is_v2_column)
    self.assertEqual('aaa_bucketized', b.name)

  def test_parse_spec(self):
    a = fc.numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    self.assertEqual({
        'aaa': parsing_ops.FixedLenFeature((2,), dtype=dtypes.int32)
    }, b.parse_example_spec)

  def test_variable_shape(self):
    a = fc.numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    # Column 'aaa` has shape [2] times three buckets -> variable_shape=[2, 3].
    self.assertAllEqual((2, 3), b.variable_shape)

  def test_num_buckets(self):
    a = fc.numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    # Column 'aaa` has shape [2] times three buckets -> num_buckets=6.
    self.assertEqual(6, b.num_buckets)

  def test_parse_example(self):
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 50])
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'price':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([bucketized_price]))
    self.assertIn('price', features)

    self.assertAllEqual([[20., 110.]], self.evaluate(features['price']))

  def test_transform_feature(self):
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformed_tensor = fc._transform_features_v2({
          'price': [[-1., 1.], [5., 6.]]
      }, [bucketized_price], None)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual([[0, 1], [3, 4]],
                          self.evaluate(transformed_tensor[bucketized_price]))

  def test_get_dense_tensor_one_input_value(self):
    """Tests _get_dense_tensor() for input with shape=[1]."""
    price = fc.numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'price': [[-1.], [1.], [5.], [6.]]
      })

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      bucketized_price_tensor = bucketized_price.get_dense_tensor(
          transformation_cache, None)
      self.assertAllClose(
          # One-hot tensor.
          [[[1., 0., 0., 0., 0.]], [[0., 1., 0., 0., 0.]],
           [[0., 0., 0., 1., 0.]], [[0., 0., 0., 0., 1.]]],
          self.evaluate(bucketized_price_tensor))

  def test_get_dense_tensor_two_input_values(self):
    """Tests _get_dense_tensor() for input with shape=[2]."""
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'price': [[-1., 1.], [5., 6.]]
      })

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      bucketized_price_tensor = bucketized_price.get_dense_tensor(
          transformation_cache, None)
      self.assertAllClose(
          # One-hot tensor.
          [[[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]],
           [[0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]]],
          self.evaluate(bucketized_price_tensor))

  def test_get_sparse_tensors_one_input_value(self):
    """Tests _get_sparse_tensors() for input with shape=[1]."""
    price = fc.numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'price': [[-1.], [1.], [5.], [6.]]
      })
      with _initialized_session() as sess:
        id_weight_pair = bucketized_price.get_sparse_tensors(
            transformation_cache, None)
        self.assertIsNone(id_weight_pair.weight_tensor)
        id_tensor_value = sess.run(id_weight_pair.id_tensor)
        self.assertAllEqual([[0, 0], [1, 0], [2, 0], [3, 0]],
                            id_tensor_value.indices)
        self.assertAllEqual([0, 1, 3, 4], id_tensor_value.values)
        self.assertAllEqual([4, 1], id_tensor_value.dense_shape)

  def test_get_sparse_tensors_two_input_values(self):
    """Tests _get_sparse_tensors() for input with shape=[2]."""
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'price': [[-1., 1.], [5., 6.]]
      })
      with _initialized_session() as sess:
        id_weight_pair = bucketized_price.get_sparse_tensors(
            transformation_cache, None)
        self.assertIsNone(id_weight_pair.weight_tensor)
        id_tensor_value = sess.run(id_weight_pair.id_tensor)
        self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1]],
                            id_tensor_value.indices)
        # Values 0-4 correspond to the first column of the input price.
        # Values 5-9 correspond to the second column of the input price.
        self.assertAllEqual([0, 6, 3, 9], id_tensor_value.values)
        self.assertAllEqual([2, 2], id_tensor_value.dense_shape)

  def test_sparse_tensor_input_not_supported(self):
    price = fc.numeric_column('price')
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 1])
    transformation_cache = fc.FeatureTransformationCache({
        'price':
            sparse_tensor.SparseTensor(
                indices=[[0, 0]], values=[0.3], dense_shape=[1, 1])
    })
    with self.assertRaisesRegex(ValueError, 'must be a Tensor'):
      bucketized_price.transform_feature(transformation_cache, None)

  def test_deep_copy(self):
    a = fc.numeric_column('aaa', shape=[2])
    a_bucketized = fc.bucketized_column(a, boundaries=[0, 1])
    a_bucketized_copy = copy.deepcopy(a_bucketized)
    self.assertEqual(a_bucketized_copy.name, 'aaa_bucketized')
    self.assertAllEqual(a_bucketized_copy.variable_shape, (2, 3))
    self.assertEqual(a_bucketized_copy.boundaries, (0, 1))

  def test_old_linear_model_one_input_value(self):
    """Tests linear_model() for input with shape=[1]."""
    price = fc.numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1.], [1.], [5.], [6.]]}
      predictions = fc_old.linear_model(features, [bucketized_price])
      bias = get_linear_model_bias()
      bucketized_price_var = get_linear_model_column_var(bucketized_price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        # One weight variable per bucket, all initialized to zero.
        self.assertAllClose([[0.], [0.], [0.], [0.], [0.]],
                            self.evaluate(bucketized_price_var))
        self.assertAllClose([[0.], [0.], [0.], [0.]],
                            self.evaluate(predictions))
        sess.run(
            bucketized_price_var.assign([[10.], [20.], [30.], [40.], [50.]]))
        # price -1. is in the 0th bucket, whose weight is 10.
        # price 1. is in the 1st bucket, whose weight is 20.
        # price 5. is in the 3rd bucket, whose weight is 40.
        # price 6. is in the 4th bucket, whose weight is 50.
        self.assertAllClose([[10.], [20.], [40.], [50.]],
                            self.evaluate(predictions))
        sess.run(bias.assign([1.]))
        self.assertAllClose([[11.], [21.], [41.], [51.]],
                            self.evaluate(predictions))

  def test_old_linear_model_two_input_values(self):
    """Tests linear_model() for input with shape=[2]."""
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1., 1.], [5., 6.]]}
      predictions = fc_old.linear_model(features, [bucketized_price])
      bias = get_linear_model_bias()
      bucketized_price_var = get_linear_model_column_var(bucketized_price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        # One weight per bucket per input column, all initialized to zero.
        self.assertAllClose(
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
            self.evaluate(bucketized_price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(
            bucketized_price_var.assign([[10.], [20.], [30.], [40.], [50.],
                                         [60.], [70.], [80.], [90.], [100.]]))
        # 1st example:
        #   price -1. is in the 0th bucket, whose weight is 10.
        #   price 1. is in the 6th bucket, whose weight is 70.
        # 2nd example:
        #   price 5. is in the 3rd bucket, whose weight is 40.
        #   price 6. is in the 9th bucket, whose weight is 100.
        self.assertAllClose([[80.], [140.]], self.evaluate(predictions))
        sess.run(bias.assign([1.]))
        self.assertAllClose([[81.], [141.]], self.evaluate(predictions))

  def test_old_linear_model_one_input_value_old_numeric(self):
    """Tests linear_model() for input with shape=[1]."""
    price = fc_old._numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1.], [1.], [5.], [6.]]}
      predictions = fc_old.linear_model(features, [bucketized_price])
      bias = get_linear_model_bias()
      bucketized_price_var = get_linear_model_column_var(bucketized_price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        # One weight variable per bucket, all initialized to zero.
        self.assertAllClose([[0.], [0.], [0.], [0.], [0.]],
                            self.evaluate(bucketized_price_var))
        self.assertAllClose([[0.], [0.], [0.], [0.]],
                            self.evaluate(predictions))
        sess.run(
            bucketized_price_var.assign([[10.], [20.], [30.], [40.], [50.]]))
        # price -1. is in the 0th bucket, whose weight is 10.
        # price 1. is in the 1st bucket, whose weight is 20.
        # price 5. is in the 3rd bucket, whose weight is 40.
        # price 6. is in the 4th bucket, whose weight is 50.
        self.assertAllClose([[10.], [20.], [40.], [50.]],
                            self.evaluate(predictions))
        sess.run(bias.assign([1.]))
        self.assertAllClose([[11.], [21.], [41.], [51.]],
                            self.evaluate(predictions))

  def test_serialization(self):
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    self.assertEqual([price], bucketized_price.parents)

    config = bucketized_price.get_config()
    self.assertEqual({
        'source_column': {
            'class_name': 'NumericColumn',
            'config': {
                'key': 'price',
                'shape': (2,),
                'default_value': None,
                'dtype': 'float32',
                'normalizer_fn': None
            }
        },
        'boundaries': (0, 2, 4, 6)
    }, config)

    new_bucketized_price = fc.BucketizedColumn.from_config(config)
    self.assertEqual(bucketized_price, new_bucketized_price)
    self.assertIsNot(price, new_bucketized_price.source_column)

    new_bucketized_price = fc.BucketizedColumn.from_config(
        config,
        columns_by_name={
            serialization._column_name_with_class_name(price): price
        })
    self.assertEqual(bucketized_price, new_bucketized_price)
    self.assertIs(price, new_bucketized_price.source_column)


class HashedCategoricalColumnTest(test.TestCase):

  def test_defaults(self):
    a = fc.categorical_column_with_hash_bucket('aaa', 10)
    self.assertEqual('aaa', a.name)
    self.assertEqual('aaa', a.key)
    self.assertEqual(10, a.hash_bucket_size)
    self.assertEqual(dtypes.string, a.dtype)
    self.assertTrue(a._is_v2_column)

  def test_key_should_be_string(self):
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.categorical_column_with_hash_bucket(('key',), 10)

  def test_bucket_size_should_be_given(self):
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be set.'):
      fc.categorical_column_with_hash_bucket('aaa', None)

  def test_bucket_size_should_be_positive(self):
    with self.assertRaisesRegex(ValueError,
                                'hash_bucket_size must be at least 1'):
      fc.categorical_column_with_hash_bucket('aaa', 0)

  def test_dtype_should_be_string_or_integer(self):
    a = fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.string)
    b = fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.int32)
    self.assertEqual(dtypes.string, a.dtype)
    self.assertEqual(dtypes.int32, b.dtype)

    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.float32)

  def test_deep_copy(self):
    original = fc.categorical_column_with_hash_bucket('aaa', 10)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(10, column.hash_bucket_size)
      self.assertEqual(10, column.num_buckets)
      self.assertEqual(dtypes.string, column.dtype)

  def test_parse_spec_string(self):
    a = fc.categorical_column_with_hash_bucket('aaa', 10)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, a.parse_example_spec)

  def test_parse_spec_int(self):
    a = fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.int32)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, a.parse_example_spec)

  def test_parse_example(self):
    a = fc.categorical_column_with_hash_bucket('aaa', 10)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_strings_should_be_hashed(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    outputs = fc._transform_features_v2({
        'wire': wire_tensor
    }, [hashed_sparse], None)
    output = outputs[hashed_sparse]
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [6, 4, 1]

    self.assertEqual(dtypes.int64, output.values.dtype)
    self.assertAllEqual(expected_values, self.evaluate(output.values))
    self.assertAllEqual(
        self.evaluate(wire_tensor.indices), self.evaluate(output.indices))
    self.assertAllEqual(
        self.evaluate(wire_tensor.dense_shape),
        self.evaluate(output.dense_shape))

  def test_tensor_dtype_should_be_string_or_integer(self):
    string_fc = fc.categorical_column_with_hash_bucket(
        'a_string', 10, dtype=dtypes.string)
    int_fc = fc.categorical_column_with_hash_bucket(
        'a_int', 10, dtype=dtypes.int32)
    float_fc = fc.categorical_column_with_hash_bucket(
        'a_float', 10, dtype=dtypes.string)
    int_tensor = sparse_tensor.SparseTensor(
        values=[101], indices=[[0, 0]], dense_shape=[1, 1])
    string_tensor = sparse_tensor.SparseTensor(
        values=['101'], indices=[[0, 0]], dense_shape=[1, 1])
    float_tensor = sparse_tensor.SparseTensor(
        values=[101.], indices=[[0, 0]], dense_shape=[1, 1])
    transformation_cache = fc.FeatureTransformationCache({
        'a_int': int_tensor,
        'a_string': string_tensor,
        'a_float': float_tensor
    })
    transformation_cache.get(string_fc, None)
    transformation_cache.get(int_fc, None)
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      transformation_cache.get(float_fc, None)

  def test_dtype_should_match_with_tensor(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
    transformation_cache = fc.FeatureTransformationCache({'wire': wire_tensor})
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      transformation_cache.get(hashed_sparse, None)

  def test_ints_should_be_hashed(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=[101, 201, 301],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    transformation_cache = fc.FeatureTransformationCache({'wire': wire_tensor})
    output = transformation_cache.get(hashed_sparse, None)
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [3, 7, 5]

    self.assertAllEqual(expected_values, self.evaluate(output.values))

  def test_int32_64_is_compatible(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=constant_op.constant([101, 201, 301], dtype=dtypes.int32),
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    transformation_cache = fc.FeatureTransformationCache({'wire': wire_tensor})
    output = transformation_cache.get(hashed_sparse, None)
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [3, 7, 5]

    self.assertAllEqual(expected_values, self.evaluate(output.values))

  def test_get_sparse_tensors(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    transformation_cache = fc.FeatureTransformationCache({
        'wire':
            sparse_tensor.SparseTensor(
                values=['omar', 'stringer', 'marlo'],
                indices=[[0, 0], [1, 0], [1, 1]],
                dense_shape=[2, 2])
    })
    id_weight_pair = hashed_sparse.get_sparse_tensors(transformation_cache,
                                                      None)
    self.assertIsNone(id_weight_pair.weight_tensor)
    self.assertEqual(
        transformation_cache.get(hashed_sparse, None), id_weight_pair.id_tensor)

  def test_get_sparse_tensors_dense_input(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    transformation_cache = fc.FeatureTransformationCache({
        'wire': (('omar', ''), ('stringer', 'marlo'))
    })
    id_weight_pair = hashed_sparse.get_sparse_tensors(transformation_cache,
                                                      None)
    self.assertIsNone(id_weight_pair.weight_tensor)
    self.assertEqual(
        transformation_cache.get(hashed_sparse, None), id_weight_pair.id_tensor)

  def test_old_linear_model(self):
    wire_column = fc.categorical_column_with_hash_bucket('wire', 4)
    self.assertEqual(4, wire_column.num_buckets)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          wire_column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), self.evaluate(wire_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
      # 'marlo' -> 3: wire_var[3] = 4
      # 'skywalker' -> 2, 'omar' -> 2: wire_var[2] + wire_var[2] = 3+3 = 6
      self.assertAllClose(((4.,), (6.,)), self.evaluate(predictions))

  def test_serialization(self):
    wire_column = fc.categorical_column_with_hash_bucket('wire', 4)
    self.assertEqual(['wire'], wire_column.parents)

    config = wire_column.get_config()
    self.assertEqual({
        'key': 'wire',
        'hash_bucket_size': 4,
        'dtype': 'string'
    }, config)

    self.assertEqual(wire_column,
                     fc.HashedCategoricalColumn.from_config(config))


class CrossedColumnTest(test.TestCase):

  def test_keys_empty(self):
    with self.assertRaisesRegex(ValueError,
                                'keys must be a list with length > 1'):
      fc.crossed_column([], 10)

  def test_keys_length_one(self):
    with self.assertRaisesRegex(ValueError,
                                'keys must be a list with length > 1'):
      fc.crossed_column(['a'], 10)

  def test_key_type_unsupported(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported key type'):
      fc.crossed_column(['a', fc.numeric_column('c')], 10)

    with self.assertRaisesRegex(
        ValueError, 'categorical_column_with_hash_bucket is not supported'):
      fc.crossed_column(
          ['a', fc.categorical_column_with_hash_bucket('c', 10)], 10)

  def test_hash_bucket_size_negative(self):
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be > 1'):
      fc.crossed_column(['a', 'c'], -1)

  def test_hash_bucket_size_zero(self):
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be > 1'):
      fc.crossed_column(['a', 'c'], 0)

  def test_hash_bucket_size_none(self):
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be > 1'):
      fc.crossed_column(['a', 'c'], None)

  def test_name(self):
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    self.assertTrue(crossed1._is_v2_column)

    crossed2 = fc.crossed_column([b, 'c', crossed1], 10)
    self.assertTrue(crossed2._is_v2_column)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_is_v2_column(self):
    a = fc_old._numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    self.assertTrue(crossed1._is_v2_column)

    crossed2 = fc.crossed_column([b, 'c', crossed1], 10)
    self.assertFalse(crossed2._is_v2_column)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_name_ordered_alphabetically(self):
    """Tests that the name does not depend on the order of given columns."""
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)

    crossed2 = fc.crossed_column([crossed1, 'c', b], 10)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_name_leaf_keys_ordered_alphabetically(self):
    """Tests that the name does not depend on the order of given columns."""
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d2', 'c'], 10)

    crossed2 = fc.crossed_column([crossed1, 'd1', b], 10)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_parse_spec(self):
    a = fc.numeric_column('a', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed = fc.crossed_column([b, 'c'], 10)
    self.assertEqual({
        'a': parsing_ops.FixedLenFeature((2,), dtype=dtypes.int32),
        'c': parsing_ops.VarLenFeature(dtypes.string),
    }, crossed.parse_example_spec)

  def test_num_buckets(self):
    a = fc.numeric_column('a', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed = fc.crossed_column([b, 'c'], 15)
    self.assertEqual(15, crossed.num_buckets)

  def test_deep_copy(self):
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    crossed2 = fc.crossed_column([b, 'c', crossed1], 15, hash_key=5)
    crossed2_copy = copy.deepcopy(crossed2)
    self.assertEqual(
        'a_bucketized_X_c_X_d1_X_d2',
        crossed2_copy.name,
    )
    self.assertEqual(15, crossed2_copy.hash_bucket_size)
    self.assertEqual(5, crossed2_copy.hash_key)

  def test_parse_example(self):
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 50])
    price_cross_wire = fc.crossed_column([bucketized_price, 'wire'], 10)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'price':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.])),
                'wire':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer'])),
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([price_cross_wire]))
    self.assertIn('price', features)
    self.assertIn('wire', features)

    self.assertAllEqual([[20., 110.]], self.evaluate(features['price']))
    wire_sparse = features['wire']
    self.assertAllEqual([[0, 0], [0, 1]], self.evaluate(wire_sparse.indices))
    # Use byte constants to pass the open-source test.
    self.assertAllEqual([b'omar', b'stringer'],
                        self.evaluate(wire_sparse.values))
    self.assertAllEqual([1, 2], self.evaluate(wire_sparse.dense_shape))

  def test_transform_feature(self):
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 50])
    hash_bucket_size = 10
    price_cross_wire = fc.crossed_column([bucketized_price, 'wire'],
                                         hash_bucket_size)
    features = {
        'price':
            constant_op.constant([[1., 2.], [5., 6.]]),
        'wire':
            sparse_tensor.SparseTensor(
                values=['omar', 'stringer', 'marlo'],
                indices=[[0, 0], [1, 0], [1, 1]],
                dense_shape=[2, 2]),
    }
    outputs = fc._transform_features_v2(features, [price_cross_wire], None)
    output = outputs[price_cross_wire]
    output_val = self.evaluate(output)
    self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]],
                        output_val.indices)
    for val in output_val.values:
      self.assertIn(val, list(range(hash_bucket_size)))
    self.assertAllEqual([2, 4], output_val.dense_shape)

  def test_get_sparse_tensors(self):
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    crossed2 = fc.crossed_column([b, 'c', crossed1], 15, hash_key=5)
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'a':
              constant_op.constant(((-1., .5), (.5, 1.))),
          'c':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
          'd1':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['d1A', 'd1B', 'd1C'],
                  dense_shape=(2, 2)),
          'd2':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['d2A', 'd2B', 'd2C'],
                  dense_shape=(2, 2)),
      })
      id_weight_pair = crossed2.get_sparse_tensors(transformation_cache, None)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      id_tensor_eval = self.evaluate(id_weight_pair.id_tensor)
      self.assertAllEqual(
          ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
           (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
           (1, 14), (1, 15)), id_tensor_eval.indices)
      # Check exact hashed output. If hashing changes this test will break.
      # All values are within [0, hash_bucket_size).
      expected_values = (6, 14, 0, 13, 8, 8, 10, 12, 2, 0, 1, 9, 8, 12, 2, 0,
                         10, 11)
      self.assertAllEqual(expected_values, id_tensor_eval.values)
      self.assertAllEqual((2, 16), id_tensor_eval.dense_shape)

  def test_get_sparse_tensors_simple(self):
    """Same as test_get_sparse_tensors, but with simpler values."""
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'a':
              constant_op.constant(((-1., .5), (.5, 1.))),
          'c':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      })
      id_weight_pair = crossed.get_sparse_tensors(transformation_cache, None)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      id_tensor_eval = self.evaluate(id_weight_pair.id_tensor)
      self.assertAllEqual(((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)),
                          id_tensor_eval.indices)
      # Check exact hashed output. If hashing changes this test will break.
      # All values are within [0, hash_bucket_size).
      expected_values = (1, 0, 1, 3, 4, 2)
      self.assertAllEqual(expected_values, id_tensor_eval.values)
      self.assertAllEqual((2, 4), id_tensor_eval.dense_shape)

  def test_old_linear_model(self):
    """Tests linear_model.

    Uses data from test_get_sparse_tensors_simple.
    """
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          'a':
              constant_op.constant(((-1., .5), (.5, 1.))),
          'c':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      }, (crossed,))
      bias = get_linear_model_bias()
      crossed_var = get_linear_model_column_var(crossed)
      with _initialized_session() as sess:
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(crossed_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        sess.run(crossed_var.assign(((1.,), (2.,), (3.,), (4.,), (5.,))))
        # Expected ids after cross = (1, 0, 1, 3, 4, 2)
        self.assertAllClose(((3.,), (14.,)), self.evaluate(predictions))
        sess.run(bias.assign((.1,)))
        self.assertAllClose(((3.1,), (14.1,)), self.evaluate(predictions))

  def test_old_linear_model_with_weights(self):

    class _TestColumnWithWeights(BaseFeatureColumnForTests,
                                 fc.CategoricalColumn,
                                 fc_old._CategoricalColumn):
      """Produces sparse IDs and sparse weights."""

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'test_column'

      @property
      def parse_example_spec(self):
        return {
            self.name:
                parsing_ops.VarLenFeature(dtypes.int32),
            '{}_weights'.format(self.name):
                parsing_ops.VarLenFeature(dtypes.float32),
        }

      @property
      def _parse_example_spec(self):
        return self.parse_example_spec

      @property
      def num_buckets(self):
        return 5

      @property
      def _num_buckets(self):
        return self.num_buckets

      def transform_feature(self, transformation_cache, state_manager):
        raise ValueError('Should not be called.')

      def _transform_feature(self, inputs):
        return (inputs.get(self.name),
                inputs.get('{}_weights'.format(self.name)))

      def get_sparse_tensors(self, transformation_cache, state_manager):
        raise ValueError('Should not be called.')

      def _get_sparse_tensors(self,
                              inputs,
                              weight_collections=None,
                              trainable=None):
        """Populates both id_tensor and weight_tensor."""
        ids_and_weights = inputs.get(self)
        return fc.CategoricalColumn.IdWeightPair(
            id_tensor=ids_and_weights[0], weight_tensor=ids_and_weights[1])

    t = _TestColumnWithWeights()
    crossed = fc.crossed_column([t, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(
          ValueError,
          'crossed_column does not support weight_tensor.*{}'.format(t.name)):
        fc_old.linear_model({
            t.name:
                sparse_tensor.SparseTensor(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=[0, 1, 2],
                    dense_shape=(2, 2)),
            '{}_weights'.format(t.name):
                sparse_tensor.SparseTensor(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=[1., 10., 2.],
                    dense_shape=(2, 2)),
            'c':
                sparse_tensor.SparseTensor(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=['cA', 'cB', 'cC'],
                    dense_shape=(2, 2)),
        }, (crossed,))

  def test_old_linear_model_old_numeric(self):
    """Tests linear_model.

    Uses data from test_get_sparse_tensors_simple.
    """
    a = fc_old._numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          'a':
              constant_op.constant(((-1., .5), (.5, 1.))),
          'c':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      }, (crossed,))
      bias = get_linear_model_bias()
      crossed_var = get_linear_model_column_var(crossed)
      with _initialized_session() as sess:
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(crossed_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        sess.run(crossed_var.assign(((1.,), (2.,), (3.,), (4.,), (5.,))))
        # Expected ids after cross = (1, 0, 1, 3, 4, 2)
        self.assertAllClose(((3.,), (14.,)), self.evaluate(predictions))
        sess.run(bias.assign((.1,)))
        self.assertAllClose(((3.1,), (14.1,)), self.evaluate(predictions))

  def test_serialization(self):
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)

    self.assertEqual([b, 'c'], crossed.parents)

    config = crossed.get_config()
    self.assertEqual({
        'hash_bucket_size':
            5,
        'hash_key':
            5,
        'keys': ({
            'config': {
                'boundaries': (0, 1),
                'source_column': {
                    'config': {
                        'dtype': 'int32',
                        'default_value': None,
                        'key': 'a',
                        'normalizer_fn': None,
                        'shape': (2,)
                    },
                    'class_name': 'NumericColumn'
                }
            },
            'class_name': 'BucketizedColumn'
        }, 'c')
    }, config)

    new_crossed = fc.CrossedColumn.from_config(config)
    self.assertEqual(crossed, new_crossed)
    self.assertIsNot(b, new_crossed.keys[0])

    new_crossed = fc.CrossedColumn.from_config(
        config,
        columns_by_name={serialization._column_name_with_class_name(b): b})
    self.assertEqual(crossed, new_crossed)
    self.assertIs(b, new_crossed.keys[0])


class OldLinearModelTest(test.TestCase):

  def test_raises_if_empty_feature_columns(self):
    with self.assertRaisesRegex(ValueError,
                                'feature_columns must not be empty'):
      fc_old.linear_model(features={}, feature_columns=[])

  def test_should_be_feature_column(self):
    with self.assertRaisesRegex(ValueError, 'must be a _FeatureColumn'):
      fc_old.linear_model(features={'a': [[0]]}, feature_columns='NotSupported')

  def test_should_be_dense_or_categorical_column(self):

    class NotSupportedColumn(
        BaseFeatureColumnForTests,
        feature_column_v2_types.FeatureColumn,
        fc_old._FeatureColumn,
    ):

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'NotSupportedColumn'

      def transform_feature(self, transformation_cache, state_manager):
        pass

      def _transform_feature(self, inputs):
        pass

      @property
      def parse_example_spec(self):
        pass

      @property
      def _parse_example_spec(self):
        pass

    with self.assertRaisesRegex(
        ValueError, 'must be either a _DenseColumn or _CategoricalColumn'):
      fc_old.linear_model(
          features={'a': [[0]]}, feature_columns=[NotSupportedColumn()])

  def test_does_not_support_dict_columns(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      fc_old.linear_model(
          features={'a': [[0]]}, feature_columns={'a': fc.numeric_column('a')})

  def test_raises_if_duplicate_name(self):
    with self.assertRaisesRegex(
        ValueError, 'Duplicate feature column name found for columns'):
      fc_old.linear_model(
          features={'a': [[0]]},
          feature_columns=[fc.numeric_column('a'),
                           fc.numeric_column('a')])

  def test_dense_bias(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc_old.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        sess.run(price_var.assign([[10.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[15.], [55.]], self.evaluate(predictions))

  def test_sparse_bias(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc_old.linear_model(features, [wire_cast])
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.], [0.], [0.]],
                            self.evaluate(wire_cast_var))
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], self.evaluate(predictions))

  def test_dense_and_sparse_bias(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor, 'price': [[1.], [5.]]}
      predictions = fc_old.linear_model(features, [wire_cast, price])
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[1015.], [10065.]], self.evaluate(predictions))

  def test_dense_and_sparse_column(self):
    """When the column is both dense and sparse, uses sparse tensors."""

    class _DenseAndSparseColumn(BaseFeatureColumnForTests, fc.DenseColumn,
                                fc.CategoricalColumn, fc_old._DenseColumn,
                                fc_old._CategoricalColumn):

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'dense_and_sparse_column'

      @property
      def parse_example_spec(self):
        return {self.name: parsing_ops.VarLenFeature(self.dtype)}

      @property
      def _parse_example_spec(self):
        return self.parse_example_spec

      def transform_feature(self, transformation_cache, state_manager):
        raise ValueError('Should not use this method.')

      def _transform_feature(self, inputs):
        return inputs.get(self.name)

      @property
      def variable_shape(self):
        return self.variable_shape

      @property
      def _variable_shape(self):
        return self.variable_shape

      def get_dense_tensor(self, transformation_cache, state_manager):
        raise ValueError('Should not use this method.')

      def _get_dense_tensor(self, inputs):
        raise ValueError('Should not use this method.')

      @property
      def num_buckets(self):
        return 4

      @property
      def _num_buckets(self):
        return self.num_buckets

      def get_sparse_tensors(self, transformation_cache, state_manager):
        raise ValueError('Should not use this method.')

      def _get_sparse_tensors(self,
                              inputs,
                              weight_collections=None,
                              trainable=None):
        sp_tensor = sparse_tensor.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1]],
            values=[2, 0, 3],
            dense_shape=[2, 2])
        return fc.CategoricalColumn.IdWeightPair(sp_tensor, None)

    dense_and_sparse_column = _DenseAndSparseColumn()
    with ops.Graph().as_default():
      sp_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {dense_and_sparse_column.name: sp_tensor}
      predictions = fc_old.linear_model(features, [dense_and_sparse_column])
      bias = get_linear_model_bias()
      dense_and_sparse_column_var = get_linear_model_column_var(
          dense_and_sparse_column)
      with _initialized_session() as sess:
        sess.run(
            dense_and_sparse_column_var.assign([[10.], [100.], [1000.],
                                                [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], self.evaluate(predictions))

  def test_dense_multi_output(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc_old.linear_model(features, [price], units=3)
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((1, 3)), self.evaluate(price_var))
        sess.run(price_var.assign([[10., 100., 1000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[15., 106., 1007.], [55., 506., 5007.]],
                            self.evaluate(predictions))

  def test_sparse_multi_output(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc_old.linear_model(features, [wire_cast], units=3)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((4, 3)), self.evaluate(wire_cast_var))
        sess.run(
            wire_cast_var.assign([[10., 11., 12.], [100., 110., 120.],
                                  [1000., 1100., 1200.],
                                  [10000., 11000., 12000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[1005., 1106., 1207.], [10015., 11017., 12019.]],
                            self.evaluate(predictions))

  def test_dense_multi_dimension(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = fc_old.linear_model(features, [price])
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([[0.], [0.]], self.evaluate(price_var))
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], self.evaluate(predictions))

  def test_sparse_multi_rank(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = array_ops.sparse_placeholder(dtypes.string)
      wire_value = sparse_tensor.SparseTensorValue(
          values=['omar', 'stringer', 'marlo', 'omar'],  # hashed = [2, 0, 3, 2]
          indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1]],
          dense_shape=[2, 2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc_old.linear_model(features, [wire_cast])
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((4, 1)), self.evaluate(wire_cast_var))
        self.assertAllClose(
            np.zeros((2, 1)),
            predictions.eval(feed_dict={wire_tensor: wire_value}))
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        self.assertAllClose(
            [[1010.], [11000.]],
            predictions.eval(feed_dict={wire_tensor: wire_value}))

  def test_sparse_combiner(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc_old.linear_model(
          features, [wire_cast], sparse_combiner='mean')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [5010.]], self.evaluate(predictions))

  def test_sparse_combiner_with_negative_weights(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    wire_cast_weights = fc.weighted_categorical_column(wire_cast, 'weights')

    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {
          'wire_cast': wire_tensor,
          'weights': constant_op.constant([[1., 1., -1.0]])
      }
      predictions = fc_old.linear_model(
          features, [wire_cast_weights], sparse_combiner='sum')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [-9985.]], self.evaluate(predictions))

  def test_dense_multi_dimension_multi_output(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = fc_old.linear_model(features, [price], units=3)
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((2, 3)), self.evaluate(price_var))
        sess.run(price_var.assign([[1., 2., 3.], [10., 100., 1000.]]))
        sess.run(bias.assign([2., 3., 4.]))
        self.assertAllClose([[23., 205., 2007.], [67., 613., 6019.]],
                            self.evaluate(predictions))

  def test_raises_if_shape_mismatch(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegex(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        fc_old.linear_model(features, [price])

  def test_dense_reshaping(self):
    price = fc.numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      predictions = fc_old.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], self.evaluate(predictions))

  def test_dense_multi_column(self):
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      predictions = fc_old.linear_model(features, [price1, price2])
      bias = get_linear_model_bias()
      price1_var = get_linear_model_column_var(price1)
      price2_var = get_linear_model_column_var(price2)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.]], self.evaluate(price1_var))
        self.assertAllClose([[0.]], self.evaluate(price2_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price1_var.assign([[10.], [100.]]))
        sess.run(price2_var.assign([[1000.]]))
        sess.run(bias.assign([7.]))
        self.assertAllClose([[3217.], [4657.]], self.evaluate(predictions))

  def test_fills_cols_to_vars(self):
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      cols_to_vars = {}
      fc_old.linear_model(features, [price1, price2], cols_to_vars=cols_to_vars)
      bias = get_linear_model_bias()
      price1_var = get_linear_model_column_var(price1)
      price2_var = get_linear_model_column_var(price2)
      self.assertEqual(cols_to_vars['bias'], [bias])
      self.assertEqual(cols_to_vars[price1], [price1_var])
      self.assertEqual(cols_to_vars[price2], [price2_var])

  def test_fills_cols_to_vars_partitioned_variables(self):
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2', shape=3)
    with ops.Graph().as_default():
      features = {
          'price1': [[1., 2.], [6., 7.]],
          'price2': [[3., 4., 5.], [8., 9., 10.]]
      }
      cols_to_vars = {}
      with variable_scope.variable_scope(
          'linear',
          partitioner=partitioned_variables.fixed_size_partitioner(2, axis=0)):
        fc_old.linear_model(
            features, [price1, price2], cols_to_vars=cols_to_vars)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertEqual([0.], self.evaluate(cols_to_vars['bias'][0]))
      # Partitioning shards the [2, 1] price1 var into 2 [1, 1] Variables.
      self.assertAllEqual([[0.]], self.evaluate(cols_to_vars[price1][0]))
      self.assertAllEqual([[0.]], self.evaluate(cols_to_vars[price1][1]))
      # Partitioning shards the [3, 1] price2 var into a [2, 1] Variable and
      # a [1, 1] Variable.
      self.assertAllEqual([[0.], [0.]], self.evaluate(cols_to_vars[price2][0]))
      self.assertAllEqual([[0.]], self.evaluate(cols_to_vars[price2][1]))

  def test_fills_cols_to_output_tensors(self):
    # Provide three _DenseColumn's to input_layer: a _NumericColumn, a
    # _BucketizedColumn, and an _EmbeddingColumn.  Only the _EmbeddingColumn
    # creates a Variable.
    apple_numeric_column = fc.numeric_column('apple_numeric_column')
    banana_dense_feature = fc.numeric_column('banana_dense_feature')
    banana_dense_feature_bucketized = fc.bucketized_column(
        banana_dense_feature, boundaries=[0.])
    cherry_sparse_column = fc.categorical_column_with_hash_bucket(
        'cherry_sparse_feature', hash_bucket_size=5)
    dragonfruit_embedding_column = fc.embedding_column(
        cherry_sparse_column, dimension=10)
    with ops.Graph().as_default():
      features = {
          'apple_numeric_column': [[3.], [4.]],
          'banana_dense_feature': [[-1.], [4.]],
          'cherry_sparse_feature': [['a'], ['x']],
      }
      cols_to_output_tensors = {}
      all_cols = [
          apple_numeric_column, banana_dense_feature_bucketized,
          dragonfruit_embedding_column
      ]
      input_layer = fc_old.input_layer(
          features, all_cols, cols_to_output_tensors=cols_to_output_tensors)

      # We check the mapping by checking that we have the right keys,
      # and that the values (output_tensors) were indeed the ones used to
      # form the input layer.
      self.assertCountEqual(all_cols, cols_to_output_tensors.keys())
      input_layer_inputs = [tensor for tensor in input_layer.op.inputs[:-1]]
      output_tensors = [tensor for tensor in cols_to_output_tensors.values()]
      self.assertCountEqual(input_layer_inputs, output_tensors)

  def test_dense_collection(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc_old.linear_model(features, [price], weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      self.assertIn(bias, my_vars)
      self.assertIn(price_var, my_vars)

  def test_sparse_collection(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc_old.linear_model(features, [wire_cast], weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, my_vars)
      self.assertIn(wire_cast_var, my_vars)

  def test_dense_trainable_default(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc_old.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertIn(bias, trainable_vars)
      self.assertIn(price_var, trainable_vars)

  def test_sparse_trainable_default(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc_old.linear_model(features, [wire_cast])
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, trainable_vars)
      self.assertIn(wire_cast_var, trainable_vars)

  def test_dense_trainable_false(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc_old.linear_model(features, [price], trainable=False)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_sparse_trainable_false(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc_old.linear_model(features, [wire_cast], trainable=False)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_column_order(self):
    price_a = fc.numeric_column('price_a')
    price_b = fc.numeric_column('price_b')
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
          'wire_cast':
              sparse_tensor.SparseTensor(
                  values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      }
      fc_old.linear_model(
          features, [price_a, wire_cast, price_b],
          weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      self.assertIn('price_a', my_vars[0].name)
      self.assertIn('price_b', my_vars[1].name)
      self.assertIn('wire_cast', my_vars[2].name)

    with ops.Graph().as_default() as g:
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
          'wire_cast':
              sparse_tensor.SparseTensor(
                  values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      }
      fc_old.linear_model(
          features, [wire_cast, price_b, price_a],
          weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      self.assertIn('price_a', my_vars[0].name)
      self.assertIn('price_b', my_vars[1].name)
      self.assertIn('wire_cast', my_vars[2].name)

  def test_static_batch_size_mismatch(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
    with self.assertRaisesRegex(
        ValueError,
        r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
      fc_old.linear_model(features, [price1, price2])

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
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc_old.linear_model(features, [price1, price2, price3])

  def test_runtime_batch_size_mismatch(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      predictions = fc_old.linear_model(features, [price1, price2])
      with _initialized_session() as sess:
        with self.assertRaisesRegex(errors.OpError,
                                    'must have the same size and shape'):
          sess.run(
              predictions, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
          'price2': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
      }
      predictions = fc_old.linear_model(features, [price1, price2])
      with _initialized_session() as sess:
        sess.run(
            predictions,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  def test_with_1d_sparse_tensor(self):
    # This test does direct variable scope access + manipulations
    # that require entering a legacy graph
    with ops.Graph().as_default():
      price = fc.numeric_column('price')
      price_buckets = fc.bucketized_column(
          price, boundaries=[
              0.,
              10.,
              100.,
          ])
      body_style = fc.categorical_column_with_vocabulary_list(
          'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])

      # Provides 1-dim tensor and dense tensor.
      features = {
          'price':
              constant_op.constant([
                  -1.,
                  12.,
              ]),
          'body-style':
              sparse_tensor.SparseTensor(
                  indices=((0,), (1,)),
                  values=('sedan', 'hardtop'),
                  dense_shape=(2,)),
      }
      self.assertEqual(1, features['price'].shape.ndims)
      self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])

      net = fc_old.linear_model(features, [price_buckets, body_style])
      with _initialized_session() as sess:
        bias = get_linear_model_bias()
        price_buckets_var = get_linear_model_column_var(price_buckets)
        body_style_var = get_linear_model_column_var(body_style)

        sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
        sess.run(bias.assign([5.]))

        self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]],
                            self.evaluate(net))

  def test_with_1d_unknown_shape_sparse_tensor(self):
    # This test needs to access variables bia variable scope & needs to be
    # run inside of a legacy graph
    with ops.Graph().as_default():
      price = fc.numeric_column('price')
      price_buckets = fc.bucketized_column(
          price, boundaries=[
              0.,
              10.,
              100.,
          ])
      body_style = fc.categorical_column_with_vocabulary_list(
          'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
      country = fc.categorical_column_with_vocabulary_list(
          'country', vocabulary_list=['US', 'JP', 'CA'])

      # Provides 1-dim tensor and dense tensor.
      features = {
          'price': array_ops.placeholder(dtypes.float32),
          'body-style': array_ops.sparse_placeholder(dtypes.string),
          'country': array_ops.placeholder(dtypes.string),
      }
      self.assertIsNone(features['price'].shape.ndims)
      self.assertIsNone(features['body-style'].get_shape().ndims)

      price_data = np.array([-1., 12.])
      body_style_data = sparse_tensor.SparseTensorValue(
          indices=((0,), (1,)), values=('sedan', 'hardtop'), dense_shape=(2,))
      country_data = np.array(['US', 'CA'])

      net = fc_old.linear_model(features, [price_buckets, body_style, country])
      bias = get_linear_model_bias()
      price_buckets_var = get_linear_model_column_var(price_buckets)
      body_style_var = get_linear_model_column_var(body_style)
      with _initialized_session() as sess:
        sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
        sess.run(bias.assign([5.]))

        self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]],
                            sess.run(
                                net,
                                feed_dict={
                                    features['price']: price_data,
                                    features['body-style']: body_style_data,
                                    features['country']: country_data
                                }))

  def test_with_rank_0_feature(self):
    price = fc.numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegex(ValueError, 'Feature .* cannot have rank 0'):
      fc_old.linear_model(features, [price])

    # This test needs to construct graph placeholders
    # w/ dynamic rank 0, so we enter a graph
    with ops.Graph().as_default():
      # Dynamic rank 0 should fail
      features = {
          'price': array_ops.placeholder(dtypes.float32),
      }
      net = fc_old.linear_model(features, [price])
      self.assertEqual(1, net.shape[1])
      with _initialized_session() as sess:
        with self.assertRaisesOpError('Feature .* cannot have rank 0'):
          sess.run(net, feed_dict={features['price']: np.array(1)})

  def test_multiple_linear_models(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features1 = {'price': [[1.], [5.]]}
      features2 = {'price': [[2.], [10.]]}
      predictions1 = fc_old.linear_model(features1, [price])
      predictions2 = fc_old.linear_model(features2, [price])
      bias1 = get_linear_model_bias(name='linear_model')
      bias2 = get_linear_model_bias(name='linear_model_1')
      price_var1 = get_linear_model_column_var(price, name='linear_model')
      price_var2 = get_linear_model_column_var(price, name='linear_model_1')
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias1))
        sess.run(price_var1.assign([[10.]]))
        sess.run(bias1.assign([5.]))
        self.assertAllClose([[15.], [55.]], self.evaluate(predictions1))
        self.assertAllClose([0.], self.evaluate(bias2))
        sess.run(price_var2.assign([[10.]]))
        sess.run(bias2.assign([5.]))
        self.assertAllClose([[25.], [105.]], self.evaluate(predictions2))

  def test_linear_model_v1_shared_embedding_all_other_v2(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price = fc.numeric_column('price')  # v2
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)  # v2
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)  # v2
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)  # v2
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)  # v2
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns(
          [categorical_column_a, categorical_column_b], dimension=2)  # v1
      all_cols = [
          price, some_embedding_column, shared_embedding_a, shared_embedding_b
      ]

      features = {
          'price': [[3.], [4.]],
          'sparse_feature': [['a'], ['x']],
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
      fc_old.linear_model(features, all_cols)
      bias = get_linear_model_bias()

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([0.], self.evaluate(bias))

  def test_linear_model_v1_shared_embedding_with_v2_cat_all_other_v2(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price = fc.numeric_column('price')  # v2
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)  # v2
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)  # v2
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)  # v2
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)  # v2
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns(
          [categorical_column_a, categorical_column_b], dimension=2)  # v1
      all_cols = [
          price, some_embedding_column, shared_embedding_a, shared_embedding_b
      ]

      features = {
          'price': [[3.], [4.]],
          'sparse_feature': [['a'], ['x']],
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
      fc_old.linear_model(features, all_cols)
      bias = get_linear_model_bias()

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([0.], self.evaluate(bias))

  def test_linear_model_v1_v2_mix(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price = fc.numeric_column('price')  # v2
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)  # v1
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)  # v1
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)  # v2
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)  # v2
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns(
          [categorical_column_a, categorical_column_b], dimension=2)  # v1
      all_cols = [
          price, some_embedding_column, shared_embedding_a, shared_embedding_b
      ]

      features = {
          'price': [[3.], [4.]],
          'sparse_feature': [['a'], ['x']],
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
      fc_old.linear_model(features, all_cols)
      bias = get_linear_model_bias()

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([0.], self.evaluate(bias))

  def test_linear_model_v2_shared_embedding_all_other_v1(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price = fc.numeric_column('price')  # v1
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)  # v1
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)  # v1
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)  # v2
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)  # v2
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b], dimension=2)  # v2
      all_cols = [
          price, some_embedding_column, shared_embedding_a, shared_embedding_b
      ]

      features = {
          'price': [[3.], [4.]],
          'sparse_feature': [['a'], ['x']],
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
      with self.assertRaisesRegex(ValueError,
                                  'SharedEmbeddingColumns are not supported'):
        fc_old.linear_model(features, all_cols)


class InputLayerTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_retrieving_input(self):
    features = {'a': [0.]}
    input_layer = fc_old.InputLayer(fc.numeric_column('a'))
    inputs = self.evaluate(input_layer(features))
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

      input_layer = fc_old.InputLayer([embedding_column])
      features = {'a': sparse_input}

      inputs = input_layer(features)
      variables = input_layer.variables

      # Sanity check: test that the inputs are correct.
      self.assertAllEqual([[1, 0], [0, 1], [1, 1]], inputs)

      # Check that only one variable was created.
      self.assertEqual(1, len(variables))

      # Check that invoking input_layer on the same features does not create
      # additional variables
      _ = input_layer(features)
      self.assertEqual(1, len(variables))
      self.assertIs(variables[0], input_layer.variables[0])

  def test_feature_column_input_layer_gradient(self):
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

      input_layer = fc_old.InputLayer([embedding_column])
      features = {'a': sparse_input}

      def scale_matrix():
        matrix = input_layer(features)
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


class FunctionalInputLayerTest(test.TestCase):

  def test_raises_if_empty_feature_columns(self):
    with self.assertRaisesRegex(ValueError,
                                'feature_columns must not be empty'):
      fc_old.input_layer(features={}, feature_columns=[])

  def test_should_be_dense_column(self):
    with self.assertRaisesRegex(ValueError, 'must be a _DenseColumn'):
      fc_old.input_layer(
          features={'a': [[0]]},
          feature_columns=[
              fc.categorical_column_with_hash_bucket('wire_cast', 4)
          ])

  def test_does_not_support_dict_columns(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      fc_old.input_layer(
          features={'a': [[0]]}, feature_columns={'a': fc.numeric_column('a')})

  def test_bare_column(self):
    with ops.Graph().as_default():
      features = features = {'a': [0.]}
      net = fc_old.input_layer(features, fc.numeric_column('a'))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0.]], self.evaluate(net))

  def test_column_generator(self):
    with ops.Graph().as_default():
      features = features = {'a': [0.], 'b': [1.]}
      columns = (fc.numeric_column(key) for key in features)
      net = fc_old.input_layer(features, columns)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0., 1.]], self.evaluate(net))

  def test_raises_if_duplicate_name(self):
    with self.assertRaisesRegex(
        ValueError, 'Duplicate feature column name found for columns'):
      fc_old.input_layer(
          features={'a': [[0]]},
          feature_columns=[fc.numeric_column('a'),
                           fc.numeric_column('a')])

  def test_one_column(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      net = fc_old.input_layer(features, [price])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1.], [5.]], self.evaluate(net))

  def test_multi_dimension(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      net = fc_old.input_layer(features, [price])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_raises_if_shape_mismatch(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegex(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        fc_old.input_layer(features, [price])

  def test_reshaping(self):
    price = fc.numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      net = fc_old.input_layer(features, [price])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_multi_column(self):
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      net = fc_old.input_layer(features, [price1, price2])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2., 3.], [5., 6., 4.]], self.evaluate(net))

  def test_fills_cols_to_vars(self):
    # Provide three _DenseColumn's to input_layer: a _NumericColumn, a
    # _BucketizedColumn, and an _EmbeddingColumn.  Only the _EmbeddingColumn
    # creates a Variable.
    price1 = fc.numeric_column('price1')
    dense_feature = fc.numeric_column('dense_feature')
    dense_feature_bucketized = fc.bucketized_column(
        dense_feature, boundaries=[0.])
    some_sparse_column = fc.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = fc.embedding_column(
        some_sparse_column, dimension=10)
    with ops.Graph().as_default():
      features = {
          'price1': [[3.], [4.]],
          'dense_feature': [[-1.], [4.]],
          'sparse_feature': [['a'], ['x']],
      }
      cols_to_vars = {}
      all_cols = [price1, dense_feature_bucketized, some_embedding_column]
      fc_old.input_layer(features, all_cols, cols_to_vars=cols_to_vars)
      self.assertCountEqual(list(cols_to_vars.keys()), all_cols)
      self.assertEqual(0, len(cols_to_vars[price1]))
      self.assertEqual(0, len(cols_to_vars[dense_feature_bucketized]))
      self.assertEqual(1, len(cols_to_vars[some_embedding_column]))
      self.assertIsInstance(cols_to_vars[some_embedding_column][0],
                            variables_lib.Variable)
      self.assertAllEqual(cols_to_vars[some_embedding_column][0].shape, [5, 10])

  def test_fills_cols_to_vars_shared_embedding(self):
    # Provide 5 DenseColumn's to input_layer: a NumericColumn, a
    # BucketizedColumn, an EmbeddingColumn, two SharedEmbeddingColumns. The
    # EmbeddingColumn creates a Variable and the two SharedEmbeddingColumns
    # shared one variable.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price1 = fc.numeric_column('price1')
      dense_feature = fc.numeric_column('dense_feature')
      dense_feature_bucketized = fc.bucketized_column(
          dense_feature, boundaries=[0.])
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns(
          [categorical_column_a, categorical_column_b], dimension=2)
      features = {
          'price1': [[3.], [4.]],
          'dense_feature': [[-1.], [4.]],
          'sparse_feature': [['a'], ['x']],
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
      cols_to_vars = {}
      all_cols = [
          price1, dense_feature_bucketized, some_embedding_column,
          shared_embedding_a, shared_embedding_b
      ]
      fc_old.input_layer(features, all_cols, cols_to_vars=cols_to_vars)
      self.assertCountEqual(list(cols_to_vars.keys()), all_cols)
      self.assertEqual(0, len(cols_to_vars[price1]))
      self.assertEqual(0, len(cols_to_vars[dense_feature_bucketized]))
      self.assertEqual(1, len(cols_to_vars[some_embedding_column]))
      self.assertEqual(1, len(cols_to_vars[shared_embedding_a]))
      # This is a bug in the current implementation and should be fixed in the
      # new one.
      self.assertEqual(0, len(cols_to_vars[shared_embedding_b]))
      self.assertIsInstance(cols_to_vars[some_embedding_column][0],
                            variables_lib.Variable)
      self.assertAllEqual(cols_to_vars[some_embedding_column][0].shape, [5, 10])
      self.assertIsInstance(cols_to_vars[shared_embedding_a][0],
                            variables_lib.Variable)
      self.assertAllEqual(cols_to_vars[shared_embedding_a][0].shape, [3, 2])

  def test_fills_cols_to_vars_partitioned_variables(self):
    price1 = fc.numeric_column('price1')
    dense_feature = fc.numeric_column('dense_feature')
    dense_feature_bucketized = fc.bucketized_column(
        dense_feature, boundaries=[0.])
    some_sparse_column = fc.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = fc.embedding_column(
        some_sparse_column, dimension=10)
    with ops.Graph().as_default():
      features = {
          'price1': [[3.], [4.]],
          'dense_feature': [[-1.], [4.]],
          'sparse_feature': [['a'], ['x']],
      }
      cols_to_vars = {}
      all_cols = [price1, dense_feature_bucketized, some_embedding_column]
      with variable_scope.variable_scope(
          'input_from_feature_columns',
          partitioner=partitioned_variables.fixed_size_partitioner(3, axis=0)):
        fc_old.input_layer(features, all_cols, cols_to_vars=cols_to_vars)
      self.assertCountEqual(list(cols_to_vars.keys()), all_cols)
      self.assertEqual(0, len(cols_to_vars[price1]))
      self.assertEqual(0, len(cols_to_vars[dense_feature_bucketized]))
      self.assertEqual(3, len(cols_to_vars[some_embedding_column]))
      self.assertEqual(
          'input_from_feature_columns/input_layer/sparse_feature_embedding/'
          'embedding_weights/part_0:0',
          cols_to_vars[some_embedding_column][0].name)
      self.assertAllEqual(cols_to_vars[some_embedding_column][0].shape, [2, 10])
      self.assertAllEqual(cols_to_vars[some_embedding_column][1].shape, [2, 10])
      self.assertAllEqual(cols_to_vars[some_embedding_column][2].shape, [1, 10])

  def test_column_order(self):
    price_a = fc.numeric_column('price_a')
    price_b = fc.numeric_column('price_b')
    with ops.Graph().as_default():
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
      }
      net1 = fc_old.input_layer(features, [price_a, price_b])
      net2 = fc_old.input_layer(features, [price_b, price_a])

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
      with self.assertRaisesRegex(Exception, 'must be a _DenseColumn'):
        fc_old.input_layer(features, [animal])

  def test_static_batch_size_mismatch(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc_old.input_layer(features, [price1, price2])

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
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc_old.input_layer(features, [price1, price2, price3])

  def test_runtime_batch_size_mismatch(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      net = fc_old.input_layer(features, [price1, price2])
      with _initialized_session() as sess:
        with self.assertRaisesRegex(errors.OpError,
                                    'Dimension 0 in both shapes must be equal'):
          sess.run(net, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
          'price2': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
      }
      net = fc_old.input_layer(features, [price1, price2])
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
      fc_old.input_layer(features, all_cols)
      fc_old.input_layer(features, all_cols)
      # Make sure that 2 variables get created in this case.
      self.assertEqual(2, len(
          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))
      expected_var_names = [
          'input_layer/sparse_feature_embedding/embedding_weights:0',
          'input_layer_1/sparse_feature_embedding/embedding_weights:0'
      ]
      self.assertCountEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])

  def test_with_1d_sparse_tensor(self):
    embedding_values = (
        (1., 2., 3., 4., 5.),  # id 0
        (6., 7., 8., 9., 10.),  # id 1
        (11., 12., 13., 14., 15.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      del shape, dtype, partition_info
      return embedding_values

    # price has 1 dimension in input_layer
    price = fc.numeric_column('price')

    # one_hot_body_style has 3 dims in input_layer.
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    one_hot_body_style = fc.indicator_column(body_style)

    # embedded_body_style has 5 dims in input_layer.
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

    if context.executing_eagerly():
      # The variables will only be initialized in TF2
      net = fc_old.input_layer(features,
                               [price, one_hot_body_style, embedded_country])
      self.assertEqual(1 + 3 + 5, net.shape[1])

      # Each row is formed by concatenating `embedded_body_style`,
      # `one_hot_body_style`, and `price` in order.
      self.assertAllEqual([[0., 0., 1., 11., 12., 13., 14., 15., 11.],
                           [1., 0., 0., 1., 2., 3., 4., 5., 12.]],
                          self.evaluate(net))

  def test_with_1d_unknown_shape_sparse_tensor(self):
    # This test needs to construct graph placeholders
    # w/ unknown shapes, so we enter a graph
    with ops.Graph().as_default():
      embedding_values = (
          (1., 2.),  # id 0
          (6., 7.),  # id 1
          (11., 12.)  # id 2
      )

      def _initializer(shape, dtype, partition_info=None):
        del shape, dtype, partition_info
        return embedding_values

      # price has 1 dimension in input_layer
      price = fc.numeric_column('price')

      # one_hot_body_style has 3 dims in input_layer.
      body_style = fc.categorical_column_with_vocabulary_list(
          'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
      one_hot_body_style = fc.indicator_column(body_style)

      # embedded_body_style has 5 dims in input_layer.
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

      net = fc_old.input_layer(features,
                               [price, one_hot_body_style, embedded_country])
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

  def test_with_rank_0_feature(self):
    # price has 1 dimension in input_layer
    price = fc.numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegex(ValueError, 'Feature .* cannot have rank 0'):
      fc_old.input_layer(features, [price])

    # This test needs to construct graph placeholders
    # w/ dynamic rank 0, so we enter a graph
    with ops.Graph().as_default():
      # Dynamic rank 0 should fail
      features = {
          'price': array_ops.placeholder(dtypes.float32),
      }
      net = fc_old.input_layer(features, [price])
      self.assertEqual(1, net.shape[1])
      with _initialized_session() as sess:
        with self.assertRaisesOpError('Feature .* cannot have rank 0'):
          sess.run(net, feed_dict={features['price']: np.array(1)})


class MakeParseExampleSpecTest(test.TestCase):

  class _TestFeatureColumn(BaseFeatureColumnForTests,
                           collections.namedtuple('_TestFeatureColumn',
                                                  ('parse_spec'))):

    @property
    def _is_v2_column(self):
      return True

    @property
    def name(self):
      return '_TestFeatureColumn'

    def transform_feature(self, transformation_cache, state_manager):
      pass

    def _transform_feature(self, inputs):
      pass

    @property
    def parse_example_spec(self):
      return self.parse_spec

    @property
    def _parse_example_spec(self):
      return self.parse_spec

  def test_no_feature_columns(self):
    actual = fc.make_parse_example_spec_v2([])
    self.assertDictEqual({}, actual)

  def test_invalid_type(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    with self.assertRaisesRegex(
        ValueError,
        'All feature_columns must be FeatureColumn instances.*invalid_column'):
      fc.make_parse_example_spec_v2((self._TestFeatureColumn({
          key1: parse_spec1
      }), 'invalid_column'))

  def test_one_feature_column(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    actual = fc.make_parse_example_spec_v2((self._TestFeatureColumn({
        key1: parse_spec1
    }),))
    self.assertDictEqual({key1: parse_spec1}, actual)

  def test_two_feature_columns(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    key2 = 'key2'
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    actual = fc.make_parse_example_spec_v2((self._TestFeatureColumn({
        key1: parse_spec1
    }), self._TestFeatureColumn({
        key2: parse_spec2
    })))
    self.assertDictEqual({key1: parse_spec1, key2: parse_spec2}, actual)

  def test_equal_keys_different_parse_spec(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    with self.assertRaisesRegex(
        ValueError,
        'feature_columns contain different parse_spec for key key1'):
      fc.make_parse_example_spec_v2((self._TestFeatureColumn({
          key1: parse_spec1
      }), self._TestFeatureColumn({
          key1: parse_spec2
      })))

  def test_equal_keys_equal_parse_spec(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    actual = fc.make_parse_example_spec_v2((self._TestFeatureColumn({
        key1: parse_spec1
    }), self._TestFeatureColumn({
        key1: parse_spec1
    })))
    self.assertDictEqual({key1: parse_spec1}, actual)

  def test_multiple_features_dict(self):
    """parse_spc for one column is a dict with length > 1."""
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    key2 = 'key2'
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    key3 = 'key3'
    parse_spec3 = parsing_ops.VarLenFeature(dtype=dtypes.int32)
    actual = fc.make_parse_example_spec_v2((self._TestFeatureColumn({
        key1: parse_spec1
    }), self._TestFeatureColumn({
        key2: parse_spec2,
        key3: parse_spec3
    })))
    self.assertDictEqual({
        key1: parse_spec1,
        key2: parse_spec2,
        key3: parse_spec3
    }, actual)


def _assert_sparse_tensor_value(test_case, expected, actual):
  test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
  test_case.assertAllEqual(expected.indices, actual.indices)

  test_case.assertEqual(
      np.array(expected.values).dtype,
      np.array(actual.values).dtype)
  test_case.assertAllEqual(expected.values, actual.values)

  test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
  test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


class VocabularyFileCategoricalColumnTest(test.TestCase):

  _FILE_FORMAT = None
  _VOCABULARY_SIZE_ERROR = (errors.OpError, 'Invalid vocab_size')

  def setUp(self):
    super(VocabularyFileCategoricalColumnTest, self).setUp()

    # Contains ints, Golden State Warriors jersey numbers: 30, 35, 11, 23, 22
    self._warriors_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/warriors_vocabulary.txt')
    self._warriors_vocabulary_size = 5

    # Contains strings, character names from 'The Wire': omar, stringer, marlo
    self._wire_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/wire_vocabulary.txt')
    self._wire_vocabulary_size = 3

    # Contains unicode characters.
    self._unicode_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/unicode_vocabulary')

  def test_defaults(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3)
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_defaults_unicode(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file=self._unicode_vocabulary_file_name)
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    if isinstance(column.num_buckets, (int, np.integer)):
      self.assertEqual(165, column.num_buckets)
    else:
      self.assertEqual(165, self.evaluate(column.num_buckets))
    self.assertEqual({'aaa': parsing_ops.VarLenFeature(dtypes.string)},
                     column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_key_should_be_string(self):
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.categorical_column_with_vocabulary_file(
          key=('aaa',), vocabulary_file='path_to_file', vocabulary_size=3)

  def test_all_constructor_args(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file='path_to_file',
        vocabulary_size=3,
        num_oov_buckets=4,
        dtype=dtypes.int32)
    self.assertEqual(7, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, column.parse_example_spec)

  def test_deep_copy(self):
    original = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file='path_to_file',
        vocabulary_size=3,
        num_oov_buckets=4,
        dtype=dtypes.int32)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(7, column.num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int32)
      }, column.parse_example_spec)

  def test_vocabulary_file_none(self):
    with self.assertRaisesRegex(ValueError, 'Missing vocabulary_file'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file=None, vocabulary_size=3)

  def test_vocabulary_file_empty_string(self):
    with self.assertRaisesRegex(ValueError, 'Missing vocabulary_file'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file='', vocabulary_size=3)

  def test_invalid_vocabulary_file(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='file_does_not_exist', vocabulary_size=10)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(errors.OpError, 'file_does_not_exist'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)
      self.evaluate(lookup_ops.tables_initializer())

  def test_invalid_vocabulary_size(self):
    with self.assertRaisesRegex(ValueError, 'Invalid vocabulary_size'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=-1)
    with self.assertRaisesRegex(ValueError, 'Invalid vocabulary_size'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=0)

  def test_too_large_vocabulary_size(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size + 1)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(*self._VOCABULARY_SIZE_ERROR):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)
      self.evaluate(lookup_ops.tables_initializer())

  def test_invalid_num_oov_buckets(self):
    with self.assertRaisesRegex(ValueError, 'Invalid num_oov_buckets'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file='path',
          vocabulary_size=3,
          num_oov_buckets=-1)

  def test_invalid_dtype(self):
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file='path',
          vocabulary_size=3,
          dtype=dtypes.float64)

  def test_invalid_buckets_and_default_value(self):
    with self.assertRaisesRegex(ValueError,
                                'both num_oov_buckets and default_value'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size,
          num_oov_buckets=100,
          default_value=2)

  def test_invalid_input_dtype_int32(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        dtype=dtypes.string)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(12, 24, 36),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_invalid_input_dtype_string(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_parse_example(self):
    a = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_get_sparse_tensors(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_none_vocabulary_size(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file=self._wire_vocabulary_file_name)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_transform_feature(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_tensor = fc._transform_features_v2({
        'aaa': inputs
    }, [column], None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape), self.evaluate(id_tensor))

  def test_get_sparse_tensors_dense_input(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': (('marlo', ''), ('skywalker', 'omar'))
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=(2, 2)), self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_default_value_in_vocabulary(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        default_value=2)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 2, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_with_oov_buckets(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=100)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (1, 2)),
        values=('marlo', 'skywalker', 'omar', 'heisenberg'),
        dense_shape=(2, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 33, 0, 62), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_small_vocabulary_size(self):
    # 'marlo' is the last entry in our vocabulary file, so be setting
    # `vocabulary_size` to 1 less than number of entries in file, we take
    # 'marlo' out of the vocabulary.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size - 1)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((-1, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (2, 2)),
        values=(11, 100, 30, 22),
        dense_shape=(3, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0, 4), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32_dense_input(self):
    default_value = -100
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32,
        default_value=default_value)
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': ((11, -1, -1), (100, 30, -1), (-1, -1, 22))
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (2, 2)),
            values=np.array((2, default_value, 0, 4), dtype=np.int64),
            dense_shape=(3, 3)), self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32_with_oov_buckets(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32,
        num_oov_buckets=100)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (2, 2)),
        values=(11, 100, 30, 22),
        dense_shape=(3, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 60, 0, 4), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_old_linear_model(self):
    wire_column = fc.categorical_column_with_vocabulary_file(
        key='wire',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=1)
    self.assertEqual(4, wire_column.num_buckets)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          wire_column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), self.evaluate(wire_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
      # 'marlo' -> 2: wire_var[2] = 3
      # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
      self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))

  def test_serialization(self):
    wire_column = fc.categorical_column_with_vocabulary_file(
        key='wire',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=1)

    self.assertEqual(['wire'], wire_column.parents)

    config = wire_column.get_config()
    self.assertEqual(
        {
            'default_value': -1,
            'dtype': 'string',
            'key': 'wire',
            'num_oov_buckets': 1,
            'vocabulary_file': self._wire_vocabulary_file_name,
            'vocabulary_size': 3,
            'file_format': self._FILE_FORMAT,
        }, config)

    self.assertEqual(wire_column,
                     fc.VocabularyFileCategoricalColumn.from_config(config))


class VocabularyTfrecordGzipFileCategoricalColumnTest(
    VocabularyFileCategoricalColumnTest):

  _FILE_FORMAT = 'tfrecord_gzip'
  _VOCABULARY_SIZE_ERROR = (errors.FailedPreconditionError,
                            'Input dataset was expected to contain 4 elements')

  def setUp(self):
    super(VocabularyTfrecordGzipFileCategoricalColumnTest, self).setUp()

    # Contains ints, Golden State Warriors jersey numbers: 30, 35, 11, 23, 22
    self._warriors_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/warriors_vocabulary.tfrecord.gz')
    self._warriors_vocabulary_size = 5

    # Contains strings, character names from 'The Wire': omar, stringer, marlo
    self._wire_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/wire_vocabulary.tfrecord.gz')
    self._wire_vocabulary_size = 3

    # Contains unicode characters.
    self._unicode_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/unicode_vocabulary.tfrecord.gz')


class VocabularyListCategoricalColumnTest(test.TestCase):

  def test_defaults_string(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_key_should_be_string(self):
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.categorical_column_with_vocabulary_list(
          key=('aaa',), vocabulary_list=('omar', 'stringer', 'marlo'))

  def test_defaults_int(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36))
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, column.parse_example_spec)

  def test_all_constructor_args(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=(12, 24, 36),
        dtype=dtypes.int32,
        default_value=-99)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, column.parse_example_spec)

  def test_deep_copy(self):
    original = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36), dtype=dtypes.int32)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(3, column.num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int32)
      }, column.parse_example_spec)

  def test_invalid_dtype(self):
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=('omar', 'stringer', 'marlo'),
          dtype=dtypes.float32)

  def test_invalid_mapping_dtype(self):
    with self.assertRaisesRegex(ValueError,
                                r'vocabulary dtype must be string or integer'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12., 24., 36.))

  def test_mismatched_int_dtype(self):
    with self.assertRaisesRegex(ValueError,
                                r'dtype.*and vocabulary dtype.*do not match'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=('omar', 'stringer', 'marlo'),
          dtype=dtypes.int32)

  def test_mismatched_string_dtype(self):
    with self.assertRaisesRegex(ValueError,
                                r'dtype.*and vocabulary dtype.*do not match'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 36), dtype=dtypes.string)

  def test_none_mapping(self):
    with self.assertRaisesRegex(ValueError,
                                r'vocabulary_list.*must be non-empty'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=None)

  def test_empty_mapping(self):
    with self.assertRaisesRegex(ValueError,
                                r'vocabulary_list.*must be non-empty'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=tuple([]))

  def test_duplicate_mapping(self):
    with self.assertRaisesRegex(ValueError, 'Duplicate keys'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 12))

  def test_invalid_num_oov_buckets(self):
    with self.assertRaisesRegex(ValueError, 'Invalid num_oov_buckets'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 36), num_oov_buckets=-1)

  def test_invalid_buckets_and_default_value(self):
    with self.assertRaisesRegex(ValueError,
                                'both num_oov_buckets and default_value'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=(12, 24, 36),
          num_oov_buckets=100,
          default_value=2)

  def test_invalid_input_dtype_int32(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(12, 24, 36),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_invalid_input_dtype_string(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_parse_example_string(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_parse_example_int(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(11, 21, 31))
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[11, 21]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]], values=[11, 21], dense_shape=[1, 2]),
        self.evaluate(features['aaa']))

  def test_get_sparse_tensors(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_transform_feature(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_tensor = fc._transform_features_v2({
        'aaa': inputs
    }, [column], None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape), self.evaluate(id_tensor))

  def test_get_sparse_tensors_dense_input(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': (('marlo', ''), ('skywalker', 'omar'))
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=(2, 2)), self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_default_value_in_vocabulary(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        default_value=2)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 2, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_with_oov_buckets(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=100)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (1, 2)),
        values=('marlo', 'skywalker', 'omar', 'heisenberg'),
        dense_shape=(2, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 33, 0, 62), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
        dtype=dtypes.int32)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (2, 2)),
        values=np.array((11, 100, 30, 22), dtype=np.int32),
        dense_shape=(3, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0, 4), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32_dense_input(self):
    default_value = -100
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
        dtype=dtypes.int32,
        default_value=default_value)
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa':
                np.array(((11, -1, -1), (100, 30, -1), (-1, -1, 22)),
                         dtype=np.int32)
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (2, 2)),
            values=np.array((2, default_value, 0, 4), dtype=np.int64),
            dense_shape=(3, 3)), self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32_with_oov_buckets(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
        dtype=dtypes.int32,
        num_oov_buckets=100)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (2, 2)),
        values=(11, 100, 30, 22),
        dense_shape=(3, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 60, 0, 4), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_old_linear_model(self):
    wire_column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=1)
    self.assertEqual(4, wire_column.num_buckets)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          wire_column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), self.evaluate(wire_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
      # 'marlo' -> 2: wire_var[2] = 3
      # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
      self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))

  def test_serialization(self):
    wire_column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=1)

    self.assertEqual(['aaa'], wire_column.parents)

    config = wire_column.get_config()
    self.assertEqual({
        'default_value': -1,
        'dtype': 'string',
        'key': 'aaa',
        'num_oov_buckets': 1,
        'vocabulary_list': ('omar', 'stringer', 'marlo')
    }, config)

    self.assertEqual(wire_column,
                     fc.VocabularyListCategoricalColumn.from_config(config))


class IdentityCategoricalColumnTest(test.TestCase):

  def test_constructor(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_key_should_be_string(self):
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.categorical_column_with_identity(key=('aaa',), num_buckets=3)

  def test_deep_copy(self):
    original = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(3, column.num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, column.parse_example_spec)

  def test_invalid_num_buckets_zero(self):
    with self.assertRaisesRegex(ValueError, 'num_buckets 0 < 1'):
      fc.categorical_column_with_identity(key='aaa', num_buckets=0)

  def test_invalid_num_buckets_negative(self):
    with self.assertRaisesRegex(ValueError, 'num_buckets -1 < 1'):
      fc.categorical_column_with_identity(key='aaa', num_buckets=-1)

  def test_invalid_default_value_too_small(self):
    with self.assertRaisesRegex(ValueError, 'default_value -1 not in range'):
      fc.categorical_column_with_identity(
          key='aaa', num_buckets=3, default_value=-1)

  def test_invalid_default_value_too_big(self):
    with self.assertRaisesRegex(ValueError, 'default_value 3 not in range'):
      fc.categorical_column_with_identity(
          key='aaa', num_buckets=3, default_value=3)

  def test_invalid_input_dtype(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'Invalid input, not integer'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_parse_example(self):
    a = fc.categorical_column_with_identity(key='aaa', num_buckets=30)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[11, 21]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([11, 21], dtype=np.int64),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_get_sparse_tensors(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)), values=(0, 1, 0), dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((0, 1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_transform_feature(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)), values=(0, 1, 0), dense_shape=(2, 2))
    id_tensor = fc._transform_features_v2({
        'aaa': inputs
    }, [column], None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((0, 1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape), self.evaluate(id_tensor))

  def test_get_sparse_tensors_dense_input(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': ((0, -1), (1, 0))
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((0, 1, 0), dtype=np.int64),
            dense_shape=(2, 2)), self.evaluate(id_weight_pair.id_tensor))

  def _test_get_sparse_tensors_with_inputs_too_small(self):
    # Inputs.
    vocabulary_size = 2
    sparse_input = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (0, 0), (1, 1), (1, 2)),
        values=(-9, 0, -6, 1),
        dense_shape=(2, 4))

    # Embedding variable.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
    )

    def _initializer(shape, dtype, partition_info=None):
      del shape, dtype, partition_info
      return embedding_values

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column.get_dense_tensor(
        fc.FeatureTransformationCache({'aaa': sparse_input}), state_manager)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())
    expected_lookups = ((1., 2.), (3., 5))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_sparse_tensors_with_inputs_too_small(self):
    self._test_get_sparse_tensors_with_inputs_too_small()

  @test_util.enable_control_flow_v2
  def test_get_sparse_tensors_with_inputs_too_small_v2(self):
    self._test_get_sparse_tensors_with_inputs_too_small()

  def _test_get_sparse_tensors_with_inputs_too_big(self):
    # Inputs.
    vocabulary_size = 2
    sparse_input = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0)), values=(2, 0), dense_shape=(4, 5))

    # Embedding variable.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
    )

    def _initializer(shape, dtype, partition_info=None):
      del shape, dtype, partition_info
      return embedding_values

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    with self.assertRaisesRegex(errors.OpError,
                                r'indices\[0\].*\[0, 2\)'):
      # Provide sparse input and get dense result.
      embedding_lookup = embedding_column.get_dense_tensor(
          fc.FeatureTransformationCache({'aaa': sparse_input}), state_manager)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.evaluate(embedding_lookup)

  def test_get_sparse_tensors_with_inputs_too_big(self):
    self._test_get_sparse_tensors_with_inputs_too_big()

  @test_util.enable_control_flow_v2
  def test_get_sparse_tensors_with_inputs_too_big_v2(self):
    self._test_get_sparse_tensors_with_inputs_too_big()

  def test_get_sparse_tensors_with_default_value(self):
    column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=4, default_value=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(1, -1, 99),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((1, 3, 3), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_with_default_value_and_placeholder_inputs(self):
    # This test needs to run in a graph because it is explicitly testing
    # graph placeholders
    with ops.Graph().as_default():
      column = fc.categorical_column_with_identity(
          key='aaa', num_buckets=4, default_value=3)
      input_indices = array_ops.placeholder(dtype=dtypes.int64)
      input_values = array_ops.placeholder(dtype=dtypes.int32)
      input_shape = array_ops.placeholder(dtype=dtypes.int64)
      inputs = sparse_tensor.SparseTensorValue(
          indices=input_indices, values=input_values, dense_shape=input_shape)
      id_weight_pair = column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)
      self.assertIsNone(id_weight_pair.weight_tensor)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=np.array(((0, 0), (1, 0), (1, 1)), dtype=np.int64),
                values=np.array((1, 3, 3), dtype=np.int64),
                dense_shape=np.array((2, 2), dtype=np.int64)),
            id_weight_pair.id_tensor.eval(
                feed_dict={
                    input_indices: ((0, 0), (1, 0), (1, 1)),
                    input_values: (1, -1, 99),
                    input_shape: (2, 2),
                }))

  def test_old_linear_model(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    self.assertEqual(3, column.num_buckets)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] = 1
      # weight_var[2] + weight_var[1] = 3+2 = 5
      self.assertAllClose(((1.,), (5.,)), self.evaluate(predictions))

  def test_serialization(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)

    self.assertEqual(['aaa'], column.parents)

    config = column.get_config()
    self.assertEqual({
        'default_value': None,
        'key': 'aaa',
        'number_buckets': 3
    }, config)

    self.assertEqual(column, fc.IdentityCategoricalColumn.from_config(config))


class TransformFeaturesTest(test.TestCase):

  # All transform tests are distributed in column test.
  # Here we only test multi column case and naming
  def transform_multi_column(self):
    bucketized_price = fc.bucketized_column(
        fc.numeric_column('price'), boundaries=[0, 2, 4, 6])
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    with ops.Graph().as_default():
      features = {
          'price': [[-1.], [5.]],
          'wire':
              sparse_tensor.SparseTensor(
                  values=['omar', 'stringer', 'marlo'],
                  indices=[[0, 0], [1, 0], [1, 1]],
                  dense_shape=[2, 2])
      }
      transformed = fc._transform_features_v2(
          features, [bucketized_price, hashed_sparse], None)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertIn(bucketized_price.name, transformed[bucketized_price].name)
      self.assertAllEqual([[0], [3]],
                          self.evaluate(transformed[bucketized_price]))
      self.assertIn(hashed_sparse.name, transformed[hashed_sparse].name)
      self.assertAllEqual([6, 4, 1],
                          self.evaluate(transformed[hashed_sparse].values))

  def test_column_order(self):
    """When the column is both dense and sparse, uses sparse tensors."""

    class _LoggerColumn(BaseFeatureColumnForTests):

      def __init__(self, name):
        super(_LoggerColumn, self).__init__()
        self._name = name

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return self._name

      def transform_feature(self, transformation_cache, state_manager):
        self.call_order = call_logger['count']
        call_logger['count'] += 1
        return 'Anything'

      @property
      def parse_example_spec(self):
        pass

    with ops.Graph().as_default():
      column1 = _LoggerColumn('1')
      column2 = _LoggerColumn('2')
      call_logger = {'count': 0}
      fc._transform_features_v2({}, [column1, column2], None)
      self.assertEqual(0, column1.call_order)
      self.assertEqual(1, column2.call_order)

      call_logger = {'count': 0}
      fc._transform_features_v2({}, [column2, column1], None)
      self.assertEqual(0, column1.call_order)
      self.assertEqual(1, column2.call_order)


class IndicatorColumnTest(test.TestCase):

  def test_indicator_column(self):
    a = fc.categorical_column_with_hash_bucket('a', 4)
    indicator_a = fc.indicator_column(a)
    self.assertEqual(indicator_a.categorical_column.name, 'a')
    self.assertEqual(indicator_a.name, 'a_indicator')
    self.assertEqual(indicator_a.variable_shape, [1, 4])
    self.assertTrue(indicator_a._is_v2_column)

    b = fc_old._categorical_column_with_hash_bucket('b', hash_bucket_size=100)
    indicator_b = fc.indicator_column(b)
    self.assertEqual(indicator_b.categorical_column.name, 'b')
    self.assertEqual(indicator_b.name, 'b_indicator')
    self.assertEqual(indicator_b.variable_shape, [1, 100])
    self.assertFalse(indicator_b._is_v2_column)

  def test_not_categorical_input(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported input type.'):
      fc.indicator_column('aaa')

  def test_1D_shape_succeeds(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_hash_bucket('animal', 4))
    transformation_cache = fc.FeatureTransformationCache({
        'animal': ['fox', 'fox']
    })
    output = transformation_cache.get(animal, None)

    self.assertAllEqual([[0., 0., 1., 0.], [0., 0., 1., 0.]],
                        self.evaluate(output))

  def test_2D_shape_succeeds(self):
    # TODO(ispir/cassandrax): Switch to categorical_column_with_keys when ready.
    animal = fc.indicator_column(
        fc.categorical_column_with_hash_bucket('animal', 4))
    transformation_cache = fc.FeatureTransformationCache({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [1, 0]],
                values=['fox', 'fox'],
                dense_shape=[2, 1])
    })
    output = transformation_cache.get(animal, None)

    self.assertAllEqual([[0., 0., 1., 0.], [0., 0., 1., 0.]],
                        self.evaluate(output))

  def test_multi_hot(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))

    transformation_cache = fc.FeatureTransformationCache({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1, 1], dense_shape=[1, 2])
    })
    output = transformation_cache.get(animal, None)

    self.assertAllEqual([[0., 2., 0., 0.]], self.evaluate(output))

  def test_multi_hot2(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    transformation_cache = fc.FeatureTransformationCache({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
    })
    output = transformation_cache.get(animal, None)

    self.assertAllEqual([[0., 1., 1., 0.]], self.evaluate(output))

  def test_deep_copy(self):
    a = fc.categorical_column_with_hash_bucket('a', 4)
    column = fc.indicator_column(a)
    column_copy = copy.deepcopy(column)
    self.assertEqual(column_copy.categorical_column.name, 'a')
    self.assertEqual(column.name, 'a_indicator')
    self.assertEqual(column.variable_shape, [1, 4])

  def test_parse_example(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_indicator = fc.indicator_column(a)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a_indicator]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_transform(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_indicator = fc.indicator_column(a)
    features = {
        'aaa':
            sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1)),
                values=('marlo', 'skywalker', 'omar'),
                dense_shape=(2, 2))
    }
    indicator_tensor = fc._transform_features_v2(features, [a_indicator],
                                                 None)[a_indicator]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([[0, 0, 1], [1, 0, 0]], self.evaluate(indicator_tensor))

  def test_transform_with_weighted_column(self):
    # Github issue 12557
    ids = fc.categorical_column_with_vocabulary_list(
        key='ids', vocabulary_list=('a', 'b', 'c'))
    weights = fc.weighted_categorical_column(ids, 'weights')
    indicator = fc.indicator_column(weights)
    features = {
        'ids': constant_op.constant([['c', 'b', 'a', 'c']]),
        'weights': constant_op.constant([[2., 4., 6., 1.]])
    }
    indicator_tensor = fc._transform_features_v2(features, [indicator],
                                                 None)[indicator]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([[6., 4., 3.]], self.evaluate(indicator_tensor))

  def test_transform_with_missing_value_in_weighted_column(self):
    # Github issue 12583
    ids = fc.categorical_column_with_vocabulary_list(
        key='ids', vocabulary_list=('a', 'b', 'c'))
    weights = fc.weighted_categorical_column(ids, 'weights')
    indicator = fc.indicator_column(weights)
    features = {
        'ids': constant_op.constant([['c', 'b', 'unknown']]),
        'weights': constant_op.constant([[2., 4., 6.]])
    }
    indicator_tensor = fc._transform_features_v2(features, [indicator],
                                                 None)[indicator]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([[0., 4., 2.]], self.evaluate(indicator_tensor))

  def test_transform_with_missing_value_in_categorical_column(self):
    # Github issue 12583
    ids = fc.categorical_column_with_vocabulary_list(
        key='ids', vocabulary_list=('a', 'b', 'c'))
    indicator = fc.indicator_column(ids)
    features = {
        'ids': constant_op.constant([['c', 'b', 'unknown']]),
    }
    indicator_tensor = fc._transform_features_v2(features, [indicator],
                                                 None)[indicator]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([[0., 1., 1.]], self.evaluate(indicator_tensor))

  def test_old_linear_model(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }

      predictions = fc_old.linear_model(features, [animal])
      weight_var = get_linear_model_column_var(animal)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      # All should be zero-initialized.
      self.assertAllClose([[0.], [0.], [0.], [0.]], self.evaluate(weight_var))
      self.assertAllClose([[0.]], self.evaluate(predictions))
      self.evaluate(weight_var.assign([[1.], [2.], [3.], [4.]]))
      self.assertAllClose([[2. + 3.]], self.evaluate(predictions))

  def test_old_linear_model_old_categorical(self):
    animal = fc.indicator_column(
        fc_old._categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }

      predictions = fc_old.linear_model(features, [animal])
      weight_var = get_linear_model_column_var(animal)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      # All should be zero-initialized.
      self.assertAllClose([[0.], [0.], [0.], [0.]], self.evaluate(weight_var))
      self.assertAllClose([[0.]], self.evaluate(predictions))
      self.evaluate(weight_var.assign([[1.], [2.], [3.], [4.]]))
      self.assertAllClose([[2. + 3.]], self.evaluate(predictions))

  def test_input_layer(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      net = fc_old.input_layer(features, [animal])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0., 1., 1., 0.]], self.evaluate(net))

  def test_input_layer_old_categorical(self):
    animal = fc.indicator_column(
        fc_old._categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      net = fc_old.input_layer(features, [animal])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0., 1., 1., 0.]], self.evaluate(net))

  def test_serialization(self):
    parent = fc.categorical_column_with_identity('animal', num_buckets=4)
    animal = fc.indicator_column(parent)

    self.assertEqual([parent], animal.parents)

    config = animal.get_config()
    self.assertEqual({
        'categorical_column': {
            'class_name': 'IdentityCategoricalColumn',
            'config': {
                'key': 'animal',
                'default_value': None,
                'number_buckets': 4
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


class _TestStateManager(fc.StateManager):

  def __init__(self, trainable=True):
    # Dict of feature_column to a dict of variables.
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
    if feature_column not in self._all_variables:
      raise ValueError('Do not recognize FeatureColumn.')
    if name in self._all_variables[feature_column]:
      return self._all_variables[feature_column][name]
    raise ValueError('Could not find variable.')


class EmbeddingColumnTest(test.TestCase, parameterized.TestCase):

  def test_defaults(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('mean', embedding_column.combiner)
    self.assertIsNone(embedding_column.ckpt_to_load_from)
    self.assertIsNone(embedding_column.tensor_name_in_ckpt)
    self.assertIsNone(embedding_column.max_norm)
    self.assertTrue(embedding_column.trainable)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column.parse_example_spec)
    self.assertTrue(embedding_column._is_v2_column)

  def test_is_v2_column(self):
    categorical_column = fc_old._categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension)
    self.assertFalse(embedding_column._is_v2_column)

  def test_all_constructor_args(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer',
        ckpt_to_load_from='my_ckpt',
        tensor_name_in_ckpt='my_ckpt_tensor',
        max_norm=42.,
        trainable=False)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('my_combiner', embedding_column.combiner)
    self.assertEqual('my_ckpt', embedding_column.ckpt_to_load_from)
    self.assertEqual('my_ckpt_tensor', embedding_column.tensor_name_in_ckpt)
    self.assertEqual(42., embedding_column.max_norm)
    self.assertFalse(embedding_column.trainable)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column.parse_example_spec)

  def test_deep_copy(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    original = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer',
        ckpt_to_load_from='my_ckpt',
        tensor_name_in_ckpt='my_ckpt_tensor',
        max_norm=42.,
        trainable=False)
    for embedding_column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', embedding_column.categorical_column.name)
      self.assertEqual(3, embedding_column.categorical_column.num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column.categorical_column.parse_example_spec)

      self.assertEqual(embedding_dimension, embedding_column.dimension)
      self.assertEqual('my_combiner', embedding_column.combiner)
      self.assertEqual('my_ckpt', embedding_column.ckpt_to_load_from)
      self.assertEqual('my_ckpt_tensor', embedding_column.tensor_name_in_ckpt)
      self.assertEqual(42., embedding_column.max_norm)
      self.assertFalse(embedding_column.trainable)
      self.assertEqual('aaa_embedding', embedding_column.name)
      self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column.parse_example_spec)

  def test_invalid_initializer(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    with self.assertRaisesRegex(ValueError, 'initializer must be callable'):
      fc.embedding_column(categorical_column, dimension=2, initializer='not_fn')

  def test_parse_example(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_embedded = fc.embedding_column(a, dimension=2)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a_embedded]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_transform_feature(self):
    a = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    a_embedded = fc.embedding_column(a, dimension=2)
    features = {
        'aaa':
            sparse_tensor.SparseTensor(
                indices=((0, 0), (1, 0), (1, 1)),
                values=(0, 1, 0),
                dense_shape=(2, 2))
    }
    outputs = fc._transform_features_v2(features, [a, a_embedded], None)
    output_a = outputs[a]
    output_embedded = outputs[a_embedded]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(self, self.evaluate(output_a),
                                self.evaluate(output_embedded))

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
        initializer=_initializer)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column.get_dense_tensor(
        fc.FeatureTransformationCache({
            'aaa': sparse_input
        }), state_manager)

    # Assert expected embedding variable and lookups.
    if not context.executing_eagerly():
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_dense_tensor_old_categorical(self):
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
    categorical_column = fc_old._categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column._get_dense_tensor(
        fc_old._LazyBuilder({
            'aaa': sparse_input
        }))

    # Assert expected embedding variable and lookups.
    if not context.executing_eagerly():
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_dense_tensor_3d(self):
    # Inputs.
    vocabulary_size = 4
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0, 0), (1, 1, 0), (1, 1, 4), (3, 0, 0), (3, 1, 2)),
        values=(2, 0, 1, 1, 2),
        dense_shape=(4, 2, 5))

    # Embedding variable.
    embedding_dimension = 3
    embedding_values = (
        (1., 2., 4.),  # id 0
        (3., 5., 1.),  # id 1
        (7., 11., 2.),  # id 2
        (2., 7., 12.)  # id 3
    )

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    # Expected lookup result, using combiner='mean'.
    expected_lookups = (
        # example 0, ids [[2], []], embedding = [[7, 11, 2], [0, 0, 0]]
        ((7., 11., 2.), (0., 0., 0.)),
        # example 1, ids [[], [0, 1]], embedding
        # = mean([[], [1, 2, 4] + [3, 5, 1]]) = [[0, 0, 0], [2, 3.5, 2.5]]
        ((0., 0., 0.), (2., 3.5, 2.5)),
        # example 2, ids [[], []], embedding = [[0, 0, 0], [0, 0, 0]]
        ((0., 0., 0.), (0., 0., 0.)),
        # example 3, ids [[1], [2]], embedding = [[3, 5, 1], [7, 11, 2]]
        ((3., 5., 1.), (7., 11., 2.)),
    )

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column.get_dense_tensor(
        fc.FeatureTransformationCache({
            'aaa': sparse_input
        }), state_manager)

    # Assert expected embedding variable and lookups.
    if not context.executing_eagerly():
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_dense_tensor_placeholder_inputs(self):
    # This test explicitly checks graph placeholders, so we need to
    # be in a graph
    with ops.Graph().as_default():
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
          initializer=_initializer)
      state_manager = _TestStateManager()
      embedding_column.create_state(state_manager)

      # Provide sparse input and get dense result.
      input_indices = array_ops.placeholder(dtype=dtypes.int64)
      input_values = array_ops.placeholder(dtype=dtypes.int64)
      input_shape = array_ops.placeholder(dtype=dtypes.int64)
      embedding_lookup = embedding_column.get_dense_tensor(
          fc.FeatureTransformationCache({
              'aaa':
                  sparse_tensor.SparseTensorValue(
                      indices=input_indices,
                      values=input_values,
                      dense_shape=input_shape)
          }), state_manager)

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())
      with _initialized_session():
        self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
        self.assertAllEqual(
            expected_lookups,
            embedding_lookup.eval(
                feed_dict={
                    input_indices: sparse_input.indices,
                    input_values: sparse_input.values,
                    input_shape: sparse_input.dense_shape,
                }))

  def test_get_dense_tensor_restore_from_ckpt(self):
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

    # Embedding variable. The checkpoint file contains _embedding_values.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )
    ckpt_path = test.test_src_dir_path(
        'python/feature_column/testdata/embedding.ckpt')
    ckpt_tensor = 'my_embedding'

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
        ckpt_to_load_from=ckpt_path,
        tensor_name_in_ckpt=ckpt_tensor)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column.get_dense_tensor(
        fc.FeatureTransformationCache({
            'aaa': sparse_input
        }), state_manager)

    # Assert expected embedding variable and lookups.
    if not context.executing_eagerly():
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_input_layer(self):
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
        initializer=_initializer)

    # Provide sparse input and get dense result.
    feature_layer = fc_old.input_layer({
        'aaa': sparse_input
    }, (embedding_column,))

    if not context.executing_eagerly():
      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('input_layer/aaa_embedding/embedding_weights:0',),
                            tuple([v.name for v in global_vars]))
      trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertCountEqual(('input_layer/aaa_embedding/embedding_weights:0',),
                            tuple([v.name for v in trainable_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(trainable_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(feature_layer))

  def test_old_linear_model(self):
    # Inputs.
    batch_size = 4
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 4), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(batch_size, 5))

    # Embedding variable.
    embedding_dimension = 2
    embedding_shape = (vocabulary_size, embedding_dimension)
    zeros_embedding_values = np.zeros(embedding_shape)

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual(embedding_shape, shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return zeros_embedding_values

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          categorical_column.name: sparse_input
      }, (embedding_column,))
      expected_var_names = (
          'linear_model/bias_weights:0',
          'linear_model/aaa_embedding/weights:0',
          'linear_model/aaa_embedding/embedding_weights:0',
      )
      self.assertCountEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])
      trainable_vars = {
          v.name: v
          for v in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      }
      self.assertCountEqual(expected_var_names, trainable_vars.keys())
      bias = trainable_vars['linear_model/bias_weights:0']
      embedding_weights = trainable_vars[
          'linear_model/aaa_embedding/embedding_weights:0']
      linear_weights = trainable_vars['linear_model/aaa_embedding/weights:0']

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      # Predictions with all zero weights.
      self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
      self.assertAllClose(zeros_embedding_values,
                          self.evaluate(embedding_weights))
      self.assertAllClose(
          np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
      self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))

      # Predictions with all non-zero weights.
      self.evaluate(
          embedding_weights.assign((
              (1., 2.),  # id 0
              (3., 5.),  # id 1
              (7., 11.)  # id 2
          )))
      self.evaluate(linear_weights.assign(((4.,), (6.,))))
      # example 0, ids [2], embedding[0] = [7, 11]
      # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
      # example 2, ids [], embedding[2] = [0, 0]
      # example 3, ids [1], embedding[3] = [3, 5]
      # sum(embeddings * linear_weights)
      # = [4*7 + 6*11, 4*2 + 6*3.5, 4*0 + 6*0, 4*3 + 6*5] = [94, 29, 0, 42]
      self.assertAllClose(((94.,), (29.,), (0.,), (42.,)),
                          self.evaluate(predictions))

  def test_old_linear_model_old_categorical(self):
    # Inputs.
    batch_size = 4
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 4), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(batch_size, 5))

    # Embedding variable.
    embedding_dimension = 2
    embedding_shape = (vocabulary_size, embedding_dimension)
    zeros_embedding_values = np.zeros(embedding_shape)

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual(embedding_shape, shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return zeros_embedding_values

    # Build columns.
    categorical_column = fc_old._categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          categorical_column.name: sparse_input
      }, (embedding_column,))
      expected_var_names = (
          'linear_model/bias_weights:0',
          'linear_model/aaa_embedding/weights:0',
          'linear_model/aaa_embedding/embedding_weights:0',
      )
      self.assertCountEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])
      trainable_vars = {
          v.name: v
          for v in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      }
      self.assertCountEqual(expected_var_names, trainable_vars.keys())
      bias = trainable_vars['linear_model/bias_weights:0']
      embedding_weights = trainable_vars[
          'linear_model/aaa_embedding/embedding_weights:0']
      linear_weights = trainable_vars['linear_model/aaa_embedding/weights:0']

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      # Predictions with all zero weights.
      self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
      self.assertAllClose(zeros_embedding_values,
                          self.evaluate(embedding_weights))
      self.assertAllClose(
          np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
      self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))

      # Predictions with all non-zero weights.
      self.evaluate(
          embedding_weights.assign((
              (1., 2.),  # id 0
              (3., 5.),  # id 1
              (7., 11.)  # id 2
          )))
      self.evaluate(linear_weights.assign(((4.,), (6.,))))
      # example 0, ids [2], embedding[0] = [7, 11]
      # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
      # example 2, ids [], embedding[2] = [0, 0]
      # example 3, ids [1], embedding[3] = [3, 5]
      # sum(embeddings * linear_weights)
      # = [4*7 + 6*11, 4*2 + 6*3.5, 4*0 + 6*0, 4*3 + 6*5] = [94, 29, 0, 42]
      self.assertAllClose(((94.,), (29.,), (0.,), (42.,)),
                          self.evaluate(predictions))

  def test_serialization_with_default_initializer(self):

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_column = fc.embedding_column(categorical_column, dimension=2)

    self.assertEqual([categorical_column], embedding_column.parents)

    config = embedding_column.get_config()
    self.assertEqual(
        {
            'categorical_column': {
                'class_name': 'IdentityCategoricalColumn',
                'config': {
                    'number_buckets': 3,
                    'key': 'aaa',
                    'default_value': None
                }
            },
            'ckpt_to_load_from': None,
            'combiner': 'mean',
            'dimension': 2,
            'initializer': {
                'class_name': 'TruncatedNormal',
                'config': {
                    'dtype': 'float32',
                    'stddev': 0.7071067811865475,
                    'seed': None,
                    'mean': 0.0
                }
            },
            'max_norm': None,
            'tensor_name_in_ckpt': None,
            'trainable': True,
            'use_safe_embedding_lookup': True
        }, config)

    new_embedding_column = fc.EmbeddingColumn.from_config(
        config, custom_objects=None)
    self.assertEqual(embedding_column.get_config(),
                     new_embedding_column.get_config())
    self.assertIsNot(categorical_column,
                     new_embedding_column.categorical_column)

    new_embedding_column = fc.EmbeddingColumn.from_config(
        config,
        custom_objects=None,
        columns_by_name={
            serialization._column_name_with_class_name(categorical_column):
                categorical_column
        })
    self.assertEqual(embedding_column.get_config(),
                     new_embedding_column.get_config())
    self.assertIs(categorical_column, new_embedding_column.categorical_column)

  def test_serialization_with_custom_initializer(self):

    def _initializer(shape, dtype, partition_info=None):
      del shape, dtype, partition_info
      return ValueError('Not expected to be called')

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=2, initializer=_initializer)

    self.assertEqual([categorical_column], embedding_column.parents)

    config = embedding_column.get_config()
    self.assertEqual(
        {
            'categorical_column': {
                'class_name': 'IdentityCategoricalColumn',
                'config': {
                    'number_buckets': 3,
                    'key': 'aaa',
                    'default_value': None
                }
            },
            'ckpt_to_load_from': None,
            'combiner': 'mean',
            'dimension': 2,
            'initializer': '_initializer',
            'max_norm': None,
            'tensor_name_in_ckpt': None,
            'trainable': True,
            'use_safe_embedding_lookup': True
        }, config)

    custom_objects = {
        '_initializer': _initializer,
    }

    # use_safe_embedding_lookup might not be populated for legacy reasons.
    del config['use_safe_embedding_lookup']

    new_embedding_column = fc.EmbeddingColumn.from_config(
        config, custom_objects=custom_objects)
    self.assertEqual(embedding_column, new_embedding_column)
    self.assertIsNot(categorical_column,
                     new_embedding_column.categorical_column)

    new_embedding_column = fc.EmbeddingColumn.from_config(
        config,
        custom_objects=custom_objects,
        columns_by_name={
            serialization._column_name_with_class_name(categorical_column):
                categorical_column
        })
    self.assertEqual(embedding_column, new_embedding_column)
    self.assertIs(categorical_column, new_embedding_column.categorical_column)


class SharedEmbeddingColumnTest(test.TestCase, parameterized.TestCase):

  def test_defaults(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      embedding_column_b, embedding_column_a = fc.shared_embedding_columns_v2(
          [categorical_column_b, categorical_column_a],
          dimension=embedding_dimension)
      self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
      self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
      self.assertIsNone(embedding_column_a.max_norm)
      self.assertIsNone(embedding_column_b.max_norm)
      self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
      self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
      self.assertEqual((embedding_dimension,),
                       embedding_column_a.variable_shape)
      self.assertEqual((embedding_dimension,),
                       embedding_column_b.variable_shape)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column_a.parse_example_spec)
      self.assertEqual({
          'bbb': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column_b.parse_example_spec)

  def test_all_constructor_args(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          combiner='my_combiner',
          initializer=lambda: 'my_initializer',
          shared_embedding_collection_name='shared_embedding_collection_name',
          ckpt_to_load_from='my_ckpt',
          tensor_name_in_ckpt='my_ckpt_tensor',
          max_norm=42.,
          trainable=False)
      self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
      self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
      self.assertEqual(42., embedding_column_a.max_norm)
      self.assertEqual(42., embedding_column_b.max_norm)
      self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
      self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
      self.assertEqual((embedding_dimension,),
                       embedding_column_a.variable_shape)
      self.assertEqual((embedding_dimension,),
                       embedding_column_b.variable_shape)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column_a.parse_example_spec)
      self.assertEqual({
          'bbb': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column_b.parse_example_spec)

  def test_deep_copy(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      original_a, _ = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          combiner='my_combiner',
          initializer=lambda: 'my_initializer',
          shared_embedding_collection_name='shared_embedding_collection_name',
          ckpt_to_load_from='my_ckpt',
          tensor_name_in_ckpt='my_ckpt_tensor',
          max_norm=42.,
          trainable=False)
      for embedding_column_a in (original_a, copy.deepcopy(original_a)):
        self.assertEqual('aaa', embedding_column_a.categorical_column.name)
        self.assertEqual(3, embedding_column_a.categorical_column.num_buckets)
        self.assertEqual({
            'aaa': parsing_ops.VarLenFeature(dtypes.int64)
        }, embedding_column_a.categorical_column.parse_example_spec)

        self.assertEqual(42., embedding_column_a.max_norm)
        self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
        self.assertEqual((embedding_dimension,),
                         embedding_column_a.variable_shape)
        self.assertEqual({
            'aaa': parsing_ops.VarLenFeature(dtypes.int64)
        }, embedding_column_a.parse_example_spec)

  def test_invalid_initializer(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      with self.assertRaisesRegex(ValueError, 'initializer must be callable'):
        fc.shared_embedding_columns_v2(
            [categorical_column_a, categorical_column_b],
            dimension=2,
            initializer='not_fn')

  def test_incompatible_column_type(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      categorical_column_c = fc.categorical_column_with_hash_bucket(
          key='ccc', hash_bucket_size=3)
      with self.assertRaisesRegex(
          ValueError, 'all categorical_columns must have the same type.*'
          'IdentityCategoricalColumn.*HashedCategoricalColumn'):
        fc.shared_embedding_columns_v2(
            [categorical_column_a, categorical_column_b, categorical_column_c],
            dimension=2)

  def test_weighted_categorical_column_ok(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      weighted_categorical_column_a = fc.weighted_categorical_column(
          categorical_column_a, weight_feature_key='aaa_weights')
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      weighted_categorical_column_b = fc.weighted_categorical_column(
          categorical_column_b, weight_feature_key='bbb_weights')
      fc.shared_embedding_columns_v2(
          [weighted_categorical_column_a, categorical_column_b], dimension=2)
      fc.shared_embedding_columns_v2(
          [categorical_column_a, weighted_categorical_column_b], dimension=2)
      fc.shared_embedding_columns_v2(
          [weighted_categorical_column_a, weighted_categorical_column_b],
          dimension=2)

  def test_parse_example(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      a = fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
      b = fc.categorical_column_with_vocabulary_list(
          key='bbb', vocabulary_list=('omar', 'stringer', 'marlo'))
      a_embedded, b_embedded = fc.shared_embedding_columns_v2([a, b],
                                                              dimension=2)
      data = example_pb2.Example(
          features=feature_pb2.Features(
              feature={
                  'aaa':
                      feature_pb2.Feature(
                          bytes_list=feature_pb2.BytesList(
                              value=[b'omar', b'stringer'])),
                  'bbb':
                      feature_pb2.Feature(
                          bytes_list=feature_pb2.BytesList(
                              value=[b'stringer', b'marlo'])),
              }))
      features = parsing_ops.parse_example(
          serialized=[data.SerializeToString()],
          features=fc.make_parse_example_spec_v2([a_embedded, b_embedded]))
      self.assertIn('aaa', features)
      self.assertIn('bbb', features)

      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([b'omar', b'stringer'], dtype=np.object_),
              dense_shape=[1, 2]), self.evaluate(features['aaa']))
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([b'stringer', b'marlo'], dtype=np.object_),
              dense_shape=[1, 2]), self.evaluate(features['bbb']))

  def test_transform_feature(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      a = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
      b = fc.categorical_column_with_identity(key='bbb', num_buckets=3)
      a_embedded, b_embedded = fc.shared_embedding_columns_v2([a, b],
                                                              dimension=2)
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
      outputs = fc._transform_features_v2(features,
                                          [a, a_embedded, b, b_embedded], None)
      output_a = outputs[a]
      output_a_embedded = outputs[a_embedded]
      output_b = outputs[b]
      output_b_embedded = outputs[b_embedded]

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      _assert_sparse_tensor_value(self, self.evaluate(output_a),
                                  self.evaluate(output_a_embedded))
      _assert_sparse_tensor_value(self, self.evaluate(output_b),
                                  self.evaluate(output_b_embedded))

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True,
          'partition_variables': False,
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False,
          'partition_variables': False,
      }, {
          'testcase_name': 'use_safe_embedding_lookup_partitioned',
          'use_safe_embedding_lookup': True,
          'partition_variables': True,
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup_partitioned',
          'use_safe_embedding_lookup': False,
          'partition_variables': True,
      })
  def test_get_dense_tensor(self, use_safe_embedding_lookup,
                            partition_variables):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 4
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
          (7., 11.),  # id 2
          (9., 13.)  # id 3
      )

      def _initializer(shape, dtype, partition_info=None):
        self.assertEqual(dtypes.float32, dtype)
        if partition_variables:
          assert partition_info is not None
          self.assertEqual([vocabulary_size, embedding_dimension],
                           partition_info.full_shape)
          self.assertAllEqual((2, embedding_dimension), shape)
          return array_ops.slice(
              embedding_values, partition_info.var_offset, shape
          )
        else:
          self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
          self.assertIsNone(partition_info)
          return embedding_values

      # Expected lookup result, using combiner='mean'.
      expected_lookups_a = (
          # example 0:
          (7., 11.),  # ids [2], embedding = [7, 11]
          # example 1:
          (2., 3.5),  # ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
      )
      if use_safe_embedding_lookup:
        expected_lookups_b = (
            # example 0:
            (1., 2.),  # ids [0], embedding = [1, 2]
            # example 1:
            (0., 0.),  # ids [], embedding = [0, 0]
        )
      else:
        expected_lookups_b = (
            # example 0:
            (1., 2.),  # ids [0], embedding = [1, 2]
        )

      # Build columns.
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)

      partitioner = None
      if partition_variables:
        partitioner = partitioned_variables.fixed_size_partitioner(2, axis=0)

      with variable_scope.variable_scope('vars', partitioner=partitioner):
        embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
            [categorical_column_a, categorical_column_b],
            dimension=embedding_dimension,
            initializer=_initializer,
            use_safe_embedding_lookup=use_safe_embedding_lookup)
        # Provide sparse input and get dense result.
        embedding_lookup_a = embedding_column_a.get_dense_tensor(
            fc.FeatureTransformationCache(input_features), None)
        embedding_lookup_b = embedding_column_b.get_dense_tensor(
            fc.FeatureTransformationCache(input_features), None)

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      if partition_variables:
        self.assertCountEqual(('vars/aaa_bbb_shared_embedding/part_0:0',
                               'vars/aaa_bbb_shared_embedding/part_1:0'),
                              tuple([v.name for v in global_vars]))
        embedding_var = array_ops.concat(global_vars, axis=0)
      else:
        self.assertCountEqual(('vars/aaa_bbb_shared_embedding:0',),
                              tuple([v.name for v in global_vars]))
        embedding_var = global_vars[0]

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(embedding_var))
      self.assertAllEqual(expected_lookups_a, self.evaluate(embedding_lookup_a))
      self.assertAllEqual(expected_lookups_b, self.evaluate(embedding_lookup_b))

      if use_safe_embedding_lookup:
        self.assertIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])
      else:
        self.assertNotIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False
      })
  def test_get_dense_tensor_valid(self, use_safe_embedding_lookup):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 3
      # -1 values are ignored.
      input_a = np.array([
          [2, 1],  # example 0, ids [2, 1]
          [0, -1]
      ])  # example 1, ids [0]
      input_b = np.array([
          [1, -1],  # example 0, ids [1]
          [1, 2]
      ])  # example 1, ids [1, 2]
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
          (5., 8.),  # ids [2, 1], embedding =  mean([3, 5] + [7, 11]) = [5, 8]
          # example 1:
          (1., 2),  # ids [0], embedding = [1, 2]
      )
      expected_lookups_b = (
          # example 0:
          (3., 5.),  # ids [1], embedding = [3, 5]
          # example 1:
          (5., 8.),  # ids [1, 2], embedding = mean([3, 5] + [7, 11]) = [5, 8]
      )

      # Build columns.
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer,
          use_safe_embedding_lookup=use_safe_embedding_lookup)

      # Provide sparse input and get dense result.
      embedding_lookup_a = embedding_column_a.get_dense_tensor(
          fc.FeatureTransformationCache(input_features), None)
      embedding_lookup_b = embedding_column_b.get_dense_tensor(
          fc.FeatureTransformationCache(input_features), None)

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('aaa_bbb_shared_embedding:0',),
                            tuple([v.name for v in global_vars]))
      embedding_var = global_vars[0]

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(embedding_var))
      self.assertAllEqual(expected_lookups_a, self.evaluate(embedding_lookup_a))
      self.assertAllEqual(expected_lookups_b, self.evaluate(embedding_lookup_b))
      if use_safe_embedding_lookup:
        self.assertIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])
      else:
        self.assertNotIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False
      })
  def test_get_dense_tensor_placeholder_inputs(self, use_safe_embedding_lookup):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
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
      # Specify shape, because dense input must have rank specified.
      input_a_placeholder = array_ops.placeholder(
          dtype=dtypes.int64, shape=[None, 3])
      input_b_placeholder = array_ops.placeholder(
          dtype=dtypes.int64, shape=[None, 3])
      input_features = {
          'aaa': input_a_placeholder,
          'bbb': input_b_placeholder,
      }
      feed_dict = {
          input_a_placeholder: input_a,
          input_b_placeholder: input_b,
      }

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

      # Build columns.
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer,
          use_safe_embedding_lookup=use_safe_embedding_lookup)

      # Provide sparse input and get dense result.
      embedding_lookup_a = embedding_column_a.get_dense_tensor(
          fc.FeatureTransformationCache(input_features), None)
      embedding_lookup_b = embedding_column_b.get_dense_tensor(
          fc.FeatureTransformationCache(input_features), None)
      if use_safe_embedding_lookup:
        self.assertIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])
      else:
        self.assertNotIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])

      with _initialized_session() as sess:
        sess.run([embedding_lookup_a, embedding_lookup_b], feed_dict=feed_dict)

  def test_serialization(self):
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      def _initializer(shape, dtype, partition_info=None):
        del shape, dtype, partition_info
        return ValueError('Not expected to be called')

      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=2,
          initializer=_initializer)

      self.assertEqual([categorical_column_a], embedding_column_a.parents)
      self.assertEqual([categorical_column_b], embedding_column_b.parents)
      # TODO(rohanj): Add tests for (from|get)_config once implemented


class WeightedCategoricalColumnTest(test.TestCase):

  def test_defaults(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    self.assertEqual('ids_weighted_by_values', column.name)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'ids': parsing_ops.VarLenFeature(dtypes.int64),
        'values': parsing_ops.VarLenFeature(dtypes.float32)
    }, column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_is_v2_column(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc_old._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    self.assertFalse(column._is_v2_column)

  def test_deep_copy(self):
    """Tests deepcopy of categorical_column_with_hash_bucket."""
    original = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('ids_weighted_by_values', column.name)
      self.assertEqual(3, column.num_buckets)
      self.assertEqual({
          'ids': parsing_ops.VarLenFeature(dtypes.int64),
          'values': parsing_ops.VarLenFeature(dtypes.float32)
      }, column.parse_example_spec)

  def test_invalid_dtype_none(self):
    with self.assertRaisesRegex(ValueError, 'is not convertible to float'):
      fc.weighted_categorical_column(
          categorical_column=fc.categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values',
          dtype=None)

  def test_invalid_dtype_string(self):
    with self.assertRaisesRegex(ValueError, 'is not convertible to float'):
      fc.weighted_categorical_column(
          categorical_column=fc.categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values',
          dtype=dtypes.string)

  def test_invalid_input_dtype(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    strings = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'Bad dtype'):
      fc._transform_features_v2({
          'ids': strings,
          'values': strings
      }, (column,), None)

  def test_column_name_collision(self):
    with self.assertRaisesRegex(ValueError, r'Parse config.*already exists'):
      fc.weighted_categorical_column(
          categorical_column=fc.categorical_column_with_identity(
              key='aaa', num_buckets=3),
          weight_feature_key='aaa').parse_example_spec()

  def test_missing_weights(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError,
                                'values is not in features dictionary'):
      fc._transform_features_v2({'ids': inputs}, (column,), None)

  def test_parse_example(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_weighted = fc.weighted_categorical_column(a, weight_feature_key='weights')
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer'])),
                'weights':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[1., 10.]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a_weighted]))
    self.assertIn('aaa', features)
    self.assertIn('weights', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([1., 10.], dtype=np.float32),
            dense_shape=[1, 2]), self.evaluate(features['weights']))

  def test_transform_features(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)), values=(0, 1, 0), dense_shape=(2, 2))
    weights = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0.5, 1.0, 0.1),
        dense_shape=(2, 2))
    id_tensor, weight_tensor = fc._transform_features_v2({
        'ids': inputs,
        'values': weights,
    }, (column,), None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array(inputs.values, dtype=np.int64),
            dense_shape=inputs.dense_shape), self.evaluate(id_tensor))
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=weights.indices,
            values=np.array(weights.values, dtype=np.float32),
            dense_shape=weights.dense_shape), self.evaluate(weight_tensor))

  def test_transform_features_dense_input(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    weights = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0.5, 1.0, 0.1),
        dense_shape=(2, 2))
    id_tensor, weight_tensor = fc._transform_features_v2({
        'ids': ((0, -1), (1, 0)),
        'values': weights,
    }, (column,), None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((0, 1, 0), dtype=np.int64),
            dense_shape=(2, 2)), self.evaluate(id_tensor))
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=weights.indices,
            values=np.array(weights.values, dtype=np.float32),
            dense_shape=weights.dense_shape), self.evaluate(weight_tensor))

  def test_transform_features_dense_weights(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)), values=(2, 1, 0), dense_shape=(2, 2))
    id_tensor, weight_tensor = fc._transform_features_v2({
        'ids': inputs,
        'values': ((.5, 0.), (1., .1)),
    }, (column,), None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array(inputs.values, dtype=np.int64),
            dense_shape=inputs.dense_shape), self.evaluate(id_tensor))
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((.5, 1., .1), dtype=np.float32),
            dense_shape=(2, 2)), self.evaluate(weight_tensor))

  def test_old_linear_model(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          'ids':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(.5, 1., .1),
                  dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] * weights[0, 0] = 1 * .5 = .5
      # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
      # = 3*1 + 2*.1 = 3+.2 = 3.2
      self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  def test_old_linear_model_mismatched_shape(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError,
                                  r'Dimensions.*are not compatible'):
        fc_old.linear_model({
            'ids':
                sparse_tensor.SparseTensorValue(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=(0, 2, 1),
                    dense_shape=(2, 2)),
            'values':
                sparse_tensor.SparseTensorValue(
                    indices=((0, 0), (0, 1), (1, 0), (1, 1)),
                    values=(.5, 11., 1., .1),
                    dense_shape=(2, 2))
        }, (column,))

  def test_old_linear_model_mismatched_dense_values(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          'ids':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values': ((.5,), (1.,))
      }, (column,),
                                        sparse_combiner='mean')
      # Disabling the constant folding optimizer here since it changes the
      # error message differently on CPU and GPU.
      config = config_pb2.ConfigProto()
      config.graph_options.rewrite_options.constant_folding = (
          rewriter_config_pb2.RewriterConfig.OFF)
      with _initialized_session(config):
        with self.assertRaisesRegex(errors.OpError, 'Incompatible shapes'):
          self.evaluate(predictions)

  def test_old_linear_model_mismatched_dense_shape(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          'ids':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values': ((.5,), (1.,), (.1,))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] * weights[0, 0] = 1 * .5 = .5
      # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
      # = 3*1 + 2*.1 = 3+.2 = 3.2
      self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  def test_old_linear_model_old_categorical(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc_old._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          'ids':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(.5, 1., .1),
                  dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] * weights[0, 0] = 1 * .5 = .5
      # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
      # = 3*1 + 2*.1 = 3+.2 = 3.2
      self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  # TODO(ptucker): Add test with embedding of weighted categorical.

  def test_serialization(self):
    categorical_column = fc.categorical_column_with_identity(
        key='ids', num_buckets=3)
    column = fc.weighted_categorical_column(
        categorical_column=categorical_column, weight_feature_key='weight')

    self.assertEqual([categorical_column, 'weight'], column.parents)

    config = column.get_config()
    self.assertEqual({
        'categorical_column': {
            'config': {
                'key': 'ids',
                'number_buckets': 3,
                'default_value': None
            },
            'class_name': 'IdentityCategoricalColumn'
        },
        'dtype': 'float32',
        'weight_feature_key': 'weight'
    }, config)

    self.assertEqual(column, fc.WeightedCategoricalColumn.from_config(config))

    new_column = fc.WeightedCategoricalColumn.from_config(
        config,
        columns_by_name={
            serialization._column_name_with_class_name(categorical_column):
                categorical_column
        })
    self.assertEqual(column, new_column)
    self.assertIs(categorical_column, new_column.categorical_column)

if __name__ == '__main__':
  test.main()
