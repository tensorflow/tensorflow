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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy

import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.client import session
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.feature_column.feature_column import _CategoricalColumn
from tensorflow.python.feature_column.feature_column import _DenseColumn
from tensorflow.python.feature_column.feature_column import _FeatureColumn
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column.feature_column import _transform_features
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


def _initialized_session():
  sess = session.Session()
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


class LazyColumnTest(test.TestCase):

  def test_transormations_called_once(self):

    class TransformCounter(_FeatureColumn):

      def __init__(self):
        self.num_transform = 0

      @property
      def name(self):
        return 'TransformCounter'

      def _transform_feature(self, cache):
        self.num_transform += 1  # Count transform calls.
        return cache.get('a')

      @property
      def _parse_example_spec(self):
        pass

    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    column = TransformCounter()
    self.assertEqual(0, column.num_transform)
    builder.get(column)
    self.assertEqual(1, column.num_transform)
    builder.get(column)
    self.assertEqual(1, column.num_transform)

  def test_returns_transform_output(self):

    class Transformer(_FeatureColumn):

      @property
      def name(self):
        return 'Transformer'

      def _transform_feature(self, cache):
        return 'Output'

      @property
      def _parse_example_spec(self):
        pass

    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    column = Transformer()
    self.assertEqual('Output', builder.get(column))
    self.assertEqual('Output', builder.get(column))

  def test_does_not_pollute_given_features_dict(self):

    class Transformer(_FeatureColumn):

      @property
      def name(self):
        return 'Transformer'

      def _transform_feature(self, cache):
        return 'Output'

      @property
      def _parse_example_spec(self):
        pass

    features = {'a': [[2], [3.]]}
    builder = _LazyBuilder(features=features)
    builder.get(Transformer())
    self.assertEqual(['a'], list(features.keys()))

  def test_error_if_feature_is_not_found(self):
    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    with self.assertRaisesRegexp(ValueError,
                                 'bbb is not in features dictionary'):
      builder.get('bbb')

  def test_not_supported_feature_column(self):

    class NotAProperColumn(_FeatureColumn):

      @property
      def name(self):
        return 'NotAProperColumn'

      def _transform_feature(self, cache):
        # It should return not None.
        pass

      @property
      def _parse_example_spec(self):
        pass

    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    with self.assertRaisesRegexp(ValueError,
                                 'NotAProperColumn is not supported'):
      builder.get(NotAProperColumn())

  def test_key_should_be_string_or_feature_colum(self):

    class NotAFeatureColumn(object):
      pass

    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    with self.assertRaisesRegexp(
        TypeError, '"key" must be either a "str" or "_FeatureColumn".'):
      builder.get(NotAFeatureColumn())


class NumericColumnTest(test.TestCase):

  def test_defaults(self):
    a = fc.numeric_column('aaa')
    self.assertEqual('aaa', a.key)
    self.assertEqual((1,), a.shape)
    self.assertIsNone(a.default_value)
    self.assertEqual(dtypes.float32, a.dtype)
    self.assertIsNone(a.normalizer_fn)

  def test_shape_saved_as_tuple(self):
    a = fc.numeric_column('aaa', shape=[1, 2], default_value=[[3, 2.]])
    self.assertEqual((1, 2), a.shape)

  def test_default_value_saved_as_tuple(self):
    a = fc.numeric_column('aaa', default_value=4.)
    self.assertEqual((4.,), a.default_value)
    a = fc.numeric_column('aaa', shape=[1, 2], default_value=[[3, 2.]])
    self.assertEqual(((3., 2.),), a.default_value)

  def test_shape_and_default_value_compatibility(self):
    fc.numeric_column('aaa', shape=[2], default_value=[1, 2.])
    with self.assertRaisesRegexp(ValueError, 'The shape of default_value'):
      fc.numeric_column('aaa', shape=[2], default_value=[1, 2, 3.])
    fc.numeric_column(
        'aaa', shape=[3, 2], default_value=[[2, 3], [1, 2], [2, 3.]])
    with self.assertRaisesRegexp(ValueError, 'The shape of default_value'):
      fc.numeric_column(
          'aaa', shape=[3, 1], default_value=[[2, 3], [1, 2], [2, 3.]])
    with self.assertRaisesRegexp(ValueError, 'The shape of default_value'):
      fc.numeric_column(
          'aaa', shape=[3, 3], default_value=[[2, 3], [1, 2], [2, 3.]])

  def test_default_value_type_check(self):
    fc.numeric_column(
        'aaa', shape=[2], default_value=[1, 2.], dtype=dtypes.float32)
    fc.numeric_column(
        'aaa', shape=[2], default_value=[1, 2], dtype=dtypes.int32)
    with self.assertRaisesRegexp(TypeError, 'must be compatible with dtype'):
      fc.numeric_column(
          'aaa', shape=[2], default_value=[1, 2.], dtype=dtypes.int32)
    with self.assertRaisesRegexp(TypeError,
                                 'default_value must be compatible with dtype'):
      fc.numeric_column('aaa', default_value=['string'])

  def test_shape_must_be_positive_integer(self):
    with self.assertRaisesRegexp(TypeError, 'shape dimensions must be integer'):
      fc.numeric_column(
          'aaa', shape=[
              1.0,
          ])

    with self.assertRaisesRegexp(ValueError,
                                 'shape dimensions must be greater than 0'):
      fc.numeric_column(
          'aaa', shape=[
              0,
          ])

  def test_dtype_is_convertable_to_float(self):
    with self.assertRaisesRegexp(ValueError,
                                 'dtype must be convertible to float'):
      fc.numeric_column('aaa', dtype=dtypes.string)

  def test_scalar_deafult_value_fills_the_shape(self):
    a = fc.numeric_column('aaa', shape=[2, 3], default_value=2.)
    self.assertEqual(((2., 2., 2.), (2., 2., 2.)), a.default_value)

  def test_parse_spec(self):
    a = fc.numeric_column('aaa', shape=[2, 3], dtype=dtypes.int32)
    self.assertEqual({
        'aaa': parsing_ops.FixedLenFeature((2, 3), dtype=dtypes.int32)
    }, a._parse_example_spec)

  def test_parse_example_no_default_value(self):
    price = fc.numeric_column('price', shape=[2])
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'price':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([price]))
    self.assertIn('price', features)
    with self.test_session():
      self.assertAllEqual([[20., 110.]], features['price'].eval())

  def test_parse_example_with_default_value(self):
    price = fc.numeric_column('price', shape=[2], default_value=11.)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'price':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.]))
        }))
    no_data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'something_else':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString(),
                    no_data.SerializeToString()],
        features=fc.make_parse_example_spec([price]))
    self.assertIn('price', features)
    with self.test_session():
      self.assertAllEqual([[20., 110.], [11., 11.]], features['price'].eval())

  def test_normalizer_fn_must_be_callable(self):
    with self.assertRaisesRegexp(TypeError, 'must be a callable'):
      fc.numeric_column('price', normalizer_fn='NotACallable')

  def test_normalizer_fn_transform_feature(self):

    def _increment_two(input_tensor):
      return input_tensor + 2.

    price = fc.numeric_column('price', shape=[2], normalizer_fn=_increment_two)
    output = _transform_features({'price': [[1., 2.], [5., 6.]]}, [price])
    with self.test_session():
      self.assertAllEqual([[3., 4.], [7., 8.]], output[price].eval())

  def test_get_dense_tensor(self):

    def _increment_two(input_tensor):
      return input_tensor + 2.

    price = fc.numeric_column('price', shape=[2], normalizer_fn=_increment_two)
    builder = _LazyBuilder({'price': [[1., 2.], [5., 6.]]})
    self.assertEqual(builder.get(price), price._get_dense_tensor(builder))

  def test_sparse_tensor_not_supported(self):
    price = fc.numeric_column('price')
    builder = _LazyBuilder({
        'price':
            sparse_tensor.SparseTensor(
                indices=[[0, 0]], values=[0.3], dense_shape=[1, 1])
    })
    with self.assertRaisesRegexp(ValueError, 'must be a Tensor'):
      price._transform_feature(builder)

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

  def test_linear_model(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], bias.eval())
        self.assertAllClose([[0.]], price_var.eval())
        self.assertAllClose([[0.], [0.]], predictions.eval())
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[10.], [50.]], predictions.eval())


class BucketizedColumnTest(test.TestCase):

  def test_invalid_source_column_type(self):
    a = fc.categorical_column_with_hash_bucket('aaa', hash_bucket_size=10)
    with self.assertRaisesRegexp(
        ValueError,
        'source_column must be a column generated with numeric_column'):
      fc.bucketized_column(a, boundaries=[0, 1])

  def test_invalid_source_column_shape(self):
    a = fc.numeric_column('aaa', shape=[2, 3])
    with self.assertRaisesRegexp(
        ValueError, 'source_column must be one-dimensional column'):
      fc.bucketized_column(a, boundaries=[0, 1])

  def test_invalid_boundaries(self):
    a = fc.numeric_column('aaa')
    with self.assertRaisesRegexp(
        ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=None)
    with self.assertRaisesRegexp(
        ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=1.)
    with self.assertRaisesRegexp(
        ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=[1, 0])
    with self.assertRaisesRegexp(
        ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=[1, 1])

  def test_name(self):
    a = fc.numeric_column('aaa', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    self.assertEqual('aaa_bucketized', b.name)

  def test_parse_spec(self):
    a = fc.numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    self.assertEqual({
        'aaa': parsing_ops.FixedLenFeature((2,), dtype=dtypes.int32)
    }, b._parse_example_spec)

  def test_variable_shape(self):
    a = fc.numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    # Column 'aaa` has shape [2] times three buckets -> variable_shape=[2, 3].
    self.assertAllEqual((2, 3), b._variable_shape)

  def test_num_buckets(self):
    a = fc.numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    # Column 'aaa` has shape [2] times three buckets -> num_buckets=6.
    self.assertEqual(6, b._num_buckets)

  def test_parse_example(self):
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 50])
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'price':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([bucketized_price]))
    self.assertIn('price', features)
    with self.test_session():
      self.assertAllEqual([[20., 110.]], features['price'].eval())

  def test_transform_feature(self):
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformed_tensor = _transform_features({
          'price': [[-1., 1.], [5., 6.]]
      }, [bucketized_price])
      with _initialized_session():
        self.assertAllEqual([[0, 1], [3, 4]],
                            transformed_tensor[bucketized_price].eval())

  def test_get_dense_tensor_one_input_value(self):
    """Tests _get_dense_tensor() for input with shape=[1]."""
    price = fc.numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      builder = _LazyBuilder({'price': [[-1.], [1.], [5.], [6.]]})
      with _initialized_session():
        bucketized_price_tensor = bucketized_price._get_dense_tensor(builder)
        self.assertAllClose(
            # One-hot tensor.
            [[[1., 0., 0., 0., 0.]],
             [[0., 1., 0., 0., 0.]],
             [[0., 0., 0., 1., 0.]],
             [[0., 0., 0., 0., 1.]]],
            bucketized_price_tensor.eval())

  def test_get_dense_tensor_two_input_values(self):
    """Tests _get_dense_tensor() for input with shape=[2]."""
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      builder = _LazyBuilder({'price': [[-1., 1.], [5., 6.]]})
      with _initialized_session():
        bucketized_price_tensor = bucketized_price._get_dense_tensor(builder)
        self.assertAllClose(
            # One-hot tensor.
            [[[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]],
             [[0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]]],
            bucketized_price_tensor.eval())

  def test_get_sparse_tensors_one_input_value(self):
    """Tests _get_sparse_tensors() for input with shape=[1]."""
    price = fc.numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      builder = _LazyBuilder({'price': [[-1.], [1.], [5.], [6.]]})
      with _initialized_session() as sess:
        id_weight_pair = bucketized_price._get_sparse_tensors(builder)
        self.assertIsNone(id_weight_pair.weight_tensor)
        id_tensor_value = sess.run(id_weight_pair.id_tensor)
        self.assertAllEqual(
            [[0, 0], [1, 0], [2, 0], [3, 0]], id_tensor_value.indices)
        self.assertAllEqual([0, 1, 3, 4], id_tensor_value.values)
        self.assertAllEqual([4, 1], id_tensor_value.dense_shape)

  def test_get_sparse_tensors_two_input_values(self):
    """Tests _get_sparse_tensors() for input with shape=[2]."""
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      builder = _LazyBuilder({'price': [[-1., 1.], [5., 6.]]})
      with _initialized_session() as sess:
        id_weight_pair = bucketized_price._get_sparse_tensors(builder)
        self.assertIsNone(id_weight_pair.weight_tensor)
        id_tensor_value = sess.run(id_weight_pair.id_tensor)
        self.assertAllEqual(
            [[0, 0], [0, 1], [1, 0], [1, 1]], id_tensor_value.indices)
        # Values 0-4 correspond to the first column of the input price.
        # Values 5-9 correspond to the second column of the input price.
        self.assertAllEqual([0, 6, 3, 9], id_tensor_value.values)
        self.assertAllEqual([2, 2], id_tensor_value.dense_shape)

  def test_sparse_tensor_input_not_supported(self):
    price = fc.numeric_column('price')
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 1])
    builder = _LazyBuilder({
        'price':
            sparse_tensor.SparseTensor(
                indices=[[0, 0]], values=[0.3], dense_shape=[1, 1])
    })
    with self.assertRaisesRegexp(ValueError, 'must be a Tensor'):
      bucketized_price._transform_feature(builder)

  def test_deep_copy(self):
    a = fc.numeric_column('aaa', shape=[2])
    a_bucketized = fc.bucketized_column(a, boundaries=[0, 1])
    a_bucketized_copy = copy.deepcopy(a_bucketized)
    self.assertEqual(a_bucketized_copy.name, 'aaa_bucketized')
    self.assertAllEqual(a_bucketized_copy._variable_shape, (2, 3))
    self.assertEqual(a_bucketized_copy.boundaries, (0, 1))

  def test_linear_model_one_input_value(self):
    """Tests linear_model() for input with shape=[1]."""
    price = fc.numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1.], [1.], [5.], [6.]]}
      predictions = fc.linear_model(features, [bucketized_price])
      bias = get_linear_model_bias()
      bucketized_price_var = get_linear_model_column_var(bucketized_price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], bias.eval())
        # One weight variable per bucket, all initialized to zero.
        self.assertAllClose(
            [[0.], [0.], [0.], [0.], [0.]], bucketized_price_var.eval())
        self.assertAllClose([[0.], [0.], [0.], [0.]], predictions.eval())
        sess.run(bucketized_price_var.assign(
            [[10.], [20.], [30.], [40.], [50.]]))
        # price -1. is in the 0th bucket, whose weight is 10.
        # price 1. is in the 1st bucket, whose weight is 20.
        # price 5. is in the 3rd bucket, whose weight is 40.
        # price 6. is in the 4th bucket, whose weight is 50.
        self.assertAllClose([[10.], [20.], [40.], [50.]], predictions.eval())
        sess.run(bias.assign([1.]))
        self.assertAllClose([[11.], [21.], [41.], [51.]], predictions.eval())

  def test_linear_model_two_input_values(self):
    """Tests linear_model() for input with shape=[2]."""
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1., 1.], [5., 6.]]}
      predictions = fc.linear_model(features, [bucketized_price])
      bias = get_linear_model_bias()
      bucketized_price_var = get_linear_model_column_var(bucketized_price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], bias.eval())
        # One weight per bucket per input column, all initialized to zero.
        self.assertAllClose(
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
            bucketized_price_var.eval())
        self.assertAllClose([[0.], [0.]], predictions.eval())
        sess.run(bucketized_price_var.assign(
            [[10.], [20.], [30.], [40.], [50.],
             [60.], [70.], [80.], [90.], [100.]]))
        # 1st example:
        #   price -1. is in the 0th bucket, whose weight is 10.
        #   price 1. is in the 6th bucket, whose weight is 70.
        # 2nd example:
        #   price 5. is in the 3rd bucket, whose weight is 40.
        #   price 6. is in the 9th bucket, whose weight is 100.
        self.assertAllClose([[80.], [140.]], predictions.eval())
        sess.run(bias.assign([1.]))
        self.assertAllClose([[81.], [141.]], predictions.eval())


class HashedCategoricalColumnTest(test.TestCase):

  def test_defaults(self):
    a = fc.categorical_column_with_hash_bucket('aaa', 10)
    self.assertEqual('aaa', a.name)
    self.assertEqual('aaa', a.key)
    self.assertEqual(10, a.hash_bucket_size)
    self.assertEqual(dtypes.string, a.dtype)

  def test_bucket_size_should_be_given(self):
    with self.assertRaisesRegexp(ValueError, 'hash_bucket_size must be set.'):
      fc.categorical_column_with_hash_bucket('aaa', None)

  def test_bucket_size_should_be_positive(self):
    with self.assertRaisesRegexp(ValueError,
                                 'hash_bucket_size must be at least 1'):
      fc.categorical_column_with_hash_bucket('aaa', 0)

  def test_dtype_should_be_string_or_integer(self):
    fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.string)
    fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.int32)
    with self.assertRaisesRegexp(ValueError, 'dtype must be string or integer'):
      fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.float32)

  def test_deep_copy(self):
    original = fc.categorical_column_with_hash_bucket('aaa', 10)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(10, column.hash_bucket_size)
      self.assertEqual(10, column._num_buckets)
      self.assertEqual(dtypes.string, column.dtype)

  def test_parse_spec_string(self):
    a = fc.categorical_column_with_hash_bucket('aaa', 10)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, a._parse_example_spec)

  def test_parse_spec_int(self):
    a = fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.int32)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, a._parse_example_spec)

  def test_parse_example(self):
    a = fc.categorical_column_with_hash_bucket('aaa', 10)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    with self.test_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([b'omar', b'stringer'], dtype=np.object_),
              dense_shape=[1, 2]),
          features['aaa'].eval())

  def test_strings_should_be_hashed(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    outputs = _transform_features({'wire': wire_tensor}, [hashed_sparse])
    output = outputs[hashed_sparse]
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [6, 4, 1]
    with self.test_session():
      self.assertEqual(dtypes.int64, output.values.dtype)
      self.assertAllEqual(expected_values, output.values.eval())
      self.assertAllEqual(wire_tensor.indices.eval(), output.indices.eval())
      self.assertAllEqual(wire_tensor.dense_shape.eval(),
                          output.dense_shape.eval())

  def test_tensor_dtype_should_be_string_or_integer(self):
    string_fc = fc.categorical_column_with_hash_bucket(
        'a_string', 10, dtype=dtypes.string)
    int_fc = fc.categorical_column_with_hash_bucket(
        'a_int', 10, dtype=dtypes.int32)
    float_fc = fc.categorical_column_with_hash_bucket(
        'a_float', 10, dtype=dtypes.string)
    int_tensor = sparse_tensor.SparseTensor(
        values=[101],
        indices=[[0, 0]],
        dense_shape=[1, 1])
    string_tensor = sparse_tensor.SparseTensor(
        values=['101'],
        indices=[[0, 0]],
        dense_shape=[1, 1])
    float_tensor = sparse_tensor.SparseTensor(
        values=[101.],
        indices=[[0, 0]],
        dense_shape=[1, 1])
    builder = _LazyBuilder({
        'a_int': int_tensor,
        'a_string': string_tensor,
        'a_float': float_tensor
    })
    builder.get(string_fc)
    builder.get(int_fc)
    with self.assertRaisesRegexp(ValueError, 'dtype must be string or integer'):
      builder.get(float_fc)

  def test_dtype_should_match_with_tensor(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
    builder = _LazyBuilder({'wire': wire_tensor})
    with self.assertRaisesRegexp(ValueError, 'dtype must be compatible'):
      builder.get(hashed_sparse)

  def test_ints_should_be_hashed(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=[101, 201, 301],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    builder = _LazyBuilder({'wire': wire_tensor})
    output = builder.get(hashed_sparse)
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [3, 7, 5]
    with self.test_session():
      self.assertAllEqual(expected_values, output.values.eval())

  def test_int32_64_is_compatible(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=constant_op.constant([101, 201, 301], dtype=dtypes.int32),
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    builder = _LazyBuilder({'wire': wire_tensor})
    output = builder.get(hashed_sparse)
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [3, 7, 5]
    with self.test_session():
      self.assertAllEqual(expected_values, output.values.eval())

  def test_get_sparse_tensors(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    builder = _LazyBuilder({
        'wire':
            sparse_tensor.SparseTensor(
                values=['omar', 'stringer', 'marlo'],
                indices=[[0, 0], [1, 0], [1, 1]],
                dense_shape=[2, 2])
    })
    id_weight_pair = hashed_sparse._get_sparse_tensors(builder)
    self.assertIsNone(id_weight_pair.weight_tensor)
    self.assertEqual(builder.get(hashed_sparse), id_weight_pair.id_tensor)

  def test_get_sparse_tensors_weight_collections(self):
    column = fc.categorical_column_with_hash_bucket('aaa', 10)
    inputs = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': inputs
        }), weight_collections=('my_weights',))

    self.assertItemsEqual(
        [], ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
    self.assertItemsEqual([], ops.get_collection('my_weights'))

  def test_get_sparse_tensors_dense_input(self):
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    builder = _LazyBuilder({'wire': (('omar', ''), ('stringer', 'marlo'))})
    id_weight_pair = hashed_sparse._get_sparse_tensors(builder)
    self.assertIsNone(id_weight_pair.weight_tensor)
    self.assertEqual(builder.get(hashed_sparse), id_weight_pair.id_tensor)

  def test_linear_model(self):
    wire_column = fc.categorical_column_with_hash_bucket('wire', 4)
    self.assertEqual(4, wire_column._num_buckets)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          wire_column.name: sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=('marlo', 'skywalker', 'omar'),
              dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)
      with _initialized_session():
        self.assertAllClose((0.,), bias.eval())
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), wire_var.eval())
        self.assertAllClose(((0.,), (0.,)), predictions.eval())
        wire_var.assign(((1.,), (2.,), (3.,), (4.,))).eval()
        # 'marlo' -> 3: wire_var[3] = 4
        # 'skywalker' -> 2, 'omar' -> 2: wire_var[2] + wire_var[2] = 3+3 = 6
        self.assertAllClose(((4.,), (6.,)), predictions.eval())


class CrossedColumnTest(test.TestCase):

  def test_keys_empty(self):
    with self.assertRaisesRegexp(
        ValueError, 'keys must be a list with length > 1'):
      fc.crossed_column([], 10)

  def test_keys_length_one(self):
    with self.assertRaisesRegexp(
        ValueError, 'keys must be a list with length > 1'):
      fc.crossed_column(['a'], 10)

  def test_key_type_unsupported(self):
    with self.assertRaisesRegexp(ValueError, 'Unsupported key type'):
      fc.crossed_column(['a', fc.numeric_column('c')], 10)

    with self.assertRaisesRegexp(
        ValueError, 'categorical_column_with_hash_bucket is not supported'):
      fc.crossed_column(
          ['a', fc.categorical_column_with_hash_bucket('c', 10)], 10)

  def test_hash_bucket_size_negative(self):
    with self.assertRaisesRegexp(
        ValueError, 'hash_bucket_size must be > 1'):
      fc.crossed_column(['a', 'c'], -1)

  def test_hash_bucket_size_zero(self):
    with self.assertRaisesRegexp(
        ValueError, 'hash_bucket_size must be > 1'):
      fc.crossed_column(['a', 'c'], 0)

  def test_hash_bucket_size_none(self):
    with self.assertRaisesRegexp(
        ValueError, 'hash_bucket_size must be > 1'):
      fc.crossed_column(['a', 'c'], None)

  def test_name(self):
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)

    crossed2 = fc.crossed_column([b, 'c', crossed1], 10)
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
    }, crossed._parse_example_spec)

  def test_num_buckets(self):
    a = fc.numeric_column('a', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed = fc.crossed_column([b, 'c'], 15)
    self.assertEqual(15, crossed._num_buckets)

  def test_deep_copy(self):
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    crossed2 = fc.crossed_column([b, 'c', crossed1], 15, hash_key=5)
    crossed2_copy = copy.deepcopy(crossed2)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2_copy.name,)
    self.assertEqual(15, crossed2_copy.hash_bucket_size)
    self.assertEqual(5, crossed2_copy.hash_key)

  def test_parse_example(self):
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 50])
    price_cross_wire = fc.crossed_column([bucketized_price, 'wire'], 10)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'price':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.])),
            'wire':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer'])),
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([price_cross_wire]))
    self.assertIn('price', features)
    self.assertIn('wire', features)
    with self.test_session():
      self.assertAllEqual([[20., 110.]], features['price'].eval())
      wire_sparse = features['wire']
      self.assertAllEqual([[0, 0], [0, 1]], wire_sparse.indices.eval())
      # Use byte constants to pass the open-source test.
      self.assertAllEqual([b'omar', b'stringer'], wire_sparse.values.eval())
      self.assertAllEqual([1, 2], wire_sparse.dense_shape.eval())

  def test_transform_feature(self):
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 50])
    hash_bucket_size = 10
    price_cross_wire = fc.crossed_column(
        [bucketized_price, 'wire'], hash_bucket_size)
    features = {
        'price': constant_op.constant([[1., 2.], [5., 6.]]),
        'wire': sparse_tensor.SparseTensor(
            values=['omar', 'stringer', 'marlo'],
            indices=[[0, 0], [1, 0], [1, 1]],
            dense_shape=[2, 2]),
    }
    outputs = _transform_features(features, [price_cross_wire])
    output = outputs[price_cross_wire]
    with self.test_session() as sess:
      output_val = sess.run(output)
      self.assertAllEqual(
          [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]], output_val.indices)
      for val in output_val.values:
        self.assertIn(val, list(range(hash_bucket_size)))
      self.assertAllEqual([2, 4], output_val.dense_shape)

  def test_get_sparse_tensors(self):
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    crossed2 = fc.crossed_column([b, 'c', crossed1], 15, hash_key=5)
    with ops.Graph().as_default():
      builder = _LazyBuilder({
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
      id_weight_pair = crossed2._get_sparse_tensors(builder)
      with _initialized_session():
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        self.assertAllEqual(
            ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
             (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
             (1, 14), (1, 15)),
            id_tensor_eval.indices)
        # Check exact hashed output. If hashing changes this test will break.
        # All values are within [0, hash_bucket_size).
        expected_values = (
            6, 14, 0, 13, 8, 8, 10, 12, 2, 0, 1, 9, 8, 12, 2, 0, 10, 11)
        self.assertAllEqual(expected_values, id_tensor_eval.values)
        self.assertAllEqual((2, 16), id_tensor_eval.dense_shape)

  def test_get_sparse_tensors_simple(self):
    """Same as test_get_sparse_tensors, but with simpler values."""
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      builder = _LazyBuilder({
          'a':
              constant_op.constant(((-1., .5), (.5, 1.))),
          'c':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      })
      id_weight_pair = crossed._get_sparse_tensors(builder)
      with _initialized_session():
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        self.assertAllEqual(
            ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)),
            id_tensor_eval.indices)
        # Check exact hashed output. If hashing changes this test will break.
        # All values are within [0, hash_bucket_size).
        expected_values = (1, 0, 1, 3, 4, 2)
        self.assertAllEqual(expected_values, id_tensor_eval.values)
        self.assertAllEqual((2, 4), id_tensor_eval.dense_shape)

  def test_linear_model(self):
    """Tests linear_model.

    Uses data from test_get_sparse_tesnsors_simple.
    """
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          'a': constant_op.constant(((-1., .5), (.5, 1.))),
          'c': sparse_tensor.SparseTensor(
              indices=((0, 0), (1, 0), (1, 1)),
              values=['cA', 'cB', 'cC'],
              dense_shape=(2, 2)),
      }, (crossed,))
      bias = get_linear_model_bias()
      crossed_var = get_linear_model_column_var(crossed)
      with _initialized_session() as sess:
        self.assertAllClose((0.,), bias.eval())
        self.assertAllClose(
            ((0.,), (0.,), (0.,), (0.,), (0.,)), crossed_var.eval())
        self.assertAllClose(((0.,), (0.,)), predictions.eval())
        sess.run(crossed_var.assign(((1.,), (2.,), (3.,), (4.,), (5.,))))
        # Expected ids after cross = (1, 0, 1, 3, 4, 2)
        self.assertAllClose(((3.,), (14.,)), predictions.eval())
        sess.run(bias.assign((.1,)))
        self.assertAllClose(((3.1,), (14.1,)), predictions.eval())

  def test_linear_model_with_weights(self):
    class _TestColumnWithWeights(_CategoricalColumn):
      """Produces sparse IDs and sparse weights."""

      @property
      def name(self):
        return 'test_column'

      @property
      def _parse_example_spec(self):
        return {
            self.name: parsing_ops.VarLenFeature(dtypes.int32),
            '{}_weights'.format(self.name): parsing_ops.VarLenFeature(
                dtypes.float32),
            }

      @property
      def _num_buckets(self):
        return 5

      def _transform_feature(self, inputs):
        return (inputs.get(self.name),
                inputs.get('{}_weights'.format(self.name)))

      def _get_sparse_tensors(self, inputs, weight_collections=None,
                              trainable=None):
        """Populates both id_tensor and weight_tensor."""
        ids_and_weights = inputs.get(self)
        return _CategoricalColumn.IdWeightPair(
            id_tensor=ids_and_weights[0], weight_tensor=ids_and_weights[1])

    t = _TestColumnWithWeights()
    crossed = fc.crossed_column([t, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError,
          'crossed_column does not support weight_tensor.*{}'.format(t.name)):
        fc.linear_model({
            t.name: sparse_tensor.SparseTensor(
                indices=((0, 0), (1, 0), (1, 1)),
                values=[0, 1, 2],
                dense_shape=(2, 2)),
            '{}_weights'.format(t.name): sparse_tensor.SparseTensor(
                indices=((0, 0), (1, 0), (1, 1)),
                values=[1., 10., 2.],
                dense_shape=(2, 2)),
            'c': sparse_tensor.SparseTensor(
                indices=((0, 0), (1, 0), (1, 1)),
                values=['cA', 'cB', 'cC'],
                dense_shape=(2, 2)),
        }, (crossed,))


def get_linear_model_bias():
  with variable_scope.variable_scope('linear_model', reuse=True):
    return variable_scope.get_variable('bias_weights')


def get_linear_model_column_var(column):
  return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                            'linear_model/' + column.name)[0]


class LinearModelTest(test.TestCase):

  def test_raises_if_empty_feature_columns(self):
    with self.assertRaisesRegexp(ValueError,
                                 'feature_columns must not be empty'):
      fc.linear_model(features={}, feature_columns=[])

  def test_should_be_feature_column(self):
    with self.assertRaisesRegexp(ValueError, 'must be a _FeatureColumn'):
      fc.linear_model(features={'a': [[0]]}, feature_columns='NotSupported')

  def test_should_be_dense_or_categorical_column(self):

    class NotSupportedColumn(_FeatureColumn):

      @property
      def name(self):
        return 'NotSupportedColumn'

      def _transform_feature(self, cache):
        pass

      @property
      def _parse_example_spec(self):
        pass

    with self.assertRaisesRegexp(
        ValueError, 'must be either a _DenseColumn or _CategoricalColumn'):
      fc.linear_model(
          features={'a': [[0]]}, feature_columns=[NotSupportedColumn()])

  def test_does_not_support_dict_columns(self):
    with self.assertRaisesRegexp(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      fc.linear_model(
          features={'a': [[0]]}, feature_columns={'a': fc.numeric_column('a')})

  def test_raises_if_duplicate_name(self):
    with self.assertRaisesRegexp(
        ValueError, 'Duplicate feature column name found for columns'):
      fc.linear_model(
          features={'a': [[0]]},
          feature_columns=[fc.numeric_column('a'),
                           fc.numeric_column('a')])

  def test_dense_bias(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], bias.eval())
        sess.run(price_var.assign([[10.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[15.], [55.]], predictions.eval())

  def test_sparse_bias(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc.linear_model(features, [wire_cast])
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose([0.], bias.eval())
        self.assertAllClose([[0.], [0.], [0.], [0.]], wire_cast_var.eval())
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], predictions.eval())

  def test_dense_and_sparse_bias(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor, 'price': [[1.], [5.]]}
      predictions = fc.linear_model(features, [wire_cast, price])
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[1015.], [10065.]], predictions.eval())

  def test_dense_and_sparse_column(self):
    """When the column is both dense and sparse, uses sparse tensors."""

    class _DenseAndSparseColumn(_DenseColumn, _CategoricalColumn):

      @property
      def name(self):
        return 'dense_and_sparse_column'

      @property
      def _parse_example_spec(self):
        return {self.name: parsing_ops.VarLenFeature(self.dtype)}

      def _transform_feature(self, inputs):
        return inputs.get(self.name)

      @property
      def _variable_shape(self):
        raise ValueError('Should not use this method.')

      def _get_dense_tensor(self, inputs, weight_collections=None,
                            trainable=None):
        raise ValueError('Should not use this method.')

      @property
      def _num_buckets(self):
        return 4

      def _get_sparse_tensors(self, inputs, weight_collections=None,
                              trainable=None):
        sp_tensor = sparse_tensor.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1]],
            values=[2, 0, 3],
            dense_shape=[2, 2])
        return _CategoricalColumn.IdWeightPair(sp_tensor, None)

    dense_and_sparse_column = _DenseAndSparseColumn()
    with ops.Graph().as_default():
      sp_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {dense_and_sparse_column.name: sp_tensor}
      predictions = fc.linear_model(features, [dense_and_sparse_column])
      bias = get_linear_model_bias()
      dense_and_sparse_column_var = get_linear_model_column_var(
          dense_and_sparse_column)
      with _initialized_session() as sess:
        sess.run(dense_and_sparse_column_var.assign(
            [[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], predictions.eval())

  def test_dense_multi_output(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc.linear_model(features, [price], units=3)
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), bias.eval())
        self.assertAllClose(np.zeros((1, 3)), price_var.eval())
        sess.run(price_var.assign([[10., 100., 1000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[15., 106., 1007.], [55., 506., 5007.]],
                            predictions.eval())

  def test_sparse_multi_output(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc.linear_model(features, [wire_cast], units=3)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), bias.eval())
        self.assertAllClose(np.zeros((4, 3)), wire_cast_var.eval())
        sess.run(
            wire_cast_var.assign([[10., 11., 12.], [100., 110., 120.], [
                1000., 1100., 1200.
            ], [10000., 11000., 12000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[1005., 1106., 1207.], [10015., 11017., 12019.]],
                            predictions.eval())

  def test_dense_multi_dimension(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = fc.linear_model(features, [price])
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([[0.], [0.]], price_var.eval())
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], predictions.eval())

  def test_sparse_multi_rank(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = array_ops.sparse_placeholder(dtypes.string)
      wire_value = sparse_tensor.SparseTensorValue(
          values=['omar', 'stringer', 'marlo', 'omar'],  # hashed = [2, 0, 3, 2]
          indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1]],
          dense_shape=[2, 2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc.linear_model(features, [wire_cast])
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((4, 1)), wire_cast_var.eval())
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
      predictions = fc.linear_model(
          features, [wire_cast], sparse_combiner='mean')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [5010.]], predictions.eval())

  def test_dense_multi_dimension_multi_output(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = fc.linear_model(features, [price], units=3)
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), bias.eval())
        self.assertAllClose(np.zeros((2, 3)), price_var.eval())
        sess.run(price_var.assign([[1., 2., 3.], [10., 100., 1000.]]))
        sess.run(bias.assign([2., 3., 4.]))
        self.assertAllClose([[23., 205., 2007.], [67., 613., 6019.]],
                            predictions.eval())

  def test_raises_if_shape_mismatch(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc.linear_model(features, [price])
      with _initialized_session():
        with self.assertRaisesRegexp(Exception, 'requested shape has 4'):
          predictions.eval()

  def test_dense_reshaping(self):
    price = fc.numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      predictions = fc.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], bias.eval())
        self.assertAllClose([[0.], [0.]], price_var.eval())
        self.assertAllClose([[0.], [0.]], predictions.eval())
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], predictions.eval())

  def test_dense_multi_column(self):
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1., 2.], [5., 6.]],
          'price2': [[3.], [4.]]
      }
      predictions = fc.linear_model(features, [price1, price2])
      bias = get_linear_model_bias()
      price1_var = get_linear_model_column_var(price1)
      price2_var = get_linear_model_column_var(price2)
      with _initialized_session() as sess:
        self.assertAllClose([0.], bias.eval())
        self.assertAllClose([[0.], [0.]], price1_var.eval())
        self.assertAllClose([[0.]], price2_var.eval())
        self.assertAllClose([[0.], [0.]], predictions.eval())
        sess.run(price1_var.assign([[10.], [100.]]))
        sess.run(price2_var.assign([[1000.]]))
        sess.run(bias.assign([7.]))
        self.assertAllClose([[3217.], [4657.]], predictions.eval())

  def test_dense_collection(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc.linear_model(features, [price], weight_collections=['my-vars'])
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
      fc.linear_model(
          features, [wire_cast], weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, my_vars)
      self.assertIn(wire_cast_var, my_vars)

  def test_dense_trainable_default(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc.linear_model(features, [price])
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
      fc.linear_model(features, [wire_cast])
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, trainable_vars)
      self.assertIn(wire_cast_var, trainable_vars)

  def test_dense_trainable_false(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc.linear_model(features, [price], trainable=False)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_sparse_trainable_false(self):
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc.linear_model(features, [wire_cast], trainable=False)
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
      fc.linear_model(
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
      fc.linear_model(
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
    with self.assertRaisesRegexp(
        ValueError,
        'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
      fc.linear_model(features, [price1, price2])

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
          'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc.linear_model(features, [price1, price2, price3])

  def test_runtime_batch_size_mismatch(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      predictions = fc.linear_model(features, [price1, price2])
      with _initialized_session() as sess:
        with self.assertRaisesRegexp(errors.OpError,
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
      predictions = fc.linear_model(features, [price1, price2])
      with _initialized_session() as sess:
        sess.run(
            predictions,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  def test_with_numpy_input_fn(self):
    price = fc.numeric_column('price')
    price_buckets = fc.bucketized_column(price, boundaries=[0., 10., 100.,])
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])

    input_fn = numpy_io.numpy_input_fn(
        x={
            'price': np.array([-1., 2., 13., 104.]),
            'body-style': np.array(['sedan', 'hardtop', 'wagon', 'sedan']),
        },
        batch_size=2,
        shuffle=False)
    features = input_fn()
    net = fc.linear_model(features, [price_buckets, body_style])
    # self.assertEqual(1 + 3 + 5, net.shape[1])
    with _initialized_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess, coord=coord)

      bias = get_linear_model_bias()
      price_buckets_var = get_linear_model_column_var(price_buckets)
      body_style_var = get_linear_model_column_var(body_style)

      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [100 - 10 + 5.]], sess.run(net))

      coord.request_stop()
      coord.join(threads)

  def test_with_1d_sparse_tensor(self):
    price = fc.numeric_column('price')
    price_buckets = fc.bucketized_column(price, boundaries=[0., 10., 100.,])
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price': constant_op.constant([-1., 12.,]),
        'body-style': sparse_tensor.SparseTensor(
            indices=((0,), (1,)),
            values=('sedan', 'hardtop'),
            dense_shape=(2,)),
    }
    self.assertEqual(1, features['price'].shape.ndims)
    self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])

    net = fc.linear_model(features, [price_buckets, body_style])
    with _initialized_session() as sess:
      bias = get_linear_model_bias()
      price_buckets_var = get_linear_model_column_var(price_buckets)
      body_style_var = get_linear_model_column_var(body_style)

      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]], sess.run(net))

  def test_with_1d_unknown_shape_sparse_tensor(self):
    price = fc.numeric_column('price')
    price_buckets = fc.bucketized_column(price, boundaries=[0., 10., 100.,])
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
        indices=((0,), (1,)),
        values=('sedan', 'hardtop'),
        dense_shape=(2,))

    net = fc.linear_model(features, [price_buckets, body_style])
    bias = get_linear_model_bias()
    price_buckets_var = get_linear_model_column_var(price_buckets)
    body_style_var = get_linear_model_column_var(body_style)
    with _initialized_session() as sess:
      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose(
          [[10 - 1000 + 5.], [1000 - 10 + 5.]],
          sess.run(net, feed_dict={
              features['price']: price_data,
              features['body-style']: body_style_data}))

    # Dense categorical_column with unknown shape is not allowed.
    with self.assertRaisesRegexp(ValueError, 'Undefined input_tensor shape.'):
      fc.linear_model(features, [price_buckets, body_style, country])

  def test_with_rank_0_feature(self):
    price = fc.numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegexp(ValueError, 'Feature .* cannot have rank 0'):
      fc.linear_model(features, [price])

    # Dynamic rank 0 should fail
    features = {
        'price': array_ops.placeholder(dtypes.float32),
    }
    net = fc.linear_model(features, [price])
    self.assertEqual(1, net.shape[1])
    with _initialized_session() as sess:
      with self.assertRaisesOpError('Feature .* cannot have rank 0'):
        sess.run(net, feed_dict={features['price']: np.array(1)})


class InputLayerTest(test.TestCase):

  def test_raises_if_empty_feature_columns(self):
    with self.assertRaisesRegexp(ValueError,
                                 'feature_columns must not be empty'):
      fc.input_layer(features={}, feature_columns=[])

  def test_should_be_dense_column(self):
    with self.assertRaisesRegexp(ValueError, 'must be a _DenseColumn'):
      fc.input_layer(
          features={'a': [[0]]},
          feature_columns=[
              fc.categorical_column_with_hash_bucket('wire_cast', 4)
          ])

  def test_does_not_support_dict_columns(self):
    with self.assertRaisesRegexp(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      fc.input_layer(
          features={'a': [[0]]}, feature_columns={'a': fc.numeric_column('a')})

  def test_raises_if_duplicate_name(self):
    with self.assertRaisesRegexp(
        ValueError, 'Duplicate feature column name found for columns'):
      fc.input_layer(
          features={'a': [[0]]},
          feature_columns=[fc.numeric_column('a'),
                           fc.numeric_column('a')])

  def test_one_column(self):
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      net = fc.input_layer(features, [price])
      with _initialized_session():
        self.assertAllClose([[1.], [5.]], net.eval())

  def test_multi_dimension(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      net = fc.input_layer(features, [price])
      with _initialized_session():
        self.assertAllClose([[1., 2.], [5., 6.]], net.eval())

  def test_raises_if_shape_mismatch(self):
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      net = fc.input_layer(features, [price])
      with _initialized_session():
        with self.assertRaisesRegexp(Exception, 'requested shape has 4'):
          net.eval()

  def test_reshaping(self):
    price = fc.numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      net = fc.input_layer(features, [price])
      with _initialized_session():
        self.assertAllClose([[1., 2.], [5., 6.]], net.eval())

  def test_multi_column(self):
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1., 2.], [5., 6.]],
          'price2': [[3.], [4.]]
      }
      net = fc.input_layer(features, [price1, price2])
      with _initialized_session():
        self.assertAllClose([[1., 2., 3.], [5., 6., 4.]], net.eval())

  def test_column_order(self):
    price_a = fc.numeric_column('price_a')
    price_b = fc.numeric_column('price_b')
    with ops.Graph().as_default():
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
      }
      net1 = fc.input_layer(features, [price_a, price_b])
      net2 = fc.input_layer(features, [price_b, price_a])
      with _initialized_session():
        self.assertAllClose([[1., 3.]], net1.eval())
        self.assertAllClose([[1., 3.]], net2.eval())

  def test_fails_for_categorical_column(self):
    animal = fc.categorical_column_with_identity('animal', num_buckets=4)
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      with self.assertRaisesRegexp(Exception, 'must be a _DenseColumn'):
        fc.input_layer(features, [animal])

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
          'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc.input_layer(features, [price1, price2])

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
          'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc.input_layer(features, [price1, price2, price3])

  def test_runtime_batch_size_mismatch(self):
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      net = fc.input_layer(features, [price1, price2])
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
      net = fc.input_layer(features, [price1, price2])
      with _initialized_session() as sess:
        sess.run(
            net,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  def test_with_numpy_input_fn(self):
    embedding_values = (
        (1., 2., 3., 4., 5.),  # id 0
        (6., 7., 8., 9., 10.),  # id 1
        (11., 12., 13., 14., 15.)  # id 2
    )
    def _initializer(shape, dtype, partition_info):
      del shape, dtype, partition_info
      return embedding_values

    # price has 1 dimension in input_layer
    price = fc.numeric_column('price')
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    # one_hot_body_style has 3 dims in input_layer.
    one_hot_body_style = fc.indicator_column(body_style)
    # embedded_body_style has 5 dims in input_layer.
    embedded_body_style = fc.embedding_column(body_style, dimension=5,
                                              initializer=_initializer)

    input_fn = numpy_io.numpy_input_fn(
        x={
            'price': np.array([11., 12., 13., 14.]),
            'body-style': np.array(['sedan', 'hardtop', 'wagon', 'sedan']),
        },
        batch_size=2,
        shuffle=False)
    features = input_fn()
    net = fc.input_layer(features,
                         [price, one_hot_body_style, embedded_body_style])
    self.assertEqual(1 + 3 + 5, net.shape[1])
    with _initialized_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess, coord=coord)

      # Each row is formed by concatenating `embedded_body_style`,
      # `one_hot_body_style`, and `price` in order.
      self.assertAllEqual(
          [[11., 12., 13., 14., 15., 0., 0., 1., 11.],
           [1., 2., 3., 4., 5., 1., 0., 0., 12]],
          sess.run(net))

      coord.request_stop()
      coord.join(threads)

  def test_with_1d_sparse_tensor(self):
    embedding_values = (
        (1., 2., 3., 4., 5.),  # id 0
        (6., 7., 8., 9., 10.),  # id 1
        (11., 12., 13., 14., 15.)  # id 2
    )
    def _initializer(shape, dtype, partition_info):
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
    embedded_country = fc.embedding_column(country, dimension=5,
                                           initializer=_initializer)

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price': constant_op.constant([11., 12.,]),
        'body-style': sparse_tensor.SparseTensor(
            indices=((0,), (1,)),
            values=('sedan', 'hardtop'),
            dense_shape=(2,)),
        # This is dense tensor for the categorical_column.
        'country': constant_op.constant(['CA', 'US']),
    }
    self.assertEqual(1, features['price'].shape.ndims)
    self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])
    self.assertEqual(1, features['country'].shape.ndims)

    net = fc.input_layer(features,
                         [price, one_hot_body_style, embedded_country])
    self.assertEqual(1 + 3 + 5, net.shape[1])
    with _initialized_session() as sess:

      # Each row is formed by concatenating `embedded_body_style`,
      # `one_hot_body_style`, and `price` in order.
      self.assertAllEqual(
          [[0., 0., 1., 11., 12., 13., 14., 15., 11.],
           [1., 0., 0., 1., 2., 3., 4., 5., 12.]],
          sess.run(net))

  def test_with_1d_unknown_shape_sparse_tensor(self):
    embedding_values = (
        (1., 2., 3., 4., 5.),  # id 0
        (6., 7., 8., 9., 10.),  # id 1
        (11., 12., 13., 14., 15.)  # id 2
    )
    def _initializer(shape, dtype, partition_info):
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
    embedded_country = fc.embedding_column(country, dimension=5,
                                           initializer=_initializer)

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
        indices=((0,), (1,)),
        values=('sedan', 'hardtop'),
        dense_shape=(2,))

    # Dense categorical_column with unknown shape is not allowed.
    with self.assertRaisesRegexp(ValueError, 'Undefined input_tensor shape.'):
      fc.input_layer(features, [price, one_hot_body_style, embedded_country])

    net = fc.input_layer(features, [price, one_hot_body_style])
    self.assertEqual(1 + 3, net.shape[1])
    with _initialized_session() as sess:

      # Each row is formed by concatenating `embedded_body_style`,
      # `one_hot_body_style`, and `price` in order.
      self.assertAllEqual(
          [[0., 0., 1., 11.], [1., 0., 0., 12.]],
          sess.run(net, feed_dict={
              features['price']: price_data,
              features['body-style']: body_style_data}))

  def test_with_rank_0_feature(self):
    # price has 1 dimension in input_layer
    price = fc.numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegexp(ValueError, 'Feature .* cannot have rank 0'):
      fc.input_layer(features, [price])

    # Dynamic rank 0 should fail
    features = {
        'price': array_ops.placeholder(dtypes.float32),
    }
    net = fc.input_layer(features, [price])
    self.assertEqual(1, net.shape[1])
    with _initialized_session() as sess:
      with self.assertRaisesOpError('Feature .* cannot have rank 0'):
        sess.run(net, feed_dict={features['price']: np.array(1)})


class MakeParseExampleSpecTest(test.TestCase):

  class _TestFeatureColumn(_FeatureColumn,
                           collections.namedtuple('_TestFeatureColumn',
                                                  ['parse_spec'])):

    @property
    def _parse_example_spec(self):
      return self.parse_spec

  def test_no_feature_columns(self):
    actual = fc.make_parse_example_spec([])
    self.assertDictEqual({}, actual)

  def test_invalid_type(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    with self.assertRaisesRegexp(
        ValueError,
        'All feature_columns must be _FeatureColumn instances.*invalid_column'):
      fc.make_parse_example_spec(
          (self._TestFeatureColumn({key1: parse_spec1}), 'invalid_column'))

  def test_one_feature_column(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    actual = fc.make_parse_example_spec(
        (self._TestFeatureColumn({key1: parse_spec1}),))
    self.assertDictEqual({key1: parse_spec1}, actual)

  def test_two_feature_columns(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    key2 = 'key2'
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    actual = fc.make_parse_example_spec(
        (self._TestFeatureColumn({key1: parse_spec1}),
         self._TestFeatureColumn({key2: parse_spec2})))
    self.assertDictEqual({key1: parse_spec1, key2: parse_spec2}, actual)

  def test_equal_keys_different_parse_spec(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    with self.assertRaisesRegexp(
        ValueError,
        'feature_columns contain different parse_spec for key key1'):
      fc.make_parse_example_spec(
          (self._TestFeatureColumn({key1: parse_spec1}),
           self._TestFeatureColumn({key1: parse_spec2})))

  def test_equal_keys_equal_parse_spec(self):
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    actual = fc.make_parse_example_spec(
        (self._TestFeatureColumn({key1: parse_spec1}),
         self._TestFeatureColumn({key1: parse_spec1})))
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
    actual = fc.make_parse_example_spec(
        (self._TestFeatureColumn({key1: parse_spec1}),
         self._TestFeatureColumn({key2: parse_spec2, key3: parse_spec3})))
    self.assertDictEqual(
        {key1: parse_spec1, key2: parse_spec2, key3: parse_spec3}, actual)


def _assert_sparse_tensor_value(test_case, expected, actual):
  test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
  test_case.assertAllEqual(expected.indices, actual.indices)

  test_case.assertEqual(
      np.array(expected.values).dtype, np.array(actual.values).dtype)
  test_case.assertAllEqual(expected.values, actual.values)

  test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
  test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


class VocabularyFileCategoricalColumnTest(test.TestCase):

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

  def test_defaults(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3)
    self.assertEqual('aaa', column.name)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, column._parse_example_spec)

  def test_all_constructor_args(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3,
        num_oov_buckets=4, dtype=dtypes.int32)
    self.assertEqual(7, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, column._parse_example_spec)

  def test_deep_copy(self):
    original = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3,
        num_oov_buckets=4, dtype=dtypes.int32)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(7, column._num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int32)
      }, column._parse_example_spec)

  def test_vocabulary_file_none(self):
    with self.assertRaisesRegexp(ValueError, 'Missing vocabulary_file'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file=None, vocabulary_size=3)

  def test_vocabulary_file_empty_string(self):
    with self.assertRaisesRegexp(ValueError, 'Missing vocabulary_file'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file='', vocabulary_size=3)

  def test_invalid_vocabulary_file(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='file_does_not_exist', vocabulary_size=10)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    with self.assertRaisesRegexp(errors.OpError, 'file_does_not_exist'):
      with self.test_session():
        lookup_ops.tables_initializer().run()

  def test_invalid_vocabulary_size(self):
    with self.assertRaisesRegexp(ValueError, 'Invalid vocabulary_size'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=None)
    with self.assertRaisesRegexp(ValueError, 'Invalid vocabulary_size'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=-1)
    with self.assertRaisesRegexp(ValueError, 'Invalid vocabulary_size'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file=self._wire_vocabulary_file_name,
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
    column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    with self.assertRaisesRegexp(errors.OpError, 'Invalid vocab_size'):
      with self.test_session():
        lookup_ops.tables_initializer().run()

  def test_invalid_num_oov_buckets(self):
    with self.assertRaisesRegexp(ValueError, 'Invalid num_oov_buckets'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file='path', vocabulary_size=3,
          num_oov_buckets=-1)

  def test_invalid_dtype(self):
    with self.assertRaisesRegexp(ValueError, 'dtype must be string or integer'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file='path', vocabulary_size=3,
          dtype=dtypes.float64)

  def test_invalid_buckets_and_default_value(self):
    with self.assertRaisesRegexp(
        ValueError, 'both num_oov_buckets and default_value'):
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
    with self.assertRaisesRegexp(ValueError, 'dtype must be compatible'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

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
    with self.assertRaisesRegexp(ValueError, 'dtype must be compatible'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

  def test_parse_example(self):
    a = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    with self.test_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([b'omar', b'stringer'], dtype=np.object_),
              dense_shape=[1, 2]),
          features['aaa'].eval())

  def test_get_sparse_tensors(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, -1, 0), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_transform_feature(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_tensor = _transform_features({'aaa': inputs}, [column])[column]
    with _initialized_session():
      _assert_sparse_tensor_value(self,
                                  sparse_tensor.SparseTensorValue(
                                      indices=inputs.indices,
                                      values=np.array(
                                          (2, -1, 0), dtype=np.int64),
                                      dense_shape=inputs.dense_shape),
                                  id_tensor.eval())

  def test_get_sparse_tensors_weight_collections(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    inputs = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': inputs
        }), weight_collections=('my_weights',))

    self.assertItemsEqual(
        [], ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
    self.assertItemsEqual([], ops.get_collection('my_weights'))

  def test_get_sparse_tensors_dense_input(self):
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    id_weight_pair = column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': (('marlo', ''), ('skywalker', 'omar'))
        }))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=np.array((2, -1, 0), dtype=np.int64),
              dense_shape=(2, 2)),
          id_weight_pair.id_tensor.eval())

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
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, 2, 0), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

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
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, 33, 0, 62), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

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
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((-1, -1, 0), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

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
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, -1, 0, 4), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_int32_dense_input(self):
    default_value = -100
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32,
        default_value=default_value)
    id_weight_pair = column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': ((11, -1, -1), (100, 30, -1), (-1, -1, 22))
        }))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1), (2, 2)),
              values=np.array((2, default_value, 0, 4), dtype=np.int64),
              dense_shape=(3, 3)),
          id_weight_pair.id_tensor.eval())

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
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, 60, 0, 4), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_linear_model(self):
    wire_column = fc.categorical_column_with_vocabulary_file(
        key='wire',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=1)
    self.assertEqual(4, wire_column._num_buckets)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          wire_column.name: sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=('marlo', 'skywalker', 'omar'),
              dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)
      with _initialized_session():
        self.assertAllClose((0.,), bias.eval())
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), wire_var.eval())
        self.assertAllClose(((0.,), (0.,)), predictions.eval())
        wire_var.assign(((1.,), (2.,), (3.,), (4.,))).eval()
        # 'marlo' -> 2: wire_var[2] = 3
        # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
        self.assertAllClose(((3.,), (5.,)), predictions.eval())


class VocabularyListCategoricalColumnTest(test.TestCase):

  def test_defaults_string(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    self.assertEqual('aaa', column.name)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, column._parse_example_spec)

  def test_defaults_int(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36))
    self.assertEqual('aaa', column.name)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, column._parse_example_spec)

  def test_all_constructor_args(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36), dtype=dtypes.int32,
        default_value=-99)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, column._parse_example_spec)

  def test_deep_copy(self):
    original = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36), dtype=dtypes.int32)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(3, column._num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int32)
      }, column._parse_example_spec)

  def test_invalid_dtype(self):
    with self.assertRaisesRegexp(ValueError, 'dtype must be string or integer'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'),
          dtype=dtypes.float32)

  def test_invalid_mapping_dtype(self):
    with self.assertRaisesRegexp(
        ValueError, r'vocabulary dtype must be string or integer'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12., 24., 36.))

  def test_mismatched_int_dtype(self):
    with self.assertRaisesRegexp(
        ValueError, r'dtype.*and vocabulary dtype.*do not match'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'),
          dtype=dtypes.int32)

  def test_mismatched_string_dtype(self):
    with self.assertRaisesRegexp(
        ValueError, r'dtype.*and vocabulary dtype.*do not match'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 36), dtype=dtypes.string)

  def test_none_mapping(self):
    with self.assertRaisesRegexp(
        ValueError, r'vocabulary_list.*must be non-empty'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=None)

  def test_empty_mapping(self):
    with self.assertRaisesRegexp(
        ValueError, r'vocabulary_list.*must be non-empty'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=tuple([]))

  def test_duplicate_mapping(self):
    with self.assertRaisesRegexp(ValueError, 'Duplicate keys'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 12))

  def test_invalid_num_oov_buckets(self):
    with self.assertRaisesRegexp(ValueError, 'Invalid num_oov_buckets'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 36),
          num_oov_buckets=-1)

  def test_invalid_buckets_and_default_value(self):
    with self.assertRaisesRegexp(
        ValueError, 'both num_oov_buckets and default_value'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=(12, 24, 36),
          num_oov_buckets=100,
          default_value=2)

  def test_invalid_input_dtype_int32(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(12, 24, 36),
        dense_shape=(2, 2))
    with self.assertRaisesRegexp(ValueError, 'dtype must be compatible'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

  def test_invalid_input_dtype_string(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=(12, 24, 36))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegexp(ValueError, 'dtype must be compatible'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

  def test_parse_example_string(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    with self.test_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([b'omar', b'stringer'], dtype=np.object_),
              dense_shape=[1, 2]),
          features['aaa'].eval())

  def test_parse_example_int(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(11, 21, 31))
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(int64_list=feature_pb2.Int64List(
                    value=[11, 21]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    with self.test_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=[11, 21],
              dense_shape=[1, 2]),
          features['aaa'].eval())

  def test_get_sparse_tensors(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, -1, 0), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_transform_feature(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_tensor = _transform_features({'aaa': inputs}, [column])[column]
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, -1, 0), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_tensor.eval())

  def test_get_sparse_tensors_weight_collections(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': inputs
        }), weight_collections=('my_weights',))

    self.assertItemsEqual(
        [], ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
    self.assertItemsEqual([], ops.get_collection('my_weights'))

  def test_get_sparse_tensors_dense_input(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'))
    id_weight_pair = column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': (('marlo', ''), ('skywalker', 'omar'))
        }))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=np.array((2, -1, 0), dtype=np.int64),
              dense_shape=(2, 2)),
          id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_default_value_in_vocabulary(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        default_value=2)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, 2, 0), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_with_oov_buckets(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=100)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (1, 2)),
        values=('marlo', 'skywalker', 'omar', 'heisenberg'),
        dense_shape=(2, 3))
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, 33, 0, 62), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_int32(self):
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
        dtype=dtypes.int32)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (2, 2)),
        values=np.array((11, 100, 30, 22), dtype=np.int32),
        dense_shape=(3, 3))
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, -1, 0, 4), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_int32_dense_input(self):
    default_value = -100
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
        dtype=dtypes.int32,
        default_value=default_value)
    id_weight_pair = column._get_sparse_tensors(
        _LazyBuilder({
            'aaa':
                np.array(
                    ((11, -1, -1), (100, 30, -1), (-1, -1, 22)), dtype=np.int32)
        }))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1), (2, 2)),
              values=np.array((2, default_value, 0, 4), dtype=np.int64),
              dense_shape=(3, 3)),
          id_weight_pair.id_tensor.eval())

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
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((2, 60, 0, 4), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_linear_model(self):
    wire_column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=1)
    self.assertEqual(4, wire_column._num_buckets)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          wire_column.name: sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=('marlo', 'skywalker', 'omar'),
              dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)
      with _initialized_session():
        self.assertAllClose((0.,), bias.eval())
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), wire_var.eval())
        self.assertAllClose(((0.,), (0.,)), predictions.eval())
        wire_var.assign(((1.,), (2.,), (3.,), (4.,))).eval()
        # 'marlo' -> 2: wire_var[2] = 3
        # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
        self.assertAllClose(((3.,), (5.,)), predictions.eval())


class IdentityCategoricalColumnTest(test.TestCase):

  def test_constructor(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    self.assertEqual('aaa', column.name)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, column._parse_example_spec)

  def test_deep_copy(self):
    original = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(3, column._num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, column._parse_example_spec)

  def test_invalid_num_buckets_zero(self):
    with self.assertRaisesRegexp(ValueError, 'num_buckets 0 < 1'):
      fc.categorical_column_with_identity(key='aaa', num_buckets=0)

  def test_invalid_num_buckets_negative(self):
    with self.assertRaisesRegexp(ValueError, 'num_buckets -1 < 1'):
      fc.categorical_column_with_identity(key='aaa', num_buckets=-1)

  def test_invalid_default_value_too_small(self):
    with self.assertRaisesRegexp(ValueError, 'default_value -1 not in range'):
      fc.categorical_column_with_identity(
          key='aaa', num_buckets=3, default_value=-1)

  def test_invalid_default_value_too_big(self):
    with self.assertRaisesRegexp(ValueError, 'default_value 3 not in range'):
      fc.categorical_column_with_identity(
          key='aaa', num_buckets=3, default_value=3)

  def test_invalid_input_dtype(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegexp(ValueError, 'Invalid input, not integer'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

  def test_parse_example(self):
    a = fc.categorical_column_with_identity(key='aaa', num_buckets=30)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(int64_list=feature_pb2.Int64List(
                    value=[11, 21]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    with self.test_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([11, 21], dtype=np.int64),
              dense_shape=[1, 2]),
          features['aaa'].eval())

  def test_get_sparse_tensors(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0, 1, 0),
        dense_shape=(2, 2))
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((0, 1, 0), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_transform_feature(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0, 1, 0),
        dense_shape=(2, 2))
    id_tensor = _transform_features({'aaa': inputs}, [column])[column]
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((0, 1, 0), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_tensor.eval())

  def test_get_sparse_tensors_weight_collections(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0, 1, 0),
        dense_shape=(2, 2))
    column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': inputs
        }), weight_collections=('my_weights',))

    self.assertItemsEqual(
        [], ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
    self.assertItemsEqual([], ops.get_collection('my_weights'))

  def test_get_sparse_tensors_dense_input(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    id_weight_pair = column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': ((0, -1), (1, 0))
        }))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=np.array((0, 1, 0), dtype=np.int64),
              dense_shape=(2, 2)),
          id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_with_inputs_too_small(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(1, -1, 0),
        dense_shape=(2, 2))
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      with self.assertRaisesRegexp(
          errors.OpError, 'assert_greater_or_equal_0'):
        id_weight_pair.id_tensor.eval()

  def test_get_sparse_tensors_with_inputs_too_big(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(1, 99, 0),
        dense_shape=(2, 2))
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      with self.assertRaisesRegexp(
          errors.OpError, 'assert_less_than_num_buckets'):
        id_weight_pair.id_tensor.eval()

  def test_get_sparse_tensors_with_default_value(self):
    column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=4, default_value=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(1, -1, 99),
        dense_shape=(2, 2))
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array((1, 3, 3), dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_with_default_value_and_placeholder_inputs(self):
    column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=4, default_value=3)
    input_indices = array_ops.placeholder(dtype=dtypes.int64)
    input_values = array_ops.placeholder(dtype=dtypes.int32)
    input_shape = array_ops.placeholder(dtype=dtypes.int64)
    inputs = sparse_tensor.SparseTensorValue(
        indices=input_indices,
        values=input_values,
        dense_shape=input_shape)
    id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
    self.assertIsNone(id_weight_pair.weight_tensor)
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=np.array(((0, 0), (1, 0), (1, 1)), dtype=np.int64),
              values=np.array((1, 3, 3), dtype=np.int64),
              dense_shape=np.array((2, 2), dtype=np.int64)),
          id_weight_pair.id_tensor.eval(feed_dict={
              input_indices: ((0, 0), (1, 0), (1, 1)),
              input_values: (1, -1, 99),
              input_shape: (2, 2),
          }))

  def test_linear_model(self):
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    self.assertEqual(3, column._num_buckets)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          column.name: sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(0, 2, 1),
              dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)
      with _initialized_session():
        self.assertAllClose((0.,), bias.eval())
        self.assertAllClose(((0.,), (0.,), (0.,)), weight_var.eval())
        self.assertAllClose(((0.,), (0.,)), predictions.eval())
        weight_var.assign(((1.,), (2.,), (3.,))).eval()
        # weight_var[0] = 1
        # weight_var[2] + weight_var[1] = 3+2 = 5
        self.assertAllClose(((1.,), (5.,)), predictions.eval())


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
      transformed = _transform_features(features,
                                        [bucketized_price, hashed_sparse])
      with _initialized_session():
        self.assertIn(bucketized_price.name, transformed[bucketized_price].name)
        self.assertAllEqual([[0], [3]], transformed[bucketized_price].eval())
        self.assertIn(hashed_sparse.name, transformed[hashed_sparse].name)
        self.assertAllEqual([6, 4, 1], transformed[hashed_sparse].values.eval())

  def test_column_order(self):
    """When the column is both dense and sparse, uses sparse tensors."""

    class _LoggerColumn(_FeatureColumn):

      def __init__(self, name):
        self._name = name

      @property
      def name(self):
        return self._name

      def _transform_feature(self, inputs):
        del inputs
        self.call_order = call_logger['count']
        call_logger['count'] += 1
        return 'Anything'

      @property
      def _parse_example_spec(self):
        pass

    with ops.Graph().as_default():
      column1 = _LoggerColumn('1')
      column2 = _LoggerColumn('2')
      call_logger = {'count': 0}
      _transform_features({}, [column1, column2])
      self.assertEqual(0, column1.call_order)
      self.assertEqual(1, column2.call_order)

      call_logger = {'count': 0}
      _transform_features({}, [column2, column1])
      self.assertEqual(0, column1.call_order)
      self.assertEqual(1, column2.call_order)


class IndicatorColumnTest(test.TestCase):

  def test_indicator_column(self):
    a = fc.categorical_column_with_hash_bucket('a', 4)
    indicator_a = fc.indicator_column(a)
    self.assertEqual(indicator_a.categorical_column.name, 'a')
    self.assertEqual(indicator_a._variable_shape, [1, 4])

    b = fc.categorical_column_with_hash_bucket('b', hash_bucket_size=100)
    indicator_b = fc.indicator_column(b)
    self.assertEqual(indicator_b.categorical_column.name, 'b')
    self.assertEqual(indicator_b._variable_shape, [1, 100])

  def test_1D_shape_succeeds(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_hash_bucket('animal', 4))
    builder = _LazyBuilder({'animal': ['fox', 'fox']})
    output = builder.get(animal)
    with self.test_session():
      self.assertAllEqual([[0., 0., 1., 0.], [0., 0., 1., 0.]], output.eval())

  def test_2D_shape_succeeds(self):
    # TODO(ispir/cassandrax): Swith to categorical_column_with_keys when ready.
    animal = fc.indicator_column(
        fc.categorical_column_with_hash_bucket('animal', 4))
    builder = _LazyBuilder({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [1, 0]],
                values=['fox', 'fox'],
                dense_shape=[2, 1])
    })
    output = builder.get(animal)
    with self.test_session():
      self.assertAllEqual([[0., 0., 1., 0.], [0., 0., 1., 0.]], output.eval())

  def test_multi_hot(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))

    builder = _LazyBuilder({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1, 1], dense_shape=[1, 2])
    })
    output = builder.get(animal)
    with self.test_session():
      self.assertAllEqual([[0., 2., 0., 0.]], output.eval())

  def test_multi_hot2(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    builder = _LazyBuilder({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
    })
    output = builder.get(animal)
    with self.test_session():
      self.assertAllEqual([[0., 1., 1., 0.]], output.eval())

  def test_deep_copy(self):
    a = fc.categorical_column_with_hash_bucket('a', 4)
    column = fc.indicator_column(a)
    column_copy = copy.deepcopy(column)
    self.assertEqual(column_copy.categorical_column.name, 'a')
    self.assertEqual(column.name, 'a_indicator')
    self.assertEqual(column._variable_shape, [1, 4])

  def test_parse_example(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_indicator = fc.indicator_column(a)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a_indicator]))
    self.assertIn('aaa', features)
    with self.test_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([b'omar', b'stringer'], dtype=np.object_),
              dense_shape=[1, 2]),
          features['aaa'].eval())

  def test_transform(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_indicator = fc.indicator_column(a)
    features = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=('marlo', 'skywalker', 'omar'),
            dense_shape=(2, 2))
    }
    indicator_tensor = _transform_features(features, [a_indicator])[a_indicator]
    with _initialized_session():
      self.assertAllEqual([[0, 0, 1], [1, 0, 0]], indicator_tensor.eval())

  def test_linear_model(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }

      predictions = fc.linear_model(features, [animal])
      weight_var = get_linear_model_column_var(animal)
      with _initialized_session():
        # All should be zero-initialized.
        self.assertAllClose([[0.], [0.], [0.], [0.]], weight_var.eval())
        self.assertAllClose([[0.]], predictions.eval())
        weight_var.assign([[1.], [2.], [3.], [4.]]).eval()
        self.assertAllClose([[2. + 3.]], predictions.eval())

  def test_input_layer(self):
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      net = fc.input_layer(features, [animal])
      with _initialized_session():
        self.assertAllClose([[0., 1., 1., 0.]], net.eval())


class EmbeddingColumnTest(test.TestCase):

  def test_defaults(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('mean', embedding_column.combiner)
    self.assertIsNotNone(embedding_column.initializer)
    self.assertIsNone(embedding_column.ckpt_to_load_from)
    self.assertIsNone(embedding_column.tensor_name_in_ckpt)
    self.assertIsNone(embedding_column.max_norm)
    self.assertTrue(embedding_column.trainable)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual(
        (embedding_dimension,), embedding_column._variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column._parse_example_spec)

  def test_all_constructor_args(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        combiner='my_combiner', initializer=lambda: 'my_initializer',
        ckpt_to_load_from='my_ckpt', tensor_name_in_ckpt='my_ckpt_tensor',
        max_norm=42., trainable=False)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('my_combiner', embedding_column.combiner)
    self.assertEqual('my_initializer', embedding_column.initializer())
    self.assertEqual('my_ckpt', embedding_column.ckpt_to_load_from)
    self.assertEqual('my_ckpt_tensor', embedding_column.tensor_name_in_ckpt)
    self.assertEqual(42., embedding_column.max_norm)
    self.assertFalse(embedding_column.trainable)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual(
        (embedding_dimension,), embedding_column._variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column._parse_example_spec)

  def test_deep_copy(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    original = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        combiner='my_combiner', initializer=lambda: 'my_initializer',
        ckpt_to_load_from='my_ckpt', tensor_name_in_ckpt='my_ckpt_tensor',
        max_norm=42., trainable=False)
    for embedding_column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', embedding_column.categorical_column.name)
      self.assertEqual(3, embedding_column.categorical_column._num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column.categorical_column._parse_example_spec)

      self.assertEqual(embedding_dimension, embedding_column.dimension)
      self.assertEqual('my_combiner', embedding_column.combiner)
      self.assertEqual('my_initializer', embedding_column.initializer())
      self.assertEqual('my_ckpt', embedding_column.ckpt_to_load_from)
      self.assertEqual('my_ckpt_tensor', embedding_column.tensor_name_in_ckpt)
      self.assertEqual(42., embedding_column.max_norm)
      self.assertFalse(embedding_column.trainable)
      self.assertEqual('aaa_embedding', embedding_column.name)
      self.assertEqual(
          (embedding_dimension,), embedding_column._variable_shape)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column._parse_example_spec)

  def test_invalid_initializer(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    with self.assertRaisesRegexp(ValueError, 'initializer must be callable'):
      fc.embedding_column(categorical_column, dimension=2, initializer='not_fn')

  def test_parse_example(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_embedded = fc.embedding_column(a, dimension=2)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a_embedded]))
    self.assertIn('aaa', features)
    with self.test_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([b'omar', b'stringer'], dtype=np.object_),
              dense_shape=[1, 2]),
          features['aaa'].eval())

  def test_transform_feature(self):
    a = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    a_embedded = fc.embedding_column(a, dimension=2)
    features = {
        'aaa': sparse_tensor.SparseTensor(
            indices=((0, 0), (1, 0), (1, 1)),
            values=(0, 1, 0),
            dense_shape=(2, 2))
    }
    outputs = _transform_features(features, [a, a_embedded])
    output_a = outputs[a]
    output_embedded = outputs[a_embedded]
    with _initialized_session():
      _assert_sparse_tensor_value(
          self, output_a.eval(), output_embedded.eval())

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
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column._get_dense_tensor(
        _LazyBuilder({
            'aaa': sparse_input
        }))

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('embedding_weights:0',), tuple([v.name for v in global_vars]))
    with _initialized_session():
      self.assertAllEqual(embedding_values, global_vars[0].eval())
      self.assertAllEqual(expected_lookups, embedding_lookup.eval())

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
        (1., 2., 4.),   # id 0
        (3., 5., 1.),   # id 1
        (7., 11., 2.),  # id 2
        (2., 7., 12.)   # id 3
    )
    def _initializer(shape, dtype, partition_info):
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
        categorical_column, dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column._get_dense_tensor(
        _LazyBuilder({
            'aaa': sparse_input
        }))

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('embedding_weights:0',), tuple([v.name for v in global_vars]))
    with _initialized_session():
      self.assertAllEqual(embedding_values, global_vars[0].eval())
      self.assertAllEqual(expected_lookups, embedding_lookup.eval())

  def test_get_dense_tensor_weight_collections(self):
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 4), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 5))

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_column = fc.embedding_column(categorical_column, dimension=2)

    # Provide sparse input and get dense result.
    embedding_column._get_dense_tensor(
        _LazyBuilder({
            'aaa': sparse_input
        }), weight_collections=('my_vars',))

    # Assert expected embedding variable and lookups.
    self.assertItemsEqual(
        [], ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
    my_vars = ops.get_collection('my_vars')
    self.assertItemsEqual(
        ('embedding_weights:0',), tuple([v.name for v in my_vars]))

  def test_get_dense_tensor_placeholder_inputs(self):
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
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    input_indices = array_ops.placeholder(dtype=dtypes.int64)
    input_values = array_ops.placeholder(dtype=dtypes.int64)
    input_shape = array_ops.placeholder(dtype=dtypes.int64)
    embedding_lookup = embedding_column._get_dense_tensor(
        _LazyBuilder({
            'aaa':
                sparse_tensor.SparseTensorValue(
                    indices=input_indices,
                    values=input_values,
                    dense_shape=input_shape)
        }))

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('embedding_weights:0',), tuple([v.name for v in global_vars]))
    with _initialized_session():
      self.assertAllEqual(embedding_values, global_vars[0].eval())
      self.assertAllEqual(expected_lookups, embedding_lookup.eval(
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
        categorical_column, dimension=embedding_dimension,
        ckpt_to_load_from=ckpt_path,
        tensor_name_in_ckpt=ckpt_tensor)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column._get_dense_tensor(
        _LazyBuilder({
            'aaa': sparse_input
        }))

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('embedding_weights:0',), tuple([v.name for v in global_vars]))
    with _initialized_session():
      self.assertAllEqual(embedding_values, global_vars[0].eval())
      self.assertAllEqual(expected_lookups, embedding_lookup.eval())

  def test_linear_model(self):
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
    def _initializer(shape, dtype, partition_info):
      self.assertAllEqual(embedding_shape, shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return zeros_embedding_values

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        initializer=_initializer)

    with ops.Graph().as_default():
      predictions = fc.linear_model({
          categorical_column.name: sparse_input
      }, (embedding_column,))
      expected_var_names = (
          'linear_model/bias_weights:0',
          'linear_model/aaa_embedding/weights:0',
          'linear_model/aaa_embedding/embedding_weights:0',
      )
      self.assertItemsEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])
      trainable_vars = {
          v.name: v for v in ops.get_collection(
              ops.GraphKeys.TRAINABLE_VARIABLES)
      }
      self.assertItemsEqual(expected_var_names, trainable_vars.keys())
      bias = trainable_vars['linear_model/bias_weights:0']
      embedding_weights = trainable_vars[
          'linear_model/aaa_embedding/embedding_weights:0']
      linear_weights = trainable_vars[
          'linear_model/aaa_embedding/weights:0']
      with _initialized_session():
        # Predictions with all zero weights.
        self.assertAllClose(np.zeros((1,)), bias.eval())
        self.assertAllClose(zeros_embedding_values, embedding_weights.eval())
        self.assertAllClose(
            np.zeros((embedding_dimension, 1)), linear_weights.eval())
        self.assertAllClose(np.zeros((batch_size, 1)), predictions.eval())

        # Predictions with all non-zero weights.
        embedding_weights.assign((
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )).eval()
        linear_weights.assign(((4.,), (6.,))).eval()
        # example 0, ids [2], embedding[0] = [7, 11]
        # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
        # example 2, ids [], embedding[2] = [0, 0]
        # example 3, ids [1], embedding[3] = [3, 5]
        # sum(embeddings * linear_weights)
        # = [4*7 + 6*11, 4*2 + 6*3.5, 4*0 + 6*0, 4*3 + 6*5] = [94, 29, 0, 42]
        self.assertAllClose(((94.,), (29.,), (0.,), (42.,)), predictions.eval())

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
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    input_layer = fc.input_layer({'aaa': sparse_input}, (embedding_column,))

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('input_layer/aaa_embedding/embedding_weights:0',),
        tuple([v.name for v in global_vars]))
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertItemsEqual(
        ('input_layer/aaa_embedding/embedding_weights:0',),
        tuple([v.name for v in trainable_vars]))
    with _initialized_session():
      self.assertAllEqual(embedding_values, trainable_vars[0].eval())
      self.assertAllEqual(expected_lookups, input_layer.eval())

  def test_input_layer_not_trainable(self):
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
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension,
        initializer=_initializer, trainable=False)

    # Provide sparse input and get dense result.
    input_layer = fc.input_layer({'aaa': sparse_input}, (embedding_column,))

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('input_layer/aaa_embedding/embedding_weights:0',),
        tuple([v.name for v in global_vars]))
    self.assertItemsEqual(
        [], ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))
    with _initialized_session():
      self.assertAllEqual(embedding_values, global_vars[0].eval())
      self.assertAllEqual(expected_lookups, input_layer.eval())


class WeightedCategoricalColumnTest(test.TestCase):

  def test_defaults(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    self.assertEqual('ids_weighted_by_values', column.name)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'ids': parsing_ops.VarLenFeature(dtypes.int64),
        'values': parsing_ops.VarLenFeature(dtypes.float32)
    }, column._parse_example_spec)

  def test_deep_copy(self):
    """Tests deepcopy of categorical_column_with_hash_bucket."""
    original = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('ids_weighted_by_values', column.name)
      self.assertEqual(3, column._num_buckets)
      self.assertEqual({
          'ids': parsing_ops.VarLenFeature(dtypes.int64),
          'values': parsing_ops.VarLenFeature(dtypes.float32)
      }, column._parse_example_spec)

  def test_invalid_dtype_none(self):
    with self.assertRaisesRegexp(ValueError, 'is not convertible to float'):
      fc.weighted_categorical_column(
          categorical_column=fc.categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values',
          dtype=None)

  def test_invalid_dtype_string(self):
    with self.assertRaisesRegexp(ValueError, 'is not convertible to float'):
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
    with self.assertRaisesRegexp(ValueError, 'Bad dtype'):
      _transform_features({'ids': strings, 'values': strings}, (column,))

  def test_column_name_collision(self):
    with self.assertRaisesRegexp(ValueError, r'Parse config.*already exists'):
      fc.weighted_categorical_column(
          categorical_column=fc.categorical_column_with_identity(
              key='aaa', num_buckets=3),
          weight_feature_key='aaa')._parse_example_spec()

  def test_missing_weights(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegexp(
        ValueError, 'values is not in features dictionary'):
      _transform_features({'ids': inputs}, (column,))

  def test_parse_example(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_weighted = fc.weighted_categorical_column(a, weight_feature_key='weights')
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer'])),
            'weights':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[1., 10.]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a_weighted]))
    self.assertIn('aaa', features)
    self.assertIn('weights', features)
    with self.test_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([b'omar', b'stringer'], dtype=np.object_),
              dense_shape=[1, 2]),
          features['aaa'].eval())
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([1., 10.], dtype=np.float32),
              dense_shape=[1, 2]),
          features['weights'].eval())

  def test_transform_features(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0, 1, 0),
        dense_shape=(2, 2))
    weights = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0.5, 1.0, 0.1),
        dense_shape=(2, 2))
    id_tensor, weight_tensor = _transform_features({
        'ids': inputs,
        'values': weights,
    }, (column,))[column]
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array(inputs.values, dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_tensor.eval())
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=weights.indices,
              values=np.array(weights.values, dtype=np.float32),
              dense_shape=weights.dense_shape),
          weight_tensor.eval())

  def test_transform_features_dense_input(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    weights = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0.5, 1.0, 0.1),
        dense_shape=(2, 2))
    id_tensor, weight_tensor = _transform_features({
        'ids': ((0, -1), (1, 0)),
        'values': weights,
    }, (column,))[column]
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=np.array((0, 1, 0), dtype=np.int64),
              dense_shape=(2, 2)),
          id_tensor.eval())
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=weights.indices,
              values=np.array(weights.values, dtype=np.float32),
              dense_shape=weights.dense_shape),
          weight_tensor.eval())

  def test_transform_features_dense_weights(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(2, 1, 0),
        dense_shape=(2, 2))
    id_tensor, weight_tensor = _transform_features({
        'ids': inputs,
        'values': ((.5, 0.), (1., .1)),
    }, (column,))[column]
    with _initialized_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=inputs.indices,
              values=np.array(inputs.values, dtype=np.int64),
              dense_shape=inputs.dense_shape),
          id_tensor.eval())
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=np.array((.5, 1., .1), dtype=np.float32),
              dense_shape=(2, 2)),
          weight_tensor.eval())

  def test_linear_model(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          'ids': sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(0, 2, 1),
              dense_shape=(2, 2)),
          'values': sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(.5, 1., .1),
              dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)
      with _initialized_session():
        self.assertAllClose((0.,), bias.eval())
        self.assertAllClose(((0.,), (0.,), (0.,)), weight_var.eval())
        self.assertAllClose(((0.,), (0.,)), predictions.eval())
        weight_var.assign(((1.,), (2.,), (3.,))).eval()
        # weight_var[0] * weights[0, 0] = 1 * .5 = .5
        # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
        # = 3*1 + 2*.1 = 3+.2 = 3.2
        self.assertAllClose(((.5,), (3.2,)), predictions.eval())

  def test_linear_model_mismatched_shape(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError, r'Dimensions.*are not compatible'):
        fc.linear_model({
            'ids': sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1)),
                values=(0, 2, 1),
                dense_shape=(2, 2)),
            'values': sparse_tensor.SparseTensorValue(
                indices=((0, 0), (0, 1), (1, 0), (1, 1)),
                values=(.5, 11., 1., .1),
                dense_shape=(2, 2))
        }, (column,))

  def test_linear_model_mismatched_dense_values(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          'ids': sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(0, 2, 1),
              dense_shape=(2, 2)),
          'values': ((.5,), (1.,))
      }, (column,))
      with _initialized_session():
        with self.assertRaisesRegexp(errors.OpError, 'Incompatible shapes'):
          predictions.eval()

  def test_linear_model_mismatched_dense_shape(self):
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          'ids': sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(0, 2, 1),
              dense_shape=(2, 2)),
          'values': ((.5,), (1.,), (.1,))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)
      with _initialized_session():
        self.assertAllClose((0.,), bias.eval())
        self.assertAllClose(((0.,), (0.,), (0.,)), weight_var.eval())
        self.assertAllClose(((0.,), (0.,)), predictions.eval())
        weight_var.assign(((1.,), (2.,), (3.,))).eval()
        # weight_var[0] * weights[0, 0] = 1 * .5 = .5
        # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
        # = 3*1 + 2*.1 = 3+.2 = 3.2
        self.assertAllClose(((.5,), (3.2,)), predictions.eval())

  # TODO(ptucker): Add test with embedding of weighted categorical.

if __name__ == '__main__':
  test.main()
