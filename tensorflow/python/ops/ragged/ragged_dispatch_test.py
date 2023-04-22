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
"""Tests for RaggedTensor operator dispatch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_dispatch
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_test_ops as test_ops
from tensorflow.python.platform import googletest


# pylint: disable=g-complex-comprehension
@test_util.run_all_in_graph_and_eager_modes
class RaggedDispatchTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def assertSameShape(self, x, y):
    """Checks that x and y have the same shape (including ragged shapes)."""
    if ragged_tensor.is_ragged(x):
      self.assertTrue(ragged_tensor.is_ragged(y))
      self.assertEqual(x.ragged_rank, y.ragged_rank)
      for (x_splits, y_splits) in zip(x.nested_row_splits, y.nested_row_splits):
        self.assertAllEqual(x_splits, y_splits)
      self.assertAllEqual(
          array_ops.shape(x.flat_values), array_ops.shape(y.flat_values))
    else:
      self.assertIsInstance(y, ops.Tensor)
      self.assertAllEqual(array_ops.shape(x), array_ops.shape(y))

  @parameterized.parameters(
      #=========================================================================
      # Test different input shapes.
      #=========================================================================
      [
          # 0-dimensional input
          {'x': 12},
          # 1-dimensional input
          {'x': [1, -2, 3]},
          # 2-dimensional input
          {'x': [[-2, 3], [-3, 4]]},
          {'x': ragged_factory_ops.constant_value(
              [[-2, 3], [-3]], ragged_rank=1)},
          # 3-dimensional inputs
          {'x': [[[-2, 3], [3, 4]], [[7, 6], [5, 4]]]},
          {'x': ragged_factory_ops.constant_value(
              [[[-2, 3], [3, 4]], [[7, 6]]],
              ragged_rank=1)},
          {'x': ragged_factory_ops.constant_value(
              [[[-2, 3, 4], []], [[7, 6]], []],
              ragged_rank=2)},
          ] +
      #=========================================================================
      # Test each unary op.
      #=========================================================================
      [{'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]), 'op': op}
       for op in test_ops.UNARY_FLOAT_OPS] +
      [{'x': ragged_factory_ops.constant_value([[True, False], [True]]),
        'op': op}
       for op in test_ops.UNARY_BOOL_OPS] +
      [{'x': ragged_factory_ops.constant_value([[18, 512], [12412]], np.int32),
        'op': op}
       for op in test_ops.UNARY_INT_OPS] +
      [{'x': ragged_factory_ops.constant_value([['abcd', 'efgh'],
                                                ['aabbccdd']]),
        'op': op}
       for op in test_ops.UNARY_STRING_OPS] +
      [
          {'op': clip_ops.clip_by_value,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'clip_value_min': 0.1, 'clip_value_max': 4.0},
          {'op': math_ops.cast,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'dtype': dtypes.int32},
          {'op': math_ops.saturate_cast,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'dtype': dtypes.int32},
          {'op': string_ops.string_to_hash_bucket,
           'x': ragged_factory_ops.constant_value(
               [['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000},
          {'op': string_ops.string_to_hash_bucket_fast,
           'x': ragged_factory_ops.constant_value(
               [['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000},
          {'op': string_ops.string_to_hash_bucket_strong,
           'x': ragged_factory_ops.constant_value(
               [['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000,
           'key': [1231, 12512]},
          {'op': string_ops.string_to_number,
           'x': ragged_factory_ops.constant_value([['-2.0', '3.0'], ['-3.0']])},
          {'op': string_ops.regex_full_match,
           'x': ragged_factory_ops.constant_value([['hello', '123'], ['1+1']]),
           'pattern': r'\w+'},
          {'op': string_ops.regex_replace,
           'x': ragged_factory_ops.constant_value([['hello', '123'], ['1+1']]),
           'pattern': r'\d',
           'rewrite': '#'},
          {'op': string_ops.substr,
           'x': ragged_factory_ops.constant_value([['hello', '123'], ['1+1']]),
           'pos': 2, 'len': 3},
          {'op': array_ops.check_numerics,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'message': 'check-numerics'},
          {'op': nn_ops.dropout,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'rate': 0.5,
           'seed': 1},
          {'op': math_ops.nextafter,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'x2': 0},
      ]
      )  # pyformat: disable
  def testUnaryElementwiseOp(self, x, op=math_ops.abs, **extra_args):
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x)
    result = op(x, **extra_args)

    # Run the wrapped op on the dense values, for comparison.
    dense_x = x.flat_values if ragged_tensor.is_ragged(x) else x
    expected_flat_values = array_ops.reshape(op(dense_x, **extra_args), [-1])

    # Check that the result has the expected shape.
    self.assertSameShape(x, result)

    # Check that the result has the expected (flattened) values.
    if ragged_tensor.is_ragged(result):
      result_flat_values = array_ops.reshape(result.flat_values, [-1])
    else:
      result_flat_values = array_ops.reshape(result, [-1])
    self.assertAllEqual(expected_flat_values, result_flat_values)

  @parameterized.parameters(
      [
          #=====================================================================
          # Without broadcasting -- i.e., shapes match exactly.
          #=====================================================================
          # Shapes: x:(), y:()
          {'x': 12,
           'y': 8},
          # Shapes: x:(3,), y:(3,)
          {'x': [7, 8, 9],
           'y': [1, -2, 3]},
          # Shapes: x:(2, 2), y:(2, 2)
          {'x': [[-2, 3], [-3, -4]],
           'y': [[1, 2], [3, 4]]},
          # Shapes: x:(2, None), y:(2, None)
          {'x': ragged_factory_ops.constant_value([[-2, 3], [-3]]),
           'y': ragged_factory_ops.constant_value([[5, 6], [7]])},
          # Shapes: x:(2, 2, 2), y:(2, 2, 2)
          {'x': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
           'y': [[[9, 3], [3, 4]], [[5, 2], [7, 6]]]},
          # Shapes: x:(2, None, None), y: (2, None, None)
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2], [3], [4]], [[], [5, 7, 8]]]),
           'y': ragged_factory_ops.constant_value(
               [[[3, 8], [2], [5]], [[], [1, 9, 8]]])},
          # Shapes: x:(2, None, 2), y: (2, None, 2)
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2]], [[3, 4], [5, 6], [7, 8]]],
              ragged_rank=1),
           'y': ragged_factory_ops.constant_value(
               [[[9, 3]], [[5, 2], [3, 4], [7, 6]]],
               ragged_rank=1)},

          #=====================================================================
          # With broadcasting
          #=====================================================================
          # Shapes: x:(), y:(3,)
          {'x': 12,                                 # Broadcast () -> (3,)
           'y': [1, -2, 3]},
          # Shapes: x:(1,), y:(3,)
          {'x': [12],                               # Broadcast (1,) -> (3,)
           'y': [1, -2, 3]},
          # Shapes: x:(), y:(2, 2)
          {'x': 12,                                 # Broadcast () -> (2, 2)
           'y': [[1, 2], [3, 4]]},
          # Shapes: x:(1,), y:(2, 2)
          {'x': 12,                                 # Broadcast (1,) -> (2, 2)
           'y': [[1, 2], [3, 4]]},
          # Shapes: x:(2, 1), y:(2, 2)
          {'x': [[10], [20]],                       # Broadcast (2, 1) -> (2, 2)
           'y': [[1, 2], [3, 4]]},
          # Shapes: x:(), y:(2, None)
          {'x': 10,                                 # Broadcast () -> (2, None)
           'y': ragged_factory_ops.constant_value(
               [[1, 2], [3]], dtype=np.int32)},
          # TODO(edloper): Add tests for more advanced broadcasting, once we add
          # support for it.

          #=====================================================================
          # Keyword Args
          #=====================================================================
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2], [3], [4]], [[], [5, 7, 8]]]),
           'y': ragged_factory_ops.constant_value(
               [[[3, 8], [2], [5]], [[], [1, 9, 8]]]),
           'use_kwargs': ('x', 'y')},
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2]], [[3, 4], [5, 6], [7, 8]]],
              ragged_rank=1),
           'y': ragged_factory_ops.constant_value(
               [[[9, 3]], [[5, 2], [3, 4], [7, 6]]],
               ragged_rank=1),
           'use_kwargs': ('x', 'y')},
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2]], [[3, 4], [5, 6], [7, 8]]],
              ragged_rank=1),
           'y': ragged_factory_ops.constant_value(
               [[[9, 3]], [[5, 2], [3, 4], [7, 6]]],
               ragged_rank=1),
           'use_kwargs': ('x',)},
      ] +
      #=========================================================================
      # Test each binary op.
      #=========================================================================
      [{'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
        'y': ragged_factory_ops.constant_value([[5.0, 1.0], [12.0]]),
        'op': op}
       for op in test_ops.BINARY_FLOAT_OPS] +
      [{'x': ragged_factory_ops.constant_value([[-2, 3], [-3]]),
        'y': ragged_factory_ops.constant_value([[5, 1], [12]]),
        'op': op}
       for op in test_ops.BINARY_INT_OPS] +
      [{'x': ragged_factory_ops.constant_value([[True, True], [False]]),
        'y': ragged_factory_ops.constant_value([[False, True], [False]]),
        'op': op}
       for op in test_ops.BINARY_BOOL_OPS]
      )  # pyformat: disable
  def testBinaryElementwiseOp(self, x, y, op=math_ops.add, **extra_args):
    use_kwargs = extra_args.pop('use_kwargs', ())
    if 'x' in use_kwargs and 'y' in use_kwargs:
      result = op(x=x, y=y, **extra_args)
    elif 'y' in use_kwargs:
      result = op(x, y=y, **extra_args)
    else:
      result = op(x, y, **extra_args)

    # Run the wrapped op on the dense values, for comparison.
    dense_x = x.flat_values if ragged_tensor.is_ragged(x) else x
    dense_y = y.flat_values if ragged_tensor.is_ragged(y) else y
    expected_flat_values = array_ops.reshape(
        op(dense_x, dense_y, **extra_args), [-1])

    # Check that the result has the expected shape.
    self.assertSameShape(y, result)

    # Check that the result has the expected (flattened) values.
    if ragged_tensor.is_ragged(result):
      result_flat_values = array_ops.reshape(result.flat_values, [-1])
    else:
      result_flat_values = array_ops.reshape(result, [-1])
    self.assertAllEqual(expected_flat_values, result_flat_values)

  @parameterized.parameters(
      [
          {'inputs': (12, 8, 3)},
          {'inputs': ([1, 2, 3], [7, 8, 9], [3, 6, 9])},
          {'inputs': ([[1, 2]], [[3, 4]], [[5, 6]])},
          {'inputs': (ragged_factory_ops.constant_value([[1, 3], [-3]]),
                      ragged_factory_ops.constant_value([[4, 7], [88]]),
                      ragged_factory_ops.constant_value([[2, 9], [12]]))},
          {'inputs': (ragged_factory_ops.constant_value(
              [[[1, 3], [-3]], [[1]]]),
                      ragged_factory_ops.constant_value(
                          [[[4, 7], [88]], [[2]]]),
                      ragged_factory_ops.constant_value(
                          [[[2, 9], [12]], [[8]]]))},
          {'inputs': (
              ragged_factory_ops.constant_value([[[1, 3], [3, 4]], [[1, 5]]],
                                                ragged_rank=1),
              ragged_factory_ops.constant_value([[[4, 7], [1, 2]], [[2, 2]]],
                                                ragged_rank=1),
              ragged_factory_ops.constant_value([[[2, 9], [5, 2]], [[8, 0]]],
                                                ragged_rank=1))},
          {'inputs': (
              ragged_factory_ops.constant_value([[[1, 3], [-3]], [[1]]]),
              ragged_factory_ops.constant_value([[[4, 7], [88]], [[2]]]),
              ragged_factory_ops.constant_value([[[2, 9], [12]], [[8]]])),
           'use_kwargs': True},
      ] + [
          {'op': math_ops.add_n,
           'inputs': (ragged_factory_ops.constant_value([[1, 3], [-3]]),
                      ragged_factory_ops.constant_value([[4, 7], [88]]),
                      ragged_factory_ops.constant_value([[2, 9], [12]]))},
          {'op': string_ops.string_join,
           'inputs': (
               ragged_factory_ops.constant_value([['a', 'b'], ['c']]),
               ragged_factory_ops.constant_value([['foo', 'bar'], ['baz']]),
               ragged_factory_ops.constant_value([['2', '9'], ['12']]))},
      ])  # pyformat: disable
  def testListValuedElementwiseOp(self,
                                  inputs,
                                  op=math_ops.add_n,
                                  **extra_args):
    use_kwargs = extra_args.pop('use_kwargs', False)
    if use_kwargs:
      result = op(inputs=inputs, **extra_args)
    else:
      result = op(inputs, **extra_args)

    # Run the wrapped op on the dense values, for comparison.
    dense_inputs = [
        x.flat_values if ragged_tensor.is_ragged(x) else x for x in inputs
    ]
    expected_flat_values = array_ops.reshape(
        op(dense_inputs, **extra_args), [-1])

    # Check that the result has the expected shape.
    self.assertSameShape(inputs[0], result)

    # Check that the result has the expected (flattened) values.
    if ragged_tensor.is_ragged(result):
      result_flat_values = array_ops.reshape(result.flat_values, [-1])
    else:
      result_flat_values = array_ops.reshape(result, [-1])
    self.assertAllEqual(expected_flat_values, result_flat_values)

  def testElementwiseOpUnknownRankError(self):
    if context.executing_eagerly():
      return
    x = ragged_factory_ops.constant([[1, 2], [3]])
    y = ragged_tensor.RaggedTensor.from_row_splits(
        array_ops.placeholder_with_default([1, 2, 3], shape=None), x.row_splits)
    with self.assertRaisesRegex(ValueError,
                                r'Unable to broadcast: unknown rank'):
      math_ops.add(x, y)

  @parameterized.parameters([
      dict(
          x=ragged_factory_ops.constant_value([[1, 2], [3]]),
          y=[[10]],
          expected=[[11, 12], [13]]),
      dict(
          x=ragged_factory_ops.constant_value([[[1, 2], [3, 4]], [[5]]],
                                              ragged_rank=2),
          y=ragged_factory_ops.constant_value([[[10], [20]], [[30]]],
                                              ragged_rank=1),
          expected=[[[11, 12], [23, 24]], [[35]]]),
      dict(
          x=ragged_factory_ops.constant_value([[[1]]]),
          y=ragged_factory_ops.constant_value([[1]]),
          expected=[[[2]]]),
  ])
  def testElementwiseOpBroadcast(self, x, y, expected):
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, dtype=dtypes.int32)
    y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, dtype=dtypes.int32)
    result = x + y
    self.assertAllEqual(result, expected)

  def testElementwiseOpShapeMismatch(self):
    x = ragged_factory_ops.constant([[1, 2, 3], [4, 5]])
    y = ragged_factory_ops.constant([[1, 2, 3], [4, 5, 6]])
    with self.assertRaises((ValueError, errors.InvalidArgumentError)):
      self.evaluate(math_ops.add(x, y))

  def testBinaryOpSparseAndRagged(self):
    x = ragged_factory_ops.constant([[1, 2, 3], [4, 5]])
    y = sparse_tensor.SparseTensor([[0, 0], [0, 1], [2, 0]], [1, 2, 3], [3, 2])
    with self.assertRaises((TypeError, ValueError)):
      self.evaluate(math_ops.add(x, y))

    with self.assertRaises((TypeError, ValueError)):
      self.evaluate(math_ops.add_n([x, y]))

  @parameterized.parameters([
      dict(
          op=array_ops.batch_gather,
          args=(ragged_factory_ops.constant_value([[5, 6, 7], [8, 9]]),
                ragged_factory_ops.constant_value([[2, 1, 0], [1]])),
          expected=ragged_factory_ops.constant_value([[7, 6, 5], [9]])),
      dict(
          op=array_ops.concat,
          args=([
              ragged_factory_ops.constant_value([[1, 2, 3], [4]],
                                                dtype=np.int32),
              np.array([[5, 6]], dtype=np.int32)
          ],),
          kwargs={'axis': 0},
          expected=ragged_factory_ops.constant_value([[1, 2, 3], [4], [5, 6]])),
      dict(
          op=array_ops.expand_dims,
          kwargs={
              'input': ragged_factory_ops.constant_value([[1, 2], [3]]),
              'axis': 0
          },
          expected=ragged_factory_ops.constant_value([[[1, 2], [3]]])),
      dict(
          op=array_ops.expand_dims_v2,
          kwargs={
              'input': ragged_factory_ops.constant_value([[1, 2], [3]]),
              'axis': -1
          },
          expected=ragged_factory_ops.constant_value([[[1], [2]], [[3]]],
                                                     ragged_rank=1),
      ),
      dict(
          op=array_ops.gather,
          kwargs={
              'params': ragged_factory_ops.constant_value([[1, 2], [3]]),
              'indices': [1, 0, 1]
          },
          expected=ragged_factory_ops.constant_value([[3], [1, 2], [3]])),
      dict(
          op=array_ops.gather_v2,
          kwargs={
              'params': ragged_factory_ops.constant_value([[1, 2], [3]]),
              'indices': ragged_factory_ops.constant_value([[1, 0], [1]])
          },
          expected=ragged_factory_ops.constant_value([[[3], [1, 2]], [[3]]])),
      dict(
          op=array_ops.gather_nd,
          kwargs={
              'params': ragged_factory_ops.constant_value([[7, 8], [9]]),
              'indices': [[0, 1], [1, 0], [0, 0]]
          },
          expected=ragged_factory_ops.constant_value([8, 9, 7])),
      dict(
          op=array_ops.one_hot,
          kwargs={
              'indices':
                  ragged_factory_ops.constant_value([[1, 2, 3], [0]],
                                                    dtype=np.int32),
              'depth':
                  4,
              'axis':
                  -1
          },
          expected=ragged_factory_ops.constant_value(
              [[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [[1, 0, 0, 0]]],
              ragged_rank=1)),
      dict(
          op=array_ops.stack,
          args=([
              ragged_factory_ops.constant_value([[1, 2, 3], [4]],
                                                dtype=np.int32),
              np.array([[5, 6]], dtype=np.int32)
          ],),
          expected=ragged_factory_ops.constant_value([[[1, 2, 3], [4]],
                                                      [[5, 6]]])),
      dict(
          op=array_ops.tile,
          args=([
              ragged_factory_ops.constant_value([[1, 2], [3]], dtype=np.int32),
              [2, 3]
          ]),
          expected=ragged_factory_ops.constant_value([[1, 2, 1, 2, 1, 2],
                                                      [3, 3, 3],
                                                      [1, 2, 1, 2, 1, 2],
                                                      [3, 3, 3]])),
      dict(
          op=array_ops.where,
          args=(ragged_factory_ops.constant_value([[True, False], [True]]),
                ragged_factory_ops.constant_value([[b'A', b'B'], [b'C']]),
                ragged_factory_ops.constant_value([[b'a', b'b'], [b'c']])),
          expected=ragged_factory_ops.constant_value([[b'A', b'b'], [b'C']])),
      dict(
          op=array_ops.where,
          args=(ragged_factory_ops.constant_value([[True, False], [True]]),),
          expected=[[0, 0], [1, 0]]),
      dict(
          op=array_ops.where_v2,
          args=(ragged_factory_ops.constant_value([[True, False], [True]]),
                ragged_factory_ops.constant_value([[b'A', b'B'], [b'C']]),
                ragged_factory_ops.constant_value([[b'a', b'b'], [b'c']])),
          expected=ragged_factory_ops.constant_value([[b'A', b'b'], [b'C']])),
      dict(
          op=math_ops.unsorted_segment_sum,
          kwargs={
              'data': ragged_factory_ops.constant_value([[1, 2], [3]]),
              'segment_ids': ragged_factory_ops.constant_value([[0, 2], [0]]),
              'num_segments': 3
          },
          expected=[4, 0, 2]),
      dict(
          op=math_ops.unsorted_segment_prod,
          kwargs={
              'data': ragged_factory_ops.constant_value([[1, 2], [3]]),
              'segment_ids': ragged_factory_ops.constant_value([[0, 2], [0]]),
              'num_segments': 3
          },
          expected=[3, 1, 2]),
      dict(
          op=math_ops.unsorted_segment_min,
          kwargs={
              'data': ragged_factory_ops.constant_value([[1, 2], [3]]),
              'segment_ids': ragged_factory_ops.constant_value([[0, 1], [0]]),
              'num_segments': 2
          },
          expected=[1, 2]),
      dict(
          op=math_ops.unsorted_segment_max,
          kwargs={
              'data': ragged_factory_ops.constant_value([[1, 2], [3]]),
              'segment_ids': ragged_factory_ops.constant_value([[0, 1], [0]]),
              'num_segments': 2
          },
          expected=[3, 2]),
      dict(
          op=math_ops.unsorted_segment_mean,
          kwargs={
              'data': ragged_factory_ops.constant_value([[1, 2], [3]]),
              'segment_ids': ragged_factory_ops.constant_value([[0, 1], [0]]),
              'num_segments': 2
          },
          expected=[2, 2]),
      dict(
          op=math_ops.unsorted_segment_sqrt_n,
          kwargs={
              'data':
                  ragged_factory_ops.constant_value([[1.0, 2.0],
                                                     [3.0, 4.0, 6.0]]),
              'segment_ids':
                  ragged_factory_ops.constant_value([[0, 1], [0, 0, 0]]),
              'num_segments':
                  2
          },
          expected=[7.0, 2.0]),
      dict(
          op=math_ops.reduce_sum,
          kwargs={
              'input_tensor':
                  ragged_factory_ops.constant_value([[1, 2], [3, 4, 5]]),
              'axis':
                  1
          },
          expected=[3, 12]),
      dict(
          op=math_ops.reduce_prod,
          kwargs={
              'input_tensor':
                  ragged_factory_ops.constant_value([[1, 2], [3, 4, 5]]),
              'axis':
                  1
          },
          expected=[2, 60]),
      dict(
          op=math_ops.reduce_min,
          kwargs={
              'input_tensor':
                  ragged_factory_ops.constant_value([[1, 2], [3, 4, 5]]),
              'axis':
                  1
          },
          expected=[1, 3]),
      dict(
          op=math_ops.reduce_max,
          kwargs={
              'input_tensor':
                  ragged_factory_ops.constant_value([[1, 2], [3, 4, 5]]),
              'axis':
                  1
          },
          expected=[2, 5]),
      dict(
          op=math_ops.reduce_mean,
          kwargs={
              'input_tensor':
                  ragged_factory_ops.constant_value([[1, 3], [3, 4, 5]]),
              'axis':
                  1
          },
          expected=[2, 4]),
      dict(
          op=math_ops.reduce_variance,
          kwargs={
              'input_tensor':
                  ragged_factory_ops.constant_value([[1, 3], [3, 6, 9]]),
              'axis':
                  1
          },
          expected=[1., 6.]),
      dict(
          op=math_ops.reduce_std,
          kwargs={
              'input_tensor':
                  ragged_factory_ops.constant_value([[1, 3], [1, 2, 2, 1]]),
              'axis':
                  1
          },
          expected=[1., 0.5]),
      dict(
          op=math_ops.reduce_any,
          kwargs={
              'input_tensor':
                  ragged_factory_ops.constant_value([[True, False],
                                                     [True, True, True]]),
              'axis':
                  1
          },
          expected=[True, True]),
      dict(
          op=math_ops.matmul,
          kwargs={
              'a': ragged_factory_ops.constant_value([[1, 2, 3], [4, 5, 6]]),
              'b': ragged_factory_ops.constant_value([[5], [4], [3]])
          },
          expected=[[22], [58]]),
      dict(
          op=string_ops.reduce_join,
          kwargs={
              'inputs':
                  ragged_factory_ops.constant_value([[
                      b'this', b'is', b'a', b'test', b'for', b'ragged',
                      b'tensors'
                  ], [b'please', b'do', b'not', b'panic', b'!']]),
              'axis':
                  0,
              'keepdims':
                  False,
              'separator':
                  ''
          },
          expected=[
              b'thisplease', b'isdo', b'anot', b'testpanic', b'for!', b'ragged',
              b'tensors'
          ]),
      dict(
          op=math_ops.reduce_all,
          kwargs={
              'input_tensor':
                  ragged_factory_ops.constant_value([[True, False],
                                                     [True, True, True]]),
              'axis':
                  1
          },
          expected=[False, True]),
      dict(
          op=array_ops.rank,
          kwargs={'input': ragged_factory_ops.constant_value([[8, 3], [5]])},
          expected=2),
      dict(
          op=array_ops.size,
          kwargs={'input': ragged_factory_ops.constant_value([[8, 3], [5]])},
          expected=3),
      dict(
          op=array_ops.size_v2,
          kwargs={'input': ragged_factory_ops.constant_value([[8, 3], [5]])},
          expected=3),
      dict(
          op=array_ops.squeeze,
          kwargs={
              'input': ragged_factory_ops.constant_value([[[1, 2, 3], [4, 5]]]),
              'axis': [0]
          },
          expected=ragged_factory_ops.constant_value([[1, 2, 3], [4, 5]])),
      dict(
          op=array_ops.squeeze_v2,
          kwargs={
              'input': ragged_factory_ops.constant_value([[[1, 2, 3], [4, 5]]]),
              'axis': [0]
          },
          expected=ragged_factory_ops.constant_value([[1, 2, 3], [4, 5]])),
      dict(
          op=data_flow_ops.dynamic_partition,
          kwargs={
              'data': ragged_factory_ops.constant_value([[1], [2, 3, 4], [5]]),
              'partitions': [2, 1, 1],
              'num_partitions': 3
          },
          expected=[
              ragged_factory_ops.constant_value([], ragged_rank=1),
              ragged_factory_ops.constant_value([[2, 3, 4], [5]]),
              ragged_factory_ops.constant_value([[1]])
          ],
          result_is_list=True),
      dict(
          op=array_ops.reverse,
          kwargs={
              'tensor': ragged_factory_ops.constant_value([[1, 2, 3], [4, 5]]),
              'axis': [0, -1]
          },
          expected=ragged_factory_ops.constant_value([[5, 4], [3, 2, 1]])),
      dict(
          op=string_ops.string_format,
          kwargs={
              'template': 'Hi {}',
              'inputs': [ragged_factory_ops.constant_value([[1, 2], [3]])]
          },
          expected='Hi [[1, 2], [3]]'),
      dict(
          op=nn_ops.softmax_v2,
          kwargs={
              'logits':
                  ragged_factory_ops.constant_value([[1., 2., 3.], [4., 5.]]),
          },
          expected=ragged_factory_ops.constant_value([
              [
                  np.exp(1) / (np.exp(1) + np.exp(2) + np.exp(3)),
                  np.exp(2) / (np.exp(1) + np.exp(2) + np.exp(3)),
                  np.exp(3) / (np.exp(1) + np.exp(2) + np.exp(3)),
              ],
              [
                  np.exp(4) / (np.exp(4) + np.exp(5)),
                  np.exp(5) / (np.exp(4) + np.exp(5)),
              ],
          ]),
          rtol=1e-6,
      ),
  ])
  def testRaggedDispatch(self,
                         op,
                         expected,
                         args=(),
                         result_is_list=False,
                         rtol=None,
                         kwargs=None):
    kwargs = kwargs or {}
    if rtol is not None:
      assert_fn = lambda x, y: self.assertAllClose(x, y, rtol=rtol)
    else:
      assert_fn = self.assertAllEqual

    result = op(*args, **kwargs)
    if result_is_list:
      self.assertLen(result, len(expected))
      for (r, e) in zip(result, expected):
        assert_fn(r, e)
    else:
      assert_fn(result, expected)

  def testUnaryElementwiseOpsPreserveUniformRowLength(self):
    # Unary elementwise op
    rt = ragged_tensor.RaggedTensor.from_uniform_row_length(
        ragged_factory_ops.constant([[1, 2], [3]]), uniform_row_length=2)
    self.assertAllEqual(rt.uniform_row_length,
                        array_ops.zeros_like(rt).uniform_row_length)

    # Unary-list elementwise op
    rt = ragged_tensor.RaggedTensor.from_uniform_row_length(
        ragged_factory_ops.constant([[1, 2], [3]]), uniform_row_length=2)
    self.assertAllEqual(rt.uniform_row_length,
                        math_ops.add_n([rt, rt]).uniform_row_length)

  def test_ragged_op_list(self):
    # Ops that should be listed as supported in both v1 and v2.
    supported_ops = [
        'bitwise.bitwise_and', 'bitwise.bitwise_or', 'bitwise.bitwise_xor',
        'bitwise.invert', 'bitwise.left_shift', 'bitwise.right_shift',
        'clip_by_value', 'concat', 'debugging.check_numerics', 'cast',
        'dtypes.complex', 'dtypes.saturate_cast', 'expand_dims', 'gather_nd',
        'gather', 'identity', 'io.decode_base64', 'io.decode_compressed',
        'io.encode_base64', 'math.abs', 'math.acos', 'math.acosh', 'math.add_n',
        'math.add', 'math.angle', 'math.asin', 'math.asinh', 'math.atan2',
        'math.atan', 'math.atanh', 'math.bessel_i0', 'math.bessel_i0e',
        'math.bessel_i1', 'math.bessel_i1e', 'math.ceil', 'math.conj',
        'math.cos', 'math.cosh', 'math.digamma', 'math.divide_no_nan',
        'math.divide', 'math.equal', 'math.erf', 'math.erfc', 'math.erfcinv',
        'math.erfinv', 'math.exp', 'math.expm1', 'math.floor', 'math.floordiv',
        'math.floormod', 'math.greater_equal', 'math.greater', 'math.imag',
        'math.is_finite', 'math.is_inf', 'math.is_nan', 'math.less_equal',
        'math.less', 'math.lgamma', 'math.log1p', 'math.log_sigmoid',
        'math.log', 'math.logical_and', 'math.logical_not', 'math.logical_or',
        'math.logical_xor', 'math.maximum', 'math.minimum', 'math.multiply',
        'math.negative', 'math.nextafter', 'math.not_equal', 'math.pow',
        'math.real', 'math.reciprocal', 'math.reciprocal_no_nan',
        'math.reduce_any', 'math.reduce_max', 'math.reduce_mean',
        'math.reduce_variance', 'math.reduce_std', 'math.reduce_min',
        'math.reduce_prod', 'math.reduce_sum', 'math.rint', 'math.round',
        'math.rsqrt', 'math.sign', 'math.sigmoid', 'math.sin', 'math.sinh',
        'math.softplus', 'math.sqrt', 'math.square', 'math.squared_difference',
        'math.subtract', 'math.tan', 'math.tanh', 'math.truediv',
        'math.unsorted_segment_max', 'math.unsorted_segment_mean',
        'math.unsorted_segment_min', 'math.unsorted_segment_prod',
        'math.unsorted_segment_sqrt_n', 'math.unsorted_segment_sum', 'one_hot',
        'ones_like', 'rank', 'realdiv', 'math.reduce_all', 'size', 'squeeze',
        'stack', 'strings.as_string', 'strings.join', 'strings.length',
        'strings.reduce_join', 'strings.regex_full_match',
        'strings.regex_replace', 'strings.strip', 'strings.substr',
        'strings.to_hash_bucket_fast', 'strings.to_hash_bucket_strong',
        'strings.to_hash_bucket', 'strings.to_number', 'strings.unicode_script',
        'tile', 'truncatediv', 'truncatemod', 'zeros_like', 'dynamic_partition',
        'reverse', 'nn.dropout', 'strings.format', 'print'
    ]

    # Ops that should be listed as supported in v1 only.
    supported_ops_v1 = ['batch_gather']

    # Ops that should be listed as supported in v2 only.
    supported_ops_v2 = ['nn.softmax']

    v1_ragged_ops = ragged_dispatch.ragged_op_list(tf_version=1)
    for element in supported_ops + supported_ops_v1:
      self.assertIn('`tf.' + element + '`', v1_ragged_ops)
    for element in supported_ops_v2:
      self.assertNotIn('`tf.' + element + '`', v1_ragged_ops)

    v2_ragged_ops = ragged_dispatch.ragged_op_list(tf_version=2)
    for element in supported_ops + supported_ops_v2:
      self.assertIn('`tf.' + element + '`', v2_ragged_ops)
    for element in supported_ops_v1:
      self.assertNotIn('`tf.' + element + '`', v2_ragged_ops)


if __name__ == '__main__':
  googletest.main()
