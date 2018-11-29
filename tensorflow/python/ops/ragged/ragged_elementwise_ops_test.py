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
"""Tests for ragged.elementwise_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest

# Constants listing various op types to test.  Each elementwise operation
# should be included in at least one list below, or tested separately if
# necessary (e.g., because it expects additional arguments).
UNARY_FLOAT_OPS = [
    ragged.abs,
    ragged.acos,
    ragged.acosh,
    ragged.angle,
    ragged.asin,
    ragged.asinh,
    ragged.atan,
    ragged.atanh,
    ragged.ceil,
    ragged.conj,
    ragged.cos,
    ragged.cosh,
    ragged.digamma,
    ragged.erf,
    ragged.erfc,
    ragged.exp,
    ragged.expm1,
    ragged.floor,
    ragged.imag,
    ragged.is_finite,
    ragged.is_inf,
    ragged.is_nan,
    ragged.lgamma,
    ragged.log,
    ragged.log1p,
    ragged.log_sigmoid,
    ragged.negative,
    ragged.real,
    ragged.reciprocal,
    ragged.rint,
    ragged.round,
    ragged.rsqrt,
    ragged.sign,
    ragged.sin,
    ragged.sinh,
    ragged.sqrt,
    ragged.square,
    ragged.tan,
    ragged.as_string,
    ragged.identity,
    ragged.ones_like,
    ragged.zeros_like,
]
UNARY_BOOL_OPS = [
    ragged.logical_not,
]
UNARY_STRING_OPS = [
    ragged.decode_base64,
    ragged.encode_base64,
    ragged.string_strip,
    ragged.decode_compressed,
]
BINARY_FLOAT_OPS = [
    ragged.add,
    ragged.atan2,
    ragged.complex,
    ragged.div,
    ragged.div_no_nan,
    ragged.divide,
    ragged.equal,
    ragged.floordiv,
    ragged.floormod,
    ragged.greater,
    ragged.greater_equal,
    ragged.less,
    ragged.less_equal,
    ragged.maximum,
    ragged.minimum,
    ragged.multiply,
    ragged.not_equal,
    ragged.pow,
    ragged.realdiv,
    ragged.squared_difference,
    ragged.subtract,
    ragged.truediv,
]
BINARY_BOOL_OPS = [
    ragged.logical_and,
    ragged.logical_or,
    ragged.logical_xor,
]
UNARY_INT_OPS = [
    ragged.unicode_script,
]
BINARY_INT_OPS = [
    ragged.truncatediv,
    ragged.truncatemod,
]


class RaggedElementwiseOpsTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):

  def assertSameShape(self, x, y):
    """Checks that x and y have the same shape (including ragged shapes)."""
    if isinstance(x, ragged.RaggedTensor):
      self.assertIsInstance(y, ragged.RaggedTensor)
      self.assertEqual(x.ragged_rank, y.ragged_rank)
      for (x_splits, y_splits) in zip(x.nested_row_splits, y.nested_row_splits):
        self.assertAllEqual(x_splits, y_splits)
      self.assertAllEqual(
          array_ops.shape(x.inner_values), array_ops.shape(y.inner_values))
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
          {'x': ragged.constant_value([[-2, 3], [-3]], ragged_rank=1)},
          # 3-dimensional inputs
          {'x': [[[-2, 3], [3, 4]], [[7, 6], [5, 4]]]},
          {'x': ragged.constant_value([[[-2, 3], [3, 4]], [[7, 6]]],
                                      ragged_rank=1)},
          {'x': ragged.constant_value([[[-2, 3, 4], []], [[7, 6]], []],
                                      ragged_rank=2)},
          ] +
      #=========================================================================
      # Test each unary op.
      #=========================================================================
      [{'x': ragged.constant_value([[-2.0, 3.0], [-3.0]]), 'op': op}
       for op in UNARY_FLOAT_OPS] +
      [{'x': ragged.constant_value([[True, False], [True]]), 'op': op}
       for op in UNARY_BOOL_OPS] +
      [{'x': ragged.constant_value([[18, 512], [12412]], np.int32), 'op': op}
       for op in UNARY_INT_OPS] +
      [{'x': ragged.constant_value([['abcd', 'efgh'], ['aabbccdd']]), 'op': op}
       for op in UNARY_STRING_OPS] +
      [
          {'op': ragged.clip_by_value,
           'x': ragged.constant_value([[-2.0, 3.0], [-3.0]]),
           'clip_value_min': 0.1, 'clip_value_max': 4.0},
          {'op': ragged.cast,
           'x': ragged.constant_value([[-2.0, 3.0], [-3.0]]),
           'dtype': dtypes.int32},
          {'op': ragged.saturate_cast,
           'x': ragged.constant_value([[-2.0, 3.0], [-3.0]]),
           'dtype': dtypes.int32},
          {'op': ragged.string_to_hash_bucket,
           'x': ragged.constant_value([['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000},
          {'op': ragged.string_to_hash_bucket_fast,
           'x': ragged.constant_value([['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000},
          {'op': ragged.string_to_hash_bucket_strong,
           'x': ragged.constant_value([['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000,
           'key': [1231, 12512]},
          {'op': ragged.string_to_number,
           'x': ragged.constant_value([['-2.0', '3.0'], ['-3.0']])},
          {'op': ragged.regex_full_match,
           'x': ragged.constant_value([['hello', '123'], ['1+1']]),
           'pattern': r'\w+'},
          {'op': ragged.regex_replace,
           'x': ragged.constant_value([['hello', '123'], ['1+1']]),
           'pattern': r'\d',
           'rewrite': '#'},
          {'op': ragged.substr,
           'x': ragged.constant_value([['hello', '123'], ['1+1']]),
           'pos': 2, 'len': 3},
          {'op': ragged.check_numerics,
           'x': ragged.constant_value([[-2.0, 3.0], [-3.0]]),
           'message': 'check-numerics'},
      ]
      )  # pyformat: disable
  def testUnaryOp(self, x, op=ragged.abs, **extra_args):
    x = ragged.convert_to_tensor_or_ragged_tensor(x)
    result = op(x, **extra_args)

    # Run the wrapped op on the dense values, for comparison.
    dense_x = x.inner_values if isinstance(x, ragged.RaggedTensor) else x
    expected_flat_values = array_ops.reshape(
        op.__wrapped__(dense_x, **extra_args), [-1])

    with self.test_session():
      # Check that the result has the expected shape.
      self.assertSameShape(x, result)

      # Check that the result has the expected (flattened) values.
      if isinstance(result, ragged.RaggedTensor):
        result_flat_values = array_ops.reshape(result.inner_values, [-1])
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
          {'x': ragged.constant_value([[-2, 3], [-3]]),
           'y': ragged.constant_value([[5, 6], [7]])},
          # Shapes: x:(2, 2, 2), y:(2, 2, 2)
          {'x': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
           'y': [[[9, 3], [3, 4]], [[5, 2], [7, 6]]]},
          # Shapes: x:(2, None, None), y: (2, None, None)
          {'x': ragged.constant_value([[[1, 2], [3], [4]], [[], [5, 7, 8]]]),
           'y': ragged.constant_value([[[3, 8], [2], [5]], [[], [1, 9, 8]]])},
          # Shapes: x:(2, None, 2), y: (2, None, 2)
          {'x': ragged.constant_value([[[1, 2]], [[3, 4], [5, 6], [7, 8]]],
                                      ragged_rank=1),
           'y': ragged.constant_value([[[9, 3]], [[5, 2], [3, 4], [7, 6]]],
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
           'y': ragged.constant_value([[1, 2], [3]], dtype=np.int32)},
          # TODO(edloper): Add tests for more advanced broadcasting, once we add
          # support for it.

          #=====================================================================
          # Keyword Args
          #=====================================================================
          {'x': ragged.constant_value([[[1, 2], [3], [4]], [[], [5, 7, 8]]]),
           'y': ragged.constant_value([[[3, 8], [2], [5]], [[], [1, 9, 8]]]),
           'use_kwargs': True},
          {'x': ragged.constant_value([[[1, 2]], [[3, 4], [5, 6], [7, 8]]],
                                      ragged_rank=1),
           'y': ragged.constant_value([[[9, 3]], [[5, 2], [3, 4], [7, 6]]],
                                      ragged_rank=1),
           'use_kwargs': True},
      ] +
      #=========================================================================
      # Test each unary op.
      #=========================================================================
      [{'x': ragged.constant_value([[-2.0, 3.0], [-3.0]]),
        'y': ragged.constant_value([[5.0, 1.0], [12.0]]),
        'op': op}
       for op in BINARY_FLOAT_OPS] +
      [{'x': ragged.constant_value([[-2, 3], [-3]]),
        'y': ragged.constant_value([[5, 1], [12]]),
        'op': op}
       for op in BINARY_INT_OPS] +
      [{'x': ragged.constant_value([[True, True], [False]]),
        'y': ragged.constant_value([[False, True], [False]]),
        'op': op}
       for op in BINARY_BOOL_OPS] +
      [
      ]
      )  # pyformat: disable
  def testBinaryOp(self, x, y, op=ragged.add, **extra_args):
    use_kwargs = extra_args.pop('use_kwargs', False)
    x = ragged.convert_to_tensor_or_ragged_tensor(x)
    y = ragged.convert_to_tensor_or_ragged_tensor(y)
    if use_kwargs:
      result = op(x=x, y=y, **extra_args)
    else:
      result = op(x, y, **extra_args)

    # Run the wrapped op on the dense values, for comparison.
    dense_x = x.inner_values if isinstance(x, ragged.RaggedTensor) else x
    dense_y = y.inner_values if isinstance(y, ragged.RaggedTensor) else y
    expected_flat_values = array_ops.reshape(
        op.__wrapped__(dense_x, dense_y, **extra_args), [-1])

    with self.test_session():
      # Check that the result has the expected shape.
      self.assertSameShape(y, result)

      # Check that the result has the expected (flattened) values.
      if isinstance(result, ragged.RaggedTensor):
        result_flat_values = array_ops.reshape(result.inner_values, [-1])
      else:
        result_flat_values = array_ops.reshape(result, [-1])
      self.assertAllEqual(expected_flat_values, result_flat_values)

  @parameterized.parameters(
      [
          {'inputs': (12, 8, 3)},
          {'inputs': ([1, 2, 3], [7, 8, 9], [3, 6, 9])},
          {'inputs': ([[1, 2]], [[3, 4]], [[5, 6]])},
          {'inputs': (ragged.constant_value([[1, 3], [-3]]),
                      ragged.constant_value([[4, 7], [88]]),
                      ragged.constant_value([[2, 9], [12]]))},
          {'inputs': (ragged.constant_value([[[1, 3], [-3]], [[1]]]),
                      ragged.constant_value([[[4, 7], [88]], [[2]]]),
                      ragged.constant_value([[[2, 9], [12]], [[8]]]))},
          {'inputs': (ragged.constant_value([[[1, 3], [3, 4]], [[1, 5]]],
                                            ragged_rank=1),
                      ragged.constant_value([[[4, 7], [1, 2]], [[2, 2]]],
                                            ragged_rank=1),
                      ragged.constant_value([[[2, 9], [5, 2]], [[8, 0]]],
                                            ragged_rank=1))},
          {'inputs': (ragged.constant_value([[[1, 3], [-3]], [[1]]]),
                      ragged.constant_value([[[4, 7], [88]], [[2]]]),
                      ragged.constant_value([[[2, 9], [12]], [[8]]])),
           'use_kwargs': True},
      ] + [
          {'op': ragged.add_n,
           'inputs': (ragged.constant_value([[1, 3], [-3]]),
                      ragged.constant_value([[4, 7], [88]]),
                      ragged.constant_value([[2, 9], [12]]))},
          {'op': ragged.string_join,
           'inputs': (ragged.constant_value([['a', 'b'], ['c']]),
                      ragged.constant_value([['foo', 'bar'], ['baz']]),
                      ragged.constant_value([['2', '9'], ['12']]))},
      ])  # pyformat: disable
  def testListValuedOp(self, inputs, op=ragged.add_n, **extra_args):
    use_kwargs = extra_args.pop('use_kwargs', False)
    inputs = [ragged.convert_to_tensor_or_ragged_tensor(x) for x in inputs]
    if use_kwargs:
      result = op(inputs=inputs, **extra_args)
    else:
      result = op(inputs, **extra_args)

    # Run the wrapped op on the dense values, for comparison.
    dense_inputs = [
        x.inner_values if isinstance(x, ragged.RaggedTensor) else x
        for x in inputs
    ]
    expected_flat_values = array_ops.reshape(
        op.__wrapped__(dense_inputs, **extra_args), [-1])

    with self.test_session():
      # Check that the result has the expected shape.
      self.assertSameShape(inputs[0], result)

      # Check that the result has the expected (flattened) values.
      if isinstance(result, ragged.RaggedTensor):
        result_flat_values = array_ops.reshape(result.inner_values, [-1])
      else:
        result_flat_values = array_ops.reshape(result, [-1])
      self.assertAllEqual(expected_flat_values, result_flat_values)

  def testUnknownRankError(self):
    x = ragged.constant([[1, 2], [3]])
    y = ragged.from_row_splits(
        array_ops.placeholder_with_default([1, 2, 3], shape=None), x.row_splits)
    with self.assertRaisesRegexp(
        ValueError, r'Unable to broadcast: unknown rank'):
      ragged.add(x, y)

  @parameterized.parameters([
      dict(
          x=ragged.constant_value([[1, 2], [3]]),
          y=[[10]],
          expected=[[11, 12], [13]]),
      dict(
          x=ragged.constant_value([[[1, 2], [3, 4]], [[5]]], ragged_rank=2),
          y=ragged.constant_value([[[10], [20]], [[30]]], ragged_rank=1),
          expected=[[[11, 12], [23, 24]], [[35]]]),
      dict(
          x=ragged.constant_value([[[1]]]),
          y=ragged.constant_value([[1]]),
          expected=[[[2]]]),
  ])
  def testBroadcastAdd(self, x, y, expected):
    x = ragged.convert_to_tensor_or_ragged_tensor(x, dtype=dtypes.int32)
    y = ragged.convert_to_tensor_or_ragged_tensor(y, dtype=dtypes.int32)
    result = x + y
    with self.cached_session():
      self.assertEqual(result.eval().tolist(), expected)

  def testShapeMismatch(self):
    x = ragged.constant([[1, 2, 3], [4, 5]])
    y = ragged.constant([[1, 2, 3], [4, 5, 6]])
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'Incompatible shapes'):
      with self.cached_session():
        ragged.add(x, y).eval()

  def testDocstring(self):
    self.assertRegexpMatches(
        ragged.add.__doc__,
        'Ragged version of the elementwise operation `tf.math.add`')
    self.assertEqual(ragged.add.__name__, 'add')


if __name__ == '__main__':
  googletest.main()
