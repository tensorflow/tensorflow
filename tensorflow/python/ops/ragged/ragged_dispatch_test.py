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

from absl.testing import parameterized
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_dispatch
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_test_ops as test_ops
from tensorflow.python.ops.ragged.dynamic_ragged_shape import DynamicRaggedShape
from tensorflow.python.platform import googletest
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest

# pylint: disable=g-complex-comprehension
# pylint: disable=g-long-lambda


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
          {'op': string_ops.string_to_hash_bucket_v1,
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
          {'op': string_ops.substr_deprecated,
           'x': ragged_factory_ops.constant_value([['hello', '123'], ['1+1']]),
           'pos': 2, 'len': 3},
          {'op': string_ops.substr_v2,
           'x': ragged_factory_ops.constant_value([['hello', '123'], ['1+1']]),
           'pos': 2, 'len': 3},
          {'op': array_ops.check_numerics,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'message': 'check-numerics'},
          {'op': nn_ops.dropout,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'rate': 0.5,
           'seed': 1},
          {'op': nn_ops.stateless_dropout,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'rate': 0.5,
           'seed': [1, 0],
           'rng_alg': 'auto_select'},
          {'op': math_ops.nextafter,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'x2': 0},
          {'op': math_ops.to_bfloat16,
           'x': ragged_factory_ops.constant_value(
               [[2.0, 3.0], [3.0]], dtype=dtypes.float32),
           'expected_dtype': dtypes.bfloat16},
          {'op': math_ops.to_complex128,
           'x': ragged_factory_ops.constant_value(
               [[2.0, 3.0], [3.0]], dtype=dtypes.float32),
           'expected_dtype': dtypes.complex128},
          {'op': math_ops.to_complex64,
           'x': ragged_factory_ops.constant_value(
               [[2.0, 3.0], [3.0]], dtype=dtypes.float32),
           'expected_dtype': dtypes.complex64},
          {'op': math_ops.to_double,
           'x': ragged_factory_ops.constant_value(
               [[2.0, 3.0], [3.0]], dtype=dtypes.float32),
           'expected_dtype': dtypes.double},
          {'op': math_ops.to_float,
           'x': ragged_factory_ops.constant_value(
               [[2.0, 3.0], [3.0]], dtype=dtypes.int32),
           'expected_dtype': dtypes.float32},
          {'op': math_ops.to_int32,
           'x': ragged_factory_ops.constant_value(
               [[2, 3], [3]], dtype=dtypes.int64),
           'expected_dtype': dtypes.int32},
          {'op': math_ops.to_int64,
           'x': ragged_factory_ops.constant_value(
               [[2, 3], [3]], dtype=dtypes.int32),
           'expected_dtype': dtypes.int64},
          {'op': image_ops_impl.convert_image_dtype,
           'x': ragged_factory_ops.constant_value([[-2, 3], [-3]]),
           'dtype': dtypes.float32,
           'expected_dtype': dtypes.float32},
          {'op': image_ops_impl.adjust_brightness,
           'x': ragged_factory_ops.constant_value([[-2, 3], [-3]]),
           'delta': 0.2},
          {'op': image_ops_impl.adjust_gamma,
           'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'gamma': 2,
           'gain': 1.2},
          {'op': image_ops_impl.stateless_random_brightness,
           'x': ragged_factory_ops.constant_value([[-2, 3], [-3]]),
           'max_delta': 0.2,
           'seed': (1, 2)},
          {'op': image_ops_impl.random_brightness,
           'x': ragged_factory_ops.constant_value([[-2, 3], [-3]]),
           'max_delta': 0.2,
           'seed': 12},
          {'op': string_ops.unicode_transcode,
           'x': ragged_factory_ops.constant_value(
               [['tensor', 'flower'], ['2.0']]),
           'input_encoding': 'UTF-8',
           'output_encoding': 'UTF-16-BE'},
      ]
      )  # pyformat: disable
  def testUnaryElementwiseOp(self,
                             x,
                             op=math_ops.abs,
                             expected_dtype=None,
                             **extra_args):
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x)
    random_seed.set_random_seed(1234)
    result = op(x, **extra_args)

    # Run the wrapped op on the dense values, for comparison.
    dense_x = x.flat_values if ragged_tensor.is_ragged(x) else x
    random_seed.set_random_seed(1234)
    expected_flat_values = array_ops.reshape(op(dense_x, **extra_args), [-1])

    # Check that the result has the expected shape.
    self.assertSameShape(x, result)

    # Check that the result has the expected (flattened) values.
    if ragged_tensor.is_ragged(result):
      result_flat_values = array_ops.reshape(result.flat_values, [-1])
    else:
      result_flat_values = array_ops.reshape(result, [-1])
    self.assertAllEqual(expected_flat_values, result_flat_values)

    if expected_dtype is not None:
      self.assertEqual(result.dtype, expected_dtype)

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
           'use_kwargs': {'x': 'x', 'y': 'y'}},
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2]], [[3, 4], [5, 6], [7, 8]]],
              ragged_rank=1),
           'y': ragged_factory_ops.constant_value(
               [[[9, 3]], [[5, 2], [3, 4], [7, 6]]],
               ragged_rank=1),
           'use_kwargs': {'x': 'x', 'y': 'y'}},
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2]], [[3, 4], [5, 6], [7, 8]]],
              ragged_rank=1),
           'y': ragged_factory_ops.constant_value(
               [[[9, 3]], [[5, 2], [3, 4], [7, 6]]],
               ragged_rank=1),
           'use_kwargs': {'y': 'y'}},
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
       for op in test_ops.BINARY_BOOL_OPS] +
      #=========================================================================
      # Test each binary op.
      #=========================================================================
      [
          {'x': 3,
           'y': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'op': math_ops.scalar_mul},
          {'x': 3,
           'y': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'op': math_ops.scalar_mul_v2},
          {'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
           'y': ragged_factory_ops.constant_value([[5.0, 1.0], [12.0]]),
           'op': nn_impl.sigmoid_cross_entropy_with_logits_v2,
           'use_kwargs': {'x': 'labels', 'y': 'logits'}},
      ])  # pyformat: disable
  def testBinaryElementwiseOp(self, x, y, op=math_ops.add, **extra_args):
    use_kwargs = extra_args.pop('use_kwargs', {})

    def compute(x, y):
      if 'x' in use_kwargs and 'y' in use_kwargs:
        extra_args[use_kwargs['x']] = x
        extra_args[use_kwargs['y']] = y
        return op(**extra_args)
      elif 'y' in use_kwargs:
        extra_args[use_kwargs['y']] = y
        return op(x, **extra_args)
      else:
        assert 'x' not in use_kwargs, use_kwargs
        return op(x, y, **extra_args)

    result = compute(x, y)

    # Run the wrapped op on the dense values, for comparison.
    dense_x = x.flat_values if ragged_tensor.is_ragged(x) else x
    dense_y = y.flat_values if ragged_tensor.is_ragged(y) else y
    expected_flat_values = array_ops.reshape(compute(dense_x, dense_y), [-1])

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
          #=====================================================================
          # Keyword Args
          #=====================================================================
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2], [3], [4]], [[], [5, 7, 8]]]),
           'y': ragged_factory_ops.constant_value(
               [[[3, 8], [2], [5]], [[], [1, 9, 8]]]),
           'use_kwargs': {'x': 'x', 'y': 'y'}},
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2]], [[3, 4], [5, 6], [7, 8]]],
              ragged_rank=1),
           'y': ragged_factory_ops.constant_value(
               [[[9, 3]], [[5, 2], [3, 4], [7, 6]]],
               ragged_rank=1),
           'use_kwargs': {'x': 'x', 'y': 'y'}},
          {'x': ragged_factory_ops.constant_value(
              [[[1, 2]], [[3, 4], [5, 6], [7, 8]]],
              ragged_rank=1),
           'y': ragged_factory_ops.constant_value(
               [[[9, 3]], [[5, 2], [3, 4], [7, 6]]],
               ragged_rank=1),
           'use_kwargs': {'y': 'y'}},
      ] +
      #=========================================================================
      # Test each binary op.
      #=========================================================================
      [{'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
        'y': ragged_factory_ops.constant_value([[5.0, 1.0], [12.0]]),
        'op': op}
       for op in test_ops.BINARY_ASSERT_OPS] +
      [{'x': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
        'y': ragged_factory_ops.constant_value([[-2.0, 3.0], [-3.0]]),
        'op': op}
       for op in test_ops.BINARY_ASSERT_OPS] +
      [{'x': ragged_factory_ops.constant_value([[5, 1], [12]]),
        'y': ragged_factory_ops.constant_value([[-2, 3], [-3]]),
        'op': op}
       for op in test_ops.BINARY_ASSERT_OPS] +
      [{'x': ragged_factory_ops.constant_value([[True, True], [False]]),
        'y': ragged_factory_ops.constant_value([[False, True], [False]]),
        'op': op}
       for op in (check_ops.assert_equal_v2, check_ops.assert_none_equal_v2)
      ])  # pyformat: disable
  def testBinaryAssertOp(self, x, y, op=check_ops.assert_equal_v2,
                         **extra_args):
    """Test the binary assert functions for ragged tensors."""

    def check_binary_assert_pass(assert_op, x, y):
      assert_passed = True
      try:
        result = assert_op(x, y)
        if result is not None:  # in graph mode
          with ops.control_dependencies([result]):
            eval_tensor = array_ops.zeros([])
          self.evaluate(eval_tensor)
      except (ValueError, errors.InvalidArgumentError):
        assert_passed = False
      return assert_passed

    op_assert_pass = check_binary_assert_pass(op, x, y)

    dense_x = x.flat_values if ragged_tensor.is_ragged(x) else x
    dense_y = y.flat_values if ragged_tensor.is_ragged(y) else y
    # Run the wrapped op on the converted tensor values, for comparison.
    expected_assert_pass = check_binary_assert_pass(op, dense_x, dense_y)

    self.assertEqual(op_assert_pass, expected_assert_pass)

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

  def testAllElementwiseOpsAreIncludedInRaggedTensorTestOps(self):
    other_tested_ops = [
        # Elementwise ops that have explicit/bespoke test cases in this file.
        string_ops.string_to_hash_bucket,
        string_ops.string_to_hash_bucket_v1,
        string_ops.string_to_hash_bucket_fast,
        string_ops.string_to_hash_bucket_strong,
        string_ops.string_to_number,
        string_ops.regex_full_match,
        string_ops.regex_replace,
        string_ops.substr,
        string_ops.substr_v2,
        string_ops.substr_deprecated,
        string_ops.unicode_transcode,
        clip_ops.clip_by_value,
        array_ops.check_numerics,
        math_ops.cast,
        math_ops.saturate_cast,
        math_ops.nextafter,
        math_ops.tensor_equals,
        math_ops.tensor_not_equals,
        math_ops.to_bfloat16,
        math_ops.to_complex128,
        math_ops.to_complex64,
        math_ops.to_double,
        math_ops.to_float,
        math_ops.to_int32,
        math_ops.to_int64,
        math_ops.scalar_mul,
        math_ops.scalar_mul_v2,
        image_ops_impl.adjust_brightness,
        image_ops_impl.adjust_gamma,
        image_ops_impl.stateless_random_brightness,
        image_ops_impl.random_brightness,
        image_ops_impl.convert_image_dtype,
        nn_impl.sigmoid_cross_entropy_with_logits_v2,
    ]
    untested_ops = (
        set(dispatch.unary_elementwise_apis() +
            dispatch.binary_elementwise_apis()) -
        set(test_ops.UNARY_FLOAT_OPS + test_ops.UNARY_BOOL_OPS +
            test_ops.UNARY_STRING_OPS + test_ops.UNARY_INT_OPS +
            test_ops.BINARY_FLOAT_OPS + test_ops.BINARY_BOOL_OPS +
            test_ops.BINARY_INT_OPS + other_tested_ops))
    untested_ops = sorted(f'{x.__module__}.{x.__name__}' for x in untested_ops)
    self.assertEmpty(
        untested_ops, 'One or more ops elementwise are not tested; please'
        ' add them to ragged_tensor_test_ops.py or ragged_dispatch_test.py')

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

  @parameterized.parameters([
      dict(
          x=ragged_factory_ops.constant_value([[1, 2, 3], [4, 5]],
                                              row_splits_dtype=dtypes.int64),
          y=[1],
          expected=[[2, 3, 4], [5, 6]],
          expected_row_splits_dtype=dtypes.int64),
      dict(
          x=ragged_factory_ops.constant_value([[1, 2, 3], [4, 5]],
                                              row_splits_dtype=dtypes.int32),
          y=[1],
          expected=[[2, 3, 4], [5, 6]],
          expected_row_splits_dtype=dtypes.int32),
      dict(
          x=[1],
          y=ragged_factory_ops.constant_value([[1, 2, 3], [4, 5]],
                                              row_splits_dtype=dtypes.int64),
          expected=[[2, 3, 4], [5, 6]],
          expected_row_splits_dtype=dtypes.int64),
      dict(
          x=[1],
          y=ragged_factory_ops.constant_value([[1, 2, 3], [4, 5]],
                                              row_splits_dtype=dtypes.int32),
          expected=[[2, 3, 4], [5, 6]],
          expected_row_splits_dtype=dtypes.int32),
  ])
  def testElementwiseOpBroadcastTensorAndRaggedTensor(
      self, x, y, expected, expected_row_splits_dtype):
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, dtype=dtypes.int32)
    y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, dtype=dtypes.int32)
    result = x + y
    self.assertAllEqual(result, expected)
    self.assertEqual(result.row_splits.dtype, expected_row_splits_dtype)

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
          op=array_ops_stack.stack,
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
          expected=[7.0, 2.0],
          rtol=1e-12,
      ),
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
          rtol=1e-6),
      dict(
          op=array_ops.bitcast,
          kwargs={
              'input':
                  ragged_factory_ops.constant_value([[1, 2], [-1]],
                                                    dtype=dtypes.int64),
              'type':
                  dtypes.uint64
          },
          expected=ragged_factory_ops.constant_value([[1, 2], [-1]],
                                                     dtype=dtypes.uint64)),
      dict(
          op=array_ops.split,
          kwargs={
              'value': ragged_factory_ops.constant_value([[1], [2, 3, 4]]),
              'num_or_size_splits': 2,
          },
          result_is_list=True,
          expected=[
              ragged_factory_ops.constant_value([[1]]),
              ragged_factory_ops.constant_value([[2, 3, 4]]),
          ]),
      dict(
          op=array_ops.reshape,
          kwargs=lambda: {
              'tensor': ragged_factory_ops.constant([[1, 2], [3]]),
              'shape': DynamicRaggedShape.from_lengths([3, (1, 0, 2)]),
          },
          expected=[[1], [], [2, 3]]),
      dict(
          op=array_ops.reshape,
          kwargs=lambda: {
              'tensor': [[1, 2], [3, 4]],
              'shape': DynamicRaggedShape.from_lengths([3, (1, 0, 3)]),
          },
          expected=[[1], [], [2, 3, 4]]),
      dict(
          op=array_ops.reshape,
          kwargs=lambda: {
              'tensor': ragged_factory_ops.constant([[1, 2], [3]]),
              'shape': [3],
          },
          expected=[1, 2, 3]),
      dict(
          op=array_ops.broadcast_to,
          kwargs=lambda: {
              'input': 3,
              'shape': DynamicRaggedShape.from_lengths([3, (1, 0, 2)])
          },
          expected=[[3], [], [3, 3]]),
      dict(
          op=array_ops.shape,
          kwargs=lambda: {
              'input': ragged_factory_ops.constant([(1, 2), (3,)]),
              'out_type': dtypes.int64
          },
          expected=lambda: DynamicRaggedShape.from_lengths([2, (2, 1)])),
      dict(
          op=array_ops.shape_v2,
          kwargs=lambda: {
              'input': ragged_factory_ops.constant([(1, 2), (3,)]),
              'out_type': dtypes.int64
          },
          expected=lambda: DynamicRaggedShape.from_lengths([2, (2, 1)])),
      dict(
          op=array_ops.broadcast_dynamic_shape,
          kwargs=lambda: {
              'shape_x': DynamicRaggedShape.from_lengths([2, (2, 3), 1]),
              'shape_y': DynamicRaggedShape.from_lengths([5])
          },
          expected=lambda: DynamicRaggedShape.from_lengths([2, (2, 3), 5])),
      dict(
          op=array_ops.broadcast_dynamic_shape,
          kwargs=lambda: {
              'shape_x': DynamicRaggedShape.from_lengths([2, (2, 3), 1]),
              'shape_y': [5],
          },
          expected=lambda: DynamicRaggedShape.from_lengths([2, (2, 3), 5])),
      dict(
          op=array_ops.ones,
          kwargs=lambda: {
              'shape': DynamicRaggedShape.from_lengths([2, (2, 3)]),
          },
          expected=[[1.0, 1.0], [1.0, 1.0, 1.0]]),
      dict(
          op=array_ops.zeros,
          kwargs=lambda: {
              'shape': DynamicRaggedShape.from_lengths([2, (2, 3)]),
          },
          expected=[[0.0, 0.0], [0.0, 0.0, 0.0]]),
      dict(
          op=array_ops.fill,
          kwargs=lambda: {
              'dims': DynamicRaggedShape.from_lengths([2, (2, 3)]),
              'value': 5
          },
          expected=[[5.0, 5.0], [5.0, 5.0, 5.0]]),
  ])
  def testRaggedDispatch(self,
                         op,
                         expected,
                         args=(),
                         result_is_list=False,
                         rtol=None,
                         kwargs=None):
    # For some tests, the inputs/outputs to the function need to be
    # constructed late, because they contain tensors.
    if callable(kwargs):
      kwargs = kwargs()
    if callable(args):
      args = args()
    if callable(expected):
      expected = expected()

    kwargs = kwargs or {}
    if rtol is not None:
      assert_fn = lambda x, y: self.assertAllClose(x, y, rtol=rtol)
    else:
      assert_fn = self.assertAllEqual

    result = op(*args, **kwargs)
    if isinstance(expected, DynamicRaggedShape):
      self.assertDynamicRaggedShapeEqual(expected, result)
    elif result_is_list:
      self.assertLen(result, len(expected))
      for (r, e) in zip(result, expected):
        assert_fn(r, e)
    else:
      assert_fn(result, expected)

  def testTensorEquals(self):
    a = ragged_factory_ops.constant([[1, 2], [3]])
    b = ragged_factory_ops.constant([[4, 5], [3]])
    c = 2
    d = ragged_factory_ops.constant([[4, 5], [3, 2, 1]])

    if tf2.enabled() and ops.executing_eagerly_outside_functions():
      # Value-based equality:
      self.assertAllEqual(
          math_ops.tensor_equals(a, b), [[False, False], [True]])
      self.assertAllEqual(
          math_ops.tensor_not_equals(a, b), [[True, True], [False]])

      # Value-based equality (w/ broadcasting):
      self.assertAllEqual(
          math_ops.tensor_equals(a, c), [[False, True], [False]])
      self.assertAllEqual(
          math_ops.tensor_not_equals(a, c), [[True, False], [True]])
      self.assertFalse(math_ops.tensor_equals(a, d),
                       msg='not broadcast-compatible')
      self.assertTrue(math_ops.tensor_not_equals(a, d),
                      msg='not broadcast-compatible')
    else:
      # Identity-based equality:
      self.assertAllEqual(math_ops.tensor_equals(a, a), True)
      self.assertAllEqual(math_ops.tensor_equals(a, b), False)
      self.assertAllEqual(math_ops.tensor_not_equals(a, b), True)

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
        'bitcast', 'bitwise.bitwise_and', 'bitwise.bitwise_or',
        'bitwise.bitwise_xor', 'bitwise.invert', 'bitwise.left_shift',
        'bitwise.right_shift', 'clip_by_value', 'concat',
        'debugging.assert_equal', 'debugging.assert_near',
        'debugging.assert_none_equal', 'debugging.assert_greater',
        'debugging.assert_greater_equal', 'debugging.assert_less',
        'debugging.assert_less_equal', 'debugging.check_numerics',
        'cast', 'dtypes.complex',
        'dtypes.saturate_cast', 'expand_dims', 'gather_nd', 'gather',
        'io.decode_base64', 'io.decode_compressed', 'io.encode_base64',
        'math.abs', 'math.acos', 'math.acosh', 'math.add_n', 'math.add',
        'math.angle', 'math.asin', 'math.asinh', 'math.atan2', 'math.atan',
        'math.atanh', 'math.bessel_i0', 'math.bessel_i0e', 'math.bessel_i1',
        'math.bessel_i1e', 'math.ceil', 'math.conj', 'math.cos', 'math.cosh',
        'math.digamma', 'math.divide_no_nan', 'math.divide', 'math.equal',
        'math.erf', 'math.erfc', 'math.erfcinv', 'math.erfinv', 'math.exp',
        'math.expm1', 'math.floor', 'math.floordiv', 'math.floormod',
        'math.greater_equal', 'math.greater', 'math.imag', 'math.is_finite',
        'math.is_inf', 'math.is_nan', 'math.less_equal', 'math.less',
        'math.lgamma', 'math.log1p', 'math.log_sigmoid', 'math.log',
        'math.logical_and', 'math.logical_not', 'math.logical_or',
        'math.logical_xor', 'math.maximum', 'math.minimum',
        'math.multiply_no_nan', 'math.multiply', 'math.negative',
        'math.nextafter', 'math.not_equal', 'math.pow', 'math.real',
        'math.reciprocal', 'math.reciprocal_no_nan', 'math.reduce_any',
        'math.reduce_max', 'math.reduce_mean', 'math.reduce_variance',
        'math.reduce_std', 'math.reduce_min', 'math.reduce_prod',
        'math.reduce_sum', 'math.rint', 'math.round', 'math.rsqrt', 'math.sign',
        'math.sigmoid', 'math.sin', 'math.sinh', 'math.softplus', 'math.sqrt',
        'math.square', 'math.squared_difference', 'math.subtract', 'math.tan',
        'math.tanh', 'math.truediv', 'math.unsorted_segment_max',
        'math.unsorted_segment_mean', 'math.unsorted_segment_min',
        'math.unsorted_segment_prod', 'math.unsorted_segment_sqrt_n',
        'math.unsorted_segment_sum', 'one_hot', 'ones_like', 'rank', 'realdiv',
        'math.reduce_all', 'size', 'split', 'squeeze', 'stack',
        'strings.as_string', 'strings.join', 'strings.length',
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

  def testDispatchWithVariable(self):
    x = ragged_factory_ops.constant([[1, 2], [3, 4, 5]])
    v = variables.Variable(10)
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(math_ops.add(x, v), [[11, 12], [13, 14, 15]])

  def testAssertType(self):
    x = ragged_factory_ops.constant([[1., 2.], [3.]])
    with ops.control_dependencies(
        [check_ops.assert_type(x, dtypes.float32)]):
      y = array_ops.identity(x)
    self.assertAllEqual(x, y)

  def assertDynamicRaggedShapeEqual(self, expected, result):
    self.assertIsInstance(result, DynamicRaggedShape)
    self.assertTrue(expected._type_spec.is_compatible_with(result))
    for (e, r) in zip(
        nest.flatten(expected, expand_composites=True),
        nest.flatten(result, expand_composites=True)):
      self.assertAllEqual(e, r)


if __name__ == '__main__':
  googletest.main()
