# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.ragged.cross and tf.ragged.matmul."""

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


def T(shape):
  """Make a dense test Tensor with the indicated shape."""
  # Values are arbitrary, but make them nontrivial because we use them to
  # test matmul against a reference implementation.
  values = math_ops.range(math_ops.reduce_prod(shape))
  return array_ops.reshape(values, shape)


@test_util.run_all_in_graph_and_eager_modes
class RaggedMatmulOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def eager_ragged_matmul(self, a, b, **kwargs):
    """Reference implementation for ragged matmul."""
    if len(a.shape) > 2:
      return [
          self.eager_ragged_matmul(a[i], b[i], **kwargs)
          for i in range(a.shape[0])
      ]
    a = self.ensure_non_ragged(a)
    b = self.ensure_non_ragged(b)
    return self.evaluate(math_ops.matmul(a, b, **kwargs)).tolist()

  def ensure_non_ragged(self, x):
    """Returns x as a Tensor.  Fails if x contains ragged rows."""
    if not isinstance(x, ragged_tensor.RaggedTensor):
      return x
    x_uniform = x.to_tensor()
    self.assertAllEqual(array_ops.size(x), array_ops.size(x_uniform))
    return x_uniform

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters([
      # The test names below have the form '<a.shape>_times_<b.shape>'.
      # Within <a.shape> and <b.shape>, constants are used for dense dimensions
      # and letters (e.g. B, I, J, and K) are used for raggged dimensions.
      dict(  # Uses math_ops.matmul
          testcase_name='dense',
          a=lambda: T([3, 4, 5]),
          b=lambda: T([3, 5, 6]),
          expected_shape=[3, 4, 6]),
      dict(  # Uses matmul_2d
          testcase_name='2x3_times_3x1',
          a=lambda: ragged_factory_ops.constant([[1, 2, 3], [4, 5, 6]]),
          b=lambda: ragged_factory_ops.constant([[5], [4], [3]]),
          expected_shape=[2, None]),
      dict(  # Uses matmul_3d_with_map_fn
          testcase_name='2xIxJ_times_2xJxK',
          a=lambda: ragged_concat_ops.stack([T([15, 32]),
                                             T([10, 20])]),
          b=lambda: ragged_concat_ops.stack([T([32, 19]),
                                             T([20, 13])]),
          expected_shape=[2, None, None]),
      dict(  # Uses matmul_3d_with_map_fn
          testcase_name='2xIxJ_times_2xJx12',
          a=lambda: ragged_concat_ops.stack([T([15, 4]), T([10, 2])]),
          b=lambda: ragged_factory_ops.constant([[[1, 2], [3, 4], [5, 6],
                                                  [7, 8]], [[9, 10], [11, 12]]],
                                                ragged_rank=1),
          expected_shape=[2, None, 2]),
      dict(  # Uses matmul_3d_with_map_fn
          testcase_name='2xIx8_times_2x8x12',
          a=lambda: ragged_concat_ops.stack([T([15, 8]), T([10, 8])]),
          b=lambda: T([2, 8, 12]),
          expected_shape=[2, None, 12]),
      dict(  # Uses matmul_3d_with_map_fn
          testcase_name='2x15x32_times_2x32xK',
          a=lambda: T([2, 15, 32]),
          b=lambda: ragged_concat_ops.stack([T([32, 19]),
                                             T([32, 13])])),
      dict(  # Uses matmul_3d_with_batch_dim_folding
          testcase_name='2xIx3_times_2x3x8',
          a=lambda: ragged_factory_ops.constant(
              [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [8, 7, 6], [5, 4, 3]]],
              ragged_rank=1),
          b=lambda: T([2, 3, 8]),
          expected_shape=[2, None, 8]),
      dict(  # Multiple batch dimensions
          testcase_name='3xBx5x7_times_3xBx7x9',
          a=lambda: ragged_tensor.RaggedTensor.from_row_lengths(
              values=T([10, 5, 7]), row_lengths=[3, 2, 0, 5]),
          b=lambda: ragged_tensor.RaggedTensor.from_row_lengths(
              values=T([10, 7, 9]), row_lengths=[3, 2, 0, 5])),
      dict(  # Multiple batch dimensions
          testcase_name='3xBx5x7_times_3x2x7x9',
          a=lambda: ragged_tensor.RaggedTensor.from_tensor(
              T([3, 2, 5, 7]), ragged_rank=1),
          b=lambda: T([3, 2, 7, 9])),
      dict(  # Multiple batch dimensions
          testcase_name='3xBxIx7_times_3x2x7x9',
          a=lambda: ragged_tensor.RaggedTensor.from_tensor(
              T([3, 2, 5, 7]), ragged_rank=2),
          b=lambda: T([3, 2, 7, 9])),
      dict(  # Multiple batch dimensions
          testcase_name='3xBxIxJ_times_3x2x7x9',
          a=lambda: ragged_tensor.RaggedTensor.from_tensor(
              T([3, 2, 5, 7]), ragged_rank=3),
          b=lambda: T([3, 2, 7, 9])),
      dict(  # Multiple batch dimensions
          testcase_name='3x2x5x7_times_3xBx7x9',
          a=lambda: T([3, 2, 5, 7]),
          b=lambda: ragged_tensor.RaggedTensor.from_tensor(
              T([3, 2, 7, 9]), ragged_rank=1)),
      dict(  # Multiple batch dimensions
          testcase_name='3x2x5x7_times_3xBxJx9',
          a=lambda: T([3, 2, 5, 7]),
          b=lambda: ragged_tensor.RaggedTensor.from_tensor(
              T([3, 2, 7, 9]), ragged_rank=2)),
      dict(  # Multiple batch dimensions
          testcase_name='3x2x5x7_times_3xBxJxK',
          a=lambda: T([3, 2, 5, 7]),
          b=lambda: ragged_tensor.RaggedTensor.from_tensor(
              T([3, 2, 7, 9]), ragged_rank=3)),
      dict(
          testcase_name='2x3xI_times_2x3x4_transpose_a',
          a=lambda: ragged_factory_ops.constant(
              [[[1], [2], [3]], [[4, 5, 6], [7, 8, 9], [10, 11, 12]]]),
          b=lambda: T([2, 3, 4]),
          transpose_a=True,
          expected_shape=[2, None, 4]),
      dict(
          testcase_name='2x3xI_times_2x4_3_transpose_a_transpose_b',
          a=lambda: ragged_factory_ops.constant(
              [[[1], [2], [3]], [[4, 5, 6], [7, 8, 9], [10, 11, 12]]]),
          b=lambda: T([2, 4, 3]),
          transpose_a=True,
          transpose_b=True,
          expected_shape=[2, None, 4]),
      dict(
          testcase_name='2xIxJ_times_2x5xJ_transpose_b',
          a=lambda: ragged_factory_ops.constant([[[1, 2], [3, 4], [5, 6]],
                                                 [[1, 2, 3], [4, 5, 6]]]),
          b=lambda: ragged_factory_ops.constant([[[3, 1], [4, 1], [5, 9], [
              1, 2
          ], [3, 4]], [[2, 4, 6], [1, 3, 5], [7, 8, 9], [1, 2, 3], [3, 2, 1]]]),
          transpose_b=True,
          expected_shape=[2, None, 5]),
  ])
  def testMatmul(self, a, b, expected_shape=None, **kwargs):
    if callable(a):
      a = a()
    if callable(b):
      b = b()
    actual = ragged_math_ops.matmul(a, b, **kwargs)
    expected = self.eager_ragged_matmul(a, b, **kwargs)
    self.assertAllEqual(actual, expected)

    if expected_shape is not None and not kwargs:
      if context.executing_eagerly():
        self.assertTrue(actual.shape.is_compatible_with(expected_shape))
      else:
        self.assertEqual(actual.shape.as_list(), expected_shape)

  @parameterized.parameters([
      dict(
          a=lambda: ragged_factory_ops.constant([[1, 2, 3], [4, 5]]),
          b=lambda: ragged_factory_ops.constant([[5], [4], [3]]),
          exc=errors.InvalidArgumentError,
          message='The matrices in `a` and `b` may not be ragged in '
          'their innermost dimension.'),
      dict(
          a=lambda: ragged_factory_ops.constant([[1, 2], [4, 5]]),
          b=lambda: ragged_factory_ops.constant([[5], [4], [3]]),
          exc=errors.InvalidArgumentError),
      dict(
          a=lambda: ragged_concat_ops.stack([T([15, 32]),
                                             T([10, 20])]),
          b=lambda: ragged_concat_ops.stack([T([32, 19]),
                                             T([22, 13])]),
          exc=errors.InvalidArgumentError),
      dict(
          a=[[1]],
          b=[[1]],
          transpose_a=True,
          adjoint_a=True,
          exc=ValueError,
          message='Only one of transpose_a and adjoint_a can be True'),
      dict(
          a=[[1]],
          b=[[1]],
          transpose_b=True,
          adjoint_b=True,
          exc=ValueError,
          message='Only one of transpose_b and adjoint_b can be True'),
      dict(
          a=lambda: ragged_factory_ops.constant([[1]]),
          b=lambda: ragged_factory_ops.constant([[1.0]]),
          exc=ValueError,
          message='`a` and `b` must have the same dtype.'),
      dict(
          a=lambda: ragged_factory_ops.constant([[1]]),
          b=lambda: ragged_factory_ops.constant([[[1]]]),
          exc=ValueError,
          message='`a` and `b` must have the same rank.'),
      dict(  # Multiple batch dimensions
          a=lambda: ragged_tensor.RaggedTensor.from_tensor(T([3, 2, 5, 7])),
          b=lambda: ragged_tensor.RaggedTensor.from_tensor(T([3, 3, 7, 9])),
          exc=errors.InvalidArgumentError,
          message='Batch dimensions of `a` and `b` do not have the same size'),
  ])
  def testMatmulError(self, a, b, exc, message=None, **kwargs):
    if callable(a):
      a = a()
    if callable(b):
      b = b()
    with self.assertRaisesRegex(exc, message):
      self.evaluate(ragged_math_ops.matmul(a, b, **kwargs))

  def testUnknownRank(self):
    no_rank_spec = ragged_tensor.RaggedTensorSpec(None, dtypes.int32, 1)
    rank_only_spec = ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32,
                                                    1)

    matmul_no_rank_for_a = def_function.function(
        input_signature=[rank_only_spec, no_rank_spec])(
            ragged_math_ops.matmul)
    matmul_no_rank_for_b = def_function.function(
        input_signature=[no_rank_spec, rank_only_spec])(
            ragged_math_ops.matmul)
    matmul_no_rank_for_a_or_b = def_function.function(
        input_signature=[no_rank_spec, no_rank_spec])(
            ragged_math_ops.matmul)

    a = ragged_factory_ops.constant([[1, 2]])
    b = ragged_factory_ops.constant([[3], [4]])
    self.assertAllEqual(matmul_no_rank_for_a(a, b), [[11]])
    self.assertAllEqual(matmul_no_rank_for_b(a, b), [[11]])
    with self.assertRaisesRegex(
        ValueError, 'matmul requires at least one input to have known '
        'rank if either input is ragged.'):
      matmul_no_rank_for_a_or_b(a, b)


if __name__ == '__main__':
  googletest.main()
