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
"""Tests for ragged_array_ops.gather."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.platform import googletest


class RaggedGatherOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testDocStringExamples(self):
    params = constant_op.constant(['a', 'b', 'c', 'd', 'e'])
    indices = constant_op.constant([3, 1, 2, 1, 0])
    ragged_params = ragged_factory_ops.constant([['a', 'b', 'c'], ['d'], [],
                                                 ['e']])
    ragged_indices = ragged_factory_ops.constant([[3, 1, 2], [1], [], [0]])
    self.assertAllEqual(
        ragged_gather_ops.gather(params, ragged_indices),
        [[b'd', b'b', b'c'], [b'b'], [], [b'a']])
    self.assertAllEqual(
        ragged_gather_ops.gather(ragged_params, indices),
        [[b'e'], [b'd'], [], [b'd'], [b'a', b'b', b'c']])
    self.assertAllEqual(
        ragged_gather_ops.gather(ragged_params, ragged_indices),
        [[[b'e'], [b'd'], []], [[b'd']], [], [[b'a', b'b', b'c']]])

  def testTensorParamsAndTensorIndices(self):
    params = ['a', 'b', 'c', 'd', 'e']
    indices = [2, 0, 2, 1]
    self.assertAllEqual(
        ragged_gather_ops.gather(params, indices), [b'c', b'a', b'c', b'b'])
    self.assertIsInstance(ragged_gather_ops.gather(params, indices), ops.Tensor)

  def testRaggedParamsAndTensorIndices(self):
    params = ragged_factory_ops.constant([['a', 'b'], ['c', 'd', 'e'], ['f'],
                                          [], ['g']])
    indices = [2, 0, 2, 1]
    self.assertAllEqual(
        ragged_gather_ops.gather(params, indices),
        [[b'f'], [b'a', b'b'], [b'f'], [b'c', b'd', b'e']])

  def testTensorParamsAndRaggedIndices(self):
    params = ['a', 'b', 'c', 'd', 'e']
    indices = ragged_factory_ops.constant([[2, 1], [1, 2, 0], [3]])
    self.assertAllEqual(
        ragged_gather_ops.gather(params, indices),
        [[b'c', b'b'], [b'b', b'c', b'a'], [b'd']])

  def testRaggedParamsAndRaggedIndices(self):
    params = ragged_factory_ops.constant([['a', 'b'], ['c', 'd', 'e'], ['f'],
                                          [], ['g']])
    indices = ragged_factory_ops.constant([[2, 1], [1, 2, 0], [3]])
    self.assertAllEqual(
        ragged_gather_ops.gather(params, indices),
        [[[b'f'], [b'c', b'd', b'e']],                # [[p[2], p[1]      ],
         [[b'c', b'd', b'e'], [b'f'], [b'a', b'b']],  #  [p[1], p[2], p[0]],
         [[]]]                                        #  [p[3]            ]]
    )  # pyformat: disable

  def testRaggedParamsAndScalarIndices(self):
    params = ragged_factory_ops.constant([['a', 'b'], ['c', 'd', 'e'], ['f'],
                                          [], ['g']])
    indices = 1
    self.assertAllEqual(
        ragged_gather_ops.gather(params, indices), [b'c', b'd', b'e'])

  def test3DRaggedParamsAnd2DTensorIndices(self):
    params = ragged_factory_ops.constant([[['a', 'b'], []],
                                          [['c', 'd'], ['e'], ['f']], [['g']]])
    indices = [[1, 2], [0, 1], [2, 2]]
    self.assertAllEqual(
        ragged_gather_ops.gather(params, indices),
        [[[[b'c', b'd'], [b'e'], [b'f']], [[b'g']]],            # [[p1, p2],
         [[[b'a', b'b'], []], [[b'c', b'd'], [b'e'], [b'f']]],  #  [p0, p1],
         [[[b'g']], [[b'g']]]]                                  #  [p2, p2]]
    )  # pyformat: disable

  def testTensorParamsAnd4DRaggedIndices(self):
    indices = ragged_factory_ops.constant(
        [[[[3, 4], [0, 6]], []], [[[2, 1], [1, 0]], [[2, 5]], [[2, 3]]],
         [[[1, 0]]]],  # pyformat: disable
        ragged_rank=2,
        inner_shape=(2,))
    params = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    self.assertAllEqual(
        ragged_gather_ops.gather(params, indices),
        [[[[b'd', b'e'], [b'a', b'g']], []],
         [[[b'c', b'b'], [b'b', b'a']], [[b'c', b'f']], [[b'c', b'd']]],
         [[[b'b', b'a']]]])  # pyformat: disable

  def testOutOfBoundsError(self):
    tensor_params = ['a', 'b', 'c']
    tensor_indices = [0, 1, 2]
    ragged_params = ragged_factory_ops.constant([['a', 'b'], ['c']])
    ragged_indices = ragged_factory_ops.constant([[0, 3]])
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r'indices\[1\] = 3 is not in \[0, 3\)'):
      self.evaluate(ragged_gather_ops.gather(tensor_params, ragged_indices))
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r'indices\[2\] = 2 is not in \[0, 2\)'):
      self.evaluate(ragged_gather_ops.gather(ragged_params, tensor_indices))
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r'indices\[1\] = 3 is not in \[0, 2\)'):
      self.evaluate(ragged_gather_ops.gather(ragged_params, ragged_indices))

  def testUnknownIndicesRankError(self):
    if context.executing_eagerly():
      return
    params = ragged_factory_ops.constant([], ragged_rank=1)
    indices = constant_op.constant([0], dtype=dtypes.int64)
    indices = array_ops.placeholder_with_default(indices, None)
    self.assertRaisesRegexp(ValueError,
                            r'indices\.shape\.ndims must be known statically',
                            ragged_gather_ops.gather, params, indices)

  # pylint: disable=bad-whitespace
  @parameterized.parameters([
      # params.shape=[2, None]; indices.shape=[3]
      dict(
          params        = [[1.0, 2.0], [3.0, 4.0, 5.0]],
          indices       = [0, 0, 1],
          expected_out  = [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0, 5.0]],
          out_grad      = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6, 0.7]],
          expected_grad = [[0.4, 0.6], [0.5, 0.6, 0.7]]),
      # params.shape=[2, None]; indices.shape=[0]
      dict(
          params        = [[1, 2], [3, 4, 5]],
          indices       = [],
          expected_out  = [],
          out_grad      = [],
          expected_grad = [[0, 0], [0, 0, 0]]),
      # params.shape=[2, None]; indices.shape=[2, 2]
      dict(
          params        = [[1.0, 2.0], [3.0, 4.0, 5.0]],
          indices       = [[0, 0], [1, 0]],
          expected_out  = [[[1.0, 2.0], [1.0, 2.0]],
                           [[3.0, 4.0, 5.0], [1.0, 2.0]]],
          out_grad      = [[[0.1, 0.2], [0.3, 0.4]],
                           [[0.5, 0.6, 0.7], [0.8, 0.9]]],
          expected_grad = [[1.2, 1.5], [0.5, 0.6, 0.7]]),
      # params.shape=[3, None, None]; indices.shape=[3]
      dict(
          params        = [[[1, 2], [3, 4, 5]], [[6.0]], [[7.0, 8.0]]],
          indices       = [2, 1, 2],
          expected_out  = [[[7.0, 8.0]], [[6.0]], [[7.0, 8.0]]],
          out_grad      = [[[0.1, 0.2]], [[0.3]], [[0.4, 0.5]]],
          expected_grad = [[[0, 0], [0, 0, 0]], [[0.3]], [[0.5, 0.7]]]),
      # params.shape=[3, None, None]; indices.shape=[0]
      dict(
          params        = [[[1, 2], [3, 4, 5]], [[6.0]], [[7.0, 8.0]]],
          indices       = [2, 1, 2],
          expected_out  = [[[7.0, 8.0]], [[6.0]], [[7.0, 8.0]]],
          out_grad      = [[[0.1, 0.2]], [[0.3]], [[0.4, 0.5]]],
          expected_grad = [[[0, 0], [0, 0, 0]], [[0.3]], [[0.5, 0.7]]]),
      # params.shape=[0, None]; indices.shape=[0]
      dict(
          params        = [],
          indices       = [],
          expected_out  = [],
          out_grad      = [],
          expected_grad = [],
          params_ragged_rank = 1),
      # params.shape=[2, None, 2]; indices.shape=[3]
      dict(
          params        = [[[1, 2], [3, 4]], [], [[5, 6]]],
          indices       = [1, 1, 2, 0, 2],
          expected_out  = [[], [], [[5, 6]], [[1, 2], [3, 4]], [[5, 6]]],
          out_grad      = [[], [], [[1, 2]], [[3, 4], [5, 6]], [[7, 7]]],
          expected_grad = [[[3, 4], [5, 6]], [], [[8, 9]]],
          params_ragged_rank = 1),
  ])  # pyformat: disable
  @test_util.run_deprecated_v1
  def testGradient(self,
                   params,
                   indices,
                   expected_out,
                   out_grad,
                   expected_grad,
                   params_ragged_rank=None):
    """Tests that ragged_gather generates the right gradient.

    Args:
      params: The `params` that should be passed to `gather`.
      indices: The `indices` that should be passed to `gather`.
      expected_out: The expected value of `gather(params, indices)`.
        `expected_out.shape = indices.shape + params.shape[1:]`.
      out_grad: The value that should be fed in as the gradient for `out`
        when testing the gradient of `ragged_gather`.  Must have the same
        shape as `expected_out`.
      expected_grad: The expected gradient for that should be returned for
        `params`.  Must have hte same shape as `params`.
      params_ragged_rank: The ragged_rank of `params`.
    """
    if context.executing_eagerly():
      return

    params = ragged_factory_ops.constant(
        params, dtype=dtypes.float32, ragged_rank=params_ragged_rank)
    indices = constant_op.constant(indices, dtype=dtypes.int32)
    out_ragged_rank = params.ragged_rank + indices.shape.ndims - 1
    out_grad = ragged_factory_ops.constant(
        out_grad, dtype=dtypes.float32, ragged_rank=out_ragged_rank)
    expected_out = ragged_factory_ops.constant(
        expected_out, dtype=dtypes.float32, ragged_rank=out_ragged_rank)
    expected_grad = ragged_factory_ops.constant(
        expected_grad,
        dtype=dtypes.float32,
        ragged_rank=params.ragged_rank)

    out = ragged_gather_ops.gather(params, indices)
    self.assertAllClose(out, expected_out)

    grads = gradients_impl.gradients(
        out.flat_values,
        (params.nested_row_splits + (params.flat_values, indices,)),
        out_grad.flat_values)
    param_nested_splits_grads = grads[:-2]
    params_flat_values_grad = grads[-2]
    indices_grad = grads[-1]
    self.assertEqual(indices_grad, None)
    for splits_grad in param_nested_splits_grads:
      self.assertEqual(splits_grad, None)

    # The gradient generates an IndexedSlices; convert back to a normal Tensor.
    self.assertIsInstance(params_flat_values_grad, indexed_slices.IndexedSlices)
    params_flat_values_grad = ops.convert_to_tensor(params_flat_values_grad)

    params_grad = params.with_flat_values(params_flat_values_grad)
    self.assertAllClose(params_grad, expected_grad, atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
  googletest.main()
