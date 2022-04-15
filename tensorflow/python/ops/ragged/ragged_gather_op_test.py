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

from absl.testing import parameterized

import numpy as np

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
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedGatherOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      # Basic gather (axis=0 and batch_dims=0)
      dict(testcase_name='Params1DTensor_Indices1DTensor',
           params=['a', 'b', 'c', 'd', 'e'],
           indices=[2, 0, 2, 1],
           expected=['c', 'a', 'c', 'b']),
      dict(testcase_name='Params1DTensor_Indices2DRagged',
           params=['a', 'b', 'c', 'd', 'e'],
           indices=[[3, 1, 2], [1], [], [0]],
           expected=[['d', 'b', 'c'], ['b'], [], ['a']]),
      dict(testcase_name='Params2DRagged_Indices0DTensor',
           params=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           indices=1,
           expected=['c', 'd', 'e']),
      dict(testcase_name='Params2DRagged_Indices1DTensor',
           params=[['a', 'b', 'c'], ['d'], [], ['e']],
           indices=[3, 1, 2, 1, 0],
           expected=[
               ['e'], ['d'], [], ['d'], ['a', 'b', 'c']]),
      dict(testcase_name='Params2DRagged_Indices2DRagged',
           params=[['a', 'b', 'c'], ['d'], [], ['e']],
           indices=[[3, 1, 2], [1], [], [0]],
           expected=[
               [['e'], ['d'], []], [['d']], [], [['a', 'b', 'c']]]),
      dict(testcase_name='Params3DRagged_Indices2DTensor',
           params=[
               [['a', 'b'], []], [['c', 'd'], ['e'], ['f']], [['g']]],
           indices=[[1, 2], [0, 1], [2, 2]],
           indices_ragged_rank=0,
           expected=[
               [[['c', 'd'], ['e'], ['f']], [['g']]],
               [[['a', 'b'], []], [['c', 'd'], ['e'], ['f']]],
               [[['g']], [['g']]]]),
      dict(testcase_name='Params3DRagged_Indices3DTensor',
           params=[[['a', 'b'], []],
                   [['c', 'd'], ['e'], ['f']],
                   [['g']]],
           indices=[[[1, 2], [0, 1], [2, 2]], [[0, 0], [1, 2], [0, 1]]],
           indices_ragged_rank=0,
           expected=[
               [[[['c', 'd'], ['e'], ['f']], [['g']]],
                [[['a', 'b'], []], [['c', 'd'], ['e'], ['f']]],
                [[['g']], [['g']]]],
               [[[['a', 'b'], []], [['a', 'b'], []]],
                [[['c', 'd'], ['e'], ['f']], [['g']]],
                [[['a', 'b'], []], [['c', 'd'], ['e'], ['f']]]]]),
      dict(testcase_name='Params1DTensor_Indices4DRaggedRank2',
           params=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
           indices=[[[[3, 4], [0, 6]], []],
                    [[[2, 1], [1, 0]], [[2, 5]], [[2, 3]]],
                    [[[1, 0]]]],
           indices_ragged_rank=2,
           expected=[
               [[['d', 'e'], ['a', 'g']], []],
               [[['c', 'b'], ['b', 'a']], [['c', 'f']], [['c', 'd']]],
               [[['b', 'a']]]]),
      # Batch gather (batch_dims=1)
      dict(testcase_name='Batch1D_Params2DRagged_Indices1DTensor',
           params=[['a', 'b'], ['c'], ['d', 'e', 'f', 'g'], ['h']],
           indices=[1, 0, 3, 0],
           batch_dims=1,
           expected=['b', 'c', 'g', 'h']),
      dict(testcase_name='Batch1D_Params2DRagged_Indices2DTensor',
           params=[['a', 'b'], ['c'], ['d', 'e', 'f', 'g'], ['h']],
           indices=[[1, 0], [0, 0], [3, 1], [0, 0]],
           indices_ragged_rank=0,
           batch_dims=1,
           expected=[['b', 'a'], ['c', 'c'], ['g', 'e'], ['h', 'h']]),
      dict(testcase_name='Batch1D_Params2DRagged_Indices2DRagged',
           params=[['a', 'b'], ['c'], ['d', 'e', 'f', 'g'], ['h']],
           indices=[[1, 0], [], [3, 2, 1], [0]],
           batch_dims=1,
           expected=[['b', 'a'], [], ['g', 'f', 'e'], ['h']]),
      dict(testcase_name='Batch1D_Params3DRagged_Indices3DRagged',
           params=[[['a'], ['b', 'c']],
                   [],
                   [['d', 'e', 'f'], ['g'], ['h', 'i'], ['j']],
                   [['k']]],
           indices=[[[1, 0], []], [], [[3, 2, 1], [0]], [[0]]],
           batch_dims=1,
           expected=[[[['b', 'c'], ['a']], []],
                     [],
                     [[['j'], ['h', 'i'], ['g']], [['d', 'e', 'f']]],
                     [[['k']]]]),
      # Batch gather (batch_dims=2)
      dict(testcase_name='Batch2D_Params3DRagged_Indices2DRagged',
           params=[[['a', 'b', 'c'], ['d', 'e'], ['f']],
                   [['g'], ['h', 'i']]],
           indices=[[0, 1, 0], [0, 1]],
           batch_dims=2,
           expected=[['a', 'e', 'f'], ['g', 'i']]),
      dict(testcase_name='Batch2D_Params3DRagged_Indices3DRagged',
           params=[[['a', 'b', 'c'], ['d', 'e'], ['f']],
                   [['g'], ['h', 'i']]],
           indices=[[[2, 1, 0], [1, 1], [0]], [[0], []]],
           batch_dims=2,
           expected=[[['c', 'b', 'a'], ['e', 'e'], ['f']], [['g'], []]]),
      # Batch gather (batch_dims=3)
      dict(testcase_name='Batch3D_Params4DRagged_Indices3DRagged',
           params=[[[['a', 'b', 'c'], ['d', 'e'], ['f']],
                    [['g'], ['h', 'i']]], [[['j']]]],
           indices=[[[0, 1, 0], [0, 1]], [[0]]],
           batch_dims=3,
           expected=[[['a', 'e', 'f'], ['g', 'i']], [['j']]]),
      # Axis gather (axis=1)
      dict(testcase_name='Params2DRagged_Indices0DTensor_axis_1',
           params=[['a', 'b'], ['c', 'd', 'e'], ['f', 'g'], ['h', 'i', 'j'],
                   ['k', 'l']],
           indices=1,
           axis=1,
           expected=['b', 'd', 'g', 'i', 'l']),
      dict(testcase_name='Params2DRagged_Indices1DTensor_axis_1',
           params=[['a', 'b'], ['c', 'd', 'e'], ['f', 'g'], ['h', 'i', 'j'],
                   ['k', 'l']],
           indices=[1, 0],
           axis=1,
           expected=[['b', 'a'], ['d', 'c'], ['g', 'f'], ['i', 'h'],
                     ['l', 'k']]),
      dict(testcase_name='Params3DRagged_Indices0DTensor_axis_1',
           params=[[['a', 'b'], ['c', 'd', 'e']],
                   [['f', 'g'], ['h', 'i', 'j'], ['k', 'l']]],
           indices=1,
           axis=1,
           expected=[['c', 'd', 'e'], ['h', 'i', 'j']]),
      dict(testcase_name='Params3DRagged_Indices1DTensor_axis_1',
           params=[[['a', 'b'], ['c', 'd', 'e']],
                   [['f', 'g'], ['h', 'i', 'j'], ['k', 'l']]],
           indices=[1, 0],
           axis=1,
           expected=[[['c', 'd', 'e'], ['a', 'b']],
                     [['h', 'i', 'j'], ['f', 'g']]]),
      # Batch/axis gather, batch = 1, axis > batch
      dict(testcase_name='Params3DRagged_Indices1DTensor_batch_1_axis_2',
           params=[[['a', 'b'], ['c', 'd', 'e']],
                   [['f', 'g'], ['h', 'i', 'j'], ['k', 'l']]],
           indices=[1, 0],
           axis=2,
           batch_dims=1,
           expected=[['b', 'd'], ['f', 'h', 'k']]),
      dict(testcase_name='Params4DRagged_Indices1DTensor_batch_1_axis_2',
           params=[[[['a', 'b'], ['c', 'd', 'e']]],
                   [[['f', 'g']], [['h', 'i', 'j'], ['k', 'l']]]],
           indices=[0, 1],
           axis=2,
           batch_dims=1,
           expected=[[['a', 'b']],
                     [['h', 'i', 'j'], ['k', 'l']]]),
  ])  # pyformat: disable
  def testRaggedGather(self,
                       params,
                       indices,
                       expected,
                       axis=None,
                       batch_dims=0,
                       params_ragged_rank=None,
                       indices_ragged_rank=None):
    params = ragged_factory_ops.constant(params, ragged_rank=params_ragged_rank)
    indices = ragged_factory_ops.constant(
        indices, ragged_rank=indices_ragged_rank)
    actual = ragged_gather_ops.gather(
        params, indices, axis=axis, batch_dims=batch_dims)
    self.assertAllEqual(actual, self._str_to_bytes(expected))

  def _str_to_bytes(self, x):
    if isinstance(x, list):
      return [self._str_to_bytes(v) for v in x]
    elif isinstance(x, str) and bytes is not str:
      return bytes(x, 'utf-8')
    else:
      return x

  def testOutOfBoundsError(self):
    tensor_params = ['a', 'b', 'c']
    tensor_indices = [0, 1, 2]
    ragged_params = ragged_factory_ops.constant([['a', 'b'], ['c']])
    ragged_indices = ragged_factory_ops.constant([[0, 3]])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'indices\[1\] = 3 is not in \[0, 3\)'):
      self.evaluate(ragged_gather_ops.gather(tensor_params, ragged_indices))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'indices\[2\] = 2 is not in \[0, 2\)'):
      self.evaluate(ragged_gather_ops.gather(ragged_params, tensor_indices))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'indices\[1\] = 3 is not in \[0, 2\)'):
      self.evaluate(ragged_gather_ops.gather(ragged_params, ragged_indices))

  def testUnknownIndicesRankError(self):
    if context.executing_eagerly():
      return
    params = ragged_factory_ops.constant([], ragged_rank=1)
    indices = constant_op.constant([0], dtype=dtypes.int64)
    indices = array_ops.placeholder_with_default(indices, None)
    self.assertRaisesRegex(ValueError,
                           r'rank\(indices\) must be known statically',
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

  @parameterized.parameters([
      # Basic gather (batch_dims == 0, axis == 0)
      dict(params_shape=[3, 4], indices_shape=[], axis=0),
      dict(params_shape=[3, 4], indices_shape=[5], axis=0),
      dict(params_shape=[3, 4], indices_shape=[2, 5], axis=0),
      # Gather over axis (axis > 0)
      dict(params_shape=[3, 4], indices_shape=[], axis=1),
      dict(params_shape=[3, 4], indices_shape=[2], axis=1),
      dict(params_shape=[3, 4], indices_shape=[2, 5], axis=1),
      dict(params_shape=[7, 3, 1], indices_shape=[2, 4], axis=1),
      dict(params_shape=[3, 4, 5, 6], indices_shape=[2, 1, 7], axis=1),
      dict(params_shape=[7, 3, 5], indices_shape=[], axis=2),
      dict(params_shape=[7, 3, 5], indices_shape=[2], axis=2),
      dict(params_shape=[7, 3, 5], indices_shape=[4, 2], axis=2),
      dict(params_shape=[7, 3, 5, 6], indices_shape=[4, 2], axis=2),
      dict(params_shape=[7, 3, 5, 6], indices_shape=[], axis=3),
      dict(params_shape=[7, 3, 5, 6], indices_shape=[4], axis=3),
      dict(params_shape=[7, 3, 5, 6], indices_shape=[8, 4], axis=3),
      dict(params_shape=[7, 3, 5, 6], indices_shape=[2, 3, 2, 3], axis=3),
      # Batched gather (batch_dims > 0)
      dict(params_shape=[7, 3], indices_shape=[7], batch_dims=1),
      dict(params_shape=[7, 3], indices_shape=[7, 5], batch_dims=1),
      dict(params_shape=[5, 3], indices_shape=[5, 7, 4, 2], batch_dims=1),
      dict(params_shape=[2, 3, 6], indices_shape=[2], batch_dims=1),
      dict(params_shape=[7, 3, 6], indices_shape=[7, 5, 4, 2], batch_dims=1),
      dict(params_shape=[7, 3, 5], indices_shape=[7, 3], batch_dims=2),
      dict(params_shape=[7, 3, 5], indices_shape=[7, 3, 2], batch_dims=2),
      dict(params_shape=[7, 3, 5, 6], indices_shape=[7, 3, 5], batch_dims=3),
      dict(params_shape=[2, 3, 5, 6], indices_shape=[2, 3, 5, 7], batch_dims=3),
      # Batched gather with axis (axis > batch_dims > 0)
      dict(params_shape=[2, 3, 6], indices_shape=[2], axis=2, batch_dims=1),
      dict(params_shape=[2, 3, 6], indices_shape=[2, 4], axis=2, batch_dims=1),
      dict(
          params_shape=[3, 1, 6, 7], indices_shape=[3, 4], axis=3,
          batch_dims=1),
      dict(
          params_shape=[3, 2, 6, 7], indices_shape=[3, 4], axis=3,
          batch_dims=1),
      dict(
          params_shape=[2, 3, 6, 7], indices_shape=[2, 3], axis=3,
          batch_dims=2),
  ])
  def testMatchesDenseGather(self,
                             params_shape,
                             indices_shape,
                             axis=None,
                             batch_dims=0):
    # Build random params & indices matrics w/ the expected shapes.
    if axis is None:
      axis = batch_dims
    params = np.random.randint(100, size=params_shape, dtype=np.int32)
    indices = np.random.randint(
        params_shape[axis], size=indices_shape, dtype=np.int32)

    # Use array_ops.gather to get the expected value.
    expected = array_ops.gather(
        params, indices, axis=axis, batch_dims=batch_dims)

    # Build ragged tensors with varying ragged_ranks from params & axis.
    params_tensors = [params] + [
        ragged_tensor.RaggedTensor.from_tensor(params, ragged_rank=i)
        for i in range(1, len(params_shape))
    ]
    indices_tensors = [indices] + [
        ragged_tensor.RaggedTensor.from_tensor(indices, ragged_rank=i)
        for i in range(1, len(indices_shape))
    ]

    # For each combination of params & axis tensors, check that
    # ragged_gather_ops.gather matches array_ops.gather.
    for params_tensor in params_tensors:
      for indices_tensor in indices_tensors:
        actual = ragged_gather_ops.gather(
            params_tensor, indices_tensor, axis=axis, batch_dims=batch_dims)
        if isinstance(actual, ragged_tensor.RaggedTensor):
          actual = actual.to_tensor()
        self.assertAllEqual(
            expected, actual, 'params.ragged_rank=%s, indices.ragged_rank=%s' %
            (getattr(params_tensor, 'ragged_rank',
                     0), getattr(indices_tensor, 'ragged_rank', 0)))


if __name__ == '__main__':
  googletest.main()
