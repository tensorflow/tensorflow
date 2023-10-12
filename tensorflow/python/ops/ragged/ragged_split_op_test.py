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
"""Tests for ragged_array_ops.split."""

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSplitOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      #=========================================================================
      # Uniform splits.
      #=========================================================================
      dict(
          descr='Uniform splits, rank-2 inputs, axis=0',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=2,
          expected=[
              [[1]],
              [[2, 3, 4]]]),
      dict(
          descr='Uniform 3 splits, rank-2 inputs, axis=0',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 2, 1],  # shape=(3, None)
          num_or_size_splits=3,
          expected=[
              [[1]],
              [[2, 3]],
              [[4]]]),
      dict(
          descr='Uniform 5 splits, rank-2 inputs, axis=0',
          pylist=[1, 2, 3, 4, 5],
          row_lengths=[1, 1, 1, 1, 1],  # shape=(5, None)
          num_or_size_splits=5,
          expected=[
              [[1]],
              [[2]],
              [[3]],
              [[4]],
              [[5]]]),
      dict(
          descr='Uniform 2 splits, rank-2 inputs(empty), axis=0',
          pylist=[1, 2, 3, 4],
          row_lengths=[4, 0],  # shape=(2, None)
          num_or_size_splits=2,
          expected=[
              [[1, 2, 3, 4]],
              [[]]]),
      dict(
          descr='Uniform 2 splits, rank-2 inputs(all empty), axis=0',
          pylist=[],
          row_lengths=[0, 0],  # shape=(2, None)
          num_or_size_splits=2,
          expected=[
              [[]],
              [[]]]),
      dict(
          descr='Uniform 1 split, rank-2 inputs, axis=0',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=1,
          expected=[
              [[1], [2, 3, 4]]]),
      dict(
          descr='Uniform 1 split, rank-2 inputs, axis=1',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=1,
          axis=1,
          expected=[
              [[1], [2, 3, 4]]]),
      dict(
          descr='Uniform 2 split, rank-3 inputs, axis=0',
          pylist=np.arange(4 * 2).reshape(4, 2),
          row_lengths=[1, 3],  # shape=(2, None, 2)
          num_or_size_splits=2,
          expected=[
              [[[0, 1]]],
              [[[2, 3], [4, 5], [6, 7]]]]),
      dict(
          descr='Uniform 2 splits, rank-3 inputs, axis=2',
          pylist=np.arange(4 * 2).reshape(4, 2),
          row_lengths=[1, 3],  # shape=(2, None, 2)
          num_or_size_splits=2,
          axis=2,
          expected=[
              [[[0]], [[2], [4], [6]]],
              [[[1]], [[3], [5], [7]]]]),
      dict(
          descr='Uniform 2 splits, rank-2 float inputs, axis=0',
          pylist=[1.0, 2.0, 3.0, 4.0],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=2,
          expected=[
              [[1.0]],
              [[2.0, 3.0, 4.0]]]),
      dict(
          descr='Uniform 2 splits, rank-2 string inputs, axis=0',
          pylist=[b'a', b'bc', b'', b'd'],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=2,
          expected=[
              [[b'a']],
              [[b'bc', b'', b'd']]]),
      #=========================================================================
      # Ragged splits.
      #=========================================================================
      dict(
          descr='Ragged 2 splits, rank-2 inputs, axis=0',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=[1, 1],
          expected=[
              [[1]],
              [[2, 3, 4]]]),
      dict(
          descr='Ragged 3 splits, rank-2 inputs, axis=0',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 2, 1],  # shape=(3, None)
          num_or_size_splits=[1, 2],
          expected=[
              [[1]],
              [[2, 3], [4]]]),
      dict(
          descr='Ragged 5 splits, rank-2 inputs(empty), axis=0',
          pylist=[1, 2, 3, 4, 5],
          row_lengths=[1, 1, 1, 1, 1],  # shape=(5, None)
          num_or_size_splits=[1, 2, 2, 0],
          expected=[
              [[1]],
              [[2], [3]],
              [[4], [5]],
              []]),
      dict(
          descr='Ragged 2 splits, rank-2 inputs(empty), axis=0',
          pylist=[1, 2, 3, 4],
          row_lengths=[4, 0, 0, 0],  # shape=(2, None)
          num_or_size_splits=[3, 1],
          expected=[
              [[1, 2, 3, 4], [], []],
              [[]]]),
      dict(
          descr='Ragged 2 splits, rank-2 inputs(all empty), axis=0',
          pylist=[],
          row_lengths=[0, 0],  # shape=(2, None)
          num_or_size_splits=[2, 0],
          expected=[
              [[], []],
              []]),
      dict(
          descr='Ragged 1 split, rank-2 inputs, axis=0',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=[2],
          expected=[
              [[1], [2, 3, 4]]]),
      dict(
          descr='Ragged 2 split, rank-3 inputs, axis=0',
          pylist=np.arange(4 * 2).reshape(4, 2),
          row_lengths=[1, 3],  # shape=(2, None, 2)
          num_or_size_splits=[1, 1],
          expected=[
              [[[0, 1]]],
              [[[2, 3], [4, 5], [6, 7]]]]),
      dict(
          descr='Ragged 2 split, rank-3 inputs, axis=-3',
          pylist=np.arange(4 * 2).reshape(4, 2),
          row_lengths=[1, 3],  # shape=(2, None, 2)
          num_or_size_splits=[1, 1],
          expected=[
              [[[0, 1]]],
              [[[2, 3], [4, 5], [6, 7]]]]),
      dict(
          descr='Ragged 2 splits, rank-3 inputs, axis=2',
          pylist=np.arange(4 * 3).reshape(4, 3),
          row_lengths=[1, 3],  # shape=(2, None, 3)
          num_or_size_splits=[2, 1],
          axis=2,
          expected=[
              [[[0, 1]], [[3, 4], [6, 7], [9, 10]]],
              [[[2]], [[5], [8], [11]]]]),
      dict(
          descr='Ragged 2 splits, rank-3 inputs, axis=-1',
          pylist=np.arange(4 * 3).reshape(4, 3),
          row_lengths=[1, 3],  # shape=(2, None, 3)
          num_or_size_splits=[2, 1],
          axis=2,
          expected=[
              [[[0, 1]], [[3, 4], [6, 7], [9, 10]]],
              [[[2]], [[5], [8], [11]]]]),
      dict(
          descr='Ragged 3 splits, rank-2 float inputs, axis=0',
          pylist=[1.0, 2.0, 3.0, 4.0],
          row_lengths=[1, 2, 1],  # shape=(2, None)
          num_or_size_splits=[2, 1],
          expected=[
              [[1.0], [2.0, 3.0]],
              [[4.0]]]),
      dict(
          descr='Ragged 3 splits with name, rank-2 float inputs, axis=0',
          pylist=[1.0, 2.0, 3.0, 4.0],
          row_lengths=[1, 2, 1],  # shape=(2, None)
          num_or_size_splits=[2, 1],
          name='ragged_split',
          expected=[
              [[1.0], [2.0, 3.0]],
              [[4.0]]]),
      dict(
          descr='Ragged 3 splits with num, rank-2 float inputs, axis=0',
          pylist=[1.0, 2.0, 3.0, 4.0],
          row_lengths=[1, 2, 1],  # shape=(2, None)
          num_or_size_splits=[2, 1],
          num=2,
          expected=[
              [[1.0], [2.0, 3.0]],
              [[4.0]]]),
      dict(
          descr='Ragged 2 splits, rank-2 string inputs, axis=0',
          pylist=[b'a', b'bc', b'', b'd'],
          row_lengths=[1, 3, 0],  # shape=(2, None)
          num_or_size_splits=[2, 1],
          expected=[
              [[b'a'], [b'bc', b'', b'd']],
              [[]]]),
  ])  # pyformat: disable
  def testSplit(self,
                descr,
                pylist,
                row_lengths,
                num_or_size_splits,
                expected,
                axis=0,
                num=None,
                name=None):
    rt = ragged_tensor.RaggedTensor.from_row_lengths(pylist, row_lengths)
    result = ragged_array_ops.split(rt, num_or_size_splits, axis, num, name)
    self.assertLen(result, len(expected))
    for res, exp in zip(result, expected):
      self.assertAllEqual(res, exp)

  @parameterized.parameters([
      #=========================================================================
      # Uniform splits errors.
      #=========================================================================
      dict(
          descr='Uniform split, can not split',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=7,
          exception=errors.InvalidArgumentError,
          message='Cannot exactly split'),
      dict(
          descr='Uniform split, ragged dimension',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=2,
          axis=1,
          exception=ValueError,
          message='ragged dimension'),
      dict(
          descr='Uniform split, zero split',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=0,
          exception=ValueError,
          message='must be >=1'),
      #=========================================================================
      # Ragged splits errors.
      #=========================================================================
      dict(
          descr='Ragged split, 2 dimensional size_splits',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=[[1, 1]],
          exception=TypeError,
          message='Python list'),
      dict(
          descr='Ragged split, ragged dimension',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=[1, 1],
          axis=1,
          exception=ValueError,
          message='ragged dimension'),
      dict(
          descr='Ragged split, cannot split',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=[1, 2],
          exception=errors.InvalidArgumentError,
          message='Cannot exactly split'),
      dict(
          descr='Ragged split, num does not match',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=[1, 1],
          num=3,
          exception=ValueError,
          message='`num` does not match'),
      dict(
          descr='Ragged split, negative split',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=[1, -1, 2],
          num=3,
          exception=errors.InvalidArgumentError,
          message='must be non-negative'),
      dict(
          descr='Ragged split, float splits',
          pylist=[1, 2, 3, 4],
          row_lengths=[1, 3],  # shape=(2, None)
          num_or_size_splits=[1.0, 2.0],
          num=2,
          exception=TypeError,
          message='integer'),
  ])  # pyformat: disable
  def testSplitError(self,
                     descr,
                     pylist,
                     row_lengths,
                     num_or_size_splits,
                     exception,
                     message,
                     axis=0,
                     num=None):
    rt = ragged_tensor.RaggedTensor.from_row_lengths(pylist, row_lengths)
    with self.assertRaises(exception):
      result = ragged_array_ops.split(rt, num_or_size_splits, axis, num)
      self.evaluate(result)

  @parameterized.named_parameters([
      ('int32', dtypes.int32),
      ('int64', dtypes.int64)])
  def testSplitTensorDtype(self, dtype):
    rt = ragged_tensor.RaggedTensor.from_row_lengths([1.0, 2.0, 3.0, 4.0],
                                                     [3, 1])
    # split_lengths is a 1-D tensor
    split_lengths = ops.convert_to_tensor([1, 1], dtype=dtype)
    result = ragged_array_ops.split(rt, split_lengths)
    expected = [
        ragged_tensor.RaggedTensor.from_row_lengths([1.0, 2.0, 3.0], [3]),
        ragged_tensor.RaggedTensor.from_row_lengths([4.0], [1])]
    self.assertLen(result, len(expected))
    for res, exp in zip(result, expected):
      self.assertAllEqual(res, exp)

  @parameterized.parameters([
      dict(rt_shape=(2, None)),
      dict(rt_shape=None),
  ])
  def testUniformSplitDynamicShape(self, rt_shape):
    rt = ragged_tensor.RaggedTensor.from_row_lengths([1.0, 2.0, 3.0, 4.0],
                                                     [3, 1])
    rt_spec = ragged_tensor.RaggedTensorSpec(rt_shape, ragged_rank=1)
    @def_function.function(input_signature=[rt_spec])
    def split_tensors(rt):
      return ragged_array_ops.split(rt, 2)

    splited_rts = split_tensors(rt)
    expected_rts = [
        ragged_tensor.RaggedTensor.from_row_lengths([1.0, 2.0, 3.0], [3]),
        ragged_tensor.RaggedTensor.from_row_lengths([4.0], [1])]
    for splited_rt, expected_rt in zip(splited_rts, expected_rts):
      self.assertAllEqual(splited_rt, expected_rt)

  @parameterized.parameters([
      dict(rt_shape=x, lengths_shape=y) for x, y in itertools.product(
          [(2, None), None],
          [(2,), (None,), None])
  ])
  def testRaggedSplitDynamicShape(self, rt_shape, lengths_shape):
    rt_spec = ragged_tensor.RaggedTensorSpec(rt_shape, ragged_rank=1)
    lengths_spec = tensor_spec.TensorSpec(lengths_shape, dtype=dtypes.int32)
    @def_function.function(input_signature=[rt_spec, lengths_spec])
    def split_tensors(rt, split_lengths):
      return ragged_array_ops.split(rt, split_lengths, num=2)

    rt = ragged_tensor.RaggedTensor.from_row_lengths([1.0, 2.0, 3.0, 4.0],
                                                     [3, 1])
    split_lengths = [1, 1]
    # split_lengths matches num at runtime
    splited_rts = split_tensors(rt, split_lengths)
    expected_rts = [
        ragged_tensor.RaggedTensor.from_row_lengths([1.0, 2.0, 3.0], [3]),
        ragged_tensor.RaggedTensor.from_row_lengths([4.0], [1])]
    for splited_rt, expected_rt in zip(splited_rts, expected_rts):
      self.assertAllEqual(splited_rt, expected_rt)

  @parameterized.parameters([
      dict(
          descr='lengths known rank, num and lengths mismatch',
          rt_shape=(2, None),
          lengths_shape=(None,),
          lengths=[1, 1, 0],
          num=2,
          exception=errors.InvalidArgumentError,
          message='inconsistent'),
      dict(
          descr='lengths unknown rank, num and lengths mismatch',
          rt_shape=None,
          lengths_shape=None,
          lengths=[1, 1, 0],
          num=2,
          exception=errors.InvalidArgumentError,
          message='inconsistent'),
      dict(
          descr='rt unknown rank, negative axis',
          rt_shape=None,
          lengths_shape=None,
          lengths=[1, 1],
          axis=-2,
          num=2,
          exception=ValueError,
          message='negative'),
      dict(
          descr='lengths unknown rank, num is None',
          rt_shape=None,
          lengths_shape=None,
          lengths=[1, 1],
          exception=ValueError,
          message='`num` must be specified'),
      dict(
          descr='lengths unknown rank, dynamic rank!=1',
          rt_shape=None,
          lengths_shape=None,
          lengths=[[1, 1]],
          num=2,
          exception=(ValueError, errors.InvalidArgumentError)),
  ])
  def testRaggedSplitDynamicShapeError(self,
                                       descr,
                                       rt_shape,
                                       lengths_shape,
                                       lengths,
                                       exception,
                                       message='',
                                       axis=0,
                                       num=None):
    rt_spec = ragged_tensor.RaggedTensorSpec(rt_shape, ragged_rank=1)
    split_lengths_spec = tensor_spec.TensorSpec(lengths_shape,
                                                dtype=dtypes.int32)
    @def_function.function(input_signature=[rt_spec, split_lengths_spec])
    def split_tensors(rt, split_lengths):
      return ragged_array_ops.split(rt, split_lengths, axis=axis, num=num)

    rt = ragged_tensor.RaggedTensor.from_row_lengths([1.0, 2.0, 3.0, 4.0],
                                                     [3, 1])
    with self.assertRaisesRegex(exception, message):
      self.evaluate(split_tensors(rt=rt, split_lengths=lengths))


if __name__ == '__main__':
  googletest.main()
