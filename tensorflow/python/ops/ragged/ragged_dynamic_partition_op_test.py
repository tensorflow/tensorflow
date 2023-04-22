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
"""Tests for ragged_array_ops.stack_dynamic_partitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSegmentStackOpTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):

  @parameterized.parameters([
      dict(  # empty inputs
          data=[],
          partitions=[],
          num_partitions=0,
          expected=[],
          expected_ragged_rank=1),
      dict(  # empty data, num_partitions>0
          data=[],
          partitions=[],
          num_partitions=3,
          expected=[[], [], []]),
      dict(  # 1D data, 1D partitions (docstring example)
          data=['a', 'b', 'c', 'd', 'e'],
          partitions=[3, 0, 2, 2, 3],
          num_partitions=5,
          expected=[['b'], [], ['c', 'd'], ['a', 'e'], []]),
      dict(  # 2D data, 1D partitions
          data=[['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']],
          data_ragged_rank=0,
          partitions=[2, 1, 2, 3],
          num_partitions=4,
          expected=[[], [['c', 'd']], [['a', 'b'], ['e', 'f']], [['g', 'h']]],
          expected_ragged_rank=1),
      dict(  # 2D ragged data, 1D partitions
          data=[['a'], ['b', 'c', 'd'], [], ['e', 'f']],
          data_ragged_rank=1,
          partitions=[2, 1, 2, 3],
          num_partitions=4,
          expected=[[], [['b', 'c', 'd']], [['a'], []], [['e', 'f']]],
          expected_ragged_rank=2),
      dict(  # 2D data, 2D partitions
          data=[['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']],
          data_ragged_rank=0,
          partitions=[[3, 0], [2, 2], [4, 3], [2, 0]],
          num_partitions=5,
          expected=[['b', 'h'], [], ['c', 'd', 'g'], ['a', 'f'], ['e']]),
      dict(  # 2D ragged data, 2D ragged partitions
          data=[['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']],
          data_ragged_rank=0,
          partitions=[[3, 0], [2, 2], [4, 3], [2, 0]],
          num_partitions=5,
          expected=[['b', 'h'], [], ['c', 'd', 'g'], ['a', 'f'], ['e']]),
      dict(  # 3D data, 1d partitions
          data=[[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]],
          data_ragged_rank=0,
          partitions=[1, 0],
          num_partitions=2,
          expected=[[[['e', 'f'], ['g', 'h']]], [[['a', 'b'], ['c', 'd']]]],
          expected_ragged_rank=1),
      dict(  # 3D data (ragged_rank=1), 1d partitions
          data=[[['a', 'b'], ['c', 'd']], [['e', 'f']]],
          data_ragged_rank=1,
          partitions=[2, 0],
          num_partitions=3,
          expected=[[[['e', 'f']]], [], [[['a', 'b'], ['c', 'd']]]],
          expected_ragged_rank=2),
      dict(  # 3D data (ragged_rank=2), 1d partitions
          data=[[['a', 'b'], ['c', 'd']], [['e', 'f', 'g', 'h']]],
          data_ragged_rank=2,
          partitions=[2, 0],
          num_partitions=3,
          expected=[[[['e', 'f', 'g', 'h']]], [], [[['a', 'b'], ['c', 'd']]]],
          expected_ragged_rank=3),
      dict(  # 3D data, 2d partitions
          data=[[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]],
          data_ragged_rank=0,
          partitions=[[1, 0], [0, 3]],
          segment_ids_ragged_rank=0,
          num_partitions=4,
          expected=[[['c', 'd'], ['e', 'f']], [['a', 'b']], [], [['g', 'h']]],
          expected_ragged_rank=1),
      dict(  # 3D data (ragged_rank=1), 2d partitions
          data=[[['a', 'b'], ['c', 'd']], [['e', 'f']]],
          data_ragged_rank=1,
          partitions=[[1, 0], [0]],
          segment_ids_ragged_rank=1,
          num_partitions=2,
          expected=[[['c', 'd'], ['e', 'f']], [['a', 'b']]],
          expected_ragged_rank=1),
      dict(  # 3D data (ragged_rank=2), 2d partitions
          data=[[['a', 'b'], ['c', 'd']], [['e', 'f', 'g', 'h']]],
          data_ragged_rank=2,
          partitions=[[1, 0], [0]],
          segment_ids_ragged_rank=1,
          num_partitions=3,
          expected=[[['c', 'd'], ['e', 'f', 'g', 'h']], [['a', 'b']], []],
          expected_ragged_rank=2),
      dict(  # 3D data (ragged_rank=2), 3d partitions (ragged_rank=2)
          data=[[['a', 'b'], ['c', 'd']], [['e', 'f', 'g', 'h']]],
          data_ragged_rank=2,
          partitions=[[[3, 0], [1, 2]], [[1, 1, 0, 1]]],
          segment_ids_ragged_rank=2,
          num_partitions=4,
          expected=[['b', 'g'], ['c', 'e', 'f', 'h'], ['d'], ['a']]),
      dict(  # 0D data, 0D partitions
          data='a',
          partitions=3,
          num_partitions=5,
          expected=[[], [], [], ['a'], []]),
      dict(  # 1D data, 0D partitions
          data=['a', 'b', 'c'],
          partitions=3,
          num_partitions=5,
          expected=[[], [], [], [['a', 'b', 'c']], []],
          expected_ragged_rank=1),
      dict(  # 2D data, 0D partitions
          data=[['a', 'b'], ['c', 'd']],
          data_ragged_rank=0,
          partitions=3,
          num_partitions=5,
          expected=[[], [], [], [[['a', 'b'], ['c', 'd']]], []],
          expected_ragged_rank=1),
      dict(  # 2D data (ragged_rank=1), 0D partitions
          data=[['a', 'b'], ['c']],
          data_ragged_rank=1,
          partitions=3,
          num_partitions=5,
          expected=[[], [], [], [[['a', 'b'], ['c']]], []],
          expected_ragged_rank=3),
  ])
  def testRaggedSegmentStack(self,
                             data,
                             partitions,
                             num_partitions,
                             expected,
                             data_ragged_rank=None,
                             segment_ids_ragged_rank=None,
                             expected_ragged_rank=None):
    for seg_dtype in [dtypes.int32, dtypes.int64]:
      data_tensor = ragged_factory_ops.constant(
          data, row_splits_dtype=seg_dtype, ragged_rank=data_ragged_rank)
      segment_ids_tensor = ragged_factory_ops.constant(
          partitions,
          dtype=seg_dtype,
          row_splits_dtype=seg_dtype,
          ragged_rank=segment_ids_ragged_rank)
      expected_tensor = ragged_factory_ops.constant(
          expected,
          row_splits_dtype=seg_dtype,
          ragged_rank=expected_ragged_rank)
      result = ragged_array_ops.stack_dynamic_partitions(
          data_tensor, segment_ids_tensor, num_partitions)
      self.assertAllEqual(result, expected_tensor)

      # Check that it's equivalent to tf.stack(dynamic_partition(...)),
      # where applicable.
      if (data_ragged_rank == 0 and segment_ids_ragged_rank == 0 and
          seg_dtype == dtypes.int32):
        equiv = ragged_concat_ops.stack(
            data_flow_ops.dynamic_partition(data_tensor, segment_ids_tensor,
                                            num_partitions))
        self.assertAllEqual(result, self.evaluate(equiv).to_list())

  @parameterized.parameters([
      dict(
          data=['a', 'b', 'c'],
          partitions=[2, -1, 0],
          num_partitions=10,
          error='must be non-negative'),
      dict(
          data=['a', 'b', 'c'],
          partitions=[2, 10, 0],
          num_partitions=1,
          error='partitions must be less than num_partitions'),
      dict(
          data=['a', 'b', 'c'],
          partitions=[2, 10, 0],
          num_partitions=10,
          error='partitions must be less than num_partitions'),
      dict(
          data=[['a', 'b'], ['c']],
          partitions=[[2], [3, 0]],
          num_partitions=10,
          error='data and partitions have incompatible ragged shapes'),
  ])
  def testRuntimeError(self, data, partitions, num_partitions, error):
    data = ragged_factory_ops.constant(data)
    partitions = ragged_factory_ops.constant(partitions, dtype=dtypes.int64)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                error):
      self.evaluate(
          ragged_array_ops.stack_dynamic_partitions(data, partitions,
                                                    num_partitions))

  @parameterized.parameters([
      dict(
          data=['a', 'b', 'c'],
          partitions=[1, 2],
          num_partitions=10,
          error=r'Shapes \(2,\) and \(3,\) are incompatible'),
      dict(
          data=[['a', 'b'], ['c', 'd']],
          partitions=[[1, 2, 3], [4, 5, 6]],
          num_partitions=10,
          error=r'Shapes \(2, 3\) and \(2, 2\) are incompatible'),
      dict(
          data=['a', 'b', 'c'],
          partitions=[1, 2, 3],
          num_partitions=[1, 2, 3],
          error='must have rank 0'),
  ])
  def testStaticError(self, data, partitions, num_partitions, error):
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                error):
      ragged_array_ops.stack_dynamic_partitions(data, partitions,
                                                num_partitions)

  def testUnknownRankError(self):
    if context.executing_eagerly():
      return
    partitions = array_ops.placeholder(dtypes.int32, None)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                'partitions must have known rank'):
      ragged_array_ops.stack_dynamic_partitions(['a', 'b', 'c'], partitions, 10)


if __name__ == '__main__':
  googletest.main()
