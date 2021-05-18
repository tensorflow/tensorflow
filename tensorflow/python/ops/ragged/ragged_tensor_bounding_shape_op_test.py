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
"""Tests for ragged.bounding_shape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorBoundingShapeOp(test_util.TensorFlowTestCase,
                                  parameterized.TestCase):

  @parameterized.named_parameters([
      # rank = 2
      dict(testcase_name='docstring_example',
           rt=[[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]],
           expected=[5, 4]),
      dict(testcase_name='shape_5_3',
           rt=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           expected=[5, 3]),
      dict(testcase_name='shape_1_7',
           rt=[['a', 'b', 'c', 'd', 'e', 'f', 'g']],
           expected=[1, 7]),
      dict(testcase_name='shape_3_7',
           rt=[[], ['a', 'b', 'c', 'd', 'e', 'f', 'g'], []],
           expected=[3, 7]),
      dict(testcase_name='shape_5_3_row_splits_int32',
           rt=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           rt_row_splits_dtype=dtypes.int32,
           expected=[5, 3]),
      dict(testcase_name='shape_0_0',
           rt=[],
           rt_ragged_rank=1,
           expected=[0, 0]),
      dict(testcase_name='shape_3_0',
           rt=[[], [], []],
           expected=[3, 0]),
      # rank = 3
      dict(testcase_name='shape_5_3_2',
           rt=[[[0, 1], [2]], [[3, 4], [], [5, 6]], [[7]], [], [[8, 9]]],
           expected=[5, 3, 2]),
      dict(testcase_name='shape_1_7_2',
           rt=[[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]],
           expected=[1, 7, 2]),
      dict(testcase_name='shape_3_7_4',
           rt=[[], [[0, 1], [2], [], [3], [4], [5, 6, 7, 8], [9]], []],
           expected=[3, 7, 4]),
      dict(testcase_name='shape_1_7_2_ragged_rank_1',
           rt=[[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]],
           rt_ragged_rank=1,
           expected=[1, 7, 2]),
      # axis != None
      dict(testcase_name='shape_5_3_axis_0',
           rt=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           axis=0,
           expected=5),
      dict(testcase_name='shape_5_3_axis_1',
           rt=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           axis=1,
           expected=3),
      dict(testcase_name='shape_5_3_axis_1_0',
           rt=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           axis=[1, 0],
           expected=[3, 5]),
      # out_type != None
      dict(testcase_name='shape_5_3_row_splits_int64_out_type_int64',
           rt=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           rt_row_splits_dtype=dtypes.int64,
           out_type=dtypes.int64,
           expected=[5, 3]),
      dict(testcase_name='shape_5_3_row_splits_int32_out_type_int32',
           rt=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           rt_row_splits_dtype=dtypes.int32,
           out_type=dtypes.int32,
           expected=[5, 3]),
      dict(testcase_name='shape_5_3_row_splits_int64_out_type_int32',
           rt=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           rt_row_splits_dtype=dtypes.int64,
           out_type=dtypes.int32,
           expected=[5, 3]),
      dict(testcase_name='shape_5_3_row_splits_int32_out_type_int64',
           rt=[['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']],
           rt_row_splits_dtype=dtypes.int32,
           out_type=dtypes.int64,
           expected=[5, 3]),
      dict(testcase_name='shape_1_3_axis_1_row_splits_int64_out_type_int32',
           rt=[[1, 2, 3]],
           rt_row_splits_dtype=dtypes.int64,
           axis=1,
           out_type=dtypes.int32,
           expected=3)
  ])  # pyformat: disable
  def testBoundingShape(self,
                        rt,
                        expected,
                        axis=None,
                        out_type=None,
                        rt_row_splits_dtype=dtypes.int64,
                        rt_ragged_rank=None):
    rt = ragged_factory_ops.constant(
        rt, ragged_rank=rt_ragged_rank, row_splits_dtype=rt_row_splits_dtype)
    bounding_shape = rt.bounding_shape(axis=axis, out_type=out_type)
    self.assertAllEqual(bounding_shape, expected)
    if out_type is not None:
      self.assertEqual(bounding_shape.dtype, out_type)
    else:
      self.assertEqual(bounding_shape.dtype, rt_row_splits_dtype)

    # If we're testing a configuration that uses `axis`, then make sure
    # that it also works if `axis` is a tensor.
    if axis is not None:
      bounding_shape = rt.bounding_shape(
          axis=constant_op.constant(axis), out_type=out_type)
      self.assertAllEqual(bounding_shape, expected)
      if out_type is not None:
        self.assertEqual(bounding_shape.dtype, out_type)
      else:
        self.assertEqual(bounding_shape.dtype, rt_row_splits_dtype)


if __name__ == '__main__':
  googletest.main()
