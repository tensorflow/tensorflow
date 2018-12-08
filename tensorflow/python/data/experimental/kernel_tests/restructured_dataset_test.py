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
"""Tests for the private `_RestructuredDataset` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class RestructuredDatasetTest(test_base.DatasetTestBase):

  @test_util.run_deprecated_v1
  def testRestructureDataset(self):
    components = (array_ops.placeholder(dtypes.int32),
                  (array_ops.placeholder(dtypes.int32, shape=[None]),
                   array_ops.placeholder(dtypes.int32, shape=[20, 30])))
    dataset = dataset_ops.Dataset.from_tensors(components)

    i32 = dtypes.int32

    test_cases = [((i32, i32, i32), None),
                  (((i32, i32), i32), None),
                  ((i32, i32, i32), (None, None, None)),
                  ((i32, i32, i32), ([17], [17], [20, 30]))]

    for new_types, new_shape_lists in test_cases:
      # pylint: disable=protected-access
      new = batching._RestructuredDataset(dataset, new_types, new_shape_lists)
      # pylint: enable=protected-access
      self.assertEqual(new_types, new.output_types)
      if new_shape_lists is not None:
        for expected_shape_list, shape in zip(
            nest.flatten(new_shape_lists), nest.flatten(new.output_shapes)):
          if expected_shape_list is None:
            self.assertIs(None, shape.ndims)
          else:
            self.assertEqual(expected_shape_list, shape.as_list())

    fail_cases = [((i32, dtypes.int64, i32), None),
                  ((i32, i32, i32, i32), None),
                  ((i32, i32, i32), ((None, None), None)),
                  ((i32, i32, i32), (None, None, None, None)),
                  ((i32, i32, i32), (None, [None], [21, 30]))]

    for new_types, new_shape_lists in fail_cases:
      with self.assertRaises(ValueError):
        # pylint: disable=protected-access
        new = batching._RestructuredDataset(dataset, new_types, new_shape_lists)
        # pylint: enable=protected-access


if __name__ == "__main__":
  test.main()
