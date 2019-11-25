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
"""Tests for the GroupByReducer serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class GroupByReducerSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_dataset(self, components):
    reducer = grouping.Reducer(
        init_func=lambda _: np.int64(0),
        reduce_func=lambda x, y: x + y,
        finalize_func=lambda x: x)

    return dataset_ops.Dataset.from_tensor_slices(components).apply(
        grouping.group_by_reducer(lambda x: x % 5, reducer))

  @combinations.generate(test_base.default_test_combinations())
  def testCoreGroupByReducer(self):
    components = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    self.verify_unused_iterator(
        lambda: self._build_dataset(components), 5, verify_exhausted=True)
    self.verify_multiple_breaks(
        lambda: self._build_dataset(components), 5, verify_exhausted=True)
    self.verify_reset_restored_iterator(
        lambda: self._build_dataset(components), 5, verify_exhausted=True)


if __name__ == '__main__':
  test.main()
