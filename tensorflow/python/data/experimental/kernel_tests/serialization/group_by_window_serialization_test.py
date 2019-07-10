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
"""Tests for the GroupByWindow serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class GroupByWindowSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_dataset(self, components):
    return dataset_ops.Dataset.from_tensor_slices(components).repeat(-1).apply(
        grouping.group_by_window(lambda x: x % 3, lambda _, xs: xs.batch(4), 4))

  def testCoreGroupByWindow(self):
    components = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0], dtype=np.int64)
    self.verify_unused_iterator(
        lambda: self._build_dataset(components), 12, verify_exhausted=False)
    self.verify_init_before_restore(
        lambda: self._build_dataset(components), 12, verify_exhausted=False)
    self.verify_multiple_breaks(
        lambda: self._build_dataset(components), 12, verify_exhausted=False)
    self.verify_reset_restored_iterator(
        lambda: self._build_dataset(components), 12, verify_exhausted=False)
    self.verify_restore_in_empty_graph(
        lambda: self._build_dataset(components), 12, verify_exhausted=False)
    diff_components = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    self.verify_restore_in_modified_graph(
        lambda: self._build_dataset(components),
        lambda: self._build_dataset(diff_components),
        12,
        verify_exhausted=False)


if __name__ == '__main__':
  test.main()
