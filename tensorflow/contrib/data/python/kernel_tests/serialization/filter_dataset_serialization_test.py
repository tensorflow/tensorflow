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
"""Tests for the FilterDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class FilterDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_filter_range_graph(self, div):
    return dataset_ops.Dataset.range(100).filter(
        lambda x: math_ops.not_equal(math_ops.mod(x, div), 2))

  def testFilterCore(self):
    div = 3
    num_outputs = np.sum([x % 3 != 2 for x in range(100)])
    self.run_core_tests(lambda: self._build_filter_range_graph(div),
                        lambda: self._build_filter_range_graph(div * 2),
                        num_outputs)

  def _build_filter_dict_graph(self):
    return dataset_ops.Dataset.range(10).map(
        lambda x: {"foo": x * 2, "bar": x ** 2}).filter(
            lambda d: math_ops.equal(d["bar"] % 2, 0)).map(
                lambda d: d["foo"] + d["bar"])

  def testFilterDictCore(self):
    num_outputs = np.sum([(x**2) % 2 == 0 for x in range(10)])
    self.run_core_tests(self._build_filter_dict_graph, None, num_outputs)

  def _build_sparse_filter(self):

    def _map_fn(i):
      return sparse_tensor.SparseTensor(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[1, 1]), i

    def _filter_fn(_, i):
      return math_ops.equal(i % 2, 0)

    return dataset_ops.Dataset.range(10).map(_map_fn).filter(_filter_fn).map(
        lambda x, i: x)

  def testSparseCore(self):
    num_outputs = 5
    self.run_core_tests(self._build_sparse_filter, None, num_outputs)


if __name__ == "__main__":
  test.main()
