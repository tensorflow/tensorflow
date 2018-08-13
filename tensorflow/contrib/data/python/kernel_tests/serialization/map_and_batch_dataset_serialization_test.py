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
"""Tests for the MapAndBatchDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib.data.python.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class MapAndBatchDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def testNumParallelBatches(self):
    range_size = 11
    num_repeats = 2
    batch_size = 5
    total_outputs = range_size * num_repeats
    num_outputs_drop_remainder = total_outputs // batch_size
    num_outputs_keep_remainder = int(math.ceil(total_outputs / batch_size))
    num_parallel_batches = 2

    def build_ds(range_start, drop_remainder=False):

      def _map_fn(x):
        return math_ops.square(x)

      return dataset_ops.Dataset.range(
          range_start, range_start + range_size).repeat(num_repeats).apply(
              batching.map_and_batch(
                  map_func=_map_fn,
                  batch_size=batch_size,
                  num_parallel_batches=num_parallel_batches,
                  drop_remainder=drop_remainder))

    self.run_core_tests(lambda: build_ds(10), lambda: build_ds(15),
                        num_outputs_keep_remainder)
    self.run_core_tests(lambda: build_ds(10, True), lambda: build_ds(15, True),
                        num_outputs_drop_remainder)

  def testNumParallelCalls(self):
    range_size = 11
    num_repeats = 2
    batch_size = 5
    total_outputs = range_size * num_repeats
    num_outputs_drop_remainder = total_outputs // batch_size
    num_outputs_keep_remainder = int(math.ceil(total_outputs / batch_size))
    num_parallel_calls = 7

    def build_ds(range_start, drop_remainder=False):

      def _map_fn(x):
        return math_ops.square(x)

      return dataset_ops.Dataset.range(
          range_start, range_start + range_size).repeat(num_repeats).apply(
              batching.map_and_batch(
                  map_func=_map_fn,
                  batch_size=batch_size,
                  num_parallel_calls=num_parallel_calls,
                  drop_remainder=drop_remainder))

    self.run_core_tests(lambda: build_ds(10), lambda: build_ds(15),
                        num_outputs_keep_remainder)
    self.run_core_tests(lambda: build_ds(10, True), lambda: build_ds(15, True),
                        num_outputs_drop_remainder)


if __name__ == "__main__":
  test.main()
