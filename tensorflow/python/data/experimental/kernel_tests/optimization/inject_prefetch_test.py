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
"""Tests for the `AutotuneBuffers` rewrite."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.compat import compat
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class InjectPrefetchTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _enable_autotune_buffers(self, dataset):
    options = dataset_ops.Options()
    options.experimental_optimization.autotune_buffers = True
    return dataset.with_options(options)

  @combinations.generate(test_base.default_test_combinations())
  def testParallelMap(self):
    dataset = dataset_ops.Dataset.range(100)
    parallel_map = "ParallelMap"
    if compat.forward_compatible(2020, 3, 6):
      parallel_map = "ParallelMapV2"
    dataset = dataset.apply(
        testing.assert_next([parallel_map, "Prefetch", "FiniteTake"]))
    dataset = dataset.map(
        lambda x: x + 1, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.take(50)
    dataset = self._enable_autotune_buffers(dataset)
    self.assertDatasetProduces(dataset, range(1, 51))

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatch(self):
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.apply(
        testing.assert_next(["MapAndBatch", "Prefetch", "FiniteTake"]))
    dataset = dataset.map(
        lambda x: x + 1, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.batch(10)
    dataset = dataset.take(5)
    dataset = self._enable_autotune_buffers(dataset)
    self.assertDatasetProduces(
        dataset, [list(range(i + 1, i + 11)) for i in range(0, 50, 10)])

  @combinations.generate(test_base.default_test_combinations())
  def testParallelInterleave(self):
    dataset = dataset_ops.Dataset.range(100)
    parallel_interleave = "ParallelInterleaveV3"
    if compat.forward_compatible(2020, 3, 6):
      parallel_interleave = "ParallelInterleaveV4"
    dataset = dataset.apply(
        testing.assert_next([parallel_interleave, "Prefetch", "FiniteTake"]))
    dataset = dataset.interleave(
        lambda x: dataset_ops.Dataset.from_tensors(x + 1),
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.take(50)
    dataset = self._enable_autotune_buffers(dataset)
    self.assertDatasetProduces(dataset, range(1, 51))

  @combinations.generate(test_base.default_test_combinations())
  def testChainedParallelDatasets(self):
    dataset = dataset_ops.Dataset.range(100)
    parallel_interleave = "ParallelInterleaveV3"
    if compat.forward_compatible(2020, 3, 6):
      parallel_interleave = "ParallelInterleaveV4"
    parallel_map = "ParallelMap"
    if compat.forward_compatible(2020, 3, 6):
      parallel_map = "ParallelMapV2"
    dataset = dataset.apply(
        testing.assert_next([
            parallel_map, "Prefetch", parallel_interleave, "Prefetch",
            "MapAndBatch", "Prefetch", "FiniteTake"
        ]))
    dataset = dataset.map(
        lambda x: x + 1, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.interleave(
        lambda x: dataset_ops.Dataset.from_tensors(x + 1),
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.map(
        lambda x: x + 1, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.batch(1)
    dataset = dataset.take(50)
    dataset = self._enable_autotune_buffers(dataset)
    self.assertDatasetProduces(dataset, [[i] for i in range(3, 53)])

  @combinations.generate(test_base.default_test_combinations())
  def testNoRegularMap(self):
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.apply(testing.assert_next(["Map", "FiniteTake"]))
    dataset = dataset.map(lambda x: x + 1).take(50)
    dataset = self._enable_autotune_buffers(dataset)
    self.assertDatasetProduces(dataset, range(1, 51))


if __name__ == "__main__":
  test.main()
