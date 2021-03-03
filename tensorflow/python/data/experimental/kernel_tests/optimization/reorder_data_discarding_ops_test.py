# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the `ReorderDataDiscardingOps` rewrite."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class ReorderDataDiscardingOpsTest(test_base.DatasetTestBase,
                                   parameterized.TestCase):

  @combinations.generate(
      combinations.combine(tf_api_version=2, mode=["eager", "graph"]))
  def testSimpleReorderingV2(self):
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.apply(
        testing.assert_next(
            ["FiniteSkip", "FiniteTake", "Shard", "ParallelMap", "Prefetch"]))
    dataset = dataset.map(lambda x: x + 1, num_parallel_calls=10)
    dataset = dataset.skip(10)
    dataset = dataset.prefetch(1)
    dataset = dataset.take(50)
    dataset = dataset.shard(2, 0)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.reorder_data_discarding_ops = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(11, 61, 2))

  @combinations.generate(
      combinations.combine(tf_api_version=1, mode=["eager", "graph"]))
  def testSimpleReorderingV1(self):
    dataset = dataset_ops.Dataset.range(100)
    # Map ops have preserve_cardinality=false in tensorflow v1.
    dataset = dataset.apply(
        testing.assert_next(
            ["ParallelMap", "FiniteSkip", "FiniteTake", "Shard", "Prefetch"]))
    dataset = dataset.map(lambda x: x + 1, num_parallel_calls=10)
    dataset = dataset.skip(10)
    dataset = dataset.prefetch(1)
    dataset = dataset.take(50)
    dataset = dataset.shard(2, 0)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.reorder_data_discarding_ops = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(11, 61, 2))


if __name__ == "__main__":
  test.main()
