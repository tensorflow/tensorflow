# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the `SeqInterleavePrefetch` optimization."""
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class SeqInterleavePrefetchTest(
    test_base.DatasetTestBase, parameterized.TestCase
):

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(cycle_length=[2, 4]),
          combinations.combine(block_length=[2, 4]),
          combinations.combine(other_arguments=[True, False]),
      )
  )
  def testOptimizationSeqInterleavePrefetch(
      self,
      cycle_length,
      block_length,
      other_arguments,
  ):
    num_input_elements = 16
    var1 = constant_op.constant(9, dtype=dtypes.int64)
    var2 = constant_op.constant(11, dtype=dtypes.int64)

    # dataset1: Deterministic parallel interleave dataset.
    dataset1 = dataset_ops.Dataset.range(num_input_elements)
    options1 = options_lib.Options()
    options1.experimental_optimization.apply_default_optimizations = False
    options1.experimental_optimization.seq_interleave_prefetch = False
    dataset1 = dataset1.with_options(options1)
    if other_arguments:
      dataset1 = dataset1.interleave(
          (lambda _: dataset_ops.Dataset.range(var1 + var2 + 1)),
          cycle_length=cycle_length,
          block_length=block_length,
          num_parallel_calls=dataset_ops.AUTOTUNE,
          deterministic=True,
      )
    else:
      dataset1 = dataset1.interleave(
          (lambda _: dataset_ops.Dataset.range(num_input_elements)),
          cycle_length=cycle_length,
          block_length=block_length,
          num_parallel_calls=dataset_ops.AUTOTUNE,
          deterministic=True,
      )

    # dataset2: Deterministic parallel interleave dataset with
    # `seq_interleave_prefetch` optimization enabled.
    dataset2 = dataset_ops.Dataset.range(num_input_elements)
    options2 = options_lib.Options()
    options2.experimental_optimization.apply_default_optimizations = False
    options2.experimental_optimization.seq_interleave_prefetch = True
    dataset2 = dataset2.with_options(options2)
    if other_arguments:
      dataset2 = dataset2.interleave(
          (lambda _: dataset_ops.Dataset.range(var1 + var2 + 1)),
          cycle_length=cycle_length,
          block_length=block_length,
          num_parallel_calls=dataset_ops.AUTOTUNE,
          deterministic=True,
      )
    else:
      dataset2 = dataset2.interleave(
          (lambda _: dataset_ops.Dataset.range(num_input_elements)),
          cycle_length=cycle_length,
          block_length=block_length,
          num_parallel_calls=dataset_ops.AUTOTUNE,
          deterministic=True,
      )

    self.assertDatasetsEqual(dataset1, dataset2)


if __name__ == "__main__":
  test.main()
