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
"""Tests for the `NoopElimination` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops.optimization_options import OptimizationOptions
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class NoopEliminationTest(test_base.DatasetTestBase):

  def testNoopElimination(self):
    a = constant_op.constant(1, dtype=dtypes.int64)
    b = constant_op.constant(2, dtype=dtypes.int64)
    some_tensor = math_ops.mul(a, b)

    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.apply(
        optimization.assert_next(
            ["FiniteRepeat", "FiniteSkip", "Prefetch", "MemoryCacheImpl"]))
    dataset = dataset.repeat(some_tensor).skip(5).take(-1).skip(0).repeat(
        1).prefetch(0).prefetch(1).cache()
    options = dataset_ops.Options()
    options.experimental_optimization = OptimizationOptions()
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=range(5))


if __name__ == "__main__":
  test.main()
