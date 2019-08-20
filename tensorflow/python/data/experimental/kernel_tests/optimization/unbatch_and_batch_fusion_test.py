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
"""Tests for the `UnbatchAndBatchFusion` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class UnbatchAndBatchFusionTest(test_base.DatasetTestBase):

  def testUnbatchAndBatchFusion(self):
    dataset = dataset_ops.Dataset.range(10).batch(5).apply(
        optimization.assert_next(
            ["Map", "ExperimentalUnbatchAndBatch"])).unbatch().batch(2)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.unbatch_and_batch_fusion = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset, expected_output=[[x, x + 1] for x in range(0, 10, 2)])


if __name__ == "__main__":
  test.main()
