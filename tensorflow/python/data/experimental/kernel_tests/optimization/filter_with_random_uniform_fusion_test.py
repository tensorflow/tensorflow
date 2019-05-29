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
"""Tests for the `FilterWithRandomUniformFusion` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class FilterWithRandomUniformFusionTest(test_base.DatasetTestBase):

  def testFilterWithRandomUniformFusion(self):
    dataset = dataset_ops.Dataset.range(10000000).apply(
        optimization.assert_next(["Sampling"]))
    dataset = dataset.filter(lambda _: random_ops.random_uniform([]) < 0.05)

    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.filter_with_random_uniform_fusion = True
    dataset = dataset.with_options(options)

    get_next = self.getNext(dataset)
    self.evaluate(get_next())


if __name__ == "__main__":
  test.main()
