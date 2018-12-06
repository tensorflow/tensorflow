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
"""Tests for the `FilterFusion` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops.optimization_options import OptimizationOptions
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _filter_fusion_test_cases():
  """Generates test cases for the FilterFusion optimization."""

  take_all = lambda x: constant_op.constant(True)
  is_zero = lambda x: math_ops.equal(x, 0)
  greater = lambda x: math_ops.greater(x + 5, 0)

  tests = []
  filters = [take_all, is_zero, greater]
  identity = lambda x: x
  for x, predicate_1 in enumerate(filters):
    for y, predicate_2 in enumerate(filters):
      tests.append(("Mixed{}{}".format(x, y), identity,
                    [predicate_1, predicate_2]))
      for z, predicate_3 in enumerate(filters):
        tests.append(("Mixed{}{}{}".format(x, y, z), identity,
                      [predicate_1, predicate_2, predicate_3]))

  take_all_multiple = lambda x, y: constant_op.constant(True)
  # Multi output
  tests.append(("Multi1", lambda x: (x, x),
                [take_all_multiple, take_all_multiple]))
  tests.append(("Multi2", lambda x: (x, 2), [
      take_all_multiple,
      lambda x, y: math_ops.equal(x * math_ops.cast(y, dtypes.int64), 0)
  ]))
  return tuple(tests)


@test_util.run_all_in_graph_and_eager_modes
class FilterFusionTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(*_filter_fusion_test_cases())
  def testFilterFusion(self, map_function, predicates):
    dataset = dataset_ops.Dataset.range(5).apply(
        optimization.assert_next(["Map", "Filter",
                                  "MemoryCacheImpl"])).map(map_function)
    for predicate in predicates:
      dataset = dataset.filter(predicate)

    dataset = dataset.cache()
    options = dataset_ops.Options()
    options.experimental_optimization = OptimizationOptions()
    options.experimental_optimization.filter_fusion = True
    dataset = dataset.with_options(options)
    expected_output = []
    for x in range(5):
      r = map_function(x)
      filtered = False
      for predicate in predicates:
        if isinstance(r, tuple):
          b = predicate(*r)  # Pass tuple as multiple arguments.
        else:
          b = predicate(r)
        if not self.evaluate(b):
          filtered = True
          break

      if not filtered:
        expected_output.append(r)
    self.assertDatasetProduces(dataset, expected_output=expected_output)


if __name__ == "__main__":
  test.main()
