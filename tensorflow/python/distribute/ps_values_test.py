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
"""Tests for the distributed values library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.central_storage_strategy_with_two_gpus
        ],
        mode=["graph", "eager"]))
class AggregatingVariableTest(test.TestCase, parameterized.TestCase):

  def testAssignOutOfScope(self, distribution):
    with distribution.scope():
      aggregating = variables_lib.Variable(1.)
    self.assertIsInstance(aggregating, ps_values.AggregatingVariable)
    self.evaluate(aggregating.assign(3.))
    self.assertEqual(self.evaluate(aggregating.read_value()), 3.)
    self.assertEqual(self.evaluate(aggregating._v.read_value()), 3.)

  def testAssignAdd(self, distribution):
    with distribution.scope():
      v = variable_scope.variable(
          1, aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def assign():
      return v.assign_add(2)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(assign)))
    self.assertAllEqual([3], per_replica_results)


if __name__ == "__main__":
  test.main()
