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
"""Tests for checkpoint_utils.init_from_checkpoint with Distribution Strategy.

These tests are located here instead of as part of
`python.training.CheckpointsTest` because they need access to distribution
strategies which are only present in contrib right now.
TODO(priyag): Move the tests to core `python.training.CheckpointsTest` when
distribution strategy moves out of contrib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.contrib.distribute.python import combinations
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import checkpoint_utils_test


class CheckpointUtilsWithDistributionStrategyTest(
    test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      distribution=[combinations.default_strategy,
                    combinations.one_device_strategy,
                    combinations.mirrored_strategy_with_gpu_and_cpu,
                    combinations.mirrored_strategy_with_two_gpus,
                    combinations.core_mirrored_strategy_with_gpu_and_cpu,
                    combinations.core_mirrored_strategy_with_two_gpus],
      in_replica_mode=[True, False],
      mode=["graph"]))
  def testInitFromCheckpoint(self, distribution, in_replica_mode):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1_value, v2_value, _, _ = checkpoint_utils_test._create_checkpoints(
          session, checkpoint_dir)

    def init_and_verify(g):
      v1 = variable_scope.get_variable("new_var1", [1, 10])
      v2 = variable_scope.get_variable(
          "new_var2", [10, 10],
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.MEAN)
      checkpoint_utils.init_from_checkpoint(checkpoint_dir, {
          "var1": "new_var1",
          "var2": "new_var2"
      })
      with self.session(graph=g) as session:
        session.run(variables.global_variables_initializer())
        self.assertAllEqual(v1_value, self.evaluate(v1))
        self.assertAllEqual(v2_value, self.evaluate(v2))

    with ops.Graph().as_default() as g, distribution.scope():
      if in_replica_mode:
        distribution.extended.call_for_each_replica(init_and_verify, args=[g])
      else:
        init_and_verify(g)


if __name__ == "__main__":
  test.main()
