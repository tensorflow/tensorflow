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
"""Tests for warm_starting_util with Distribution Strategy.

These tests are located here instead of as part of `WarmStartingUtilTest`
because they need access to distribution strategies which are only present in
contrib right now.
TODO(priyag): Move the tests to core `WarmStartingUtilTest` when distribution
strategy moves out of contrib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized

from tensorflow.contrib.distribute.python import combinations
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import warm_starting_util as ws_util


class WarmStartingUtilWithDistributionStrategyTest(
    test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(
      distribution=[combinations.default_strategy,
                    combinations.one_device_strategy,
                    combinations.mirrored_strategy_with_gpu_and_cpu,
                    combinations.mirrored_strategy_with_two_gpus],
      save_with_distribution=[True, False],
      restore_with_distribution=[True, False],
      mode=["graph"]))
  def testWarmStart(self, distribution, save_with_distribution,
                    restore_with_distribution):

    var_name = "v"
    original_value = [[1., 2.], [3., 4.]]

    # Create variable and save checkpoint from which to warm-start.
    def create_var(g):
      with self.test_session(graph=g) as sess:
        var = variable_scope.get_variable(var_name, initializer=original_value)
        sess.run(variables.global_variables_initializer())
        saver = saver_lib.Saver()
        ckpt_prefix = os.path.join(self.get_temp_dir(), "model")
        saver.save(sess, ckpt_prefix, global_step=0)
        return var, sess.run(var)

    if save_with_distribution:
      with ops.Graph().as_default() as g, distribution.scope():
        _, prev_init_val = create_var(g)
    else:
      with ops.Graph().as_default() as g:
        _, prev_init_val = create_var(g)

    # Verify we initialized the values correctly.
    self.assertAllEqual(original_value, prev_init_val)

    def warm_start(g):
      with self.test_session(graph=g) as sess:
        # Initialize with zeros.
        var = variable_scope.get_variable(
            var_name, initializer=[[0., 0.], [0., 0.]])
        ws_util.warm_start(self.get_temp_dir())
        sess.run(variables.global_variables_initializer())
        # Verify weights were correctly warm-started to previous values.
        self.assertAllEqual(original_value, self.evaluate(var))

    # Warm start in a new graph.
    if restore_with_distribution:
      with ops.Graph().as_default() as g, distribution.scope():
        warm_start(g)
    else:
      with ops.Graph().as_default() as g:
        warm_start(g)


if __name__ == "__main__":
  test.main()
