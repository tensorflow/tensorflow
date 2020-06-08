# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for make_template used with MirroredStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class TemplateMirroredStrategyTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_merge_call(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")

    def fn():
      var1 = variable_scope.get_variable(
          "var1", shape=[], initializer=init_ops.constant_initializer(21.))
      ds_context.get_replica_context().merge_call(lambda _: ())
      var2 = variable_scope.get_variable(
          "var2", shape=[], initializer=init_ops.constant_initializer(2.))
      return var1 * var2

    temp = template.make_template("my_template", fn)

    strategy = mirrored_strategy.MirroredStrategy(["/cpu:0", "/gpu:0"])
    out = strategy.experimental_local_results(
        strategy.run(temp))

    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual([42., 42.], self.evaluate(out))


if __name__ == "__main__":
  test.main()
